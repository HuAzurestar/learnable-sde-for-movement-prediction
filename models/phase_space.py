"""PIRC-26 dynamics in physical seconds and metres, with velocity-only noise.

These are model components, not unmanaged research runners. Training and
forecast jobs must be composed through the shared research supervisor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from domain import ModelContext
from .base import ParameterGroupProvider, ParameterRole, SDEModel


class ModelContractError(ValueError):
    """MODEL_CONTRACT_ERROR: invalid or incompatible dynamics input."""


def _require(condition, message):
    if not condition:
        raise ModelContractError("MODEL_CONTRACT_ERROR: " + message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DynamicsSpec:
    """Frozen normalization/context provenance; normalization never fits here."""

    coordinate_frame: str
    train_binding_hash: str
    normalizer_hash: str
    context_hash: str
    context_dim: int = 0
    means: tuple[float, ...] = (0., 0., 0., 0.)
    scales: tuple[float, ...] = (1., 1., 1., 1.)

    def __post_init__(self):
        # Detach mutable caller arrays even though the dataclass is frozen.
        object.__setattr__(self, "means", tuple(self.means))
        object.__setattr__(self, "scales", tuple(self.scales))
        _require(isinstance(self.coordinate_frame, str) and bool(self.coordinate_frame.strip()),
                 "explicit coordinate frame required")
        _require(type(self.context_dim) is int and 0 <= self.context_dim <= 128,
                 "context dimension outside registered bound")
        for value in (self.train_binding_hash, self.normalizer_hash, self.context_hash):
            _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
                     "frozen provenance hash required")
        _require(len(self.means) == len(self.scales) == 4 + self.context_dim,
                 "normalizer dimension mismatch")
        _require(all(type(x) in (int, float) and math.isfinite(x) for x in self.means)
                 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in self.scales),
                 "normalizer must be finite with positive scales")


class AffineAccelerationDrift(nn.Module):
    """M0: A v + B p + a; optional frozen context has an explicit linear map."""

    family = "M0"

    def __init__(self, context_dim=0):
        super().__init__()
        _require(type(context_dim) is int and 0 <= context_dim <= 128, "context dimension")
        self.context_dim = context_dim
        self.A = nn.Parameter(torch.zeros(2, 2))
        self.B = nn.Parameter(torch.zeros(2, 2))
        self.a = nn.Parameter(torch.zeros(2))
        self.context_weight = nn.Parameter(torch.zeros(2, context_dim))

    def forward(self, t, z, c):
        return F.linear(z[..., 2:], self.A) + F.linear(z[..., :2], self.B, self.a) + F.linear(c, self.context_weight)

    def configuration(self):
        return {"family": self.family}


class _ResidualDrift(nn.Module):
    def __init__(self, affine, spec, features):
        super().__init__()
        _require(isinstance(affine, AffineAccelerationDrift) and affine.context_dim == spec.context_dim,
                 "residual requires compatible affine base")
        _require(isinstance(features, (tuple, list)) and 1 <= len(features) <= 8
                 and all(type(i) is int and 0 <= i < 4 + spec.context_dim for i in features)
                 and len(set(features)) == len(features), "bounded unique feature selector required")
        self.affine = affine
        self.spec = spec
        self.features = tuple(features)
        self.residual_enabled = True
        self.register_buffer("means", torch.tensor(spec.means))
        self.register_buffer("scales", torch.tensor(spec.scales))

    def selected_features(self, z, c):
        values = (torch.cat((z, c), dim=-1) - self.means) / self.scales
        return values[..., list(self.features)]

    def forward(self, t, z, c):
        base = self.affine(t, z, c)
        return base + self.residual(self.selected_features(z, c)) if self.residual_enabled else base

    def configuration(self):
        return {"family": self.family, "features": list(self.features),
                "residual_enabled": self.residual_enabled}


class RBFResidualDrift(_ResidualDrift):
    """M1-R: fixed train-bound centres, never an N by N kernel matrix."""

    family = "M1-R"

    def __init__(self, affine, spec, features, centers, bandwidth):
        super().__init__(affine, spec, features)
        centers = torch.as_tensor(centers, dtype=torch.float32)
        bandwidth = torch.as_tensor(bandwidth, dtype=torch.float32)
        _require(centers.ndim == 2 and 1 <= centers.shape[0] <= 128
                 and centers.shape[1] == len(features), "RBF basis/input bound")
        _require(bandwidth.shape == (centers.shape[0],) and torch.isfinite(bandwidth).all()
                 and (bandwidth > 0).all() and torch.isfinite(centers).all(), "finite positive RBF widths required")
        self.register_buffer("centers", centers.clone())
        self.register_buffer("bandwidth", bandwidth.clone())
        self.coefficients = nn.Parameter(torch.zeros(centers.shape[0], 2))

    def basis(self, u):
        delta = (u.unsqueeze(-2) - self.centers) / self.bandwidth.unsqueeze(-1)
        return torch.exp(-0.5 * delta.square().sum(-1))

    def residual(self, u):
        return self.basis(u) @ self.coefficients

    def configuration(self):
        return {**super().configuration(), "centers": self.centers.detach().cpu().tolist(),
                "bandwidth": self.bandwidth.detach().cpu().tolist()}


class SplineResidualDrift(_ResidualDrift):
    """M1-S: bounded additive B-splines, clamped extrapolation, no tensor product."""

    family = "M1-S"

    def __init__(self, affine, spec, features, knots, degree=3):
        super().__init__(affine, spec, features)
        knots = torch.as_tensor(knots, dtype=torch.float32)
        _require(type(degree) is int and 1 <= degree <= 3, "spline degree must be 1..3")
        _require(knots.ndim == 2 and knots.shape[0] == len(features)
                 and knots.shape[1] >= 2 * (degree + 1), "open spline knot shape")
        q = len(features) * (knots.shape[1] - degree - 1)
        _require(q <= 128 and torch.isfinite(knots).all()
                 and (knots[:, 1:] >= knots[:, :-1]).all(), "spline basis/input bound or knot ordering")
        _require((knots[:, :degree + 1] == knots[:, :1]).all()
                 and (knots[:, -degree - 1:] == knots[:, -1:]).all()
                 and (knots[:, 0] < knots[:, -1]).all(), "clamped nonempty knot domain required")
        self.degree = degree
        self.register_buffer("knots", knots.clone())
        self.coefficients = nn.Parameter(torch.zeros(q, 2))

    def basis(self, u):
        parts = []
        for index, knots in enumerate(self.knots):
            # nextafter avoids an empty half-open interval at the right boundary.
            upper = torch.nextafter(knots[-1], knots[0])
            x = u[..., index].clamp(min=knots[0], max=upper).unsqueeze(-1)
            basis = ((x >= knots[:-1]) & (x < knots[1:])).to(u.dtype)
            for order in range(1, self.degree + 1):
                count = len(knots) - order - 1
                left = knots[order:order + count] - knots[:count]
                right = knots[order + 1:order + count + 1] - knots[1:count + 1]
                left_weight = (x - knots[:count]) / torch.where(left > 0, left, torch.ones_like(left))
                right_weight = (knots[order + 1:order + count + 1] - x) / torch.where(right > 0, right, torch.ones_like(right))
                basis = (left_weight * (left > 0) * basis[..., :count]
                         + right_weight * (right > 0) * basis[..., 1:count + 1])
            parts.append(basis)
        return torch.cat(parts, dim=-1)

    def residual(self, u):
        return self.basis(u) @ self.coefficients

    def configuration(self):
        return {**super().configuration(), "knots": self.knots.detach().cpu().tolist(),
                "degree": self.degree, "extrapolation": "clamp"}


class NeuralResidualAccelerationDrift(_ResidualDrift):
    """M2: affine-equivalent initialization with a trainable, bounded residual."""

    family = "M2"

    def __init__(self, affine, spec, features, hidden=(16,), *, seed=0,
                 amplitude=1., spectral_bound=1., alpha=1.):
        super().__init__(affine, spec, features)
        _require(isinstance(hidden, (list, tuple)) and 1 <= len(hidden) <= 2
                 and all(type(width) is int and width in (16, 32) for width in hidden), "unregistered MLP capacity")
        _require(type(seed) is int and 0 <= seed < 2**63, "initialization seed")
        _require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                     for v in (amplitude, spectral_bound, alpha)), "positive physical residual bounds required")
        self.hidden, self.seed = tuple(hidden), seed
        self.amplitude, self.spectral_bound, self.alpha = float(amplitude), float(spectral_bound), float(alpha)
        # Restore global RNG: construction does not alter paired Brownian identities.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            widths = (len(features), *hidden, 2)
            self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def residual(self, u):
        for index, layer in enumerate(self.layers):
            # Frobenius norm bounds the spectral norm and has a finite gradient at
            # the zero final layer. The constraint applies on every evaluation.
            norm = torch.linalg.vector_norm(layer.weight)
            weight = layer.weight / (norm / self.spectral_bound).clamp(min=1.)
            u = F.linear(u, weight, layer.bias)
            if index < len(self.layers) - 1:
                u = torch.tanh(u)
        return self.alpha * self.amplitude * torch.tanh(u / self.amplitude)

    def configuration(self):
        return {**super().configuration(), "hidden": list(self.hidden), "seed": self.seed,
                "amplitude": self.amplitude, "spectral_bound": self.spectral_bound, "alpha": self.alpha}

    def lipschitz_bound(self):
        inverse_scale = self.scales[list(self.features)].reciprocal().max().item()
        return self.alpha * self.spectral_bound ** len(self.layers) * inverse_scale


class PhaseSpaceSDE(SDEModel, ParameterGroupProvider):
    """Canonical four-dimensional contract; position rows cannot be learned."""

    def __init__(self, acceleration, velocity_factor, spec):
        super().__init__()
        _require(isinstance(spec, DynamicsSpec), "frozen dynamics specification required")
        _require(type(acceleration) in (AffineAccelerationDrift, RBFResidualDrift,
                                       SplineResidualDrift, NeuralResidualAccelerationDrift), "unknown drift family")
        base = acceleration if isinstance(acceleration, AffineAccelerationDrift) else acceleration.affine
        _require(base.context_dim == spec.context_dim and
                 (not isinstance(acceleration, _ResidualDrift) or acceleration.spec == spec), "context/normalizer mismatch")
        factor = torch.as_tensor(velocity_factor, dtype=torch.float32)
        _require(factor.shape == (2, 2) and torch.isfinite(factor).all(), "velocity factor must be finite 2 by 2")
        self.acceleration_model, self.spec = acceleration, spec
        self.register_buffer("velocity_factor", factor.clone())

    @property
    def state_dim(self):
        return 4

    @property
    def noise_dim(self):
        return 2

    def _inputs(self, t, z, context):
        _require(isinstance(z, torch.Tensor) and z.ndim == 2 and z.shape[1] == 4
                 and z.dtype in (torch.float32, torch.float64) and torch.isfinite(z).all(), "finite state [B,4] required")
        _require(z.dtype == self.velocity_factor.dtype and z.device == self.velocity_factor.device,
                 "state and model dtype/device differ")
        _require(isinstance(t, torch.Tensor) and t.shape == z.shape[:1]
                 and t.dtype == z.dtype and t.device == z.device and torch.isfinite(t).all(), "time must be finite [B] in seconds")
        _require(isinstance(context, ModelContext) and context.regime is None, "K=1 requires no latent regime")
        c = context.condition
        if c is None and self.spec.context_dim == 0:
            c = z.new_empty((len(z), 0))
        _require(isinstance(c, torch.Tensor) and c.shape == (len(z), self.spec.context_dim)
                 and c.dtype == z.dtype and c.device == z.device and torch.isfinite(c).all(), "frozen context [B,C] required")
        return c

    def acceleration(self, t, z, context):
        c = self._inputs(t, z, context)
        result = self.acceleration_model(t, z, c)
        _require(result.shape == (len(z), 2) and torch.isfinite(result).all(), "nonfinite acceleration")
        return result

    def drift(self, t, z, context):
        return torch.cat((z[:, 2:], self.acceleration(t, z, context)), dim=-1)

    def diffusion(self, t, z, context):
        self._inputs(t, z, context)
        _require(torch.isfinite(self.velocity_factor).all(), "nonfinite diffusion")
        factor = torch.cat((self.velocity_factor.new_zeros((2, 2)), self.velocity_factor), dim=0)
        return factor.expand(len(z), 4, 2)

    def parameter_groups(self):
        return {ParameterRole.DRIFT: tuple(self.acceleration_model.parameters()), ParameterRole.DIFFUSION: ()}

    def supports(self, objective, engine, resume_level):
        if objective in ("L1", "L2"):
            return {"supported": False, "reason": "INAPPLICABLE: current observed-state model has no registered latent observation/recognition contract"}
        allowed = objective in ("O1", "O2") and engine == "generic-rollout" and resume_level == "restart-only"
        return {"supported": allowed, "reason": None if allowed else "model capability not implemented/qualified"}

    def model_card(self):
        return {"schema_version": "pirc26-model-card-v1", "state_names": ["x", "y", "vx", "vy"],
                "units": ["m", "m", "m/s", "m/s"], "time_unit": "s", "noise_dim": 2,
                "diffusion_support": ["vx", "vy"], "spec": {**asdict(self.spec),
                    "means": list(self.spec.means), "scales": list(self.spec.scales)},
                "family": self.acceleration_model.family, "dtype": str(self.velocity_factor.dtype),
                "device": str(self.velocity_factor.device), "qualification": "unqualified",
                "velocity_covariance": (self.velocity_factor @ self.velocity_factor.T).detach().cpu().tolist()}

    def checkpoint(self):
        state = {}
        _require(sum(t.numel() for t in self.state_dict().values()) <= 100_000, "checkpoint tensor quota")
        for name, value in self.state_dict().items():
            _require(torch.isfinite(value).all(), "nonfinite checkpoint tensor")
            data = value.detach().cpu().tolist()
            item = {"shape": list(value.shape), "dtype": str(value.dtype), "data": data}
            state[name] = {**item, "sha256": _hash(item)}
        payload = {"schema_version": "pirc26-dynamics-checkpoint-v1", "model_card": self.model_card(),
                   "configuration": self.acceleration_model.configuration(), "state": state}
        return {**payload, "sha256": _hash(payload)}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Load bounded JSON data only; no pickle or user-supplied executable factory."""
        try:
            from infrastructure.pirc26_checkpoint_contract import inspect_checkpoint, CheckpointContractError
            try:
                document = inspect_checkpoint(checkpoint)["document"]
            except CheckpointContractError as exc:
                raise ModelContractError(str(exc)) from exc
            identity = document.pop("sha256")
            _require(identity == _hash(document) and document["schema_version"] == "pirc26-dynamics-checkpoint-v1",
                     "checkpoint identity/schema mismatch")
            card, config = document["model_card"], document["configuration"]
            _require(card["state_names"] == ["x", "y", "vx", "vy"] and card["units"] == ["m", "m", "m/s", "m/s"]
                     and card["time_unit"] == "s" and card["noise_dim"] == 2
                     and card["diffusion_support"] == ["vx", "vy"] and card["family"] == config["family"],
                     "model card contract mismatch")
            _require(card["dtype"] in ("torch.float32", "torch.float64") and card["device"] == "cpu",
                     "checkpoint loader requires an explicit CPU float32/64 export")
            spec = DynamicsSpec(**card["spec"])
            affine = AffineAccelerationDrift(spec.context_dim)
            family = config["family"]
            if family == "M0":
                acceleration = affine
            elif family == "M1-R":
                acceleration = RBFResidualDrift(affine, spec, config["features"], config["centers"], config["bandwidth"])
            elif family == "M1-S":
                _require(config["extrapolation"] == "clamp", "spline extrapolation contract")
                acceleration = SplineResidualDrift(affine, spec, config["features"], config["knots"], config["degree"])
            elif family == "M2":
                acceleration = NeuralResidualAccelerationDrift(affine, spec, config["features"], config["hidden"],
                    seed=config["seed"], amplitude=config["amplitude"], spectral_bound=config["spectral_bound"], alpha=config["alpha"])
            else:
                raise ModelContractError("MODEL_CONTRACT_ERROR: unknown checkpoint model")
            if isinstance(acceleration, _ResidualDrift):
                _require(type(config["residual_enabled"]) is bool, "residual gate type")
                acceleration.residual_enabled = config["residual_enabled"]
            model = cls(acceleration, torch.zeros(2, 2), spec).to(dtype=getattr(torch, card["dtype"].split(".")[-1]))
            expected = model.state_dict()
            _require(document["state"].keys() == expected.keys(), "checkpoint parameter names mismatch")
            tensors = {}
            for name, template in expected.items():
                item = dict(document["state"][name])
                tensor_hash = item.pop("sha256")
                _require(tensor_hash == _hash(item) and item["shape"] == list(template.shape)
                         and item["dtype"] == str(template.dtype), "checkpoint tensor identity/shape/dtype mismatch")
                value = torch.tensor(item["data"], dtype=template.dtype)
                # JSON cannot encode the trailing dimension of an empty [2,0] tensor.
                _require(value.numel() == template.numel(), "checkpoint tensor size mismatch")
                _require(value.numel() == 0 or value.shape == template.shape, "checkpoint tensor nesting mismatch")
                value = value.reshape(template.shape)
                _require(torch.isfinite(value).all(), "checkpoint tensor is nonfinite")
                tensors[name] = value
            model.load_state_dict(tensors)
            _require(model.model_card() == card and model.acceleration_model.configuration() == config,
                     "checkpoint metadata does not describe its weights")
            if isinstance(acceleration, _ResidualDrift):
                _require(torch.equal(acceleration.means, torch.tensor(spec.means, dtype=torch.float32).to(model.velocity_factor.dtype))
                         and torch.equal(acceleration.scales, torch.tensor(spec.scales, dtype=torch.float32).to(model.velocity_factor.dtype)),
                         "checkpoint normalizer differs from frozen provenance")
            return model
        except (KeyError, TypeError, RuntimeError, OverflowError, ValueError) as exc:
            if isinstance(exc, ModelContractError):
                raise
            raise ModelContractError("MODEL_CONTRACT_ERROR: malformed checkpoint") from exc


def stability_diagnostics(model, t, z, context):
    """Local stress diagnostics; this does not certify long-horizon stability."""
    state = z.detach().clone().requires_grad_(True)
    acceleration = model.acceleration(t, state, context)
    rows = [torch.autograd.grad(acceleration[:, i].sum(), state, retain_graph=True)[0] for i in range(2)]
    jacobian = torch.stack(rows, dim=1)
    drift = model.drift(t, state, context)
    result = {"samples": len(z), "max_acceleration": acceleration.norm(dim=-1).max().item(),
              "max_jacobian_norm": torch.linalg.matrix_norm(jacobian, ord=2).max().item(),
              "max_radial_growth": (state * drift).sum(-1).max().item(), "drift_clipping": False}
    if isinstance(model.acceleration_model, NeuralResidualAccelerationDrift):
        result["residual_lipschitz_upper_bound"] = model.acceleration_model.lipschitz_bound()
    return result
