"""PIRC-26 model identity/export and adapters to existing affine contracts."""

from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess

import torch

from domain import ModelContext
from infrastructure.research_files import source_file_hash
from infrastructure.research_store import ResearchError, digest
from models.base import ExactGaussianKernelMixin
from models.phase_space import (AffineAccelerationDrift, DynamicsSpec, ModelContractError,
                               PhaseSpaceSDE)
from .research_contracts import accept_model


SOURCE_FILES = ("application/pirc26_dynamics.py", "models/phase_space.py", "models/base.py",
                "domain/__init__.py", "domain/types.py", "domain/errors.py", "numerics.py",
                "experiments/nex326/phase_space.py")


def dynamics_identity(root=None):
    """Literal Git head and bounded, reverified source identities; no data reads."""
    root = Path(root or Path(__file__).resolve().parents[1]).absolute()
    files = {name: source_file_hash(root, root / name, maximum_bytes=2 * 1024 * 1024)
             for name in SOURCE_FILES}
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ResearchError("CONTRACT_MISMATCH", "literal dynamics code SHA missing")
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", *SOURCE_FILES],
        check=True, capture_output=True, text=True, timeout=10).stdout.strip())
    return {"schema_version": "pirc26-dynamics-code-v1", "code_sha": head,
            "source_tree_hash": digest(files), "files": files, "source_tree_dirty": dirty}


def freeze_dynamics(model, *, protocol_hash, root=None):
    """Export an explicitly unqualified package; study qualification comes later.

    A model component passing fixtures cannot manufacture a formal training,
    forecast or preregistration receipt. PIRC-27/28 consumers can inspect this
    artifact, but formal nonlinear comparisons will reject its qualification.
    """
    if not isinstance(model, PhaseSpaceSDE) or not re.fullmatch(r"[0-9a-f]{64}", str(protocol_hash)):
        raise ResearchError("CONTRACT_MISMATCH", "model and frozen protocol identity required")
    identity = dynamics_identity(root)
    checkpoint = model.checkpoint()
    card = {**model.model_card(), "model_id": model.acceleration_model.family + "-" + checkpoint["sha256"][:16],
            "implementation": identity, "implementation_hash": identity["source_tree_hash"],
            "code_sha": identity["code_sha"], "checkpoint_hash": checkpoint["sha256"]}
    package = {"schema_version": "pirc25-package-v1", "kind": "FrozenDynamicsPackage",
        "state_order": card["state_names"], "units": card["units"],
        "capabilities": ["generic-rollout"], "resume_level": "restart-only",
        "code_hash": digest(identity), "data_hash": model.spec.train_binding_hash,
        "input_hash": digest(card["spec"]), "output_hash": checkpoint["sha256"],
        "protocol_hash": protocol_hash, "qualification": "fixture",
        "model_card": card, "checkpoint": checkpoint}
    if isinstance(model.acceleration_model, AffineAccelerationDrift):
        package["capabilities"].append("exact-transition")
    accept_model(package)
    return package


def load_frozen_dynamics(package, *, protocol_hash, root=None, formal=False):
    """Cross-check the package, code, card and weights before returning a model."""
    if formal or package.get("qualification") != "fixture":
        raise ResearchError("UNQUALIFIED", "formal dynamics need independently admitted study qualification")
    accept_model(package, formal=formal)
    identity = dynamics_identity(root)
    card = package.get("model_card", {})
    if (package["protocol_hash"] != protocol_hash or package["code_hash"] != digest(identity)
            or card.get("implementation") != identity or card.get("code_sha") != identity["code_sha"]
            or card.get("implementation_hash") != identity["source_tree_hash"]):
        raise ResearchError("CONTRACT_MISMATCH", "frozen model code/protocol identity differs")
    try:
        model = PhaseSpaceSDE.from_checkpoint(package["checkpoint"])
        expected = freeze_dynamics(model, protocol_hash=protocol_hash, root=root)
    except (ModelContractError, KeyError, TypeError, ValueError) as exc:
        raise ResearchError("CORRUPT_ARTIFACT", "frozen dynamics checkpoint incompatible") from exc
    # Qualification fields are independently validated by the shared contract.
    # No extra caller claims can change model/data/normalizer/capability identity.
    for key, value in expected.items():
        if key != "qualification" and package.get(key) != value:
            raise ResearchError("CORRUPT_ARTIFACT", "frozen dynamics metadata differs from verified weights")
    return model


class AffinePhaseSpaceOracle(ExactGaussianKernelMixin):
    """Adapt M0 to the existing analytic affine transition implementation.

    This is an oracle diagnostic, not the main shared forecast engine. Its
    optional context is a single frozen vector constant over the interval.
    """

    def __init__(self, model):
        self.model = model
        self._base()

    def _base(self):
        acceleration = self.model.acceleration_model
        if isinstance(acceleration, AffineAccelerationDrift):
            return acceleration
        if acceleration.residual_enabled is False:
            return acceleration.affine
        raise ModelContractError("MODEL_CONTRACT_ERROR: nonlinear residual has no exact affine oracle")

    def _condition(self, context):
        base = self._base()
        if not isinstance(context, ModelContext) or context.regime is not None:
            raise ModelContractError("MODEL_CONTRACT_ERROR: oracle requires K=1 context")
        condition = context.condition
        if condition is None and base.context_dim == 0:
            return base.a.new_empty((0,))
        if (not isinstance(condition, torch.Tensor) or condition.shape != (base.context_dim,)
                or condition.dtype != base.a.dtype or condition.device != base.a.device
                or not torch.isfinite(condition).all()):
            raise ModelContractError("MODEL_CONTRACT_ERROR: oracle context must be a constant [C] vector")
        return condition

    def constant_drift(self, context):
        base = self._base()
        acceleration = base.a + base.context_weight @ self._condition(context)
        return torch.cat((base.a.new_zeros(2), acceleration))

    def drift_matrix(self, context):
        base = self._base()
        self._condition(context)
        top = torch.cat((base.A.new_zeros((2, 2)), torch.eye(2, dtype=base.A.dtype, device=base.A.device)), dim=1)
        return torch.cat((top, torch.cat((base.B, base.A), dim=1)), dim=0)

    def diffusion_matrix(self, context):
        self._condition(context)
        factor = self.model.velocity_factor
        if not torch.isfinite(factor).all():
            raise ModelContractError("MODEL_CONTRACT_ERROR: nonfinite oracle velocity factor")
        return torch.cat((factor.new_zeros((2, 2)), factor), dim=0)

    def affine_transition(self, dt, context):
        if isinstance(dt, torch.Tensor):
            if dt.ndim != 0 or dt.dtype != self.model.velocity_factor.dtype or dt.device != self.model.velocity_factor.device:
                raise ModelContractError("MODEL_CONTRACT_ERROR: scalar compatible oracle interval required")
            interval = dt.item()
        elif type(dt) in (float, int):
            interval = dt
        else:
            raise ModelContractError("MODEL_CONTRACT_ERROR: scalar oracle interval required")
        if not math.isfinite(interval) or interval <= 0:
            raise ModelContractError("MODEL_CONTRACT_ERROR: positive finite oracle interval required")
        result = super().affine_transition(dt, context)
        if not all(torch.isfinite(value).all() for value in result):
            raise ModelContractError("MODEL_CONTRACT_ERROR: affine transition overflow")
        return result


def adapt_legacy_affine(legacy, spec: DynamicsSpec):
    """Recover the physical coefficients of the existing direct-feature model.

    Directional/terrain feature bases contain nonlinear derived terms and are
    refused here rather than being mislabeled as an affine four-dimensional M0.
    """
    from experiments.nex326.phase_space import AffineVelocityModel
    if not isinstance(legacy, AffineVelocityModel) or legacy.feature_basis != "direct":
        raise ModelContractError("MODEL_CONTRACT_ERROR: only direct legacy affine features can be adapted")
    if len(legacy.condition_names) != spec.context_dim or tuple(legacy.feature_names) != ("vx", "vy", *legacy.condition_names):
        raise ModelContractError("MODEL_CONTRACT_ERROR: legacy context/features differ")
    mean = torch.as_tensor(legacy.feature_mean, dtype=torch.float64)
    scale = torch.as_tensor(legacy.feature_scale, dtype=torch.float64)
    weights = torch.as_tensor(legacy.weights, dtype=torch.float64)
    covariance = torch.as_tensor(legacy.diffusion_covariance, dtype=torch.float64)
    count = 2 + spec.context_dim
    if (mean.shape != (count,) or scale.shape != (count,) or weights.shape != (count + 1, 2)
            or covariance.shape != (2, 2) or not all(torch.isfinite(t).all() for t in (mean, scale, weights, covariance))
            or not (scale > 0).all() or not torch.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12)):
        raise ModelContractError("MODEL_CONTRACT_ERROR: invalid legacy normalization or covariance")
    values, vectors = torch.linalg.eigh(covariance)
    if (values < 0).any():
        raise ModelContractError("MODEL_CONTRACT_ERROR: legacy covariance is not positive semidefinite")
    factor = vectors @ torch.diag(values.sqrt())
    affine = AffineAccelerationDrift(spec.context_dim).double()
    with torch.no_grad():
        coefficients = weights[1:] / scale[:, None]
        affine.A.copy_(coefficients[:2].T)
        affine.context_weight.copy_(coefficients[2:].T)
        affine.a.copy_(weights[0] - mean @ coefficients)
    # Preserve float64 adapter precision: PhaseSpaceSDE normally builds f32 L.
    model = PhaseSpaceSDE(affine, torch.zeros((2, 2)), spec).double()
    model.velocity_factor.copy_(factor)
    return model
