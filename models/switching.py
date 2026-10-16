"""K<=2 CTMC switching acceleration SDE with exact *conditional* kernels.

The joint switching transition is not Gaussian. This model deliberately does
not implement ExactTransitionProvider: only a fixed-mode segment has an exact
affine Gaussian kernel. State order is [x, y, vx, vy], time is in seconds.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from domain import ModelContext
from models.base import ExactGaussianKernelMixin, SDEModel


class SwitchingError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise SwitchingError("CONTRACT_MISMATCH", f"{name} must be finite with shape {shape}")
    return array.copy()


def psd_array(value, shape, name):
    array = finite_array(value, shape, name)
    if not np.allclose(array, np.swapaxes(array, -1, -2), rtol=0, atol=1e-12):
        raise SwitchingError("CONTRACT_MISMATCH", f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(array)) < -1e-12:
        raise SwitchingError("CONTRACT_MISMATCH", f"{name} must be positive semidefinite")
    return array


def gaussian_sample(mean, covariance, rng):
    """Preserve deterministic zero-noise cases; do not add sampling jitter."""
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(values))))
    if not np.isfinite(values).all() or np.min(values) < -tolerance:
        raise SwitchingError("NONFINITE", "conditional covariance is not positive semidefinite")
    return mean + (vectors * np.sqrt(np.maximum(values, 0))) @ rng.standard_normal(len(mean))


class SwitchingSDEModel(SDEModel):
    """dp=v dt; dv=(A[mode] p+B[mode] v+a[mode])dt+L[mode]dW.

    These frozen parameter buffers are estimates, not an inference algorithm.
    An explicit mode is required for drift/diffusion calls. Labels carry no
    behavioral meaning, and permuting labels permutes Q on both axes.
    """

    def __init__(self, Q, A, B, a, L, *, rate_bounds=(0.0, 100.0)):
        super().__init__()
        q = np.asarray(Q, dtype=np.float64)
        if q.ndim != 2 or q.shape[0] not in (1, 2) or q.shape[1] != q.shape[0]:
            raise SwitchingError("CONTRACT_MISMATCH", "first-round K must be 1 or 2")
        k = q.shape[0]
        q = finite_array(q, (k, k), "Q")
        lower, upper = rate_bounds
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper):
            raise SwitchingError("CONTRACT_MISMATCH", "rate bounds must be finite and ordered")
        off = q.copy()
        np.fill_diagonal(off, 0)
        selected = off[~np.eye(k, dtype=bool)]
        if (np.any(selected < lower) or np.any(selected > upper)
                or not np.allclose(q.sum(axis=1), 0, rtol=0, atol=1e-12)
                or np.any(np.diag(q) > 0)):
            raise SwitchingError("INVALID_GENERATOR", "Q needs bounded nonnegative rates and zero row sums")
        # Canonicalize only the checked diagonal residual; this also makes K=1
        # exactly absorbing rather than attempting a jump with zero destinations.
        np.fill_diagonal(q, -off.sum(axis=1))
        for name, value, shape in (("Q", q, (k, k)), ("A", A, (k, 2, 2)),
                ("B", B, (k, 2, 2)), ("a", a, (k, 2)), ("L", L, (k, 2, 2))):
            self.register_buffer(name, torch.from_numpy(finite_array(value, shape, name)))
        self.rate_bounds = (float(lower), float(upper))

    @classmethod
    def from_rate_logits(cls, logits, A, B, a, L, *, rate_bounds=(0.0, 100.0)):
        logits = finite_array(logits, (2,), "rate logits")
        rates = np.logaddexp(0, logits)
        q = [[-rates[0], rates[0]], [rates[1], -rates[1]]]
        return cls(q, A, B, a, L, rate_bounds=rate_bounds)

    @property
    def state_dim(self):
        return 4

    @property
    def noise_dim(self):
        return 2

    @property
    def n_modes(self):
        return len(self.Q)

    def _mode(self, mode):
        if type(mode) is not int or not 0 <= mode < self.n_modes:
            raise SwitchingError("CONTRACT_MISMATCH", "an explicit valid integer mode is required")
        return mode

    def coefficients(self, mode):
        mode = self._mode(mode)
        matrix = self.A.new_zeros((4, 4))
        matrix[:2, 2:] = torch.eye(2, dtype=self.A.dtype, device=self.A.device)
        matrix[2:, :2], matrix[2:, 2:] = self.A[mode], self.B[mode]
        offset = self.a.new_zeros(4)
        offset[2:] = self.a[mode]
        noise = self.L.new_zeros((4, 2))
        noise[2:] = self.L[mode]
        return matrix, offset, noise

    def drift(self, t, x, context: ModelContext):
        matrix, offset, _ = self.coefficients(context.regime)
        return x @ matrix.T + offset

    def diffusion(self, t, x, context: ModelContext):
        _, _, noise = self.coefficients(context.regime)
        return noise.expand(x.shape[:-1] + noise.shape)

    def conditional_affine_transition(self, mode, dt):
        if isinstance(dt, bool) or not math.isfinite(float(dt)) or float(dt) < 0:
            raise SwitchingError("CONTRACT_MISMATCH", "segment duration must be finite and nonnegative")
        matrix, offset, noise = self.coefficients(mode)
        augmented = matrix.new_zeros((5, 5))
        augmented[:4, :4], augmented[:4, 4] = matrix, offset
        exponential = torch.linalg.matrix_exp(augmented * float(dt))
        covariance = ExactGaussianKernelMixin.covariance_from_coefficients(matrix, noise, float(dt))
        if not torch.isfinite(exponential).all() or not torch.isfinite(covariance).all():
            raise SwitchingError("NONFINITE", "affine segment overflowed")
        return exponential[:4, :4], exponential[:4, 4], covariance

    def mode_transition(self, dt):
        if isinstance(dt, bool) or not math.isfinite(float(dt)) or float(dt) < 0:
            raise SwitchingError("CONTRACT_MISMATCH", "mode duration must be finite and nonnegative")
        return torch.linalg.matrix_exp(self.Q * float(dt))

    def permuted(self, order):
        if sorted(order) != list(range(self.n_modes)) or any(type(i) is not int for i in order):
            raise SwitchingError("CONTRACT_MISMATCH", "label map must be a permutation")
        arrays = {name: getattr(self, name).detach().cpu().numpy() for name in ("Q", "A", "B", "a", "L")}
        return type(self)(arrays["Q"][np.ix_(order, order)],
            *(arrays[name][order] for name in ("A", "B", "a", "L")), rate_bounds=self.rate_bounds)

    def to_dict(self):
        return {"schema_version": "pirc28-switching-model-v1", "K": self.n_modes,
            "state_order": ["x", "y", "vx", "vy"], "units": ["m", "m", "m/s", "m/s"],
            "time_unit": "s", "rate_unit": "s^-1", "rate_bounds": list(self.rate_bounds),
            "labels": [f"state-{i+1}" for i in range(self.n_modes)],
            **{name: getattr(self, name).detach().cpu().tolist() for name in ("Q", "A", "B", "a", "L")}}

    @classmethod
    def from_dict(cls, value):
        if (value.get("schema_version") != "pirc28-switching-model-v1"
                or value.get("state_order") != ["x", "y", "vx", "vy"]
                or value.get("units") != ["m", "m", "m/s", "m/s"]
                or value.get("time_unit") != "s" or value.get("rate_unit") != "s^-1"):
            raise SwitchingError("CONTRACT_MISMATCH", "model schema, state order or units differ")
        model = cls(*(value[name] for name in ("Q", "A", "B", "a", "L")), rate_bounds=value["rate_bounds"])
        if value.get("K") != model.n_modes or value.get("labels") != model.to_dict()["labels"]:
            raise SwitchingError("CONTRACT_MISMATCH", "model K or label declarations differ")
        return model
