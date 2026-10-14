"""Exact constant-affine moments and endpoint half-space probabilities.

Covariance uses a short-interval Van Loan exponential and semigroup doubling.
This avoids the exponentially growing inverse block at a long stable horizon.
It is a floating-point analytic reference, not a PDE or first-passage solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import FrozenDynamicsPackage


def _finite_tensor(value, shape, name):
    try:
        result = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise DataValidationError(name + " is not a numerical array") from exc
    if tuple(result.shape) != shape or not torch.isfinite(result).all():
        raise DataValidationError(name + " has invalid shape or nonfinite values")
    return result


def _covariance(value, name):
    result = _finite_tensor(value, (4, 4), name)
    if not torch.allclose(result, result.T, rtol=0, atol=1e-12):
        raise DataValidationError(name + " is not symmetric")
    # Accept only roundoff-sized negative eigenvalues. Never project or add jitter.
    scale = max(1.0, float(torch.linalg.matrix_norm(result, ord=2)))
    if float(torch.linalg.eigvalsh(result).min()) < -1e-12 * scale:
        raise DataValidationError(name + " is not positive semidefinite")
    return result


@dataclass(frozen=True)
class AffineTransition:
    F: torch.Tensor
    offset: torch.Tensor
    covariance: torch.Tensor
    doubling_steps: int


def exact_transition(package: FrozenDynamicsPackage, dt: float) -> AffineTransition:
    """Return F, c, Q for X(t+dt)|X(t) ~ N(F X(t)+c, Q)."""
    if not isinstance(package, FrozenDynamicsPackage):
        raise DataValidationError("a frozen dynamics package is required")
    package.validate()
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt < 0:
        raise DataValidationError("transition interval must be finite and nonnegative")
    parameters = package.manifest()["parameters"]
    A = torch.tensor(parameters["A"], dtype=torch.float64)
    b = torch.tensor(parameters["b"], dtype=torch.float64)
    L = torch.tensor(parameters["L"], dtype=torch.float64)
    if dt == 0:
        return AffineTransition(torch.eye(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64),
                                torch.zeros((4, 4), dtype=torch.float64), 0)
    scaled_norm = float(torch.linalg.matrix_norm(A, ord=1)) * dt
    if not math.isfinite(scaled_norm):
        raise NumericalError("affine horizon exceeds finite numerical range")
    doublings = max(0, math.ceil(math.log2(scaled_norm))) if scaled_norm > 1 else 0
    if doublings > 60:
        raise NumericalError("affine horizon exceeds the declared 60-doubling limit")
    interval = math.ldexp(float(dt), -doublings)
    augmented = torch.zeros((5, 5), dtype=torch.float64)
    augmented[:4, :4], augmented[:4, 4] = A, b
    exponential = torch.linalg.matrix_exp(augmented * interval)
    F, offset = exponential[:4, :4], exponential[:4, 4]
    van_loan = torch.zeros((8, 8), dtype=torch.float64)
    van_loan[:4, :4], van_loan[:4, 4:], van_loan[4:, 4:] = A, L @ L.T, -A.T
    Q = torch.linalg.matrix_exp(van_loan * interval)[:4, 4:] @ F.T
    Q = (Q + Q.T) / 2
    for _ in range(doublings):
        Q, offset, F = Q + F @ Q @ F.T, offset + F @ offset, F @ F
        Q = (Q + Q.T) / 2
    if not all(torch.isfinite(value).all() for value in (F, offset, Q)):
        raise NumericalError("nonfinite affine transition")
    try:
        _covariance(Q, "transition covariance")
    except DataValidationError as exc:
        raise NumericalError("invalid analytic transition covariance") from exc
    return AffineTransition(F, offset, Q, doublings)


def exact_moments(package, initial_mean, initial_covariance, horizon):
    """Integrate a Gaussian initial distribution, including its uncertainty."""
    mean = _finite_tensor(initial_mean, (4,), "initial mean")
    covariance = _covariance(initial_covariance, "initial covariance")
    transition = exact_transition(package, horizon)
    result_mean = transition.F @ mean + transition.offset
    result_covariance = transition.F @ covariance @ transition.F.T + transition.covariance
    if not torch.isfinite(result_mean).all() or not torch.isfinite(result_covariance).all():
        raise NumericalError("nonfinite affine moments")
    return result_mean, _covariance(result_covariance, "forecast covariance")


def endpoint_halfspace_probability(mean, covariance, normal, threshold, *, closed=True):
    """P(normal @ X >= threshold), or > for an open boundary.

Only an endpoint event. A degenerate projection uses the stated boundary rule.
The erfc tail retains small nonzero probabilities without 1-CDF cancellation.
"""
    mean = _finite_tensor(mean, (4,), "endpoint mean")
    covariance = _covariance(covariance, "endpoint covariance")
    normal = _finite_tensor(normal, (4,), "half-space normal")
    if not torch.any(normal != 0) or type(threshold) not in (int, float) or not math.isfinite(threshold) or type(closed) is not bool:
        raise DataValidationError("invalid half-space geometry or boundary")
    # Scaling leaves the geometry unchanged and prevents avoidable overflow.
    scale = float(normal.abs().max())
    normal = normal / scale
    threshold = threshold / scale
    projected_mean = float(normal @ mean)
    variance = float(normal @ covariance @ normal)
    if not math.isfinite(projected_mean) or not math.isfinite(variance) or variance < 0:
        raise NumericalError("invalid projected Gaussian moments")
    if variance == 0:
        return float(projected_mean >= threshold if closed else projected_mean > threshold)
    return 0.5 * math.erfc((threshold - projected_mean) / math.sqrt(2 * variance))
