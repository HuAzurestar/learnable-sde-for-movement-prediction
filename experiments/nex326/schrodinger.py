"""Finite-particle Schrödinger bridge under an isotropic Brownian reference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class SchrodingerBridgeError(ValueError):
    """The registered particle bridge cannot produce a converged coupling."""


@dataclass(frozen=True)
class ParticleBridgeResult:
    endpoint_samples: np.ndarray
    paths: np.ndarray
    diagnostics: dict[str, object]


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    reduced = np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(maximum + reduced, axis=axis)


def _sinkhorn_coupling(
    cost: np.ndarray,
    epsilon: float,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int, float]:
    count = cost.shape[0]
    log_kernel = -cost / epsilon
    log_mass = np.full(count, -math.log(count))
    log_u = np.zeros(count)
    log_v = np.zeros(count)
    marginal_error = float("inf")
    coupling = np.empty_like(cost)
    for iteration in range(1, max_iterations + 1):
        log_u = log_mass - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_mass - _logsumexp(log_kernel.T + log_u[None, :], axis=1)
        log_coupling = log_u[:, None] + log_kernel + log_v[None, :]
        coupling = np.exp(log_coupling)
        marginal_error = float(
            max(
                np.max(np.abs(coupling.sum(axis=1) - 1.0 / count)),
                np.max(np.abs(coupling.sum(axis=0) - 1.0 / count)),
            )
        )
        if marginal_error <= tolerance:
            return coupling, iteration, marginal_error
    raise SchrodingerBridgeError(
        "particle Sinkhorn coupling did not converge: "
        f"error={marginal_error:.6g}, tolerance={tolerance:.6g}, "
        f"iterations={max_iterations}"
    )


def solve_particle_schrodinger_bridge(
    forecast_samples: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    rng: np.random.Generator,
    *,
    epsilon_scale: float,
    time_steps: int,
    max_iterations: int,
    tolerance: float,
) -> ParticleBridgeResult:
    """Solve a discrete endpoint coupling and sample Brownian bridge paths.

    The forecast particles define the initial marginal and fresh samples from the
    registered Gaussian prior define the terminal marginal.  Log-domain Sinkhorn
    scaling solves the finite entropic coupling.  Conditional Brownian bridges then
    provide complete paths for sampled endpoint pairs.
    """
    samples = np.asarray(forecast_samples, dtype=float)
    mean = np.asarray(prior_mean, dtype=float)
    covariance = np.asarray(prior_covariance, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 2:
        raise SchrodingerBridgeError("forecast marginal requires at least two 2D particles")
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise SchrodingerBridgeError("endpoint marginal must be a 2D Gaussian")
    if epsilon_scale <= 0.0 or time_steps < 2 or max_iterations < 1 or tolerance <= 0.0:
        raise SchrodingerBridgeError("registered bridge solver parameters are invalid")
    if not np.isfinite(samples).all() or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise SchrodingerBridgeError("bridge marginals contain non-finite values")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise SchrodingerBridgeError("endpoint covariance is not symmetric")
    if float(np.linalg.eigvalsh(covariance).min()) <= 0.0:
        raise SchrodingerBridgeError("endpoint covariance is not positive definite")

    count = len(samples)
    targets = rng.multivariate_normal(mean, covariance, size=count)
    differences = samples[:, None, :] - targets[None, :, :]
    cost = 0.5 * np.sum(differences**2, axis=2)
    forecast_covariance = np.cov(samples, rowvar=False)
    reference_variance = max(
        float(0.5 * (np.trace(forecast_covariance) + np.trace(covariance))),
        1e-12,
    )
    epsilon = epsilon_scale * reference_variance
    coupling, iterations, marginal_error = _sinkhorn_coupling(
        cost,
        epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    flat = coupling.ravel()
    flat /= flat.sum()
    pair_indices = rng.choice(flat.size, size=count, replace=True, p=flat)
    start_indices, target_indices = np.unravel_index(pair_indices, coupling.shape)
    starts = samples[start_indices]
    endpoints = targets[target_indices]
    times = np.linspace(0.0, 1.0, time_steps + 1)
    increments = rng.normal(
        scale=math.sqrt(epsilon / time_steps),
        size=(count, time_steps, 2),
    )
    brownian = np.concatenate(
        [np.zeros((count, 1, 2)), np.cumsum(increments, axis=1)],
        axis=1,
    )
    bridge_noise = brownian - times[None, :, None] * brownian[:, -1:, :]
    paths = (
        (1.0 - times[None, :, None]) * starts[:, None, :]
        + times[None, :, None] * endpoints[:, None, :]
        + bridge_noise
    )
    endpoint_covariance = np.cov(endpoints, rowvar=False)
    diagnostics: dict[str, object] = {
        "solver": "log_sinkhorn_brownian_particle_bridge",
        "reference_process": "isotropic_brownian",
        "particle_count": count,
        "epsilon_scale": epsilon_scale,
        "epsilon_absolute": epsilon,
        "time_steps": time_steps,
        "max_iterations": max_iterations,
        "iterations": iterations,
        "tolerance": tolerance,
        "max_marginal_error": marginal_error,
        "converged": True,
        "transport_cost": float(np.sum(coupling * cost)),
        "terminal_mean_error": float(np.linalg.norm(endpoints.mean(axis=0) - mean)),
        "terminal_covariance_error": float(np.linalg.norm(endpoint_covariance - covariance)),
    }
    return ParticleBridgeResult(
        endpoint_samples=endpoints,
        paths=paths,
        diagnostics=diagnostics,
    )


__all__ = [
    "ParticleBridgeResult",
    "SchrodingerBridgeError",
    "solve_particle_schrodinger_bridge",
]
