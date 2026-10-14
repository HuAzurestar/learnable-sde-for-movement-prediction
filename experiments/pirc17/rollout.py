"""Causal Euler-Maruyama kernel; not an alias for other NEX326 integrators.

Callbacks receive predicted state only, never a trajectory/target container.
Map providers must query static assets at these positions. The step-size and
history sampling semantics belong to this version, not the frozen PIRC-22 fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .origins import HISTORY_POINTS, Origin, frozen_array, motion

ROLLOUT_VERSION = "pirc17-causal-euler-maruyama-v4"
DEFAULT_HISTORY_STEP_SECONDS = 5.0


@dataclass(frozen=True)
class PredictedState:
    elapsed_seconds: float
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    history_positions_m: np.ndarray  # particle, visible/predicted point, xy
    history_elapsed_seconds: np.ndarray

    @property
    def history_direction(self):
        if self.history_positions_m.shape[1] < 2:
            # Known velocity or a sampled point-only prior supplies direction.
            return motion(self.velocities_mps)[1:]
        displacement = self.history_positions_m[:, -1] - self.history_positions_m[:, 0]
        return motion(displacement)[1:]


@dataclass(frozen=True)
class Features:
    values: np.ndarray
    valid: np.ndarray

    def model_matrix(self, particles):
        values = np.asarray(self.values, dtype=float)
        valid = np.asarray(self.valid)
        if values.ndim != 2 or values.shape[0] != particles or valid.shape != values.shape:
            raise ValueError("feature values/masks do not match particles")
        if valid.dtype != np.bool_ or not np.isfinite(values[valid]).all():
            raise ValueError("boolean masks and finite valid features required")
        # Same numeric-zero + explicit validity convention as PIRC-21.
        return frozen_array(np.column_stack((np.where(valid, values, 0.), valid.astype(float))))


@dataclass(frozen=True)
class Forecast:
    elapsed_seconds: np.ndarray
    positions_m: np.ndarray  # particle, requested horizon, xy
    invalid_feature_rows: int
    feature_query_rows: int
    version: str = ROLLOUT_VERSION


def _horizons(values):
    horizons = frozen_array(values)
    if horizons.ndim != 1 or len(horizons) == 0 or horizons[0] <= 0 or np.any(np.diff(horizons) <= 0):
        raise ValueError("strictly increasing positive horizons required")
    return horizons


def inertial(origin: Origin, horizons_seconds, *, particles: int, seed: int):
    """Equal-origin-information diagnostic, not a new numbered method arm."""
    horizons = _horizons(horizons_seconds)
    velocities = origin.sample_velocities(particles, np.random.default_rng(seed))
    positions = origin.position_m[None, None, :] + velocities[:, None, :] * horizons[None, :, None]
    return Forecast(horizons, frozen_array(positions), 0, 0, "pirc17-inertial-v1")


def rollout(origin: Origin, horizons_seconds, *, particles: int, seed: int,
            max_step_seconds: float, base_drift: Callable, diffusion: Callable,
            terrain: Callable | None = None, conditioner: Callable | None = None,
            history_step_seconds: float = DEFAULT_HISTORY_STEP_SECONDS,
            brownian_increments: Callable | None = None):
    """Integrate dx = (base + conditioner) dt + diffusion dW in metric units.

    Diffusion must return (particle, 2, noise_dimension), in m/sqrt(s).
    Terrain returns numeric values and validity, conditioner consumes their
    zero-imputed + mask matrix. Missing queries are counted, never dropped.
    Drift may use estimated velocity, but no model weights are fitted here.
    History is sampled at fixed physical times k * history_step_seconds, not at
    integration steps or requested outputs. Between ticks velocity/history are
    held; terrain still sees current predicted positions. Cadence is a model
    parameter that must be sealed separately from numerical step size.
    An optional registered Brownian driver supplies dW (already scaled in sqrt(s))
    at the requested interval. It permits shared paths during step refinement.
    """
    horizons = _horizons(horizons_seconds)
    if not np.isfinite(max_step_seconds) or max_step_seconds <= 0:
        raise ValueError("positive finite step required")
    if not np.isfinite(history_step_seconds) or history_step_seconds <= 0:
        raise ValueError("positive finite history cadence required")
    if (terrain is None) != (conditioner is None):
        raise ValueError("terrain and conditioner must be supplied together")
    rng = np.random.default_rng(seed)
    velocities = origin.sample_velocities(particles, rng)
    positions = np.broadcast_to(origin.position_m, (particles, 2)).copy()
    history = np.broadcast_to(origin.history_positions_m, (particles, *origin.history_positions_m.shape)).copy()
    history_times = origin.history_times_seconds - origin.epoch_seconds
    elapsed = 0.
    history_tick = 1
    next_history_time = float(history_step_seconds)
    output = []
    missing = queries = 0
    for horizon in horizons:
        while elapsed < horizon:
            end_time = min(elapsed + max_step_seconds, float(horizon), next_history_time)
            dt = end_time - elapsed
            if dt <= 0:
                raise ValueError("step too small for floating-point time resolution")
            state = PredictedState(elapsed, frozen_array(positions), frozen_array(velocities),
                                   frozen_array(history), frozen_array(history_times))
            drift = frozen_array(base_drift(state), (particles, 2)).copy()
            if terrain is not None:
                features = terrain(state)
                matrix = features.model_matrix(particles)
                queries += particles
                missing += int(np.count_nonzero(~np.asarray(features.valid).all(axis=1)))
                drift += frozen_array(conditioner(state, matrix), (particles, 2))
            noise = frozen_array(diffusion(state))
            if noise.ndim != 3 or noise.shape[:2] != (particles, 2) or noise.shape[2] < 1:
                raise ValueError("diffusion must have shape (particles,2,noise_dimension)")
            if brownian_increments is None:
                noise_delta = np.einsum("pij,pj->pi", noise,
                    rng.standard_normal((particles, noise.shape[2]))) * np.sqrt(dt)
            else:
                dW = frozen_array(brownian_increments(elapsed,end_time,particles,noise.shape[2]),
                                  (particles,noise.shape[2]))
                noise_delta = np.einsum("pij,pj->pi",noise,dW)
            delta = drift * dt + noise_delta
            positions = frozen_array(positions + delta)
            elapsed = end_time
            if elapsed == next_history_time:
                history = np.concatenate((history, positions[:, None, :]), axis=1)[:, -HISTORY_POINTS:]
                history_times = np.append(history_times, elapsed)[-HISTORY_POINTS:]
                # Same three-point secant convention as causal_prefix(), using
                # predicted positions at physical ticks, never true future fixes.
                velocities = (history[:, -1] - history[:, 0]) / (history_times[-1] - history_times[0])
                history_tick += 1
                next_history_time = history_tick * float(history_step_seconds)
        output.append(positions.copy())
    return Forecast(horizons, frozen_array(np.stack(output, axis=1)), missing, queries)
