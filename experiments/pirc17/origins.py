"""Versioned forecast origins. Public constructors accept no future observations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

ORIGIN_VERSION = "pirc17-origin-v1"
HISTORY_POINTS = 3  # PIRC-21 materialization default; asserted by the handoff.


def frozen_array(value, shape=None):
    result = np.array(value, dtype=float, copy=True)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("wrong shape or nonfinite input")
    result.setflags(write=False)
    return result


def motion(velocity):
    """Return speed and direction separately; stationary has no valid heading."""
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape[-1:] != (2,) or not np.isfinite(velocity).all():
        raise ValueError("finite 2D velocities required")
    speed = np.linalg.norm(velocity, axis=-1)
    valid = speed > 1e-12
    direction = velocity / np.where(valid, speed, 1.0)[..., None]
    return speed, direction, valid


@dataclass(frozen=True)
class Origin:
    mode: str
    epoch_seconds: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    history_positions_m: np.ndarray
    history_times_seconds: np.ndarray
    velocity_source: str
    velocity_prior_mps: np.ndarray | None = None
    velocity_observed_at_seconds: float | None = None
    velocity_error_mps: float | None = None  # None means unknown, never zero error.

    def __post_init__(self):
        if self.mode not in {"known_velocity", "causal_prefix", "point_only"}:
            raise ValueError("unknown origin mode")
        if not np.isfinite(self.epoch_seconds) or not self.velocity_source.strip():
            raise ValueError("finite origin time and velocity provenance required")
        if self.velocity_observed_at_seconds is not None and (
            not np.isfinite(self.velocity_observed_at_seconds) or self.velocity_observed_at_seconds > self.epoch_seconds
        ):
            raise ValueError("velocity observation must be finite and no later than origin")
        if self.velocity_error_mps is not None and (
            not np.isfinite(self.velocity_error_mps) or self.velocity_error_mps < 0
        ):
            raise ValueError("velocity error must be nonnegative or explicitly unknown")
        for key in ("position_m", "velocity_mps"):
            object.__setattr__(self, key, frozen_array(getattr(self, key), (2,)))
        positions = frozen_array(self.history_positions_m)
        times = frozen_array(self.history_times_seconds)
        if times.ndim != 1 or len(times) < 1 or positions.shape != (len(times), 2):
            raise ValueError("history must contain matched times and positions")
        if np.any(np.diff(times) <= 0) or times[-1] != self.epoch_seconds:
            raise ValueError("history must increase strictly and end at the origin")
        if not np.array_equal(positions[-1], self.position_m):
            raise ValueError("history endpoint differs from origin")
        if self.mode == "causal_prefix" and len(times) < 2:
            raise ValueError("causal prefix requires at least two observations")
        if self.mode != "causal_prefix" and len(times) != 1:
            raise ValueError("point/known-velocity origins cannot hide extra history")
        object.__setattr__(self, "history_positions_m", positions)
        object.__setattr__(self, "history_times_seconds", times)
        if self.mode == "point_only":
            prior = frozen_array(self.velocity_prior_mps)
            if prior.ndim != 2 or prior.shape[1] != 2 or len(prior) < 1:
                raise ValueError("point-only requires a train-fitted velocity prior")
            if not np.array_equal(self.velocity_mps, prior.mean(axis=0)):
                raise ValueError("point-only representative velocity must be prior mean")
            object.__setattr__(self, "velocity_prior_mps", prior)
        elif self.velocity_prior_mps is not None:
            raise ValueError("only point-only origins accept a prior")

    def sample_velocities(self, count: int, rng: np.random.Generator):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("positive integer particle count required")
        if self.velocity_prior_mps is not None:
            return self.velocity_prior_mps[rng.integers(len(self.velocity_prior_mps), size=count)].copy()
        return np.broadcast_to(self.velocity_mps, (count, 2)).copy()


def known_velocity(position_m, epoch_seconds, velocity_mps, *, source,
                   observed_at_seconds=None, error_mps=None):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("known velocity needs explicit observation provenance")
    return Origin("known_velocity", epoch_seconds, position_m, velocity_mps,
                  [position_m], [epoch_seconds], source,
                  velocity_observed_at_seconds=epoch_seconds if observed_at_seconds is None else observed_at_seconds,
                  velocity_error_mps=error_mps)


def causal_prefix(positions_m, times_seconds, *, max_gap_seconds=60.0):
    positions = frozen_array(positions_m)
    times = frozen_array(times_seconds)
    if times.ndim != 1 or len(times) < 2 or positions.shape != (len(times), 2):
        raise ValueError("at least two matched prefix observations required")
    gaps = np.diff(times)
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("positive finite maximum gap required")
    if np.any(gaps <= 0) or np.any(gaps > max_gap_seconds):
        raise ValueError("prefix contains a repeated, backward or excessive time gap")
    # Secant across the last three visible observations; no label-derived slope.
    start = max(0, len(times) - HISTORY_POINTS)
    velocity = (positions[-1] - positions[start]) / (times[-1] - times[start])
    return Origin("causal_prefix", float(times[-1]), positions[-1], velocity,
                  positions[start:], times[start:], "visible-last-three-secant-v1",
                  velocity_observed_at_seconds=float(times[-1]))


@dataclass(frozen=True)
class VelocityPrior:
    velocities_mps: np.ndarray
    training_identity: str

    def __post_init__(self):
        values = frozen_array(self.velocities_mps)
        if values.ndim != 2 or values.shape[1] != 2 or not len(values):
            raise ValueError("nonempty 2D velocity population required")
        if not isinstance(self.training_identity, str) or not self.training_identity.strip():
            raise ValueError("training identity required")
        object.__setattr__(self, "velocities_mps", values)

    @classmethod
    def fit(cls, origins, *, split, training_identity):
        if split != "train":
            raise ValueError("velocity prior can only be fitted on train")
        origins = tuple(origins)
        if not origins or any(o.mode != "causal_prefix" for o in origins):
            raise ValueError("fit requires explicit training prefix origins")
        return cls(np.stack([o.velocity_mps for o in origins]), training_identity)

    @property
    def identity(self):
        digest = hashlib.sha256(self.training_identity.encode("utf-8"))
        digest.update(np.asarray(self.velocities_mps, dtype="<f8").tobytes())
        return digest.hexdigest()

    def at(self, position_m, epoch_seconds):
        return Origin("point_only", epoch_seconds, position_m, self.velocities_mps.mean(axis=0),
                      [position_m], [epoch_seconds], f"train-prior:{self.identity}", self.velocities_mps)
