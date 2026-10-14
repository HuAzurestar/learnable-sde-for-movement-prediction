"""Train-only causal drift + frozen-capacity conditioner + physical diffusion.

This is a versioned PIRC-17 base model, not the positional PIRC-22 benchmark
base and not a substitute for the separately registered NEX326 method arms.
The frozen selection controls representations/capacity, not fitted weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from experiments.pirc22.conditioners import TrainingConfig, fit_conditioner, predict_conditioner
from experiments.pirc22.consumer import load_benchmark_selection_binding
from .origins import frozen_array, motion

DYNAMICS_VERSION = "pirc17-causal-kinematic-drift-v1"
BASE_COLUMNS = ("intercept", "velocity_east_mps", "velocity_north_mps", "speed_mps",
                "history_direction_east", "history_direction_north", "direction_valid")


def baseline_design(velocities_mps, history_direction=None):
    velocity = np.asarray(velocities_mps, dtype=float)
    if velocity.ndim != 2 or velocity.shape[1] != 2 or not np.isfinite(velocity).all():
        raise ValueError("finite velocity matrix required")
    speed, direction, valid = motion(velocity)
    if history_direction is not None:
        direction, valid = history_direction
        direction = np.asarray(direction, dtype=float)
        valid = np.asarray(valid)
        if direction.shape != velocity.shape or valid.shape != speed.shape or valid.dtype != np.bool_:
            raise ValueError("invalid history direction shape or mask")
        if not np.isfinite(direction).all() or not np.allclose(np.linalg.norm(direction[valid], axis=1), 1):
            raise ValueError("valid directions must be finite unit vectors")
        direction = np.where(valid[:, None], direction, 0.)
    return np.column_stack((np.ones(len(velocity)), velocity, speed, direction, valid.astype(float)))


@dataclass(frozen=True)
class TransitionRows:
    role: str
    independent_blocks: tuple[str, ...]
    velocities_mps: np.ndarray
    history_direction: np.ndarray
    direction_valid: np.ndarray
    displacement_m: np.ndarray
    elapsed_seconds: np.ndarray
    feature_matrix: np.ndarray

    def __post_init__(self):
        if self.role not in {"train", "validation"}:
            raise ValueError("fitting rows must be train or validation")
        dt = frozen_array(self.elapsed_seconds)
        if dt.ndim != 1 or len(dt) < 2 or np.any(dt <= 0):
            raise ValueError("at least two positive-time transitions required")
        n = len(dt)
        if len(self.independent_blocks) != n or any(not b for b in self.independent_blocks):
            raise ValueError("each transition needs its independent block identity")
        for name in ("velocities_mps", "history_direction", "displacement_m"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), (n, 2)))
        valid = np.array(self.direction_valid, copy=True)
        if valid.dtype != np.bool_ or valid.shape != (n,):
            raise ValueError("boolean direction validity per transition required")
        valid.setflags(write=False)
        features = frozen_array(self.feature_matrix)
        if features.ndim != 2 or len(features) != n:
            raise ValueError("feature matrix row count mismatch")
        object.__setattr__(self, "elapsed_seconds", dt)
        object.__setattr__(self, "direction_valid", valid)
        object.__setattr__(self, "feature_matrix", features)
        self.design()

    def design(self):
        return baseline_design(self.velocities_mps, (self.history_direction, self.direction_valid))


def diffusion_covariance(displacement_m, drift_mps, elapsed_seconds):
    """Brownian MLE: mean[(dx - b dt)(dx - b dt)^T / dt], units m²/s."""
    dx = np.asarray(displacement_m, dtype=float)
    drift = np.asarray(drift_mps, dtype=float)
    dt = np.asarray(elapsed_seconds, dtype=float)
    if dx.ndim != 2 or dx.shape[1] != 2 or drift.shape != dx.shape or dt.shape != (len(dx),):
        raise ValueError("diffusion input shapes differ")
    if len(dx) < 2 or not all(np.isfinite(a).all() for a in (dx, drift, dt)) or np.any(dt <= 0):
        raise ValueError("finite positive-time residuals required")
    residuals = (dx - drift * dt[:, None]) / np.sqrt(dt[:, None])
    covariance = residuals.T @ residuals / len(residuals)
    return frozen_array((covariance + covariance.T) / 2)


@dataclass(frozen=True)
class LearnedDynamics:
    base_weights: np.ndarray
    conditioner_model: object
    conditioner_checkpoint: dict
    diffusion_root: np.ndarray
    identity: dict

    def base_drift(self, state):
        return baseline_design(state.velocities_mps, state.history_direction) @ self.base_weights

    def correction(self, state, features):
        # PIRC-22's Torch helper wraps arrays without copying; supply writable input.
        return predict_conditioner(self.conditioner_model, np.array(features, copy=True))

    def diffusion(self, state):
        return np.broadcast_to(self.diffusion_root, (len(state.positions_m), 2, 2))


def fit_dynamics(train: TransitionRows, validation: TransitionRows, *, seed: int,
                 training_identity: str, configuration_identity: str):
    if train.role != "train" or validation.role != "validation":
        raise ValueError("train/validation roles reversed")
    if set(train.independent_blocks) & set(validation.independent_blocks):
        raise ValueError("train and validation independent blocks overlap")
    if not training_identity or not configuration_identity:
        raise ValueError("training and feature-configuration identities required")
    if train.feature_matrix.shape[1] != validation.feature_matrix.shape[1]:
        raise ValueError("train/validation feature dimensions differ")
    binding = load_benchmark_selection_binding()
    frozen = binding["selected_configuration"]
    train_x, validation_x = train.design(), validation.design()
    train_y = train.displacement_m / train.elapsed_seconds[:, None]
    validation_y = validation.displacement_m / validation.elapsed_seconds[:, None]
    # Same base context in every factor configuration, fitted on train only.
    weights = np.linalg.solve(train_x.T @ train_x + 1e-6*np.eye(train_x.shape[1]), train_x.T @ train_y)
    fit = fit_conditioner(frozen["conditioner_id"], np.array(train.feature_matrix),
        train_y-train_x@weights, np.array(validation.feature_matrix), validation_y-validation_x@weights,
        seed=seed, config=TrainingConfig(**frozen["training_config"]))
    fitted_drift = train_x@weights + predict_conditioner(fit.model, np.array(train.feature_matrix))
    covariance = diffusion_covariance(train.displacement_m, fitted_drift, train.elapsed_seconds)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    root = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.)))
    identity = {"version":DYNAMICS_VERSION,"base_columns":list(BASE_COLUMNS),"ridge":1e-6,
        "training_identity":training_identity,"configuration_identity":configuration_identity,
        "pirc22_consumer_identity":binding["consumer_identity_sha256"],"seed":seed,
        "base_weights":weights.tolist(),"diffusion_covariance_m2_per_s":covariance.tolist(),
        "diffusion_estimator":"train_brownian_residual_mle_v1","conditioner_checkpoint":dict(fit.checkpoint),
        "train_transition_count":len(train.elapsed_seconds),"validation_transition_count":len(validation.elapsed_seconds)}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",",":"), allow_nan=False).encode()).hexdigest()
    return LearnedDynamics(frozen_array(weights), fit.model, dict(fit.checkpoint), frozen_array(root), identity)
