"""Distinct train-only linear optimizer candidate, not the PIRC-22 Adam fit.

The causal drift, selected feature capacity and residual diffusion are retained.
Only the conditioner optimizer changes. No epochs or validation-based checkpoint
selection are invented for this deterministic solve; old checkpoints stay old.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

import numpy as np
import torch

from experiments.pirc22.conditioners import build_conditioner
from experiments.pirc22.consumer import load_benchmark_selection_binding
from experiments.pirc22.representations import _COMPOSITIONS, _VARIANT_DIMENSIONS
from .configurations import terrain_configurations
from .dynamics import BASE_COLUMNS, LearnedDynamics, TransitionRows, diffusion_covariance
from .inference import SEEDS
from .origins import frozen_array

POLICY_VERSION = "pirc17-direct-linear-training-candidate-v1"
MODEL_VERSION = "pirc17-direct-linear-causal-dynamics-v1"
CONDITIONER_VERSION = "pirc17-direct-linear-conditioner-v1"
MAX_AUGMENTED_CONDITION = 1e8
STATIONARITY_RTOL = 1e-10
OBJECTIVE_RTOL = 1e-12


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def training_policy():
    binding = load_benchmark_selection_binding()
    selected = binding["selected_configuration"]
    if (selected["conditioner_id"] != "linear" or selected["hidden_widths"] != []
            or selected["training_config"]["weight_decay"] != 1e-4):
        raise ValueError("direct candidate requires the frozen linear capacity and L2 strength")
    policy = {
        "version": POLICY_VERSION, "status": "candidate-unsealed",
        "pirc22_consumer_identity": binding["consumer_identity_sha256"],
        "optimizer": "numpy_lstsq_augmented_ridge_float64",
        "objective": "mean((X_aug@theta-Y)^2)+weight_decay/2*sum(theta^2)",
        "weight_decay": 1e-4, "regularize_bias": True, "base_ridge": 1e-6,
        "normal_equation_penalty": "n_train*n_outputs*weight_decay/2",
        "seed_use": "identity_only_deterministic_fit; forecast_streams_remain_separate",
        "data_roles": {"solve": "train", "diagnostic": "validation", "selection": "none"},
        "population": "one first-future transition per qualified origin; candidate only",
        "max_augmented_condition": MAX_AUGMENTED_CONDITION,
        "stationarity_rtol": STATIONARITY_RTOL, "objective_rtol": OBJECTIVE_RTOL,
        "superseded_optimizer_settings": dict(selected["training_config"]),
        "final_eval_authorized": False,
    }
    return {**policy, "sha256": digest(policy)}


def configuration_width(name):
    config = terrain_configurations().get(name)
    if config is None:
        raise ValueError("unknown terrain configuration")
    return 2 * (sum(_VARIANT_DIMENSIONS[v] for v in config["variant_ids"])
                + sum(_COMPOSITIONS[c][0] for c in config["composition_ids"]))


def _matrix(values, name):
    result = frozen_array(values)
    if result.ndim != 2 or len(result) < 2:
        raise ValueError(f"{name} needs a finite matrix with at least two rows")
    return result


def solve_linear(features, targets, *, weight_decay=1e-4):
    """Solve MSE + lambda/2 L2 using augmented least squares, including bias."""
    x, y = _matrix(features, "features"), _matrix(targets, "targets")
    if len(x) != len(y) or y.shape[1] != 2:
        raise ValueError("aligned features and two output columns required")
    if isinstance(weight_decay, bool) or not np.isfinite(weight_decay) or weight_decay <= 0:
        raise ValueError("positive finite weight decay required")
    design = np.column_stack((x, np.ones(len(x))))
    penalty = len(x) * y.shape[1] * weight_decay / 2
    augmented = np.vstack((design, np.sqrt(penalty) * np.eye(design.shape[1])))
    response = np.vstack((y, np.zeros((design.shape[1], y.shape[1]))))
    theta, _, rank, singular = np.linalg.lstsq(augmented, response, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if rank != design.shape[1] or not np.isfinite(condition) or condition > MAX_AUGMENTED_CONDITION:
        raise ValueError("augmented linear system is rank deficient or ill-conditioned")
    residual = design @ theta - y
    gradient_data = 2 / y.size * (design.T @ residual)
    gradient_penalty = weight_decay * theta
    relative_gradient = float(np.linalg.norm(gradient_data + gradient_penalty) /
        max(1., np.linalg.norm(gradient_data), np.linalg.norm(gradient_penalty),
            2 / y.size * np.linalg.norm(design.T @ y)))
    mse = float(np.mean(residual**2))
    regularization = float(weight_decay / 2 * np.sum(theta**2))
    objective, zero_objective = mse + regularization, float(np.mean(y**2))
    numbers = [relative_gradient, mse, regularization, objective, zero_objective]
    if not np.isfinite(theta).all() or not np.isfinite(numbers).all():
        raise ValueError("nonfinite linear solution or diagnostics")
    if relative_gradient > STATIONARITY_RTOL:
        raise ValueError("linear objective stationarity check failed")
    if objective > zero_objective + OBJECTIVE_RTOL * max(1., zero_objective):
        raise ValueError("linear objective is worse than the feasible zero correction")
    return frozen_array(theta), {
        "train_rows": len(x), "input_dim": x.shape[1], "output_dim": 2,
        "weight_decay": float(weight_decay), "normal_equation_penalty": float(penalty),
        "augmented_rank": int(rank), "augmented_condition": condition,
        "relative_gradient_norm": relative_gradient, "train_mse_m2_per_s2": mse,
        "regularization": regularization, "objective": objective,
        "zero_correction_objective": zero_objective,
    }


def _validate_key(training_identity, configuration, seed):
    if not isinstance(training_identity, str) or not re.fullmatch(r"[0-9a-f]{64}", training_identity):
        raise ValueError("training identity must be a SHA256")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("one of the five registered seeds required")
    configs = terrain_configurations()
    if configuration not in configs:
        raise ValueError("unknown terrain configuration")
    return configs[configuration]


def _conditioner(theta, seed):
    model = build_conditioner("linear", len(theta) - 1, 2, seed=seed)
    # Initialization never enters the solve; all parameters are replaced.
    with torch.no_grad():
        model.network[0].weight.copy_(torch.tensor(theta[:-1].T.copy(), dtype=torch.float64))
        model.network[0].bias.copy_(torch.tensor(theta[-1].copy(), dtype=torch.float64))
    model.eval()
    return model


def fit_direct_dynamics(train: TransitionRows, validation: TransitionRows, *, seed,
                        training_identity, configuration):
    if train.role != "train" or validation.role != "validation":
        raise ValueError("train/validation roles reversed")
    if set(train.independent_blocks) & set(validation.independent_blocks):
        raise ValueError("train and validation independent blocks overlap")
    config = _validate_key(training_identity, configuration, seed)
    width = configuration_width(configuration)
    if train.feature_matrix.shape[1] != width or validation.feature_matrix.shape[1] != width:
        raise ValueError("feature dimensions differ from the frozen configuration")
    policy = training_policy()
    design = train.design()
    target = train.displacement_m / train.elapsed_seconds[:, None]
    base = np.linalg.solve(design.T @ design + policy["base_ridge"] * np.eye(design.shape[1]),
                           design.T @ target)
    residual = target - design @ base
    theta, report = solve_linear(train.feature_matrix, residual, weight_decay=policy["weight_decay"])
    diagnostics = {}
    for role, batch in (("train", train), ("validation", validation)):
        y = batch.displacement_m / batch.elapsed_seconds[:, None] - batch.design() @ base
        correction = np.column_stack((batch.feature_matrix, np.ones(len(y)))) @ theta
        diagnostics[role] = {"base_only_mse_m2_per_s2": float(np.mean(y**2)),
                             "fitted_mse_m2_per_s2": float(np.mean((correction-y)**2))}
    correction = np.column_stack((train.feature_matrix, np.ones(len(train.elapsed_seconds)))) @ theta
    covariance = diffusion_covariance(train.displacement_m, design @ base + correction,
                                      train.elapsed_seconds)
    conditioner = {"version": CONDITIONER_VERSION, "input_dim": width, "output_dim": 2,
                   "theta_feature_rows_then_bias": theta.tolist(), "solver_report": report}
    conditioner["sha256"] = digest(conditioner)
    identity = {"version": MODEL_VERSION, "training_policy": policy,
        "training_identity": training_identity, "configuration": configuration,
        "configuration_identity": config["sha256"], "seed": seed,
        "base_columns": list(BASE_COLUMNS), "ridge": policy["base_ridge"],
        "base_weights": base.tolist(), "diffusion_covariance_m2_per_s": covariance.tolist(),
        "diffusion_estimator": "train_brownian_residual_mle_v1",
        "conditioner_checkpoint": conditioner, "diagnostics": diagnostics,
        "train_transition_count": len(train.elapsed_seconds),
        "validation_transition_count": len(validation.elapsed_seconds)}
    identity["sha256"] = digest(identity)
    return restore_direct_dynamics(identity)


def restore_direct_dynamics(identity):
    """Reject old Adam checkpoints and altered candidate semantics, even rehashed."""
    payload = deepcopy(identity)
    actual_hash = payload.pop("sha256", None)
    if actual_hash != digest(payload) or payload.get("version") != MODEL_VERSION:
        raise ValueError("direct dynamics version/hash mismatch")
    if set(payload) != {"version", "training_policy", "training_identity", "configuration",
        "configuration_identity", "seed", "base_columns", "ridge", "base_weights",
        "diffusion_covariance_m2_per_s", "diffusion_estimator", "conditioner_checkpoint",
        "diagnostics", "train_transition_count", "validation_transition_count"}:
        raise ValueError("direct dynamics fields differ from the candidate schema")
    policy = training_policy()
    config = _validate_key(payload.get("training_identity"), payload.get("configuration"), payload.get("seed"))
    if (payload.get("training_policy") != policy or payload.get("configuration_identity") != config["sha256"]
            or payload.get("base_columns") != list(BASE_COLUMNS) or payload.get("ridge") != policy["base_ridge"]
            or payload.get("diffusion_estimator") != "train_brownian_residual_mle_v1"):
        raise ValueError("direct dynamics policy or input semantics changed")
    for key in ("train_transition_count", "validation_transition_count"):
        if type(payload.get(key)) is not int or payload[key] < 2:
            raise ValueError("invalid transition count")
    cp = deepcopy(payload["conditioner_checkpoint"])
    cp_hash = cp.pop("sha256", None)
    width = configuration_width(payload["configuration"])
    if (cp_hash != digest(cp) or cp.get("version") != CONDITIONER_VERSION
            or set(cp) != {"version", "input_dim", "output_dim", "theta_feature_rows_then_bias", "solver_report"}
            or cp.get("input_dim") != width or cp.get("output_dim") != 2):
        raise ValueError("direct conditioner version/hash/dimensions differ")
    theta = frozen_array(cp["theta_feature_rows_then_bias"], (width + 1, 2))
    report = cp["solver_report"]
    rows = payload["train_transition_count"]
    expected = {"train_rows": rows, "input_dim": width, "output_dim": 2,
                "weight_decay": policy["weight_decay"],
                "normal_equation_penalty": rows * policy["weight_decay"], "augmented_rank": width+1}
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("direct solver normalization or dimensions changed")
    diagnostic_names = ("augmented_condition", "relative_gradient_norm", "train_mse_m2_per_s2",
                        "regularization", "objective", "zero_correction_objective")
    numbers = np.array([report[k] for k in diagnostic_names], dtype=float)
    if not np.isfinite(numbers).all() or np.any(numbers < 0):
        raise ValueError("nonfinite or negative direct solver diagnostics")
    if (not 1 <= report["augmented_condition"] <= MAX_AUGMENTED_CONDITION
            or report["relative_gradient_norm"] > STATIONARITY_RTOL
            or report["objective"] > report["zero_correction_objective"]
               + OBJECTIVE_RTOL * max(1., report["zero_correction_objective"])):
        raise ValueError("direct solver qualification failed")
    if (not np.isclose(report["regularization"], policy["weight_decay"]/2*np.sum(theta**2), rtol=1e-12, atol=1e-12)
            or not np.isclose(report["objective"], report["train_mse_m2_per_s2"]+report["regularization"], rtol=1e-12, atol=1e-12)):
        raise ValueError("direct solver objective does not match coefficients")
    diagnostics = payload["diagnostics"]
    if set(diagnostics) != {"train", "validation"} or any(
        set(row) != {"base_only_mse_m2_per_s2", "fitted_mse_m2_per_s2"}
        or not all(np.isfinite(v) and v >= 0 for v in row.values()) for row in diagnostics.values()
    ):
        raise ValueError("invalid role-specific loss diagnostics")
    if (diagnostics["train"]["fitted_mse_m2_per_s2"] != report["train_mse_m2_per_s2"]
            or diagnostics["train"]["base_only_mse_m2_per_s2"] != report["zero_correction_objective"]):
        raise ValueError("training diagnostics do not match solver report")
    covariance = frozen_array(payload["diffusion_covariance_m2_per_s"], (2, 2))
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0) or np.min(eigenvalues) < -1e-10:
        raise ValueError("diffusion covariance is not positive semidefinite")
    return LearnedDynamics(frozen_array(payload["base_weights"], (len(BASE_COLUMNS), 2)),
        _conditioner(theta, payload["seed"]), deepcopy(identity["conditioner_checkpoint"]),
        frozen_array(vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.)))), deepcopy(identity))
