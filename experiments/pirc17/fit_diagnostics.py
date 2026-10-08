"""Project aggregate diagnostics from pinned saved fits; never fit or forecast.

The whitelist deliberately excludes parameters, source paths and segment IDs.
Augmented ridge ranks are not ranks of the unregularized feature matrix.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .protocol_core import canonical, read_json, unpack


ROLES = ("train", "adapt", "validation")


def transition_counts(training):
    counts = Counter()
    transitions = Counter()
    tails = Counter()
    for row in training["per_segment"]:
        role = row["role"]
        if role not in ROLES:
            raise ValueError("unknown development role")
        number = row["transition_count"]
        if type(number) is not int or number < 0:
            raise ValueError("nonnegative integer transition count required")
        if row["reference_interval_seconds"] != training["reference_interval_seconds"]:
            raise ValueError("transition interval differs from saved fit")
        if row["scoring_observations_changed"] is not False:
            raise ValueError("saved fitting resampling altered scoring observations")
        if type(row["removed_partial_tail"]) is not bool:
            raise ValueError("Boolean partial-tail flag required")
        counts[role] += 1
        transitions[role] += number
        tails[role] += row["removed_partial_tail"]
    result = {
        "windows": {role: counts[role] for role in ROLES},
        "transitions": {role: transitions[role] for role in ROLES},
        "windows_with_removed_partial_tail": {role: tails[role] for role in ROLES},
    }
    for actual, saved in (("windows", "sample_counts"),
                          ("transitions", "transitions_by_role"),
                          ("windows_with_removed_partial_tail", "removed_tails_by_role")):
        if result[actual] != training[saved]:
            raise ValueError("saved aggregate differs from per-window counts")
    return result


def terrain_diagnostics(model):
    checkpoint = model["conditioner_checkpoint"]
    solver = checkpoint["solver_report"]
    n = model["train_transition_count"]
    width = checkpoint["input_dim"]
    if (solver["train_rows"] != n or solver["input_dim"] != width
            or solver["output_dim"] != 2):
        raise ValueError("terrain solver dimensions disagree")
    penalty = n * solver["output_dim"] * solver["weight_decay"] / 2
    if not np.isclose(solver["normal_equation_penalty"], penalty, rtol=1e-12, atol=0):
        raise ValueError("ridge normal-equation normalization disagrees")
    q = np.asarray(model["diffusion_covariance_m2_per_s"], dtype=float)
    if q.shape != (2, 2) or not np.isfinite(q).all() or not np.allclose(q, q.T):
        raise ValueError("finite symmetric two-dimensional covariance required")
    eigenvalues = np.linalg.eigvalsh(q)
    if eigenvalues[0] < -1e-12:
        raise ValueError("saved covariance is not positive semidefinite")
    diagnostics = model["diagnostics"]
    return {
        "configuration": model["configuration"],
        "train_transitions": n,
        "validation_transitions": model["validation_transition_count"],
        "conditioner_input_columns": width,
        "augmented_ridge_rank": solver["augmented_rank"],
        "augmented_ridge_condition": solver["augmented_condition"],
        "relative_gradient_norm": solver["relative_gradient_norm"],
        "base_ridge_unnormalized": model["training_policy"]["base_ridge"],
        "conditioner_weight_decay": solver["weight_decay"],
        "conditioner_normal_equation_penalty": solver["normal_equation_penalty"],
        "train_base_mse_m2_per_s2": diagnostics["train"]["base_only_mse_m2_per_s2"],
        "train_fitted_mse_m2_per_s2": diagnostics["train"]["fitted_mse_m2_per_s2"],
        "validation_base_mse_m2_per_s2": diagnostics["validation"]["base_only_mse_m2_per_s2"],
        "validation_fitted_mse_m2_per_s2": diagnostics["validation"]["fitted_mse_m2_per_s2"],
        "diffusion_eigenvalues_m2_per_s": eigenvalues.tolist(),
    }


def project(inventory_path, inventory_sha256):
    inventory = read_json(inventory_path, expected_file_sha256=inventory_sha256)
    if (inventory["model_count"], inventory["method_fit_count"],
            inventory["terrain_fit_count"]) != (26, 16, 10):
        raise ValueError("expected pinned 26-fit inventory")
    methods, terrain, base_weights = [], [], []
    seen = set()
    protocols, original_executions = set(), set()
    for binding in inventory["models"]:
        identity = binding["fit_identity"]
        if identity in seen:
            raise ValueError("duplicate fit identity")
        seen.add(identity)
        payload = unpack(read_json(
            binding["original_model_record_path"],
            expected_file_sha256=binding["original_model_record_file_sha256"]),
            expected_sha256=binding["original_model_record_content_sha256"])
        if (payload["fit_identity"] != identity
                or payload["parameter_identity"] != binding["parameter_identity"]
                or payload["matrix_sha256"] != inventory["matrix_sha256"]):
            raise ValueError("saved fit identity binding mismatch")
        protocols.add(payload["protocol_sha256"])
        original_executions.add(payload["execution_sha256"])
        artifact = payload["artifact"]
        if binding["matrix"] == "NEX326-methods":
            model = artifact["model"]
            counts = transition_counts(artifact["training"])
            methods.append({
                "representative_slot": binding["representative_slot"],
                "prediction_slots": sorted(binding["prediction_configs"]),
                "reference_interval_seconds": artifact["training"]["reference_interval_seconds"],
                **counts,
                "model_kind": model["model_kind"],
                "mode_count": binding["mode_count"],
                "algorithm_training_sample_label_not_unique_rows": model["training_sample_count"],
                "selected_covariance_scale": model["covariance_scale"],
                "selected_estimator_drift_fraction": model["estimator_drift_fraction"],
                "original_record_sha256": binding["original_model_record_file_sha256"],
            })
        elif binding["matrix"] == "terrain":
            model = artifact["model"]
            if model["configuration"] != binding["configuration"]:
                raise ValueError("terrain configuration binding mismatch")
            terrain.append({**terrain_diagnostics(model),
                            "original_record_sha256": binding["original_model_record_file_sha256"]})
            base_weights.append(np.asarray(model["base_weights"], dtype=float))
        else:
            raise ValueError("unexpected fitting family")
    if len(methods) != 16 or len(terrain) != 10 or len(protocols) != 1:
        raise ValueError("saved fit population/protocol mismatch")
    if any(w.shape != (7, 2) or not np.isfinite(w).all() for w in base_weights):
        raise ValueError("finite baseline coefficient matrices required")
    if any(not np.array_equal(base_weights[0], w) for w in base_weights[1:]):
        raise ValueError("terrain configurations do not share identical baseline coefficients")
    if sum(len(m["prediction_slots"]) for m in methods) != 28:
        raise ValueError("expected 28 method prediction configurations")
    result = {
        "schema_version": "pirc17-saved-fit-diagnostics-v1",
        "inventory_sha256": inventory_sha256,
        "matrix_sha256": inventory["matrix_sha256"],
        "protocol_sha256": next(iter(protocols)),
        "original_fit_execution_sha256": sorted(original_executions),
        "continuation_execution_sha256": inventory["execution_sha256"],
        "method_fit_count": len(methods), "terrain_fit_count": len(terrain),
        "terrain_baseline_identical_across_all_fits": True,
        "methods": sorted(methods, key=lambda m: m["representative_slot"]),
        "terrain": sorted(terrain, key=lambda m: m["configuration"]),
        "scope": {
            "new_fits": 0, "new_predictions": 0, "new_particle_scores": 0,
            "independent_forecast_audit_performed": False,
            "raw_feature_rank_available": False,
            "feature_mask_validity_rates_available": False,
            "coefficient_matrices_exported": False,
            "private_source_paths_or_segment_ids_exported": False,
            "qualification": "Saved development-fit diagnostics only; not final forecast inference.",
        },
    }
    canonical(result)  # reject nonfinite diagnostics before publication
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = project(args.inventory, args.inventory_sha256)
    # Exclusive publication; do not overwrite an earlier evidence snapshot.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(f"Projected {result['method_fit_count']} method and {result['terrain_fit_count']} terrain fits; no fitting or forecasts.")


if __name__ == "__main__":
    main()
