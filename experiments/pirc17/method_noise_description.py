"""Describe probabilities and noise in original saved fits, without refitting.

Final blended mode probabilities are parameters, not per-role sample counts.
The stored rate covariance already includes selected scale squared; Q=tau*R
does not multiply it by that scale again. No forecast constructors are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .fit_diagnostics import project as project_base
from .protocol_core import canonical, digest, file_hash, read_json, unpack


SOURCE_HASHES = {
    "experiments/nex326/model.py": "8863e936419a55d95b4fe31ddb8790fcbb0a923b57cbc3f77cde3e095dea9aa2",
    "experiments/pirc17/method_training.py": "65c78934a74f2725bf7f792cd851178fcc539d67a912fc8d33253d45a1d15598",
    "experiments/pirc17/method_rollout.py": "076e3964d8f2f18ea45c93c964962be01a74eff62c380cc8719b5e3af4ac3560",
}
EMBEDDING = "Q_m2_per_s = R_rate_m2_per_s2 * reference_interval_seconds"
KINDS = {"seg_constant_mode", "pointwise_mixture", "single_gaussian",
         "explicit_decomp", "gmm_kernel"}


def covariance_description(value, tau):
    """Saved 2x2 arithmetic using the existing bind-time PSD tolerance."""
    if type(tau) not in (float, int) or not np.isfinite(tau) or tau <= 0:
        raise ValueError("positive finite reference interval required")
    r = np.asarray(value, dtype=float)
    if r.shape != (2, 2) or not np.isfinite(r).all():
        raise ValueError("finite 2x2 rate covariance required")
    tolerance = 128 * np.finfo(float).eps * max(1., float(np.linalg.norm(r, ord=2)))
    asymmetry = float(np.max(np.abs(r - r.T)))
    if asymmetry > tolerance:
        raise ValueError("symmetric rate covariance required")
    saved = np.linalg.eigvalsh(.5 * (r + r.T))
    if saved[0] < -tolerance:
        raise ValueError("materially indefinite saved rate covariance")
    bound = np.maximum(saved, 0.)
    return {
        "saved_rate_eigenvalues_m2_per_s2": saved.tolist(),
        "bound_rate_eigenvalues_m2_per_s2": bound.tolist(),
        "embedded_diffusion_eigenvalues_m2_per_s": (tau * bound).tolist(),
        "saved_rate_asymmetry_m2_per_s2": asymmetry,
        "rate_psd_roundoff_tolerance_m2_per_s2": tolerance,
        "bind_time_negative_roundoff_repair_required": bool(saved[0] < 0),
    }


def describe_method(artifact, binding):
    model, training, dynamics = (artifact[name] for name in ("model", "training", "dynamics"))
    tau = training["reference_interval_seconds"]
    if (type(tau) not in (float, int) or not np.isfinite(tau) or tau <= 0
            or binding["reference_interval_seconds"] != tau
            or dynamics["reference_interval_seconds"] != tau
            or dynamics["noise_embedding"] != EMBEDDING
            or dynamics["embedding_is_continuous_time_fit_qualification"] is not False
            or dynamics["fit_identity"] != binding["parameter_identity"]
            or dynamics["fit_identity"] != digest({
                "training_identity_sha256": training["training_identity_sha256"], "model": model})
            or dynamics["model_sha256"] != digest(model)):
        raise ValueError("saved dynamics/noise/model identity binding mismatch")
    kind = model["model_kind"]
    if kind not in KINDS:
        raise ValueError("unsupported model family")
    n = 1 if kind in {"single_gaussian", "explicit_decomp"} else 3
    if (type(binding["mode_count"]) is not int or binding["mode_count"] != n
            or len(model["weights"]) != n or len(model["covariances"]) != n):
        raise ValueError("saved mode count differs from registered family")
    columns = (6 if kind == "explicit_decomp" else 3) + len(model["condition_names"])
    for weights in model["weights"]:
        w = np.asarray(weights, dtype=float)
        if w.shape != (columns, 2) or not np.isfinite(w).all():
            raise ValueError("finite registered coefficient shape required")
    probabilities = np.asarray(model["mode_probabilities"], dtype=float)
    if (probabilities.shape != (n,) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("normalized nonnegative mode probabilities required")
    scale = model["covariance_scale"]
    if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite saved scale required")
    modes = [{"mode_index": i, "final_mode_probability": float(p),
              **covariance_description(r, tau)}
             for i, (p, r) in enumerate(zip(probabilities, model["covariances"]))]
    return {
        "representative_slot": binding["representative_slot"],
        "prediction_slots": sorted(binding["prediction_configs"]),
        "original_record_sha256": binding["original_model_record_file_sha256"],
        "model_sha256": dynamics["model_sha256"],
        "model_kind": kind, "reference_interval_seconds": tau,
        "mode_count": n, "selected_covariance_scale_already_in_saved_R": scale,
        "selected_estimator_drift_fraction": model["estimator_drift_fraction"],
        "modes": modes,
        "diffusion_eigenvalue_envelope_m2_per_s": [
            min(m["embedded_diffusion_eigenvalues_m2_per_s"][0] for m in modes),
            max(m["embedded_diffusion_eigenvalues_m2_per_s"][1] for m in modes)],
        "per_role_mode_counts_retained_in_saved_artifact": False,
        "final_probabilities_multiplied_by_exposure_label_to_invent_counts": False,
    }


def project(inventory_path, inventory_sha256, base_path, base_sha256):
    base = read_json(base_path, expected_file_sha256=base_sha256)
    if project_base(inventory_path, inventory_sha256) != base:
        raise ValueError("original aggregate fitting diagnostics changed")
    root = Path(__file__).resolve().parents[2]
    for name, expected in SOURCE_HASHES.items():
        if file_hash(root / name) != expected:
            raise ValueError("original fitting/noise source changed")
    inventory = read_json(inventory_path, expected_file_sha256=inventory_sha256)
    methods = []
    for binding in inventory["models"]:
        if binding["matrix"] != "NEX326-methods":
            continue
        payload = unpack(read_json(binding["original_model_record_path"],
                                 expected_file_sha256=binding["original_model_record_file_sha256"]),
                         expected_sha256=binding["original_model_record_content_sha256"])
        artifact = payload["artifact"]
        for name in ("experiments/nex326/model.py", "experiments/pirc17/method_training.py"):
            if artifact["training"]["source_sha256"][name] != SOURCE_HASHES[name]:
                raise ValueError("saved fit source provenance mismatch")
        if (payload["fit_identity"] != binding["fit_identity"]
                or payload["parameter_identity"] != binding["parameter_identity"]):
            raise ValueError("saved fit inventory binding mismatch")
        methods.append(describe_method(artifact, binding))
    methods.sort(key=lambda x: x["representative_slot"])
    if [m["representative_slot"] for m in methods] != [m["representative_slot"] for m in base["methods"]]:
        raise ValueError("saved method fit population mismatch")
    for m, old in zip(methods, base["methods"]):
        if any(m[new] != old[old_name] for new, old_name in (
                ("prediction_slots", "prediction_slots"), ("original_record_sha256", "original_record_sha256"),
                ("reference_interval_seconds", "reference_interval_seconds"),
                ("selected_covariance_scale_already_in_saved_R", "selected_covariance_scale"),
                ("selected_estimator_drift_fraction", "selected_estimator_drift_fraction"))):
            raise ValueError("saved method diagnostics disagree with original aggregate")
    result = {
        "schema_version": "pirc17-saved-method-noise-description-v1",
        "inventory_sha256": inventory_sha256, "base_projection_sha256": base_sha256,
        "matrix_sha256": base["matrix_sha256"], "protocol_sha256": base["protocol_sha256"],
        "continuation_execution_sha256": base["continuation_execution_sha256"],
        "source_sha256": SOURCE_HASHES, "method_fit_count": len(methods),
        "prediction_configuration_count": sum(len(m["prediction_slots"]) for m in methods),
        "mode_parameter_records": sum(m["mode_count"] for m in methods),
        "methods": methods,
        "scope": {
            "saved_covariances_already_include_selected_scale_squared": True,
            "scale_applied_again": False, "reference_interval_not_integration_step": True,
            "per_role_mode_counts_available": False, "mode_probabilities_are_fitted_parameters": True,
            "mixture_predictive_covariance_or_final_calibration_inferred_from_Q": False,
            "new_fits_forecasts_particle_scores_resampling_or_map_queries": 0,
            "coefficient_matrices_private_paths_or_trajectory_identifiers_exported": False,
            "independent_saved_output_audit_completed": False,
            "all_review_items_or_paper_complete": False,
        },
    }
    canonical(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--base-projection", type=Path, required=True)
    parser.add_argument("--base-projection-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = project(args.inventory, args.inventory_sha256, args.base_projection, args.base_projection_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(f"Described {result['method_fit_count']} original fits/{result['mode_parameter_records']} modes; no fitting or forecasts.")


if __name__ == "__main__":
    main()
