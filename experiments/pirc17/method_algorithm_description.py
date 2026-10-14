"""Describe pinned saved method fits and frozen algorithms; never train or predict."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from .protocol_core import canonical, file_hash, read_json, unpack


INVENTORY_SHA = "675b755e7b234698cc2954e5d743b36d41e37284c4abff200ac888da212fe250"
CONSTANTS = ("RIDGE", "COVARIANCE_JITTER", "MIN_META_TASK_SEGMENTS",
             "REPTILE_META_EPOCHS", "REPTILE_INITIAL_STEP", "REPTILE_INNER_STEPS",
             "REPTILE_INNER_RATE")
MODEL_FIELDS = ("meta_algorithm", "meta_task_count", "meta_outer_epochs", "meta_inner_steps",
                "meta_inner_rate", "meta_inner_objective_before", "meta_inner_objective_after",
                "adaptation_parameter_delta", "transfer_method", "finetune_method",
                "covariance_scale", "estimator_method", "estimator_drift_fraction")


def source_constants(source):
    """Parse literal defaults, without importing/calling the fitting module."""
    tree = ast.parse(source)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in CONSTANTS:
                    values[target.id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "_blend_models":
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if arg.arg == "weight":
                    values["target_adaptation_weight"] = ast.literal_eval(default)
    if set(values) != set(CONSTANTS) | {"target_adaptation_weight"}:
        raise ValueError("complete literal algorithm constants required")
    return values


def describe_model(model):
    result = {name: model[name] for name in MODEL_FIELDS}
    for name in MODEL_FIELDS:
        if isinstance(result[name], (int, float)) and not np.isfinite(result[name]):
            raise ValueError("finite saved algorithm diagnostic required")
    canonical(result)
    return result


def project(inventory_path, bundle_path):
    inventory = read_json(inventory_path, expected_file_sha256=INVENTORY_SHA)
    bundle = unpack(read_json(bundle_path))
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    if bundle["protocol"]["sha256"] != execution["protocol_sha256"]:
        raise ValueError("execution/protocol algorithm binding differs")
    source_root = Path(__file__).resolve().parents[2]
    required = ("experiments/nex326/model.py", "experiments/pirc17/method_rollout.py",
                "experiments/pirc17/method_mechanisms.py", "experiments/pirc17/formal_forecasts.py")
    sources = {}
    for relative in required:
        expected = execution["source_sha256"]["PSDE-SDE/" + relative]
        if file_hash(source_root / relative) != expected:
            raise ValueError("frozen algorithm source differs")
        if relative != "experiments/pirc17/formal_forecasts.py":
            if protocol["source_sha256"]["PSDE-SDE/" + relative] != expected:
                raise ValueError("protocol algorithm source differs")
        sources[relative] = expected
    constants = source_constants((source_root / required[0]).read_text(encoding="utf-8"))
    descriptions, models, fit_ids, prediction_slots = [], {}, set(), set()
    for binding in inventory["models"]:
        if binding["matrix"] != "NEX326-methods":
            continue
        record = unpack(read_json(binding["original_model_record_path"],
                                  expected_file_sha256=binding["original_model_record_file_sha256"]),
                        expected_sha256=binding["original_model_record_content_sha256"])
        if (record["fit_identity"] != binding["fit_identity"]
                or record["parameter_identity"] != binding["parameter_identity"]
                or record["matrix_sha256"] != inventory["matrix_sha256"]
                or record["protocol_sha256"] != bundle["protocol"]["sha256"]
                or record["fit_identity"] in fit_ids):
            raise ValueError("original saved method identity differs or repeats")
        fit_ids.add(record["fit_identity"])
        artifact = record["artifact"]
        for relative, expected in artifact["training"]["source_sha256"].items():
            if file_hash(source_root / relative) != expected:
                raise ValueError("original fit source differs")
        model = artifact["model"]
        slots = sorted(binding["prediction_configs"])
        if prediction_slots.intersection(slots):
            raise ValueError("duplicate saved prediction slot")
        prediction_slots.update(slots)
        for slot in slots:
            models[slot] = model
        descriptions.append({"representative_slot": binding["representative_slot"],
                             "prediction_slots": slots, "original_record_sha256": binding["original_model_record_file_sha256"],
                             **describe_model(model)})
    if len(descriptions) != 16 or len(prediction_slots) != 28:
        raise ValueError("complete original 16-fit/28-slot method population required")
    meta, full = models["arm-14/reptile"], models["arm-01/full"]
    if (meta["meta_outer_epochs"] != constants["REPTILE_META_EPOCHS"]
            or meta["meta_inner_steps"] != constants["REPTILE_INNER_STEPS"]
            or meta["meta_inner_rate"] != constants["REPTILE_INNER_RATE"]
            or meta["meta_task_count"] < 2):
        raise ValueError("saved Reptile settings differ from bound source")
    equality = {name: bool(np.array_equal(np.asarray(meta[name]), np.asarray(full[name])))
                for name in ("weights", "covariances", "mode_probabilities")}
    result = {"schema_version": "pirc17-saved-method-algorithm-description-v1",
              "inventory_sha256": INVENTORY_SHA, "protocol_sha256": bundle["protocol"]["sha256"],
              "continuation_execution_sha256": bundle["execution"]["sha256"],
              "source_sha256": sources, "constants": constants,
              "methods": sorted(descriptions, key=lambda row: row["representative_slot"]),
              "saved_reptile_and_full_parameter_arrays_exactly_equal": equality,
              "scope": {"new_fits": 0, "new_forecasts": 0, "new_particle_scores": 0,
                        "coefficient_matrices_private_paths_or_segment_ids_exported": False,
                        "pre_meta_and_pre_target_parameters_saved": False,
                        "retained_initialization_flow": "Final target blending retains weight 0.35 on the preceding model; not a full re-solve from that initialization. Saved final model differences do not establish meta-learning performance gains.",
                        "independent_saved_forecast_audit": False,
                        "numerical_or_meta_learning_qualification_added": False}}
    canonical(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = project(args.inventory, args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print("Described 16 saved method fits and pinned algorithms; no fits or forecasts.")


if __name__ == "__main__":
    main()
