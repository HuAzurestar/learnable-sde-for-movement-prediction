"""One bounded diagnostic replay of CLOSED validation cost-probe artifacts.

This is not a new fit, forecast, power study or completed formal method matrix.
No trajectory/map/source-data loader is called. All input hashes are fixed
before execution; output is exclusive. The original cost-only flags survive.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.nex326.model import ModelState
from .method_mechanisms import (VERSION, mechanism_registry, model_gate, score_diagnostic,
                                validate_gate, _digest)
from .method_rollout import MethodDynamics
from .method_training import TRAINING_FIELDS
from .workload import method_inventory

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"artifacts/pirc17/dev10/method-cost-p512-h5-v1"
OUTPUT = ROOT/"artifacts/pirc17/dev10/cached-method-mechanisms-v1.json"
PLAN_SHA = "31a012915b76c67f554480a20e57d52ae2681d2812cd0efa2af6f43281e30c9a"
RESULT_SHA = "55b2ac2c4f7ffbb43a66babe9708d3cad6a5e748c10604afe4fbeb9a173f14ca"
EXPECTED_FIT_SLOTS = {"arm-01/full", "arm-04/gmm_kernel", "arm-05/explicit_decomp", "arm-09/mixed", "arm-14/reptile"}


def _read(path, sha):
    if path.stat().st_size > 2*1024**2:
        raise ValueError("bounded cached input exceeded 2MiB")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("closed cost artifact hash changed")
    return json.loads(raw)


def restore_cached_fit(row):
    data = dict(row["model"])
    data["condition_names"] = tuple(data["condition_names"])
    for key in ("weights", "covariances"):
        data[key] = tuple(np.asarray(x, dtype=float) for x in data[key])
    data["mode_probabilities"] = np.asarray(data["mode_probabilities"], dtype=float)
    d = row["dynamics"]
    model = ModelState(**data)
    result = MethodDynamics.bind(model, reference_interval_seconds=d["reference_interval_seconds"],
        fit_identity=d["fit_identity"], noise_binding_rationale=d["noise_binding_rationale"])
    if result.identity() != d or d["fit_identity"] != _digest({
            "training_identity_sha256": row["training"]["training_identity_sha256"], "model": model.to_dict()}):
        raise ValueError("stored model/training/dynamics identity does not reproduce")
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError("cached diagnostic receipt exists; do not overwrite or rerun")
    started = time.perf_counter()
    def guard():
        if time.perf_counter()-started > 60:
            raise TimeoutError("fixed cached diagnostic wall cap exhausted; no retry")
    plan_path = ROOT/"experiments/pirc17/plans/method-cost-p512-h5-v1.json"
    plan, closed = _read(plan_path, PLAN_SHA), _read(SOURCE/"result.json", RESULT_SHA)
    if (closed["status"] != "complete" or closed["plan_sha256"] != PLAN_SHA
            or closed["final_eval_reads"] != 0 or closed["formal_training_accepted"] is not False
            or closed["numerically_qualified"] is not False or closed["cost_only"] is not True
            or closed["input_sha256"] != plan["expected_input_sha256"]
            or {r["slot_id"] for r in closed["fits"]} != EXPECTED_FIT_SLOTS or len(closed["fits"]) != 5):
        raise ValueError("complete original development cost-only provenance required")
    fits, sources = [], {str(plan_path.relative_to(ROOT)): PLAN_SHA,
                         str((SOURCE/"result.json").relative_to(ROOT)): RESULT_SHA}
    for i, item in enumerate(closed["fits"]):
        if item["path"] != f"fit-{i:02d}.json":
            raise ValueError("registered cached fit order/path changed")
        path = SOURCE/item["path"]
        row = _read(path, item["sha256"])
        if (row["slot_id"] != item["slot_id"] or row["formal_training_accepted"] is not False
                or row["training"]["input_sha256"] != closed["input_sha256"]):
            raise ValueError("cached fit role/identity changed")
        fits.append((row, restore_cached_fit(row), item))
        sources[str(path.relative_to(ROOT))] = item["sha256"]
        guard()
    registry, gates, missing = mechanism_registry(), {}, []
    inventory = {r["slot_id"]: r for r in method_inventory()["slots"]}
    for name, definition in registry["slots"].items():
        if definition.get("source") != "fitted-model":
            continue
        c = inventory[name]["components"]
        key = {k: c[k] for k in TRAINING_FIELDS if k in c}
        matches = [(row, d, item) for row, d, item in fits if row["training"]["training_components"] == key]
        if not matches:
            missing.append(name)
            continue
        if len(matches) != 1:
            raise ValueError("ambiguous cached training-component identity")
        row, dynamics, item = matches[0]
        receipt = model_gate(name, dynamics)
        validate_gate(receipt)
        gates[name] = {"source_fit_slot": item["slot_id"], "source_fit_sha256": item["sha256"], "gate": receipt}
    item = closed["forecasts"][0]
    if item["slot_id"] != "arm-01/full" or item["path"] != "forecast-00.json":
        raise ValueError("exact preregistered cached Full forecast required")
    path = SOURCE/item["path"]
    saved = _read(path, item["sha256"])
    array_path = SOURCE/"forecast-00.npz"
    if (array_path.stat().st_size > 2*1024**2 or hashlib.sha256(array_path.read_bytes()).hexdigest() != saved["artifact_sha256"]
            or saved["cost_only"] is not True or saved["accuracy_claim_authorized"] is not False
            or saved["cost_origin_in_calibration_population"] is not True
            or saved["diagnostics"]["seed"] != plan["seed"] or saved["diagnostics"]["origin_id"] != plan["sample_id"]
            or saved["diagnostics"]["particles"] != 512 or saved["diagnostics"]["max_step_seconds"] != 5.
            or saved["diagnostics"]["dynamics"] != fits[0][1].identity()):
        raise ValueError("bound saved array or original cost-only scope changed")
    sources[str(path.relative_to(ROOT))] = item["sha256"]
    sources[str(array_path.relative_to(ROOT))] = saved["artifact_sha256"]
    with np.load(array_path, allow_pickle=False) as arrays:
        if arrays["elapsed_seconds"].tolist() != closed["actual_horizons_seconds"]:
            raise ValueError("cached observed-time binding changed")
        scores = {name: score_diagnostic(name, arrays["positions_m"], arrays["target_positions_m"],
            seed=plan["seed"], origin_id=plan["sample_id"], input_identity_sha256=closed["input_sha256"])
            for name in ("arm-10/d2_mc", "arm-10/d2_closed")}
    for receipt in scores.values():
        validate_gate(receipt)
    guard()
    report = {"schema_version": "pirc17-cached-method-mechanism-replay-v1", "status": "complete-cached-replay-only",
        "mechanism_registry_sha256": registry["sha256"], "source_artifact_sha256": sources,
        "source_code_sha256": {**registry["source_sha256"], "experiments/pirc17/cached_method_mechanisms.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "source_partition": "validation", "input_identity_sha256": closed["input_sha256"],
        "source_cost_only_flags_preserved": True, "source_origin_is_calibration_exposed": True,
        "model_gates": gates, "required_model_gates_without_cached_fit": missing,
        "cached_Full_score_diagnostic": scores,
        "score_scope": "Both algorithms replay the SAME saved arm01 Full array, not registered arm10 forecast executions or independent evidence.",
        "integration_gate_evidence": "not_available_in_cache: no same-grid exact-kernel reference; no substitution of an unrelated forecast",
        "variance_gate_evidence": "not_available_in_cache: only one origin/seed; no duplication into five seeds or false zero variance",
        "elapsed_seconds_excluding_import_startup": time.perf_counter()-started,
        "wall_cap_seconds": 60, "extra_scoring_only_Gaussian_draws": 2*4*256,
        "new_empirical_fits": 0, "new_empirical_forecasts": 0, "final_eval_reads": 0,
        "formal_matrix_complete": False, "numerically_qualified": False, "power_qualified": False,
        "scientific_claim_authorized": False, "automatic_retry": False}
    report["sha256"] = _digest(report)
    with OUTPUT.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({"status": report["status"], "model_gate_count": len(report["model_gates"]),
        "missing_model_gate_count": len(report["required_model_gates_without_cached_fit"]),
        "cached_score_diagnostics": len(report["cached_Full_score_diagnostic"]),
        "sha256": report["sha256"], "elapsed_seconds": report["elapsed_seconds_excluding_import_startup"]}))
