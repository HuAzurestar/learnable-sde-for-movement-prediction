"""Public-safe extraction of ALREADY CLOSED development qualification.

This is not another audit or experiment. Exact immutable report hashes bind the
previous independent audits; only their aggregate planning inputs are exported.
No trajectories, particle arrays, model fitting or final-evaluation data access.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .inference import PRIMARY_FAMILY, SEEDS
from .method_comparisons import DELTA_M, FAMILY_DEFINITIONS

VERSION = "pirc17-closed-qualification-record-v1"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/pirc17/evidence/closed-qualification-v1.json"
BINDINGS = {
    "terrain_power": ("candidate-primary-five-seed-power-p512-v1.json",
        "29e92c11d2642f0ec334920bb51fafaad8b4b24dec0b0be7431a2497aed3a771"),
    "terrain_fine": ("small-budget-offline-v1.json",
        "a70a367e7bce829f45f96e81365f46ccd5224b8551dd6af159c5018cf7b61877"),
    "terrain_coarse": ("small-budget-full-p512-h5-v1.audit.json",
        "f098e1351fb7e122c171f42f01ca9ed7284db9d19bbaae81f18da6b52f3b2def"),
    "methods": ("method-qualification-p512-h5-v1/result.json",
        "fbedee50aee338f230bc7251f3b1c2cea6414f3e320ec51d8b266e212bdcc7e0"),
    "method_audit": ("method-qualification-p512-h5-v1/audit.json",
        "b52e0a91bf1e371ef50d3a213e5306d10053a7ac1ec718ba001aa73d5c4cfa7d"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_bound(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("closed qualification evidence hash mismatch")
    return json.loads(raw)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite real aggregate required")
    return float(value)


def build_record(directory):
    sources = {name: read_bound(Path(directory) / path, sha) for name, (path, sha) in BINDINGS.items()}
    power, fine, coarse, methods, audit = (sources[k] for k in BINDINGS)
    if (methods["status"] != "complete" or methods["final_eval_reads"] != 0
            or audit["status"] != "audited-saved-evidence"
            or audit["result_sha256"] != BINDINGS["methods"][1]
            or audit["forecast_arrays_checked"] != 435
            or audit["observed_time_score_records_checked"] != 1740
            or audit["fitted_models_checked"] != 16
            or audit["method_planning_comparisons_checked"] != 21
            or power["final_eval_label_prediction_metric_reads"] != 0
            or coarse["final_eval_label_prediction_metric_reads"] != 0):
        raise ValueError("complete closed development evidence required")
    terrain = {}
    for name in PRIMARY_FAMILY:
        fine_row, = [r for r in fine["aggregate_precision"] if r["kind"] == "small_budget_contrast"
                     and r["comparison"] == name and r["particles"] == 512]
        coarse_row, = [r for r in coarse["aggregate_precision"] if r["kind"] == "coarse_contrast"
                       and r["comparison"] == name]
        change, = [r for r in coarse["aggregate_precision"] if r["kind"] == "coarse_minus_fine"
                   and r["comparison"] == name]
        fm = _finite(fine_row["simultaneous_normal_mc_margin_m"])
        cm = _finite(coarse_row["normal_mc_margin_m"])
        step = _finite(change["absolute_change_plus_margin_m"])
        terrain[name] = {"paired_block_sd_m": _finite(power["planning"]["results"][name]["paired_block_sd_m"]),
            "planning_sd_step_seconds": .3125, "fine_mc_margin_m": fm, "coarse_mc_margin_m": cm,
            "step_change_plus_margin_m": step, "combined_pilot_margin_m": max(fm, cm) + step,
            "epsilon_m": DELTA_M / 4}
    families = {}
    for family_id, contrasts in FAMILY_DEFINITIONS.items():
        actual = methods["analysis"]["families"][family_id]
        if actual["status"] != "computed" or set(actual["results"]) != set(contrasts):
            raise ValueError("complete registered method planning family required")
        families[family_id] = {name: {
            "paired_block_sd_m": _finite(actual["results"][name]["paired_block_sd_m"]),
            "simulation_mean_standard_error_m": _finite(actual["results"][name]["simulation_mean_standard_error_m"]),
        } for name in contrasts}
    variance = methods["analysis"]["variance"]["gates"]["arm-21/crn"]["evidence"]
    payload = {"schema_version": VERSION, "source_file_sha256": {k: v[1] for k, v in BINDINGS.items()},
        "partition": "validation", "independent_blocks": 3, "origins_per_block": 1,
        "registered_seeds": list(SEEDS), "delta_m": DELTA_M,
        "terrain_primary": terrain, "method_families": families,
        "crn_difference_variance": {k: variance[k] for k in
            ("numerator_m2", "denominator_m2", "descriptive_block_bootstrap_interval",
             "independent_block_count", "seed_count", "variance_ddof")},
        "verified_method_counts": {k: audit[k] for k in ("fitted_models_checked", "model_gates_checked",
            "forecast_arrays_checked", "observed_time_score_records_checked", "method_planning_comparisons_checked")},
        "limitations": ["Only three calibration-exposed development blocks; not final effects or a representative population certificate.",
            "Terrain power SD comes from the finite N512/h0.3125 reference, not a final-data or coarse-grid variance estimate.",
            "Terrain margins concern only five weighted-ES primary contrasts, not LIO, secondary origin modes, every time slot or other metrics.",
            "Finite N512 reference, no five-seed N512-to-larger-N convergence certificate.",
            "Method simulation SE is descriptive, not a total numerical-error or particle-refinement bound.",
            "Same-grid exact affine-kernel diagnostics are not a globally exact nonlinear SDE reference.",
            "Near-zero CRN-FP variance is pair-conditional; ratio is not speedup or a universal variance-reduction guarantee."],
        "new_fits": 0, "new_forecasts": 0, "final_eval_reads": 0,
        "numerically_qualified": False, "scientific_claim_authorized": False}
    return {**payload, "sha256": digest(payload)}


def write_record(directory, output=OUTPUT):
    record = build_record(directory)
    with Path(output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record


if __name__ == "__main__":
    record = write_record(ROOT / "artifacts/pirc17/dev10")
    print(json.dumps({"path": str(OUTPUT.relative_to(ROOT)), "sha256": record["sha256"],
                      "new_forecasts": 0, "final_eval_reads": 0}))
