"""Read-only combined cost projection; never an admission or runtime bound.

Use closed measurements for both matrices. Unmeasured FP mechanisms use the
largest of the three observed FP costs as an explicit planning assumption, not
as evidence that every mechanism has that cost. Keep non-prediction gaps open.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .workload import method_inventory

ROOT = Path(__file__).resolve().parents[2]
BINDINGS = {
    "terrain": ("experiments/pirc17/evidence/terrain-cost-v1.json",
                "905aed0dcbc2ed448af10fc9ae8541fd1067b719bb45e45ae38426e3a9de4600"),
    "methods": ("experiments/pirc17/evidence/method-cost-v1.json",
                "61c98536ee15ebcfda3df821030cd6355e2fc3bf6aa695acea917313ad775d65"),
}


def read_evidence():
    output = {}
    for name, (relative, expected) in BINDINGS.items():
        raw = (ROOT / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("closed public cost evidence changed")
        output[name] = json.loads(raw)
    return output


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("measured positive finite seconds required")
    return value


def combined_scenario(evidence, *, independent_blocks):
    if type(independent_blocks) is not int or independent_blocks < 30:
        raise ValueError("scenario requires at least the registered 30 independent blocks")
    terrain, methods = evidence["terrain"], evidence["methods"]
    for report in (terrain, methods):
        if report["execution_admitted"] is not False or report["final_eval_authorized"] is not False:
            raise ValueError("cost evidence cannot admit execution")
        if report["settings"]["particles"] != 512 or report["settings"]["maximum_step_seconds"] != 5:
            raise ValueError("only matched N512/h5 measurements can be combined")
    inventory = method_inventory()
    if inventory["required_propagator_counts"] != {"fp": 26, "mc": 1, "crn": 1}:
        raise ValueError("required method propagator counts changed")
    fp_keys = {"arm-01/full", "arm-04/gmm_kernel", "arm-05/explicit_decomp"}
    forecasts = methods["by_forecast"]
    if set(forecasts) != fp_keys | {"arm-21/mc", "arm-21/crn"}:
        raise ValueError("all five measured method forecasts required")
    if any(r["count"] != 1 for r in forecasts.values()):
        raise ValueError("single-trial method observations must not be inflated")
    fp_cost = max(_seconds(forecasts[k]["forecast_seconds"]) for k in fp_keys)
    score_cost = max(_seconds(r["common_score_seconds"]) for r in forecasts.values())
    if len(terrain["by_configuration"]) != 10:
        raise ValueError("complete ten-configuration terrain costs required")
    terrain_per_repeat = math.fsum(_seconds(r["mean_seconds"]) for r in terrain["by_configuration"].values())
    method_per_repeat = 26 * fp_cost + math.fsum(_seconds(forecasts[k]["forecast_seconds"])
                                              for k in ("arm-21/mc", "arm-21/crn"))
    repeats = independent_blocks * 5
    terrain_seconds = repeats * terrain_per_repeat
    method_seconds = repeats * method_per_repeat
    score_seconds = repeats * 38 * score_cost
    return {
        "independent_blocks_scenario_only": independent_blocks, "origins_per_block": 1, "seed_count": 5,
        "method_forecasts": repeats * 28, "terrain_forecasts": repeats * 10,
        "complete_forecasts": repeats * 38, "scoring_slots_per_forecast": 4,
        "assumed_fp_seconds_per_complete_forecast": fp_cost,
        "assumed_common_score_seconds_per_forecast": score_cost,
        "terrain_prediction_seconds": terrain_seconds, "method_prediction_seconds": method_seconds,
        "common_scoring_seconds": score_seconds,
        "prediction_and_common_scoring_hours": (terrain_seconds + method_seconds + score_seconds) / 3600,
        "scope": "same hardware/runtime/settings conditional serial projection, not a cap or full project ETA",
        "method_cost_assumption": "all 26 FP slots use the largest of three observed FP costs; MC/CRN use their own single trial",
        "score_cost_assumption": "all 38 slots/configurations use the largest observed method common-score cost; excludes extra method-specific metrics",
        "final_eval_available_blocks": None, "formal_physical_fit_count": None,
        "full_feature_wall_time_estimate_seconds": None, "execution_admitted": False,
    }


def report():
    evidence = read_evidence()
    return {
        "schema_version": "pirc17-combined-cost-projection-v1", "execution_admitted": False,
        "final_eval_authorized": False, "new_empirical_fits": 0, "new_empirical_forecasts": 0,
        "evidence": {name: {"path": p, "sha256": s} for name, (p, s) in BINDINGS.items()},
        "method_sampling": "5 single-trial fits, 5 single-trial forecasts, one calibration-exposed validation origin/seed",
        "scenarios": [combined_scenario(evidence, independent_blocks=n) for n in (30, 46, 62)],
        "not_included": ["formal data qualification/loading", "sealed training population and physical fit policy",
            "extra method-specific scores", "independent replay/audit", "failure and resource reserve",
            "software tests/review", "full manuscript/PDF", "human decision latency"],
        "next": "select exact finite formal workload and phase/total caps, seal protocol/matrix, test/review, obtain ACCEPT-01",
        "no_new_probe_requested": True, "full_feature_wall_time_estimate_seconds": None,
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
