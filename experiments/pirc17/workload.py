"""Read-only, candidate workload accounting; never authorizes an experiment.

The method and terrain matrices are separate. Unknown costs stay unknown, and
identical method components are not permission to reuse arm-dependent streams.
Only the already closed development timing ledger is read by the CLI.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean

from experiments.nex326.specification import load_experiment_spec
from .configurations import terrain_configurations
from .inference import SEEDS

VERSION = "pirc17-candidate-workload-v1"
ROOT = Path(__file__).resolve().parents[2]
MEASURED_CONFIGURATIONS = (
    "base", "all-terrain", "loo-road", "loo-river", "loo-worldcover", "loo-surface",
)
PILOT_BINDINGS = {
    "plan": ("experiments/pirc17/plans/small-budget-full-p512-h5-v1.json",
             "bfa6fa6039ba07d8c97fa0a6dcc8a8e9c3b04e0fcb50714aea42b75aa14dc467"),
    "ledger": ("artifacts/pirc17/dev10/small-budget-full-p512-h5-v1.jsonl",
               "5a67eb8b954382ee2526b8c5bb8b37b106969098254aae83b0a6061119a60d16"),
    "audit": ("artifacts/pirc17/dev10/small-budget-full-p512-h5-v1.audit.json",
              "f098e1351fb7e122c171f42f01ca9ed7284db9d19bbaae81f18da6b52f3b2def"),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_integer(value):
    if type(value) is not int or value < 1:
        raise ValueError("positive integer workload axis required")
    return value


def _seconds(value):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ValueError("finite nonnegative elapsed seconds required")
    return value


def method_inventory():
    """Count all 36 slots, including the eight specifically exempted slots."""
    spec = load_experiment_spec()
    policy = json.loads((ROOT / "experiments/nex326/pirc19_scope_policy.json").read_text())
    exclusions = {row["arm_id"]: row["reason"] for row in policy["approved_excluded_arms"]}
    if set(exclusions) != {13, 17, 22} or len(policy["approved_excluded_arms"]) != 3:
        raise ValueError("PIRC-17 carries only the three human-approved method exemptions")
    rows = []
    for arm, subconfig in spec.executions:
        components = {**spec.full_components, **subconfig["components"]}
        excluded = arm.arm_id in exclusions
        rows.append({
            "slot_id": f"arm-{arm.arm_id:02d}/{subconfig['subconfig_id']}",
            "arm_id": arm.arm_id, "group": arm.group,
            "disposition": "EXCLUDED" if excluded else "REQUIRED",
            "exclusion_reason": exclusions.get(arm.arm_id),
            "full_anchor": bool(arm.control.get("full_anchor", False)),
            "components": components, "component_identity_sha256": _digest(components),
            "propagator": components["poa"],
            "observation_interval_seconds": components["dt_seconds"],
            "prediction_cost_seconds": None,
        })
    if (len(rows) != 36 or sum(r["disposition"] == "REQUIRED" for r in rows) != 28
            or len({r["slot_id"] for r in rows}) != 36):
        raise ValueError("frozen 36-slot / 28-required scope changed")
    groups = defaultdict(list)
    for row in rows:
        if row["disposition"] == "REQUIRED":
            groups[row["component_identity_sha256"]].append(row["slot_id"])
    return {
        "slots": rows, "conceptual_arms": 22, "ledger_slots": 36,
        "required_slots": 28, "excluded_slots": 8,
        "required_propagator_counts": dict(Counter(
            r["propagator"] for r in rows if r["disposition"] == "REQUIRED")),
        "identical_component_groups": [v for v in groups.values() if len(v) > 1],
        "physical_execution_reuse_approved": False,
        "reuse_caveat": "Current runner seeds include arm and subconfig; equal components do not imply equal predictions or interchangeable mechanism records.",
        "observation_interval_caveat": "NEX326 dt changes training/observation resampling; it is not the terrain integrator's maximum step.",
        "formal_adapter_ready": False,
        "adapter_gap": "Legacy runner predicts from each resampled segment midpoint to its endpoint. Bind explicit causal origins, actual horizons, four scoring slots, block IDs and paired seeds before formal PIRC-17 use.",
    }


def summarize_pilot_timing(plan, rows, audit, *, plan_sha256, ledger_sha256):
    """Validate a complete timing grid; do not rescore or requalify predictions."""
    if (plan.get("particles") != 512 or plan.get("step_seconds") != 5.
            or plan.get("scoring_slots") != 4 or plan.get("time_weights") != [.25]*4
            or plan.get("configurations") != list(MEASURED_CONFIGURATIONS)
            or plan.get("seeds") != list(SEEDS) or plan.get("expected_run_count") != 90
            or plan.get("final_eval_authorized") is not False
            or len(plan.get("sample_ids", [])) != 3
            or len(set(plan.get("sample_ids", []))) != 3):
        raise ValueError("timings must describe the registered N512/h5 full development pilot")
    if (len(rows) != 93 or rows[0].get("type") != "initialization"
            or rows[1].get("type") != "header" or rows[-1].get("type") != "completion"
            or any(r.get("type") != "run" for r in rows[2:-1])):
        raise ValueError("complete ordered 90-run timing ledger required")
    start, header, finish = rows[0], rows[1], rows[-1]
    if (start.get("plan") != plan or start.get("plan_sha256") != plan_sha256
            or start.get("final_eval_label_prediction_metric_reads") != 0
            or header.get("physical_history_step_seconds") != 5.
            or finish.get("status") != "complete" or finish.get("resource_stopped") is not False
            or finish.get("final_eval_label_prediction_metric_reads") != 0):
        raise ValueError("closed development ledger identity/status mismatch")
    expected_counts = {"expected_run_count": 90, "attempted_run_count": 90,
                       "success_count": 90, "failure_count": 0, "unattempted_run_count": 0}
    if (any(finish.get(k) != v for k, v in expected_counts.items())
            or finish.get("terminal_error_count") != 0
            or audit.get("counts") != expected_counts or audit.get("status") != "complete"
            or audit.get("ledger_sha256") != ledger_sha256
            or audit.get("plan_sha256") != plan_sha256
            or audit.get("source_sha256") != start.get("source_sha256")):
        raise ValueError("timing audit must bind the complete successful ledger and sources")
    expected = set(itertools.product(plan["sample_ids"], MEASURED_CONFIGURATIONS, SEEDS))
    seen, origin_blocks, horizons_by_origin = set(), {}, {}
    times = defaultdict(list)
    for row in rows[2:-1]:
        key = row["sample_id"], row["configuration"], row["seed"]
        if key not in expected or key in seen or row.get("status") != "success":
            raise ValueError("missing, failed, extra or duplicate timing grid row")
        seen.add(key)
        if row.get("particles") != 512 or row.get("max_step_seconds") != 5.:
            raise ValueError("timing settings differ within pilot")
        origin, block = row["sample_id"], row["independent_block_id"]
        if not block or origin_blocks.setdefault(origin, block) != block:
            raise ValueError("each pilot origin must keep its independent block")
        horizons = row["actual_horizons_seconds"]
        if (len(horizons) != 4 or any(type(v) not in (int, float) or not math.isfinite(v)
                                    or abs(v-nominal) > 30
                                    for v, nominal in zip(horizons, (60, 300, 900, 1800)))
                or horizons_by_origin.setdefault(origin, horizons) != horizons):
            raise ValueError("paired actual forecast horizons changed")
        times[row["configuration"]].append(
            _seconds(row["rollout_wall_seconds_including_lazy_map_initialization"]))
    if seen != expected or len(set(origin_blocks.values())) != 3:
        raise ValueError("complete three-block timing grid required")
    elapsed = _seconds(finish["elapsed_seconds"])
    initialization = _seconds(header["input_validation_seconds"])
    rollout = math.fsum(v for values in times.values() for v in values)
    remainder = elapsed-initialization-rollout
    if remainder < 0:
        raise ValueError("rollout plus initialization exceeds whole-batch elapsed time")
    return {
        "scope": "closed development pilot, not an isolated performance benchmark or numerical certificate",
        "independent_blocks": 3, "origins": 3, "seeds": list(SEEDS),
        "particles": 512, "maximum_step_seconds": 5., "nominal_horizon_seconds": 1800,
        "scoring_slots_from_same_forecast": 4,
        "timing_unit": "seconds per complete 512-particle forecast, including lazy map initialization",
        "by_configuration": {k: {"count": len(v), "mean_seconds": fmean(v),
                                  "min_seconds": min(v), "max_seconds": max(v)}
                             for k, v in times.items()},
        "batch_internal_seconds": elapsed, "initialization_seconds": initialization,
        "rollout_seconds": rollout, "other_internal_seconds": remainder,
        "final_eval_label_prediction_metric_reads": 0,
        "recomputed_scientific_or_numerical_qualification": False,
    }


def read_pilot_timing():
    raw = {}
    for name, (relative, expected) in PILOT_BINDINGS.items():
        value = (ROOT/relative).read_bytes()
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError(f"closed pilot {name} hash changed")
        raw[name] = value
    plan = json.loads(raw["plan"])
    rows = [json.loads(line) for line in raw["ledger"].splitlines()]
    audit = json.loads(raw["audit"])
    for relative, expected in rows[0]["source_sha256"].items():
        if Path(relative).name != relative:
            raise ValueError("unexpected pilot source locator")
        if _file_sha(Path(__file__).parent/relative) != expected:
            raise ValueError(f"pilot runtime source changed: {relative}")
    return summarize_pilot_timing(plan, rows, audit,
        plan_sha256=PILOT_BINDINGS["plan"][1], ledger_sha256=PILOT_BINDINGS["ledger"][1])


def cost_scenario(timing, *, independent_blocks, origins_per_block=1):
    """Extrapolate only like-for-like pilot work, with LIO assumptions separate."""
    blocks = _positive_integer(independent_blocks)
    origins = blocks*_positive_integer(origins_per_block)
    by_config = timing["by_configuration"]
    if set(by_config) != set(MEASURED_CONFIGURATIONS):
        raise ValueError("all six measured configuration costs required")
    means = {k: _seconds(v["mean_seconds"]) for k, v in by_config.items()}
    known = origins*len(SEEDS)*math.fsum(means.values())
    lio_assumed_each = fmean(v for k, v in means.items() if k != "base")
    assumed = origins*len(SEEDS)*4*lio_assumed_each
    return {
        "independent_blocks_scenario_only": blocks, "origins_per_block": origins_per_block,
        "final_eval_available_blocks": None,
        "terrain_forecasts_six_measured_configurations": origins*len(SEEDS)*6,
        "terrain_forecasts_four_unmeasured_lio_configurations": origins*len(SEEDS)*4,
        "method_slot_seed_origin_units_without_reuse": origins*len(SEEDS)*28,
        "total_prediction_units_if_same_origins_without_reuse": origins*len(SEEDS)*(10+28),
        "terrain_six_configuration_rollout_estimate_seconds": known,
        "terrain_four_lio_assumption_seconds": assumed,
        "terrain_ten_configuration_rollout_estimate_hours": (known+assumed)/3600,
        "lio_assumption": "Each LIO costs the mean of five measured terrain-bearing configurations; unmeasured, not an upper bound.",
        "full_feature_wall_time_estimate_seconds": None,
        "not_included": ["actual LIO cost", "method fitting and propagation", "formal input qualification",
                         "full metric registry and independent replay", "failure handling", "tests/review",
                         "manuscript and PDF", "human decision latency"],
        "interpretation": "Serial, same hardware/model/settings conditional projection, not observed formal runtime or a compute cap.",
    }


def workload_report():
    methods = method_inventory()
    terrain = terrain_configurations()
    timing = read_pilot_timing()
    sources = ("experiments/nex326/specification.py", "experiments/nex326/experiment.json",
               "experiments/nex326/pirc19_scope_policy.json", "experiments/nex326/runner.py",
               "experiments/nex326/model.py", "experiments/nex326/cohort.py",
               "experiments/pirc17/configurations.py", "experiments/pirc17/workload.py")
    payload = {
        "schema_version": VERSION, "state": "candidate-accounting-only",
        "execution_admitted": False, "final_eval_authorized": False,
        "source_sha256": {p: _file_sha(ROOT/p) for p in sources},
        "pilot_bindings": {k: {"path": p, "sha256": s} for k, (p, s) in PILOT_BINDINGS.items()},
        "registered_seeds": list(SEEDS), "method_matrix": methods,
        "terrain_matrix": list(terrain.values()), "pilot_timing": timing,
        "scenarios": [cost_scenario(timing, independent_blocks=n) for n in (30, 46, 62)],
        "scenario_count_basis": "30 is the existing inferential floor; 46 is the prior limited pilot power scenario; 62 is development validation availability, not final-eval availability. None selects the formal sample count.",
        "no_cartesian_cross": "28 required method slots PLUS 10 terrain configurations, never 28 times 10.",
        "no_scoring_slot_multiplier": "Four terrain scoring times share one forecast; they are not four independent full rollouts.",
        "stop_policy": "No automatic particle/step/seed/horizon expansion or old N2048 restart. Exact sealed workloads and finite phase/total caps require ACCEPT-01 before final evaluation.",
        "open_before_formal_budget": ["method explicit-origin/horizon adapter and cost evidence",
            "four LIO costs and applicable numerical-evidence limitations", "deterministic block/origin selection",
            "secondary-origin dispositions", "formal fit reuse/refit policy", "complete scoring/audit overhead",
            "hard phase and cumulative caps with non-cherry-picked failure dispositions"],
    }
    return {**payload, "sha256": _digest(payload)}


if __name__ == "__main__":
    print(json.dumps(workload_report(), indent=2, allow_nan=False))
