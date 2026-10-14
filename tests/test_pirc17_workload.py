import copy
import itertools
import json
import math

import pytest

from experiments.pirc17 import workload
from experiments.pirc17.configurations import terrain_configurations
from experiments.pirc17.inference import SEEDS


def timing_fixture():
    """Software-only ledger; no empirical costs or numerical results claimed."""
    plan = {
        "particles": 512, "step_seconds": 5., "scoring_slots": 4,
        "time_weights": [.25]*4, "configurations": list(workload.MEASURED_CONFIGURATIONS),
        "seeds": list(SEEDS), "sample_ids": ["origin-a", "origin-b", "origin-c"],
        "expected_run_count": 90, "final_eval_authorized": False,
    }
    counts = {"expected_run_count": 90, "attempted_run_count": 90,
              "success_count": 90, "failure_count": 0, "unattempted_run_count": 0}
    rows = [{"type": "initialization", "plan": copy.deepcopy(plan), "plan_sha256": "p",
             "source_sha256": {"runtime.py": "s"}, "final_eval_label_prediction_metric_reads": 0},
            {"type": "header", "physical_history_step_seconds": 5., "input_validation_seconds": 2.}]
    for origin, configuration, seed in itertools.product(
            plan["sample_ids"], plan["configurations"], SEEDS):
        rows.append({"type": "run", "status": "success", "sample_id": origin,
            "configuration": configuration, "seed": seed, "particles": 512,
            "max_step_seconds": 5., "independent_block_id": origin+"-block",
            "actual_horizons_seconds": [60., 300., 900., 1800.],
            "rollout_wall_seconds_including_lazy_map_initialization": 1.})
    rows.append({"type": "completion", "status": "complete", **counts,
                 "resource_stopped": False, "terminal_error_count": 0, "elapsed_seconds": 100.,
                 "final_eval_label_prediction_metric_reads": 0})
    audit = {"counts": counts, "status": "complete", "ledger_sha256": "l",
             "plan_sha256": "p", "source_sha256": {"runtime.py": "s"}}
    return plan, rows, audit


def summary(plan, rows, audit):
    return workload.summarize_pilot_timing(plan, rows, audit, plan_sha256="p", ledger_sha256="l")


def test_inventory_keeps_all_slots_exemptions_and_distinguishes_two_matrices():
    inventory = workload.method_inventory()
    assert inventory["ledger_slots"] == 36
    assert inventory["conceptual_arms"] == 22
    assert inventory["required_slots"] == 28 and inventory["excluded_slots"] == 8
    assert {r["arm_id"] for r in inventory["slots"] if r["disposition"] == "EXCLUDED"} == {13, 17, 22}
    assert all(r["exclusion_reason"] for r in inventory["slots"] if r["disposition"] == "EXCLUDED")
    assert inventory["required_propagator_counts"] == {"fp": 26, "mc": 1, "crn": 1}
    assert not inventory["physical_execution_reuse_approved"]
    assert not inventory["formal_adapter_ready"]
    assert all(r["prediction_cost_seconds"] is None for r in inventory["slots"])
    assert len(terrain_configurations()) == 10  # Arm 17 exemption cannot remove these.
    anchors = [r for r in inventory["slots"] if r["full_anchor"]]
    assert len(anchors) == 6 and len({r["component_identity_sha256"] for r in anchors}) == 1
    groups = inventory["identical_component_groups"]
    assert any({r["slot_id"] for r in anchors} <= set(group) for group in groups)
    assert ["arm-19/em", "arm-19/euler"] in groups


def test_complete_timing_summary_accounts_for_units_and_not_numerical_qualification():
    timing = summary(*timing_fixture())
    assert timing["rollout_seconds"] == 90.
    assert timing["initialization_seconds"] == 2.
    assert timing["other_internal_seconds"] == 8.
    assert timing["independent_blocks"] == 3
    assert all(v["count"] == 15 and v["mean_seconds"] == 1. for v in timing["by_configuration"].values())
    assert not timing["recomputed_scientific_or_numerical_qualification"]
    json.dumps(timing, allow_nan=False)


@pytest.mark.parametrize("change", ["duplicate", "failure", "particle", "step", "seed", "origin",
                                  "block", "horizon", "negative", "infinite", "boolean"])
def test_corrupt_row_cannot_support_a_cost_projection(change):
    plan, rows, audit = timing_fixture()
    if change == "duplicate":
        rows[3] = copy.deepcopy(rows[2])
    elif change == "failure":
        rows[2]["status"] = "failure"
    elif change == "particle":
        rows[2]["particles"] = 256
    elif change == "step":
        rows[2]["max_step_seconds"] = 2.5
    elif change == "seed":
        rows[2]["seed"] = 100
    elif change == "origin":
        rows[2]["sample_id"] = "extra"
    elif change == "block":
        rows[2]["independent_block_id"] = "other"
    elif change == "horizon":
        rows[2]["actual_horizons_seconds"][-1] = 300.
    else:
        rows[2]["rollout_wall_seconds_including_lazy_map_initialization"] = {
            "negative": -1., "infinite": math.inf, "boolean": True}[change]
    with pytest.raises(ValueError):
        summary(plan, rows, audit)


@pytest.mark.parametrize("change", ["partial", "unfinished", "resources", "errors", "reads",
                                  "count", "hash", "source", "elapsed", "authorization", "weights"])
def test_incomplete_or_mismatched_whole_batch_is_rejected(change):
    plan, rows, audit = timing_fixture()
    if change == "partial":
        rows.pop(3)
    elif change == "unfinished":
        rows[-1]["status"] = "running"
    elif change == "resources":
        rows[-1]["resource_stopped"] = True
    elif change == "errors":
        rows[-1]["terminal_error_count"] = 1
    elif change == "reads":
        rows[-1]["final_eval_label_prediction_metric_reads"] = 1
    elif change == "count":
        rows[-1]["success_count"] = 89
    elif change == "hash":
        audit["ledger_sha256"] = "changed"
    elif change == "source":
        audit["source_sha256"] = {"runtime.py": "changed"}
    elif change == "elapsed":
        rows[-1]["elapsed_seconds"] = 80.
    elif change == "authorization":
        plan["final_eval_authorized"] = True
    else:
        plan["time_weights"] = [0., 0., 0., 1.]
    with pytest.raises(ValueError):
        summary(plan, rows, audit)


def test_scenario_does_not_multiply_by_four_scores_or_cross_method_and_terrain():
    timing = summary(*timing_fixture())
    result = workload.cost_scenario(timing, independent_blocks=30)
    assert result["terrain_forecasts_six_measured_configurations"] == 900
    assert result["terrain_forecasts_four_unmeasured_lio_configurations"] == 600
    assert result["method_slot_seed_origin_units_without_reuse"] == 4200
    assert result["total_prediction_units_if_same_origins_without_reuse"] == 5700
    assert result["terrain_six_configuration_rollout_estimate_seconds"] == 900.
    assert result["terrain_four_lio_assumption_seconds"] == 600.
    assert result["terrain_ten_configuration_rollout_estimate_hours"] == 1500/3600
    assert result["full_feature_wall_time_estimate_seconds"] is None
    assert result["final_eval_available_blocks"] is None
    twice = workload.cost_scenario(timing, independent_blocks=30, origins_per_block=2)
    assert twice["total_prediction_units_if_same_origins_without_reuse"] == 11400


@pytest.mark.parametrize("n", [True, 0, -1, 3.5, None])
def test_invalid_workload_axis_is_rejected(n):
    with pytest.raises(ValueError):
        workload.cost_scenario(summary(*timing_fixture()), independent_blocks=n)


def test_missing_cost_cannot_silently_become_zero():
    timing = summary(*timing_fixture())
    del timing["by_configuration"]["loo-river"]
    with pytest.raises(ValueError):
        workload.cost_scenario(timing, independent_blocks=30)


def test_committed_accounting_is_integral_and_never_an_authorization():
    path = workload.ROOT/"experiments/pirc17/plans/full-delivery-workload-v1.json"
    report = json.loads(path.read_text())
    digest = report.pop("sha256")
    assert workload._digest(report) == digest
    assert report["state"] == "candidate-accounting-only"
    assert report["execution_admitted"] is False and report["final_eval_authorized"] is False
    assert report["method_matrix"] == workload.method_inventory()
    assert report["terrain_matrix"] == list(terrain_configurations().values())
    assert len(report["scenarios"]) == 3
    assert all(s["full_feature_wall_time_estimate_seconds"] is None for s in report["scenarios"])


def test_cli_reader_rejects_changed_bound_file_before_json_or_forecast_access(tmp_path, monkeypatch):
    monkeypatch.setattr(workload, "ROOT", tmp_path)
    monkeypatch.setattr(workload, "PILOT_BINDINGS", {"plan": ("bad.json", "0"*64)})
    (tmp_path/"bad.json").write_text("not even JSON")
    with pytest.raises(ValueError, match="hash changed"):
        workload.read_pilot_timing()
