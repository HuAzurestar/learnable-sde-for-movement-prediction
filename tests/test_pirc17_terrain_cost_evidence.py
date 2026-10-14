"""Portable arithmetic/provenance checks; no private data or forecasts are read."""
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re

from experiments.pirc17.configurations import terrain_configurations


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/pirc17/evidence/terrain-cost-v1.json"


def evidence():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_public_provenance_keeps_closed_input_bindings_and_old_means():
    report = evidence()
    for binding in report["provenance"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])
        assert not PureWindowsPath(binding["path"]).is_absolute()
        assert ".." not in Path(binding["path"]).parts
    for name in ("original_workload_report", "lio_plan"):
        binding = report["provenance"][name]
        assert hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
    old = json.loads((ROOT / report["provenance"]["original_workload_report"]["path"]).read_text())
    for name, stats in old["pilot_timing"]["by_configuration"].items():
        assert report["by_configuration"][name] == stats
    assert report["sampling"]["six_configurations"]["seeds"] == old["registered_seeds"]


def test_ten_configuration_costs_keep_one_vs_fifteen_observation_denominators():
    report = evidence()
    costs = report["by_configuration"]
    assert set(costs) == set(terrain_configurations())
    assert sum(row["count"] for row in costs.values()) == 94
    for name, row in costs.items():
        assert row["count"] == (1 if name.startswith("lio-") else 15)
        assert all(math.isfinite(row[key]) and row[key] > 0
                   for key in ("mean_seconds", "min_seconds", "max_seconds"))
        assert row["min_seconds"] <= row["mean_seconds"] <= row["max_seconds"]
        if row["count"] == 1:
            assert row["min_seconds"] == row["mean_seconds"] == row["max_seconds"]
    assert report["sampling"]["four_lio"]["origins"] == 1
    assert report["sampling"]["four_lio"]["seeds"] == [20260814]


def test_conditional_projection_has_no_scoring_or_method_cartesian_multiplier():
    report = evidence()
    per_seed = math.fsum(row["mean_seconds"] for row in report["by_configuration"].values())
    assert [row["independent_blocks_scenario_only"] for row in report["scenarios"]] == [30, 46, 62]
    for row in report["scenarios"]:
        assert row["origins_per_block"] == 1 and row["seed_count"] == 5
        repeats = row["independent_blocks_scenario_only"] * row["origins_per_block"] * row["seed_count"]
        assert row["terrain_forecasts"] == repeats * 10
        assert math.isclose(row["terrain_only_prediction_hours"], repeats * per_seed / 3600, rel_tol=1e-12)
        assert row["full_feature_wall_time_estimate_seconds"] is None
        assert row["final_eval_available_blocks"] is None


def test_closed_lio_batch_and_budget_arithmetic_are_not_extra_authority():
    report = evidence()
    batch, budget = report["lio_batch"], report["new_preparation_empirical_budget"]
    assert batch["status"] == "complete"
    assert batch["expected_run_count"] == batch["attempted_run_count"] == batch["success_count"] == 4
    assert batch["failure_count"] == batch["unattempted_run_count"] == batch["terminal_error_count"] == 0
    assert batch["job_active_processes_at_close"] == batch["numerical_sensitivities"] == 0
    assert math.isclose(batch["internal_seconds"], sum(batch[key] for key in
                        ("input_validation_seconds", "rollout_seconds", "other_internal_seconds")), rel_tol=1e-12)
    lio_total = math.fsum(row["mean_seconds"] for name, row in report["by_configuration"].items()
                          if name.startswith("lio-"))
    assert math.isclose(batch["rollout_seconds"], lio_total, rel_tol=1e-12)
    assert batch["internal_seconds"] < budget["inner_batch_cap_seconds"] == 600
    assert batch["internal_seconds"] < batch["outer_seconds"] < budget["outer_batch_cap_seconds"] == 660
    assert budget["charged_outer_seconds"] == batch["outer_seconds"]
    assert budget["cap_seconds"] == budget["prior_seconds"] + budget["charged_outer_seconds"] + budget["remaining_seconds"]
    assert budget["additional_execution_authorized"] is False


def test_cost_observations_do_not_become_scientific_qualification_or_formal_permission():
    report = evidence()
    for flag in ("execution_admitted", "final_eval_authorized", "numerically_qualified"):
        assert report[flag] is False
    assert report["final_eval_label_prediction_metric_reads"] == 0
    assert report["settings"]["particles"] == 512
    assert report["settings"]["maximum_step_seconds"] == 5
    assert report["settings"]["nominal_horizon_seconds"] == 1800
    assert report["settings"]["scoring_slots_from_same_forecast"] == 4
    assert report["recorded_runtime"]["hardware_identity"] is None
    assert len(report["caveats"]) == 8
