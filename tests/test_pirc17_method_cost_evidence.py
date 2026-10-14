"""Portable cost/provenance tests; no private inputs, fitting or forecasting."""
import copy
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import pytest

from experiments.pirc17 import delivery_budget as budget


def test_public_method_evidence_preserves_exact_sources_and_preregistration():
    value = budget.read_evidence()["methods"]
    for binding in value["provenance"].values():
        assert not PureWindowsPath(binding["path"]).is_absolute()
        assert ".." not in Path(binding["path"]).parts
        assert len(binding["sha256"]) == 64
    for path, expected in value["source_sha256"].items():
        assert hashlib.sha256((budget.ROOT/path).read_bytes()).hexdigest() == expected
    bound = value["provenance"]["plan"]
    assert hashlib.sha256((budget.ROOT/bound["path"]).read_bytes()).hexdigest() == bound["sha256"]
    plan = json.loads((budget.ROOT/bound["path"]).read_text())
    assert value["input_identity_sha256"] == plan["expected_input_sha256"]
    assert set(value["by_fit"]) == set(plan["fit_slots"])
    assert set(value["by_forecast"]) == {r["slot_id"] for r in plan["forecasts"]}
    assert value["batch_internal_seconds"] < 600 and value["batch_outer_seconds"] < 660
    assert math.isclose(value["cumulative_preparation_empirical_seconds"],
                        plan["prior_preparation_empirical_seconds"]+value["batch_outer_seconds"], rel_tol=1e-12)


def test_saved_projection_reproduces_without_private_inputs_or_new_runs():
    saved = json.loads((budget.ROOT/"experiments/pirc17/plans/delivery-cost-projection-v1.json").read_text())
    assert saved == budget.report()
    assert saved["new_empirical_fits"] == saved["new_empirical_forecasts"] == 0
    assert saved["no_new_probe_requested"] and not saved["execution_admitted"]


def test_no_accuracy_claim_or_fictitious_total_budget_from_cost_observations():
    value = budget.read_evidence()["methods"]
    for key in ("execution_admitted", "final_eval_authorized", "numerically_qualified",
                "formal_training_accepted", "accuracy_claim_authorized"):
        assert value[key] is False
    assert value["cost_origin_in_calibration_population"] is True
    assert value["final_eval_label_prediction_metric_reads"] == 0
    assert value["full_feature_wall_time_estimate_seconds"] is None
    assert all(r["count"] == 1 for r in value["by_fit"].values())
    assert all(r["count"] == 1 for r in value["by_forecast"].values())
    assert value["independent_audit"]["scoring_time_rows_recomputed"] == 20


@pytest.mark.parametrize("blocks", [30, 46, 62])
def test_projection_keeps_matrices_additive_and_separates_assumptions(blocks):
    source = budget.read_evidence()
    result = budget.combined_scenario(source, independent_blocks=blocks)
    repeat = blocks * 5
    assert result["complete_forecasts"] == repeat * (28+10)
    assert result["method_forecasts"] == repeat * 28
    assert result["terrain_forecasts"] == repeat * 10
    forecasts = source["methods"]["by_forecast"]
    fp_max = max(r["forecast_seconds"] for r in forecasts.values() if r["propagation"] == "fp")
    method = repeat * (26*fp_max+forecasts["arm-21/mc"]["forecast_seconds"]+forecasts["arm-21/crn"]["forecast_seconds"])
    assert math.isclose(method, result["method_prediction_seconds"], rel_tol=1e-12)
    assert result["prediction_and_common_scoring_hours"] > result["terrain_prediction_seconds"]/3600
    assert result["full_feature_wall_time_estimate_seconds"] is None
    assert result["formal_physical_fit_count"] is None
    assert not result["execution_admitted"]


@pytest.mark.parametrize("blocks", [True, 0, 29, 30.0, "30"])
def test_invalid_block_counts_are_not_silently_cast(blocks):
    with pytest.raises(ValueError):
        budget.combined_scenario(budget.read_evidence(), independent_blocks=blocks)


@pytest.mark.parametrize("change", ["setting", "missing", "inflated", "nan", "admitted"])
def test_incompatible_evidence_cannot_produce_a_budget(change):
    source = copy.deepcopy(budget.read_evidence())
    methods = source["methods"]
    if change == "setting": methods["settings"]["particles"] = 1024
    elif change == "missing": del methods["by_forecast"]["arm-21/crn"]
    elif change == "inflated": methods["by_forecast"]["arm-01/full"]["count"] = 5
    elif change == "nan": methods["by_forecast"]["arm-01/full"]["forecast_seconds"] = float("nan")
    else: methods["execution_admitted"] = True
    with pytest.raises(ValueError):
        budget.combined_scenario(source, independent_blocks=30)
