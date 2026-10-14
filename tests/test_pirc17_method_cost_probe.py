"""Dispatch/resource software tests only; no empirical cost experiments."""
import copy
from dataclasses import replace
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pirc17 import method_cost_probe as probe
from experiments.pirc17.method_development import MethodDevelopment
from experiments.pirc17.method_inputs import AssignedSample, bind_method_prefix, development_training_segment
from tests.test_pirc17_method_inputs import EPOCH, sample


def plan():
    return {**copy.deepcopy(probe.CONSTANTS), "source_sha256": probe.source_hashes()}


def prepared():
    s = replace(sample(split="validation"), sample_id=probe.CONSTANTS["sample_id"], target_end=6)
    times = EPOCH + np.array([0, 5, 10, 70, 310, 910, 1810], dtype=np.int64) * 10**9
    positions = np.column_stack((114 + np.arange(7)*.0001, 22 + np.arange(7)*.00005))
    prefix = bind_method_prefix(AssignedSample(s, "validation", "software-only-population"),
        times[:3], positions[:3], source_identity="software-only-source")
    segment, _ = development_training_segment(prefix, times, positions,
        region="software-region", region_identity="software-only-region")
    return MethodDevelopment((prefix,), (segment,), {"sha256": "software-only-input"})


def test_exact_plan_keeps_five_fits_five_predictions_and_one_existing_seed():
    value = plan()
    assert probe.validate_plan(value) == value
    assert len(value["fit_slots"]) == len(value["forecasts"]) == 5
    assert value["prior_preparation_empirical_seconds"] + value["outer_wall_seconds"] < 1800
    assert value["cost_origin_in_calibration_population"] and not value["accuracy_claim_authorized"]
    assert not value["formal_fit_reuse_authorized"]


def test_redacted_saved_preregistration_is_not_a_new_executable_binding():
    saved = json.loads((probe.ROOT / "experiments/pirc17/plans/method-cost-p512-h5-v1.json").read_text())
    assert saved["source_trajectory"] == "private/source-trajectories/unified_full_leg.parquet"
    assert saved["source_sha256"] != probe.source_hashes()
    with pytest.raises(ValueError, match="exact fixed method cost plan"):
        probe.validate_plan(saved)


@pytest.mark.parametrize("key,value", [
    ("particles", 1024), ("max_step_seconds", 2.5), ("seed", 20260815),
    ("nominal_seconds", [60, 300, 900, 3600]), ("origin_count", 2),
    ("outer_wall_seconds", 720), ("automatic_retry", True), ("final_eval_authorized", True),
    ("expected_input_sha256", "0"*64), ("max_transitions", 80000),
    ("fit_slots", ["arm-13/animal_pretrain"]), ("source_sha256", {}),
])
def test_scope_or_source_changes_fail_before_loading(key, value):
    changed = plan()
    changed[key] = value
    with pytest.raises(ValueError, match="exact fixed"):
        probe.validate_plan(changed)


def test_target_binding_uses_original_observations_and_common_origin_frame():
    source = prepared()
    prefix, horizons, targets, frame = probe.cost_targets(source, probe.CONSTANTS["sample_id"])
    np.testing.assert_array_equal(horizons, [60, 300, 900, 1800])
    np.testing.assert_allclose(probe.positions_in_scoring_frame(prefix, prefix.origin.position_m[None], frame), 0, atol=1e-9)
    np.testing.assert_array_equal(targets, probe.positions_in_scoring_frame(prefix, source.segments[0].state[3:], frame))
    with pytest.raises(ValueError, match="exact admitted"):
        probe.cost_targets(source, "different-sample")
    with pytest.raises(ValueError, match="exact admitted"):
        probe.cost_targets(replace(source, prefixes=source.prefixes*2, segments=source.segments*2), probe.CONSTANTS["sample_id"])


def test_budget_refuses_work_after_total_cap_or_low_memory(monkeypatch):
    value = plan()
    monkeypatch.setattr(probe, "available_physical_bytes", lambda: 2**40)
    budget = probe.PhaseBudget(value, io.StringIO())
    budget.started -= 601
    with pytest.raises(TimeoutError):
        budget.run("fit", "slot", lambda: pytest.fail("no over-budget fit"))
    budget = probe.PhaseBudget(value, io.StringIO())
    monkeypatch.setattr(probe, "available_physical_bytes", lambda: 0)
    with pytest.raises(MemoryError):
        budget.run("fit", "slot", lambda: pytest.fail("no low-memory fit"))


def test_one_failure_stops_dispatch_without_retry(monkeypatch, tmp_path):
    value = plan()
    calls = []
    def fail(*args, **kwargs):
        calls.append(args[1])
        raise ValueError("software failure")
    monkeypatch.setattr(probe, "fit_development_method", fail)
    monkeypatch.setattr(probe, "available_physical_bytes", lambda: 2**40)
    monkeypatch.setattr(probe, "forecast_method", lambda *a, **k: pytest.fail("no predictions after failure"))
    with pytest.raises(ValueError, match="software failure"):
        probe.execute(prepared(), value, tmp_path, probe.PhaseBudget(value, io.StringIO()))
    assert calls == ["arm-01/full"] and not list(tmp_path.iterdir())


def test_existing_attempt_is_never_restarted(monkeypatch, tmp_path):
    path = tmp_path / "attempt"
    path.mkdir()
    probe.publish(path / "preserved.json", {"existing": True})
    value = plan()
    monkeypatch.setattr(probe, "OUTPUT", path)
    monkeypatch.setattr(probe, "read_bound", lambda *a: value)
    monkeypatch.setattr(probe, "run_owned", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(FileExistsError):
        probe.supervise("plan", "a"*64)
    assert json.loads((path / "preserved.json").read_text()) == {"existing": True}


def test_complete_dispatch_fits_once_per_recipe_and_passes_no_targets_to_prediction(monkeypatch, tmp_path):
    value = plan()
    value["particles"] = 4  # Software seam only, not an admitted production plan.
    fit_calls, forecast_calls = [], []
    model = SimpleNamespace(condition_names=("solar_elev",), to_dict=lambda: {"software_only": True})
    dynamics = SimpleNamespace(model=model, identity=lambda: {"software_only": True})
    def fit(data, slot, **kw):
        fit_calls.append(slot)
        return SimpleNamespace(dynamics=dynamics, training={"transitions_by_role": value["expected_transitions"]}, fit_seconds=.01)
    def forecast(dynamics, origin, horizons, **kwargs):
        forecast_calls.append(kwargs)
        assert "target" not in kwargs and not hasattr(origin, "segment")
        positions = np.broadcast_to(origin.position_m, (4, 4, 2)).copy()
        return SimpleNamespace(forecast=SimpleNamespace(positions_m=positions), diagnostics={"software_only": True},
                               conditional_means_m=None, conditional_covariances_m2=None)
    monkeypatch.setattr(probe, "fit_development_method", fit)
    monkeypatch.setattr(probe, "forecast_method", forecast)
    monkeypatch.setattr(probe, "available_physical_bytes", lambda: 2**40)
    result = probe.execute(prepared(), value, tmp_path, probe.PhaseBudget(value, io.StringIO()))
    assert fit_calls == value["fit_slots"]
    assert [r["propagation"] for r in forecast_calls] == ["fp", "fp", "fp", "mc", "crn"]
    assert forecast_calls[0]["crn_pair_id"] == forecast_calls[-1]["crn_pair_id"]
    assert forecast_calls[-2]["crn_pair_id"] is None
    assert len(result["fits"]) == len(result["forecasts"]) == 5
    assert len(list(tmp_path.glob("*.npz"))) == 5
    with pytest.raises(FileExistsError):
        probe.publish(tmp_path / "fit-00.json", {"overwrite": True})
