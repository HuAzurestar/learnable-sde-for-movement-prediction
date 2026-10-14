import copy
import json

import pytest

from experiments.pirc17 import lio_cost_probe as probe


def plan():
    return json.loads((probe.Path(__file__).resolve().parents[1]/
        "experiments/pirc17/plans/lio-cost-p512-h5-v1.json").read_text())


def test_only_four_fixed_cost_forecasts_and_original_models_are_admitted():
    value = plan()
    assert probe.validate_plan(value) == value
    assert value["axes"]["configurations"] == ["lio-road", "lio-river", "lio-worldcover", "lio-surface"]
    assert value["expected_run_count"] == 4
    assert value["outer_wall_seconds"] == 660
    assert value["final_eval_authorized"] is False


@pytest.mark.parametrize("change", ["N", "h", "seed", "H", "origins", "config", "cap", "retry", "source", "model", "extra"])
def test_probe_rejects_scope_expansion_or_model_changes(change):
    value = copy.deepcopy(plan())
    if change == "N":
        value["axes"]["particles"] = [1024]
    elif change == "h":
        value["axes"]["steps"] = [2.5]
    elif change == "seed":
        value["axes"]["seeds"].append(20260815)
    elif change == "H":
        value["nominal_horizon_seconds"] = 3600.
    elif change == "origins":
        value["axes"]["limit_origins"] = 2
    elif change == "config":
        value["axes"]["configurations"].append("all-terrain")
    elif change == "cap":
        value["outer_wall_seconds"] = 1200
    elif change == "retry":
        value["automatic_retry"] = True
    elif change == "source":
        value["source_sha256"]["direct_linear_rollout.py"] = "0"*64
    elif change == "model":
        value["evidence"]["fit_sha256"] = "0"*64
    else:
        value["approve_final_eval"] = True
    with pytest.raises(ValueError):
        probe.validate_plan(value)


def test_existing_output_forbids_a_second_launch(tmp_path, monkeypatch):
    value = plan()
    monkeypatch.setattr(probe, "OUTPUT", tmp_path/"probe.jsonl")
    probe.OUTPUT.with_suffix(".probe.json").write_text("preserved")
    monkeypatch.setattr(probe, "read_bound", lambda *args: value)
    monkeypatch.setattr(probe, "validate_plan", lambda p: p)
    monkeypatch.setattr(probe, "run_owned", lambda *args, **kwargs: pytest.fail("must not spawn"))
    with pytest.raises(FileExistsError):
        probe.supervise("plan", "a"*64)
    assert probe.OUTPUT.with_suffix(".probe.json").read_text() == "preserved"
