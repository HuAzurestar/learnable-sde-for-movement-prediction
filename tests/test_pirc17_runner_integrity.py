"""Reject forged gate verdicts and misspelled execution routes."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.nex326.model import train_model
from experiments.nex326.runner import NEX326Runner, RunError, predict_segments, validate_run_record


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    runner = NEX326Runner.from_paths(
        root / "experiments/nex326/fixtures/registered_cohort.json",
        tmp_path_factory.mktemp("integrity"), n_samples=12,
    )
    arm = runner.spec.arms[0]
    return runner.run_one(arm, arm.subconfigs[0])


@pytest.mark.parametrize("mutation", [
    {"value": float("nan")}, {"threshold": float("inf")},
    {"operator": "typo"}, {"value": 0.0, "threshold": 1.0, "operator": "ge", "passed": True},
    {"value": 2.0, "threshold": 1.0, "operator": "le", "passed": True},
])
def test_gate_tampering_is_rejected(record, mutation):
    changed = deepcopy(record)
    changed["mechanism_gates"][0].update(mutation)
    with pytest.raises(RunError, match="mechanism gate"):
        validate_run_record(changed)


@pytest.mark.parametrize("field", ["model", "estimator", "transfer", "finetune"])
def test_unknown_route_fails_before_data_access(field):
    config = {"model": "seg_constant_mode", field: "unregistered"}
    with pytest.raises(ValueError, match="unknown"):
        train_model([], [], [], [], config)


@pytest.mark.parametrize("field", ["poa", "integrator", "bridge"])
def test_unknown_prediction_route_fails_before_data_access(field):
    with pytest.raises(RunError, match="unsupported"):
        predict_segments(None, [], {field: "unregistered"}, seed=1, n_samples=12)
