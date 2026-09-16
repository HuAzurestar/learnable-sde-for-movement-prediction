from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.nex326.scoped_runner import ScopeRunError, run_approved_scope
from experiments.nex326.specification import load_experiment_spec


class _FakeRunner:
    def __init__(self, output_root: Path) -> None:
        self.spec = load_experiment_spec()
        self.output_root = output_root
        self.replicate_seed = self.spec.seed
        self.n_samples = 64
        self.implementation = {"execution_identity_sha256": "fixture"}
        self.called: list[int] = []

    def run_one(self, arm, subconfig):
        self.called.append(arm.arm_id)
        subconfig_id = str(subconfig["subconfig_id"])
        return {
            "arm_id": arm.arm_id,
            "run_id": f"arm-{arm.arm_id}-{subconfig_id}",
            "run_status": "succeeded",
        }


def test_scoped_runner_executes_only_the_28_approved_slots(tmp_path):
    runner = _FakeRunner(tmp_path / "runs")
    records = run_approved_scope(runner)  # type: ignore[arg-type]
    manifest = json.loads((runner.output_root / "manifest.json").read_text())

    assert len(records) == 28
    assert set(runner.called).isdisjoint({13, 17, 22})
    assert len(set(runner.called)) == 19
    assert manifest["schema_version"] == "nex326-scoped-run-manifest-v1"
    assert manifest["scope"]["required_execution_count"] == 28
    assert {
        item["arm_id"] for item in manifest["scope"]["approved_excluded_arms"]
    } == {13, 17, 22}


def test_scoped_runner_rejects_scope_drift(tmp_path):
    policy = json.loads(
        Path("experiments/nex326/pirc19_scope_policy.json").read_text(encoding="utf-8")
    )
    policy["approved_excluded_arms"].pop()
    policy_path = tmp_path / "changed-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ScopeRunError, match="exactly 13, 17, and 22"):
        run_approved_scope(_FakeRunner(tmp_path / "runs"), policy_path)  # type: ignore[arg-type]
