from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import experiments.nex326_multi_seed as multi_seed


def test_multi_seed_applies_scope_policy_to_every_replicate(tmp_path, monkeypatch):
    class FakeRunner:
        @classmethod
        def from_paths(cls, *args, **kwargs):
            return SimpleNamespace(cohort=object(), condition_resolver=None)

        def __init__(
            self,
            spec,
            cohort,
            output_root,
            *,
            n_samples,
            replicate_seed,
            strict_environment,
            condition_resolver,
        ):
            self.spec = spec
            self.output_root = Path(output_root)
            self.n_samples = n_samples
            self.replicate_seed = replicate_seed

    calls: list[tuple[int, Path]] = []

    def fake_scoped_run(runner, policy_path):
        policy = Path(policy_path)
        calls.append((runner.replicate_seed, policy))
        runner.output_root.mkdir(parents=True)
        (runner.output_root / "manifest.json").write_text(
            json.dumps({"seed": runner.replicate_seed}), encoding="utf-8"
        )
        return tuple(
            {
                "run_id": f"slot-{slot}",
                "run_status": "succeeded",
                "dataset": {"fingerprint": "cohort-fingerprint"},
                "implementation": {"execution_identity_sha256": "implementation"},
            }
            for slot in range(28)
        )

    monkeypatch.setattr(multi_seed, "NEX326Runner", FakeRunner)
    monkeypatch.setattr(multi_seed, "run_approved_scope", fake_scoped_run)
    policy = tmp_path / "scope.json"
    policy.write_text('{"approved": true}\n', encoding="utf-8")
    cohort = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "nex326"
        / "fixtures"
        / "registered_cohort.json"
    )

    manifest = multi_seed.run_replicates(
        cohort,
        tmp_path / "runs",
        [101, 202],
        scope_policy=policy,
    )

    assert calls == [(101, policy), (202, policy)]
    assert manifest["executions_per_replicate"] == 28
    assert manifest["total_executions"] == 56
    assert manifest["scope_policy"] == {
        "source_file": policy.name,
        "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
    }
    assert [entry["record_count"] for entry in manifest["replicates"]] == [28, 28]
