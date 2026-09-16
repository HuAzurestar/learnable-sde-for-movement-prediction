from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.nex326_multi_seed_compose as compose_module
from experiments.nex326_multi_seed_compose import ComposeError, compose_replicates


def _inputs(tmp_path: Path):
    batch_root = tmp_path / "runs"
    roots = tuple(batch_root / f"seed-{seed}" for seed in (202, 101))
    for root in roots:
        root.mkdir(parents=True)
        (root / "manifest.json").write_text("{}", encoding="utf-8")
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {"schema_version": "pirc20-cohort-v1", "cohort_id": "pirc20-fixture"}
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "scope.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "pirc19-reproduction-scope-v1",
                "experiment_id": "NEX326",
                "required_execution_count": 28,
                "approved_excluded_arms": [
                    {"arm_id": arm_id, "reason": "fixture"}
                    for arm_id in (13, 17, 22)
                ],
            }
        ),
        encoding="utf-8",
    )
    return batch_root, roots, cohort, policy


def test_compose_replicates_orders_and_binds_verified_roots(tmp_path, monkeypatch):
    batch_root, roots, cohort, policy = _inputs(tmp_path)
    policy_hash = compose_module._sha256(policy)

    def fake_verified(root):
        seed = int(root.name.removeprefix("seed-"))
        implementation = {
            "source_bundle_sha256": "a" * 64,
            "execution_identity_sha256": "b" * 64,
        }
        manifest = {
            "scope": {"policy_sha256": policy_hash},
            "record_count": 28,
            "run_status": {"succeeded": 28},
            "replicate_seed": seed,
            "protocol_seed": 7,
            "requested_prediction_samples": 64,
            "spec_version": "nex326-process-v2",
            "implementation": implementation,
        }
        records = tuple(
            {
                "replicate_seed": seed,
                "run_status": "succeeded",
                "dataset": {
                    "dataset_id": "pirc20-fixture",
                    "fingerprint": "f" * 64,
                },
            }
            for _ in range(28)
        )
        return manifest, records

    monkeypatch.setattr(compose_module, "_verified_run", fake_verified)
    output = batch_root / "combined.json"
    manifest = compose_replicates(
        cohort, roots, output, scope_policy=policy
    )

    assert manifest["replicate_seeds"] == [101, 202]
    assert manifest["executions_per_replicate"] == 28
    assert manifest["total_executions"] == 56
    assert manifest["scope_policy"]["sha256"] == policy_hash
    assert output.is_file()


def test_compose_replicates_rejects_root_outside_manifest_parent(tmp_path, monkeypatch):
    batch_root, roots, cohort, policy = _inputs(tmp_path)
    outside = tmp_path / "outside-seed"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(compose_module, "_verified_run", lambda root: ({}, ()))

    with pytest.raises(ComposeError, match="escapes the combined manifest root"):
        compose_replicates(
            cohort,
            [outside, roots[0]],
            batch_root / "combined.json",
            scope_policy=policy,
        )
