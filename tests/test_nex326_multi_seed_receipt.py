from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.nex326.multi_seed_receipt import (
    MultiSeedReceiptError,
    build_multi_seed_receipt,
)


def _reference(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _receipt_inputs(tmp_path: Path, *, scoped: bool = False):
    seeds = (101, 202)
    fingerprint = "f" * 64
    replicate_entries = []
    aggregate_inputs = []
    for seed in seeds:
        records_root = tmp_path / f"seed-{seed}"
        records_root.mkdir()
        manifest = records_root / "manifest.json"
        manifest_payload: dict[str, object] = {"replicate_seed": seed}
        if scoped:
            manifest_payload.update(
                {
                    "record_count": 28,
                    "run_status": {"succeeded": 28},
                    "arm_ids": [
                        arm_id
                        for arm_id in range(1, 23)
                        if arm_id not in {13, 17, 22}
                    ],
                    "scope": {"policy_sha256": "a" * 64},
                }
            )
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        replicate_entries.append(
            {
                "replicate_seed": seed,
                "manifest": _reference(manifest, tmp_path),
                **(
                    {"record_count": 28, "run_status": {"succeeded": 28}}
                    if scoped
                    else {}
                ),
            }
        )
        aggregate_root = tmp_path / f"aggregate-seed-{seed}"
        aggregate_root.mkdir()
        comparisons = aggregate_root / "nex326_pilot_comparisons.csv"
        comparisons.write_text("arm_id,delta\n1,0\n", encoding="utf-8")
        summary = aggregate_root / "nex326_summary.json"
        summary.write_text(
            json.dumps(
                {
                    "replicate_seed": seed,
                    "artifacts": {
                        "nex326_pilot_comparisons.csv": _reference(
                            comparisons, aggregate_root
                        )
                    },
                }
            ),
            encoding="utf-8",
        )
        aggregate_inputs.append(
            {
                "replicate_seed": seed,
                "summary_path": summary.relative_to(tmp_path).as_posix(),
                "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "comparisons_path": comparisons.relative_to(tmp_path).as_posix(),
                "comparisons_sha256": hashlib.sha256(comparisons.read_bytes()).hexdigest(),
            }
        )
    batch = {
        "schema_version": "nex326-multi-seed-manifest-v1",
        "spec_version": "nex326-process-v2",
        "protocol_seed": 20260814,
        "replicate_seeds": list(seeds),
        "cohort": {"fingerprint": fingerprint},
        "replicates": replicate_entries,
    }
    if scoped:
        batch.update(
            {
                "executions_per_replicate": 28,
                "total_executions": 28 * len(seeds),
                "scope_policy": {
                    "source_file": "pirc19_scope_policy.json",
                    "sha256": "a" * 64,
                },
            }
        )
    batch_path = tmp_path / "multi_seed_manifest.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    combined_root = tmp_path / "aggregate-replicates"
    combined_root.mkdir()
    table = combined_root / "nex326_cross_replicate_comparisons.csv"
    table.write_text("arm_id,mean\n1,0\n", encoding="utf-8")
    combined = {
        "schema_version": "nex326-tsde-replicate-aggregate-v1",
        "spec_version": "nex326-process-v2",
        "scientific_status": "exploratory_only",
        "assessment": "not_assessed",
        "method": "descriptive_across_replicate_seeds_no_inferential_ci",
        "protocol_seed": 20260814,
        "replicate_seeds": list(seeds),
        "replicate_count": len(seeds),
        "execution_count_per_replicate": 28 if scoped else 36,
        "dataset_fingerprint": fingerprint,
        "batch_manifest_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
        "comparison_status": {"reference": 8, "exploratory_point_estimate": 22},
        "replicate_status": (
            {"succeeded": 28}
            if scoped
            else {"succeeded": 30, "data_unavailable": 6}
        ),
        "inputs": aggregate_inputs,
        "artifacts": {table.name: _reference(table, combined_root)},
    }
    if scoped:
        combined["scope"] = {
            "policy_file": "pirc19_scope_policy.json",
            "policy_sha256": "a" * 64,
            "approved_excluded_arm_ids": [13, 17, 22],
        }
    combined_path = combined_root / "nex326_replicate_summary.json"
    combined_path.write_text(json.dumps(combined), encoding="utf-8")
    return batch_path, combined_path, table


def test_multi_seed_receipt_verifies_the_complete_hash_chain(tmp_path):
    batch, summary, table = _receipt_inputs(tmp_path)
    receipt = build_multi_seed_receipt(batch, summary, tmp_path / "receipt.json")
    assert receipt["replicate_seeds"] == [101, 202]
    assert receipt["total_execution_count"] == 72
    assert receipt["assessment"] == "not_assessed"

    table.write_text("tampered", encoding="utf-8")
    with pytest.raises(MultiSeedReceiptError, match="size mismatch|SHA-256 mismatch"):
        build_multi_seed_receipt(batch, summary, tmp_path / "invalid.json")


def test_multi_seed_receipt_binds_the_approved_scoped_matrix(tmp_path):
    batch, summary, _ = _receipt_inputs(tmp_path, scoped=True)
    receipt = build_multi_seed_receipt(batch, summary, tmp_path / "receipt.json")
    assert receipt["execution_count_per_replicate"] == 28
    assert receipt["total_execution_count"] == 56
    assert receipt["scope"] == {
        "policy_file": "pirc19_scope_policy.json",
        "policy_sha256": "a" * 64,
        "approved_excluded_arm_ids": [13, 17, 22],
        "required_execution_count": 28,
    }


def test_multi_seed_receipt_rejects_scope_hash_disagreement(tmp_path):
    batch, summary, _ = _receipt_inputs(tmp_path, scoped=True)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["scope"]["policy_sha256"] = "b" * 64
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MultiSeedReceiptError, match="approved 28-slot matrix"):
        build_multi_seed_receipt(batch, summary, tmp_path / "invalid.json")


def test_checked_in_pirc20_receipt_is_privacy_safe_and_audit_bound():
    root = Path(__file__).parents[1] / "experiments" / "nex326"
    receipt_path = root / "pirc20_r1t_multi_seed_receipt.json"
    audit = json.loads((root / "pirc20_runtime_audit.json").read_text(encoding="utf-8"))
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)

    assert receipt["replicate_seeds"] == [20260814, 20260815, 20260816]
    assert receipt["total_execution_count"] == 84
    assert receipt["scope"]["approved_excluded_arm_ids"] == [13, 17, 22]
    assert audit["multi_seed_evaluation"]["portable_receipt"]["sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert all(
        sensitive not in receipt_text
        for sensitive in ('"sample_id"', '"file_id"', '"coordinates"', '"timestamps"')
    )
