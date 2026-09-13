from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.nex326.confirmation_readiness import (
    ConfirmationReadinessError,
    build_confirmation_readiness,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cohort(root: Path, *, version: str, identity_prefix: str) -> Path:
    root.mkdir()
    rows = []
    for index, split in enumerate(("train", "validation", "final_eval")):
        identity = {
            "data_version": version,
            "file_id": f"{identity_prefix}-file-{index}",
            "segment_id": f"{identity_prefix}:segment-{index}",
            "history_start": 0,
            "history_end": 1,
            "target_start": 2,
            "target_end": 2,
        }
        sample_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        rows.append(
            {
                "sample_id": sample_id,
                **identity,
                "split": split,
                "independent_block_id": f"{identity_prefix}-block-{index}",
                "factor_availability": {},
            }
        )
    manifest = root / "samples.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    ordered = hashlib.sha256()
    for row in rows:
        ordered.update(row["sample_id"].encode())
        ordered.update(b"\n")
    cohort = root / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "schema_version": "pirc20-cohort-v1",
                "cohort_id": version,
                "data_version": version,
                "sample_manifest": manifest.name,
                "sample_manifest_sha256": _sha(manifest),
                "ordered_sample_ids_sha256": ordered.hexdigest(),
                "ordered_sample_ids_hash_encoding": "utf8_sample_id_newline_in_manifest_order",
                "sample_count": 3,
                "sample_counts_by_split": {
                    "train": 1,
                    "validation": 1,
                    "final_eval": 1,
                },
                "window": {"history_points": 2, "target_points": 1},
                "final_eval_access": "sealed_identity_only",
            }
        ),
        encoding="utf-8",
    )
    return cohort


def _inputs(tmp_path: Path, *, overlap: bool = False):
    discovery = _cohort(tmp_path / "discovery", version="discovery-v1", identity_prefix="d")
    confirmation = _cohort(
        tmp_path / "confirmation",
        version="confirmation-v1",
        identity_prefix="d" if overlap else "c",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "pirc20-confirmation-policy-v1",
                "discovery_evidence": {
                    "cohort_id": "discovery-v1",
                    "cohort_sha256": _sha(discovery),
                },
                "confirmation_data": {
                    "required_splits": ["train", "validation", "final_eval"],
                    "required_zero_overlap_identifiers": [
                        "sample_id",
                        "file_id",
                        "segment_id",
                        "independent_block_id",
                    ],
                },
                "confirmation_execution": {
                    "replicate_seed_count": 3,
                    "required_execution_count_per_seed": 28,
                    "approved_excluded_arm_ids": [13, 17, 22],
                },
                "decision_rule": {
                    "guardrails": [
                        {
                            "metric": "hdr90_abs_error_from_90",
                            "noninferiority_margin": None,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "pirc20-candidate-selection-v1",
                "policy": {"sha256": _sha(policy)},
                "candidate_count": 1,
                "candidates": [
                    {
                        "arm_id": 6,
                        "subconfig_id": "dt30",
                        "reference_arm_id": 6,
                        "reference_subconfig_id": "dt60",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return policy, selection, discovery, confirmation


def test_confirmation_readiness_proves_zero_overlap_before_unlock(tmp_path):
    policy, selection, discovery, confirmation = _inputs(tmp_path)
    result = build_confirmation_readiness(
        policy, selection, discovery, confirmation, tmp_path / "readiness.json"
    )
    assert result["status"] == "waiting_for_decision_threshold"
    assert result["independence"]["overlap_counts"] == {
        "sample_id": 0,
        "file_id": 0,
        "segment_id": 0,
        "independent_block_id": 0,
    }
    assert len(set(result["execution"]["replicate_seeds"])) == 3
    assert result["execution"]["final_eval_unlock"] == "confirmation-v1"


def test_confirmation_readiness_rejects_source_identity_overlap(tmp_path):
    policy, selection, discovery, confirmation = _inputs(tmp_path, overlap=True)
    with pytest.raises(ConfirmationReadinessError, match="overlaps discovery"):
        build_confirmation_readiness(
            policy, selection, discovery, confirmation, tmp_path / "readiness.json"
        )
