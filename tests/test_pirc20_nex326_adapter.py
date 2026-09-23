from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.nex326.pirc20_adapter import (
    PIRC20AdapterError,
    load_pirc20_nex326_cohort,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_id(row: dict[str, object]) -> str:
    identity = {
        key: row[key]
        for key in (
            "data_version",
            "file_id",
            "segment_id",
            "history_start",
            "history_end",
            "target_start",
            "target_end",
        )
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_release(tmp_path: Path, *, duplicate_time: bool = False):
    release = tmp_path / "release"
    release.mkdir()
    conditions = tmp_path / "conditions"
    conditions.mkdir()
    data_version = "pirc20-runtime-fixture-v1"
    definitions = (
        ("train-a", "train", "block-a"),
        ("train-b", "train", "block-b"),
        ("validation-a", "validation", "block-c"),
        ("evaluation-a", "final_eval", "block-d"),
    )

    samples = []
    alignment = []
    trajectory_rows = []
    condition_manifest = []
    for sample_index, (file_id, split, block_id) in enumerate(definitions):
        stable_segment = f"r1t:{file_id}:0:0"
        source_segment = f"source-{sample_index}"
        sample = {
            "data_version": data_version,
            "file_id": file_id,
            "segment_id": stable_segment,
            "split": split,
            "independent_block_id": block_id,
            "history_start": 0,
            "history_end": 2,
            "target_start": 3,
            "target_end": 5,
            "factor_availability": {},
        }
        sample["sample_id"] = _sample_id(sample)
        samples.append(sample)
        epochs = [0, 1, 1 if duplicate_time and sample_index == 0 else 2, 3, 4, 5]
        for point_index, epoch in enumerate(epochs):
            alignment.append(
                {
                    "segment_id": stable_segment,
                    "file_id": file_id,
                    "source_segment_id": source_segment,
                    "segment_point_index": point_index,
                    "source_point_index": point_index,
                    "relative_time_s": float(point_index),
                    "absolute_epoch_ns": 1_700_000_000_000_000_000
                    + epoch * 1_000_000_000,
                }
            )
            trajectory_rows.append(
                {
                    "file_id": file_id,
                    "segment_id": source_segment,
                    "t": float(point_index),
                    "x": float(sample_index * 10 + point_index),
                    "y": float(-sample_index * 10 + point_index / 2),
                    "region": "fixture-region",
                    "city": f"fixture-city-{sample_index}",
                }
            )
        condition_path = conditions / f"{file_id}_cond.parquet"
        pd.DataFrame(
            {
                "file_id": [file_id] * 6,
                "solar_elev": [10.0 + value for value in range(6)],
                "is_day": [1.0] * 6,
                "dem_elev": [100.0 + value for value in range(6)],
                "dem_slope": [0.1 + value / 100 for value in range(6)],
            }
        ).to_parquet(condition_path, index=False)
        condition_manifest.append(
            {
                "file_id": file_id,
                "relative_path": condition_path.name,
                "sha256": _sha256(condition_path),
            }
        )

    samples_path = release / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples),
        encoding="utf-8",
    )
    ordered = hashlib.sha256()
    for sample in samples:
        ordered.update(str(sample["sample_id"]).encode())
        ordered.update(b"\n")
    cohort = {
        "schema_version": "pirc20-cohort-v1",
        "cohort_id": data_version,
        "data_version": data_version,
        "sample_manifest": samples_path.name,
        "sample_manifest_sha256": _sha256(samples_path),
        "ordered_sample_ids_sha256": ordered.hexdigest(),
        "ordered_sample_ids_hash_encoding": "utf8_sample_id_newline_in_manifest_order",
        "sample_count": len(samples),
        "sample_counts_by_split": {"train": 2, "validation": 1, "final_eval": 1},
        "window": {
            "mode": "nex326_midpoint",
            "history_points": None,
            "target_points": None,
            "stride_points": 1,
        },
        "final_eval_access": "sealed_identity_only",
    }
    cohort_path = release / "cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    alignment_path = release / "alignment.jsonl"
    alignment_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in alignment),
        encoding="utf-8",
    )
    condition_manifest_path = release / "condition_file_manifest.jsonl"
    condition_manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in condition_manifest),
        encoding="utf-8",
    )
    trajectory_path = tmp_path / "trajectory.parquet"
    pd.DataFrame(trajectory_rows).to_parquet(trajectory_path, index=False)
    dataset = {
        "schema_version": "pirc20-release-v1",
        "dataset_id": data_version,
        "artifacts": {
            name: {"sha256": _sha256(release / name)}
            for name in (
                "cohort.json",
                "alignment.jsonl",
                "condition_file_manifest.jsonl",
            )
        },
        "source": {"trajectory": {"sha256": _sha256(trajectory_path)}},
    }
    (release / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    return cohort_path, trajectory_path, conditions, data_version


def test_adapter_derives_block_disjoint_adapt_and_keeps_final_eval_sealed(tmp_path):
    cohort_path, trajectory, conditions, _ = _write_release(tmp_path)
    cohort = load_pirc20_nex326_cohort(cohort_path, trajectory, conditions)

    assert len(cohort.splits["train"]) == 1
    assert len(cohort.splits["adapt"]) == 1
    assert len(cohort.splits["validation"]) == 1
    assert cohort.splits["evaluation"] == ()
    assert "sealed" in cohort.unavailable_reasons["evaluation"]
    assert {
        segment.segment_id
        for split in ("train", "adapt")
        for segment in cohort.splits[split]
    } == {"r1t:train-a:0:0", "r1t:train-b:0:0"}
    assert all(
        segment.has_terrain
        for split in ("train", "adapt", "validation")
        for segment in cohort.splits[split]
    )
    assert {
        segment.independent_block_id
        for split in ("train", "adapt", "validation")
        for segment in cohort.splits[split]
    } == {"block-a", "block-b", "block-c"}


def test_bounded_pilot_selection_is_explicit_and_fingerprinted(tmp_path):
    cohort_path, trajectory, conditions, _ = _write_release(tmp_path)
    full = load_pirc20_nex326_cohort(cohort_path, trajectory, conditions)
    bounded = load_pirc20_nex326_cohort(
        cohort_path,
        trajectory,
        conditions,
        maximum_segments_per_role={"train": 1, "adapt": 1, "validation": 1},
    )

    assert {
        name: len(bounded.splits[name]) for name in ("train", "adapt", "validation")
    } == {
        "train": 1,
        "adapt": 1,
        "validation": 1,
    }
    assert bounded.fingerprint != full.fingerprint
    with pytest.raises(PIRC20AdapterError, match="maximum_segments_per_role"):
        load_pirc20_nex326_cohort(
            cohort_path,
            trajectory,
            conditions,
            maximum_segments_per_role={"train": 0},
        )


def test_exact_cohort_id_unlocks_final_eval_and_preserves_sample_boundary(tmp_path):
    cohort_path, trajectory, conditions, cohort_id = _write_release(tmp_path)
    cohort = load_pirc20_nex326_cohort(
        cohort_path,
        trajectory,
        conditions,
        final_eval_unlock=cohort_id,
    )

    assert len(cohort.splits["evaluation"]) == 1
    evaluation = cohort.splits["evaluation"][0]
    assert max(1, len(evaluation.time) // 2 - 1) == 2
    assert evaluation.segment_id == "r1t:evaluation-a:0:0"
    with pytest.raises(PIRC20AdapterError, match="acknowledgement"):
        load_pirc20_nex326_cohort(
            cohort_path,
            trajectory,
            conditions,
            final_eval_unlock="wrong-cohort",
        )


def test_duplicate_exact_time_keeps_first_observed_state_without_target_leakage(
    tmp_path,
):
    cohort_path, trajectory, conditions, _ = _write_release(
        tmp_path, duplicate_time=True
    )
    cohort = load_pirc20_nex326_cohort(cohort_path, trajectory, conditions)
    segment = next(
        segment
        for split in ("train", "adapt")
        for segment in cohort.splits[split]
        if "train-a" in segment.segment_id
    )

    assert len(segment.time) == 5
    start = max(1, len(segment.time) // 2 - 1)
    assert start == 1
    assert segment.state[start, 0] == 1.0
    assert (segment.time[1:] > segment.time[:-1]).all()


def test_adapter_rejects_changed_trajectory_source(tmp_path):
    cohort_path, trajectory, conditions, _ = _write_release(tmp_path)
    changed = pd.read_parquet(trajectory)
    changed.loc[0, "x"] += 1.0
    changed.to_parquet(trajectory, index=False)
    with pytest.raises(PIRC20AdapterError, match="trajectory source hash mismatch"):
        load_pirc20_nex326_cohort(cohort_path, trajectory, conditions)
