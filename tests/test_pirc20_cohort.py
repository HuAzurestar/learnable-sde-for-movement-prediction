from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data.pirc20 import PIRC20CohortError, load_pirc20_cohort


FIXTURE = Path(__file__).parent / "fixtures" / "pirc20" / "cohort.json"


def _write_fixture(tmp_path):
    rows = []
    for split_index, split in enumerate(("train", "validation", "final_eval")):
        identity = {
            "data_version": "fixture-v1",
            "file_id": f"file-{split_index}",
            "segment_id": f"r1:file-{split_index}:0",
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
                "independent_block_id": f"block-{split_index}",
                "factor_availability": {
                    "dem_elev": {
                        "history_valid_points": 2,
                        "history_complete": True,
                        "target_valid_points": 1,
                        "target_complete": True,
                    }
                },
            }
        )
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    ordered = hashlib.sha256()
    for row in rows:
        ordered.update(row["sample_id"].encode())
        ordered.update(b"\n")
    cohort = {
        "schema_version": "pirc20-cohort-v1",
        "cohort_id": "fixture-v1",
        "data_version": "fixture-v1",
        "sample_manifest": "samples.jsonl",
        "sample_manifest_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "ordered_sample_ids_sha256": ordered.hexdigest(),
        "ordered_sample_ids_hash_encoding": "utf8_sample_id_newline_in_manifest_order",
        "sample_count": 3,
        "sample_counts_by_split": {"train": 1, "validation": 1, "final_eval": 1},
        "window": {"history_points": 2, "target_points": 1, "stride_points": 1},
        "final_eval_access": "sealed_identity_only",
    }
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    return cohort_path, sample_path


def test_reader_validates_explicit_hash_bound_cohort(tmp_path):
    cohort_path, _ = _write_fixture(tmp_path)
    cohort = load_pirc20_cohort(cohort_path)
    assert [sample.split for sample in cohort.iter_samples()] == [
        "train",
        "validation",
        "final_eval",
    ]
    assert len(list(cohort.samples_for_model("train"))) == 1
    with pytest.raises(PIRC20CohortError, match="sealed"):
        cohort.samples_for_model("final_eval")
    assert len(list(cohort.samples_for_model("final_eval", final_eval_unlocked=True))) == 1


def test_public_reader_loads_registered_pirc20_fixture():
    cohort = load_pirc20_cohort(FIXTURE)
    assert cohort.cohort_id == "pirc20-psde-fixture-v1"
    assert cohort.sample_counts_by_split == {
        "train": 1,
        "validation": 1,
        "final_eval": 1,
    }


def test_reader_rejects_changed_sample_manifest(tmp_path):
    cohort_path, sample_path = _write_fixture(tmp_path)
    sample_path.write_text(sample_path.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(PIRC20CohortError, match="hash mismatch"):
        load_pirc20_cohort(cohort_path)


def test_reader_rejects_independent_block_leakage(tmp_path):
    cohort_path, sample_path = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    rows[1]["independent_block_id"] = rows[0]["independent_block_id"]
    sample_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    cohort = json.loads(cohort_path.read_text())
    cohort["sample_manifest_sha256"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    with pytest.raises(PIRC20CohortError, match="independent block crosses"):
        load_pirc20_cohort(cohort_path)
