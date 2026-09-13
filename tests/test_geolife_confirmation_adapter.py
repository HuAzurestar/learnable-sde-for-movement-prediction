import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.nex326.geolife_confirmation_adapter import (
    GeoLifeConfirmationAdapterError,
    load_geolife_confirmation_cohort,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    release_id = "pirc20-geolife-fixture-v1"
    release = tmp_path / release_id
    release.mkdir()
    source_rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    ordered_digest = hashlib.sha256()
    counts = {"train": 0, "validation": 0, "final_eval": 0}
    for user in range(10):
        split = "train" if user < 7 else "validation" if user < 9 else "final_eval"
        file_id = f"GL_{user:03d}_20200101000000"
        source_segment_id = f"{user}_0"
        for point in range(6):
            source_rows.append(
                {
                    "file_id": file_id,
                    "segment_id": source_segment_id,
                    "t": float(point * 5),
                    "x": float(point + user),
                    "y": float(user - point),
                    "region": "Beijing",
                    "city": "Beijing",
                }
            )
        identity = {
            "data_version": release_id,
            "file_id": file_id,
            "segment_id": f"geolife:{source_segment_id}",
            "history_start": 0,
            "history_end": 2,
            "target_start": 3,
            "target_end": 5,
        }
        sample_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        samples.append(
            {
                "sample_id": sample_id,
                **identity,
                "split": split,
                "independent_block_id": hashlib.sha256(
                    f"geolife-user-v1:{user:03d}".encode()
                ).hexdigest(),
                "factor_availability": {},
            }
        )
        counts[split] += 1
        ordered_digest.update(f"{sample_id}\n".encode())
    trajectory = tmp_path / "geolife.parquet"
    pq.write_table(pa.Table.from_pylist(source_rows), trajectory)
    condition_rows = [
        {
            "file_id": row["file_id"],
            "segment_id": row["segment_id"],
            "point_index": index % 6,
            "solar_elev": float((index % 6) - 3),
            "is_day": int(index % 6 > 3),
        }
        for index, row in enumerate(source_rows)
    ]
    condition_path = tmp_path / "geolife_solar_conditions.parquet"
    pq.write_table(pa.Table.from_pylist(condition_rows), condition_path)
    condition_receipt = tmp_path / "receipt.json"
    _write_json(
        condition_receipt,
        {
            "schema_version": "pirc20-geolife-solar-conditions-v1",
            "source": {"cleaned_trajectory_sha256": _sha256(trajectory)},
            "artifact": {
                "path": condition_path.name,
                "sha256": _sha256(condition_path),
            },
        },
    )
    samples_path = release / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in samples),
        encoding="utf-8",
    )
    cohort = {
        "schema_version": "pirc20-cohort-v1",
        "cohort_id": release_id,
        "data_version": release_id,
        "sample_manifest": "samples.jsonl",
        "sample_manifest_sha256": _sha256(samples_path),
        "ordered_sample_ids_sha256": ordered_digest.hexdigest(),
        "ordered_sample_ids_hash_encoding": "utf8_sample_id_newline_in_manifest_order",
        "sample_count": len(samples),
        "sample_counts_by_split": counts,
        "window": {"mode": "nex326_midpoint"},
        "final_eval_access": "sealed_identity_only",
    }
    _write_json(release / "cohort.json", cohort)
    for name in ("split.json", "coverage_report.json", "leakage_report.json"):
        _write_json(release / name, {"fixture": True})
    artifacts = {
        name: {"sha256": _sha256(release / name)}
        for name in (
            "cohort.json",
            "samples.jsonl",
            "split.json",
            "coverage_report.json",
            "leakage_report.json",
        )
    }
    _write_json(
        release / "dataset.json",
        {
            "schema_version": "pirc20-geolife-release-v1",
            "dataset_id": release_id,
            "source": {"trajectory": {"sha256": _sha256(trajectory)}},
            "artifacts": artifacts,
        },
    )
    return release / "cohort.json", trajectory, condition_receipt, release_id


def test_adapter_keeps_evaluation_sealed_and_loads_user_partition(tmp_path):
    cohort_path, trajectory, condition_receipt, release_id = _fixture(tmp_path)

    sealed = load_geolife_confirmation_cohort(
        cohort_path, trajectory, condition_receipt
    )
    assert not sealed.splits["evaluation"]
    assert sealed.splits["train"]
    assert sealed.splits["adapt"]
    assert sealed.splits["validation"]
    assert "evaluation" in sealed.unavailable_reasons
    assert all(
        set(segment.conditions) == {"solar_elev", "is_day"}
        for values in sealed.splits.values()
        for segment in values
    )

    opened = load_geolife_confirmation_cohort(
        cohort_path,
        trajectory,
        condition_receipt,
        final_eval_unlock=release_id,
    )
    assert len(opened.splits["evaluation"]) == 1
    assert "evaluation" not in opened.unavailable_reasons
    assert opened.purpose == "pirc20_external_domain_independent_confirmation"


def test_adapter_rejects_wrong_final_eval_acknowledgement(tmp_path):
    cohort_path, trajectory, condition_receipt, _ = _fixture(tmp_path)
    with pytest.raises(GeoLifeConfirmationAdapterError, match="acknowledgement"):
        load_geolife_confirmation_cohort(
            cohort_path,
            trajectory,
            condition_receipt,
            final_eval_unlock="wrong",
        )
