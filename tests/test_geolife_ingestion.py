import hashlib
import json
import sys

import pandas as pd
import pytest

from data.geolife import audit_geolife_ingestion
from data.loader import SegmentLoader
from data.validation import DataValidationError
from scripts.audit_geolife_ingestion import main as audit_main


def _trajectory_row(file_index, file_id, step, *, geolife):
    return {
        "track_id": file_index,
        "file_id": file_id,
        "cluster_A": -1 if geolife else file_index,
        "country": "China" if geolife else "Fixture",
        "region": "Beijing" if geolife else "Region",
        "city": "Beijing" if geolife else "City",
        "segment_id": "0_0",
        "t": float(step * (4 if geolife else 1)),
        "x": float(file_index + step),
        "y": float(file_index - step),
        "z": float(10 + step) if geolife else float("nan"),
        "vx": 0.25,
        "vy": -0.25,
        "speed": 0.5,
    }


def _write_sources(tmp_path):
    osm_root = tmp_path / "osm"
    gl_root = tmp_path / "geolife"
    osm_root.mkdir()
    gl_root.mkdir()

    gl_rows = []
    for file_index in range(10):
        # Two files per user deliberately expose that file-level splits are not
        # a user-independent evaluation contract.
        file_id = f"GL_{file_index // 2:03d}_{file_index:03d}"
        for step in range(3):
            gl_rows.append(
                _trajectory_row(file_index, file_id, step, geolife=True)
            )
    gl_frame = pd.DataFrame(gl_rows)
    gl_path = gl_root / "geolife_leg.parquet"
    gl_frame.to_parquet(gl_path, index=False)

    manifest = {
        "split": "file_id",
        "ratio": "70/15/15",
        "seed": 20260816,
        "files": {"train": 7, "val": 1, "eval": 2},
        "segments": {"train": 7, "val": 1, "eval": 2},
        "points": {"train": 21, "val": 3, "eval": 6},
    }
    (gl_root / "geolife_splits.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (gl_root / "geolife_longduration_tiers.json").write_text(
        json.dumps(
            {
                "ge6h": {"segments": 0, "users": 0},
                "ge8h": {"segments": 0, "users": 0},
                "ge10h": {"segments": 0, "users": 0},
                "ge12h": {"segments": 0, "users": 0},
                "max_duration_h": 0.01,
                "gap_definition_s": 300.0,
                "note": "synthetic fixture",
            }
        ),
        encoding="utf-8",
    )

    osm_rows = []
    osm_ids = [f"OSM_{index:03d}" for index in range(4)]
    for file_index, file_id in enumerate(osm_ids):
        for step in range(3):
            osm_rows.append(
                _trajectory_row(file_index, file_id, step, geolife=False)
            )
    pd.DataFrame(osm_rows).to_parquet(
        osm_root / "unified_full_leg.parquet", index=False
    )
    (osm_root / "global_splits.json").write_text(
        json.dumps(
            {"train": osm_ids[:2], "val": [osm_ids[2]], "eval": [osm_ids[3]]}
        ),
        encoding="utf-8",
    )

    digest = hashlib.sha256(gl_path.read_bytes()).hexdigest()
    build_stats = {
        "status": "pass",
        "n_failed": 0,
        "schema_identical": True,
        "columns": list(gl_frame.columns),
        "points_cleaned": len(gl_frame),
        "segments": 10,
        "files_with_segments": 10,
        "bytes": gl_path.stat().st_size,
        "sha256": digest,
    }
    (gl_root / "geolife_build_stats.json").write_text(
        json.dumps(build_stats), encoding="utf-8"
    )

    registry = {
        "registry_id": "TEST-GL-001",
        "dataset": "GeoLife test fixture",
        "provider": "fixture provider",
        "url": "https://example.invalid/geolife",
        "version": "test",
        "license": "fixture only; not independently reviewed",
        "cleaned_product": {"sha256": digest},
        "caveats": ["synthetic fixture"],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return osm_root, gl_root, registry_path


def _mutate_build_stats(gl_root, mutate):
    path = gl_root / "geolife_build_stats.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_geolife_split_loads_without_cross_file_segment_merge(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    loader = SegmentLoader(
        data_root=osm_root, geolife_root=gl_root, split="geolife_train"
    )
    assert loader.load() == 7
    assert len({segment.segment_id for segment in loader.segments}) == 7
    assert all(segment.source == "geolife_leg" for segment in loader.segments)
    assert all("raw_segment_id" in segment.meta for segment in loader.segments)


def test_unified_split_contains_both_file_id_namespaces(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    loader = SegmentLoader(
        data_root=osm_root, geolife_root=gl_root, split="unified_train"
    )
    assert loader.load() == 9
    assert {segment.source for segment in loader.segments} == {
        "unified_full_leg",
        "geolife_leg",
    }
    assert len({segment.segment_id for segment in loader.segments}) == 9


def test_unified_max_segments_is_one_total_cap_not_one_per_source(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    loader = SegmentLoader(
        data_root=osm_root,
        geolife_root=gl_root,
        split="unified_train",
        max_segments=1,
    )
    assert loader.load() == 1
    assert len(loader.segments) == 1


def test_negative_max_segments_is_rejected(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        SegmentLoader(
            data_root=osm_root,
            geolife_root=gl_root,
            split="unified_train",
            max_segments=-1,
        )


def test_unified_split_fails_on_cross_source_file_id_namespace_overlap(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    osm = pd.read_parquet(osm_root / "unified_full_leg.parquet")
    osm.loc[0, "file_id"] = "GL_000_000"
    osm.to_parquet(osm_root / "unified_full_leg.parquet", index=False)
    loader = SegmentLoader(
        data_root=osm_root, geolife_root=gl_root, split="unified_train"
    )
    with pytest.raises(DataValidationError, match="namespace overlap"):
        loader.load()


def test_registered_split_count_mismatch_fails_fast(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    manifest_path = gl_root / "geolife_splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["train"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loader = SegmentLoader(
        data_root=osm_root, geolife_root=gl_root, split="geolife_train"
    )
    with pytest.raises(DataValidationError, match="file counts disagree"):
        loader.load()


def test_unknown_split_does_not_silently_load_all_rows(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    loader = SegmentLoader(
        data_root=osm_root, geolife_root=gl_root, split="geolife_everything"
    )
    with pytest.raises(DataValidationError, match="unsupported split"):
        loader.load()


def test_audit_records_missing_provenance_without_hiding_technical_pass(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    report = audit_geolife_ingestion(
        gl_root, osm_root=osm_root, registry_path=None
    )
    assert report["status"] == "provisional"
    assert report["technical_status"] == "pass"
    assert report["release_status"] == "blocked_missing_or_inconsistent_registration"
    assert report["provenance"]["status"] == "unconfirmed"
    assert any(item["area"] == "provenance_registration" for item in report["failure_ledger"])


def test_audit_technical_pass_still_blocks_unreviewed_legal_release(tmp_path):
    osm_root, gl_root, registry_path = _write_sources(tmp_path)
    report = audit_geolife_ingestion(
        gl_root, osm_root=osm_root, registry_path=registry_path
    )
    assert report["status"] == "provisional"
    assert report["technical_status"] == "pass"
    assert report["release_status"] == "blocked_pending_independent_legal_review"
    assert report["quality_evidence"]["status"] == "pass"
    assert report["provenance"]["sha256_match"] is True
    assert report["schema"]["dtypes"]["file_id"] == "string"
    assert report["split"]["partitions"]["train"]["points"] == 21
    assert report["split"]["user_partitioning"]["user_independent"] is False
    namespace = report["cross_source_file_id_namespace"]
    assert namespace["scope"] == "file_id_namespace_only"
    assert namespace["intersection"] == 0
    assert namespace["content_duplicate_check"]["status"] == "not_performed"
    tiers = report["long_duration_boundaries"]
    assert tiers["cleaned_leg_gap60s"]["gap_definition_s"] == 60.0
    assert tiers["walk_filtered_gap300s"]["buckets"]["ge12h"]["segments"] == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(status="failed"),
        lambda value: value.update(n_failed=7),
        lambda value: value.update(schema_identical=False),
        lambda value: value.pop("sha256"),
    ],
)
def test_bad_or_incomplete_build_stats_fail_the_hard_gate(tmp_path, mutate):
    osm_root, gl_root, registry_path = _write_sources(tmp_path)
    _mutate_build_stats(gl_root, mutate)
    report = audit_geolife_ingestion(
        gl_root, osm_root=osm_root, registry_path=registry_path
    )
    assert report["status"] == "failed"
    assert report["technical_status"] == "failed"
    assert report["quality_evidence"]["status"] == "failed"
    assert any(item["area"] == "quality" for item in report["failure_ledger"])


def test_cli_returns_nonzero_and_writes_failure_report_for_negative_probe(
    tmp_path, monkeypatch
):
    osm_root, gl_root, registry_path = _write_sources(tmp_path)
    _mutate_build_stats(
        gl_root,
        lambda value: value.update(status="failed", n_failed=7, schema_identical=False),
    )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_geolife_ingestion.py",
            "--geolife-root",
            str(gl_root),
            "--osm-root",
            str(osm_root),
            "--registry",
            str(registry_path),
            "--output",
            str(output),
        ],
    )
    assert audit_main() == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["quality_evidence"]["status"] == "failed"


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        pytest.param(
            lambda value: value.pop("status"),
            "missing required field: status",
            id="missing-status",
        ),
        pytest.param(
            lambda value: value.update(status="unknown"),
            "status must be one of the successful build states",
            id="unknown-status",
        ),
    ],
)
def test_cli_fails_closed_for_missing_or_unknown_build_status(
    tmp_path, monkeypatch, mutate, expected_error
):
    osm_root, gl_root, registry_path = _write_sources(tmp_path)
    _mutate_build_stats(gl_root, mutate)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_geolife_ingestion.py",
            "--geolife-root",
            str(gl_root),
            "--osm-root",
            str(osm_root),
            "--registry",
            str(registry_path),
            "--output",
            str(output),
        ],
    )
    assert audit_main() == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["technical_status"] == "failed"
    assert report["quality_evidence"]["status"] == "failed"
    assert any(
        expected_error in error for error in report["quality_evidence"]["errors"]
    )
