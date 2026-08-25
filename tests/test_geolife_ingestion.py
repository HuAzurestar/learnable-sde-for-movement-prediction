import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from data.geolife import audit_geolife_ingestion
from data.loader import SegmentLoader
from data.validation import DataValidationError


def _write_sources(tmp_path):
    osm_root = tmp_path / "osm"
    gl_root = tmp_path / "geolife"
    osm_root.mkdir()
    gl_root.mkdir()

    gl_rows = []
    for file_index in range(10):
        file_id = f"GL_{file_index:03d}"
        # raw segment IDs intentionally repeat across files; the adapter must
        # not merge these ten independent trajectories.
        for step in range(3):
            gl_rows.append(
                {
                    "file_id": file_id,
                    "segment_id": "0_0",
                    "t": float(step * 4),
                    "x": float(file_index + step),
                    "y": float(file_index - step),
                }
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
    (gl_root / "geolife_build_stats.json").write_text(
        json.dumps({"n_failed": 0, "schema_identical": True}), encoding="utf-8"
    )

    osm_rows = []
    osm_ids = [f"OSM_{index:03d}" for index in range(4)]
    for file_index, file_id in enumerate(osm_ids):
        for step in range(3):
            osm_rows.append(
                {
                    "file_id": file_id,
                    "segment_id": "0_0",
                    "t": float(step),
                    "x": float(file_index + step),
                    "y": float(file_index - step),
                }
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
    registry = {
        "registry_id": "TEST-GL-001",
        "dataset": "GeoLife test fixture",
        "provider": "fixture provider",
        "url": "https://example.invalid/geolife",
        "version": "test",
        "license": "fixture only",
        "cleaned_product": {"sha256": digest},
        "caveats": ["synthetic fixture"],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return osm_root, gl_root, registry_path


def test_geolife_split_loads_without_cross_file_segment_merge(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    loader = SegmentLoader(
        data_root=osm_root, geolife_root=gl_root, split="geolife_train"
    )
    assert loader.load() == 7
    assert len({segment.segment_id for segment in loader.segments}) == 7
    assert all(segment.source == "geolife_leg" for segment in loader.segments)
    assert all("raw_segment_id" in segment.meta for segment in loader.segments)


def test_unified_split_contains_both_disjoint_sources(tmp_path):
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


def test_unified_split_fails_on_cross_source_file_id_leakage(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    osm = pd.read_parquet(osm_root / "unified_full_leg.parquet")
    osm.loc[0, "file_id"] = "GL_000"
    osm.to_parquet(osm_root / "unified_full_leg.parquet", index=False)
    loader = SegmentLoader(
        data_root=osm_root, geolife_root=gl_root, split="unified_train"
    )
    with pytest.raises(DataValidationError, match="leakage"):
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


def test_audit_records_missing_provenance_in_failure_ledger(tmp_path):
    osm_root, gl_root, _ = _write_sources(tmp_path)
    report = audit_geolife_ingestion(
        gl_root, osm_root=osm_root, registry_path=None, verify_sha256=False
    )
    assert report["status"] == "provisional"
    assert report["provenance"]["status"] == "unconfirmed"
    assert report["failure_ledger"][0]["area"] == "provenance"


def test_audit_passes_registered_hash_and_exact_counts(tmp_path):
    osm_root, gl_root, registry_path = _write_sources(tmp_path)
    report = audit_geolife_ingestion(
        gl_root, osm_root=osm_root, registry_path=registry_path
    )
    assert report["status"] == "pass"
    assert report["provenance"]["sha256_match"] is True
    assert report["split"]["partitions"]["train"]["points"] == 21
    assert report["cross_source_leakage"]["intersection"] == 0
