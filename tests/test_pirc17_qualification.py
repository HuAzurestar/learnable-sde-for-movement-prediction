import hashlib
import json

import pytest

from experiments.pirc17.qualification import audit_windows


def make_release(root):
    samples, alignment = [], []
    for name, split, times in [
        ("ok", "train", [0, 10, 40, 70]),
        ("gap", "train", [0, 10, 90, 120]),
        ("duplicate", "validation", [0, 0, 40, 70]),
        ("sealed", "final_eval", [0, 10, 40, 70]),
    ]:
        samples.append(dict(sample_id=name, segment_id=name, split=split,
            independent_block_id=name, history_start=0, history_end=1,
            target_start=2, target_end=3))
        for i, time in enumerate(times):
            alignment.append(dict(segment_id=name, split=split, segment_point_index=i,
                file_id=name, source_point_index=i,
                absolute_epoch_ns=time * 10**9))
    for filename, rows in [("samples.jsonl", samples), ("alignment.jsonl", alignment)]:
        (root / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "cohort.json").write_text(json.dumps({"cohort_id": "fixture",
        "sample_manifest_sha256": digest("samples.jsonl")}), encoding="utf-8")
    (root / "dataset.json").write_text(json.dumps({"dataset_id": "fixture", "artifacts": {
        name: {"sha256": digest(name)} for name in ("alignment.jsonl", "cohort.json")}}), encoding="utf-8")


def test_elapsed_time_gaps_and_sealed_role(tmp_path):
    make_release(tmp_path)
    report = audit_windows(tmp_path, horizons=(1, 2))
    assert report["splits"]["train"]["denominator_samples"] == 2
    assert report["splits"]["train"]["horizons"] == [
        {"horizon_minutes": 1, "samples": 1, "independent_blocks": 1},
        {"horizon_minutes": 2, "samples": 0, "independent_blocks": 0}]
    assert report["splits"]["validation"]["horizons"][0]["samples"] == 0
    assert "final_eval" not in report["splits"]
    assert report["final_eval_label_prediction_metric_reads"] == 0


def test_changed_timestamp_manifest_is_rejected(tmp_path):
    make_release(tmp_path)
    with (tmp_path / "alignment.jsonl").open("a", encoding="utf-8") as out:
        out.write("{}\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        audit_windows(tmp_path)


@pytest.mark.parametrize("missing_road", [False, True])
def test_common_coverage_uses_joint_status_and_does_not_open_eval(tmp_path, monkeypatch, missing_road):
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    from experiments.nex326 import pirc21_adapter

    make_release(tmp_path)
    rows = []
    for i, seconds in enumerate((0, 10, 40, 70)):
        rows.append(dict(file_id="ok", point_index=i, absolute_epoch_ns=seconds*10**9,
            split="train", dem_surface_status="valid", hydrorivers_river_status="valid",
            worldcover_status="valid", historical_motion_status="valid",
            overture_road_status="source_missing" if missing_road and i == 2 else "valid"))
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet")
    # The nonexistent eval path is a read trap: the metadata audit must exclude it.
    manifest = {"snapshot_id": "test", "content_inventory_sha256": "fixture",
        "files": [{"path": "train.parquet", "split": "train"},
                  {"path": "forbidden-eval.parquet", "split": "final_eval"}]}
    monkeypatch.setattr(pirc21_adapter, "FeatureSnapshotAdapter", lambda *args:
        SimpleNamespace(root=tmp_path, dataset_id="fixture", manifest=manifest))
    report = audit_windows(tmp_path, horizons=(1,), snapshot=tmp_path, eligibility_horizon_minutes=1)
    row = report["splits"]["train"]["horizons"][0]
    assert row["samples"] == 1
    assert row["common_coverage_samples"] == (0 if missing_road else 1)
    assert row["common_coverage_blocks"] == (0 if missing_road else 1)
    assert report["splits"]["validation"]["horizons"][0]["common_coverage_samples"] == 0
    eligible=report["eligibility"]["rows"]
    assert len(eligible)==(0 if missing_road else 1)
    if eligible:
        assert eligible[0]["sample_id"]=="ok" and eligible[0]["file_id"]=="ok"
        assert eligible[0]["split"]=="train"
