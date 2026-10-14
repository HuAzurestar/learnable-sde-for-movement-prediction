"""Audit elapsed-time support using identity/timestamp metadata only.

Never opens trajectory positions, features, labels, predictions or metrics.
Final-eval rows are excluded from horizon qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import duckdb


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_windows(release: Path, horizons=(1, 5, 15, 30, 60, 120), max_gap_seconds=60,
                  snapshot: Path | None = None, eligibility_horizon_minutes=None):
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0 or not horizons or any(not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError("positive horizons and gap tolerance are required")
    if eligibility_horizon_minutes is not None and (snapshot is None or not math.isfinite(eligibility_horizon_minutes) or eligibility_horizon_minutes<=0):
        raise ValueError("eligibility manifest requires a snapshot and positive finite horizon")
    release = Path(release)
    dataset_path, cohort_path = release / "dataset.json", release / "cohort.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if dataset["dataset_id"] != cohort["cohort_id"]:
        raise ValueError("dataset/cohort identity mismatch")
    sources = {name: release / name for name in ("samples.jsonl", "alignment.jsonl")}
    hashes = {name: _hash(path) for name, path in sources.items()}
    if hashes["samples.jsonl"] != cohort["sample_manifest_sha256"]:
        raise ValueError("sample manifest hash mismatch")
    for name in ("alignment.jsonl", "cohort.json"):
        actual = hashes.get(name) or _hash(release / name)
        if actual != dataset["artifacts"][name]["sha256"]:
            raise ValueError(f"release artifact hash mismatch: {name}")
    db = duckdb.connect()
    try:
        db.execute("SET threads=4")
        db.execute("SET memory_limit='2GB'")
        # Explicit projection makes the data boundary auditable.
        db.execute("""CREATE TEMP TABLE samples AS
            SELECT * FROM read_json(?, columns={sample_id:'VARCHAR', segment_id:'VARCHAR',
              independent_block_id:'VARCHAR', split:'VARCHAR', history_start:'BIGINT',
              history_end:'BIGINT', target_start:'BIGINT', target_end:'BIGINT'})
            WHERE split IN ('train', 'validation')""", [str(sources["samples.jsonl"])])
        db.execute("""CREATE TEMP TABLE selected AS
                SELECT s.*, a.segment_point_index, a.absolute_epoch_ns,
                  a.file_id, a.source_point_index,
                  a.absolute_epoch_ns - lag(a.absolute_epoch_ns) OVER
                    (PARTITION BY s.sample_id ORDER BY a.segment_point_index) AS gap_ns
                FROM read_json(?, columns={segment_id:'VARCHAR', split:'VARCHAR',
                    segment_point_index:'BIGINT', absolute_epoch_ns:'BIGINT',
                    file_id:'VARCHAR', source_point_index:'BIGINT'}) a
                JOIN samples s ON a.segment_id=s.segment_id AND a.split=s.split
                WHERE a.split IN ('train','validation')
                  AND a.segment_point_index BETWEEN s.history_start AND s.target_end""",
            [str(sources["alignment.jsonl"])])
        db.execute("""CREATE TEMP TABLE windows AS
            SELECT sample_id, independent_block_id, split,
              (max(CASE WHEN segment_point_index=target_end THEN absolute_epoch_ns END)
                - max(CASE WHEN segment_point_index=history_end THEN absolute_epoch_ns END)) / 1e9 AS followup_s,
              max(gap_ns)/1e9 AS max_gap_s,
              min(gap_ns)/1e9 AS min_gap_s,
              count(*)=max(target_end-history_start+1)
                AND count(DISTINCT segment_point_index)=count(*) AS complete,
              max(history_end-history_start+1) AS history_points
            FROM selected GROUP BY sample_id, independent_block_id, split""")
        snapshot_identity = None
        if snapshot is not None:
            from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
            adapter = FeatureSnapshotAdapter(snapshot, FeatureSelection())
            if adapter.dataset_id != dataset["dataset_id"]:
                raise ValueError("feature snapshot/cohort dataset mismatch")
            paths = [str(adapter.root / item["path"]) for item in adapter.manifest["files"]
                     if item["split"] in {"train", "validation"}]
            if not paths:
                raise ValueError("snapshot has no train/validation files")
            # Only validity and identity metadata; no feature values or held-out files.
            db.execute("""CREATE TEMP TABLE coverage AS
                SELECT file_id, point_index, absolute_epoch_ns, split,
                  dem_surface_status='valid' AS surface_valid,
                  overture_road_status='valid' AS road_valid,
                  hydrorivers_river_status='valid' AS river_valid,
                  worldcover_status='valid' AS worldcover_valid,
                  historical_motion_status='valid' AS history_valid
                FROM read_parquet(?) WHERE split IN ('train','validation')""", [paths])
            duplicate = db.execute("""SELECT count(*) FROM (
                SELECT 1 FROM coverage GROUP BY file_id, point_index, absolute_epoch_ns, split
                HAVING count(*)>1)""").fetchone()[0]
            if duplicate:
                raise ValueError("duplicate feature identities")
            db.execute("""CREATE TEMP TABLE window_coverage AS
                SELECT s.sample_id,
                  bool_and(coalesce(c.surface_valid,false)) AS surface_valid,
                  bool_and(coalesce(c.road_valid,false)) AS road_valid,
                  bool_and(coalesce(c.river_valid,false)) AS river_valid,
                  bool_and(coalesce(c.worldcover_valid,false)) AS worldcover_valid,
                  bool_and(coalesce(c.history_valid,false)) AS history_valid
                FROM selected s LEFT JOIN coverage c ON s.file_id=c.file_id
                  AND s.source_point_index=c.point_index
                  AND s.absolute_epoch_ns=c.absolute_epoch_ns AND s.split=c.split
                WHERE s.segment_point_index>=s.history_end GROUP BY s.sample_id""")
            db.execute("""CREATE TEMP VIEW joint_windows AS
                SELECT w.*, c.surface_valid AND c.road_valid AND c.river_valid
                  AND c.worldcover_valid AND c.history_valid AS joint_valid
                FROM windows w JOIN window_coverage c USING(sample_id)""")
            snapshot_identity = {"snapshot_id": adapter.manifest["snapshot_id"],
                "content_inventory_sha256": adapter.manifest["content_inventory_sha256"],
                "coverage_window": "origin_through_registered_target_end",
                "factor_groups": ["surface", "road", "river", "worldcover"],
                "baseline_context": "history.direction"}
        splits = {}
        for split in ("train", "validation"):
            denominator = db.execute("SELECT count(*), count(DISTINCT independent_block_id) FROM samples WHERE split=?", [split]).fetchone()
            qualified = []
            for horizon in horizons:
                counts = db.execute("""SELECT count(*), count(DISTINCT independent_block_id)
                    FROM windows WHERE split=? AND complete AND min_gap_s>0
                    AND max_gap_s<=? AND followup_s>=? AND history_points>=2""",
                    [split, max_gap_seconds, horizon * 60]).fetchone()
                row = {"horizon_minutes": horizon, "samples": counts[0], "independent_blocks": counts[1]}
                if snapshot is not None:
                    common = db.execute("""SELECT count(*), count(DISTINCT independent_block_id)
                        FROM joint_windows WHERE split=? AND complete AND min_gap_s>0
                        AND max_gap_s<=? AND followup_s>=? AND history_points>=2 AND joint_valid""",
                        [split, max_gap_seconds, horizon * 60]).fetchone()
                    row["common_coverage_samples"], row["common_coverage_blocks"] = common
                qualified.append(row)
            splits[split] = {"denominator_samples": denominator[0], "denominator_blocks": denominator[1], "horizons": qualified}
        result = {"schema_version": "pirc17-window-qualification-v2", "dataset_id": dataset["dataset_id"],
            "snapshot": snapshot_identity,
            "inputs_sha256": hashes, "final_eval_label_prediction_metric_reads": 0,
            "qualified_roles": ["train", "validation"], "max_gap_seconds": max_gap_seconds,
            "origin_mode": "causal_history_prefix_candidate", "minimum_history_points": 2,
            "splits": splits, "limitations": ["Snapshot validity is not a guarantee of coverage at future predicted positions; rollout must handle missing queries explicitly.",
                "Historical final-eval exposure is not reset by this audit.", "No interpolation or target-dependent scoring was performed."]}
        if eligibility_horizon_minutes is not None:
            eligible = db.execute("""SELECT s.*, f.file_id, w.followup_s, w.max_gap_s FROM samples s
                JOIN joint_windows w USING(sample_id,independent_block_id,split)
                JOIN (SELECT sample_id,min(file_id) file_id FROM selected GROUP BY sample_id) f USING(sample_id)
                WHERE w.complete AND w.min_gap_s>0 AND w.max_gap_s<=?
                  AND w.followup_s>=? AND w.history_points>=2 AND w.joint_valid
                ORDER BY s.split,s.sample_id""", [max_gap_seconds,eligibility_horizon_minutes*60]).to_arrow_table().to_pylist()
            result["eligibility"]={"schema_version":"pirc17-development-eligibility-v1",
                "horizon_minutes":eligibility_horizon_minutes,"roles":["train","validation"],"rows":eligible}
        return result
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--eligibility-horizon-minutes", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result=audit_windows(args.release, snapshot=args.snapshot,eligibility_horizon_minutes=args.eligibility_horizon_minutes)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as target:
            json.dump(result,target,indent=2,allow_nan=False)
        print(json.dumps({"output":str(args.output),"splits":result["splits"],
                          "eligible_rows":len(result.get("eligibility",{}).get("rows",[]))},indent=2))
    else:
        print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
