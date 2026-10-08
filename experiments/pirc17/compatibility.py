"""Bounded real train/validation handoff check, not an experiment or score.

Run with the DSDE source root on PYTHONPATH. Chooses the first registered sample
in each permitted role; only its last three history points are read as positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from .features import CanonicalEncoder, LocalFrame, PredictedPositionFeatures
from .origins import causal_prefix
from .qualification import _hash
from .rollout import PredictedState, rollout


def check(release, snapshot, data_root, *, common_origin=False):
    from trajectory.online_terrain import RawMapQuery
    encoder = CanonicalEncoder.frozen_pirc22(snapshot)
    dataset = json.loads((release / "dataset.json").read_text(encoding="utf-8"))
    cohort = json.loads((release / "cohort.json").read_text(encoding="utf-8"))
    if dataset["dataset_id"] != encoder.adapter.dataset_id or cohort["cohort_id"] != dataset["dataset_id"]:
        raise ValueError("dataset identities differ")
    hashes = {}
    for name in ("alignment.jsonl", "condition_file_manifest.jsonl", "cohort.json"):
        hashes[name] = _hash(release / name)
        if hashes[name] != dataset["artifacts"][name]["sha256"]:
            raise ValueError(f"release artifact changed: {name}")
    hashes["samples.jsonl"] = _hash(release / "samples.jsonl")
    if hashes["samples.jsonl"] != cohort["sample_manifest_sha256"]:
        raise ValueError("sample manifest changed")
    selected = {}
    candidates = []
    with (release / "samples.jsonl").open(encoding="utf-8") as source:
        for line in source:
            sample = json.loads(line)
            if sample["split"] in {"train", "validation"} and sample["history_end"] - sample["history_start"] >= 2:
                selected.setdefault(sample["split"], sample)
                candidates.append(sample)
    if common_origin:
        # Eligibility uses status metadata, not final-eval values or model scores.
        eligibility = duckdb.connect()
        try:
            eligibility.execute("SET threads=4")
            eligibility.execute("SET memory_limit='1GB'")
            eligibility.execute("CREATE TEMP TABLE candidates(ordinal BIGINT, segment_id VARCHAR, split VARCHAR, origin_index BIGINT)")
            eligibility.executemany("INSERT INTO candidates VALUES(?,?,?,?)", [(i,s["segment_id"],s["split"],s["history_end"]) for i,s in enumerate(candidates)])
            paths = [str(encoder.adapter.root / f["path"]) for f in encoder.adapter.manifest["files"] if f["split"] in {"train","validation"}]
            indexes = eligibility.execute("""WITH numbered AS (
                SELECT segment_id,split,row_number() OVER (PARTITION BY segment_id ORDER BY point_index)-1 idx,
                    dem_surface_status='valid' AND overture_road_status='valid'
                    AND hydrorivers_river_status='valid' AND worldcover_status='valid'
                    AND historical_motion_status='valid' AS valid
                FROM read_parquet(?) WHERE split IN ('train','validation'))
                SELECT min(c.ordinal) FROM candidates c JOIN numbered n
                ON c.segment_id=n.segment_id AND c.split=n.split AND c.origin_index=n.idx
                WHERE n.valid GROUP BY c.split""", [paths]).fetchall()
            selected = {candidates[i]["split"]: candidates[i] for (i,) in indexes}
        finally:
            eligibility.close()
    if set(selected) != {"train", "validation"}:
        raise ValueError("both development roles required")
    with (release / "condition_file_manifest.jsonl").open(encoding="utf-8") as source:
        conditions = {r["file_id"]: r for r in map(json.loads, source)}
    receipts = [data_root / name for name in (
        "registry/terrain_expansion_receipt.json", "registry/linear_expansion_receipt.json",
        "runs/pirc18-production-001/terrain-receipt.json",
        "runs/pirc18-production-001/remaining-terrain-receipt.json",
        "runs/pirc18-production-001/linear-receipt.json")]
    maps = RawMapQuery(data_root, receipts)
    db = duckdb.connect()
    reports = []
    try:
        db.execute("SET threads=4")
        db.execute("SET memory_limit='1GB'")
        metadata_paths = [str(encoder.adapter.root / f["path"]) for f in encoder.adapter.manifest["files"] if f["split"] in {"train","validation"}]
        for column in ("dem_surface_parent_asset_id", "worldcover_parent_asset_id"):
            parents = db.execute(f"SELECT DISTINCT {column} FROM read_parquet(?) WHERE split IN ('train','validation') AND {column} IS NOT NULL", [metadata_paths]).fetchall()
            for (parent,) in parents:
                if parent.startswith("registered-file:"):
                    maps.register_snapshot_parent(parent)
        db.execute("CREATE TEMP TABLE wanted(segment_id VARCHAR, split VARCHAR, first_index BIGINT, last_index BIGINT)")
        db.executemany("INSERT INTO wanted VALUES(?,?,?,?)", [(s["segment_id"], role, s["history_end"]-2, s["history_end"]) for role,s in selected.items()])
        table = db.execute("""SELECT a.* FROM read_json(?, columns={segment_id:'VARCHAR',split:'VARCHAR',
            segment_point_index:'BIGINT',source_point_index:'BIGINT',absolute_epoch_ns:'BIGINT',file_id:'VARCHAR'}) a
            JOIN wanted w ON a.segment_id=w.segment_id AND a.split=w.split
            WHERE a.segment_point_index BETWEEN w.first_index AND w.last_index
            ORDER BY a.split,a.segment_point_index""", [str(release / "alignment.jsonl")]).to_arrow_table()
        for role, sample in selected.items():
            alignment = [r for r in table.to_pylist() if r["split"] == role]
            if len(alignment) != 3 or any(r["file_id"] != sample["file_id"] for r in alignment):
                raise ValueError("history alignment mismatch")
            entry = conditions[sample["file_id"]]
            condition_path = (data_root / "cond_slices" / entry["relative_path"]).resolve()
            if not condition_path.is_relative_to((data_root / "cond_slices").resolve()):
                raise ValueError("unsafe condition locator")
            if _hash(condition_path) != entry["sha256"]:
                raise ValueError("condition source changed")
            points = pq.read_table(condition_path, columns=["lon", "lat"]).take([r["source_point_index"] for r in alignment]).to_pylist()
            lonlat = np.array([[p["lon"], p["lat"]] for p in points])
            frame = LocalFrame(*lonlat[-1])
            # Subtract integer ns before casting, avoiding epoch float cancellation.
            epochs = np.array([r["absolute_epoch_ns"] for r in alignment], dtype=np.int64)
            times = (epochs - epochs[-1]).astype(float) / 1e9
            origin = causal_prefix(frame.from_lonlat(lonlat), times)
            state = PredictedState(0., origin.position_m[None], origin.velocity_mps[None],
                                   origin.history_positions_m[None], origin.history_times_seconds)
            provider = PredictedPositionFeatures(encoder, maps, frame)
            online = provider(state)
            offline = encoder.adapter._load(role, final_eval_unlock=None,
                file_ids=[sample["file_id"]], segment_ids=[sample["segment_id"]])
            rows = [r for r in offline.to_pylist() if r["absolute_epoch_ns"] == int(epochs[-1])]
            if len(rows) != 1:
                raise ValueError("ambiguous frozen origin")
            expected = encoder.encode(rows)
            both = online.valid & expected.valid
            differences = {name: float(abs(online.values[0,i]-expected.values[0,i]))
                for i,name in enumerate(encoder.columns) if both[0,i] and not np.isclose(online.values[0,i],expected.values[0,i],rtol=1e-5,atol=1e-6)}
            # Wiring check only: inertial base, zero correction/noise. Not a fitted model.
            forecast = rollout(origin, [1., 2.], particles=1, seed=20260926, max_step_seconds=1.,
                base_drift=lambda s:s.velocities_mps, diffusion=lambda s:np.zeros((1,2,2)),
                terrain=provider, conditioner=lambda s,m:np.zeros((1,2)))
            reports.append({"role":role,"sample_id":sample["sample_id"],"segment_id":sample["segment_id"],
                "condition_sha256":entry["sha256"],"prefix_points":3,"model_input_dim":online.model_matrix(1).shape[1],
                "origin_valid_columns":int(online.valid.sum()),"frozen_valid_columns":int(expected.valid.sum()),
                "mask_equal":bool(np.array_equal(online.valid,expected.valid)),"value_mismatches":differences,
                "rollout_query_rows":forecast.feature_query_rows,"rollout_invalid_feature_rows":forecast.invalid_feature_rows})
        return {"schema_version":"pirc17-online-compatibility-v1", "purpose":"bounded_real_prefix_wiring_check_not_model_evaluation",
            "dataset_id":dataset["dataset_id"], "snapshot_id":encoder.adapter.manifest["snapshot_id"],
            "selection":"first_common_origin_per_role" if common_origin else "first_origin_per_role",
            "release_sha256":hashes,"maps":maps.identity,"checks":reports,
            "final_eval_label_prediction_metric_reads":0,
            "passed":all(r["mask_equal"] and not r["value_mismatches"] for r in reports)}
    finally:
        maps.close()
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "snapshot", "data-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--common-origin", action="store_true")
    args = parser.parse_args()
    result = check(args.release, args.snapshot, args.data_root, common_origin=args.common_origin)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
