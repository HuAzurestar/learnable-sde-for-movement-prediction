"""Real validation-only measurement smoke; no scientific model comparison.

Uses a specified existing validation sample, its three observed history points,
and the first three actual target timestamps. No interpolation, no final eval.
The timed workload is explicitly inertial drift + full map queries + zero
conditioner/noise, NOT a trained production-model performance result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .features import CanonicalEncoder, LocalFrame, PredictedPositionFeatures
from .metrics import EntropyGrid, score_path
from .origins import causal_prefix
from .qualification import _hash
from .rollout import inertial, rollout
from .runtime import benchmark_cpu


def check(release, snapshot, data_root, sample_id):
    from trajectory.online_terrain import RawMapQuery
    encoder = CanonicalEncoder.frozen_pirc22(snapshot)
    cohort = json.loads((release/"cohort.json").read_text(encoding="utf-8"))
    dataset = json.loads((release/"dataset.json").read_text(encoding="utf-8"))
    if cohort["cohort_id"] != encoder.adapter.dataset_id or dataset["dataset_id"] != cohort["cohort_id"]:
        raise ValueError("dataset identity mismatch")
    if _hash(release/"samples.jsonl") != cohort["sample_manifest_sha256"]:
        raise ValueError("sample manifest changed")
    for name in ("cohort.json","condition_file_manifest.jsonl"):
        if _hash(release/name) != dataset["artifacts"][name]["sha256"]:
            raise ValueError("release artifact changed")
    with (release/"samples.jsonl").open(encoding="utf-8") as source:
        sample = next((s for s in map(json.loads,source) if s["sample_id"]==sample_id),None)
    if sample is None or sample["split"] != "validation":
        raise ValueError("measurement check admits only a specified validation sample")
    if sample["history_end"]-sample["history_start"]<2 or sample["target_end"]-sample["target_start"]<2:
        raise ValueError("sample needs three real history and target observations")
    entry = next(f for f in encoder.adapter.manifest["files"] if f["file_id"]==sample["file_id"] and f["split"]=="validation")
    metadata = pq.read_table(encoder.adapter.root/entry["path"], columns=["segment_id","point_index","absolute_epoch_ns",
        "dem_surface_parent_asset_id","worldcover_parent_asset_id"])
    rows = sorted((r for r in metadata.to_pylist() if r["segment_id"]==sample["segment_id"]),key=lambda r:r["point_index"])
    chosen = rows[sample["history_end"]-2:sample["history_end"]+4]
    if len(chosen)!=6 or sample["target_start"]!=sample["history_end"]+1:
        raise ValueError("registered sample boundary mismatch")
    with (release/"condition_file_manifest.jsonl").open(encoding="utf-8") as source:
        condition = next(r for r in map(json.loads,source) if r["file_id"]==sample["file_id"])
    path = (data_root/"cond_slices"/condition["relative_path"]).resolve()
    if not path.is_relative_to((data_root/"cond_slices").resolve()) or _hash(path)!=condition["sha256"]:
        raise ValueError("condition locator/hash mismatch")
    points = pq.read_table(path,columns=["lon","lat"]).take([r["point_index"] for r in chosen]).to_pylist()
    lonlat = np.array([[p["lon"],p["lat"]] for p in points])
    epochs = np.array([r["absolute_epoch_ns"] for r in chosen],dtype=np.int64)
    times = (epochs-epochs[2]).astype(float)/1e9
    if np.any(np.diff(times)<=0) or np.any(np.diff(times)>60):
        raise ValueError("unsupported exact-time gap")
    frame = LocalFrame(*lonlat[2])
    xy = frame.from_lonlat(lonlat)
    origin = causal_prefix(xy[:3],times[:3])
    horizons = times[3:]
    targets = xy[3:]  # Only the offline scorer receives these target positions.
    receipts = [data_root/name for name in ("registry/terrain_expansion_receipt.json","registry/linear_expansion_receipt.json",
        "runs/pirc18-production-001/terrain-receipt.json","runs/pirc18-production-001/remaining-terrain-receipt.json",
        "runs/pirc18-production-001/linear-receipt.json")]
    resources = []
    forecasts = []
    def factory():
        maps = RawMapQuery(data_root,receipts)
        for r in chosen[:3]:
            for name in ("dem_surface_parent_asset_id","worldcover_parent_asset_id"):
                parent = r[name]
                if parent and parent.startswith("registered-file:"):
                    maps.register_snapshot_parent(parent)
        resources.append(maps)
        provider = PredictedPositionFeatures(encoder,maps,frame)
        def execute(stages):
            def terrain(state):
                with stages.span("terrain_io_and_query"):
                    return provider(state)
            with stages.span("rollout"):
                result = rollout(origin,horizons,particles=16,seed=20260926,max_step_seconds=1,history_step_seconds=5.,
                    base_drift=lambda s:np.broadcast_to(origin.velocity_mps,(16,2)),diffusion=lambda s:np.zeros((16,2,2)),
                    terrain=terrain,conditioner=lambda s,m:np.zeros((16,2)))
            forecasts.append(result)
        execute.close = maps.close
        return execute
    try:
        timing = benchmark_cpu(factory,repetitions=2,horizon_seconds=float(horizons[-1]),particles=16,
                               step_seconds=1.,history_step_seconds=5.,batch_size=1,precision="float64")
        if not forecasts:
            raise ValueError("all measurement forecasts failed")
        forecast = forecasts[-1]
        reference = inertial(origin,horizons,particles=16,seed=20260926)
        consistent = bool(np.allclose(forecast.positions_m,reference.positions_m,atol=1e-7,rtol=1e-7))
        scores = score_path(forecast.positions_m,targets,horizons,time_weights=np.full(3,1/3),
                            entropy_grid=EntropyGrid(tuple(np.arange(-2000,2001,100)),tuple(np.arange(-2000,2001,100))))
        return {"schema_version":"pirc17-measurement-check-v1","purpose":"real_validation_inertial_map_wiring_not_model_comparison",
            "sample_id":sample_id,"dataset_id":dataset["dataset_id"],"condition_sha256":condition["sha256"],
            "actual_horizons_seconds":horizons.tolist(),"truth_interpolation":False,
            "final_eval_label_prediction_metric_reads":0,"inertial_consistency":consistent,"scores":scores,"timing":timing,
            "passed":consistent and all(timing[m]["summary"]["failure_count"]==0 for m in ("cold","warm"))}
    finally:
        for resource in resources:
            resource.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("release","snapshot","data-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--sample-id",required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=check(args.release,args.snapshot,args.data_root,args.sample_id)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as output:
            json.dump(result,output,indent=2,allow_nan=False)
        print(json.dumps({"output":str(args.output),"passed":result["passed"],
            "inertial_consistency":result["inertial_consistency"],"actual_horizons_seconds":result["actual_horizons_seconds"],
            "energy_score_m":result["scores"]["time_weighted_energy_score_m"],
            "timing":{m:result["timing"][m]["summary"] for m in ("cold","warm")}},indent=2))
    else:
        print(json.dumps(result,indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
