"""Bounded real validation rollout qualification, never final-eval execution.

Preserves each attempted run, including failures, in an exclusive JSONL ledger.
The first sorted validation origins are an engineering pilot, not a score-selected
or representative test population. No prediction-effect conclusions are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .calibration import TIME_WEIGHTS
from .brownian import BROWNIAN_VERSION, BrownianPath
from .checkpoints import restore_dynamics
from .configurations import configuration_encoder, terrain_configurations
from .development import load_development
from .features import CanonicalEncoder, PredictedPositionFeatures
from .metrics import EntropyGrid, score_path
from .pilot_selection import SELECTIONS, select_windows
from .precision import energy_precision
from .qualification import _hash
from .rollout import ROLLOUT_VERSION, rollout

MAP_BACKENDS = ("legacy", "batched", "cached", "multicell")


def resolve_map_backend(name):
    """Load only the explicit backend and bind its raster and geometry sources."""
    if name not in MAP_BACKENDS:
        raise ValueError("unknown map backend")
    from trajectory import online_terrain, linear_materialization
    from map_data import terrain_features
    modules = [online_terrain, linear_materialization, terrain_features]
    query_type = online_terrain.RawMapQuery
    if name in {"batched", "cached", "multicell"}:
        from trajectory import batched_terrain
        modules.append(batched_terrain)
        query_type = batched_terrain.BatchedRawMapQuery
    if name in {"cached", "multicell"}:
        from trajectory import cached_terrain
        modules.append(cached_terrain)
        query_type = cached_terrain.CachedBatchedRawMapQuery
    if name == "multicell":
        from trajectory import multicell_terrain
        modules.append(multicell_terrain)
        query_type = multicell_terrain.MultiCellRawMapQuery
    return query_type, modules


def save_particle_artifact(ledger_path, row, positions, targets, horizons):
    """Exclusive, hash-bound local diagnostic arrays; never append private rows to Git."""
    key = {name: row[name] for name in ("sample_id", "configuration", "seed", "particles", "max_step_seconds")}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    folder = ledger_path.with_suffix(".particles")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (digest+".npz")
    with path.open("xb") as target:
        np.savez_compressed(target, positions_m=positions, target_positions_m=targets, elapsed_seconds=horizons)
    return {"path": path.relative_to(ledger_path.parent).as_posix(), "sha256": _hash(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fit","eligibility","release","snapshot","data-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fit-sha256",required=True)
    parser.add_argument("--eligibility-sha256",required=True)
    parser.add_argument("--configurations",nargs="+",default=["base","all-terrain"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[20260814])
    parser.add_argument("--particles",nargs="+",type=int,default=[32])
    parser.add_argument("--steps",nargs="+",type=float,default=[5.,2.5])
    parser.add_argument("--limit-origins",type=int,default=1)
    parser.add_argument("--selection-policy",choices=tuple(SELECTIONS),default="lexical_origins")
    parser.add_argument("--map-backend",choices=MAP_BACKENDS,default="legacy")
    parser.add_argument("--save-particles",action="store_true")
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a run ledger")
    if args.limit_origins<1 or any(p<2 for p in args.particles) or any(not np.isfinite(s) or s<=0 for s in args.steps):
        raise ValueError("positive origins/steps and at least two particles required")
    if any(len(set(values))!=len(values) for values in (args.configurations,args.seeds,args.particles,args.steps)):
        raise ValueError("duplicate workload settings are ambiguous")
    if _hash(args.fit)!=args.fit_sha256:
        raise ValueError("fit artifact changed")
    fitted=json.loads(args.fit.read_text(encoding="utf-8"))
    configs=terrain_configurations()
    if fitted["schema_version"]!="pirc17-development-fit-v1" or fitted["configurations"]!=configs:
        raise ValueError("fit configuration registry differs")
    models={name:{seed:restore_dynamics(fitted["models"][name][str(seed)]) for seed in args.seeds} for name in args.configurations}
    for name,by_seed in models.items():
        for seed,model in by_seed.items():
            if (model.identity["configuration_identity"]!=configs[name]["sha256"] or
                model.identity["training_identity"]!=fitted["development_identity"]["sha256"] or model.identity["seed"]!=seed):
                raise ValueError("checkpoint mislabeled by configuration, population or seed")
    encoder=CanonicalEncoder.frozen_pirc22(args.snapshot)
    print("Verified frozen encoder",flush=True)
    windows,parents,identity=load_development(args.eligibility,args.eligibility_sha256,args.release,args.data_root,encoder)
    if identity!=fitted["development_identity"]:
        raise ValueError("fit and rollout development populations differ")
    selected=select_windows(windows["validation"],count=args.limit_origins,policy=args.selection_policy)
    RawMapQuery, map_modules = resolve_map_backend(args.map_backend)
    receipts=[args.data_root/name for name in ("registry/terrain_expansion_receipt.json","registry/linear_expansion_receipt.json",
        "runs/pirc18-production-001/terrain-receipt.json","runs/pirc18-production-001/remaining-terrain-receipt.json",
        "runs/pirc18-production-001/linear-receipt.json")]
    maps=RawMapQuery(args.data_root,receipts)
    for parent in parents:
        maps.register_snapshot_parent(parent)
    header={"schema_version":"pirc17-development-rollout-v1","type":"header","purpose":"bounded_validation_engineering_pilot",
        "expected_run_count":len(selected)*len(args.configurations)*len(args.seeds)*len(args.particles)*len(args.steps),
        "fit_sha256":args.fit_sha256,"development_identity":identity["sha256"],"rollout_version":ROLLOUT_VERSION,
        "selection":SELECTIONS[args.selection_policy],"selection_policy":args.selection_policy,
        "selected_independent_block_count":len({w.block_id for w in selected}),
        "sample_ids":[w.sample_id for w in selected],"configurations":args.configurations,"seeds":args.seeds,
        "particle_counts":args.particles,"max_steps_seconds":args.steps,"physical_history_step_seconds":5.,
        "map_backend":args.map_backend,"save_particles":args.save_particles,
        "brownian_driver":BROWNIAN_VERSION,"coupling":"same atomic Gaussian increments across configurations, steps and particle prefixes within each origin/seed",
        "scoring_grid":"actual observed timestamps nearest 60/300/900/1800 within 30s; no interpolation",
        "time_weights":list(TIME_WEIGHTS),"entropy_grid_m":{"minimum":-10000,"maximum":10000,"cell_size":250},
        "failure_policy":"retain every attempted workload; no failed origins removed from denominator",
        "final_eval_label_prediction_metric_reads":0,
        "map_source_sha256":{module.__name__:_hash(Path(module.__file__)) for module in map_modules},
        "source_sha256":{name:_hash(Path(__file__).with_name(name)) for name in (
            "development_rollout.py","rollout.py","brownian.py","dynamics.py","checkpoints.py","development.py","features.py","configurations.py","metrics.py","pilot_selection.py","precision.py")}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    failure_count=0
    completed_count=0
    try:
        with args.output.open("x",encoding="utf-8") as target:
            def emit(row):
                target.write(json.dumps(row,allow_nan=False)+"\n")
                target.flush()
            emit(header)
            for window in selected:
                drivers={seed:BrownianPath(window.horizon_seconds,args.steps,history_step_seconds=5.,
                    particles=max(args.particles),seed=seed,stream_id=window.sample_id) for seed in args.seeds}
                for name,by_seed in models.items():
                    selected_encoder=configuration_encoder(encoder,name)
                    # No terrain means no map lookup; origin history is unchanged.
                    query=(lambda positions:[{} for _ in positions]) if name=="base" else maps
                    provider=PredictedPositionFeatures(selected_encoder,query,window.frame)
                    for seed,model in by_seed.items():
                        for particles in args.particles:
                            for step in args.steps:
                                row={"type":"run","sample_id":window.sample_id,"independent_block_id":window.block_id,
                                    "configuration":name,"seed":seed,"particles":particles,"max_step_seconds":step,
                                    "brownian_identity":drivers[seed].identity,
                                    "actual_horizons_seconds":window.horizon_seconds.tolist(),"status":"failure"}
                                start=time.perf_counter()
                                try:
                                    prediction=rollout(window.origin,window.horizon_seconds,particles=particles,seed=seed,
                                        max_step_seconds=step,history_step_seconds=5.,base_drift=model.base_drift,
                                        diffusion=model.diffusion,terrain=provider,conditioner=model.correction,
                                        brownian_increments=drivers[seed])
                                    row["rollout_wall_seconds_including_lazy_map_initialization"]=time.perf_counter()-start
                                    row["invalid_feature_rows"]=prediction.invalid_feature_rows
                                    row["feature_query_rows"]=prediction.feature_query_rows
                                    row["scores"]=score_path(prediction.positions_m,window.target_positions_m,window.horizon_seconds,
                                        time_weights=TIME_WEIGHTS,entropy_grid=EntropyGrid(tuple(np.arange(-10000,10001,250)),tuple(np.arange(-10000,10001,250))))
                                    if particles >= 3:
                                        row["particle_precision"] = energy_precision(prediction.positions_m,window.target_positions_m,TIME_WEIGHTS)
                                    if args.save_particles:
                                        row["particle_artifact"] = save_particle_artifact(args.output, row,
                                            prediction.positions_m,window.target_positions_m,window.horizon_seconds)
                                    row["status"]="success"
                                except Exception as exc:
                                    failure_count+=1
                                    row["elapsed_seconds_before_failure"]=time.perf_counter()-start
                                    row["error_type"]=type(exc).__name__
                                    row["error_message"]=str(exc)[:300]
                                emit(row)
                                completed_count+=1
                                print(json.dumps({k:v for k,v in row.items() if k!="scores"}),flush=True)
            emit({"type":"completion","attempted_run_count":completed_count,"failure_count":failure_count,
                "success_count":completed_count-failure_count,"maps":maps.identity,
                "not_qualified_by_this_pilot":["representative paired power","final forecast efficacy","all NEX326 methods","full experiment budget"]})
    finally:
        maps.close()
    if failure_count:
        raise SystemExit(1)


if __name__=="__main__":
    main()
