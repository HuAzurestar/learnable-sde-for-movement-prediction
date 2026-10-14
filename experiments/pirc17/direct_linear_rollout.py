"""Bounded closed-loop prediction for a distinct, unaccepted training candidate.

The original Adam runner, models and ledgers stay unchanged. This entry requires
the complete new fit and its attempt ledger, saves every successful particle
array, retains failed/unattempted work, and never authorizes final evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import psutil
import torch

from .brownian import BROWNIAN_VERSION, BrownianPath
from .calibration import TIME_WEIGHTS
from .configurations import configuration_encoder, terrain_configurations
from .development import load_development
from .development_rollout import MAP_BACKENDS, resolve_map_backend, save_particle_artifact
from .direct_linear import MODEL_VERSION, configuration_width
from .direct_linear_evidence import add_evidence_arguments, load_candidate
from .direct_linear_fit import FIT_VERSION, MINIMUM_FREE_BYTES
from .features import CanonicalEncoder, PredictedPositionFeatures
from .inference import SEEDS
from .metrics import EntropyGrid, score_path
from .pilot_selection import SELECTIONS, select_windows
from .precision import energy_precision
from .qualification import _hash
from .rollout import ROLLOUT_VERSION, rollout

VERSION = "pirc17-direct-linear-rollout-v1"
PURPOSE = "bounded_validation_training_candidate_pilot"
SCORING_GRID = "actual observed timestamps nearest 60/300/900/1800 within 30s; no interpolation"
ENTROPY_GRID = {"minimum":-10000, "maximum":10000, "cell_size":250}
RECEIPTS = ("registry/terrain_expansion_receipt.json", "registry/linear_expansion_receipt.json",
    "runs/pirc18-production-001/terrain-receipt.json", "runs/pirc18-production-001/remaining-terrain-receipt.json",
    "runs/pirc18-production-001/linear-receipt.json")


def source_hashes():
    names = ("direct_linear_rollout.py", "direct_linear_evidence.py", "direct_linear.py", "direct_linear_fit.py",
        "development_rollout.py", "rollout.py", "brownian.py", "dynamics.py", "checkpoints.py", "development.py",
        "features.py", "configurations.py", "metrics.py", "pilot_selection.py", "precision.py", "origins.py",
        "calibration.py", "inference.py", "qualification.py")
    return {name:_hash(Path(__file__).with_name(name)) for name in names}


def validate_workload(configurations, seeds, particles, steps, limit_origins, selection_policy, map_backend, wall_seconds):
    axes = (configurations, seeds, particles, steps)
    if any(not values or len(set(values)) != len(values) for values in axes):
        raise ValueError("nonempty unique workload axes required")
    if (any(name not in terrain_configurations() for name in configurations)
            or any(type(seed) is not int or seed not in SEEDS for seed in seeds)
            or any(type(p) is not int or p < 3 for p in particles)
            or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not np.isfinite(s) or s <= 0 for s in steps)
            or type(limit_origins) is not int or limit_origins < 1
            or selection_policy not in SELECTIONS or map_backend not in MAP_BACKENDS
            or isinstance(wall_seconds, bool) or not np.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("invalid bounded candidate workload")


def validate_population(bundle, windows, parents, identity):
    if (identity != bundle["development_identity"] or parents != bundle["snapshot_parent_assets"]
            or set(windows) != {"train", "validation"}):
        raise ValueError("fit and rollout development identities or parents differ")
    seen, blocks = set(), {}
    for role, group in windows.items():
        ids = [w.sample_id for w in group]
        blocks[role] = {w.block_id for w in group}
        if (ids != identity["sample_ids"][role]
                or any(w.role != role or not w.block_id for w in group) or len(set(ids)) != len(ids)
                or seen.intersection(ids) or len(group) != bundle["population"]["samples"][role]
                or len(blocks[role]) != bundle["population"]["independent_blocks"][role]):
            raise ValueError("fit and rollout roles, samples or independent blocks differ")
        seen.update(ids)
    if blocks["train"] & blocks["validation"]:
        raise ValueError("train and validation independent blocks overlap")


def run(*, fit, fit_sha256, fit_ledger, fit_ledger_sha256, training_policy_sha256,
        eligibility, eligibility_sha256, release, snapshot, data_root, output,
        configurations, seeds, particles, steps, limit_origins, wall_seconds,
        selection_policy="lexical_independent_blocks", map_backend="multicell", available_memory=None, progress=None):
    validate_workload(configurations, seeds, particles, steps, limit_origins, selection_policy, map_backend, wall_seconds)
    output = Path(output).resolve()
    if output.exists() or output.with_suffix(".particles").exists():
        raise FileExistsError("refusing to overwrite candidate ledger or particle evidence")
    if output.suffix != ".jsonl":
        raise ValueError("candidate rollout ledger must use .jsonl")
    evidence = dict(fit=fit, fit_sha256=fit_sha256, fit_ledger=fit_ledger,
                    fit_ledger_sha256=fit_ledger_sha256, training_policy_sha256=training_policy_sha256)
    binding = {k:v for k,v in evidence.items() if k.endswith("sha256")}
    expected = limit_origins*len(configurations)*len(seeds)*len(particles)*len(steps)
    available_memory = available_memory or (lambda:psutil.virtual_memory().available)
    started = time.perf_counter()
    attempted = successful = failures = terminal_errors = 0
    stopped, maps, maps_identity = False, None, None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        def emit(row):
            target.write(json.dumps(row, allow_nan=False)+"\n")
            target.flush()

        def guard():
            if time.perf_counter()-started >= wall_seconds:
                raise TimeoutError("registered candidate wall-time budget exhausted")
            if available_memory() < MINIMUM_FREE_BYTES:
                raise MemoryError("candidate rollout requires at least 2 GiB available RAM")

        def terminal(exc):
            nonlocal terminal_errors, stopped
            terminal_errors += 1
            stopped = stopped or isinstance(exc, (MemoryError, TimeoutError))
            emit({"type":"failure", "error_type":type(exc).__name__, "error_message":str(exc)[:300]})

        emit({"type":"initialization", "schema_version":VERSION, "purpose":PURPOSE, **binding,
            "started_at":datetime.now(timezone.utc).isoformat(), "expected_run_count":expected,
            "configurations":list(configurations), "seeds":list(seeds), "particle_counts":list(particles),
            "max_steps_seconds":list(steps), "limit_origins":limit_origins, "selection_policy":selection_policy,
            "map_backend":map_backend, "eligibility_sha256":eligibility_sha256,
            "wall_seconds":wall_seconds, "minimum_free_bytes":MINIMUM_FREE_BYTES,
            "runtime":{"python":platform.python_version(), "numpy":np.__version__, "torch":torch.__version__,
                       "torch_intraop_threads":torch.get_num_threads(), "torch_interop_threads":torch.get_num_interop_threads()},
            "resource_policy":"cooperative checks before preflight, streams, every run and every feature query; outer process cap remains separate",
            "certified":False, "final_eval_label_prediction_metric_reads":0})
        sources = None
        try:
            guard()
            sources = source_hashes()
            bundle, models, fitted_eligibility = load_candidate(**evidence)
            if fitted_eligibility != eligibility_sha256:
                raise ValueError("rollout eligibility differs from original candidate fit ledger")
            encoder = CanonicalEncoder.frozen_pirc22(Path(snapshot))
            windows, parents, identity = load_development(Path(eligibility), eligibility_sha256,
                                                         Path(release), Path(data_root), encoder)
            validate_population(bundle, windows, parents, identity)
            selected = select_windows(windows["validation"], count=limit_origins, policy=selection_policy)
            encoders = {name:configuration_encoder(encoder, name) for name in configurations}
            if any(len(e.columns)*2 != configuration_width(name) for name,e in encoders.items()):
                raise ValueError("online encoder differs from the frozen candidate configuration")
            query_type, map_modules = resolve_map_backend(map_backend)
            map_sources = {m.__name__:_hash(Path(m.__file__)) for m in map_modules}
            emit({"type":"header", "schema_version":VERSION, "purpose":PURPOSE, **binding,
                "fit_version":FIT_VERSION, "model_version":MODEL_VERSION,
                "expected_run_count":expected, "development_identity":identity["sha256"],
                "model_identities":{name:{str(seed):model.identity["sha256"] for seed,model in group.items()}
                                    for name,group in models.items()},
                "sample_ids":[w.sample_id for w in selected], "selected_independent_block_count":len({w.block_id for w in selected}),
                "selection":SELECTIONS[selection_policy], "selection_policy":selection_policy,
                "configurations":list(configurations), "seeds":list(seeds), "particle_counts":list(particles),
                "max_steps_seconds":list(steps), "physical_history_step_seconds":5., "rollout_version":ROLLOUT_VERSION,
                "map_backend":map_backend, "save_particles":True, "brownian_driver":BROWNIAN_VERSION,
                "coupling":"same atomic increments across configurations, steps and particle prefixes within each origin/seed",
                "scoring_grid":SCORING_GRID, "time_weights":list(TIME_WEIGHTS), "entropy_grid_m":ENTROPY_GRID,
                "source_sha256":sources, "map_source_sha256":map_sources,
                "input_validation_seconds":time.perf_counter()-started,
                "failure_policy":"retain attempted failures and the complete unattempted denominator",
                "final_eval_label_prediction_metric_reads":0, "formal_training_accepted":False, "certified":False})
            if progress:
                progress({"phase":"inputs_verified", "expected_runs":expected})
            guard()
            maps = query_type(Path(data_root), [Path(data_root)/p for p in RECEIPTS])
            for parent in parents:
                maps.register_snapshot_parent(parent)
            grid = EntropyGrid(tuple(np.arange(-10000,10001,250)), tuple(np.arange(-10000,10001,250)))
            # One stream resident at a time; identical origin/seed/grid/prefix
            # semantics to the old runner, independent of configuration order.
            for window in selected:
                for seed in seeds:
                    guard()
                    driver = BrownianPath(window.horizon_seconds, steps, history_step_seconds=5.,
                                          particles=max(particles), seed=seed, stream_id=window.sample_id)
                    for name in configurations:
                        model = models[name][seed]
                        query = (lambda positions:[{} for _ in positions]) if name == "base" else maps
                        provider = PredictedPositionFeatures(encoders[name], query, window.frame)

                        def guarded_features(state):
                            guard()
                            return provider(state)

                        for count in particles:
                            for step in steps:
                                guard()
                                attempted += 1
                                row = {"type":"run", "sample_id":window.sample_id, "independent_block_id":window.block_id,
                                    "configuration":name, "seed":seed, "particles":count, "max_step_seconds":step,
                                    "model_identity_sha256":model.identity["sha256"], "brownian_identity":driver.identity,
                                    "actual_horizons_seconds":window.horizon_seconds.tolist(), "status":"failure"}
                                begin = time.perf_counter()
                                try:
                                    prediction = rollout(window.origin, window.horizon_seconds, particles=count, seed=seed,
                                        max_step_seconds=step, history_step_seconds=5., base_drift=model.base_drift,
                                        diffusion=model.diffusion, terrain=guarded_features, conditioner=model.correction,
                                        brownian_increments=driver)
                                    row["rollout_wall_seconds_including_lazy_map_initialization"] = time.perf_counter()-begin
                                    row.update(invalid_feature_rows=prediction.invalid_feature_rows,
                                               feature_query_rows=prediction.feature_query_rows)
                                    guard()
                                    row["scores"] = score_path(prediction.positions_m, window.target_positions_m,
                                        window.horizon_seconds, time_weights=TIME_WEIGHTS, entropy_grid=grid)
                                    row["particle_precision"] = energy_precision(prediction.positions_m, window.target_positions_m, TIME_WEIGHTS)
                                    row["particle_artifact"] = save_particle_artifact(output, row, prediction.positions_m,
                                        window.target_positions_m, window.horizon_seconds)
                                    guard()
                                    row["status"] = "success"
                                    successful += 1
                                except Exception as exc:
                                    failures += 1
                                    row.update(error_type=type(exc).__name__, error_message=str(exc)[:300],
                                               elapsed_seconds_before_failure=time.perf_counter()-begin)
                                    emit(row)
                                    if isinstance(exc, (MemoryError, TimeoutError)):
                                        raise
                                else:
                                    emit(row)
                                if progress:
                                    progress({"phase":"run", "attempted":attempted, "expected":expected,
                                              "configuration":name, "seed":seed, "status":row["status"]})
                    del driver
            if {m.__name__:_hash(Path(m.__file__)) for m in map_modules} != map_sources:
                raise ValueError("map source files changed during execution")
        except Exception as exc:
            terminal(exc)
        finally:
            if maps is not None:
                try:
                    maps_identity = maps.identity
                except Exception as exc:
                    terminal(exc)
                try:
                    maps.close()
                except Exception as exc:
                    terminal(exc)
            if sources is not None:
                try:
                    if source_hashes() != sources:
                        raise ValueError("candidate source files changed during execution")
                except Exception as exc:
                    terminal(exc)
        completion = {"type":"completion", "status":"complete" if successful == expected and not terminal_errors else "failed",
            "expected_run_count":expected, "attempted_run_count":attempted, "success_count":successful,
            "failure_count":failures, "unattempted_run_count":expected-attempted,
            "terminal_error_count":terminal_errors, "resource_stopped":stopped, "maps":maps_identity,
            "elapsed_seconds":time.perf_counter()-started, "certified":False, "formal_training_accepted":False,
            "not_qualified_by_this_pilot":["formal training", "representative convergence or power", "final forecast efficacy",
                                          "all NEX326 methods", "full experiment budget"]}
        emit(completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_arguments(parser)
    for name in ("eligibility", "release", "snapshot", "data-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--eligibility-sha256", required=True)
    parser.add_argument("--configurations", nargs="+", default=list(terrain_configurations()))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--particles", nargs="+", type=int, required=True)
    parser.add_argument("--steps", nargs="+", type=float, required=True)
    parser.add_argument("--limit-origins", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--selection-policy", choices=tuple(SELECTIONS), default="lexical_independent_blocks")
    parser.add_argument("--map-backend", choices=MAP_BACKENDS, default="multicell")
    result = run(**vars(parser.parse_args()), progress=lambda row:print(json.dumps(row), flush=True))
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
