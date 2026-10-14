"""One registered method cost batch, not numerical or scientific qualification.

Five deterministic fits, five forecasts, one validation origin/seed. Rehydrate
the already audited development population for fitting; never rerun its audit.
Per-phase limits are cooperative, and an independent owned Windows Job enforces
the hard total deadline. No retry, parameter escalation or final-eval access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from .calibration import NOMINAL_SECONDS, TIME_WEIGHTS, observed_indices
from .features import LocalFrame
from .method_development import PreparationBudget, _hash, load_method_development
from .method_inputs import positions_in_scoring_frame
from .method_rollout import forecast_method
from .method_training import TRAINING_FIELDS, fit_development_method, required_slot
from .metrics import EntropyGrid, score_path
from .physical_memory import available_physical_bytes
from .seed_resume_session import require_owned_job, run_owned

VERSION = "pirc17-method-cost-probe-v1"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/pirc17/dev10/method-cost-p512-h5-v1"
FIT_SLOTS = ["arm-01/full", "arm-04/gmm_kernel", "arm-05/explicit_decomp",
             "arm-09/mixed", "arm-14/reptile"]
FORECASTS = [
    {"slot_id": "arm-01/full", "fit_slot_id": "arm-01/full", "pair_id": VERSION},
    {"slot_id": "arm-04/gmm_kernel", "fit_slot_id": "arm-04/gmm_kernel", "pair_id": None},
    {"slot_id": "arm-05/explicit_decomp", "fit_slot_id": "arm-05/explicit_decomp", "pair_id": None},
    {"slot_id": "arm-21/mc", "fit_slot_id": "arm-01/full", "pair_id": None},
    {"slot_id": "arm-21/crn", "fit_slot_id": "arm-01/full", "pair_id": VERSION},
]
CONSTANTS = {
    "schema_version": VERSION, "purpose": "bounded_method_fit_forecast_and_common_score_cost_only",
    "fit_slots": FIT_SLOTS, "forecasts": FORECASTS,
    "expected_fit_count": 5, "expected_forecast_count": 5,
    "particles": 512, "max_step_seconds": 5.0, "seed": 20260814,
    "nominal_seconds": list(NOMINAL_SECONDS), "time_weights": list(TIME_WEIGHTS),
    "selection": "same lexical-first eligible validation sample as the closed LIO cost probe",
    "sample_id": "00d6711e02d7224c4f68497b0b751391bab28a2844cf2bda1e8d7e57762a1a8b",
    "origin_count": 1, "max_steps": 400, "max_particle_steps": 204800,
    "max_transitions": 40000, "expected_transitions": {"train": 26009, "adapt": 5762, "validation": 6149},
    "preparation": {"max_samples": 485, "max_total_points": 1100000,
        "max_file_bytes": 33554432, "max_file_rows": 250000, "wall_seconds": 180},
    "phase_cooperative_caps_seconds": {"load": 180, "fit": 90, "forecast": 30, "score": 10},
    "inner_wall_seconds": 600, "outer_wall_seconds": 660, "minimum_free_bytes": 2147483648,
    "preparation_empirical_cap_seconds": 1800,
    "prior_preparation_empirical_seconds": 327.5257883,
    "source_trajectory": "private/source-trajectories/unified_full_leg.parquet",
    "locations_source": {"path": "artifacts/pirc17/dev10/native-primary-p1024-missing-v1.extension/root.json",
        "sha256": "381c000b3ba9919d08325968255a4708a001d4a2a550940765428d0a37ecbc7d"},
    "prepared_evidence": {"path": "artifacts/pirc17/dev10/method-development-preparation-v1.json",
        "sha256": "eec69a9d9ee502e6f5f574a1b7ebc9ea0ec561c7789bf23b1e3b142aa9e3201e"},
    "expected_input_sha256": "3b17325c864ff1db5c1800533aa227f58a4a0a4e1e957e1430a773019d02c1d3",
    "fit_reuse_scope": "within this cost probe only: identical full training components for FP/MC/CRN",
    "formal_fit_reuse_authorized": False, "formal_training_accepted": False,
    "cost_origin_in_calibration_population": True, "accuracy_claim_authorized": False,
    "numerically_qualified": False, "final_eval_authorized": False, "automatic_retry": False,
}
SOURCE_FILES = (
    "data/pirc20.py", "experiments/nex326/cohort.py", "experiments/nex326/model.py",
    "experiments/nex326/pirc20_adapter.py", "experiments/nex326/specification.py",
    "experiments/nex326/experiment.json", "experiments/nex326/pirc19_scope_policy.json",
    *("experiments/pirc17/" + name + ".py" for name in (
        "method_cost_probe", "method_training", "method_development", "method_inputs", "method_rollout",
        "origins", "rollout", "calibration", "metrics", "features", "workload",
        "physical_memory", "seed_resume_session", "seed_resume_lineage")),
)


def source_hashes():
    return {name: _hash(ROOT / name) for name in SOURCE_FILES}


def read_bound(path, digest):
    if _hash(path) != digest:
        raise ValueError("cost probe bound file changed")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_plan(plan):
    expected = {**CONSTANTS, "source_sha256": source_hashes()}
    if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("exact fixed method cost plan and source identities required")
    if plan["prior_preparation_empirical_seconds"] + plan["outer_wall_seconds"] > plan["preparation_empirical_cap_seconds"]:
        raise ValueError("cost batch exceeds cumulative preparation ceiling")
    for request in FORECASTS:
        fitted = required_slot(request["fit_slot_id"])["components"]
        predicted = required_slot(request["slot_id"])["components"]
        if {k: fitted.get(k) for k in TRAINING_FIELDS} != {k: predicted.get(k) for k in TRAINING_FIELDS}:
            raise ValueError("probe cannot substitute another fitted method")
    return plan


def publish(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def emit(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + "\n")
    stream.flush()
    # Keep model arrays and private input metadata out of terminal output.
    print(json.dumps(row, allow_nan=False), flush=True)


class PhaseBudget:
    def __init__(self, plan, log):
        self.plan, self.log, self.started = plan, log, time.perf_counter()
        self.current = None

    def guard(self):
        if time.perf_counter() - self.started >= self.plan["inner_wall_seconds"]:
            raise TimeoutError("method cost batch cooperative total cap reached")
        if available_physical_bytes() < self.plan["minimum_free_bytes"]:
            raise MemoryError("method cost batch requires 2 GiB available RAM")

    def run(self, kind, identity, function):
        self.guard()
        self.current = {"kind": kind, "identity": identity}
        cap = self.plan["phase_cooperative_caps_seconds"][kind]
        emit(self.log, {"event": "phase_started", **self.current, "cooperative_cap_seconds": cap})
        started = time.perf_counter()
        result = function()
        seconds = time.perf_counter() - started
        if seconds >= cap:
            raise TimeoutError(f"{kind} cooperative phase cap reached after {seconds:.6f}s; no retry")
        self.guard()
        emit(self.log, {"event": "phase_completed", **self.current, "elapsed_seconds": seconds})
        self.current = None
        return result, seconds


def cost_targets(prepared, sample_id):
    matches = [(p, s) for p, s in zip(prepared.prefixes, prepared.segments)
               if p.assignment.sample.sample_id == sample_id]
    if len(matches) != 1 or matches[0][0].assignment.method_role != "validation":
        raise ValueError("cost origin must be the exact admitted validation sample")
    prefix, segment = matches[0]
    n = len(prefix.visible_epoch_ns)
    if segment.segment_id != prefix.assignment.sample.segment_id or segment.time[n-1] != 0:
        raise ValueError("cost scoring segment/relative origin mismatch")
    times = segment.time[n:]
    indexes = observed_indices(times)
    lonlat = prefix.condition_at.frame.to_lonlat(prefix.origin.position_m[None])[0]
    frame = LocalFrame(*lonlat)
    targets = positions_in_scoring_frame(prefix, segment.state[n:][indexes], frame)
    return prefix, times[indexes], targets, frame


def execute(prepared, plan, output, budget):
    """No loaders in this seam; software tests can exercise exact dispatch."""
    fitted, fits, forecasts = {}, [], []
    prefix, horizons, targets, frame = cost_targets(prepared, plan["sample_id"])
    for slot in plan["fit_slots"]:
        result, seconds = budget.run("fit", slot,
            lambda: fit_development_method(prepared, slot, max_transitions=plan["max_transitions"]))
        if result.training["transitions_by_role"] != plan["expected_transitions"]:
            raise ValueError("registered training transitions changed")
        fitted[slot] = result
        path = output / f"fit-{len(fits):02d}.json"
        publish(path, {"slot_id": slot, "training": result.training, "model": result.dynamics.model.to_dict(),
            "dynamics": result.dynamics.identity(), "fit_only_seconds": result.fit_seconds,
            "fit_and_prepare_seconds": seconds, "formal_training_accepted": False})
        row = {"event": "fit_saved", "slot_id": slot, "path": path.name, "sha256": _hash(path),
               "fit_only_seconds": result.fit_seconds, "fit_and_prepare_seconds": seconds}
        fits.append(row)
        emit(budget.log, row)
    grid = EntropyGrid(tuple(np.arange(-10000, 10001, 250)), tuple(np.arange(-10000, 10001, 250)))
    for request in plan["forecasts"]:
        slot = request["slot_id"]
        components = required_slot(slot)["components"]
        model = fitted[request["fit_slot_id"]]
        result, seconds = budget.run("forecast", slot, lambda: forecast_method(
            model.dynamics, prefix.origin, horizons, propagation=components["poa"],
            integrator=components["integrator"], particles=plan["particles"], seed=plan["seed"],
            max_step_seconds=plan["max_step_seconds"], history_step_seconds=components["dt_seconds"],
            max_steps=plan["max_steps"], max_particle_steps=plan["max_particle_steps"],
            origin_id=plan["sample_id"], run_id=slot, condition_names=model.dynamics.model.condition_names,
            condition_at=prefix.condition_at, crn_pair_id=request["pair_id"]))
        positions = positions_in_scoring_frame(prefix, result.forecast.positions_m, frame)
        scores, score_seconds = budget.run("score", slot, lambda: score_path(positions, targets, horizons,
            time_weights=plan["time_weights"], entropy_grid=grid))
        array_path = output / f"forecast-{len(forecasts):02d}.npz"
        arrays = {"positions_m": positions, "target_positions_m": targets, "elapsed_seconds": horizons}
        if result.conditional_means_m is not None:
            arrays.update(conditional_means_method_frame_m=result.conditional_means_m,
                          conditional_covariances_method_frame_m2=result.conditional_covariances_m2)
        with array_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        detail_path = array_path.with_suffix(".json")
        publish(detail_path, {"request": request, "diagnostics": result.diagnostics, "scores": scores,
            "forecast_seconds": seconds, "common_score_seconds": score_seconds,
            "artifact_sha256": _hash(array_path), "cost_only": True,
            "cost_origin_in_calibration_population": True, "accuracy_claim_authorized": False})
        row = {"event": "forecast_saved", "slot_id": slot, "path": detail_path.name,
            "sha256": _hash(detail_path), "forecast_seconds": seconds, "common_score_seconds": score_seconds}
        forecasts.append(row)
        emit(budget.log, row)
    return {"fits": fits, "forecasts": forecasts, "actual_horizons_seconds": horizons.tolist()}


def worker(plan_path, digest):
    plan = validate_plan(read_bound(plan_path, digest))
    start = read_bound(OUTPUT / "start.json", _hash(OUTPUT / "start.json"))
    if start.get("plan_sha256") != digest or start.get("schema_version") != VERSION:
        raise ValueError("matching original method cost supervisor required")
    require_owned_job(start["job_name"])
    with (OUTPUT / "ledger.jsonl").open("x", encoding="utf-8") as log:
        budget = PhaseBudget(plan, log)
        emit(log, {"event": "initialization", "plan_sha256": digest, "worker_pid": os.getpid(),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "expected_fits": 5, "expected_forecasts": 5, "final_eval_reads": 0})
        try:
            evidence = plan["prepared_evidence"]
            prepared_record = read_bound(ROOT / evidence["path"], evidence["sha256"])
            previous = prepared_record["identity"]
            if previous["sha256"] != plan["expected_input_sha256"]:
                raise ValueError("prior preparation input identity changed")
            locations = plan["locations_source"]
            paths = read_bound(ROOT / locations["path"], locations["sha256"])["data_locations"]
            prepared, seconds = budget.run("load", "existing-485-window-population", lambda: load_method_development(
                paths["eligibility"], previous["eligibility_sha256"], paths["release"], paths["snapshot"],
                Path(paths["data_root"]) / "cond_slices", plan["source_trajectory"],
                sample_ids=previous["sample_ids"], budget=PreparationBudget(**plan["preparation"])))
            if prepared.identity != previous:
                raise ValueError("rehydrated method inputs differ from audited preparation")
            result = execute(prepared, plan, OUTPUT, budget)
            if source_hashes() != plan["source_sha256"]:
                raise ValueError("method cost sources changed during execution")
            budget.guard()
            result.update(schema_version=VERSION, status="complete", plan_sha256=digest,
                input_sha256=previous["sha256"], load_seconds=seconds,
                elapsed_seconds=time.perf_counter()-budget.started, cost_only=True,
                numerically_qualified=False, formal_training_accepted=False, final_eval_reads=0)
            publish(OUTPUT / "result.json", result)
            emit(log, {"event": "complete", "fits": len(result["fits"]), "forecasts": len(result["forecasts"]),
                "elapsed_seconds": result["elapsed_seconds"], "result_sha256": _hash(OUTPUT / "result.json")})
            return 0
        except Exception as exc:
            emit(log, {"event": "failed", "current_phase": budget.current,
                "error_type": type(exc).__name__, "error_message": str(exc)[:500],
                "elapsed_seconds": time.perf_counter()-budget.started, "automatic_retry": False})
            return 1


def supervise(plan_path, digest):
    plan = validate_plan(read_bound(plan_path, digest))
    OUTPUT.mkdir(parents=True, exist_ok=False)  # Exclusive reservation survives any failure.
    started = time.perf_counter()
    job_name = "PIRC17-METHOD-COST-" + digest[:24]
    publish(OUTPUT / "start.json", {"schema_version": VERSION, "plan_sha256": digest,
        "job_name": job_name, "supervisor_pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
        "outer_wall_seconds": plan["outer_wall_seconds"]})
    command = [sys.executable, "-u", "-m", "experiments.pirc17.method_cost_probe", "--worker",
               "--plan", str(Path(plan_path).resolve()), "--plan-sha256", digest]
    result = run_owned(command, timeout=plan["outer_wall_seconds"]-(time.perf_counter()-started), job_name=job_name)
    result.update(finished_at=datetime.now(timezone.utc).isoformat(), plan_sha256=digest,
                  elapsed_seconds_including_supervision=time.perf_counter()-started)
    publish(OUTPUT / "supervision.json", result)
    print(json.dumps({"event": "owned_method_cost_closed", **result}), flush=True)
    return 0 if (result["worker_returncode"] == 0 and result["process_tree_closed"]
        and not result["timed_out"] and not result["interrupted"] and result["supervisor_error"] is None) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    return (worker if args.worker else supervise)(args.plan, args.plan_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
