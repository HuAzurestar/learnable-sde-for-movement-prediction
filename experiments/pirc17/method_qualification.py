"""One finite development-method qualification attempt, not final evaluation.

Reuse five CLOSED deterministic fits, fit the eleven missing recipes once, and
run exactly three existing development origins x five seeds x (28 slots + one
same-grid Full reference). No precision grid, cost probe, retry or final labels.
An exclusive reservation and owned Windows Job preserve the attempt/deadline.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from . import method_cost_probe as cost
from .cached_method_mechanisms import restore_cached_fit, PLAN_SHA as COST_PLAN_SHA, RESULT_SHA as COST_RESULT_SHA
from .calibration import NOMINAL_SECONDS, TIME_WEIGHTS
from .inference import SEEDS
from .method_comparisons import method_comparison_registry
from .method_development import PreparationBudget, load_method_development, _hash
from .method_inputs import positions_in_scoring_frame
from .method_mechanisms import (EXACT_REFERENCE, NUMERICAL_SLOTS, SCORE_SLOTS, mechanism_registry,
    forecast_stream_binding, model_gate, score_diagnostic, integration_diagnostic, _digest)
from .method_qualification_analysis import analyze
from .method_rollout import forecast_method
from .method_training import FIT_SOURCES, TRAINING_FIELDS, FittedMethod, fit_development_method, required_slot
from .metrics import EntropyGrid, score_path
from .physical_memory import available_physical_bytes
from .seed_resume_session import require_owned_job, run_owned
from .workload import method_inventory

VERSION = "pirc17-method-qualification-p512-h5-v1"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/pirc17/dev10/method-qualification-p512-h5-v1"
PLAN_PATH = ROOT / "experiments/pirc17/plans/method-qualification-p512-h5-v1.json"
CACHED = ROOT / "artifacts/pirc17/dev10/method-cost-p512-h5-v1"
ORIGINS = {
    "50da7cfb18afb6bde2ee56bbbfa81698b47af8e262b2cc8554955da23faf225a": [
        "00d6711e02d7224c4f68497b0b751391bab28a2844cf2bda1e8d7e57762a1a8b"],
    "a5b32b31086bc6e7c3d26999dab18b0b439bf4ad93223e7fc67dbccc17642e7c": [
        "0924849bf85b7c13eb7fc8378d0725b63e50b46c3e074e08f204a2303fdd6149"],
    "b1c20f396c1e4e06c9547cbd7f9a494a1a497187caf6397979206784dffbaaba": [
        "0bbad69a73cae2f606dee4d88314ba1af8cfcc41d34b8c99432ef745573e02fa"],
}
NEW_FITS = ["arm-02/pointwise", "arm-03/single_gaussian", "arm-06/dt30", "arm-06/dt120",
    "arm-06/dt300", "arm-06/dt600", "arm-08/qmle", "arm-09/pure_es", "arm-12/scratch",
    "arm-15/drift_only", "arm-15/two_step"]
TRANSITIONS = {
    "30": {"train": 52170, "adapt": 11567, "validation": 12340},
    "60": {"train": 26009, "adapt": 5762, "validation": 6149},
    "120": {"train": 12918, "adapt": 2861, "validation": 3055},
    "300": {"train": 5079, "adapt": 1124, "validation": 1196},
    "600": {"train": 2467, "adapt": 545, "validation": 578},
}
SOURCE_FILES = tuple(dict.fromkeys((*cost.SOURCE_FILES,
    *("experiments/pirc17/"+name+".py" for name in ("method_qualification", "method_qualification_analysis",
        "method_mechanisms", "cached_method_mechanisms", "method_comparisons", "comparison_registry",
        "inference", "inference_guard")))))


def source_hashes():
    return {name: _hash(ROOT/name) for name in SOURCE_FILES}


def recipe(slot):
    components = required_slot(slot)["components"]
    return {k: components[k] for k in TRAINING_FIELDS if k in components}


def plan_definition():
    slots = [r["slot_id"] for r in method_inventory()["slots"] if r["disposition"] == "REQUIRED"]
    keys = [_digest(recipe(s)) for s in (*cost.FIT_SLOTS, *NEW_FITS)]
    if len(keys) != 16 or len(set(keys)) != 16 or not {_digest(recipe(s)) for s in slots} <= set(keys):
        raise ValueError("five cached plus eleven new recipes must cover all28 required slots exactly")
    return {
        "schema_version": VERSION, "purpose": "finite_development_method_mechanism_and_paired_planning_evidence",
        "source_sha256": source_hashes(), "method_registry_sha256": method_comparison_registry()["sha256"],
        "mechanism_registry_sha256": mechanism_registry()["sha256"],
        "expected_origins_by_block": copy.deepcopy(ORIGINS), "origin_mode": "causal_prefix",
        "selection": "the same three pre-existing terrain development origins, no outcome-based replacement",
        "new_fit_slots": list(NEW_FITS), "cached_fit_slots": list(cost.FIT_SLOTS),
        "required_slots": slots, "forecast_order_per_origin_seed": [EXACT_REFERENCE, *slots],
        "new_fits": 11, "reused_fits": 5, "scientific_slot_forecasts": 420,
        "same_grid_reference_forecasts": 15, "expected_forecasts": 435,
        "particles": 512, "max_step_seconds": 5., "seeds": list(SEEDS),
        "nominal_seconds": list(NOMINAL_SECONDS), "time_weights": list(TIME_WEIGHTS),
        "max_steps": 400, "max_particle_steps": 204800, "max_transitions": 80000,
        "expected_transitions_by_interval": copy.deepcopy(TRANSITIONS), "planning_blocks": 46,
        "score_diagnostic_extra_Gaussian_draws": 3*5*2*4*256,
        "preparation": copy.deepcopy(cost.CONSTANTS["preparation"]),
        "phase_cooperative_caps_seconds": {"load": 180, "fit": 90, "forecast": 30, "score": 10, "analysis": 60},
        "inner_wall_seconds": 1000, "outer_wall_seconds": 1060,
        "termination_reserve_seconds": 40,
        "independent_saved_output_audit_cap_seconds": 240,
        "preparation_empirical_cap_seconds": 1800, "prior_preparation_empirical_seconds": 393.318805,
        "minimum_free_bytes": 2147483648, "max_output_bytes": 256*1024**2, "max_artifact_bytes": 2*1024**2,
        "locations_source": copy.deepcopy(cost.CONSTANTS["locations_source"]),
        "prepared_evidence": copy.deepcopy(cost.CONSTANTS["prepared_evidence"]),
        "expected_input_sha256": cost.CONSTANTS["expected_input_sha256"],
        "source_trajectory": cost.CONSTANTS["source_trajectory"],
        "cached_cost_plan_sha256": COST_PLAN_SHA, "cached_cost_result_sha256": COST_RESULT_SHA,
        "cached_mechanism_report": {"path": "artifacts/pirc17/dev10/cached-method-mechanisms-v1.json",
            "sha256": "c3ec457b6920d51e514a5ebea8baa1f8e8400b8317795797d3229e89a7c46133"},
        "failure_policy": "one attempt per item; retain failed/unavailable rows, continue independent items only while resources remain; timeout/memory/output limit stops batch; never select successful intersections or rerun this reservation",
        "gate_failure_policy": "failed mechanism gates are results, not reasons to tune, refit or repeat",
        "numerical_scope": "Full same-grid exact-affine-kernel diagnostics plus descriptive five-seed simulation SE; not all-method h/N convergence or a global nonlinear solution",
        "calibration_exposure": "validation origins overlap the estimator calibration population; planning/mechanism evidence only",
        "secondary_origin_modes_in_this_batch": [], "automatic_retry": False, "automatic_expansion": False,
        "formal_training_accepted": False, "numerically_qualified": False, "power_qualified": False,
        "scientific_claim_authorized": False, "final_eval_authorized": False,
    }


def validate_plan(plan):
    if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(plan_definition(), sort_keys=True, allow_nan=False):
        raise ValueError("exact finite method qualification plan and source identities required")
    if (plan["prior_preparation_empirical_seconds"]+plan["outer_wall_seconds"]
            +plan["independent_saved_output_audit_cap_seconds"] > plan["preparation_empirical_cap_seconds"]):
        raise ValueError("qualification plus audit exceed cumulative preparation cap")
    return plan


def read_bound(path, sha):
    if _hash(path) != sha:
        raise ValueError("qualification bound source hash changed")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emit(log, row):
    log.write(json.dumps(row, allow_nan=False)+"\n")
    log.flush()
    if row["event"] in {"initialization", "fit_saved", "fit_reused", "item_failed", "group_complete", "complete", "failed"}:
        print(json.dumps(row, allow_nan=False), flush=True)


class Budget:
    def __init__(self, plan, log):
        self.plan, self.log, self.started = plan, log, time.perf_counter()
        self.current, self.written_bytes = None, 0

    def guard(self):
        if time.perf_counter()-self.started >= self.plan["inner_wall_seconds"]:
            raise TimeoutError("qualification total cap reached; no expansion or retry")
        if available_physical_bytes() < self.plan["minimum_free_bytes"]:
            raise MemoryError("qualification requires2GiB free RAM")
        if self.written_bytes > self.plan["max_output_bytes"]:
            raise OSError("qualification saved-output cap exceeded")

    def run(self, phase, identity, function):
        self.guard()
        self.current = {"phase": phase, "identity": identity}
        emit(self.log, {"event": "phase_started", **self.current})
        started = time.perf_counter()
        value = function()
        seconds = time.perf_counter()-started
        if seconds >= self.plan["phase_cooperative_caps_seconds"][phase]:
            raise TimeoutError(f"qualification {phase} phase cap reached; no retry")
        self.guard()
        emit(self.log, {"event": "phase_complete", **self.current, "seconds": seconds})
        self.current = None
        return value, seconds

    def save(self, path, value, *, arrays=False):
        if arrays:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **value)
            raw = buffer.getvalue()
        else:
            raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
        if len(raw) > self.plan["max_artifact_bytes"] or self.written_bytes+len(raw) > self.plan["max_output_bytes"]:
            raise OSError("qualification saved-output cap would be exceeded")
        with Path(path).open("xb") as stream:
            stream.write(raw)
        self.written_bytes += len(raw)
        return _hash(path)


def fit_record(fit, *, seconds, reused):
    return {"slot_id": fit.slot_id, "training": fit.training, "model": fit.dynamics.model.to_dict(),
        "dynamics": fit.dynamics.identity(), "fit_only_seconds": fit.fit_seconds,
        "fit_and_prepare_seconds": seconds, "reused_closed_fit": reused, "formal_training_accepted": False}


def check_fit(row, plan):
    slot, training = row["slot_id"], row["training"]
    body = {k: v for k, v in training.items() if k != "training_identity_sha256"}
    tau = str(int(required_slot(slot)["components"]["dt_seconds"]))
    if (training["input_sha256"] != plan["expected_input_sha256"]
            or training["training_components"] != recipe(slot)
            or training["training_identity_sha256"] != _digest(body)
            or training["sample_counts"] != {"train": 328, "adapt": 76, "validation": 81}
            or training["transitions_by_role"] != plan["expected_transitions_by_interval"][tau]
            or training["numpy_version"] != np.__version__
            or set(training["source_sha256"]) != set(FIT_SOURCES)
            or any(plan["source_sha256"].get(k) != v for k, v in training["source_sha256"].items())
            or row["formal_training_accepted"] is not False or training["formal_training_accepted"] is not False):
        raise ValueError("fitted recipe/input/transition/source identity changed")
    dynamics = restore_cached_fit(row)
    return FittedMethod(dynamics, training, slot, row["fit_only_seconds"])


def cached_fits(plan):
    closed = read_bound(CACHED/"result.json", plan["cached_cost_result_sha256"])
    old_plan = read_bound(ROOT/"experiments/pirc17/plans/method-cost-p512-h5-v1.json", plan["cached_cost_plan_sha256"])
    cost.validate_plan(old_plan)
    if (closed["status"] != "complete" or closed["plan_sha256"] != COST_PLAN_SHA
            or closed["final_eval_reads"] != 0 or not closed["cost_only"]
            or closed["formal_training_accepted"] or closed["numerically_qualified"]
            or [r["slot_id"] for r in closed["fits"]] != plan["cached_fit_slots"]):
        raise ValueError("complete closed cost provenance required for deterministic fit reuse")
    prior = plan["cached_mechanism_report"]
    read_bound(ROOT/prior["path"], prior["sha256"])
    result = []
    for i, item in enumerate(closed["fits"]):
        if item["path"] != f"fit-{i:02d}.json":
            raise ValueError("exact cached fit path/order required")
        path = CACHED/item["path"]
        row = read_bound(path, item["sha256"])
        if row["slot_id"] != item["slot_id"]:
            raise ValueError("cached fit manifest label differs")
        result.append((check_fit(row, plan), {"slot_id": item["slot_id"], "status": "success", "reused": True,
            "path": path.relative_to(ROOT).as_posix(), "sha256": item["sha256"], "new_fit_seconds": 0.}))
    return result


def execute(prepared, plan, output, budget, reused):
    """Target-free forecast dispatch, one physical execution per declared slot."""
    fitted, fit_rows, model_gates, rows, diagnostics = {}, [], {}, [], []
    for fit, row in reused:
        fitted[_digest(recipe(fit.slot_id))] = fit
        fit_rows.append(row)
        emit(budget.log, {"event": "fit_reused", "slot_id": fit.slot_id, "sha256": row["sha256"]})
    for i, slot in enumerate(plan["new_fit_slots"]):
        try:
            fit, seconds = budget.run("fit", slot, lambda: fit_development_method(
                prepared, slot, max_transitions=plan["max_transitions"]))
            detail = fit_record(fit, seconds=seconds, reused=False)
            check_fit(detail, plan)
            path = output/f"fit-{i:02d}.json"
            digest = budget.save(path, detail)
            fitted[_digest(recipe(slot))] = fit
            row = {"slot_id": slot, "status": "success", "reused": False,
                "path": path.relative_to(ROOT).as_posix(), "sha256": digest,
                "fit_only_seconds": fit.fit_seconds, "new_fit_seconds": seconds}
            emit(budget.log, {"event": "fit_saved", "slot_id": slot, "sha256": digest, "seconds": seconds})
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            row = {"slot_id": slot, "status": "failed", "reused": False,
                "reason": f"{type(exc).__name__}: {exc}"[:500]}
            emit(budget.log, {"event": "item_failed", "kind": "fit", **row})
            budget.guard()  # Resource exhaustion is not a recoverable item failure.
        fit_rows.append(row)
    registry = mechanism_registry()
    for slot, definition in registry["slots"].items():
        if definition.get("source") == "fitted-model":
            fit = fitted.get(_digest(recipe(slot)))
            model_gates[slot] = ({"status": "computed", "gate": model_gate(slot, fit.dynamics)} if fit else
                {"status": "unavailable", "reason": "registered fitted recipe failed; no substitution"})
    grid = EntropyGrid(tuple(np.arange(-10000, 10001, 250)), tuple(np.arange(-10000, 10001, 250)))
    for block, origins in plan["expected_origins_by_block"].items():
        for origin_id in origins:
            prefix, horizons, targets, frame = cost.cost_targets(prepared, origin_id)
            if prefix.assignment.sample.independent_block_id != block:
                raise ValueError("registered qualification origin changed block")
            context = {"input_sha256": prepared.identity["sha256"], "prefix": prefix.identity(),
                "elapsed_seconds": horizons.tolist(), "target_positions_scoring_frame_m": targets.tolist()}
            context_sha = _digest(context)
            for seed in plan["seeds"]:
                group = {}
                for slot in plan["forecast_order_per_origin_seed"]:
                    scientific_slot = "arm-01/full" if slot == EXACT_REFERENCE else slot
                    c = required_slot(scientific_slot)["components"]
                    fit = fitted.get(_digest(recipe(scientific_slot)))
                    identity = {"origin_id": origin_id, "block_id": block, "seed": seed, "slot_id": slot,
                        "split": "validation", "context_sha256": context_sha, "elapsed_seconds": horizons.tolist()}
                    if fit is None:
                        row = {**identity, "status": "unavailable", "reason": "registered fitted recipe failed"}
                        rows.append(row)
                        emit(budget.log, {"event": "forecast_unavailable", **row})
                        continue
                    binding = forecast_stream_binding(slot, plan["origin_mode"])
                    try:
                        result, seconds = budget.run("forecast", identity, lambda: forecast_method(
                            fit.dynamics, prefix.origin, horizons, propagation=c["poa"],
                            integrator="exact" if slot == EXACT_REFERENCE else c["integrator"],
                            particles=plan["particles"], seed=seed, max_step_seconds=plan["max_step_seconds"],
                            history_step_seconds=c["dt_seconds"], max_steps=plan["max_steps"],
                            max_particle_steps=plan["max_particle_steps"], origin_id=origin_id, run_id=slot,
                            condition_names=fit.dynamics.model.condition_names,
                            condition_at=prefix.condition_at if fit.dynamics.model.condition_names else None,
                            crn_pair_id=binding["crn_pair_id"]))
                        positions = positions_in_scoring_frame(prefix, result.forecast.positions_m, frame)
                        scores, score_seconds = budget.run("score", identity, lambda: score_path(
                            positions, targets, horizons, time_weights=plan["time_weights"], entropy_grid=grid))
                        arrays = {"positions_m": positions, "target_positions_m": targets, "elapsed_seconds": horizons}
                        if result.conditional_means_m is not None:
                            arrays.update(conditional_means_method_frame_m=result.conditional_means_m,
                                conditional_covariances_method_frame_m2=result.conditional_covariances_m2)
                        stem = output/f"forecast-{len(rows):04d}"
                        array_sha = budget.save(stem.with_suffix(".npz"), arrays, arrays=True)
                        gate = None
                        if slot in SCORE_SLOTS:
                            gate = score_diagnostic(slot, positions, targets, seed=seed, origin_id=origin_id,
                                input_identity_sha256=prepared.identity["sha256"])
                        detail = {**identity, "status": "success", "context": context,
                            "diagnostics": result.diagnostics, "scores": scores, "score_gate": gate,
                            "artifact_sha256": array_sha, "forecast_seconds": seconds, "score_seconds": score_seconds,
                            "source_input_sha256": prepared.identity["sha256"],
                            "development_calibration_exposed": True, "scientific_claim_authorized": False}
                        detail_sha = budget.save(stem.with_suffix(".json"), detail)
                        row = {**identity, "status": "success", "score_m": scores["time_weighted_energy_score_m"],
                            "path": stem.with_suffix(".json").name, "sha256": detail_sha,
                            "diagnostics": result.diagnostics, "score_gate": gate}
                        group[slot] = result
                        emit(budget.log, {"event": "forecast_saved", **identity,
                            "path": row["path"], "sha256": detail_sha, "seconds": seconds})
                    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                        row = {**identity, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:500]}
                        emit(budget.log, {"event": "item_failed", "kind": "forecast", **row})
                        budget.guard()
                    rows.append(row)
                for slot in sorted(NUMERICAL_SLOTS):
                    if slot in group and EXACT_REFERENCE in group:
                        gate = integration_diagnostic(slot, group[slot], group[EXACT_REFERENCE],
                            approximate_context_sha256=context_sha, reference_context_sha256=context_sha)
                        diagnostics.append({"origin_id": origin_id, "seed": seed, "slot_id": slot,
                            "status": "computed", "gate": gate})
                    else:
                        diagnostics.append({"origin_id": origin_id, "seed": seed, "slot_id": slot,
                            "status": "unavailable", "reason": "paired approximate/reference forecast missing"})
                budget.guard()
                emit(budget.log, {"event": "group_complete", "origin_id": origin_id, "seed": seed,
                    "records": len(rows), "expected": plan["expected_forecasts"]})
    # Large per-forecast provenance lives in individual files, not result.json.
    analysis, seconds = budget.run("analysis", "whole-five-seed-method-evidence", lambda: analyze(rows,
        expected_origins_by_block=plan["expected_origins_by_block"],
        slots=plan["forecast_order_per_origin_seed"], planning_blocks=plan["planning_blocks"]))
    return {"fits": fit_rows, "forecasts": [{k: v for k, v in row.items() if k not in {"diagnostics", "score_gate"}}
            for row in rows], "model_gates": model_gates, "integration_gates": diagnostics,
        "score_gates": [{"origin_id": r["origin_id"], "seed": r["seed"], "slot_id": r["slot_id"],
            "status": r["status"], "gate": r.get("score_gate"), "reason": r.get("reason")}
            for r in rows if r["slot_id"] in SCORE_SLOTS], "analysis": analysis, "analysis_seconds": seconds}


def worker(plan_path, digest):
    plan = validate_plan(read_bound(plan_path, digest))
    start = read_bound(OUTPUT/"start.json", _hash(OUTPUT/"start.json"))
    if start.get("plan_sha256") != digest or start.get("schema_version") != VERSION:
        raise ValueError("matching original qualification supervisor required")
    require_owned_job(start["job_name"])
    with (OUTPUT/"ledger.jsonl").open("x", encoding="utf-8") as log:
        budget = Budget(plan, log)
        emit(log, {"event": "initialization", "plan_sha256": digest, "worker_pid": os.getpid(),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "expected_new_fits": 11, "expected_forecasts": 435, "final_eval_reads": 0})
        try:
            reused = cached_fits(plan)
            evidence = plan["prepared_evidence"]
            previous = read_bound(ROOT/evidence["path"], evidence["sha256"])["identity"]
            if previous["sha256"] != plan["expected_input_sha256"]:
                raise ValueError("audited preparation identity changed")
            locations = plan["locations_source"]
            paths = read_bound(ROOT/locations["path"], locations["sha256"])["data_locations"]
            prepared, load_seconds = budget.run("load", "existing-485-window-fit-inputs", lambda: load_method_development(
                paths["eligibility"], previous["eligibility_sha256"], paths["release"], paths["snapshot"],
                Path(paths["data_root"])/"cond_slices", plan["source_trajectory"], sample_ids=previous["sample_ids"],
                budget=PreparationBudget(**plan["preparation"])))
            if prepared.identity != previous:
                raise ValueError("rehydrated inputs differ from already audited development population")
            result = execute(prepared, plan, OUTPUT, budget, reused)
            if source_hashes() != plan["source_sha256"]:
                raise ValueError("qualification code changed during execution")
            budget.guard()
            complete = (result["analysis"]["successful_forecasts"] == plan["expected_forecasts"]
                        and all(r["status"] == "success" for r in result["fits"]))
            result.update(schema_version=VERSION, status="complete" if complete else "complete-with-unavailable-items",
                plan_sha256=digest, input_sha256=previous["sha256"], load_seconds=load_seconds,
                elapsed_seconds=time.perf_counter()-budget.started,
                final_eval_reads=0, formal_training_accepted=False, numerically_qualified=False,
                power_qualified=False, scientific_claim_authorized=False, automatic_retry=False)
            result_sha = budget.save(OUTPUT/"result.json", result)
            emit(log, {"event": "complete", "status": result["status"], "result_sha256": result_sha,
                "successful_forecasts": result["analysis"]["successful_forecasts"],
                "elapsed_seconds": result["elapsed_seconds"]})
            return 0  # Gate failure/unavailable evidence is not a worker crash.
        except Exception as exc:
            emit(log, {"event": "failed", "current_phase": budget.current,
                "error_type": type(exc).__name__, "error_message": str(exc)[:500],
                "elapsed_seconds": time.perf_counter()-budget.started, "automatic_retry": False})
            return 1


def supervise(plan_path, digest):
    plan = validate_plan(read_bound(plan_path, digest))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    job_name = "PIRC17-METHOD-QUALIFICATION-"+digest[:24]
    cost.publish(OUTPUT/"start.json", {"schema_version": VERSION, "plan_sha256": digest,
        "job_name": job_name, "supervisor_pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
        "outer_wall_seconds": plan["outer_wall_seconds"]})
    command = [sys.executable, "-u", "-m", "experiments.pirc17.method_qualification", "--worker",
        "--plan", str(Path(plan_path).resolve()), "--plan-sha256", digest]
    result = run_owned(command, timeout=plan["outer_wall_seconds"]-plan["termination_reserve_seconds"]
        -(time.perf_counter()-started), job_name=job_name)
    result.update(finished_at=datetime.now(timezone.utc).isoformat(), plan_sha256=digest,
        elapsed_seconds_including_supervision=time.perf_counter()-started)
    cost.publish(OUTPUT/"supervision.json", result)
    print(json.dumps({"event": "owned_method_qualification_closed", **result}), flush=True)
    return 0 if (result["worker_returncode"] == 0 and result["process_tree_closed"] and not result["timed_out"]
        and not result["interrupted"] and result["supervisor_error"] is None) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    return (worker if args.worker else supervise)(args.plan, args.plan_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
