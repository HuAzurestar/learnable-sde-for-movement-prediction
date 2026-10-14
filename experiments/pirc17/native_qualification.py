"""Bounded primary-family numerical or seed-planning pilots using the unchanged engine.

A hash-bound plan specifies every workload and both process/resource budgets.
The original complete candidate audit is reproduced before new predictions.
Completion means complete auditable execution, never numerical qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from os import getpid
from pathlib import Path
import re
import subprocess
import sys
import time

from . import direct_linear_rollout as engine
from . import physical_memory
from .direct_linear_evidence import read_bound
from .inference import PRIMARY_FAMILY
from .precision_check import load_particle_evidence
from .qualification import _hash
from .resource_replay import MemoryObserver, audited, numerical_summary, source_hashes as replay_sources

VERSION = "pirc17-native-primary-qualification-v2"
PLAN_VERSION = "pirc17-native-primary-plan-v1"
SEED_PLAN_VERSION = "pirc17-native-primary-seed-plan-v1"
PLAN_PURPOSES = {PLAN_VERSION: "complete_primary_family_validation_numerical_pilot",
                 SEED_PLAN_VERSION: "complete_primary_family_validation_seed_planning"}
DIGESTS = ("reference_ledger_sha256", "reference_audit_sha256", "fit_sha256",
           "fit_ledger_sha256", "training_policy_sha256", "eligibility_sha256")
AXES = ("configurations", "seeds", "particles", "steps", "limit_origins",
        "selection_policy", "map_backend", "wall_seconds")
PLAN_KEYS = {"schema_version", "purpose", "expected_run_count", "outer_wall_seconds",
             "minimum_free_bytes", "tolerance_m", "certified", "formal_training_accepted",
             "final_eval_authorized", *DIGESTS, *AXES}


def source_hashes():
    return {**engine.source_hashes(), **replay_sources(), "native_qualification.py": _hash(Path(__file__))}


def load_plan(path, digest):
    plan = read_bound(Path(path), digest)
    if (set(plan) != PLAN_KEYS or plan["schema_version"] not in PLAN_PURPOSES
            or plan["purpose"] != PLAN_PURPOSES[plan["schema_version"]]
            or any(plan[key] is not False for key in ("certified", "formal_training_accepted", "final_eval_authorized"))
            or type(plan["minimum_free_bytes"]) is not int or plan["minimum_free_bytes"] != engine.MINIMUM_FREE_BYTES
            or any(not isinstance(plan[key], str) or not re.fullmatch("[0-9a-f]{64}", plan[key]) for key in DIGESTS)):
        raise ValueError("exact unsealed validation-only plan, source bindings and RAM floor required")
    engine.validate_workload(**{key: plan[key] for key in AXES})
    family = {name for pair in PRIMARY_FAMILY.values() for name in pair}
    expected = plan["limit_origins"]
    for key in ("configurations", "seeds", "particles", "steps"):
        expected *= len(plan[key])
    numerical = plan["schema_version"] == PLAN_VERSION
    valid_axes = (len(plan["particles"]) >= 2 and len(plan["steps"]) >= 2) if numerical else (
        len(plan["particles"]) == len(plan["steps"]) == 1
        and plan["seeds"] in (list(engine.SEEDS[1:3]), list(engine.SEEDS[3:5])))
    if (set(plan["configurations"]) != family or plan["selection_policy"] != "lexical_independent_blocks"
            or plan["map_backend"] != "multicell" or not valid_axes
            or type(plan["expected_run_count"]) is not int or plan["expected_run_count"] != expected
            or type(plan["outer_wall_seconds"]) is not int
            or not plan["wall_seconds"] < plan["outer_wall_seconds"] <= 12*3600
            or isinstance(plan["tolerance_m"], bool) or not isinstance(plan["tolerance_m"], (int, float))
            or not 0 < plan["tolerance_m"] < float("inf")):
        raise ValueError("complete primary family, profile-specific axes, fixed denominator and bounded outer cap required")
    return plan


def seed_reference_contract(spec, reference, report):
    """Only after the complete, unchanged reference audit has reproduced."""
    if spec["schema_version"] != SEED_PLAN_VERSION:
        return None
    init, header = reference[:2]
    if (header["seeds"] != [engine.SEEDS[0]]
            or len(header["particle_counts"]) < 2 or len(header["max_steps_seconds"]) < 2
            or spec["configurations"] != header["configurations"]
            or spec["limit_origins"] != init["limit_origins"]
            or spec["selection_policy"] != init["selection_policy"]
            or spec["map_backend"] != init["map_backend"]
            or spec["particles"] != [max(header["particle_counts"])]
            or spec["steps"] != [min(header["max_steps_seconds"])]):
        raise ValueError("seed planning requires the whole first-seed primary refinement pilot and its finest/largest setting")
    return {"reference_seeds": header["seeds"], "new_seeds": spec["seeds"],
        "required_union_seeds": list(engine.SEEDS), "sample_ids": header["sample_ids"],
        "particles": spec["particles"][0], "step_seconds": spec["steps"][0],
        "reference_numerical": numerical_summary(report), "numerically_qualified": False,
        "scope": "disjoint seed-planning inputs only; no new refinement or scientific power result"}


def validate_seed_result(spec, reference, candidate):
    """Keep the full reference; only seeds, resolution axes and timing may vary."""
    changes = {"seeds", "particle_counts", "max_steps_seconds", "expected_run_count"}
    for old, new, allowed in ((reference[0], candidate[0], changes | {"started_at", "wall_seconds"}),
                              (reference[1], candidate[1], changes | {"input_validation_seconds"})):
        if {k: v for k, v in old.items() if k not in allowed} != {
                k: v for k, v in new.items() if k not in allowed}:
            raise ValueError("seed planning changes reference cohort, model, runtime, map, source or scoring identity")
    header = candidate[1]
    if (header["seeds"] != spec["seeds"] or header["particle_counts"] != spec["particles"]
            or header["max_steps_seconds"] != spec["steps"]
            or header["expected_run_count"] != spec["expected_run_count"]):
        raise ValueError("seed planning output differs from the exact registered workload")
    originals = {(row["sample_id"], row["configuration"]): row for row in reference[2:-1]
        if row["particles"] == spec["particles"][0] and row["max_step_seconds"] == spec["steps"][0]}
    return originals


def output_paths(output):
    output = Path(output).resolve()
    if output.suffix != ".jsonl":
        raise ValueError("native qualification envelope must use .jsonl")
    return {"envelope": output, "forecast": output.with_suffix(".forecast.jsonl"),
            "particles": output.with_suffix(".forecast.particles"), "audit": output.with_suffix(".audit.json"),
            "supervisor": output.with_suffix(".supervisor.json")}


def require_absent(paths):
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite qualification envelope, forecasts, particles, audit or supervisor")


def run(*, plan, plan_sha256, reference_ledger, reference_audit, fit, fit_ledger,
        eligibility, release, snapshot, data_root, output, progress=None, _supervised=False):
    plan_path = Path(plan).resolve()
    spec = load_plan(plan_path, plan_sha256)
    paths = output_paths(output)
    if _supervised:
        supervisor = json.loads(paths["supervisor"].read_text(encoding="utf-8"))
        if (supervisor.get("schema_version") != VERSION+"-supervisor" or supervisor.get("status") != "running"
                or supervisor.get("plan_sha256") != plan_sha256):
            raise ValueError("worker requires its matching supervisor reservation")
        require_absent({key: path for key, path in paths.items() if key != "supervisor"})
    else:
        require_absent(paths)
    evidence = {"fit": Path(fit), "fit_ledger": Path(fit_ledger), **{
        key: spec[key] for key in ("fit_sha256", "fit_ledger_sha256", "training_policy_sha256")}}
    bound = {plan_path: plan_sha256, Path(reference_ledger): spec["reference_ledger_sha256"],
             Path(reference_audit): spec["reference_audit_sha256"], Path(fit): spec["fit_sha256"],
             Path(fit_ledger): spec["fit_ledger_sha256"], Path(eligibility): spec["eligibility_sha256"]}
    sources, observer_identity = source_hashes(), physical_memory.identity()
    observer, errors = MemoryObserver(), []
    started = time.perf_counter()
    execution = report = seed_contract = None
    paths["envelope"].parent.mkdir(parents=True, exist_ok=True)
    with paths["envelope"].open("x", encoding="utf-8") as target:
        def emit(row):
            target.write(json.dumps(row, allow_nan=False)+"\n")
            target.flush()

        def failure(exc):
            row = {"type": "failure", "error_type": type(exc).__name__, "error_message": str(exc)[:300]}
            errors.append(row)
            emit(row)

        emit({"type": "initialization", "schema_version": VERSION,
            "purpose": spec["purpose"],
            "started_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_sha256, "plan": spec,
            "worker_pid": getpid(),
            "source_sha256": sources, "observer": observer_identity,
            "forecast_path": paths["forecast"].name, "audit_path": paths["audit"].name,
            "expected_run_count": spec["expected_run_count"], "certified": False,
            "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0})
        try:
            observer.guard()
            reference = read_bound(Path(reference_ledger), spec["reference_ledger_sha256"], jsonl=True)
            original = read_bound(Path(reference_audit), spec["reference_audit_sha256"])
            checked = audited(reference, Path(reference_ledger).resolve().parent,
                              spec["reference_ledger_sha256"], spec["tolerance_m"], evidence)
            if checked != original or checked["status"] != "complete":
                raise ValueError("original complete candidate audit and unchanged tolerance must exactly reproduce")
            if reference[0]["eligibility_sha256"] != spec["eligibility_sha256"]:
                raise ValueError("new workload must retain the original fit eligibility")
            seed_contract = seed_reference_contract(spec, reference, checked)
            runtime = {"python": engine.platform.python_version(), "numpy": engine.np.__version__,
                "torch": engine.torch.__version__, "torch_intraop_threads": engine.torch.get_num_threads(),
                "torch_interop_threads": engine.torch.get_num_interop_threads()}
            if reference[0].get("runtime") != runtime:
                raise ValueError("candidate runtime versions or thread counts changed from the reference")
            verified = {"type": "reference_verified", "numerical": numerical_summary(original)}
            if seed_contract is not None:
                verified["seed_planning_contract"] = seed_contract
            emit(verified)
            if progress:
                progress({"phase": "reference_verified", "expected_runs": spec["expected_run_count"]})
            observer.guard()
            execution = engine.run(**evidence, eligibility=Path(eligibility), eligibility_sha256=spec["eligibility_sha256"],
                release=Path(release), snapshot=Path(snapshot), data_root=Path(data_root), output=paths["forecast"],
                **{key: spec[key] for key in AXES}, available_memory=observer, progress=progress)
            forecast_sha = _hash(paths["forecast"])
            bound[paths["forecast"]] = forecast_sha
            emit({"type": "engine_completed", "ledger_sha256": forecast_sha, "completion": execution})
            observer.guard()
            candidate = read_bound(paths["forecast"], forecast_sha, jsonl=True)
            report = audited(candidate, paths["forecast"].parent, forecast_sha, spec["tolerance_m"], evidence)
            with paths["audit"].open("x", encoding="utf-8") as audit_target:
                json.dump(report, audit_target, indent=2, allow_nan=False)
            bound[paths["audit"]] = _hash(paths["audit"])
            emit({"type": "audit", "path": paths["audit"].name, "sha256": bound[paths["audit"]],
                  "numerical": numerical_summary(report)})
            if report["status"] != "complete":
                raise ValueError("incomplete candidate pilot; do not salvage a smaller workload")
            if seed_contract is not None:
                originals = validate_seed_result(spec, reference, candidate)
                for row in candidate[2:-1]:
                    previous = originals[(row["sample_id"], row["configuration"])]
                    old = load_particle_evidence(previous, Path(reference_ledger).resolve().parent)
                    new = load_particle_evidence(row, paths["forecast"].parent)
                    if (row["independent_block_id"] != previous["independent_block_id"]
                            or row["actual_horizons_seconds"] != previous["actual_horizons_seconds"]
                            or not all(engine.np.array_equal(a, b) for a, b in zip(old[1:], new[1:]))):
                        raise ValueError("seed planning changes paired reference block, observed targets or times")
            observer.guard()
        except Exception as exc:
            failure(exc)
        finally:
            try:
                if (source_hashes() != sources or physical_memory.identity() != observer_identity
                        or any(_hash(path) != digest for path, digest in bound.items())):
                    raise ValueError("qualification sources, plan, bound inputs or output evidence changed")
            except Exception as exc:
                failure(exc)
        expected = spec["expected_run_count"]
        counts = {key: execution[key] if execution else default for key, default in
                  (("attempted_run_count", 0), ("success_count", 0), ("failure_count", 0), ("unattempted_run_count", expected))}
        complete = bool(not errors and report and report["status"] == "complete"
                        and counts["success_count"] == expected and observer.calls and not observer.failures)
        completion = {"type": "completion", "status": "complete" if complete else "failed",
            "expected_run_count": expected, **counts, "terminal_error_count": len(errors),
            "engine_terminal_error_count": execution["terminal_error_count"] if execution else 0,
            "resource_stopped": bool((execution and execution["resource_stopped"]) or
                any(row["error_type"] in {"MemoryError", "TimeoutError"} for row in errors)),
            "observer": observer.summary(), "candidate_numerical": numerical_summary(report) if report else None,
            "seed_planning_contract": seed_contract, "numerically_qualified": False,
            "elapsed_seconds": time.perf_counter()-started, "certified": False,
            "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0}
        emit(completion)
    return completion


def terminate_owned(child):
    """Only a still-live process created and retained by this supervisor is targeted."""
    if child.poll() is not None:
        return
    if os.name != "nt":
        raise RuntimeError("native Windows qualification supervisor requires Windows")
    result = subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode and child.poll() is None:
        raise RuntimeError("owned worker tree termination could not be confirmed")
    child.wait(timeout=20)


def supervise(args, worker_arguments):
    """Hard outer cap is enforced here, independently of cooperative engine guards."""
    spec = load_plan(args["plan"], args["plan_sha256"])
    paths = output_paths(args["output"])
    require_absent(paths)
    paths["supervisor"].parent.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    record = {"schema_version": VERSION+"-supervisor", "plan_sha256": args["plan_sha256"],
        "status": "running", "supervisor_pid": getpid(),
        "started_at": started_at.isoformat(), "outer_wall_seconds": spec["outer_wall_seconds"],
        "deadline_at": (started_at+timedelta(seconds=spec["outer_wall_seconds"])).isoformat(),
        "worker_pid": None, "worker_returncode": None, "timed_out": False, "termination_confirmed": False,
        "certified": False, "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0}
    child = None
    error = None
    # Reserve the output atomically BEFORE launching a worker. A second main
    # invocation must fail even before the first worker emits initialization.
    # Only this held, newly created handle is updated on terminal completion.
    with paths["supervisor"].open("x", encoding="utf-8") as target:
        json.dump(record, target, indent=2, allow_nan=False)
        target.flush()
        try:
            if os.name != "nt":
                raise RuntimeError("native Windows qualification supervisor requires Windows")
            command = [sys.executable, "-u", "-m", "experiments.pirc17.native_qualification", "--worker", *worker_arguments]
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            record["worker_pid"] = child.pid
            print(json.dumps({"phase": "worker_started", "supervisor_pid": record["supervisor_pid"],
                "worker_pid": child.pid, "deadline_at": record["deadline_at"]}), flush=True)
            try:
                child.wait(timeout=max(0., spec["outer_wall_seconds"]-(time.perf_counter()-begin)))
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                terminate_owned(child)
            record["worker_returncode"] = child.poll()
            record["termination_confirmed"] = record["worker_returncode"] is not None
        except BaseException as exc:
            error = exc
            record.update(error_type=type(exc).__name__, error_message=str(exc)[:300])
            if child is not None and child.poll() is None:
                try:
                    terminate_owned(child)
                except Exception as stop_error:
                    record["termination_error"] = str(stop_error)[:300]
            if child is not None:
                record["worker_returncode"] = child.poll()
                record["termination_confirmed"] = record["worker_returncode"] is not None
        finally:
            record.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-begin,
                status="complete" if not error and not record["timed_out"] and record["worker_returncode"] == 0 else "failed")
            # A supervisor timeout never fabricates an inner completion row.
            target.seek(0)
            json.dump(record, target, indent=2, allow_nan=False)
            target.truncate()
            target.flush()
            print(json.dumps({"phase": "supervisor_completion", **record}), flush=True)
    if error:
        raise error
    return 124 if record["timed_out"] else record["worker_returncode"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "reference-ledger", "reference-audit", "fit", "fit-ledger", "eligibility",
                 "release", "snapshot", "data-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = vars(parser.parse_args())
    worker = args.pop("worker")
    if not worker:
        return supervise(args, sys.argv[1:])
    result = run(**args, _supervised=True, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
