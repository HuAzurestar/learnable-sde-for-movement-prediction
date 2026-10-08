"""One four-forecast development cost probe, reusing the unchanged engine.

No parameter search, retries, final-eval access, or numerical qualification.
An existing owned Windows Job enforces the hard outer process-tree deadline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from . import direct_linear_rollout as engine
from .direct_linear_evidence import read_bound
from .qualification import _hash
from .resource_replay import MemoryObserver
from .seed_resume_session import require_owned_job, run_owned

VERSION = "pirc17-lio-cost-probe-v1"
AXES = {"configurations": ["lio-road", "lio-river", "lio-worldcover", "lio-surface"],
        "seeds": [20260814], "particles": [512], "steps": [5.], "limit_origins": 1,
        "selection_policy": "lexical_independent_blocks", "map_backend": "multicell", "wall_seconds": 600}
PLAN_CONSTANTS = {"schema_version": VERSION, "purpose": "four_LIO_costs_only_no_precision_search",
    "axes": AXES, "expected_run_count": 4, "outer_wall_seconds": 660,
    "preparation_empirical_cap_seconds": 1800, "prior_preparation_empirical_seconds": 0,
    "final_eval_authorized": False, "numerical_qualification_claim": False, "automatic_retry": False,
    "nominal_horizon_seconds": 1800., "expected_sample_id": "00d6711e02d7224c4f68497b0b751391bab28a2844cf2bda1e8d7e57762a1a8b"}
OUTPUT = Path("artifacts/pirc17/dev10/lio-cost-p512-h5-v1.jsonl")


def source_hashes():
    names = ("lio_cost_probe.py", "physical_memory.py", "resource_replay.py", "seed_resume_session.py")
    return {**engine.source_hashes(), **{n: _hash(Path(__file__).with_name(n)) for n in names}}


def validate_plan(plan):
    if (set(plan) != {*PLAN_CONSTANTS, "source_sha256", "evidence", "locations_source"}
            or any(plan.get(k) != v for k, v in PLAN_CONSTANTS.items())
            or plan["source_sha256"] != source_hashes()):
        raise ValueError("exact fixed four-run development cost plan and source identity required")
    engine.validate_workload(**plan["axes"])
    if plan["locations_source"] != {
        "path": "artifacts/pirc17/dev10/native-primary-p1024-missing-v1.extension/root.json",
        "sha256": "381c000b3ba9919d08325968255a4708a001d4a2a550940765428d0a37ecbc7d",
        "use": "existing data locators only; never restart authority"}:
        raise ValueError("only the previously bound data locators may be reused")
    evidence = read_bound("experiments/pirc17/plans/small-budget-offline-v1.json",
        "f8b48a75ff0bc1a662960c60301531d84f778a3f3e2f37d4d6376b9792d5e462")["candidate_evidence"]
    if plan["evidence"] != evidence:
        raise ValueError("cost probe cannot refit or select different models")
    return plan


def _emit(log, row):
    log.write(json.dumps(row, allow_nan=False)+"\n")
    log.flush()
    print(json.dumps(row, allow_nan=False), flush=True)


def worker(plan_path, digest):
    plan = validate_plan(read_bound(plan_path, digest))
    supervision = OUTPUT.with_suffix(".supervisor.jsonl")
    start = json.loads(supervision.read_text().splitlines()[0])
    if start.get("plan_sha256") != digest or start.get("phase") != "owned_cost_started":
        raise ValueError("matching original supervisor required")
    require_owned_job(start["job_name"])
    observer = MemoryObserver()
    locations = plan["locations_source"]
    root = read_bound(locations["path"], locations["sha256"])
    paths = root["data_locations"]
    result = engine.run(**plan["evidence"], **{k: paths[k] for k in ("eligibility", "release", "snapshot", "data_root")},
        eligibility_sha256="7b6773ec628b7a14f586400ab315965d1f88b2227365b0b9a6cb35be9a7f9690",
        output=OUTPUT, **plan["axes"], available_memory=observer,
        progress=lambda row: print(json.dumps(row), flush=True))
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("cost probe source changed")
    rows = [json.loads(line) for line in OUTPUT.read_text().splitlines()]
    if result["status"] == "complete":
        if rows[1]["sample_ids"] != [plan["expected_sample_id"]]:
            raise ValueError("cost probe selected a different validation origin")
    summary = {"schema_version": VERSION+"-worker", "plan_sha256": digest,
        "ledger_sha256": _hash(OUTPUT), "completion": result, "memory_observer": observer.summary(),
        "cost_only": True, "certified": False, "final_eval_label_prediction_metric_reads": 0}
    with OUTPUT.with_suffix(".probe.json").open("x", encoding="utf-8") as target:
        json.dump(summary, target, indent=2, allow_nan=False)
    print(json.dumps({"phase": "cost_worker_completion", "status": result["status"],
        "success_count": result["success_count"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return 0 if result["status"] == "complete" else 1


def supervise(plan_path, digest):
    plan = validate_plan(read_bound(plan_path, digest))
    paths = (OUTPUT, OUTPUT.with_suffix(".particles"), OUTPUT.with_suffix(".probe.json"),
             OUTPUT.with_suffix(".supervisor.jsonl"))
    if any(path.exists() for path in paths):
        raise FileExistsError("original cost attempt already exists; no retry or overwrite")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    job_name = "PIRC17-LIO-COST-"+digest[:24]
    with paths[-1].open("x", encoding="utf-8") as log:
        started = time.perf_counter()
        _emit(log, {"phase": "owned_cost_started", "plan_sha256": digest, "supervisor_pid": os.getpid(),
            "job_name": job_name, "started_at": datetime.now(timezone.utc).isoformat(),
            "outer_wall_seconds": plan["outer_wall_seconds"], "expected_run_count": 4})
        command = [sys.executable, "-u", "-m", "experiments.pirc17.lio_cost_probe", "--worker",
                   "--plan", str(plan_path), "--plan-sha256", digest]
        result = run_owned(command, timeout=plan["outer_wall_seconds"]-(time.perf_counter()-started), job_name=job_name)
        _emit(log, {"phase": "owned_cost_completion", "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds_including_supervision": time.perf_counter()-started, **result})
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
