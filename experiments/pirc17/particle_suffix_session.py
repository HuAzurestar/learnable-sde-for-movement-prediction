"""Owned Windows suffix launch/stop and independent complete scientific replay.

Private non-breakaway kill-on-close jobs are reused unchanged. An unclosed
reservation is never treated as proof of process death or permission to retry.
No inspection here accepts final scientific efficacy, training or resources.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from . import particle_suffix_execution as execution
from . import particle_suffix_lineage as lineage
from .seed_resume_session import require_owned_job, run_owned

science, maps = execution.science, execution.maps
VERSION = science.VERSION+"-session"


def _stop_requested(reservation):
    path = reservation["attempt"]/"stop.json"
    if not path.exists():
        return False
    value = lineage.admission.read_closed(path, lineage.native._hash(path))
    if value != {"schema_version": VERSION+"-stop", "start_sha256": reservation["start_sha256"]}:
        raise ValueError("stop request must bind the exact suffix attempt")
    return True


def request_stop(directory, root_sha256):
    directory = Path(directory).resolve()
    paths = lineage._attempts(directory)
    if not paths:
        raise ValueError("no suffix reservation to stop")
    _, _, _, digest = lineage.history(directory, root_sha256, current_index=len(paths)-1)
    lineage.journal._publish_json(paths[-1]/"stop.json", {"schema_version": VERSION+"-stop", "start_sha256": digest})


def supervise(*, source):
    reservation = lineage.reserve(source=source)
    root = lineage.read_root(reservation["directory"], reservation["root_sha256"])
    if root["source_sha256"].get("particle_suffix_session.py") != lineage.native._hash(Path(__file__)):
        raise ValueError("suffix root must bind this supervisor before launch")
    command = [sys.executable, "-u", "-m", "experiments.pirc17.particle_suffix_session", "worker",
        "--directory", str(reservation["directory"]), "--root-sha256", reservation["root_sha256"],
        "--index", str(reservation["index"]), "--start-sha256", reservation["start_sha256"]]
    result = run_owned(command,
        timeout=reservation["start"]["outer_remaining_seconds"]-(time.perf_counter()-reservation["started_monotonic"]),
        stop_requested=lambda: _stop_requested(reservation), job_name=reservation["start"]["job_name"])
    lineage.journal._publish_json(reservation["attempt"]/"work-process.json", result)
    closure = lineage.seal(reservation)
    print(json.dumps({"phase": "particle_suffix_supervisor_closed", "directory": str(reservation["directory"]),
        "root_sha256": reservation["root_sha256"], "index": reservation["index"],
        "closure_sha256": lineage.native._hash(reservation["attempt"]/"closure.json"),
        **{k: closure[k] for k in ("status", "new_committed_count", "new_failure_count", "continuable", "process_tree_closed", "elapsed_seconds")}}), flush=True)
    return 0 if closure["status"] == "computed_closed" else 124 if closure["timed_out"] else 130 if closure["interrupted"] else 1


def worker(directory, root_sha256, index, start_sha256):
    _, _, start, actual = lineage.history(directory, root_sha256, current_index=index)
    if actual != start_sha256:
        raise ValueError("suffix worker reservation digest changed")
    require_owned_job(start["job_name"])
    result = execution.run(directory=directory, root_sha256=root_sha256, index=index, start_sha256=start_sha256,
        progress=lambda event: print(json.dumps(event), flush=True))
    return 0 if result["status"] == "computed_pending_supervisory_closure" else 1


def inspect_closed(directory, root_sha256):
    directory = Path(directory).resolve()
    root, closed, _, _ = lineage.history(directory, root_sha256)
    if not closed or closed[-1]["closure"]["status"] != "computed_closed":
        raise ValueError("whole suffix inspection requires complete owned closure")
    before = lineage.storage.inventory(directory)
    observer = lineage.native.MemoryObserver()
    prepared = science.prepare(source=root["source"], guard=observer.guard)
    registered = maps.catalog(prepared, guard=observer.guard)
    originals, folders, _, _, _ = lineage.inherit(prepared, directory, root_sha256, registered=registered, guard=observer.guard)
    execution.load_inputs(prepared, guard=observer.guard)
    last, expected = closed[-1], len(prepared.keys)
    work, saved = last["directory"]/"work", last["saved"]
    if saved is None or len(originals) != expected:
        raise ValueError("closed suffix is missing registered scientific forecasts")
    result = lineage.admission.read_closed(work/"core-result.json", lineage.native._hash(work/"core-result.json"))
    inherited = lineage.admission.read_closed(work/"inherited.json", lineage.native._hash(work/"inherited.json"))
    count = saved.manifest["inherited_count"]
    if (set(inherited) != {"root_sha256", "start_sha256", "rows"} or inherited["root_sha256"] != root_sha256
            or inherited["start_sha256"] != last["start_sha256"] or len(inherited["rows"]) != count):
        raise ValueError("suffix materialization changes inherited root, reservation or count")
    rows = execution.assembled(work, saved)
    if len(rows) != expected:
        raise ValueError("materialized suffix differs from the complete denominator")
    for index, (row, original, folder) in enumerate(zip(rows, originals, folders)):
        observer.guard()
        intended = deepcopy(original)
        intended["particle_artifact"]["path"] = f"inherited/{index:06d}.npz" if index < count else f"journal/particles/{index:06d}.npz"
        if row != intended:
            raise ValueError("suffix materialization changed original ordered scientific ancestry")
        execution.load_particle_evidence(row, work)
        execution.load_particle_evidence(original, folder)
    if (set(result) != execution.RESULT_FIELDS or result["schema_version"] != execution.VERSION
            or result["status"] != "computed_pending_supervisory_closure"
            or result["expected_run_count"] != expected or result["inherited_success_count"] != count
            or result["new_committed_count"] != len(saved.records) or result["new_failure_count"] != 0
            or result["missing_completed_record_count"] != 0 or result["errors"]
            or result["journal_tip"] != saved.tip or result["uncommitted_files"] != saved.uncommitted_files
            or result["resources"] != lineage.resources(prepared.spec, last["start"]["prior_elapsed_seconds"])
            or result["reference_resource_charges"] != prepared.reference_resource_charges
            or result["map_catalog_sha256"] != science._digest(registered) or not lineage._unqualified(result)
            or not lineage.storage._seconds(result["elapsed_seconds"]) or result["elapsed_seconds"] > last["closure"]["elapsed_seconds"]):
        raise ValueError("suffix result changes counts, charges, evidence or qualification boundaries")
    memory = result["observer"]
    if (type(memory.get("calls")) is not int or memory["calls"] <= 0 or memory.get("failed_calls") != 0
            or type(memory.get("minimum_observed_available_bytes")) is not int
            or memory["minimum_observed_available_bytes"] < prepared.spec["minimum_free_bytes"]
            or memory.get("measurement_cache") is not False
            or science.science.runtime_identity() != prepared.original_reference[0]["runtime"]):
        raise ValueError("suffix runtime or fresh-memory observations differ")
    data_root = prepared.parent_root["data_locations"]["data_root"]
    observation_report = maps.verify_attempt(work, saved, root_sha256=root_sha256, start_sha256=last["start_sha256"],
        registered=registered, data_root=data_root, guard=observer.guard)
    if result["map_observations"] != observation_report:
        raise ValueError("suffix map observation summary differs from durable records")
    nonbase = [r for r in saved.records if r["row"]["configuration"] != "base"]
    final_maps = None
    if nonbase:
        path = work/"map-observations"/f"{nonbase[-1]['index']:06d}.json"
        final_maps = lineage.admission.read_closed(path, lineage.native._hash(path))["map_identity"]
    if result["new_attempt_maps"] != final_maps:
        raise ValueError("suffix result invents or changes current-attempt map use")
    whole = science.whole_audit(prepared, rows, work, guard=observer.guard)
    stored = lineage.admission.read_closed(work/"whole-science.json", result["whole_science_sha256"])
    if whole != stored:
        raise ValueError("whole suffix science does not independently reproduce")
    # Recheck new assets from EVERY attempt, including ones outside the original
    # parent's visited set and assets used only by an earlier interruption.
    for item in closed:
        if item["saved"]:
            maps.verify_attempt(item["directory"]/"work", item["saved"], root_sha256=root_sha256,
                start_sha256=item["start_sha256"], registered=registered, data_root=data_root, guard=observer.guard)
    prepared.recheck(observer.guard)
    if before != lineage.storage.inventory(directory) or lineage.source_hashes() != root["source_sha256"]:
        raise ValueError("suffix evidence changed during independent whole replay")
    return {"schema_version": VERSION+"-inspection", "status": "verified_closed_whole_workload",
        "root_sha256": root_sha256, "last_closure_sha256": last["closure_sha256"], "attempt_count": len(closed),
        "expected_run_count": expected, "new_committed_count": sum(c["closure"]["new_committed_count"] for c in closed),
        "parent_runs_covered": whole["parent_runs_covered"], "reference_runs_covered": whole["reference_runs_covered"],
        "exact_parent_prefix_checks": whole["exact_parent_prefix_checks"],
        "total_charged_elapsed_seconds": lineage.math.fsum(c["closure"]["elapsed_seconds"] for c in closed),
        "reference_resource_charges": prepared.reference_resource_charges,
        "whole_science_sha256": result["whole_science_sha256"], **science.UNQUALIFIED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    launch = modes.add_parser("run")
    for name in sorted(science.SOURCE_FIELDS):
        launch.add_argument("--"+name.replace("_", "-"), required=True)
    work = modes.add_parser("worker", help=argparse.SUPPRESS)
    stop, inspect = modes.add_parser("stop"), modes.add_parser("inspect")
    for sub in (work, stop, inspect):
        sub.add_argument("--directory", type=Path, required=True)
        sub.add_argument("--root-sha256", required=True)
    work.add_argument("--index", type=int, required=True)
    work.add_argument("--start-sha256", required=True)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    if mode == "worker":
        return worker(**args)
    if mode == "stop":
        request_stop(**args)
        return 0
    if mode == "inspect":
        print(json.dumps(inspect_closed(**args)), flush=True)
        return 0
    return supervise(source=args)


if __name__ == "__main__":
    raise SystemExit(main())
