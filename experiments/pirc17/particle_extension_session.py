"""Owned Windows missing-particle run/stop and independent whole inspection.

Every process belongs to a private non-breakaway kill-on-close Job Object.
This profile is distinct from the unchanged native and seed-resume profiles.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from . import particle_extension_execution as execution
from . import particle_extension_lineage as lineage
from .seed_resume_session import require_owned_job, run_owned

VERSION = execution.science.VERSION+"-session"


def _stop_requested(reservation):
    path = reservation["attempt"]/"stop.json"
    if not path.exists():
        return False
    request = lineage.admission.read_closed(path,lineage.native._hash(path))
    if request != {"schema_version":VERSION+"-stop","start_sha256":reservation["start_sha256"]}:
        raise ValueError("stop request must bind the exact extension attempt")
    return True


def request_stop(directory,root_sha256):
    """Request a stop; only subsequent owned closure proves it actually stopped."""
    directory=Path(directory).resolve()
    paths=lineage._attempts(directory)
    if not paths:
        raise ValueError("no extension reservation to stop")
    _,_,_,digest=lineage.history(directory,root_sha256,current_index=len(paths)-1)
    lineage.journal._publish_json(paths[-1]/"stop.json",{"schema_version":VERSION+"-stop","start_sha256":digest})


def supervise(*,source,locations):
    reservation=lineage.reserve(source=source,locations=locations)
    root=lineage.read_root(reservation["directory"],reservation["root_sha256"])
    if root["source_sha256"].get("particle_extension_session.py") != lineage.native._hash(Path(__file__)):
        raise ValueError("extension root must bind its supervising session before launch")
    start=reservation["start"]
    command=[sys.executable,"-u","-m","experiments.pirc17.particle_extension_session","worker",
        "--directory",str(reservation["directory"]),"--root-sha256",reservation["root_sha256"],
        "--index",str(reservation["index"]),"--start-sha256",reservation["start_sha256"]]
    outcome=run_owned(command,timeout=start["outer_remaining_seconds"]-(time.perf_counter()-reservation["started_monotonic"]),
        stop_requested=lambda:_stop_requested(reservation),job_name=start["job_name"])
    lineage.journal._publish_json(reservation["attempt"]/"work-process.json",outcome)
    closure=lineage.seal(reservation)
    print(json.dumps({"phase":"particle_extension_supervisor_closed","directory":str(reservation["directory"]),
        "root_sha256":reservation["root_sha256"],"index":reservation["index"],
        "closure_sha256":lineage.native._hash(reservation["attempt"]/"closure.json"),
        **{k:closure[k] for k in ("status","new_committed_count","new_failure_count","continuable",
                                 "process_tree_closed","elapsed_seconds")}}),flush=True)
    return 0 if closure["status"]=="computed_closed" else 124 if closure["timed_out"] else 130 if closure["interrupted"] else 1


def worker(directory,root_sha256,index,start_sha256):
    _,_,start,actual=lineage.history(directory,root_sha256,current_index=index)
    if actual != start_sha256:
        raise ValueError("worker reservation digest changed")
    require_owned_job(start["job_name"])
    result=execution.run(directory=directory,root_sha256=root_sha256,index=index,start_sha256=start_sha256,
        progress=lambda event:print(json.dumps(event),flush=True))
    return 0 if result["status"]=="computed_pending_supervisory_closure" else 1


def inspect_closed(directory,root_sha256):
    """Replay complete original/new evidence; never certify scientific efficacy."""
    directory=Path(directory).resolve()
    root,closed,_,_=lineage.history(directory,root_sha256)
    if not closed or closed[-1]["closure"]["status"]!="computed_closed":
        raise ValueError("whole extension inspection requires complete owned closure")
    before=lineage.storage.inventory(directory)
    observer=lineage.native.MemoryObserver()
    prepared=execution.science.prepare(source=root["source"],guard=observer.guard)
    originals,folders,_,_,_=lineage.inherit(prepared,directory,root_sha256,guard=observer.guard)
    last=closed[-1]
    work,saved=last["directory"]/"work",last["saved"]
    if saved is None or len(originals)!=prepared.spec["expected_run_count"]:
        raise ValueError("closed extension is missing registered scientific records")
    result_path=work/"core-result.json"
    result=lineage.admission.read_closed(result_path,lineage.native._hash(result_path))
    inherited=lineage.admission.read_closed(work/"inherited.json",lineage.native._hash(work/"inherited.json"))
    count=saved.manifest["inherited_count"]
    if (set(inherited)!={"root_sha256","start_sha256","rows"} or inherited["root_sha256"]!=root_sha256
            or inherited["start_sha256"]!=last["start_sha256"] or len(inherited["rows"])!=count):
        raise ValueError("inherited materialization changes bound root/reservation/count")
    rows=execution.assembled(work,saved)
    if len(rows)!=len(originals):
        raise ValueError("materialized extension differs from the complete ordered workload")
    for index,(row,original,folder) in enumerate(zip(rows,originals,folders)):
        observer.guard()
        expected=deepcopy(original)
        expected["particle_artifact"]["path"]=(f"inherited/{index:06d}.npz" if index<count else f"journal/particles/{index:06d}.npz")
        if row!=expected:
            raise ValueError("assembled result changed original ordered scientific ancestry")
        execution.load_particle_evidence(row,work)
        execution.load_particle_evidence(original,folder)
    if (result["schema_version"]!=execution.VERSION or result["status"]!="computed_pending_supervisory_closure"
            or result["expected_run_count"]!=len(rows) or result["inherited_success_count"]!=count
            or result["new_committed_count"]!=len(saved.records) or result["new_failure_count"]!=0
            or result["missing_completed_record_count"]!=0 or result["errors"]
            or result["journal_tip"]!=saved.tip or result["uncommitted_files"]!=saved.uncommitted_files
            or result["resources"]!=lineage.resources(prepared.spec,last["start"]["prior_elapsed_seconds"])
            or result["production_resume_ready"] is not False or not lineage._unqualified(result)
            or not lineage.storage._seconds(result["elapsed_seconds"])
            or result["elapsed_seconds"]>last["closure"]["elapsed_seconds"]):
        raise ValueError("core result changed scientific counts, resource charge or certification boundary")
    memory=result["observer"]
    if (type(memory.get("calls")) is not int or memory["calls"]<=0 or memory.get("failed_calls")!=0
            or type(memory.get("minimum_observed_available_bytes")) is not int
            or memory["minimum_observed_available_bytes"]<prepared.spec["minimum_free_bytes"]
            or memory.get("measurement_cache") is not False or result["new_attempt_maps"] is None
            or execution.science.runtime_identity()!=prepared.reference[0]["runtime"]):
        raise ValueError("extension runtime, maps or fresh-memory observations differ")
    maps=result["new_attempt_maps"]
    assets={Path(p):sha for p,sha in maps.get("receipt_sha256",{}).items()}
    assets.update({Path(root["data_locations"]["data_root"])/p:sha for p,sha in maps.get("verified_assets",{}).items()})
    if any(lineage.native._hash(p)!=sha for p,sha in assets.items()):
        raise ValueError("extension map receipt or asset changed")
    whole=execution.science.whole_audit(prepared,rows,work,guard=observer.guard)
    stored=lineage.admission.read_closed(work/"whole-science.json",result["whole_science_sha256"])
    if whole!=stored:
        raise ValueError("whole extension scientific audit does not independently reproduce")
    if before!=lineage.storage.inventory(directory) or lineage.source_hashes()!=root["source_sha256"]:
        raise ValueError("extension evidence changed during independent whole replay")
    prepared.recheck(observer.guard)
    if any(lineage.native._hash(p)!=sha for p,sha in assets.items()):
        raise ValueError("extension map receipt or asset changed during whole replay")
    return {"schema_version":VERSION+"-inspection","status":"verified_closed_whole_workload",
        "root_sha256":root_sha256,"last_closure_sha256":last["closure_sha256"],"attempt_count":len(closed),
        "expected_run_count":len(rows),"new_committed_count":sum(c["closure"]["new_committed_count"] for c in closed),
        "reference_runs_covered":whole["reference_runs_covered"],"exact_reference_prefix_checks":whole["exact_reference_prefix_checks"],
        "total_charged_elapsed_seconds":lineage.math.fsum(c["closure"]["elapsed_seconds"] for c in closed),
        "whole_science_sha256":result["whole_science_sha256"],"reference_numerical":whole["reference_numerical"],
        "resource_certified":False,**lineage.UNQUALIFIED}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest="mode",required=True)
    launch=modes.add_parser("run")
    for name in sorted(execution.science.SOURCE_ARGUMENTS|lineage.LOCATION_ARGUMENTS):
        launch.add_argument("--"+name.replace("_","-"),required=True)
    work=modes.add_parser("worker",help=argparse.SUPPRESS)
    stop=modes.add_parser("stop")
    inspect=modes.add_parser("inspect")
    for sub in (work,stop,inspect):
        sub.add_argument("--directory",type=Path,required=True)
        sub.add_argument("--root-sha256",required=True)
    work.add_argument("--index",type=int,required=True)
    work.add_argument("--start-sha256",required=True)
    args=vars(parser.parse_args())
    mode=args.pop("mode")
    if mode=="worker":return worker(**args)
    if mode=="stop":
        request_stop(**args)
        return 0
    if mode=="inspect":
        print(json.dumps(inspect_closed(**args)),flush=True)
        return 0
    return supervise(source={k:args[k] for k in execution.science.SOURCE_ARGUMENTS},
                     locations={k:args[k] for k in lineage.LOCATION_ARGUMENTS})


if __name__=="__main__":
    raise SystemExit(main())
