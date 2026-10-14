"""Ordered missing-particle attempts, with immutable ancestry and no renewed caps.

This is storage/admission, not a worker or a launch CLI. Whole scientific
reference replay belongs inside the bounded worker. Only a supervising owner
may seal the actual closed process record; a stale reservation never proves
that its worker died. Existing native and seed-resume evidence is untouched.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from . import particle_extension as science
from . import resume_journal as journal
from . import seed_resume_lineage as storage
from .precision_check import KEYS

native, admission = science.native, science.admission
VERSION = science.VERSION+"-lineage"
LOCATION_ARGUMENTS = {"eligibility", "release", "snapshot", "data_root"}
UNQUALIFIED = science.UNQUALIFIED
ROOT_FIELDS = {"schema_version", "source", "data_locations", "source_sha256", "plan", *UNQUALIFIED}
START_FIELDS = {"schema_version", "root_sha256", "index", "previous_closure_sha256", "started_at",
    "deadline_at", "supervisor_pid", "prior_elapsed_seconds", "cooperative_remaining_seconds",
    "outer_remaining_seconds", "inherited_count", "job_name", *UNQUALIFIED}
CLOSURE_FIELDS = {"schema_version", "root_sha256", "start_sha256", "index", "finished_at", "elapsed_seconds",
    "worker_returncode", "process_tree_closed", "timed_out", "interrupted", "supervisor_error",
    "source_sha256", "inventory", "journal_tip", "new_committed_count", "new_failure_count",
    "process_sha256", "continuable", "status", *UNQUALIFIED}


def source_hashes():
    # Generic inventory/process/journal helpers are reused without changing any
    # old profile's schemas or source bytes. Their entire dependency set stays
    # pinned too, including the current worker and supervising CLI.
    return {**storage.source_hashes(), **science.source_hashes(),
            **{name:native._hash(Path(__file__).with_name(name)) for name in
               ("particle_extension_lineage.py", "particle_extension_execution.py", "particle_extension_session.py")}}


def _unqualified(value):
    return (all(value.get(k) is False for k in UNQUALIFIED if k != "final_eval_label_prediction_metric_reads")
        and type(value.get("final_eval_label_prediction_metric_reads")) is int
        and value["final_eval_label_prediction_metric_reads"] == 0)


def normalize(source, locations):
    if set(locations) != LOCATION_ARGUMENTS:
        raise ValueError("exact development-only data-location arguments required")
    return science.normalize(source), {k: str(Path(v).resolve()) for k, v in locations.items()}


def resources(spec, spent):
    if not storage._seconds(spent):
        raise ValueError("finite cumulative elapsed charge required")
    return {"prior_elapsed_upper_bound_seconds": spent,
        "cooperative_remaining_seconds": spec["wall_seconds"]-spent,
        "outer_remaining_seconds": spec["outer_wall_seconds"]-spent,
        "outer_cap_enforced": False,
        "policy": "all extension attempts share original caps; old completed reference is not re-executed"}


def _attempts(directory):
    path = Path(directory)/"attempts"
    if not path.is_dir() or path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("extension attempts must be a real local directory")
    return storage._attempts(directory)


def read_root(directory, digest):
    directory = Path(directory).resolve()
    root = admission.read_closed(directory/"root.json", digest)
    if set(root) != ROOT_FIELDS or root["schema_version"] != VERSION or not _unqualified(root):
        raise ValueError("invalid immutable particle-extension root")
    source, locations = normalize(root["source"], root["data_locations"])
    if (root["source"] != source or root["data_locations"] != locations
            or root["source_sha256"] != source_hashes()
            or root["plan"] != science.load_plan(source["plan"], source["plan_sha256"])
            or directory != science.directory_for(source, root["plan"])):
        raise ValueError("extension root path, code, locations or plan changed")
    return root


def initialize(*, source, locations):
    """Reserve a contract only; no unbounded scientific replay before the timer."""
    source, locations = normalize(source, locations)
    spec = science.load_plan(source["plan"], source["plan_sha256"])
    directory = science.directory_for(source, spec)
    if directory.exists():
        path = directory/"root.json"
        if not path.is_file():
            raise FileExistsError("unsealed root reservation; do not reopen or erase it")
        digest = native._hash(path)
        root = read_root(directory, digest)
        if root["source"] != source or root["data_locations"] != locations:
            raise ValueError("cannot fork an existing extension contract")
        return directory, digest, root
    root = {"schema_version": VERSION, "source": source, "data_locations": locations,
            "source_sha256": source_hashes(), "plan": spec, **UNQUALIFIED}
    directory.mkdir()
    (directory/"attempts").mkdir()
    digest = journal._publish_json(directory/"root.json", root)
    return directory, digest, root


def _validate_start(start, root, digest, index, previous, spent, inherited):
    spec = root["plan"]
    if (set(start) != START_FIELDS or start["schema_version"] != VERSION+"-start" or not _unqualified(start)
            or type(start["index"]) is not int or start["index"] != index
            or start["root_sha256"] != digest or start["previous_closure_sha256"] != previous
            or not storage._seconds(start["prior_elapsed_seconds"]) or start["prior_elapsed_seconds"] != spent
            or any(not storage._seconds(start[k]) or start[k] <= 0 for k in
                   ("cooperative_remaining_seconds", "outer_remaining_seconds"))
            or start["cooperative_remaining_seconds"] != spec["wall_seconds"]-spent
            or start["outer_remaining_seconds"] != spec["outer_wall_seconds"]-spent
            or type(start["inherited_count"]) is not int or start["inherited_count"] != inherited
            or not 0 <= inherited <= spec["expected_run_count"]
            or type(start["supervisor_pid"]) is not int or start["supervisor_pid"] <= 0
            or not isinstance(start["job_name"], str)
            or not re.fullmatch(r"Local\\pirc17-extension-[0-9a-f]{32}", start["job_name"])
            or storage._date(start["deadline_at"]) != storage._date(start["started_at"])
                +timedelta(seconds=start["outer_remaining_seconds"])):
        raise ValueError("extension reservation changes ancestry, denominator or cumulative caps")


def _outcome(work, start, spec, process, elapsed, saved):
    count = len(saved.records) if saved else 0
    failures = saved.failure_count if saved else 0
    path = work/"core-result.json"
    result = admission.read_closed(path, native._hash(path)) if path.is_file() else None
    within = elapsed < min(start["cooperative_remaining_seconds"], start["outer_remaining_seconds"])
    healthy = not process["timed_out"] and not process["supervisor_error"] and not failures
    complete = bool(within and healthy and not process["interrupted"] and process["worker_returncode"] == 0
        and result and result.get("status") == "computed_pending_supervisory_closure"
        and _unqualified(result) and not result.get("errors") and saved
        and all(type(result.get(k)) is int for k in ("expected_run_count", "inherited_success_count",
            "new_committed_count", "new_failure_count", "missing_completed_record_count"))
        and result.get("expected_run_count") == spec["expected_run_count"]
        and start["inherited_count"]+count == spec["expected_run_count"]
        and result.get("inherited_success_count") == start["inherited_count"]
        and result.get("new_committed_count") == count and result.get("new_failure_count") == 0
        and result.get("missing_completed_record_count") == 0 and result.get("journal_tip") == saved.tip
        and result.get("whole_science_sha256") == native._hash(work/"whole-science.json"))
    continuable = bool(within and healthy and not complete and process["interrupted"]
        and process["worker_returncode"] != 0 and result is None)
    return {"new_committed_count": count, "new_failure_count": failures,
        "status": "computed_closed" if complete else "interrupted" if continuable else "failed",
        "continuable": continuable}


def history(directory, digest, *, current_index=None):
    directory = Path(directory).resolve()
    root = read_root(directory, digest)
    paths = _attempts(directory)
    if current_index is not None and (type(current_index) is not int or current_index != len(paths)-1):
        raise ValueError("worker must use the sole latest extension reservation")
    spent, inherited, previous, closed = 0., 0, None, []
    for index, path in enumerate(paths):
        start_path, closure_path = path/"start.json", path/"closure.json"
        if not start_path.is_file():
            raise FileExistsError("unclosed attempt reservation; no duplicate launch")
        start_digest = native._hash(start_path)
        start = admission.read_closed(start_path, start_digest)
        _validate_start(start, root, digest, index, previous, spent, inherited)
        if index == current_index:
            if closure_path.exists():
                raise FileExistsError("extension attempt already closed")
            return root, closed, start, start_digest
        if not closure_path.is_file():
            raise FileExistsError("unclosed attempt reservation; no duplicate launch")
        closure_digest = native._hash(closure_path)
        closure = admission.read_closed(closure_path, closure_digest)
        if (set(closure) != CLOSURE_FIELDS or closure["schema_version"] != VERSION+"-closure"
                or not _unqualified(closure) or closure["root_sha256"] != digest
                or closure["start_sha256"] != start_digest or type(closure["index"]) is not int
                or closure["index"] != index or closure["source_sha256"] != root["source_sha256"]
                or not storage._seconds(closure["elapsed_seconds"])
                or any(type(closure[k]) is not int or closure[k] < 0 for k in ("new_committed_count", "new_failure_count"))
                or any(type(closure[k]) is not bool for k in ("process_tree_closed", "timed_out", "interrupted", "continuable"))
                or type(closure["worker_returncode"]) is not int
                or storage._date(closure["finished_at"]) < storage._date(start["started_at"])
                or abs((storage._date(closure["finished_at"])-storage._date(start["started_at"])).total_seconds()
                       -closure["elapsed_seconds"]) > 2
                or closure["inventory"] != storage.inventory(path/"work")):
            raise ValueError("extension closure identity, elapsed time or inventory changed")
        process = storage._process(path/"work-process.json", closure["process_sha256"])
        if (any(process[k] != closure[k] for k in ("worker_returncode", "process_tree_closed", "timed_out",
                                                  "interrupted", "supervisor_error"))
                or process.get("job_name") != start["job_name"] or process["elapsed_seconds"] > closure["elapsed_seconds"]):
            raise ValueError("extension closure changed actual owned process facts")
        saved = storage._snapshot(path/"work", closure["journal_tip"])
        if (saved is None) != (closure["journal_tip"] is None):
            raise ValueError("closed extension journal tip is missing or invented")
        if saved and (saved.manifest["inherited_count"] != inherited
                      or len(saved.manifest["workloads"]) != root["plan"]["expected_run_count"]):
            raise ValueError("closed extension journal changed full denominator or inherited count")
        expected = _outcome(path/"work", start, root["plan"], process, closure["elapsed_seconds"], saved)
        if any(closure[k] != v for k, v in expected.items()) or (index < len(paths)-1 and not closure["continuable"]):
            raise ValueError("extension closure outcome changed or noncontinuable attempt was retried")
        closed.append({"index": index, "directory": path, "start": start, "start_sha256": start_digest,
                       "closure": closure, "closure_sha256": closure_digest, "saved": saved})
        previous = closure_digest
        spent = math.fsum(item["closure"]["elapsed_seconds"] for item in closed)
        inherited += closure["new_committed_count"]
    return root, closed, None, None


def reserve(*, source, locations):
    directory, digest, root = initialize(source=source, locations=locations)
    _, closed, _, _ = history(directory, digest)
    if closed and not closed[-1]["closure"]["continuable"]:
        raise ValueError("last extension attempt is complete or noncontinuable; no automatic retry")
    spent = math.fsum(item["closure"]["elapsed_seconds"] for item in closed)
    remaining = resources(root["plan"], spent)
    if min(remaining["cooperative_remaining_seconds"], remaining["outer_remaining_seconds"]) <= 0:
        raise TimeoutError("original cumulative extension budget exhausted; no renewal")
    index = len(closed)
    begun = time.perf_counter()
    now = datetime.now(timezone.utc)
    start = {"schema_version": VERSION+"-start", "root_sha256": digest, "index": index,
        "previous_closure_sha256": closed[-1]["closure_sha256"] if closed else None,
        "started_at": now.isoformat(), "deadline_at": (now+timedelta(seconds=remaining["outer_remaining_seconds"])).isoformat(),
        "supervisor_pid": os.getpid(), "prior_elapsed_seconds": spent,
        "job_name": "Local\\pirc17-extension-"+uuid4().hex,
        "cooperative_remaining_seconds": remaining["cooperative_remaining_seconds"],
        "outer_remaining_seconds": remaining["outer_remaining_seconds"],
        "inherited_count": sum(item["closure"]["new_committed_count"] for item in closed), **UNQUALIFIED}
    attempt = directory/"attempts"/f"{index:06d}"
    attempt.mkdir()  # Exactly one concurrent caller wins; no lock-file guessing.
    start_digest = journal._publish_json(attempt/"start.json", start)
    return {"directory": directory, "root_sha256": digest, "index": index, "start_sha256": start_digest,
            "attempt": attempt, "start": start, "started_monotonic": begun}


def seal(reservation):
    """Pin a previously published, actually closed owned-process record.

    The supervising caller must retain process ownership. This function cannot
    determine whether manually fabricated process JSON reflects a live OS job.
    Validation tests use explicitly synthetic closure records, never claim a
    real process experiment. Native supervision is a separate required layer.
    """
    directory, digest, index = (reservation[k] for k in ("directory", "root_sha256", "index"))
    root, _, start, start_digest = history(directory, digest, current_index=index)
    if start_digest != reservation["start_sha256"]:
        raise ValueError("extension reservation digest changed")
    process_path = reservation["attempt"]/"work-process.json"
    process_digest = native._hash(process_path)
    process = storage._process(process_path, process_digest)
    if process.get("job_name") != start["job_name"]:
        raise ValueError("closure must bind the reserved owned process job")
    work = reservation["attempt"]/"work"
    saved = storage._snapshot(work)
    files = storage.inventory(work)
    if files != storage.inventory(work) or source_hashes() != root["source_sha256"]:
        raise ValueError("extension evidence or code changed during terminal inventory")
    elapsed = time.perf_counter()-reservation["started_monotonic"]
    if not storage._seconds(elapsed) or elapsed < process["elapsed_seconds"]:
        raise ValueError("closure cannot undercharge the actual worker")
    outcome = _outcome(work, start, root["plan"], process, elapsed, saved)
    closure = {"schema_version": VERSION+"-closure", "root_sha256": digest, "start_sha256": start_digest,
        "index": index, "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
        **{k: process[k] for k in ("worker_returncode", "process_tree_closed", "timed_out", "interrupted", "supervisor_error")},
        "source_sha256": root["source_sha256"], "inventory": files, "journal_tip": saved.tip if saved else None,
        "process_sha256": process_digest, **outcome, **UNQUALIFIED}
    journal._publish_json(reservation["attempt"]/"closure.json", closure)
    return closure


def journal_contract(root, digest, index, start_digest, prepared):
    return {"schema_version": VERSION+"-journal-contract", "root_sha256": digest, "index": index,
        "start_sha256": start_digest, "source_sha256": root["source_sha256"],
        "scientific_header": prepared.scientific_header,
        "reference_numerical": science.numerical_summary(prepared.reference_report),
        "observer": native.physical_memory.identity()}


def inherit(prepared, directory, digest, *, current_index=None, guard=lambda: None):
    """Replay ALL committed new-particle ancestors; originals remain references."""
    root, closed, start, start_digest = history(directory, digest, current_index=current_index)
    if root["source"] != prepared.source or root["plan"] != prepared.spec:
        raise ValueError("fresh whole-reference admission differs from extension root")
    rows, directories, refs = [], [], []
    for item in closed:
        guard()
        saved = item["saved"]
        if saved:
            if (saved.manifest["workloads"] != [dict(zip(KEYS, key)) for key in prepared.keys]
                    or saved.manifest["ancestry"] != {"prior_closures": refs}
                    or saved.manifest["contract"] != journal_contract(root, digest, item["index"], item["start_sha256"], prepared)):
                raise ValueError("ancestor journal changes scientific contract, ordered ancestry or denominator")
            added = [deepcopy(record["row"]) for record in saved.records]
            folders = [str(item["directory"]/"work/journal")]*len(added)
            science.validate_rows(prepared, added, folders, offset=len(rows), guard=guard)
            rows.extend(added)
            directories.extend(folders)
        refs.append({"index": item["index"], "closure_sha256": item["closure_sha256"]})
    if start is not None and start["inherited_count"] != len(rows):
        raise ValueError("reserved count differs from scientifically verified inherited rows")
    prepared.recheck(guard)
    return rows, directories, {"prior_closures": refs}, start, start_digest
