"""One ordered, immutable continuation history per closed legacy envelope.

An unclosed reservation is a barrier, not permission to guess that its process
has died. Only an owned supervisor may seal an attempt after its whole process
tree is closed. Every attempt (including preflight failure) consumes budget.
This layer checks storage/ancestry; scientific inheritance is replayed separately.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os
import math
from pathlib import Path
import re
import time
from uuid import uuid4

from . import resume_journal as journal
from . import seed_resume as admission
from .precision_check import KEYS

native = admission.native
VERSION = "pirc17-seed-resume-lineage-v1"
SOURCE_ARGUMENTS = {"plan", "plan_sha256", "reference_ledger", "reference_audit", "fit", "fit_ledger",
                    "envelope", "envelope_sha256", "forecast_sha256", "supervisor_sha256"}
LOCATION_ARGUMENTS = {"eligibility", "release", "snapshot", "data_root"}
UNQUALIFIED = {"certified": False, "formal_training_accepted": False,
               "final_eval_label_prediction_metric_reads": 0}
ROOT_FIELDS = {"schema_version", "source", "data_locations", "source_sha256", "plan",
               "initialization", "header", "validated_prefix", *UNQUALIFIED}
START_FIELDS = {"schema_version", "root_sha256", "index", "previous_closure_sha256", "started_at",
                "deadline_at", "supervisor_pid", "prior_elapsed_seconds", "cooperative_remaining_seconds",
                "outer_remaining_seconds", "inherited_count", "job_name", *UNQUALIFIED}
CLOSURE_FIELDS = {"schema_version", "root_sha256", "start_sha256", "index", "finished_at", "elapsed_seconds",
    "worker_returncode", "process_tree_closed", "timed_out", "interrupted", "supervisor_error",
    "source_sha256", "inventory", "journal_tip", "new_committed_count", "new_failure_count",
    "process_sha256", "continuable", "status", *UNQUALIFIED}


def source_hashes():
    directory = Path(__file__).parent
    return {**admission.source_hashes(), **{name: native._hash(directory/name) for name in
        ("resume_journal.py", "seed_resume_execution.py", "seed_resume_lineage.py", "seed_resume_session.py")}}


def normalize(source, locations):
    if set(source) != SOURCE_ARGUMENTS or set(locations) != LOCATION_ARGUMENTS:
        raise ValueError("exact closed-source and data-location arguments required")
    return ({key: value if key.endswith("_sha256") else str(Path(value).resolve()) for key, value in source.items()},
            {key: str(Path(value).resolve()) for key, value in locations.items()})


def directory_for(source):
    return Path(source["envelope"]).resolve().with_suffix(".resume")


def _unqualified(value):
    return (all(value.get(key) is False for key in ("certified", "formal_training_accepted"))
            and type(value.get("final_eval_label_prediction_metric_reads")) is int
            and value["final_eval_label_prediction_metric_reads"] == 0)


def _seconds(value):
    return type(value) in (int, float) and 0 <= value < float("inf")


def _date(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timezone-aware attempt time required")
    return result


def inventory(directory):
    """Hash every file, including uncommitted orphans; refuse escaping links."""
    directory = Path(directory).resolve()
    if not directory.exists():
        return []
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or path.resolve() != path.absolute():
            raise ValueError("linked or escaping resume evidence is not admissible")
        if path.is_file():
            result.append({"path": path.relative_to(directory).as_posix(), "sha256": native._hash(path)})
        elif not path.is_dir():
            raise ValueError("nonregular resume evidence")
    return result


def read_root(directory, digest):
    directory = Path(directory).resolve()
    root = admission.read_closed(directory/"root.json", digest)
    if set(root) != ROOT_FIELDS or root["schema_version"] != VERSION or not _unqualified(root):
        raise ValueError("invalid immutable resume root")
    source, locations = normalize(root["source"], root["data_locations"])
    if (root["source"] != source or root["data_locations"] != locations or directory != directory_for(source)
            or root["source_sha256"] != source_hashes()
            or root["plan"] != native.load_plan(source["plan"], source["plan_sha256"])):
        raise ValueError("resume root path, sources, inputs or plan changed")
    return root


def initialize(*, source, locations):
    source, locations = normalize(source, locations)
    directory = directory_for(source)
    if directory.exists():
        path = directory/"root.json"
        if not path.is_file():
            raise FileExistsError("unsealed root reservation; do not reopen or erase it")
        digest = native._hash(path)
        root = read_root(directory, digest)
        if root["source"] != source or root["data_locations"] != locations:
            raise ValueError("cannot fork or reset the existing continuation contract")
        return directory, digest, root
    prefix = admission.inspect(**source)
    root = {"schema_version": VERSION, "source": source, "data_locations": locations,
            "source_sha256": source_hashes(), "plan": prefix.spec, "initialization": prefix.initialization,
            "header": prefix.header, "validated_prefix": prefix.report, **UNQUALIFIED}
    directory.mkdir()  # Concurrent initialization loses here, before any worker.
    (directory/"attempts").mkdir()
    digest = journal._publish_json(directory/"root.json", root)
    return directory, digest, root


def _attempts(directory):
    paths = sorted((directory/"attempts").iterdir())
    for index, path in enumerate(paths):
        if (not path.is_dir() or path.is_symlink() or path.name != f"{index:06d}"
                or path.resolve() != path.absolute()):
            raise ValueError("missing, reordered or linked resume attempt")
    return paths


def _legacy_elapsed(root):
    source = root["source"]
    old = admission.read_closed(native.output_paths(source["envelope"])["supervisor"], source["supervisor_sha256"])
    if not _seconds(old.get("elapsed_seconds")):
        raise ValueError("finite legacy resource charge required")
    return old["elapsed_seconds"]


def resources(spec, spent):
    return {"prior_elapsed_upper_bound_seconds": spent, "cooperative_remaining_seconds": spec["wall_seconds"]-spent,
        "outer_remaining_seconds": spec["outer_wall_seconds"]-spent, "outer_cap_enforced": False,
        "policy": "charge full prior supervisor elapsed against both original caps; no implicit renewal"}


def _validate_start(start, root, digest, index, previous, spent, inherited):
    spec = root["plan"]
    if (set(start) != START_FIELDS or start["schema_version"] != VERSION+"-start" or not _unqualified(start)
            or type(start["index"]) is not int or start["index"] != index
            or start["root_sha256"] != digest or start["previous_closure_sha256"] != previous
            or not _seconds(start["prior_elapsed_seconds"]) or start["prior_elapsed_seconds"] != spent
            or not _seconds(start["cooperative_remaining_seconds"]) or not _seconds(start["outer_remaining_seconds"])
            or start["cooperative_remaining_seconds"] != spec["wall_seconds"]-spent
            or start["outer_remaining_seconds"] != spec["outer_wall_seconds"]-spent
            or start["cooperative_remaining_seconds"] <= 0 or start["outer_remaining_seconds"] <= 0
            or type(start["inherited_count"]) is not int or start["inherited_count"] != inherited
            or type(start["supervisor_pid"]) is not int or start["supervisor_pid"] <= 0
            or not isinstance(start["job_name"], str) or not re.fullmatch(r"Local\\pirc17-resume-[0-9a-f]{32}", start["job_name"])
            or _date(start["deadline_at"]) != _date(start["started_at"])+timedelta(seconds=start["outer_remaining_seconds"])):
        raise ValueError("attempt reservation breaks ancestry, denominator or cumulative caps")


def _snapshot(work, expected_tip=None):
    path = work/"journal/manifest.json"
    if not path.is_file():
        if expected_tip is not None:
            raise ValueError("closed journal manifest disappeared")
        return None
    return journal.read(path.parent, native._hash(path), expected_tip=expected_tip)


def _process(path, digest):
    process = admission.read_closed(path, digest)
    if (process.get("process_tree_closed") is not True or type(process.get("worker_returncode")) is not int
            or type(process.get("worker_pid")) is not int or process["worker_pid"] <= 0
            or process.get("accounting", {}).get("active_processes") != 0
            or type(process.get("accounting", {}).get("active_processes")) is not int
            or any(type(process.get(key)) is not bool for key in ("timed_out", "interrupted"))
            or not _seconds(process.get("elapsed_seconds"))):
        raise ValueError("invalid owned job process closure")
    return process


def _outcome(work, start, *, elapsed, returncode, timed_out, interrupted, error, saved):
    count = len(saved.records) if saved else 0
    failures = saved.failure_count if saved else 0
    result_path = work/"core-result.json"
    result = admission.read_closed(result_path, native._hash(result_path)) if result_path.is_file() else None
    within = elapsed < start["outer_remaining_seconds"] and not timed_out
    complete = bool(within and returncode == 0 and not error and not interrupted and result
                    and result.get("status") == "computed_pending_supervisory_closure"
                    and not result.get("errors") and saved and result.get("journal_tip") == saved.tip
                    and result.get("whole_math_sha256") == native._hash(work/"whole-math.json")
                    and not failures and not result.get("missing_completed_record_count"))
    # A forced interruption may precede the journal or its result. Unknown
    # abnormal exits and core failures are retained but are never auto-retried.
    continuable = bool(within and not complete and not failures and not error and interrupted
                       and returncode != 0 and result is None)
    return {"new_committed_count": count, "new_failure_count": failures,
            "status": "computed_closed" if complete else "interrupted" if continuable else "failed",
            "continuable": continuable}


def history(directory, digest, *, current_index=None):
    """Validate every closed predecessor, optionally one final live reservation."""
    directory = Path(directory).resolve()
    root = read_root(directory, digest)
    paths = _attempts(directory)
    if current_index is not None and (type(current_index) is not int or current_index != len(paths)-1):
        raise ValueError("worker must use the sole latest reserved attempt")
    legacy_elapsed = _legacy_elapsed(root)
    spent = legacy_elapsed
    inherited = root["validated_prefix"]["reusable_success_count"]
    previous, closed = None, []
    for index, path in enumerate(paths):
        start_path = path/"start.json"
        if not start_path.is_file():
            raise FileExistsError("unclosed attempt reservation; no duplicate launch")
        start_digest = native._hash(start_path)
        start = admission.read_closed(start_path, start_digest)
        _validate_start(start, root, digest, index, previous, spent, inherited)
        closure_path = path/"closure.json"
        if index == current_index:
            if closure_path.exists():
                raise FileExistsError("reserved attempt is already closed")
            return root, closed, start, start_digest
        if not closure_path.is_file():
            raise FileExistsError("unclosed attempt reservation; no duplicate launch")
        closure_digest = native._hash(closure_path)
        closure = admission.read_closed(closure_path, closure_digest)
        if (set(closure) != CLOSURE_FIELDS or closure["schema_version"] != VERSION+"-closure"
                or not _unqualified(closure) or closure["root_sha256"] != digest
                or closure["start_sha256"] != start_digest or type(closure["index"]) is not int
                or closure["index"] != index or closure["process_tree_closed"] is not True
                or type(closure["worker_returncode"]) is not int or not _seconds(closure["elapsed_seconds"])
                or any(type(closure[key]) is not int or closure[key] < 0 for key in ("new_committed_count", "new_failure_count"))
                or any(type(closure[key]) is not bool for key in ("timed_out", "interrupted", "continuable"))
                or closure["source_sha256"] != root["source_sha256"]
                or _date(closure["finished_at"]) < _date(start["started_at"])
                or abs((_date(closure["finished_at"])-_date(start["started_at"])).total_seconds()-closure["elapsed_seconds"]) > 2
                or closure["inventory"] != inventory(path/"work")):
            raise ValueError("closed attempt identity, process closure, elapsed charge or inventory changed")
        saved = _snapshot(path/"work", closure["journal_tip"])
        process = _process(path/"work-process.json", closure["process_sha256"])
        if (any(process[key] != closure[key] for key in
                ("worker_returncode", "process_tree_closed", "timed_out", "interrupted", "supervisor_error"))
                or process.get("job_name") != start["job_name"]
                or process["elapsed_seconds"] > closure["elapsed_seconds"]):
            raise ValueError("closure changed the owned job's terminal facts")
        if (saved is None) != (closure["journal_tip"] is None):
            raise ValueError("closed journal tip is missing or invented")
        if saved and saved.manifest["inherited_count"] != inherited:
            raise ValueError("closed journal omitted inherited records")
        expected = _outcome(path/"work", start, elapsed=closure["elapsed_seconds"],
            returncode=closure["worker_returncode"], timed_out=closure["timed_out"],
            interrupted=closure["interrupted"], error=closure["supervisor_error"], saved=saved)
        if any(closure[key] != value for key, value in expected.items()):
            raise ValueError("closed attempt outcome or continuation disposition changed")
        if index < len(paths)-1 and not closure["continuable"]:
            raise ValueError("a completed or failed attempt cannot be silently retried")
        closed.append({"index": index, "directory": path, "start": start, "start_sha256": start_digest,
                       "closure": closure, "closure_sha256": closure_digest, "saved": saved})
        previous = closure_digest
        spent = math.fsum([legacy_elapsed, *[item["closure"]["elapsed_seconds"] for item in closed]])
        inherited += closure["new_committed_count"]
    return root, closed, None, None


def reserve(*, source, locations):
    directory, digest, root = initialize(source=source, locations=locations)
    _, closed, _, _ = history(directory, digest)
    if closed and not closed[-1]["closure"]["continuable"]:
        raise ValueError("last attempt is completed or noncontinuable; no automatic retry")
    spent = math.fsum([_legacy_elapsed(root), *[item["closure"]["elapsed_seconds"] for item in closed]])
    if spent >= min(root["plan"]["wall_seconds"], root["plan"]["outer_wall_seconds"]):
        raise TimeoutError("original cumulative budget exhausted; no renewal")
    index = len(closed)
    started_monotonic = time.perf_counter()
    now = datetime.now(timezone.utc)
    start = {"schema_version": VERSION+"-start", "root_sha256": digest, "index": index,
        "previous_closure_sha256": closed[-1]["closure_sha256"] if closed else None,
        "started_at": now.isoformat(), "deadline_at": (now+timedelta(seconds=root["plan"]["outer_wall_seconds"]-spent)).isoformat(),
        "supervisor_pid": os.getpid(), "prior_elapsed_seconds": spent,
        "job_name": "Local\\pirc17-resume-"+uuid4().hex,
        "cooperative_remaining_seconds": root["plan"]["wall_seconds"]-spent,
        "outer_remaining_seconds": root["plan"]["outer_wall_seconds"]-spent,
        "inherited_count": root["validated_prefix"]["reusable_success_count"]+sum(item["closure"]["new_committed_count"] for item in closed),
        **UNQUALIFIED}
    attempt = directory/"attempts"/f"{index:06d}"
    attempt.mkdir()  # Linearization point: exactly one concurrent caller wins.
    start_digest = journal._publish_json(attempt/"start.json", start)
    return {"directory": directory, "root_sha256": digest, "index": index, "start_sha256": start_digest,
            "attempt": attempt, "start": start, "started_monotonic": started_monotonic}


def seal(reservation, *, elapsed, returncode, tree_closed, timed_out=False, interrupted=False, error=None):
    """Called only after the supervisor has reaped its owned entire process tree."""
    if tree_closed is not True or type(returncode) is not int or not _seconds(elapsed):
        raise ValueError("confirmed whole process-tree closure and measured elapsed time required")
    directory, digest, index = (reservation[key] for key in ("directory", "root_sha256", "index"))
    root, _, start, start_digest = history(directory, digest, current_index=index)
    if start_digest != reservation["start_sha256"]:
        raise ValueError("attempt reservation changed")
    work = reservation["attempt"]/"work"
    process_digest = native._hash(reservation["attempt"]/"work-process.json")
    process = _process(reservation["attempt"]/"work-process.json", process_digest)
    if (process["worker_returncode"] != returncode or process["timed_out"] != timed_out
            or process["interrupted"] != interrupted or process["supervisor_error"] != error
            or process.get("job_name") != start["job_name"]
            or process["elapsed_seconds"] > elapsed):
        raise ValueError("seal must preserve actual owned job termination facts")
    saved = _snapshot(work)
    files = inventory(work)
    outcome = _outcome(work, start, elapsed=elapsed, returncode=returncode, timed_out=timed_out,
                       interrupted=interrupted, error=error, saved=saved)
    closure = {"schema_version": VERSION+"-closure", "root_sha256": digest, "start_sha256": start_digest,
        "index": index, "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
        "worker_returncode": returncode, "process_tree_closed": True, "timed_out": timed_out,
        "interrupted": interrupted, "supervisor_error": error, "source_sha256": source_hashes(),
        "inventory": files, "journal_tip": saved.tip if saved else None, "process_sha256": process_digest,
        **outcome, **UNQUALIFIED}
    if root["source_sha256"] != closure["source_sha256"] or files != inventory(work):
        raise ValueError("sources or closed evidence changed during terminal inventory")
    journal._publish_json(reservation["attempt"]/"closure.json", closure)
    return closure


def inherit(prefix, source, locations, directory, digest, index, reference, *, guard):
    """Replay every ancestor's scientific rows; never accept a caller-chosen tip."""
    root, closed, start, _ = history(directory, digest, current_index=index)
    source, locations = normalize(source, locations)
    if (root["source"] != source or root["data_locations"] != locations or root["plan"] != prefix.spec
            or root["initialization"] != prefix.initialization or root["header"] != prefix.header
            or root["validated_prefix"] != prefix.report):
        raise ValueError("current scientific admission differs from the immutable continuation root")
    rows = deepcopy(prefix.rows)
    directories = [str(native.output_paths(source["envelope"])["forecast"].parent)]*len(rows)
    refs = []
    for item in closed:
        guard()
        saved = item["saved"]
        if saved:
            from .seed_resume_execution import VERSION as execution_version
            ancestry, contract = saved.manifest["ancestry"], saved.manifest["contract"]
            expected_ancestry = {"legacy_arguments": source, "validated_prefix": prefix.report,
                "inherited_rows": rows, "inherited_directories": directories,
                "continuation": {"root_sha256": digest, "index": item["index"], "prior_closures": refs}}
            expected_contract = {"schema_version": execution_version, "source_sha256": root["source_sha256"],
                "plan": prefix.spec, "scientific_initialization": prefix.initialization,
                "scientific_header": prefix.header, "data_locations": locations,
                "resources": resources(prefix.spec, item["start"]["prior_elapsed_seconds"]),
                "observer": native.physical_memory.identity()}
            if (ancestry != expected_ancestry or contract != expected_contract
                    or saved.manifest["workloads"] != [dict(zip(KEYS, key)) for key in admission.workload_keys(prefix.spec, prefix.header["sample_ids"]) ]):
                raise ValueError("ancestor journal changed inherited prefix, scientific contract or whole denominator")
            new_rows = [deepcopy(record["row"]) for record in saved.records]
            new_directory = item["directory"]/"work/journal"
            admission.validate_prefix_rows(prefix.spec, reference, [prefix.initialization, prefix.header, *new_rows],
                new_directory, Path(source["reference_ledger"]).parent, guard=guard, offset=len(rows))
            rows.extend(new_rows)
            directories.extend([str(new_directory)]*len(new_rows))
        refs.append({"index": item["index"], "closure_sha256": item["closure_sha256"]})
    if start is not None and start["inherited_count"] != len(rows):
        raise ValueError("reserved prefix count differs from independently inherited rows")
    prefix.rows = rows
    keys = admission.workload_keys(prefix.spec, prefix.header["sample_ids"])
    prefix.remaining_keys = keys[len(rows):]
    return directories, {"root_sha256": digest, "index": index, "prior_closures": refs}, start
