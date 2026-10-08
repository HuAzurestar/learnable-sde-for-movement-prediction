"""Separate immutable suffix attempts; only owned closed interruptions resume.

Full child forecasts inherited after interruption differ from the original
particle prefix inside each forecast. Every attempt shares the child's pinned
caps; already completed parent resource charges remain separate references.
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

from . import particle_suffix_contract as science
from . import particle_suffix_maps as maps
from . import resume_journal as journal
from . import seed_resume_lineage as storage
from .precision_check import KEYS

native, admission = science.native, science.admission
VERSION = science.VERSION+"-lineage"
UNQUALIFIED = science.UNQUALIFIED
ROOT_FIELDS = {"schema_version", "source", "source_sha256", "plan", *UNQUALIFIED}
START_FIELDS = {"schema_version", "root_sha256", "index", "previous_closure_sha256", "started_at",
    "deadline_at", "supervisor_pid", "prior_elapsed_seconds", "cooperative_remaining_seconds",
    "outer_remaining_seconds", "inherited_count", "job_name", *UNQUALIFIED}
CLOSURE_FIELDS = {"schema_version", "root_sha256", "start_sha256", "index", "finished_at", "elapsed_seconds",
    "worker_returncode", "process_tree_closed", "timed_out", "interrupted", "supervisor_error",
    "source_sha256", "inventory", "journal_tip", "new_committed_count", "new_failure_count",
    "process_sha256", "continuable", "status", *UNQUALIFIED}


def source_hashes():
    return {**science.source_hashes(), **{name: native._hash(Path(__file__).with_name(name)) for name in
        ("particle_suffix_maps.py", "particle_suffix_lineage.py", "particle_suffix_execution.py", "particle_suffix_session.py")}}


def _unqualified(value):
    return (all(value.get(k) is False for k in UNQUALIFIED if k != "final_eval_label_prediction_metric_reads")
            and type(value.get("final_eval_label_prediction_metric_reads")) is int
            and value["final_eval_label_prediction_metric_reads"] == 0)


def resources(spec, spent):
    if not storage._seconds(spent):
        raise ValueError("finite cumulative suffix attempt charge required")
    return {"prior_elapsed_upper_bound_seconds": spent,
        "cooperative_remaining_seconds": spec["wall_seconds"]-spent,
        "outer_remaining_seconds": spec["outer_wall_seconds"]-spent, "outer_cap_enforced": False,
        "policy": "all suffix attempts share original child caps; completed parent charges are separate, never erased"}


def _attempts(directory):
    path = Path(directory)/"attempts"
    if not path.is_dir() or path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("suffix attempts must be a real local directory")
    return storage._attempts(directory)


def read_root(directory, digest):
    directory = Path(directory).resolve()
    root = admission.read_closed(directory/"root.json", digest)
    if set(root) != ROOT_FIELDS or root["schema_version"] != VERSION or not _unqualified(root):
        raise ValueError("invalid immutable suffix root")
    source = science.normalize(root["source"])
    if (source != root["source"] or root["source_sha256"] != source_hashes()
            or root["plan"] != science.load_plan(source["plan"], source["plan_sha256"])
            or directory != science.directory_for(source, root["plan"])):
        raise ValueError("suffix root path, sources or plan changed")
    return root


def initialize(*, source):
    # No scientific parent replay before the supervised attempt's timer.
    source = science.normalize(source)
    spec = science.load_plan(source["plan"], source["plan_sha256"])
    directory = science.directory_for(source, spec)
    if directory.exists():
        path = directory/"root.json"
        if not path.is_file():
            raise FileExistsError("unsealed suffix root reservation; never reopen or erase")
        digest = native._hash(path)
        root = read_root(directory, digest)
        if root["source"] != source:
            raise ValueError("cannot fork an existing suffix contract")
        return directory, digest, root
    root = {"schema_version": VERSION, "source": source, "source_sha256": source_hashes(),
            "plan": spec, **UNQUALIFIED}
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
            or not isinstance(start["job_name"], str) or not re.fullmatch(r"Local\\pirc17-suffix-[0-9a-f]{32}", start["job_name"])
            or storage._date(start["deadline_at"]) != storage._date(start["started_at"])
                +timedelta(seconds=start["outer_remaining_seconds"])):
        raise ValueError("suffix reservation changes ancestry, denominator or cumulative caps")


def _outcome(work, start, spec, process, elapsed, saved):
    # The immutable parent's helper checks only generic counts/charges/result
    # presence. Require this profile's schema/flags additionally, never relabel
    # a legacy result as a suffix execution.
    outcome = science.lineage._outcome(work, start, spec, process, elapsed, saved)
    if outcome["status"] == "computed_closed":
        result = admission.read_closed(work/"core-result.json", native._hash(work/"core-result.json"))
        if result.get("schema_version") != science.VERSION+"-execution" or not _unqualified(result):
            outcome.update(status="failed", continuable=False)
    return outcome


def history(directory, digest, *, current_index=None):
    directory = Path(directory).resolve()
    root = read_root(directory, digest)
    paths = _attempts(directory)
    if current_index is not None and (type(current_index) is not int or current_index != len(paths)-1):
        raise ValueError("worker must use the sole latest suffix reservation")
    spent, inherited, previous, closed = 0., 0, None, []
    for index, path in enumerate(paths):
        start_path, closure_path = path/"start.json", path/"closure.json"
        if not start_path.is_file():
            raise FileExistsError("unclosed suffix reservation; no duplicate launch")
        start_digest = native._hash(start_path)
        start = admission.read_closed(start_path, start_digest)
        _validate_start(start, root, digest, index, previous, spent, inherited)
        if index == current_index:
            if closure_path.exists():
                raise FileExistsError("suffix attempt already closed")
            return root, closed, start, start_digest
        if not closure_path.is_file():
            raise FileExistsError("unclosed suffix reservation; no duplicate launch")
        closure_digest = native._hash(closure_path)
        closure = admission.read_closed(closure_path, closure_digest)
        if (set(closure) != CLOSURE_FIELDS or closure["schema_version"] != VERSION+"-closure"
                or not _unqualified(closure) or closure["root_sha256"] != digest
                or closure["start_sha256"] != start_digest or type(closure["index"]) is not int or closure["index"] != index
                or closure["source_sha256"] != root["source_sha256"] or not storage._seconds(closure["elapsed_seconds"])
                or any(type(closure[k]) is not int or closure[k] < 0 for k in ("new_committed_count", "new_failure_count"))
                or any(type(closure[k]) is not bool for k in ("process_tree_closed", "timed_out", "interrupted", "continuable"))
                or type(closure["worker_returncode"]) is not int
                or storage._date(closure["finished_at"]) < storage._date(start["started_at"])
                or abs((storage._date(closure["finished_at"])-storage._date(start["started_at"])).total_seconds()
                       -closure["elapsed_seconds"]) > 2
                or closure["inventory"] != storage.inventory(path/"work")):
            raise ValueError("suffix closure identity, elapsed time or inventory changed")
        process = storage._process(path/"work-process.json", closure["process_sha256"])
        if (any(process[k] != closure[k] for k in ("worker_returncode", "process_tree_closed", "timed_out", "interrupted", "supervisor_error"))
                or process.get("job_name") != start["job_name"] or process["elapsed_seconds"] > closure["elapsed_seconds"]):
            raise ValueError("suffix closure changed actual owned process facts")
        saved = storage._snapshot(path/"work", closure["journal_tip"])
        if (saved is None) != (closure["journal_tip"] is None):
            raise ValueError("closed suffix journal tip is missing or invented")
        if saved and (saved.manifest["inherited_count"] != inherited
                      or len(saved.manifest["workloads"]) != root["plan"]["expected_run_count"]):
            raise ValueError("suffix journal changes complete denominator or inherited forecast count")
        outcome = _outcome(path/"work", start, root["plan"], process, closure["elapsed_seconds"], saved)
        if any(closure[k] != v for k, v in outcome.items()) or (index < len(paths)-1 and not closure["continuable"]):
            raise ValueError("suffix closure outcome changed or noncontinuable attempt retried")
        closed.append({"index": index, "directory": path, "start": start, "start_sha256": start_digest,
                       "closure": closure, "closure_sha256": closure_digest, "saved": saved})
        previous = closure_digest
        spent = math.fsum(item["closure"]["elapsed_seconds"] for item in closed)
        inherited += closure["new_committed_count"]
    return root, closed, None, None


def reserve(*, source):
    begun, now = time.perf_counter(), datetime.now(timezone.utc)
    directory, digest, root = initialize(source=source)
    _, closed, _, _ = history(directory, digest)
    if closed and not closed[-1]["closure"]["continuable"]:
        raise ValueError("last suffix attempt is complete or noncontinuable; no automatic retry")
    spent = math.fsum(item["closure"]["elapsed_seconds"] for item in closed)
    remaining = resources(root["plan"], spent)
    if min(remaining["cooperative_remaining_seconds"], remaining["outer_remaining_seconds"]) <= 0:
        raise TimeoutError("original cumulative suffix budget exhausted; no renewal")
    index = len(closed)
    start = {"schema_version": VERSION+"-start", "root_sha256": digest, "index": index,
        "previous_closure_sha256": closed[-1]["closure_sha256"] if closed else None,
        "started_at": now.isoformat(), "deadline_at": (now+timedelta(seconds=remaining["outer_remaining_seconds"])).isoformat(),
        "supervisor_pid": os.getpid(), "prior_elapsed_seconds": spent,
        "job_name": "Local\\pirc17-suffix-"+uuid4().hex,
        "cooperative_remaining_seconds": remaining["cooperative_remaining_seconds"],
        "outer_remaining_seconds": remaining["outer_remaining_seconds"],
        "inherited_count": sum(c["closure"]["new_committed_count"] for c in closed), **UNQUALIFIED}
    attempt = directory/"attempts"/f"{index:06d}"
    attempt.mkdir()
    digest_start = journal._publish_json(attempt/"start.json", start)
    return {"directory": directory, "root_sha256": digest, "index": index, "start_sha256": digest_start,
            "attempt": attempt, "start": start, "started_monotonic": begun}


def seal(reservation):
    """Bind an actually closed owned-process record; JSON alone cannot prove OS death."""
    directory, digest, index = (reservation[k] for k in ("directory", "root_sha256", "index"))
    root, _, start, start_digest = history(directory, digest, current_index=index)
    if start_digest != reservation["start_sha256"]:
        raise ValueError("suffix reservation digest changed")
    process_path = reservation["attempt"]/"work-process.json"
    process_digest = native._hash(process_path)
    process = storage._process(process_path, process_digest)
    if process.get("job_name") != start["job_name"]:
        raise ValueError("suffix closure must bind the reserved owned job")
    work = reservation["attempt"]/"work"
    saved, files = storage._snapshot(work), storage.inventory(work)
    if files != storage.inventory(work) or source_hashes() != root["source_sha256"]:
        raise ValueError("suffix code or evidence changed during terminal inventory")
    elapsed = time.perf_counter()-reservation["started_monotonic"]
    if not storage._seconds(elapsed) or elapsed < process["elapsed_seconds"]:
        raise ValueError("suffix closure cannot undercharge the actual worker")
    closure = {"schema_version": VERSION+"-closure", "root_sha256": digest, "start_sha256": start_digest,
        "index": index, "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
        **{k: process[k] for k in ("worker_returncode", "process_tree_closed", "timed_out", "interrupted", "supervisor_error")},
        "source_sha256": root["source_sha256"], "inventory": files, "journal_tip": saved.tip if saved else None,
        "process_sha256": process_digest, **_outcome(work, start, root["plan"], process, elapsed, saved), **UNQUALIFIED}
    journal._publish_json(reservation["attempt"]/"closure.json", closure)
    return closure


def journal_contract(root, digest, index, start_digest, prepared, registered):
    return {"schema_version": VERSION+"-journal-contract", "root_sha256": digest, "index": index,
        "start_sha256": start_digest, "source_sha256": root["source_sha256"],
        "scientific_header": prepared.scientific_header, "map_catalog_sha256": science._digest(registered),
        "parent_whole_science_sha256": prepared.spec["parent_whole_science_sha256"],
        "reference_resource_charges": prepared.reference_resource_charges, "observer": native.physical_memory.identity()}


def inherit(prepared, directory, digest, *, registered, current_index=None, guard=lambda: None):
    root, closed, start, start_digest = history(directory, digest, current_index=current_index)
    if root["source"] != prepared.source or root["plan"] != prepared.spec:
        raise ValueError("fresh parent admission differs from the suffix root")
    rows, folders, refs = [], [], []
    for item in closed:
        guard()
        saved = item["saved"]
        if saved:
            if (saved.manifest["workloads"] != [dict(zip(KEYS, k)) for k in prepared.keys]
                    or saved.manifest["ancestry"] != {"prior_closures": refs}
                    or saved.manifest["contract"] != journal_contract(root, digest, item["index"], item["start_sha256"], prepared, registered)):
                raise ValueError("suffix ancestor changes ordered ancestry or scientific contract")
            added = [deepcopy(record["row"]) for record in saved.records]
            added_folders = [item["directory"]/"work/journal"]*len(added)
            science.validate_rows(prepared, added, added_folders, offset=len(rows), guard=guard)
            maps.verify_attempt(item["directory"]/"work", saved, root_sha256=digest,
                start_sha256=item["start_sha256"], registered=registered,
                data_root=prepared.parent_root["data_locations"]["data_root"], guard=guard)
            rows.extend(added)
            folders.extend(added_folders)
        refs.append({"index": item["index"], "closure_sha256": item["closure_sha256"]})
    if start is not None and start["inherited_count"] != len(rows):
        raise ValueError("suffix reservation differs from independently replayed inherited forecasts")
    prepared.recheck(guard)
    return rows, folders, {"prior_closures": refs}, start, start_digest
