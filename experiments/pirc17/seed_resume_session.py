"""Owned Windows seed continuation: reserve, run, close, independently inspect.

The small stdlib-only bootstrap cannot spawn the research worker until assigned
to a private kill-on-close Job Object. No breakaway is allowed. Completion means
zero active processes in that job, not merely a dead launcher PID.

Windows contract: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
ABI: https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information
Accounting: https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_accounting_information
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import seed_resume_lineage as lineage
from . import seed_resume_execution as execution

VERSION = "pirc17-seed-resume-session-v1"
BOOTSTRAP = (
    "import sys; token=sys.stdin.buffer.read(1); "
    "token == b'R' or sys.exit(125); "
    "import subprocess; sys.exit(subprocess.call(sys.argv[1:], "
    "stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW))"
)


class _Limits(ctypes.Structure):
    _fields_ = [("per_process_user", ctypes.c_int64), ("per_job_user", ctypes.c_int64),
        ("flags", ctypes.c_uint32), ("min_working_set", ctypes.c_size_t), ("max_working_set", ctypes.c_size_t),
        ("active_limit", ctypes.c_uint32), ("affinity", ctypes.c_size_t), ("priority", ctypes.c_uint32),
        ("scheduling", ctypes.c_uint32)]


class _Extended(ctypes.Structure):
    _fields_ = [("basic", _Limits), ("io", ctypes.c_uint64*6), ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [("user", ctypes.c_int64), ("kernel", ctypes.c_int64), ("period_user", ctypes.c_int64),
        ("period_kernel", ctypes.c_int64), ("page_faults", ctypes.c_uint32), ("total", ctypes.c_uint32),
        ("active", ctypes.c_uint32), ("terminated", ctypes.c_uint32)]


class OwnedJob:
    """A retained fresh job handle; never enumerate/terminate unrelated PIDs."""

    def __init__(self, name=None):
        if os.name != "nt":
            raise OSError("production seed continuation requires native Windows job supervision")
        self.api = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        pointer, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        for function_name, arguments, result in (
            ("CreateJobObjectW", [pointer, ctypes.c_wchar_p], pointer),
            ("SetInformationJobObject", [pointer, ctypes.c_int, pointer, dword], boolean),
            ("AssignProcessToJobObject", [pointer, pointer], boolean),
            ("QueryInformationJobObject", [pointer, ctypes.c_int, pointer, dword, pointer], boolean),
            ("TerminateJobObject", [pointer, dword], boolean), ("CloseHandle", [pointer], boolean)):
            function = getattr(self.api, function_name)
            function.argtypes, function.restype = arguments, result
        ctypes.set_last_error(0)
        self.handle = self.api.CreateJobObjectW(None, name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS: never adopt another job.
            self.close()
            raise FileExistsError("refusing to reuse an existing process job")
        limits = _Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
        try:
            self._check(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        self._check(self.api.AssignProcessToJobObject(self.handle, int(process._handle)))

    def accounting(self):
        state = _Accounting()
        self._check(self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(state), ctypes.sizeof(state), None))
        return {"total_processes": int(state.total), "active_processes": int(state.active),
                "user_seconds": state.user/10000000., "kernel_seconds": state.kernel/10000000.}

    def terminate(self):
        self._check(self.api.TerminateJobObject(self.handle, 130))

    def close(self):
        if self.handle:
            self._check(self.api.CloseHandle(self.handle))
            self.handle = None


def require_owned_job(name):
    """A direct call to the internal worker CLI is not an owned launch."""
    if os.name != "nt":
        raise OSError("native Windows process-job membership required")
    api = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    api.OpenJobObjectW.argtypes, api.OpenJobObjectW.restype = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_wchar_p], ctypes.c_void_p
    api.GetCurrentProcess.argtypes, api.GetCurrentProcess.restype = [], ctypes.c_void_p
    api.IsProcessInJob.argtypes, api.IsProcessInJob.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)], ctypes.c_int32
    api.CloseHandle.argtypes, api.CloseHandle.restype = [ctypes.c_void_p], ctypes.c_int32
    handle = api.OpenJobObjectW(4, False, name)  # JOB_OBJECT_QUERY only; not inherited.
    if not handle:
        raise PermissionError("worker must belong to its live reserved process job")
    try:
        member = ctypes.c_int32()
        if not api.IsProcessInJob(api.GetCurrentProcess(), handle, ctypes.byref(member)):
            raise ctypes.WinError(ctypes.get_last_error())
        if member.value != 1:
            raise PermissionError("worker must belong to its live reserved process job")
    finally:
        # Only the supervisor retains a job handle during scientific execution:
        # losing it must still trigger kill-on-close for the entire tree.
        if not api.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def run_owned(command, *, timeout, stop_requested=lambda: False, job_name=None):
    """Bound the entire descendant tree; the command seam is for software tests."""
    if not lineage._seconds(timeout) or timeout <= 0:
        raise TimeoutError("no remaining outer budget")
    begin = time.perf_counter()
    job, child, assigned = OwnedJob(job_name), None, False
    timed_out, interrupted, error = False, False, None
    try:
        # -I -S disables user/site startup code before the assignment handshake.
        # Use the base executable to avoid the venv launcher's pre-assignment child.
        child = subprocess.Popen([sys._base_executable, "-I", "-S", "-u", "-c", BOOTSTRAP, *command],
            stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(child)
        assigned = True
        child.stdin.write(b"R")
        child.stdin.flush()
        child.stdin.close()
        while child.poll() is None or job.accounting()["active_processes"]:
            if stop_requested():
                interrupted = True
                break
            if time.perf_counter()-begin >= timeout:
                timed_out = True
                break
            time.sleep(min(.1, max(0., timeout-(time.perf_counter()-begin))))
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as exc:
        error = {"error_type": type(exc).__name__, "error_message": str(exc)[:300]}
    finally:
        try:
            if assigned and job.accounting()["active_processes"]:
                job.terminate()
            elif child is not None and child.poll() is None:
                # The blocked bootstrap has spawned nothing if assignment failed.
                child.kill()
            if child is not None:
                child.wait(timeout=20)
                if child.stdin and not child.stdin.closed:
                    child.stdin.close()
            drain = time.perf_counter()
            while job.accounting()["active_processes"]:
                if time.perf_counter()-drain > 20:
                    raise RuntimeError("owned job termination not confirmed; reservation must remain unclosed")
                time.sleep(.01)
            accounting = job.accounting()
        finally:
            job.close()  # Crash/error fallback is kill-on-close, never broad PID matching.
    if child is None:
        raise RuntimeError(f"worker creation failed; reservation remains unclosed: {error}")
    return {"worker_pid": child.pid, "worker_returncode": child.returncode, "job_name": job_name,
        "process_tree_closed": accounting["active_processes"] == 0,
        "accounting": accounting, "timed_out": timed_out, "interrupted": interrupted,
        "supervisor_error": error, "elapsed_seconds": time.perf_counter()-begin}


def _stop_requested(reservation):
    path = reservation["attempt"]/"stop.json"
    if not path.exists():
        return False
    request = lineage.admission.read_closed(path, lineage.native._hash(path))
    if request != {"schema_version": VERSION+"-stop", "start_sha256": reservation["start_sha256"]}:
        raise ValueError("stop request does not bind the owned attempt")
    return True


def request_stop(directory, root_sha256):
    """Request cooperative supervisory stop; this is NOT proof of termination."""
    directory = Path(directory).resolve()
    paths = lineage._attempts(directory)
    if not paths:
        raise ValueError("no reserved attempt to stop")
    _, _, _, digest = lineage.history(directory, root_sha256, current_index=len(paths)-1)
    lineage.journal._publish_json(paths[-1]/"stop.json", {"schema_version": VERSION+"-stop", "start_sha256": digest})


def supervise(*, source, locations):
    reservation = lineage.reserve(source=source, locations=locations)
    start = reservation["start"]
    command = [sys.executable, "-u", "-m", "experiments.pirc17.seed_resume_session", "worker",
        "--directory", str(reservation["directory"]), "--root-sha256", reservation["root_sha256"],
        "--index", str(reservation["index"]), "--start-sha256", reservation["start_sha256"]]
    elapsed = lambda: time.perf_counter()-reservation["started_monotonic"]
    outcome = run_owned(command, timeout=start["outer_remaining_seconds"]-elapsed(),
                        stop_requested=lambda: _stop_requested(reservation), job_name=start["job_name"])
    # Record actual job accounting before the immutable closure inventory. A
    # failure here leaves an unclosed reservation, never an assumed safe retry.
    lineage.journal._publish_json(reservation["attempt"]/"work-process.json", outcome)
    closure = lineage.seal(reservation, elapsed=elapsed(), returncode=outcome["worker_returncode"],
        tree_closed=outcome["process_tree_closed"], timed_out=outcome["timed_out"],
        interrupted=outcome["interrupted"], error=outcome["supervisor_error"])
    print(json.dumps({"phase": "resume_supervisor_closed", "directory": str(reservation["directory"]),
                      "root_sha256": reservation["root_sha256"], **closure}), flush=True)
    return 0 if closure["status"] == "computed_closed" else 124 if closure["timed_out"] else 130 if closure["interrupted"] else 1


def worker(directory, root_sha256, index, start_sha256):
    root, _, start, actual = lineage.history(directory, root_sha256, current_index=index)
    if actual != start_sha256:
        raise ValueError("worker reservation digest changed")
    require_owned_job(start["job_name"])
    result = execution.run(source=root["source"], **root["data_locations"],
        output=Path(directory)/"attempts"/f"{index:06d}"/"work", resume_directory=directory,
        root_sha256=root_sha256, attempt_index=index,
        progress=lambda event: print(json.dumps(event), flush=True))
    return 0 if result["status"] == "computed_pending_supervisory_closure" else 1


def inspect_closed(directory, root_sha256):
    """Read-only whole-chain consumer; completion is never numerical acceptance."""
    directory = Path(directory).resolve()
    root, closed, _, _ = lineage.history(directory, root_sha256)
    if not closed or closed[-1]["closure"]["status"] != "computed_closed":
        raise ValueError("whole consumer requires a complete supervisory closure")
    before = lineage.inventory(directory)
    observer = lineage.native.MemoryObserver()
    source, spec = root["source"], root["plan"]
    prefix = execution.admission.inspect(**source, guard=observer.guard)
    reference = execution.admission.read_closed(source["reference_ledger"], spec["reference_ledger_sha256"], jsonl=True)
    directories, _, _ = lineage.inherit(prefix, source, root["data_locations"], directory, root_sha256,
                                       None, reference, guard=observer.guard)
    last = closed[-1]
    work, saved = last["directory"]/"work", last["saved"]
    if saved is None or len(prefix.rows) != spec["expected_run_count"]:
        raise ValueError("closed history is missing registered scientific rows")
    result_path = work/"core-result.json"
    result = execution.admission.read_closed(result_path, lineage.native._hash(result_path))
    inherited_path = work/"inherited.json"
    inherited = execution.admission.read_closed(inherited_path, lineage.native._hash(inherited_path))
    count = saved.manifest["inherited_count"]
    if (set(inherited) != {"source_forecast_sha256", "rows"}
            or inherited["source_forecast_sha256"] != source["forecast_sha256"]
            or len(inherited["rows"]) != count):
        raise ValueError("materialized inherited prefix changed")
    rows = deepcopy(inherited["rows"])
    for record in saved.records:
        row = deepcopy(record["row"])
        row["particle_artifact"]["path"] = "journal/"+row["particle_artifact"]["path"]
        rows.append(row)
    for index, (row, original, original_directory) in enumerate(zip(rows, prefix.rows, directories)):
        observer.guard()
        relative = f"inherited/{index:06d}.npz" if index < count else f"journal/particles/{index:06d}.npz"
        expected = deepcopy(original)
        expected["particle_artifact"]["path"] = relative
        if row != expected:
            raise ValueError("assembled result differs from the original ordered scientific ancestry")
        execution.load_particle_evidence(row, work)
        execution.load_particle_evidence(original, original_directory)
    if (result["schema_version"] != execution.VERSION or result["status"] != "computed_pending_supervisory_closure"
            or result["expected_run_count"] != len(rows) or result["inherited_success_count"] != count
            or result["new_committed_count"] != len(saved.records) or result["new_failure_count"] != 0
            or result["missing_completed_record_count"] != 0 or result["errors"]
            or result["journal_tip"] != saved.tip or result["uncommitted_files"] != saved.uncommitted_files
            or result["resources"] != lineage.resources(spec, last["start"]["prior_elapsed_seconds"])
            or result["production_resume_ready"] is not False or not lineage._unqualified(result)
            or not lineage._seconds(result["elapsed_seconds"]) or result["elapsed_seconds"] > last["closure"]["elapsed_seconds"]):
        raise ValueError("core result changes closed counts, errors, resources or certification boundary")
    memory = result["observer"]
    if (type(memory.get("calls")) is not int or memory["calls"] <= 0 or memory.get("failed_calls") != 0
            or type(memory.get("minimum_observed_available_bytes")) is not int
            or memory["minimum_observed_available_bytes"] < spec["minimum_free_bytes"]
            or memory.get("measurement_cache") is not False or result["new_attempt_maps"] is None
            or execution.runtime_identity() != prefix.initialization["runtime"]):
        raise ValueError("resumed runtime, maps or fresh-memory observations differ")
    # Asset manifests describe only what each new attempt actually verified;
    # absent historical completion/observer/map evidence remains absent.
    maps = result["new_attempt_maps"]
    map_files = {Path(path): sha for path, sha in maps.get("receipt_sha256", {}).items()}
    map_files.update({Path(root["data_locations"]["data_root"])/path: sha
                      for path, sha in maps.get("verified_assets", {}).items()})
    if any(lineage.native._hash(path) != sha for path, sha in map_files.items()):
        raise ValueError("resumed map receipt or asset changed")
    whole = execution._whole_math(prefix, rows, work, reference, Path(source["reference_ledger"]).parent, observer.guard)
    stored = execution.admission.read_closed(work/"whole-math.json", result["whole_math_sha256"])
    if stored != whole:
        raise ValueError("whole resumed scientific audit does not independently reproduce")
    if before != lineage.inventory(directory) or lineage.source_hashes() != root["source_sha256"]:
        raise ValueError("closed continuation evidence changed during independent replay")
    return {"schema_version": VERSION+"-inspection", "status": "verified_closed_whole_workload",
        "root_sha256": root_sha256, "last_closure_sha256": last["closure_sha256"],
        "attempt_count": len(closed), "expected_run_count": len(rows),
        "legacy_inherited_count": root["validated_prefix"]["reusable_success_count"],
        "new_committed_count": sum(item["closure"]["new_committed_count"] for item in closed),
        "total_charged_elapsed_seconds": lineage.math.fsum([lineage._legacy_elapsed(root),
            *[item["closure"]["elapsed_seconds"] for item in closed]]),
        "whole_math_sha256": result["whole_math_sha256"], "reference_numerical": prefix.report["reference_numerical"],
        "historical_resource_gaps": whole["historical_resource_gaps"], "numerically_qualified": False,
        "resource_certified": False, **lineage.UNQUALIFIED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    launch = modes.add_parser("run")
    for name in sorted(lineage.SOURCE_ARGUMENTS | lineage.LOCATION_ARGUMENTS):
        launch.add_argument("--"+name.replace("_", "-"), required=True)
    worker_parser = modes.add_parser("worker", help=argparse.SUPPRESS)
    stop_parser = modes.add_parser("stop")
    inspect_parser = modes.add_parser("inspect")
    for sub in (worker_parser, stop_parser, inspect_parser):
        sub.add_argument("--directory", type=Path, required=True)
        sub.add_argument("--root-sha256", required=True)
    worker_parser.add_argument("--index", type=int, required=True)
    worker_parser.add_argument("--start-sha256", required=True)
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
    return supervise(source={key: args[key] for key in lineage.SOURCE_ARGUMENTS},
                     locations={key: args[key] for key in lineage.LOCATION_ARGUMENTS})


if __name__ == "__main__":
    raise SystemExit(main())
