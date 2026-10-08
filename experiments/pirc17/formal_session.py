"""Persistent, owned native worker transport for the future formal controller.

This layer consumes an already durable pending ledger reservation. It neither
grants final-eval authority nor settles that reservation. The controller must
publish the returned observation, account its own IO/idle/closure overhead,
validate the scientific artifact manifest, and settle before dispatching again.
There is deliberately no production command-line launcher in this module.

One synchronous handler is constructed once, after the first request. Models
and maps can stay loaded across requests. The bounded immutable file mailbox
avoids blocking pipe reads and output-buffer deadlocks. A separate watchdog
remains active during supervisor IO. Native termination is asynchronous: the
observed overrun is retained, never advertised as a zero-latency hard cutoff.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from uuid import uuid4

from . import formal_budget as budget
from .physical_memory import available_physical_bytes
from .protocol_core import canonical, envelope, read_json, sha256, under, unpack

VERSION = "pirc17-owned-persistent-session-v1"
MAX_MESSAGE_BYTES = 64 * 1024
MIN_AVAILABLE_BYTES = 2 * 1024**3
POLL_SECONDS = .01
MEMORY_POLL_NS = 100_000_000
CLOSE_LIMIT_NS = 5_000_000_000


class SessionFailure(RuntimeError):
    """No successful completion barrier; the reservation remains chargeable."""


class UnclosedTree(SessionFailure):
    """Never settle/re-admit on a merely requested, unconfirmed termination."""


def _read(path, expected=None):
    record = read_json(path, max_bytes=MAX_MESSAGE_BYTES)
    unpack(record, expected_sha256=expected)
    return record


def _write(path, payload):
    if len(canonical(envelope(payload))) + 1 > MAX_MESSAGE_BYTES:
        raise ValueError("bounded metadata only; save arrays outside the mailbox")
    return budget._publish(path, payload)


def _integer(value, *, positive=False):
    return budget._integer(value, positive=positive)


def _native_job(name):
    # Reuse the sealed implementation without changing its source. Its imports
    # occur under the first reservation's clock, never once per forecast.
    from .seed_resume_session import OwnedJob
    return OwnedJob(name)


def _membership(name):
    from .seed_resume_session import require_owned_job
    require_owned_job(name)


def _command(value):
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(v, str) or not v or "\0" in v for v in value)):
        raise ValueError("explicit argument vector required; no shell command")
    return list(value)


def _job_name(value):
    prefix = "PIRC17-FORMAL-"
    if (not isinstance(value, str) or not value.startswith(prefix) or len(value) != len(prefix)+32
            or any(c not in "0123456789abcdef" for c in value[len(prefix):])):
        raise ValueError("exact fresh formal job name required")
    return value


def _dispatch_claim(directory, pending, root_sha256):
    claim = _read(under(directory, "dispatches/"+sha256(pending["reservation_sha256"])+".json"))
    p = unpack(claim)
    budget._fields(p, ("schema_version", "ledger_root_sha256", "reservation_sha256", "work_id",
                      "session_directory", "job_name"))
    if (p["schema_version"] != VERSION+"-dispatch" or p["ledger_root_sha256"] != root_sha256
            or p["reservation_sha256"] != pending["reservation_sha256"] or p["work_id"] != pending["work_id"]
            or not isinstance(p["session_directory"], str) or not Path(p["session_directory"]).is_absolute()
            or str(Path(p["session_directory"]).resolve()) != p["session_directory"]):
        raise ValueError("dispatch claim changed its pending work/session")
    _job_name(p["job_name"])
    return claim


def _query_job(name):
    """Read-only named-Job recovery. Never adopt/terminate by name or PID."""
    if os.name != "nt":
        raise OSError("native Windows closure observation required")
    from .seed_resume_session import _Accounting
    api = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    pointer, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
    api.OpenJobObjectW.argtypes, api.OpenJobObjectW.restype = [dword, boolean, ctypes.c_wchar_p], pointer
    api.QueryInformationJobObject.argtypes = [pointer, ctypes.c_int, pointer, dword, pointer]
    api.QueryInformationJobObject.restype = boolean
    api.CloseHandle.argtypes, api.CloseHandle.restype = [pointer], boolean
    handle = api.OpenJobObjectW(4, False, _job_name(name))  # QUERY only, not inherited.
    if not handle:
        code = ctypes.get_last_error()
        if code == 2:  # Only ERROR_FILE_NOT_FOUND means the named object is gone.
            return {"exists": False, "accounting": None}
        raise ctypes.WinError(code)
    try:
        state = _Accounting()
        if not api.QueryInformationJobObject(handle, 1, ctypes.byref(state), ctypes.sizeof(state), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"exists": True, "accounting": {"active_processes": int(state.active),
            "total_processes": int(state.total), "user_seconds": state.user/1e7, "kernel_seconds": state.kernel/1e7}}
    finally:
        if not api.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def recover_pending_closure(ledger):
    """Observe old-tree closure; caller still conservatively charges unknown time.

A missing/invalid claim is not inferred to mean that work never started. An
active old Job blocks recovery; this function cannot kill or retry it. Named
object absence implies completed destruction under the Windows Job contract,
not merely an exited/recycled launcher PID.
"""
    if ledger._lock.stream.closed or ledger._poisoned:
        raise ValueError("recovery requires the held healthy ledger writer")
    pending = ledger.summary()["pending"]
    if pending is None:
        raise ValueError("no pending reservation to recover")
    claim = _dispatch_claim(ledger.directory, pending, ledger.root_sha256)
    p = unpack(claim)
    observation = _query_job(p["job_name"])
    if observation["exists"] and observation["accounting"]["active_processes"] != 0:
        raise UnclosedTree("old owned worker is still active; no settlement or new admission")
    return envelope({"schema_version": VERSION+"-recovery-closure", "ledger_root_sha256": ledger.root_sha256,
        "reservation_sha256": pending["reservation_sha256"], "work_id": pending["work_id"],
        "dispatch_claim_sha256": claim["sha256"], "job_name": p["job_name"], "observation": observation,
        "observed_ns": time.monotonic_ns(), "process_tree_closed": True, "elapsed_ns": None,
        "required_conservative_charge_ns": pending["reserved_ns"]})


def _load_session(directory, expected):
    directory = Path(directory).resolve()
    record = _read(under(directory, "session.json"), expected)
    p = unpack(record)
    budget._fields(p, ("schema_version", "directory", "job_name", "worker_command",
                      "ledger_directory", "ledger_root_sha256", "execution_sha256"))
    if p["schema_version"] != VERSION or p["directory"] != str(directory):
        raise ValueError("session location/version/job identity changed")
    _job_name(p["job_name"])
    _command(p["worker_command"])
    root = read_json(under(Path(p["ledger_directory"]), "ledger.json"))
    contract = budget.validate_contract(unpack(root, expected_sha256=p["ledger_root_sha256"]))
    if (contract["ledger_directory"] != p["ledger_directory"]
            or contract["execution_sha256"] != sha256(p["execution_sha256"])):
        raise ValueError("session does not bind its canonical execution ledger")
    return p, contract


def _request(record, *, session_sha256, previous_sha256, sequence, session, work):
    p = unpack(record)
    budget._fields(p, ("schema_version", "session_sha256", "previous_barrier_sha256", "sequence",
                      "work_id", "reservation_sha256", "reservation_event_index", "started_ns", "deadline_ns"))
    if (p["schema_version"] != VERSION+"-request" or p["session_sha256"] != session_sha256
            or p["previous_barrier_sha256"] != previous_sha256
            or type(p["sequence"]) is not int or p["sequence"] != sequence):
        raise ValueError("request sequence/session/barrier chain changed")
    index = _integer(p["reservation_event_index"])
    event = unpack(_read(under(Path(session["ledger_directory"]), f"events/{index:06d}.json"), p["reservation_sha256"]))
    budget._fields(event, ("schema_version", "index", "root_sha256", "previous_sha256", "type", "row"))
    row = event["row"]
    budget._fields(row, ("work_id", "phase", "reserved_ns", "generated_forecasts", "reservation_name"))
    if (event["schema_version"] != budget.VERSION+"-event" or event["type"] != "reserve"
            or type(event["index"]) is not int or event["index"] != index
            or event["root_sha256"] != session["ledger_root_sha256"]
            or row["work_id"] != p["work_id"] or p["work_id"] not in work):
        raise ValueError("request must name its actual durable reservation")
    registered = work[p["work_id"]]
    if (row["phase"] != registered["phase"] or type(row["generated_forecasts"]) is not int
            or row["generated_forecasts"] != registered["generated_forecasts"]
            or _integer(row["reserved_ns"], positive=True) > registered["max_active_ns"]):
        raise ValueError("reservation changes its registered work")
    start, deadline = _integer(p["started_ns"]), _integer(p["deadline_ns"], positive=True)
    if not start < deadline <= start+row["reserved_ns"]:
        raise ValueError("deadline must fit inside the pending reservation")
    return p, registered


def _bootstrap_request(record, *, session_sha256, session, contract):
    """Read-only admission of restoration, not a scientific reservation.

    Replay only the anchored startup prefix. Credit topups may be published
    concurrently by its sole controller; no writer or old ledger is opened.
    The absolute deadline comes from the remaining ORIGINAL input/total caps
    before this control span, never from a caller-supplied fresh allowance.
    """
    from .formal_controller import VERSION as CONTROL_VERSION
    p = unpack(record)
    budget._fields(p, ('schema_version', 'session_sha256', 'control_sha256',
                      'started_ns', 'deadline_ns'))
    if p['schema_version'] != VERSION+'-bootstrap-request' or p['session_sha256'] != session_sha256:
        raise ValueError('bootstrap must name its owned session')
    directory = Path(session['ledger_directory'])
    key = sha256(p['control_sha256'])
    c = unpack(_read(under(directory, 'controls/'+key+'.json'), key))
    budget._fields(c, ('schema_version', 'ledger_root_sha256', 'phase', 'started_ns',
                      'phase_deadline_ns', 'worker_job_name', 'session_directory', 'worker_command'))
    phase = contract['workloads'][0]['phase']
    if (c['schema_version'] != CONTROL_VERSION+'-control'
            or c['ledger_root_sha256'] != session['ledger_root_sha256'] or c['phase'] != phase
            or c['worker_job_name'] != session['job_name'] or c['session_directory'] != session['directory']
            or canonical(c['worker_command']) != canonical(session['worker_command'])
            or _integer(p['started_ns']) != _integer(c['started_ns'])
            or _integer(p['deadline_ns'], positive=True) != _integer(c['phase_deadline_ns'], positive=True)):
        raise ValueError('bootstrap control/session/input deadline changed')
    head = unpack(_read(under(directory, 'head.json')))
    budget._fields(head, ('root_sha256', 'event_count', 'last_event_sha256'))
    count = _integer(head['event_count'], positive=True)
    if count > 10000 or head['root_sha256'] != session['ledger_root_sha256']:
        raise ValueError('bounded startup prefix required')
    state = budget._State(contract, session['ledger_root_sha256'])
    opened = False
    for index in range(count):
        event = _read(under(directory, f'events/{index:06d}.json'))
        e = unpack(event)
        if e['root_sha256'] != session['ledger_root_sha256']:
            raise ValueError('bootstrap event names another ledger')
        if e['type'] not in {'partial_predecessor', 'resource_predecessor', 'control_open', 'control_credit'}:
            raise ValueError('restoration must precede all new scientific work')
        if e['type'] == 'control_open':
            remaining = min(contract['phase_caps_ns'][phase]-state.charged[phase],
                            contract['total_cap_ns']-sum(state.charged.values()))
            if opened or e['row']['control_sha256'] != key or c['phase_deadline_ns'] != c['started_ns']+remaining:
                raise ValueError('bootstrap cannot reset or borrow the original input budget')
            opened = True
        state.apply(e, event['sha256'])
    if (state.tip != head['last_event_sha256'] or not opened or state.active_control != key
            or state.pending is not None or state.halted is not None or not state.imported_success
            or any(v != ('success' if k in state.imported_success else 'unattempted')
                   for k, v in state.status.items())
            or time.monotonic_ns() >= p['deadline_ns']):
        raise ValueError('fresh unexpired partial-recovery control required')
    return p


def serve(directory, session_sha256, handler_factory):
    """Internal CPU worker seam; actual authorized entrypoint is still required.

handler_factory(session, contract) is called ONCE under the first request.
The returned handler(registered_work, output_directory) must finish all CPU
work and durably save its artifacts before returning a small dict manifest.
No asynchronous tasks/subprocesses may survive its return. This is a bound
source contract, not a claim that a digest proves absence of async threads.
"""
    directory = Path(directory).resolve()
    session, contract = _load_session(directory, sha256(session_sha256))
    _membership(session["job_name"])
    work = {row["work_id"]: row for row in contract["workloads"]}
    # Python's first Windows platform.platform() synchronously invokes two
    # version-query helpers. Runtime construction later reads this SAME cached
    # identity. Finish that bounded metadata setup before ready, so the strict
    # lifetime process-count baseline does not reject those completed helpers.
    # The first reservation/watchdog already covers this startup; query errors
    # propagate before readiness. Do not move the baseline after scientific
    # work or weaken the active/total-process and worker-liveness guard.
    platform.platform()
    _write(under(directory, "ready.json"), {"schema_version": VERSION+"-ready",
        "session_sha256": session_sha256, "worker_pid": os.getpid()})
    sequence, previous, handler, seen_work = 0, session_sha256, None, set()
    bootstrapped = False
    while True:
        bootstrap_path = under(directory, 'bootstrap-request.json')
        if not bootstrapped and bootstrap_path.exists():
            if sequence != 0 or handler is not None:
                raise ValueError('bootstrap must precede handler construction and work')
            request = _read(bootstrap_path)
            p = _bootstrap_request(request, session_sha256=session_sha256, session=session, contract=contract)
            value, error_type = None, None
            try:
                handler = handler_factory(session, contract)
                restore = getattr(handler, 'bootstrap', None)
                if not callable(handler) or not callable(restore):
                    raise TypeError('explicit synchronous restoration handler required')
                output = under(directory, 'bootstrap'); output.mkdir()
                value = restore(output)
                if not isinstance(value, dict):
                    raise TypeError('bounded restoration manifest required')
                if time.monotonic_ns() >= p['deadline_ns']:
                    raise TimeoutError('restoration exceeded original input deadline')
            except BaseException as exc:
                value, error_type = None, type(exc).__name__
            _write(under(directory, 'bootstrap-barrier.json'), {
                'schema_version': VERSION+'-bootstrap-barrier', 'session_sha256': session_sha256,
                'request_sha256': request['sha256'], 'control_sha256': p['control_sha256'],
                'worker_pid': os.getpid(), 'status': 'failure' if error_type else 'success',
                'value': value, 'error_type': error_type})
            if error_type:
                return 1
            bootstrapped = True
        path = under(directory, f"requests/{sequence:06d}.json")
        if not path.exists():
            time.sleep(POLL_SECONDS)
            continue
        request = _read(path)
        p, registered = _request(request, session_sha256=session_sha256,
            previous_sha256=previous, sequence=sequence, session=session, work=work)
        claim = unpack(_dispatch_claim(Path(session["ledger_directory"]), p, session["ledger_root_sha256"]))
        if claim["session_directory"] != str(directory) or claim["job_name"] != session["job_name"]:
            raise ValueError("reservation dispatch belongs to another owned session")
        if p["work_id"] in seen_work:
            raise ValueError("worker must not execute registered work twice")
        seen_work.add(p["work_id"])
        if time.monotonic_ns() >= p["deadline_ns"]:
            raise TimeoutError("worker received an expired reservation")
        result_sha, error_type = None, None
        try:
            if handler is None:
                handler = handler_factory(session, contract)
                if not callable(handler):
                    raise TypeError("synchronous handler required")
            output = under(directory, f"outputs/{sequence:06d}")
            output.mkdir()
            value = handler(dict(registered), output)
            if not isinstance(value, dict):
                raise TypeError("small result manifest must be an object")
            result = _write(under(output, "result.json"), {"schema_version": VERSION+"-result",
                "session_sha256": session_sha256, "sequence": sequence,
                "reservation_sha256": p["reservation_sha256"], "work_id": p["work_id"], "value": value})
            result_sha = result["sha256"]
        except Exception as exc:
            # No exception message/private data in the transport control record.
            error_type = type(exc).__name__[:128]
        barrier = _write(under(directory, f"barriers/{sequence:06d}.json"), {
            "schema_version": VERSION+"-barrier", "session_sha256": session_sha256,
            "sequence": sequence, "previous_barrier_sha256": previous, "request_sha256": request["sha256"],
            "reservation_sha256": p["reservation_sha256"], "work_id": p["work_id"],
            "worker_pid": os.getpid(), "status": "failure" if error_type else "success",
            "result_sha256": result_sha, "error_type": error_type})
        if error_type:
            return 1
        previous, sequence = barrier["sha256"], sequence+1


def _barrier_result(record, request, *, directory, session_sha256, sequence, previous_sha256, worker_pid):
    """Validate saved barrier/result bindings only; this is NOT a live OS check."""
    p = unpack(record)
    budget._fields(p, ("schema_version", "session_sha256", "sequence", "previous_barrier_sha256",
        "request_sha256", "reservation_sha256", "work_id", "worker_pid", "status", "result_sha256", "error_type"))
    q = unpack(request)
    expected = {"schema_version": VERSION+"-barrier", "session_sha256": session_sha256,
        "sequence": sequence, "previous_barrier_sha256": previous_sha256,
        "request_sha256": request["sha256"], "reservation_sha256": q["reservation_sha256"],
        "work_id": q["work_id"], "worker_pid": worker_pid}
    if any(type(p[k]) is not type(v) or p[k] != v for k, v in expected.items()):
        raise ValueError("completion barrier does not bind this exact request")
    if p["status"] == "failure":
        if (p["result_sha256"] is not None or not isinstance(p["error_type"], str)
                or not p["error_type"] or len(p["error_type"]) > 128):
            raise ValueError("malformed worker failure barrier")
        return None
    if p["status"] != "success" or p["error_type"] is not None:
        raise ValueError("unknown completion disposition")
    result = unpack(_read(under(directory, f"outputs/{sequence:06d}/result.json"), p["result_sha256"]))
    budget._fields(result, ("schema_version", "session_sha256", "sequence", "reservation_sha256", "work_id", "value"))
    if (result["schema_version"] != VERSION+"-result" or not isinstance(result["value"], dict)
            or any(type(result[k]) is not type(q[k]) or result[k] != q[k]
                   for k in ("session_sha256", "sequence", "reservation_sha256", "work_id"))):
        raise ValueError("saved manifest does not match this completion")
    return result


class OwnedSession:
    """One retained kill-on-close Job, one worker, one in-flight reservation.

All injected callables are explicit software-test seams. Production must use
the defaults and bind the concrete command/source/runtime in its execution
seal. A session is intentionally not restartable after any transport failure.
"""
    def __init__(self, directory, ledger, worker_command, *, _clock=time.monotonic_ns,
                 _memory=available_physical_bytes, _job_factory=_native_job, _stop=lambda: False):
        self.directory = Path(directory).resolve()
        self.ledger, self.command = ledger, _command(worker_command)
        self.clock, self.memory, self.job_factory, self.stop_requested = _clock, _memory, _job_factory, _stop
        self.job_name = "PIRC17-FORMAL-"+uuid4().hex
        self.job = self.child = self.session = self.baseline = None
        self.sequence, self.previous, self.worker_pid = 0, None, None
        self.closed, self.failed = False, False
        self._reason, self._watch = None, None
        self._finish_watch = threading.Event()
        self._state_lock = threading.Lock()
        self._in_call = threading.Lock()
        self._last_end = None
        self._wait_tick = None
        self._bootstrap_attempted = False

    def _claim(self, pending):
        # A new session/path/process must not dispatch a pending reservation a
        # second time. Crash recovery consumes this durable claim and proves the
        # old job closed; it must not launch this work again.
        claims = under(self.ledger.directory, "dispatches")
        claims.mkdir(exist_ok=True)
        return _write(under(claims, pending["reservation_sha256"]+".json"), {
            "schema_version": VERSION+"-dispatch", "ledger_root_sha256": self.ledger.root_sha256,
            "reservation_sha256": pending["reservation_sha256"], "work_id": pending["work_id"],
            "session_directory": str(self.directory), "job_name": self.job_name})

    def _trip(self, reason):
        with self._state_lock:
            if self._reason is not None:
                return
            self._reason = {"reason": reason, "observed_ns": self.clock(), "termination_error_type": None}
            if self.job is not None:
                try:
                    self.job.terminate()
                except Exception as exc:
                    self._reason["termination_error_type"] = type(exc).__name__

    def request_termination(self, reason):
        """Retained-owner cancellation, also usable by the cumulative watchdog.

This does not prove tree closure or settle anything. The owner must stop
admission and call close() after the per-request watcher has joined.
"""
        if reason not in {"deadline", "operator_stop", "resource_pressure", "supervision_error"}:
            raise ValueError("bounded supervisory termination reason required")
        self._trip(reason)

    def _guard(self, *, memory=False):
        if self.clock() >= self.deadline:
            self._trip("deadline")
        elif self.stop_requested():
            self._trip("operator_stop")
        elif memory:
            available = self.memory()
            if type(available) is not int or available < MIN_AVAILABLE_BYTES:
                self._trip("resource_pressure")
        if self._reason is not None:
            raise SessionFailure(self._reason["reason"])

    def _watchdog(self):
        next_memory = 0
        while not self._finish_watch.is_set():
            try:
                now = self.clock()
                self._guard(memory=now >= next_memory)
                if now >= next_memory:
                    next_memory = now+MEMORY_POLL_NS
            except SessionFailure:
                return
            except BaseException:
                self._trip("supervision_error")
                return
            self._finish_watch.wait(min(POLL_SECONDS, max(0, self.deadline-self.clock())/1e9))

    def _wait_record(self, relative):
        path = under(self.directory, relative)
        while True:
            if self._wait_tick is not None:
                self._wait_tick()  # Owner thread only; watchdog never writes the ledger.
            self._guard()
            if path.exists():
                record = _read(path)
                self._guard()  # Slow IO cannot turn a late response into success.
                return record
            if self.child.poll() is not None or self.job.accounting()["active_processes"] == 0:
                raise SessionFailure("worker_exit_without_barrier")
            time.sleep(min(POLL_SECONDS, max(0, self.deadline-self.clock())/1e9))

    def _start(self):
        if os.name != "nt":
            raise OSError("formal native supervision requires Windows")
        self.directory.mkdir()  # Exclusive; never reuse another session directory.
        for name in ("requests", "barriers", "outputs"):
            (self.directory/name).mkdir()
        contract = self.ledger._state.contract
        self.session = _write(self.directory/"session.json", {"schema_version": VERSION,
            "directory": str(self.directory), "job_name": self.job_name, "worker_command": self.command,
            "ledger_directory": str(self.ledger.directory), "ledger_root_sha256": self.ledger.root_sha256,
            "execution_sha256": contract["execution_sha256"]})
        self.previous = self.session["sha256"]
        self.job = self.job_factory(self.job_name)
        self._guard(memory=True)
        # The sealed bootstrap reads exactly one byte before spawning anything.
        from .seed_resume_session import BOOTSTRAP
        command = [*self.command, "--session-directory", str(self.directory),
                   "--session-sha256", self.session["sha256"]]
        self.child = subprocess.Popen([sys._base_executable, "-I", "-S", "-u", "-c", BOOTSTRAP, *command],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        self.job.assign(self.child)
        self._guard()
        self.child.stdin.write(b"R")
        self.child.stdin.flush()
        self.child.stdin.close()
        ready = unpack(self._wait_record("ready.json"))
        budget._fields(ready, ("schema_version", "session_sha256", "worker_pid"))
        if ready["schema_version"] != VERSION+"-ready" or ready["session_sha256"] != self.session["sha256"]:
            raise ValueError("worker ready message names another session")
        self.worker_pid = _integer(ready["worker_pid"], positive=True)
        self.baseline = self.job.accounting()
        if self.baseline["active_processes"] < 2:
            raise ValueError("owned launcher and synchronous worker must remain alive")

    def _barrier(self, record, request):
        result = _barrier_result(record, request, directory=self.directory, session_sha256=self.session["sha256"],
            sequence=self.sequence, previous_sha256=self.previous, worker_pid=self.worker_pid)
        if result is None:
            return None
        accounting = self.job.accounting()
        if (self.child.poll() is not None or any(accounting[k] != self.baseline[k]
                for k in ("active_processes", "total_processes"))):
            raise ValueError("persistent CPU handler spawned descendants or exited before its barrier")
        return result

    def bootstrap(self, control_sha256, *, started_ns, deadline_ns, tick):
        """Construct and restore the retained worker under input control time.

        No reserve/dispatch claim, work attempt, generation debit or settlement
        is made here. The controller must validate/publish the returned receipt
        and pay every startup/validation/closure interval from its input span.
        """
        if not callable(tick):
            raise ValueError('explicit owner-thread control metering required')
        if not self._in_call.acquire(blocking=False):
            raise RuntimeError('one in-flight request only')
        try:
            summary = self.ledger.summary()
            key = sha256(control_sha256)
            if (self.closed or self.failed or self._bootstrap_attempted or self.session is not None
                    or self.ledger._poisoned or self.ledger._lock.stream.closed
                    or summary['pending'] is not None or summary['halted_reason'] is not None
                    or summary['active_control_sha256'] != key or not self.ledger._state.imported_success):
                raise ValueError('fresh healthy partial-recovery control required')
            c = unpack(_read(under(self.ledger.directory, 'controls/'+key+'.json'), key))
            if (c['phase'] != self.ledger._state.contract['workloads'][0]['phase']
                    or c['worker_job_name'] != self.job_name or c['session_directory'] != str(self.directory)
                    or canonical(c['worker_command']) != canonical(self.command)
                    or c['started_ns'] != _integer(started_ns)
                    or c['phase_deadline_ns'] != _integer(deadline_ns, positive=True)
                    or not started_ns <= self.clock() < deadline_ns):
                raise ValueError('exact active input control deadline required')
            self._bootstrap_attempted = True
            self.deadline, self._wait_tick = deadline_ns, tick
            self._finish_watch.clear()
            self._watch = threading.Thread(target=self._watchdog, name='pirc17-owned-bootstrap', daemon=True)
            self._watch.start()
            error = None
            try:
                tick()
                self._guard(memory=True)
                self._start()
                request = _write(under(self.directory, 'bootstrap-request.json'), {
                    'schema_version': VERSION+'-bootstrap-request', 'session_sha256': self.session['sha256'],
                    'control_sha256': key, 'started_ns': started_ns, 'deadline_ns': deadline_ns})
                barrier = self._wait_record('bootstrap-barrier.json')
                p = unpack(barrier)
                budget._fields(p, ('schema_version', 'session_sha256', 'request_sha256',
                                   'control_sha256', 'worker_pid', 'status', 'value', 'error_type'))
                expected = dict(schema_version=VERSION+'-bootstrap-barrier', session_sha256=self.session['sha256'],
                    request_sha256=request['sha256'], control_sha256=key, worker_pid=self.worker_pid,
                    status='success', error_type=None)
                if (any(type(p[k]) is not type(v) or p[k] != v for k, v in expected.items())
                        or not isinstance(p['value'], dict)):
                    raise SessionFailure('restoration did not return its exact success barrier')
                accounting = self.job.accounting()
                if self.child.poll() is not None or any(accounting[k] != self.baseline[k]
                        for k in ('active_processes', 'total_processes')):
                    raise SessionFailure('restoration spawned descendants or exited')
                tick()
                self._guard(memory=True)
            except BaseException as exc:
                error = exc
                self._trip('supervision_error')
            finally:
                self._wait_tick = None
                self._finish_watch.set()
                self._watch.join(timeout=1)
                if self._watch.is_alive():
                    self.failed = True
                    raise UnclosedTree('bootstrap watcher has not stopped')
            if error is not None:
                self.failed = True
                self.close()
                raise error
            self._last_end = self.clock()
            return envelope(dict(schema_version=VERSION+'-bootstrap-observation',
                ledger_root_sha256=self.ledger.root_sha256, session_sha256=self.session['sha256'],
                control_sha256=key, request_sha256=request['sha256'], barrier=barrier,
                started_ns=started_ns, ended_ns=self._last_end, elapsed_ns=self._last_end-started_ns,
                deadline_ns=deadline_ns, worker_pid=self.worker_pid, accounting=accounting))
        finally:
            self._in_call.release()

    def run(self, reservation, *, started_ns=None):
        """Return an observation, not a settled budget/scientific verdict.

Pass the controller's timestamp taken BEFORE Ledger.reserve to include that
IO in the deadline. Inter-call gaps are reported separately, never silently
claimed as charged. The controller must charge them to the proper phase.
"""
        if not self._in_call.acquire(blocking=False):
            raise RuntimeError("one in-flight request only")
        try:
            return self._run(reservation, started_ns=started_ns)
        finally:
            self._in_call.release()

    def _run(self, reservation, *, started_ns):
        if self.closed or self.failed:
            raise RuntimeError("closed/failed session cannot restart")
        summary = self.ledger.summary()
        pending = summary["pending"]
        if (summary["halted_reason"] is not None or self.ledger._lock.stream.closed or self.ledger._poisoned
                or pending is None or canonical(reservation) != canonical({**pending, "ledger_root_sha256": self.ledger.root_sha256})
                or self.ledger.tip["last_event_sha256"] != pending["reservation_sha256"]):
            raise ValueError("an exact current durable, unhalted reservation is required")
        if self.session is not None and self.ledger.root_sha256 != self.session["payload"]["ledger_root_sha256"]:
            raise ValueError("a persistent session cannot switch ledgers")
        now = self.clock()
        begin = now if started_ns is None else _integer(started_ns)
        if begin > now or (self._last_end is not None and begin < self._last_end):
            raise ValueError("invalid or overlapping controller clock interval")
        gap = None if self._last_end is None else {"started_ns": self._last_end, "ended_ns": begin,
                                                   "elapsed_ns": begin-self._last_end}
        self.deadline, self._reason = begin+pending["reserved_ns"], None
        self._claim(pending)  # Exclusive and durable BEFORE launching/dispatching.
        self._finish_watch.clear()
        self._watch = threading.Thread(target=self._watchdog, name="pirc17-owned-deadline", daemon=True)
        barrier = result = closure = None
        error_type, status = None, "failure"
        self._watch.start()
        try:
            self._guard(memory=True)
            if self.session is None:
                self._start()
            self._guard(memory=True)
            request = _write(under(self.directory, f"requests/{self.sequence:06d}.json"), {
                "schema_version": VERSION+"-request", "session_sha256": self.session["sha256"],
                "previous_barrier_sha256": self.previous, "sequence": self.sequence,
                "work_id": pending["work_id"], "reservation_sha256": pending["reservation_sha256"],
                "reservation_event_index": self.ledger.tip["event_count"]-1,
                "started_ns": begin, "deadline_ns": self.deadline})
            barrier = self._wait_record(f"barriers/{self.sequence:06d}.json")
            result = self._barrier(barrier, request)
            self._guard(memory=True)
            if result is None:
                self._trip("worker_failure")
            else:
                status = "success"
        except KeyboardInterrupt:
            self._trip("operator_stop")
        except BaseException as exc:
            error_type = type(exc).__name__
            self._trip("transport_failure")
        finally:
            self._finish_watch.set()
            self._watch.join(timeout=1)
            if self._watch.is_alive():
                # Never close a handle while a watchdog may still be using it.
                self.failed = True
                if self.job is not None:
                    self.job.terminate()
                raise UnclosedTree("watchdog did not stop; keep reservation pending")
        end = self.clock()
        if end >= self.deadline:
            self._trip("deadline")
        if self._reason is not None:
            self.failed = True
            closure = self.close()
            status = "timeout" if self._reason["reason"] == "deadline" else "interrupted" if self._reason["reason"] == "operator_stop" else "failure"
            result = None
            end = self.clock()
        if end >= self.deadline:
            status = "timeout"  # Include termination/closure latency, never hide overrun.
        observation = {"schema_version": VERSION+"-observation", "session_sha256": None if self.session is None else self.session["sha256"],
            "ledger_root_sha256": self.ledger.root_sha256, "reservation_sha256": pending["reservation_sha256"],
            "work_id": pending["work_id"], "sequence": self.sequence, "status": status,
            "started_ns": begin, "ended_ns": end, "elapsed_ns": end-begin, "deadline_ns": self.deadline,
            "inter_call_gap": gap, "barrier_sha256": None if barrier is None else barrier["sha256"],
            "result_sha256": None if result is None else barrier["payload"]["result_sha256"],
            "worker_pid": self.worker_pid, "stop_observation": self._reason, "error_type": error_type,
            "closure": closure, "accounting": None if self.job is None else self.job.accounting()}
        if status == "success":
            self.previous, self.sequence = barrier["sha256"], self.sequence+1
        self._last_end = end
        return envelope(observation)

    def close(self):
        """Close only this retained tree; caller must charge this elapsed time.

Failure raises UnclosedTree, retaining the handle for a later explicit closure
attempt. It does NOT allow a new scientific attempt or infer a dead tree from
a dead launcher. Closing a normal session is cancellation of its idle loop,
not a failed forecast and not an extra forecast call.
"""
        if self._watch is not None and self._watch.is_alive():
            raise UnclosedTree("stop and join the watchdog before releasing its job")
        begin, cutoff = self.clock(), time.monotonic_ns()+CLOSE_LIMIT_NS
        self.closed = True
        accounting = None
        if self.job is not None:
            if self.job.accounting()["active_processes"]:
                self.job.terminate()
            if self.child is not None and self.child.poll() is None:
                # Assignment may have failed: this gated bootstrap has not been
                # released and has no children outside our retained job.
                if self.job.accounting()["active_processes"] == 0:
                    self.child.kill()
            while True:
                accounting = self.job.accounting()
                if accounting["active_processes"] == 0 and (self.child is None or self.child.poll() is not None):
                    break
                if time.monotonic_ns() >= cutoff:
                    raise UnclosedTree("owned process tree closure not confirmed")
                time.sleep(POLL_SECONDS)
            # The cumulative controller may request termination concurrently.
            # Retain the same lock it uses until the handle is no longer visible.
            with self._state_lock:
                self.job.close()
                self.job = None
        if self.child is not None and self.child.stdin is not None and not self.child.stdin.closed:
            self.child.stdin.close()
        end = self.clock()
        return {"schema_version": VERSION+"-closure", "job_name": self.job_name,
            "process_tree_closed": True, "accounting": accounting,
            "launcher_returncode": None if self.child is None else self.child.returncode,
            "started_ns": begin, "ended_ns": end, "elapsed_ns": end-begin}
