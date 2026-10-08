"""Cumulative formal execution controller, without a data-access entrypoint.

The caller must bind the real authorization and scientific manifest validators
in the approved execution sources. This module has no CLI, data reader, fake
approval receipt, or fallback validator. Software seams are not final-eval
authorization. A single retained OwnedSession spans all registered phases.

Administrative time is covered by nonrefundable, prepaid control credit FROM
each original phase budget. Five seconds initially, then one-second topups
before expiry; unused credit is conservative cost, never a new budget. Work
intervals cover reserve, startup, prediction, validation and observation IO.
The complementary intervals cover settlement, phase switches and shutdown.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
import time

from . import formal_budget as budget
from . import formal_session as native
from .physical_memory import available_physical_bytes
from .protocol_core import canonical, envelope, file_hash, read_json, sha256, under, unpack

VERSION = "pirc17-cumulative-controller-v1"
INITIAL_CREDIT_NS = 5_000_000_000
TOPUP_CREDIT_NS = 1_000_000_000
TOPUP_MARGIN_NS = 100_000_000
BOOTSTRAP_DOMAIN_WINDOW_NS = 30_000_000_000
MAX_VERIFICATION_BYTES = 8*1024*1024  # Full 12,370-candidate input file inventory, not IPC.


class ControllerStopped(RuntimeError):
    """No implicit retry/reset after a stopped controller."""


def _publish(directory, payload):
    record = envelope(payload)
    if len(canonical(record))+1 > native.MAX_MESSAGE_BYTES:
        raise ValueError("controller evidence must remain bounded metadata")
    path = under(directory, record["sha256"]+".json")
    budget._publish(path, payload)
    return record


def _control_record(ledger, key):
    p = unpack(read_json(under(ledger.directory, "controls/"+sha256(key)+".json")), expected_sha256=key)
    budget._fields(p, ("schema_version", "ledger_root_sha256", "phase", "started_ns",
        "phase_deadline_ns", "worker_job_name", "session_directory", "worker_command"))
    if (p["schema_version"] != VERSION+"-control" or p["ledger_root_sha256"] != ledger.root_sha256
            or p["phase"] != ledger._state.controls[key]["phase"]):
        raise ValueError("controller start does not bind the real ledger/phase")
    native._job_name(p["worker_job_name"])
    native._command(p["worker_command"])
    session_directory = Path(p["session_directory"])
    if (not session_directory.is_absolute() or str(session_directory.resolve()) != p["session_directory"]
            or not session_directory.resolve().is_relative_to(ledger.directory)):
        raise ValueError("controller worker session must stay inside the canonical ledger")
    return p


def _authority(value, ledger, work):
    budget._fields(value, ("schema_version", "ledger_root_sha256", "execution_sha256",
                          "approval_sha256", "work_id", "granted"))
    expected = {"schema_version": VERSION+"-work-authority", "ledger_root_sha256": ledger.root_sha256,
        "execution_sha256": ledger._state.contract["execution_sha256"],
        "approval_sha256": ledger._state.contract["approval_sha256"], "work_id": work["work_id"], "granted": True}
    if canonical(value) != canonical(expected):
        raise PermissionError("work authority must bind the exact execution/approval/ledger/item")
    return value


def verify_artifacts(value, work, directory):
    """Verify actual saved bytes and complete owned-file inventory, not a bool.

The bound domain validator supplies scientific checks in details. Generic
integrity checks do not certify the scientific meaning of an arbitrary array.
"""
    budget._fields(value, ("schema_version", "work_id", "verified", "artifacts", "details"))
    if (value["schema_version"] != VERSION+"-artifact-verification" or value["work_id"] != work["work_id"]
            or value["verified"] is not True or not isinstance(value["artifacts"], dict)
            or not isinstance(value["details"], dict) or not value["details"]):
        raise ValueError("bound nonempty domain verification and artifact inventory required")
    directory = Path(directory).resolve()
    actual = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        checked = under(directory, relative)
        if checked.is_file():
            actual.add(relative)
    actual.discard("result.json")  # Separately content-hash verified transport manifest.
    if actual != set(value["artifacts"]):
        raise ValueError("missing/extra owned scientific artifact")
    for relative, binding in value["artifacts"].items():
        budget._fields(binding, ("file_sha256", "bytes"))
        path = under(directory, relative)
        if (path.stat().st_size != budget._integer(binding["bytes"])
                or file_hash(path) != sha256(binding["file_sha256"])):
            raise ValueError("saved scientific artifact bytes/hash changed")
    return value


def _save_verification(ledger, work, value):
    """Keep normal per-forecast metadata inline; bind a large input inventory.

    Publication is inside the measured work interval. The native message and
    controller-observation limits stay unchanged. No inventory is truncated.
    """
    if len(canonical(value)) <= native.MAX_MESSAGE_BYTES//2:
        return value
    payload = dict(schema_version=VERSION+'-verification-file', ledger_root_sha256=ledger.root_sha256,
        work_id=work['work_id'], verification=value)
    record = envelope(payload)
    if len(canonical(record))+1 > MAX_VERIFICATION_BYTES:
        raise ValueError('complete scientific verification exceeds its bounded file size')
    directory = under(ledger.directory, 'verifications'); directory.mkdir(exist_ok=True)
    path = under(directory, record['sha256']+'.json')
    budget._publish(path, payload)
    return dict(schema_version=VERSION+'-verification-reference', ledger_root_sha256=ledger.root_sha256,
        work_id=work['work_id'], content_sha256=record['sha256'], file_sha256=file_hash(path), bytes=path.stat().st_size)


def read_verification(ledger, work, value):
    """Resolve a bound complete inventory, never treat its reference as PASS."""
    if not isinstance(value, dict) or value.get('schema_version') != VERSION+'-verification-reference':
        return value  # Existing small inline proofs still go through verify_artifacts.
    budget._fields(value, ('schema_version', 'ledger_root_sha256', 'work_id', 'content_sha256', 'file_sha256', 'bytes'))
    if value['ledger_root_sha256'] != ledger.root_sha256 or value['work_id'] != work['work_id']:
        raise ValueError('scientific verification reference belongs to another ledger/work')
    size = budget._integer(value['bytes'], positive=True)
    if size > MAX_VERIFICATION_BYTES:
        raise ValueError('scientific verification file exceeds its size bound')
    path = under(ledger.directory, 'verifications/'+sha256(value['content_sha256'])+'.json')
    if path.stat().st_size != size:
        raise ValueError('scientific verification file length changed')
    payload = unpack(read_json(path, expected_file_sha256=value['file_sha256'], max_bytes=MAX_VERIFICATION_BYTES),
                     expected_sha256=value['content_sha256'])
    budget._fields(payload, ('schema_version', 'ledger_root_sha256', 'work_id', 'verification'))
    if (payload['schema_version'] != VERSION+'-verification-file' or payload['ledger_root_sha256'] != ledger.root_sha256
            or payload['work_id'] != work['work_id']):
        raise ValueError('saved scientific verification belongs to another ledger/work')
    return payload['verification']


def recover_interrupted(ledger):
    """Conservative terminal recovery, never a forecast retry or new allowance.

The exact old owned Job must be absent/empty. A pending work item consumes its
full old reservation; all prepaid controller credit remains charged. Unknown
controller interruption halts subsequent admission rather than inventing a
clock or treating an open control span as free. Completed outputs are retained.
"""
    if ledger._poisoned or ledger._lock.stream.closed:
        raise ValueError("reopen and hold the healthy writer before recovery")
    key = ledger.summary()["active_control_sha256"]
    if key is None:
        raise ValueError("no open controller span to recover")
    start = _control_record(ledger, key)
    old = native._query_job(start["worker_job_name"])
    if old["exists"] and old["accounting"]["active_processes"]:
        raise native.UnclosedTree("old controller worker still active; recovery cannot adopt or kill it")
    evidence = _publish(under(ledger.directory, "controls"), {"schema_version": VERSION+"-recovery",
        "ledger_root_sha256": ledger.root_sha256, "control_sha256": key, "old_job": old,
        "observed_ns": time.monotonic_ns(), "process_tree_closed": True, "elapsed_ns": None})
    pending = ledger.summary()["pending"]
    if pending is not None:
        ledger.settle(pending["reservation_sha256"], status="interrupted", elapsed_ns=None,
            completion_evidence_sha256=evidence["sha256"], result_sha256=None, reason="verified old tree closure; unknown elapsed")
    ledger.close_control(key, observed_ns=None, evidence_sha256=evidence["sha256"], reason="recovered_unknown")
    return evidence


class _Meter:
    """Disjoint work/control intervals with one uninterrupted phase clock."""
    def __init__(self, start, deadline, credit, clock, cancel, memory, stop):
        self.start, self.deadline, self.credit = start, deadline, credit
        self.clock, self.cancel, self.memory, self.stop = clock, cancel, memory, stop
        self.control_ns, self.boundary, self.work_deadline = 0, start, None
        self.reason, self.error_type = None, None
        self._lock, self._done = threading.Lock(), threading.Event()
        self.thread = threading.Thread(target=self._watch, name="pirc17-cumulative-budget", daemon=True)

    def overhead(self, now=None):
        with self._lock:
            now = self.clock() if now is None else max(now, self.boundary)
            return self.control_ns+(now-self.boundary if self.work_deadline is None else 0)

    def trip(self, reason):
        with self._lock:
            if self.reason is not None:
                return
            self.reason = reason
        self.cancel("deadline" if "deadline" in reason else reason if reason in
                    {"resource_pressure", "operator_stop"} else "supervision_error")

    def guard(self, *, check_memory=False):
        with self._lock:
            now = self.clock()
            work_deadline, credit = self.work_deadline, self.credit
            overhead = self.control_ns+(now-self.boundary if work_deadline is None else 0)
        if now >= self.deadline:
            self.trip("phase_deadline")
        elif work_deadline is not None and now >= work_deadline:
            self.trip("work_deadline")
        elif overhead >= credit:
            self.trip("control_credit_exhausted")
        elif self.stop():
            self.trip("operator_stop")
        elif check_memory:
            available = self.memory()
            if type(available) is not int or available < native.MIN_AVAILABLE_BYTES:
                self.trip("resource_pressure")
        if self.reason is not None:
            raise ControllerStopped(self.reason)

    def begin_work(self, allocation):
        self.guard(check_memory=True)
        now = self.clock()
        with self._lock:
            if self.work_deadline is not None:
                raise RuntimeError("overlapping empirical intervals")
            self.control_ns += now-self.boundary
            self.boundary, self.work_deadline = now, min(now+allocation, self.deadline)
        self.guard()
        return now

    def end_work(self):
        now = self.clock()
        with self._lock:
            if self.work_deadline is None:
                raise RuntimeError("no empirical interval to finish")
            elapsed = now-self.boundary
            self.boundary, self.work_deadline = now, None
        return now, elapsed

    def abort_unreserved_work(self):
        """A failed handoff with NO durable pending work is control time."""
        with self._lock:
            if self.work_deadline is not None:
                now = self.clock()
                self.control_ns += now-self.boundary
                self.boundary, self.work_deadline = now, None

    def add_credit(self, amount):
        self.guard()  # A topup that became durable too late cannot revive work.
        with self._lock:
            self.credit += amount

    def _watch(self):
        next_memory = 0
        while not self._done.is_set():
            try:
                now = self.clock()
                self.guard(check_memory=now >= next_memory)
                if now >= next_memory:
                    next_memory = now+native.MEMORY_POLL_NS
            except ControllerStopped:
                return
            except BaseException as exc:
                self.error_type = type(exc).__name__
                self.trip("supervision_error")
                return
            self._done.wait(native.POLL_SECONDS)

    def close(self):
        self._done.set()
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            self.trip("supervision_error")
            raise native.UnclosedTree("cumulative watcher has not stopped")


class Controller:
    """Run the complete registered inventory, with explicit bound callbacks.

authorize_work(work) must validate current granted scope without data access.
validate_result(work, manifest, output_directory) must verify real scientific
artifacts and return a small evidence dict. Neither callback may queue work or
launch descendants; scientific computation belongs inside the owned worker.
Actual guarded final-eval data entrypoints still must be supplied and sealed.
"""
    def __init__(self, ledger, worker_command, *, authorize_work, validate_result, validate_bootstrap=None,
                 _clock=time.monotonic_ns, _memory=available_physical_bytes,
                 _stop=lambda: False, _session_factory=native.OwnedSession):
        if not callable(authorize_work) or not callable(validate_result):
            raise ValueError("explicit authorization and artifact validators required")
        if validate_bootstrap is not None and not callable(validate_bootstrap):
            raise ValueError('explicit restoration validator required')
        self.ledger, self.clock, self.memory, self.stop = ledger, _clock, _memory, _stop
        self.authorize, self.validate = authorize_work, validate_result
        self.validate_bootstrap = validate_bootstrap
        self.bootstrap_receipt = None
        self.meter, self.control, self.phase, self.finished = None, None, None, False
        self.results = []
        self._next_phase_start = None
        self._initial_started_ns = None
        self._terminal_meter = None
        directory = under(ledger.directory, f"session-{ledger.tip['event_count']:06d}")
        self.session = _session_factory(directory, ledger, worker_command, _clock=_clock, _memory=_memory,
            _stop=lambda: self.stop() or (self.meter is not None and self.meter.reason is not None))

    def _healthy(self):
        if self.ledger._poisoned or self.ledger._lock.stream.closed:
            raise ControllerStopped("ledger writer unavailable")
        if self.ledger.summary()["halted_reason"] is not None:
            raise ControllerStopped("ledger halted")

    def _open_phase(self, phase):
        self._healthy()
        state = self.ledger.summary()
        if state["pending"] is not None or state["active_control_sha256"] is not None:
            raise ControllerStopped("old work/control must be recovered before new admission")
        start = self.clock() if self._next_phase_start is None else self._next_phase_start
        self._next_phase_start = None
        remaining = min(state["remaining_ns_by_phase"][phase], state["remaining_total_ns"])
        if remaining <= 0:
            raise ControllerStopped("phase budget exhausted")
        self.control = envelope({"schema_version": VERSION+"-control",
            "ledger_root_sha256": self.ledger.root_sha256, "phase": phase,
            "started_ns": start, "phase_deadline_ns": start+remaining,
            "worker_job_name": self.session.job_name, "session_directory": str(self.session.directory),
            "worker_command": self.session.command})
        # A concrete entrypoint can carry its pre-ledger metadata/setup clock
        # into the FIRST phase. Debit that already elapsed prefix, plus the
        # ordinary control credit, from this same phase. Never backdate later
        # phases or revive an expired absolute phase deadline.
        prefix = max(0, self.clock()-start) if self._initial_started_ns == start else 0
        self._initial_started_ns = None
        initial = min(INITIAL_CREDIT_NS+prefix, remaining)
        self.phase = phase
        self.meter = _Meter(start, start+remaining, initial, self.clock,
            self.session.request_termination, self.memory, self.stop)
        self.meter.thread.start()
        # The credit event precedes control-manifest IO as well as scientific
        # dispatch. Failed/slow startup cannot disappear from a later ledger.
        self.ledger.open_control(self.control["sha256"], phase=phase, credit_ns=initial)
        directory = under(self.ledger.directory, "controls")
        directory.mkdir(exist_ok=True)
        _publish(directory, unpack(self.control))
        self._healthy()
        self.meter.guard(check_memory=True)

    def _credit(self, *, window_ns=TOPUP_CREDIT_NS):
        self.meter.guard()
        budget._integer(window_ns,positive=True)
        observed = self.meter.overhead()
        left = self.meter.credit-observed
        if left <= window_ns+TOPUP_MARGIN_NS:
            state = self.ledger.summary()
            available = min(state["remaining_ns_by_phase"][self.phase], state["remaining_total_ns"])
            wanted = TOPUP_CREDIT_NS if window_ns == TOPUP_CREDIT_NS else max(TOPUP_CREDIT_NS,window_ns-left+TOPUP_CREDIT_NS)
            amount = min(wanted, available)
            if amount <= 0:
                raise ControllerStopped("no phase credit for necessary supervision")
            self.ledger.topup_control(self.control["sha256"], credit_ns=amount, observed_ns=observed)
            self.meter.add_credit(amount)
            self._healthy()

    def _halt(self, reason, evidence):
        if self.ledger.summary()["halted_reason"] is None:
            self.ledger.halt(reason, evidence)

    def _bootstrap(self):
        """Non-generating restoration in the ORIGINAL input-phase clock.

        The validator is execution-bound domain code, not a truthy PASS flag;
        it receives a metering callback for bounded loops/IO. Worker restore
        and validation never enter begin_work or reserve a completed item.
        """
        observed = self.session.bootstrap(self.control['sha256'],
            started_ns=unpack(self.control)['started_ns'], deadline_ns=self.meter.deadline, tick=self._credit)
        p = unpack(observed)
        expected = dict(schema_version=native.VERSION+'-bootstrap-observation',
            ledger_root_sha256=self.ledger.root_sha256, control_sha256=self.control['sha256'],
            started_ns=unpack(self.control)['started_ns'], deadline_ns=self.meter.deadline)
        if (any(type(p.get(k)) is not type(v) or p.get(k) != v for k, v in expected.items())
                or budget._integer(p['ended_ns']) > self.clock()
                or budget._integer(p['elapsed_ns']) != p['ended_ns']-p['started_ns']):
            raise ValueError('restoration observation does not bind current measured input span')
        from .formal_entrypoint import MODULE
        concrete = self.session.command[1:5] == ['-u','-m',MODULE,'worker']
        # Whole source/context/model checks contain bounded synchronous blocks
        # longer than a forecast's cheap metadata loop. Prepay a sliding 30s
        # administration window FROM original remaining input/total caps;
        # unused credit is retained, not refunded or shifted to another phase.
        tick = (lambda: self._credit(window_ns=BOOTSTRAP_DOMAIN_WINDOW_NS)) if concrete else self._credit
        tick()
        if concrete:
            from .formal_paid_validation import paid_call
            # Keep the original validators/30s credit window. Multi-file
            # synchronous checks must pulse the owning writer while they run,
            # not wait until after an already expired credit interval.
            verified = paid_call(self.validate_bootstrap, deepcopy(p),
                under(self.session.directory, 'bootstrap'), tick, _tick=tick, _clock=self.clock)
        else:
            verified = self.validate_bootstrap(deepcopy(p), under(self.session.directory, 'bootstrap'), tick)
        if not isinstance(verified, dict) or not verified:
            raise ValueError('nonempty domain restoration verification required')
        tick()
        self.bootstrap_receipt = _publish(under(self.ledger.directory, 'controls'), {
            'schema_version': VERSION+'-bootstrap-observation', 'ledger_root_sha256': self.ledger.root_sha256,
            'control_sha256': self.control['sha256'], 'native_observation': observed,
            'restoration_verification': verified, 'checked_through_ns': self.clock()})
        from .formal_partial_imports import VERSION as IMPORT_VERSION
        if verified.get('schema_version') == IMPORT_VERSION:
            tick()
            self.ledger.bind_partial_imports(verified['manifest_reference'],
                bootstrap_observation_sha256=self.bootstrap_receipt['sha256'])
        else:
            if concrete:
                raise ValueError('actual formal worker requires committed scientific import admission')
        self._healthy()
        self.meter.guard(check_memory=True)

    def _work(self, work):
        self._credit()
        allocation = self.ledger._state.reservation(work["work_id"])["reserved_ns"]
        start = self.meter.begin_work(allocation)
        reservation = self.ledger.reserve(work["work_id"])
        observed, validation, authorization, error_type = None, None, None, None
        status, result_sha = "failure", None
        try:
            # This callback is compulsory, but only the later full execution
            # seal/source review proves it is the real authorization consumer.
            authorization = _authority(self.authorize(deepcopy(work)), self.ledger, work)
            self.meter.guard()
            observed = self.session.run(reservation, started_ns=start)
            native_result = unpack(observed)
            expected = {"schema_version": native.VERSION+"-observation", "ledger_root_sha256": self.ledger.root_sha256,
                "reservation_sha256": reservation["reservation_sha256"], "work_id": work["work_id"],
                "started_ns": start, "deadline_ns": start+reservation["reserved_ns"]}
            if (any(type(native_result.get(k)) is not type(v) or native_result.get(k) != v for k, v in expected.items())
                    or native_result.get("status") not in budget.STATUSES
                    or budget._integer(native_result["ended_ns"]) > self.clock()
                    or budget._integer(native_result["elapsed_ns"]) != native_result["ended_ns"]-start):
                raise ValueError("native observation does not bind this exact measured work")
            budget._integer(native_result["sequence"])
            status = native_result["status"]
            if status == "success":
                output = under(self.session.directory, f"outputs/{native_result['sequence']:06d}")
                manifest = unpack(read_json(under(output, "result.json")), expected_sha256=native_result["result_sha256"])
                verified = verify_artifacts(self.validate(deepcopy(work), deepcopy(manifest["value"]), output), work, output)
                validation = _save_verification(self.ledger, work, verified)
                result_sha = native_result["result_sha256"]
            self.meter.guard(check_memory=True)
        except KeyboardInterrupt:
            self.meter.trip("operator_stop")
            status = "interrupted"
        except BaseException as exc:
            error_type = type(exc).__name__
            status = "failure"
            self.meter.trip("supervision_error")
        if status != "success" or self.meter.reason is not None:
            self.session.request_termination("supervision_error")
            closure = self.session.close()
            result_sha = None
        else:
            closure = None
        evidence = _publish(under(self.ledger.directory, "controls"), {"schema_version": VERSION+"-work-observation",
            "ledger_root_sha256": self.ledger.root_sha256, "control_sha256": self.control["sha256"],
            "reservation_sha256": reservation["reservation_sha256"], "work_id": work["work_id"],
            "started_ns": start, "checked_through_ns": self.clock(), "native_observation": observed,
            "work_authority": authorization, "scientific_manifest_validation": validation, "closure": closure,
            "candidate_status": status, "error_type": error_type, "stop_reason": self.meter.reason})
        end, elapsed = self.meter.end_work()  # Includes observation publication.
        if elapsed >= reservation["reserved_ns"] or end >= self.meter.deadline or self.meter.reason in {"work_deadline", "phase_deadline"}:
            status = "timeout"
        elif self.meter.reason == "operator_stop":
            status = "interrupted"
        elif self.meter.reason is not None:
            status = "failure"
        if status != "success":
            result_sha = None
        settlement = self.ledger.settle(reservation["reservation_sha256"], status=status, elapsed_ns=elapsed,
            completion_evidence_sha256=evidence["sha256"], result_sha256=result_sha,
            reason="controller measured through saved observation; separate prepaid control overhead")
        self.results.append({"work_id": work["work_id"], "status": status, "settlement_sha256": settlement["sha256"]})
        if status != "success":
            reason = "operator_stop" if status == "interrupted" else "resource_pressure" if self.meter.reason == "resource_pressure" else "supervision_error"
            self._halt(reason, evidence["sha256"])
            raise ControllerStopped("registered work stopped; no automatic retry")
        self._healthy()
        self.meter.guard()

    def _close_phase(self, *, last=False, stopped=False):
        if self.meter is None:
            return
        meter, key = self.meter, self.control["sha256"]
        try:
            if not stopped:
                self._credit()
            closure = self.session.close() if last or stopped else None
            evidence = _publish(under(self.ledger.directory, "controls"), {"schema_version": VERSION+"-phase-observation",
                "ledger_root_sha256": self.ledger.root_sha256, "control_sha256": key,
                "checked_through_ns": self.clock(), "closure": closure,
                "stop_reason": meter.reason, "control_observed_ns": meter.overhead(),
                "control_credit_ns": meter.credit})
            self.ledger.close_control(key, observed_ns=meter.overhead(), evidence_sha256=evidence["sha256"],
                reason="stopped" if stopped else "phase_complete")
            # Closing-record IO is still covered by the prepaid clock. A late
            # known overrun is retained after the closed span and halts admission.
            measured = meter.overhead()
            recorded = self.ledger._state.controls[key]["observed_ns"]
            if measured > max(meter.credit, recorded):
                self.ledger.retain_control_overrun(key, observed_ns=measured, evidence_sha256=evidence["sha256"])
            if not stopped:
                meter.guard()
        finally:
            if last or stopped:
                self._terminal_meter = (meter, key)
            meter.close()
            if not self.ledger._poisoned and self.ledger._state.controls[key]["closed"]:
                measured = meter.overhead()
                recorded = self.ledger._state.controls[key]["observed_ns"]
                if measured > max(meter.credit, recorded):
                    self.ledger.retain_control_overrun(key, observed_ns=measured,
                        evidence_sha256=self.ledger._state.controls[key]["evidence_sha256"])
            # Everything from this boundary to opening the next phase belongs
            # to that next phase. No unmetered inter-phase idle window.
            self._next_phase_start = self.clock()
            if not stopped:
                meter.guard()
            self.meter = None

    def finish_terminal(self, evidence_sha256):
        """Debit concrete launcher's terminal publication to the last phase.

        The receipt must already exist. This is not a new allowance, a retry,
        or a claim to measure Python/OS teardown after returning to the CLI.
        Last append IO is covered by the same prepaid credit and checked after
        it returns; an observed overrun is durable and cannot return success.
        """
        if (not self.finished or self._terminal_meter is None or self.meter is not None
                or self.ledger._poisoned or self.ledger._lock.stream.closed):
            raise ControllerStopped("measured terminal controller and held writer required")
        meter, key = self._terminal_meter
        closure = self.session.close()  # .closed alone is NOT proof of Job closure.
        if closure.get("process_tree_closed") is not True:
            raise native.UnclosedTree("terminal owned tree is not verified closed")
        record = self.ledger.finish_control_tail(key, observed_ns=meter.overhead(),
                                                evidence_sha256=evidence_sha256)
        measured = meter.overhead()
        recorded = self.ledger._state.controls[key]["observed_ns"]
        if measured > max(meter.credit, recorded):
            self.ledger.retain_control_overrun(key, observed_ns=measured, evidence_sha256=evidence_sha256)
        meter.guard()
        return record

    def run_all(self, phase_order, *, started_ns=None):
        if self.finished:
            raise ControllerStopped("controller instance is one-shot")
        if started_ns is not None:
            budget._integer(started_ns)
            if started_ns > self.clock():
                raise ValueError("entrypoint start cannot be in the future")
        self.finished = True
        self._next_phase_start = self.clock() if started_ns is None else started_ns
        self._initial_started_ns = started_ns
        contract = self.ledger._state.contract
        if (not isinstance(phase_order, (list, tuple)) or len(set(phase_order)) != len(phase_order)
                or set(phase_order) != set(contract["phase_caps_ns"])):
            raise ValueError("complete explicit phase order required")
        self._healthy()
        before = self.ledger.summary()
        if before["pending"] is not None or before["active_control_sha256"] is not None:
            raise ControllerStopped("recover the previous owned controller first")
        # Even a clean phase boundary is not a reason to overlap its old worker.
        if before["control_spans"]:
            prior_key = next(reversed(before["control_spans"]))
            prior = _control_record(self.ledger, prior_key)
            old = native._query_job(prior["worker_job_name"])
            if old["exists"] and old["accounting"]["active_processes"]:
                raise native.UnclosedTree("previous owned session is still active")
        dispositions = self.ledger.dispositions()
        if self.ledger._state.imported_success and self.validate_bootstrap is None:
            raise ControllerStopped('imported science requires explicit metered restoration')
        phases = [(phase, [dict(row) for row in contract["workloads"] if row["phase"] == phase
            and dispositions[row["work_id"]] == "unattempted"]) for phase in phase_order]
        phases = [(phase, rows) for phase, rows in phases if rows]
        try:
            if self.validate_bootstrap is not None:
                if (not self.ledger._state.imported_success or before['control_spans']
                        or set(k for k, v in dispositions.items() if v != 'unattempted')
                           != set(self.ledger._state.imported_success)
                        or any(dispositions[k] != 'success' for k in self.ledger._state.imported_success)
                        or phase_order[0] != contract['workloads'][0]['phase']):
                    raise ControllerStopped('partial restoration must precede all successor work')
                self._open_phase(phase_order[0])
                self._bootstrap()
                self._close_phase(last=not phases)
            for index, (phase, rows) in enumerate(phases):
                self._open_phase(phase)
                for work in rows:
                    self._work(work)
                self._close_phase(last=index == len(phases)-1)
            return {"schema_version": VERSION+"-result", "ledger_root_sha256": self.ledger.root_sha256,
                "settlements": deepcopy(self.results), "summary": self.ledger.summary(),
                "dispositions": self.ledger.dispositions()}
        except BaseException as exc:
            # A poisoned ledger cannot be repaired in-place or converted into a
            # success. Preserve pending reservations/control and close our tree.
            try:
                if self.meter is not None and self.ledger.summary()["pending"] is None:
                    self.meter.abort_unreserved_work()
                self.session.request_termination("supervision_error")
                self.session.close()
                if self.meter is not None:
                    if not self.ledger._poisoned and self.ledger.summary()["pending"] is None:
                        directory = under(self.ledger.directory, "controls")
                        directory.mkdir(exist_ok=True)
                        failure = _publish(directory, {
                            "schema_version": VERSION+"-stopped", "control_sha256": self.control["sha256"],
                            "control_start": self.control, "observed_ns": self.clock(),
                            "reason": self.meter.reason, "error_type": type(exc).__name__})
                        self._halt("supervision_error", failure["sha256"])
                        if self.ledger.summary()["active_control_sha256"] is not None:
                            self._close_phase(last=True, stopped=True)
            finally:
                if self.meter is not None:
                    self.meter.close()
                    self.meter = None
            raise
