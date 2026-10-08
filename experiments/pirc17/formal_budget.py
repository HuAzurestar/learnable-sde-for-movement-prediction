"""Durable one-attempt cumulative budget accounting for the formal executor.

No data reader, process launcher or authorization bypass lives here. Admission
must separately validate ACCEPT01 and pin this contract's canonical directory
in the execution/runtime binding. Supervision must supply real work-completion
evidence and monotonic elapsed time, not a caller-invented success flag.

Reserve the maximum permissible active time BEFORE launching work. Settle with
observed elapsed time only after a bound synchronous completion barrier. A
persistent owned worker may process multiple items: do not launch a fresh
Python/model/map process for every forecast. Each barrier must bind the exact
reservation and durable outputs, with no queued/asynchronous work. Recovery
after a lost barrier additionally requires closure of the old owned tree. If
an interrupted measurement is unavailable, retain the full reservation as a
conservative charge (not a claim that that CPU time was actually observed).
One replay on open, then O(1) bookkeeping per append; no quadratic rescans.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time

from .protocol_core import canonical, digest, envelope, read_json, sha256, under, unpack

VERSION = "pirc17-cumulative-phase-budget-v1"
NANOSECONDS = 1_000_000_000
HEAD_REPLACE_RETRY_SECONDS = 0.5
STATUSES = {"success", "failure", "timeout", "interrupted"}


def _fields(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("exact budget record fields required")


def _integer(value, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError("nonnegative integer nanoseconds/count required")
    return value


def validate_contract(value):
    """Fresh public validation; callers cannot supply a policy PASS or cache."""
    return _validate_contract(value)


def _validate_contract(value, *, checked_policy=None):
    """All contract predicates; optional policy is private same-call reuse.

    Only contract_for_matrix supplies the policy it just fully validated.
    This is not exposed on either public validation/projection interface.
    """
    has_resource_policy = isinstance(value, dict) and 'resource_policy' in value
    if checked_policy is not None and not has_resource_policy:
        raise ValueError('same-call policy requires its exact bound resource record')
    _fields(value, ("schema_version", "protocol_sha256", "execution_sha256", "matrix_sha256",
        "runtime_manifest_sha256", "approval_sha256", "ledger_directory", "phase_caps_ns",
        "total_cap_ns", "max_generated_forecasts", "max_attempts_per_item", "workloads",
        *(('resource_policy',) if has_resource_policy else ())))
    if value["schema_version"] != VERSION or type(value["max_attempts_per_item"]) is not int or value["max_attempts_per_item"] != 1:
        raise ValueError("fixed one-attempt budget schema required")
    for name in ("protocol_sha256", "execution_sha256", "matrix_sha256", "runtime_manifest_sha256", "approval_sha256"):
        sha256(value[name])
    directory = value["ledger_directory"]
    if not isinstance(directory, str) or not Path(directory).is_absolute() or str(Path(directory).resolve()) != directory:
        raise ValueError("exact canonical approved ledger directory required")
    phases = value["phase_caps_ns"]
    if not isinstance(phases, dict) or not phases or any(not isinstance(k, str) or not k for k in phases):
        raise ValueError("explicit phase budgets required")
    for cap in phases.values():
        _integer(cap, positive=True)
    if _integer(value["total_cap_ns"], positive=True) != sum(phases.values()):
        raise ValueError("total must equal the fixed phase budgets; no credit transfer")
    _integer(value["max_generated_forecasts"])
    rows = value["workloads"]
    if not isinstance(rows, list) or not rows or len(rows) > 100000:
        raise ValueError("bounded complete work inventory required")
    seen, calls, reduced_phase_rows = set(), 0, []
    for row in rows:
        _fields(row, ("work_id", "phase", "max_active_ns", "generated_forecasts"))
        key = sha256(row["work_id"])
        if key in seen or row["phase"] not in phases:
            raise ValueError("unique registered work IDs and phases required")
        seen.add(key)
        cap = _integer(row["max_active_ns"], positive=True)
        if cap > phases[row["phase"]]:
            if not has_resource_policy:
                raise ValueError("work cap exceeds its own phase")
            # Preserve the sealed nominal work descriptor when an explicitly
            # approved overlay REDUCES its phase cap. Actual reservation is
            # still min(nominal work cap, remaining phase, remaining total).
            reduced_phase_rows.append(row)
        calls += _integer(row["generated_forecasts"])
    additional = 0
    if has_resource_policy:
        if checked_policy is None:
            from .formal_resource_policy import validate_policy
            policy = validate_policy(value['resource_policy'])
        else:
            if canonical(unpack(value['resource_policy'])) != canonical(checked_policy):
                raise ValueError('same-call checked policy differs from the bound record')
            policy = checked_policy
        if any(row['max_active_ns'] > policy['original_phase_caps_ns'][row['phase']]
               for row in reduced_phase_rows):
            raise ValueError('nominal work cap exceeds the unchanged approved original phase')
        if (value['phase_caps_ns'] != policy['phase_caps_ns'] or value['total_cap_ns'] != policy['total_cap_ns']
                or value['protocol_sha256'] != policy['protocol_sha256'] or value['matrix_sha256'] != policy['matrix_sha256']
                or digest(rows) != policy['original_workloads_sha256']
                or calls != policy['original_generation_limit']
                or directory == policy['predecessor_ledger_directory']
                or value['execution_sha256'] == policy['predecessor_execution_sha256']
                or value['approval_sha256'] == policy['predecessor_approval_sha256']
                or value['runtime_manifest_sha256'] == policy['predecessor_runtime_manifest_sha256']):
            raise ValueError('resource successor must preserve exact science and use distinct registered authority')
        additional = policy['additional_generation_allowance']
    if calls + additional != value["max_generated_forecasts"]:
        raise ValueError("entire generated-forecast denominator must be registered")
    return value


def contract_for_matrix(matrix, *, protocol_sha256, execution_sha256, runtime_manifest_sha256,
                        approval_sha256, ledger_directory, resource_policy=None):
    """Pure projection; the caller must validate matrix and human scope first."""
    p = unpack(matrix)
    if p["protocol_sha256"] != protocol_sha256:
        raise ValueError("matrix names a different protocol")
    rows = [{"work_id": r["work_id"], "phase": r["phase"],
             "max_active_ns": _integer(r["max_active_seconds"], positive=True)*NANOSECONDS,
             "generated_forecasts": r["generated_forecasts"]} for r in p["workloads"]]
    phases = {k: _integer(v, positive=True)*NANOSECONDS for k, v in p["phase_caps_seconds"].items()}
    calls = p['max_generated_forecasts']
    extra, policy = {}, None
    if resource_policy is not None:
        from .formal_resource_policy import validate_policy
        policy = deepcopy(validate_policy(resource_policy))
        if policy['original_phase_caps_ns'] != phases:
            raise ValueError('resource reallocation must start from the unchanged original matrix')
        phases, calls = deepcopy(policy['phase_caps_ns']), policy['effective_generation_limit']
        extra = dict(resource_policy=deepcopy(resource_policy))
    return _validate_contract({"schema_version": VERSION, "protocol_sha256": protocol_sha256,
        "execution_sha256": execution_sha256, "matrix_sha256": matrix["sha256"],
        "runtime_manifest_sha256": runtime_manifest_sha256, "approval_sha256": approval_sha256,
        "ledger_directory": str(Path(ledger_directory).resolve()), "phase_caps_ns": phases,
        "total_cap_ns": sum(phases.values()), "max_generated_forecasts": calls,
        "max_attempts_per_item": 1, "workloads": rows, **extra}, checked_policy=policy)


def _startup_floor(runtime_manifest, contract):
    """Pure replay validation; live eligibility is checked by the writer API.

    No caller charge is accepted. The entire predecessor binding must be inside
    the contract's exact runtime digest, with unchanged phase/total/matrix caps.
    Replaying a new ledger must not reopen/query the historical process or data.
    Full runtime/approval semantics are still mandatory entrypoint checks.
    """
    from . import formal_carryover as costs, formal_predecessor as predecessor
    runtime = unpack(runtime_manifest, expected_sha256=contract['runtime_manifest_sha256'])
    if (runtime.get('protocol_sha256') != contract['protocol_sha256']
            or runtime.get('matrix_sha256') != contract['matrix_sha256']
            or runtime.get('ledger_directory') != contract['ledger_directory']):
        raise ValueError('startup floor must belong to this exact replacement runtime')
    binding = runtime.get('predecessor')
    b = unpack(binding)
    c = unpack(b['cost_snapshot'])
    from . import formal_input_interruption as interruption
    if b['schema_version'] == interruption.VERSION:
        interruption.validate_binding_scope(b)
        predecessor = interruption
    elif (b['schema_version'] != predecessor.VERSION or b['eligible_closed_input_startup'] is not True
          or b['process_tree_closed'] is not True or b['read_only'] is not True
          or b['authorizes_execution'] is not False or type(b['generated_forecasts']) is not int
          or b['generated_forecasts'] != 0 or type(b['final_eval_reads']) is not int or b['final_eval_reads'] != 0):
        raise ValueError('registered predecessor eligibility required')
    if (c['schema_version'] != costs.VERSION or c['read_only'] is not True or c['authorizes_execution'] is not False
            or c['ledger_root_sha256'] != predecessor.REGISTERED['expected_root_sha256']
            or c['head_sha256'] != predecessor.REGISTERED['expected_head_sha256']
            or c['terminal_proof_sha256'] != predecessor.REGISTERED['expected_terminal_proof_sha256']
            or c['protocol_sha256'] != contract['protocol_sha256'] or c['matrix_sha256'] != contract['matrix_sha256']
            or c['phase_caps_ns'] != contract['phase_caps_ns'] or c['total_cap_ns'] != contract['total_cap_ns']
            or c['execution_sha256'] == contract['execution_sha256'] or c['approval_sha256'] == contract['approval_sha256']
            or c['ledger_directory'] != b['ledger_directory'] or c['ledger_directory'] == contract['ledger_directory']
            or type(c['generated_forecasts_reserved']) is not int or c['generated_forecasts_reserved'] != 0
            or type(c['final_eval_reads']) is not int or c['final_eval_reads'] != 0
            or c['work_inventory_count'] != len(contract['workloads']) or c['halted_reason'] != 'supervision_error'
            or c['work_dispositions'] != {contract['workloads'][0]['work_id']:'failure'}
            or contract['workloads'][0]['phase'] != 'input_qualification_and_binding'
            or contract['workloads'][0]['generated_forecasts'] != 0):
        raise ValueError('registered predecessor and unchanged replacement budget scope required')
    phase = 'input_qualification_and_binding'
    names = ('charged_ns_by_phase', 'measured_ns_by_phase', 'conservatively_charged_ns_by_phase',
             'control_charged_ns_by_phase', 'control_observed_ns_by_phase')
    for name in names:
        _fields(c[name], contract['phase_caps_ns'])
        for key, amount in c[name].items():
            _integer(amount)
            if key != phase and amount != 0:
                raise ValueError('startup floor cannot import scientific phase costs')
    for key in contract['phase_caps_ns']:
        charge = c['charged_ns_by_phase'][key]
        if (charge != c['measured_ns_by_phase'][key]+c['conservatively_charged_ns_by_phase'][key]
                or charge > contract['phase_caps_ns'][key]
                or c['control_charged_ns_by_phase'][key] > charge
                or c['control_observed_ns_by_phase'][key] > c['control_charged_ns_by_phase'][key]):
            raise ValueError('startup floor categories/caps do not reconcile')
    total = _integer(c['charged_total_ns'], positive=True)
    if (total != sum(c['charged_ns_by_phase'].values()) or total > contract['total_cap_ns']
            or total != predecessor.REGISTERED_CHARGED_NS):
        raise ValueError('startup floor total differs from its original phase charges')
    return binding, c


def _sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _replace_head(temporary, path):
    """Bound transient Windows reader/AV sharing conflicts, not permissions.

    Retrying the same flushed head is not a new ledger event or work attempt.
    The writer remains held and all delay stays inside the caller's existing
    measured interval. Persistent denial still propagates and poisons it.
    Never fall back to truncation, delete, a different head, or an unbounded
    loop. Non-sharing errors and non-Windows replacement remain immediate.
    """
    deadline = time.monotonic()+HEAD_REPLACE_RETRY_SECONDS
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            remaining = deadline-time.monotonic()
            if os.name != "nt" or getattr(error, "winerror", None) not in {5, 32, 33} or remaining <= 0:
                raise
            time.sleep(min(0.05, remaining))


def _publish(path, payload, *, replace=False):
    """Fully flush a same-directory temp, then no-replace publish the event.

Only head.json is replaceable under the held writer lock. It witnesses the
latest immutable event so deletion of a committed suffix cannot look empty.
Pending temp files are never interpreted as authority. Filesystem/power-loss
guarantees remain those of fsync/_commit and the host, not zero-loss promises.
"""
    path = Path(path)
    record = envelope(payload)
    raw = canonical(record)+b"\n"
    fd, temporary = tempfile.mkstemp(prefix="."+path.name+".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            if path.name != "head.json" or path.is_symlink() or not path.is_file():
                raise ValueError("only the existing ledger head may be replaced")
            _replace_head(temporary, path)
        else:
            os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)  # This call's exact temp only.
    return record


class _WriterLock:
    """OS lock, released on process death; a stale filename grants no lease."""
    def __init__(self, path):
        if Path(path).is_symlink():
            raise ValueError("writer lock must not be a symlink")
        self.stream = Path(path).open("r+b")
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        if not self.stream.closed:
            self.stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()


class _State:
    def __init__(self, contract, root_sha256):
        self.contract = contract
        self.work = {r["work_id"]: r for r in contract["workloads"]}
        self.charged = dict.fromkeys(contract["phase_caps_ns"], 0)
        self.measured = dict(self.charged)
        self.conservative = dict(self.charged)
        self.status = {}
        self.pending = None
        self.generated = 0
        self.halted = None
        self.controls = {}
        self.active_control = None
        self.terminal_control = None
        self.control_charged = dict(self.charged)
        self.control_observed = dict(self.charged)
        self.predecessor_floor = None
        self.imported_success = {}
        self.partial_imports = None
        self.carried_generation_reservations = {}
        self.consumed_generation_transfers = {}
        self.count, self.tip = 0, root_sha256

    def control_credit(self, phase, amount):
        """Nonrefundable supervision credit, inside the original phase cap."""
        _integer(amount, positive=True)
        if 'resource_policy' in self.contract and self.predecessor_floor is None:
            raise ValueError('resource continuation requires its closed predecessor floor first')
        if self.terminal_control is not None:
            raise ValueError("terminal execution cannot buy new control credit")
        if phase not in self.charged or self.pending is not None:
            raise ValueError("control credit requires a registered idle phase")
        if self.halted is not None:
            raise TimeoutError("halted ledger cannot buy more control credit")
        remaining = min(self.contract["phase_caps_ns"][phase]-self.charged[phase],
                        self.contract["total_cap_ns"]-sum(self.charged.values()))
        if amount > remaining:
            raise TimeoutError("control credit cannot expand or borrow a phase budget")

    def observe_control(self, control, observed):
        if observed is None:
            return
        observed = _integer(observed)
        if observed < control["observed_ns"]:
            raise ValueError("control observation cannot roll its clock backward")
        self.control_observed[control["phase"]] += observed-control["observed_ns"]
        control["observed_ns"] = observed

    def reservation(self, work_id):
        if 'resource_policy' in self.contract and self.predecessor_floor is None:
            raise ValueError('resource continuation requires its closed predecessor floor first')
        if (self.predecessor_floor is not None
                and self.predecessor_floor.get('requires_metered_source_admission') is True
                and self.partial_imports is None):
            raise ValueError('metadata source requires complete metered domain admission before work reservation')
        if self.terminal_control is not None:
            raise ValueError("terminal execution cannot admit more work")
        if self.halted is not None:
            raise TimeoutError("ledger halted; no implicit budget extension")
        if self.pending is not None:
            raise RuntimeError("prior work has no verified completion or recovery closure")
        if work_id not in self.work or work_id in self.status:
            raise ValueError("unregistered/already attempted work; no retry")
        work = self.work[work_id]
        phase = work["phase"]
        if self.active_control is not None and self.controls[self.active_control]["phase"] != phase:
            raise ValueError("work cannot move outside its current control phase")
        available = min(work["max_active_ns"], self.contract["phase_caps_ns"][phase]-self.charged[phase],
                        self.contract["total_cap_ns"]-sum(self.charged.values()))
        # A qualified pre-dispatch predecessor token is retained, not refunded.
        # Its first REAL dispatch consumes that exact old token once. This is
        # not a second attempt at any dispatched or settled source work.
        generated_debit = 0 if work_id in self.carried_generation_reservations else work["generated_forecasts"]
        if available <= 0 or self.generated+generated_debit > self.contract["max_generated_forecasts"]:
            raise TimeoutError("registered phase/total/generation budget exhausted")
        return {"work_id": work_id, "phase": phase, "reserved_ns": available,
            "generated_forecasts": work["generated_forecasts"],
            "reservation_name": "PIRC17-FINAL-"+self.contract["execution_sha256"][:16]+f"-{self.count:06d}-"+work_id[:16]}

    def apply(self, event, record_sha256):
        _fields(event, ("schema_version", "index", "root_sha256", "previous_sha256", "type", "row"))
        if (event["schema_version"] != VERSION+"-event" or type(event["index"]) is not int
                or event["index"] != self.count or event["previous_sha256"] != self.tip):
            raise ValueError("budget event index/hash chain changed")
        row = event["row"]
        if ('resource_policy' in self.contract and self.count == 0
                and event['type'] != 'resource_predecessor'):
            raise ValueError('resource continuation requires its closed predecessor floor first')
        if event["type"] == "startup_predecessor":
            _fields(row, ('runtime_manifest',))
            if self.count != 0 or self.predecessor_floor is not None or self.pending is not None or self.controls:
                raise ValueError('startup predecessor must be the once-only first budget event')
            binding, c = _startup_floor(row['runtime_manifest'], self.contract)
            self.charged = dict(c['charged_ns_by_phase'])
            self.measured = dict(c['measured_ns_by_phase'])
            self.conservative = dict(c['conservatively_charged_ns_by_phase'])
            self.control_charged = dict(c['control_charged_ns_by_phase'])
            self.control_observed = dict(c['control_observed_ns_by_phase'])
            # No old work disposition, halt, control span, generation credit or
            # result is reused. Only a registered zero-fit/forecast interruption
            # is eligible; post-read input exposure stays in its distinct binding.
            self.predecessor_floor = dict(binding_sha256=binding['sha256'],
                ledger_root_sha256=c['ledger_root_sha256'], head_sha256=c['head_sha256'],
                terminal_proof_sha256=c['terminal_proof_sha256'], charged_total_ns=c['charged_total_ns'])
            if any(self.charged[k] >= cap for k, cap in self.contract['phase_caps_ns'].items()):
                self.halted = 'phase_cap_reached'
            if sum(self.charged.values()) >= self.contract['total_cap_ns']:
                self.halted = 'total_cap_reached'
        elif event["type"] in {"partial_predecessor", "resource_predecessor"}:
            _fields(row, ('runtime_manifest',))
            if self.count != 0 or self.predecessor_floor is not None or self.pending is not None or self.controls:
                raise ValueError('partial predecessor must be the once-only first budget event')
            is_resource = event['type'] == 'resource_predecessor'
            if is_resource:
                from .formal_resource_predecessor import resource_floor
                binding, c, s = resource_floor(row['runtime_manifest'], self.contract)
            else:
                from .formal_partial_predecessor import partial_floor
                binding, c, s = partial_floor(row['runtime_manifest'], self.contract)
            self.charged = dict(c['charged_ns_by_phase'])
            self.measured = dict(c['measured_ns_by_phase'])
            self.conservative = dict(c['conservatively_charged_ns_by_phase'])
            self.control_charged = dict(c['control_charged_ns_by_phase'])
            self.control_observed = dict(c['control_observed_ns_by_phase'])
            self.imported_success = deepcopy(s['completed_sources'])
            self.status = dict.fromkeys(self.imported_success, 'success')
            self.generated = c['generated_forecasts_reserved'] if is_resource else c['generation_reservations_retained']
            if not is_resource:
                pending = deepcopy(c['pending_reservation'])
                self.carried_generation_reservations = {pending['work_id']: pending}
            self.predecessor_floor = dict(binding_sha256=binding['sha256'],
                ledger_root_sha256=c['ledger_root_sha256'], head_sha256=c['head_sha256'],
                terminal_proof_sha256=c['terminal_proof_sha256'] if is_resource else c['terminal_sha256'],
                charged_total_ns=c['charged_total_ns'],
                kind='verified_resource_predecessor' if is_resource else 'verified_partial_predecessor',
                imported_success_count=len(self.imported_success))
            if is_resource:
                policy = unpack(unpack(binding)['resource_policy'])
                self.predecessor_floor.update(retry_work_id=policy['retry_work_id'],
                    retry_old_reservation_sha256=policy['retry_old_reservation_sha256'])
                from .formal_resource_predecessor import METADATA_VERSION, RECOVERY_VERSION, CONTINUATION_VERSION
                if unpack(binding)['schema_version'] in {METADATA_VERSION, RECOVERY_VERSION, CONTINUATION_VERSION}:
                    self.predecessor_floor['requires_metered_source_admission'] = True
            if any(self.charged[k] >= cap for k, cap in self.contract['phase_caps_ns'].items()):
                self.halted = 'phase_cap_reached'
            if sum(self.charged.values()) >= self.contract['total_cap_ns']:
                self.halted = 'total_cap_reached'
        elif event["type"] == "bind_partial_imports":
            from .formal_partial_imports import validate_binding
            self.partial_imports = validate_binding(self, row)
        elif event["type"] == "reserve":
            if not isinstance(row, dict) or canonical(row) != canonical(self.reservation(row.get("work_id"))):
                raise ValueError("reservation changed fixed workload/budget allocation")
            self.charged[row["phase"]] += row["reserved_ns"]
            carried = self.carried_generation_reservations.pop(row['work_id'],None)
            if carried is None:
                self.generated += row["generated_forecasts"]
            else:
                self.consumed_generation_transfers[row['work_id']] = dict(
                    predecessor_reservation_sha256=carried['reservation_sha256'],
                    successor_reservation_sha256=record_sha256)
            self.status[row["work_id"]] = "reserved"
            self.pending = {**row, "reservation_sha256": record_sha256}
        elif event["type"] == "settle":
            _fields(row, ("reservation_sha256", "status", "elapsed_ns", "completion_evidence_sha256",
                          "result_sha256", "reason"))
            pending = self.pending
            if pending is None or row["reservation_sha256"] != pending["reservation_sha256"] or row["status"] not in STATUSES:
                raise ValueError("settlement must close exactly the pending owned work")
            sha256(row["completion_evidence_sha256"])
            if not isinstance(row["reason"], str) or not row["reason"].strip() or len(row["reason"]) > 1024:
                raise ValueError("explicit bounded settlement reason required")
            if row["result_sha256"] is not None:
                sha256(row["result_sha256"])
            if row["status"] == "success" and row["result_sha256"] is None:
                raise ValueError("success requires immutable output identity")
            elapsed = row["elapsed_ns"]
            if elapsed is None:
                if row["status"] != "interrupted":
                    raise ValueError("unknown elapsed time cannot become a success or free budget")
                charge = pending["reserved_ns"]
            else:
                charge = _integer(elapsed)
                if charge >= pending["reserved_ns"] and row["status"] != "timeout":
                    raise ValueError("deadline reach/overrun must remain timeout, not success")
            phase = pending["phase"]
            self.charged[phase] += charge-pending["reserved_ns"]
            (self.conservative if elapsed is None else self.measured)[phase] += charge
            self.status[pending["work_id"]] = row["status"]
            self.pending = None
            if row["status"] == "timeout":
                self.halted = "work_deadline_reached"
            if self.charged[phase] >= self.contract["phase_caps_ns"][phase]:
                self.halted = "phase_cap_reached"
            if sum(self.charged.values()) >= self.contract["total_cap_ns"]:
                self.halted = "total_cap_reached"
        elif event["type"] == "halt":
            _fields(row, ("reason", "evidence_sha256"))
            sha256(row["evidence_sha256"])
            if row["reason"] not in {"resource_pressure", "supervision_error", "integrity_failure", "operator_stop"}:
                raise ValueError("registered terminal halt reason required")
            if self.halted is not None:
                raise ValueError("halt is terminal, not a new attempt")
            self.halted = row["reason"]
        elif event["type"] in {"control_open", "control_credit"}:
            _fields(row, ("control_sha256", "phase", "credit_ns", "observed_ns"))
            key = sha256(row["control_sha256"])
            self.control_credit(row["phase"], row["credit_ns"])
            observed = _integer(row["observed_ns"])
            if event["type"] == "control_open":
                if self.active_control is not None or key in self.controls or observed != 0:
                    raise ValueError("fresh exclusive control span required")
                control = {"phase": row["phase"], "credit_ns": 0, "observed_ns": 0, "closed": False}
            else:
                if key != self.active_control:
                    raise ValueError("credit must name the current control span")
                control = self.controls[key]
                if row["phase"] != control["phase"] or observed >= control["credit_ns"]:
                    raise ValueError("expired control credit cannot be retroactively extended")
                if observed < control["observed_ns"]:
                    raise ValueError("control observation cannot roll its clock backward")
            self.controls[key] = control
            self.active_control = key
            self.observe_control(control, observed)
            control["credit_ns"] += row["credit_ns"]
            self.charged[row["phase"]] += row["credit_ns"]
            self.conservative[row["phase"]] += row["credit_ns"]
            self.control_charged[row["phase"]] += row["credit_ns"]
            if self.charged[row["phase"]] >= self.contract["phase_caps_ns"][row["phase"]]:
                self.halted = "phase_cap_reached"
            if sum(self.charged.values()) >= self.contract["total_cap_ns"]:
                self.halted = "total_cap_reached"
        elif event["type"] == "control_close":
            _fields(row, ("control_sha256", "observed_ns", "evidence_sha256", "reason"))
            key = sha256(row["control_sha256"])
            sha256(row["evidence_sha256"])
            if key != self.active_control or self.pending is not None:
                raise ValueError("close exactly the idle current control span")
            if row["reason"] not in {"phase_complete", "stopped", "recovered_unknown"}:
                raise ValueError("explicit control closure disposition required")
            if (row["observed_ns"] is None) != (row["reason"] == "recovered_unknown"):
                raise ValueError("unknown overhead is recovery, never measured completion")
            control = self.controls[key]
            self.observe_control(control, row["observed_ns"])
            # Normally the prepaid amount covers closing-record IO as well.
            # Known overrun is retained and terminal, never silently clamped.
            overrun = max(0, control["observed_ns"]-control["credit_ns"])
            if overrun:
                self.charged[control["phase"]] += overrun
                self.control_charged[control["phase"]] += overrun
                self.measured[control["phase"]] += overrun
                self.halted = "control_credit_exhausted"
            if row["reason"] == "recovered_unknown":
                self.halted = "unknown_controller_interruption"
            control.update(closed=True, reason=row["reason"], evidence_sha256=row["evidence_sha256"])
            self.active_control = None
        elif event["type"] == "control_terminal_tail":
            _fields(row, ("control_sha256", "observed_ns", "evidence_sha256"))
            key = sha256(row["control_sha256"])
            sha256(row["evidence_sha256"])
            control = self.controls.get(key)
            if (control is None or not control["closed"] or self.active_control is not None
                    or self.pending is not None or key != next(reversed(self.controls))
                    or control["reason"] == "recovered_unknown"):
                raise ValueError("terminal tail requires the latest measured closed idle span")
            observed = _integer(row["observed_ns"])
            if observed < control["observed_ns"]:
                raise ValueError("terminal observation cannot roll its clock backward")
            extra = max(0, observed-max(control["credit_ns"], control["observed_ns"]))
            self.observe_control(control, observed)
            self.charged[control["phase"]] += extra
            self.control_charged[control["phase"]] += extra
            self.measured[control["phase"]] += extra
            if observed >= control["credit_ns"]:
                self.halted = "control_credit_exhausted"
            control["terminal_evidence_sha256"] = row["evidence_sha256"]
            self.terminal_control = key
        elif event["type"] == "control_late_overrun":
            _fields(row, ("control_sha256", "observed_ns", "evidence_sha256"))
            key = sha256(row["control_sha256"])
            sha256(row["evidence_sha256"])
            control = self.controls.get(key)
            if control is None or not control["closed"]:
                raise ValueError("late overhead observation requires a closed control span")
            observed = _integer(row["observed_ns"])
            if observed <= max(control["credit_ns"], control["observed_ns"]):
                raise ValueError("late event must retain a newly observed control overrun")
            extra = observed-max(control["credit_ns"], control["observed_ns"])
            self.observe_control(control, observed)
            self.charged[control["phase"]] += extra
            self.control_charged[control["phase"]] += extra
            self.measured[control["phase"]] += extra
            self.halted = "control_credit_exhausted"
        else:
            raise ValueError("unknown budget event")
        self.count += 1
        self.tip = record_sha256


class Ledger:
    """Single held writer; never reset counters by opening another attempt."""
    @classmethod
    def create(cls, directory, contract):
        validate_contract(contract)
        directory = Path(directory).resolve()
        if str(directory) != contract["ledger_directory"]:
            raise ValueError("changing directories cannot reset the approved ledger")
        directory.mkdir()  # Exclusive reservation, never exist_ok or overwrite.
        (directory/"events").mkdir()
        with (directory/"writer.lock").open("xb") as lock:
            lock.write(b"\0")
            lock.flush()
            os.fsync(lock.fileno())
        root = _publish(directory/"ledger.json", contract)
        _publish(directory/"head.json", {"root_sha256": root["sha256"], "event_count": 0, "last_event_sha256": root["sha256"]})
        return cls.open(directory, expected_root_sha256=root["sha256"])

    @classmethod
    def open(cls, directory, *, expected_root_sha256, expected_tip=None):
        directory = Path(directory).resolve()
        lock = _WriterLock(under(directory, "writer.lock"))
        try:
            root = read_json(under(directory, "ledger.json"))
            contract = validate_contract(unpack(root, expected_sha256=expected_root_sha256))
            if str(directory) != contract["ledger_directory"]:
                raise ValueError("ledger directory differs from its accepted identity")
            state = _State(contract, root["sha256"])
            head = unpack(read_json(under(directory, "head.json")))
            _fields(head, ("root_sha256", "event_count", "last_event_sha256"))
            _integer(head["event_count"])
            sha256(head["last_event_sha256"])
            if head["root_sha256"] != root["sha256"]:
                raise ValueError("ledger head names another root")
            tips = [root["sha256"]]
            files = sorted(under(directory, "events").iterdir())
            orphan_temps = []
            for path in files:
                if path.name.startswith(".") and path.name.endswith(".pending") and path.is_file() and not path.is_symlink():
                    orphan_temps.append(path.name)
                    continue
                if path.name != f"{state.count:06d}.json":
                    raise ValueError("budget event missing/reordered/extra/partial")
                event = read_json(under(directory, "events/"+path.name), max_bytes=64*1024)
                payload = unpack(event)
                if payload["root_sha256"] != root["sha256"]:
                    raise ValueError("budget event root identity changed")
                state.apply(payload, event["sha256"])
                tips.append(event["sha256"])
            if head["event_count"] >= len(tips) or tips[head["event_count"]] != head["last_event_sha256"]:
                raise ValueError("committed suffix deleted or ledger head altered")
            if expected_tip is not None:
                _fields(expected_tip, ("root_sha256", "event_count", "last_event_sha256"))
                count = _integer(expected_tip["event_count"])
                if (expected_tip["root_sha256"] != root["sha256"] or count >= len(tips)
                        or tips[count] != expected_tip["last_event_sha256"]):
                    raise ValueError("ledger is not a descendant of the independently pinned checkpoint")
            self = object.__new__(cls)
            self.directory, self._lock, self._state = directory, lock, state
            self.root_sha256, self._poisoned = root["sha256"], False
            self.recovered_unanchored_events = state.count-head["event_count"]
            self.orphan_temps = tuple(orphan_temps)
            # Crash after immutable event publication but before head update:
            # retain the event and its debit; never discard it or retry its work.
            if self.recovered_unanchored_events:
                _publish(under(directory, "head.json"), self.tip, replace=True)
            return self
        except BaseException:
            lock.close()
            raise

    def close(self):
        self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def tip(self):
        return {"root_sha256": self.root_sha256, "event_count": self._state.count, "last_event_sha256": self._state.tip}

    def summary(self):
        s = self._state
        return {**self.tip, "charged_ns_by_phase": dict(s.charged), "measured_ns_by_phase": dict(s.measured),
            "conservatively_charged_ns_by_phase": dict(s.conservative),
            "remaining_ns_by_phase": {k: max(0, v-s.charged[k]) for k, v in s.contract["phase_caps_ns"].items()},
            "remaining_total_ns": max(0, s.contract["total_cap_ns"]-sum(s.charged.values())),
            "generated_forecasts_reserved": s.generated, "pending": deepcopy(s.pending),
            "attempted_work_items": len(s.status), "unattempted_work_items": len(s.work)-len(s.status),
            "halted_reason": s.halted, "recovered_unanchored_events": self.recovered_unanchored_events,
            "orphan_temporary_files": list(self.orphan_temps),
            "control_charged_ns_by_phase": dict(s.control_charged),
            "control_observed_ns_by_phase": dict(s.control_observed),
            "active_control_sha256": s.active_control, "terminal_control_sha256": s.terminal_control,
            "predecessor_floor": deepcopy(s.predecessor_floor),
            "imported_success_count": len(s.imported_success),
            "partial_imports": deepcopy(s.partial_imports),
            "carried_generation_reservations": deepcopy(s.carried_generation_reservations),
            "consumed_generation_transfers": deepcopy(s.consumed_generation_transfers),
            "control_spans": deepcopy(s.controls)}

    def dispositions(self):
        return {key: self._state.status.get(key, "unattempted") for key in self._state.work}

    def _append(self, kind, row):
        if self._lock.stream.closed or self._poisoned:
            raise RuntimeError("closed/failed writer must not launch more work")
        event = {"schema_version": VERSION+"-event", "index": self._state.count,
            "root_sha256": self.root_sha256, "previous_sha256": self._state.tip, "type": kind, "row": deepcopy(row)}
        record = envelope(event)
        index = self._state.count
        # All transition validation precedes mutation; mutation precedes IO only
        # inside this held writer. Any publication failure poisons it permanently.
        self._state.apply(event, record["sha256"])
        try:
            _publish(under(self.directory, f"events/{index:06d}.json"), event)
            _publish(under(self.directory, "head.json"), self.tip, replace=True)
        except BaseException:
            self._poisoned = True
            raise
        return deepcopy(record)

    def import_startup_floor(self, runtime_manifest):
        """Verify actual predecessor now, then durably debit before any work.

        The fixed original matrix contract is unchanged; the approved runtime
        binds the full predecessor, not a free amount or user-supplied credit.
        This API itself grants no human permission. Production run/worker must
        separately require the bound first event; generic Ledger APIs cannot
        infer a runtime's semantics from its digest alone.
        """
        if self._lock.stream.closed or self._poisoned or self._state.count != 0:
            raise ValueError('healthy empty ledger required for once-only startup floor')
        binding, _ = _startup_floor(runtime_manifest, self._state.contract)
        from . import formal_input_interruption as interruption
        if unpack(binding)['schema_version'] == interruption.VERSION:
            interruption.verify_input_interruption(binding)
        else:
            from .formal_predecessor import verify_startup_predecessor
            verify_startup_predecessor(binding)
        return self._append('startup_predecessor', {'runtime_manifest': runtime_manifest})

    def reserve(self, work_id):
        row = self._state.reservation(work_id)
        record = self._append("reserve", row)
        return {**row, "reservation_sha256": record["sha256"], "ledger_root_sha256": self.root_sha256}

    def import_partial_floor(self, runtime_manifest):
        """Read-only source requalification before the once-only durable debit.

        Not a launcher, an approval bypass, or a way to retry dispatched work.
        The actual entrypoint must separately pin and qualify this new runtime
        and supervise/charge the startup inspection under the original caps.
        """
        if self._lock.stream.closed or self._poisoned or self._state.count != 0:
            raise ValueError('healthy empty ledger required for once-only partial import')
        from .formal_partial_predecessor import partial_floor, verify_partial_predecessor
        binding, _, _ = partial_floor(runtime_manifest,self._state.contract)
        verify_partial_predecessor(binding)
        return self._append('partial_predecessor', {'runtime_manifest':runtime_manifest})

    def bind_partial_imports(self, manifest_reference, *, bootstrap_observation_sha256):
        """Bind owned restoration without any new scientific debit/settlement."""
        return self._append('bind_partial_imports', dict(manifest_reference=manifest_reference,
            bootstrap_observation_sha256=bootstrap_observation_sha256))

    def import_resource_floor(self, runtime_manifest):
        """Retain closed-cap history once, after metered live source validation.

        This does not transfer the old failed token or grant launch authority.
        Every new dispatch, including the single approved timeout retry, buys
        its normal token and time reservation under the successor contract.
        """
        if self._lock.stream.closed or self._poisoned or self._state.count != 0:
            raise ValueError('healthy empty ledger required for once-only resource import')
        from .formal_resource_predecessor import resource_floor, verify_resource_predecessor
        binding, _, _ = resource_floor(runtime_manifest, self._state.contract)
        verify_resource_predecessor(binding)
        return self._append('resource_predecessor', {'runtime_manifest': runtime_manifest})

    def settle(self, reservation_sha256, *, status, elapsed_ns, completion_evidence_sha256, result_sha256, reason):
        """Only the bound supervisor may supply measured time/completion identity.

This API verifies accounting, not an IPC barrier or a physical process tree.
Native deadline/barrier/recovery enforcement is a mandatory executor layer.
"""
        return self._append("settle", {"reservation_sha256": reservation_sha256, "status": status,
            "elapsed_ns": elapsed_ns, "completion_evidence_sha256": completion_evidence_sha256,
            "result_sha256": result_sha256, "reason": reason})

    def halt(self, reason, evidence_sha256):
        return self._append("halt", {"reason": reason, "evidence_sha256": evidence_sha256})

    def open_control(self, control_sha256, *, phase, credit_ns):
        return self._append("control_open", {"control_sha256": control_sha256, "phase": phase,
            "credit_ns": credit_ns, "observed_ns": 0})

    def topup_control(self, control_sha256, *, credit_ns, observed_ns):
        control = self._state.controls[control_sha256]
        return self._append("control_credit", {"control_sha256": control_sha256, "phase": control["phase"],
            "credit_ns": credit_ns, "observed_ns": observed_ns})

    def close_control(self, control_sha256, *, observed_ns, evidence_sha256, reason):
        return self._append("control_close", {"control_sha256": control_sha256, "observed_ns": observed_ns,
            "evidence_sha256": evidence_sha256, "reason": reason})

    def retain_control_overrun(self, control_sha256, *, observed_ns, evidence_sha256):
        return self._append("control_late_overrun", {"control_sha256": control_sha256,
            "observed_ns": observed_ns, "evidence_sha256": evidence_sha256})

    def finish_control_tail(self, control_sha256, *, observed_ns, evidence_sha256):
        """Observe terminal IO within the old credit; never reopen a budget."""
        return self._append("control_terminal_tail", {"control_sha256": control_sha256,
            "observed_ns": observed_ns, "evidence_sha256": evidence_sha256})
