"""Eligibility of the single consumed, zero-science PIRC-17 startup attempt.

This is a read-only historical verifier, NOT an approval or retry API. Source
pins prevent callers from choosing a cheaper predecessor. Historical execution
bytes are not checked against today's repaired sources. All metadata is bound
to the old hash chain; native named-Job queries establish closure now. No writer,
process adoption/termination, raw data, or scientific adapter is used.
"""
import hashlib
from pathlib import Path

from . import formal_budget as budget, formal_carryover as costs, formal_session as native
from .protocol_core import canonical, digest, envelope, sha256, under, unpack

VERSION = 'pirc17-closed-startup-predecessor-v1'
HISTORICAL_ENTRY = 'pirc17-concrete-formal-entrypoint-v1'
# Independently audited original input-startup charge. Used only to retain a
# conservative claim if replacement initialization cannot create its ledger;
# it is NOT a fallback for eligibility, approval or permission to start work.
REGISTERED_CHARGED_NS = 40_423_000_000
REGISTERED = dict(
    expected_root_sha256='c0cf2a78fbe32f3cd21f75436f262311e17f4d1170b1a9dde881d1d2d586ec5c',
    expected_head_sha256='f5e09eff4877160951ed21320bdca5a0a4fa5be86ea856ef279a50982754d656',
    expected_terminal_proof_sha256='75146bfe3e61bee4becd6c64ec3353b96804e6bf166db5be9129f833c5134e08')
MAX_METADATA_NODES = 64


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inventory(directory):
    """Bound traversal as well as bytes; reject links, including directory links."""
    pending, found = [directory], {}
    while pending:
        for path in pending.pop().iterdir():
            _require(len(found) < MAX_METADATA_NODES, 'bounded startup metadata inventory required')
            _require(not path.is_symlink() and path.resolve().is_relative_to(directory),
                     'startup metadata links/escapes forbidden')
            relative = path.relative_to(directory).as_posix()
            found[relative] = 'directory' if path.is_dir() else 'file'
            if path.is_dir():
                pending.append(path)
            else:
                _require(path.is_file(), 'regular startup metadata required')
    return found


class _Reads:
    def __init__(self):
        self.files = {}

    def _remember(self, path, checksum, bound, json_record):
        if str(path) in self.files:
            _require(self.files[str(path)] == (checksum, bound, json_record), 'predecessor changed between linked reads')
        self.files[str(path)] = checksum, bound, json_record

    def read(self, path, expected=None, *, large=False):
        path = Path(path)
        _require(path.is_absolute() and str(path.resolve()) == str(path),
                 'canonical absolute predecessor metadata required')
        bound = costs.MAX_ROOT_BYTES if large else costs.MAX_RECORD_BYTES
        record, checksum = costs._read_bound(path, bound)
        self._remember(path, checksum, bound, True)
        unpack(record, expected_sha256=expected)
        return record

    def raw(self, path, bound):
        _require(not path.is_symlink() and path.is_file() and path.stat().st_size <= bound,
                 'bounded regular predecessor bytes required')
        with path.open('rb') as stream:
            raw = stream.read(bound+1)
        _require(len(raw) <= bound, 'predecessor bytes grew beyond their bound')
        self._remember(path, hashlib.sha256(raw).hexdigest(), bound, False)

    def verify_unchanged(self):
        for name, (checksum, bound, json_record) in self.files.items():
            if json_record:
                _, actual = costs._read_bound(Path(name), bound)
                _require(actual == checksum, 'predecessor bytes changed during eligibility inspection')
            else:
                self.raw(Path(name), bound)


def _closed_job(name):
    observation = native._query_job(name)  # QUERY only, not Create/Terminate/Open writer.
    _require(isinstance(observation, dict) and type(observation.get('exists')) is bool,
             'actual native closure observation required')
    if observation['exists']:
        accounting = observation.get('accounting')
        _require(isinstance(accounting, dict) and type(accounting.get('active_processes')) is int
                 and accounting['active_processes'] == 0, 'predecessor owned Job still active or unverified')
    else:
        _require(observation.get('accounting') is None, 'absent native Job has invalid accounting')
    return observation


def inspect_startup_predecessor(directory):
    """Prove fixed zero-science eligibility and costs, without authorizing a run.

    The returned byte binding can be sealed in a replacement runtime. Reinspect
    immediately before admission: a saved closure flag is not current liveness.
    No timestamp participates in the binding; native closure is queried twice.
    Observed changes are rejected, not repaired. This is not a hostile-owner OS
    atomic snapshot or a substitute for new exact-version human acceptance.
    """
    supplied = Path(directory)
    _require(supplied.is_absolute(), 'absolute predecessor ledger required')
    directory = supplied.resolve(strict=True)
    reads = _Reads()
    before = _inventory(directory)
    claim = under(directory.parent, directory.name+'.launch')
    claim_before = _inventory(claim)
    _require(claim_before == {'start.json':'file', 'terminal.json':'file'},
             'exact consumed startup claim required')
    snapshot = costs.inspect_closed_ledger(directory, **REGISTERED)
    c = unpack(snapshot)
    _require(c['charged_total_ns'] == REGISTERED_CHARGED_NS, 'registered startup charge changed')
    _require(c['event_types'] == ['control_open','reserve','settle','halt','control_close','control_terminal_tail']
             and c['generated_forecasts_reserved'] == 0 and c['halted_reason'] == 'supervision_error',
             'only the closed first input-startup failure is eligible')
    root = reads.read(under(directory, 'ledger.json'), c['ledger_root_sha256'], large=True)
    contract = unpack(root)
    events = [reads.read(under(directory, f'events/{i:06d}.json')) for i in range(6)]
    rows = [unpack(event)['row'] for event in events]
    opened, reservation, settlement, halted, closed, tail = rows
    work_id = reservation['work_id']
    phase = 'input_qualification_and_binding'
    _require(c['work_dispositions'] == {work_id:'failure'} and reservation['phase'] == phase
             and reservation['generated_forecasts'] == 0 and settlement['status'] == 'failure'
             and settlement['result_sha256'] is None and halted['evidence_sha256'] == settlement['completion_evidence_sha256']
             and all(c['charged_ns_by_phase'][k] == 0 for k in c['phase_caps_ns'] if k != phase),
             'scientific/partial/recovered attempt cannot become a startup predecessor')
    terminal_record = reads.read(under(claim, 'terminal.json'))
    terminal = unpack(terminal_record)
    _require(digest(dict(content_sha256=terminal_record['sha256'],
             file_sha256=reads.files[str(under(claim,'terminal.json'))][0])) == c['terminal_proof_sha256'],
             'terminal meaning/bytes do not match the pinned tail proof')
    _require(terminal['schema_version'] == HISTORICAL_ENTRY+'-launch-terminal'
             and terminal['ledger_root_sha256'] == root['sha256'] and terminal['process_tree_closed'] is True
             and terminal['candidate_complete'] is False and terminal['approval_verified'] is True
             and terminal['startup_transferred_to_ledger'] is True and terminal['startup_conservative_charge_ns'] == 0
             and terminal['accounting_complete'] is False and terminal['human_accepted'] is False
             and terminal['scientific_claim_authorized'] is False
             and terminal['ledger_tip_before_terminal'] == dict(root_sha256=root['sha256'],event_count=5,
                 last_event_sha256=events[4]['sha256']), 'historical launch terminal differs from closed startup')
    launch = unpack(reads.read(under(claim,'start.json'), terminal['launch_sha256']))
    _require(launch['schema_version'] == HISTORICAL_ENTRY+'-launch' and launch['ledger_directory'] == str(directory)
             and launch['startup_phase'] == phase and launch['startup_reservation_ns'] == contract['phase_caps_ns'][phase]
             and launch['incomplete_claim_is_terminal'] is True and launch['final_eval_authorized'] is False,
             'historical launch claim changed')
    bundle_record = reads.read(Path(launch['bundle_path']), terminal['bundle_sha256'], large=True)
    _require(bundle_record['sha256'] == launch['bundle_sha256']
             and reads.files[launch['bundle_path']][0] == launch['bundle_file_sha256'], 'historical bundle bytes changed')
    bundle = unpack(bundle_record)
    _require(bundle['schema_version'] == HISTORICAL_ENTRY+'-bundle', 'historical bundle version changed')
    for name, key in [('protocol','protocol_sha256'),('execution','execution_sha256'),
                      ('matrix','matrix_sha256'),('runtime','runtime_manifest_sha256')]:
        unpack(bundle[name], expected_sha256=contract[key])
    matrix, runtime = unpack(bundle['matrix']), unpack(bundle['runtime'])
    expected = budget.contract_for_matrix(bundle['matrix'], protocol_sha256=contract['protocol_sha256'],
        execution_sha256=contract['execution_sha256'], runtime_manifest_sha256=contract['runtime_manifest_sha256'],
        approval_sha256=contract['approval_sha256'], ledger_directory=directory)
    _require(canonical(expected) == canonical(contract) and matrix['workloads'][0]['work_id'] == work_id
             and runtime['ledger_directory'] == str(directory), 'historical complete matrix/runtime contract changed')
    authority = launch['authority']
    _require(authority['approval_sha256'] == contract['approval_sha256']
             and authority['journal_directory'] == str(under(directory,'access')), 'historical authority linkage changed')
    approval = unpack(reads.read(Path(authority['approval_path']), contract['approval_sha256']))
    # Historical only: do not ask today's repaired sources to match old TEST
    # evidence or turn the consumed old approval into replacement permission.
    _require(approval['schema_version'] == 'pirc17-human-accept01-v1' and approval['task'] == 'ACCEPT-01'
             and approval['decision'] == 'CONFIRMED' and approval['human_confirmation']['kind'] == 'user-message',
             'pinned historical human decision required')
    for key, task in [('test','TEST-01'),('review','REVIEW-01')]:
        check = unpack(reads.read(Path(authority[key+'_path']), approval[key+'_sha256']))
        _require(check['schema_version'] == 'pirc17-pre-eval-check-v1' and check['task'] == task
                 and check['result'] == 'PASS' and check['protocol_sha256'] == contract['protocol_sha256']
                 and check['execution_sha256'] == contract['execution_sha256'], 'historical check linkage changed')
    _require(approval['protocol_sha256'] == contract['protocol_sha256']
             and approval['execution_sha256'] == contract['execution_sha256'], 'historical approval scope changed')
    control_record = reads.read(under(directory,'controls/'+sha256(opened['control_sha256'])+'.json'),opened['control_sha256'])
    control = unpack(control_record)
    session_directory = under(directory,'session-000000')
    _require(control['schema_version'] == 'pirc17-cumulative-controller-v1-control'
             and control['ledger_root_sha256'] == root['sha256'] and control['phase'] == phase
             and control['started_ns'] == launch['started_ns'] and control['session_directory'] == str(session_directory),
             'first startup control/session linkage changed')
    session_record = reads.read(under(session_directory,'session.json'))
    session = unpack(session_record)
    from .formal_entrypoint import worker_command
    _require(session == dict(schema_version=native.VERSION,directory=str(session_directory),
             job_name=control['worker_job_name'],worker_command=control['worker_command'],ledger_directory=str(directory),
             ledger_root_sha256=root['sha256'],execution_sha256=contract['execution_sha256'])
             and session['worker_command'] == worker_command(launch['bundle_path'],bundle_record['sha256'],authority,bundle['runtime']),
             'actual owned session/command differs from historical sealed launch')
    native._job_name(session['job_name'])
    dispatch = unpack(reads.read(under(directory,'dispatches/'+events[1]['sha256']+'.json')))
    _require(dispatch == dict(schema_version=native.VERSION+'-dispatch',ledger_root_sha256=root['sha256'],
             reservation_sha256=events[1]['sha256'],work_id=work_id,session_directory=str(session_directory),job_name=session['job_name']),
             'first durable dispatch claim changed')
    request_record = reads.read(under(session_directory,'requests/000000.json'))
    request = unpack(request_record)
    _require(request['schema_version'] == native.VERSION+'-request' and request['sequence'] == 0
             and request['session_sha256'] == session_record['sha256'] and request['previous_barrier_sha256'] == session_record['sha256']
             and request['reservation_sha256'] == events[1]['sha256'] and request['reservation_event_index'] == 1
             and request['work_id'] == work_id and 0 <= request['started_ns'] < request['deadline_ns']
             <= request['started_ns']+reservation['reserved_ns'], 'first request/reservation binding changed')
    barrier_record = reads.read(under(session_directory,'barriers/000000.json'))
    barrier = unpack(barrier_record)
    _require(barrier == dict(schema_version=native.VERSION+'-barrier',sequence=0,session_sha256=session_record['sha256'],
             previous_barrier_sha256=session_record['sha256'],request_sha256=request_record['sha256'],
             reservation_sha256=events[1]['sha256'],work_id=work_id,worker_pid=barrier['worker_pid'],
             status='failure',result_sha256=None,error_type='ValueError'), 'first factory failure barrier changed')
    ready = unpack(reads.read(under(session_directory,'ready.json')))
    _require(ready == dict(schema_version=native.VERSION+'-ready',session_sha256=session_record['sha256'],
             worker_pid=barrier['worker_pid']) and budget._integer(barrier['worker_pid'],positive=True), 'actual ready/worker identity changed')
    observed = unpack(reads.read(under(directory,'controls/'+settlement['completion_evidence_sha256']+'.json'),
                                 settlement['completion_evidence_sha256']))
    n = unpack(observed['native_observation'])
    _require(observed['schema_version'] == 'pirc17-cumulative-controller-v1-work-observation'
             and observed['ledger_root_sha256'] == root['sha256'] and observed['control_sha256'] == control_record['sha256']
             and observed['work_id'] == work_id and observed['reservation_sha256'] == events[1]['sha256']
             and observed['candidate_status'] == 'failure' and observed['scientific_manifest_validation'] is None
             and observed['work_authority'] == dict(schema_version='pirc17-cumulative-controller-v1-work-authority',
                 ledger_root_sha256=root['sha256'],execution_sha256=contract['execution_sha256'],approval_sha256=contract['approval_sha256'],
                 work_id=work_id,granted=True), 'historical controller/work authority changed')
    _require(n['schema_version'] == native.VERSION+'-observation' and n['ledger_root_sha256'] == root['sha256']
             and n['reservation_sha256'] == events[1]['sha256'] and n['work_id'] == work_id
             and n['sequence'] == 0 and n['session_sha256'] == session_record['sha256'] and n['barrier_sha256'] == barrier_record['sha256']
             and n['worker_pid'] == barrier['worker_pid'] and n['status'] == 'failure' and n['result_sha256'] is None
             and n['started_ns'] == request['started_ns'] and n['deadline_ns'] == request['deadline_ns']
             and n['elapsed_ns'] == n['ended_ns']-n['started_ns'] == settlement['elapsed_ns']
             and n['closure']['process_tree_closed'] is True and n['closure']['job_name'] == session['job_name']
             and type(n['closure']['accounting']['active_processes']) is int and n['closure']['accounting']['active_processes'] == 0,
             'actual measured native failure/closure linkage changed')
    phase_observation = unpack(reads.read(under(directory,'controls/'+closed['evidence_sha256']+'.json'),closed['evidence_sha256']))
    _require(phase_observation['schema_version'] == 'pirc17-cumulative-controller-v1-phase-observation'
             and phase_observation['ledger_root_sha256'] == root['sha256'] and phase_observation['control_sha256'] == control_record['sha256']
             and phase_observation['control_credit_ns'] == opened['credit_ns'] and phase_observation['control_observed_ns'] == closed['observed_ns']
             and phase_observation['closure']['process_tree_closed'] is True and phase_observation['closure']['job_name'] == session['job_name'],
             'actual phase shutdown linkage changed')
    # Every file in the old ledger must be accounted for; in particular access,
    # runtime, result, partial output, a second request, or a second session fail.
    expected_files = {Path(name).relative_to(directory).as_posix() for name in reads.files if Path(name).is_relative_to(directory)}
    expected_files.update(['head.json','writer.lock'])
    extra_controls = [name for name,kind in before.items() if kind == 'file' and name.startswith('controls/') and name not in expected_files]
    _require(len(extra_controls) == 1, 'exact stopped-controller receipt required')
    stopped_record = reads.read(under(directory,extra_controls[0]))
    stopped = unpack(stopped_record)
    _require(extra_controls[0] == 'controls/'+stopped_record['sha256']+'.json'
             and stopped['schema_version'] == 'pirc17-cumulative-controller-v1-stopped'
             and stopped['control_sha256'] == control_record['sha256'] and stopped['control_start'] == control_record,
             'stopped-controller receipt linkage changed')
    expected_files.add(extra_controls[0])
    expected_dirs = {'controls','dispatches','events','session-000000',
                     'session-000000/requests','session-000000/barriers','session-000000/outputs'}
    _require(before == {**dict.fromkeys(expected_files,'file'),**dict.fromkeys(expected_dirs,'directory')},
             'access/runtime/output/extra startup metadata makes predecessor ineligible')
    # Bind head bytes as well as its content. writer.lock is inert historical
    # binary metadata: never open it as a writer lock or change it.
    reads.read(under(directory,'head.json'), c['head_sha256'])
    reads.raw(under(directory,'writer.lock'), 16)
    _closed_job(session['job_name'])
    reads.verify_unchanged()
    after_costs = costs.inspect_closed_ledger(directory, **REGISTERED)
    _require(after_costs == snapshot and _inventory(directory) == before and _inventory(claim) == claim_before,
             'predecessor costs/inventory changed during eligibility inspection')
    closure_now = _closed_job(session['job_name'])
    return envelope(dict(schema_version=VERSION,ledger_directory=str(directory),cost_snapshot=snapshot,
        source_file_sha256={name:checksum for name,(checksum,_,_) in reads.files.items()},
        bundle_sha256=bundle_record['sha256'],launch_sha256=terminal['launch_sha256'],terminal_sha256=terminal_record['sha256'],
        native_job_name=session['job_name'],native_job_observation=closure_now,process_tree_closed=True,
        eligible_closed_input_startup=True,final_eval_reads=0,generated_forecasts=0,read_only=True,authorizes_execution=False))


def verify_startup_predecessor(binding):
    """Recheck source pins, recorded bytes and actual closure; never trust a flag."""
    value = unpack(binding)
    _require(value.get('schema_version') == VERSION, 'exact startup predecessor binding required')
    fresh = inspect_startup_predecessor(value['ledger_directory'])
    _require(canonical(fresh) == canonical(binding), 'sealed predecessor binding changed since registration')
    return unpack(fresh)
