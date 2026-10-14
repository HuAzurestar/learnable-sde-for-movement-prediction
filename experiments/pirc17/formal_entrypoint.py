"""Fixed formal prepare/run and owned-worker CLI; never an approval generator.

No direct-data, reduced-matrix, retry, thread-count or forecast-parameter
override. Preparation reads metadata/code only. A run consumes one exclusive
launch claim, even if initialization fails before its ledger can be created.
Legacy partial/resource restoration launches are retired: continue the saved
experiment with checkpoint_resume, without restoration-wide admission.
"""
import time
_MODULE_STARTED_NS = time.monotonic_ns()

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from .protocol_core import canonical, digest, envelope, file_hash, publish, read_json, sha256, under, unpack

VERSION = 'pirc17-concrete-formal-entrypoint-v3'
PARTIAL_VERSION = 'pirc17-concrete-formal-entrypoint-v4'
RESOURCE_VERSION = 'pirc17-concrete-formal-entrypoint-v5'
MODULE = 'experiments.pirc17.formal_entrypoint'
ENTRYPOINT = 'PSDE-SDE/experiments/pirc17/formal_entrypoint.py'
PHASE_ORDER = ('input_qualification_and_binding', 'method_training', 'terrain_training',
    'method_forecasts', 'terrain_forecasts', 'offline_common_scores',
    'mechanism_and_paired_inference', 'runtime_and_forecast_replay',
    'independent_saved_output_reanalysis', 'aggregate_export_and_integrity')
INPUT_PATHS = ('release', 'snapshot', 'data_root', 'trajectory_path', 'development_eligibility_path')


def is_restoration(runtime):
    return runtime.get('schema_version') in {PARTIAL_VERSION+'-runtime', RESOURCE_VERSION+'-runtime'}


def _runtime_version(runtime):
    if runtime.get('schema_version') == RESOURCE_VERSION+'-runtime':
        return RESOURCE_VERSION
    return PARTIAL_VERSION if runtime.get('schema_version') == PARTIAL_VERSION+'-runtime' else VERSION


def runtime_resource_policy(runtime):
    """Resource overlay is explicit in v5 only; historical seals stay strict.

    Execution's original budget fields describe the unchanged scientific plan.
    The bound v5 runtime and actual ledger separately enforce the approved
    phase reallocation and exactly one additional charged failed-item attempt.
    """
    if runtime.get('schema_version') != RESOURCE_VERSION+'-runtime':
        return None
    from .formal_resource_predecessor import VERSION as PREDECESSOR_VERSION, METADATA_VERSION, RECOVERY_VERSION, CONTINUATION_VERSION
    predecessor = unpack(runtime['predecessor'])
    if predecessor.get('schema_version') not in {PREDECESSOR_VERSION, METADATA_VERSION, RECOVERY_VERSION, CONTINUATION_VERSION}:
        raise ValueError('v5 requires the distinct closed-cap resource predecessor')
    return predecessor['resource_policy']


def budget_contract(bundle, *, approval_sha256):
    """Complete actual ledger projection, never a caller-selected override."""
    from . import formal_budget as budget
    p, e, m, r = (bundle[k] for k in ('protocol', 'execution', 'matrix', 'runtime'))
    runtime = unpack(r)
    return budget.contract_for_matrix(m, protocol_sha256=p['sha256'], execution_sha256=e['sha256'],
        runtime_manifest_sha256=r['sha256'], approval_sha256=approval_sha256,
        ledger_directory=runtime['ledger_directory'], resource_policy=runtime_resource_policy(runtime))


def _fields(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('exact formal launch fields required')


def _absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or str(Path(value).resolve()) != value:
        raise ValueError('canonical absolute formal location required')
    return value


def runtime_binding(protocol, matrix, environment, *, ledger_directory, input_paths, predecessor):
    """Bind the one ledger/location/phase order BEFORE human authorization.

    Only path syntax is examined here. Raw input files are verified by the
    guarded input work; no final-eval index, position or feature is read here.
    """
    _fields(input_paths, INPUT_PATHS)
    value = envelope(dict(schema_version=VERSION+'-runtime', protocol_sha256=protocol['sha256'],
        matrix_sha256=matrix['sha256'], environment=deepcopy(environment),
        ledger_directory=str(Path(ledger_directory).resolve()),
        input_paths={k: str(Path(v).resolve()) for k, v in input_paths.items()},
        working_directory=str(Path(__file__).resolve().parents[2]),
        worker_module=MODULE, phase_order=list(PHASE_ORDER), predecessor=deepcopy(predecessor), final_eval_authorized=False))
    validate_runtime_binding(value, protocol, matrix)
    return value


def validate_runtime_binding(value, protocol, matrix):
    from . import formal_budget as budget
    from .formal_environment import VERSION as ENVIRONMENT_VERSION
    p, m, r = unpack(protocol), unpack(matrix), unpack(value)
    partial = is_restoration(r)
    version = _runtime_version(r)
    _fields(r, ('schema_version', 'protocol_sha256', 'matrix_sha256', 'environment',
        'ledger_directory', 'input_paths', 'working_directory', 'worker_module', 'phase_order', 'predecessor', 'final_eval_authorized',
        *(('predecessor_reference',) if partial else ())))
    e = unpack(r['environment'])
    if (r['schema_version'] != version+'-runtime' or r['protocol_sha256'] != protocol['sha256']
            or r['matrix_sha256'] != matrix['sha256'] or m['protocol_sha256'] != protocol['sha256']
            or r['worker_module'] != MODULE or r['phase_order'] != list(PHASE_ORDER)
            or set(PHASE_ORDER) != set(m['phase_caps_seconds']) or r['final_eval_authorized'] is not False
            or e['schema_version'] != ENVIRONMENT_VERSION or e['protocol_sha256'] != protocol['sha256']
            or e['runtime_policy_sha256'] != digest(p['resource_contract']['runtime_binding'])
            or e['final_eval_authorized'] is not False or e['empirical_operations'] != 0 or e['final_eval_reads'] != 0):
        raise ValueError('runtime binding differs from fixed protocol/matrix/entrypoint')
    _fields(r['input_paths'], INPUT_PATHS)
    for path in (r['ledger_directory'], r['working_directory'], *r['input_paths'].values()): _absolute(path)
    if r['working_directory'] != str(Path(__file__).resolve().parents[2]):
        raise ValueError('formal source root moved since sealing')
    # Syntax/scope only, not live eligibility or human permission. The exact
    # matrix projection retains original caps; no phase or generation reset.
    contract = budget.contract_for_matrix(matrix, protocol_sha256=protocol['sha256'],
        execution_sha256=digest(dict(schema_validation_only=value['sha256'])),
        runtime_manifest_sha256=value['sha256'], approval_sha256=digest('SCHEMA VALIDATION NOT HUMAN AUTHORITY'),
        ledger_directory=r['ledger_directory'], resource_policy=runtime_resource_policy(r))
    if partial:
        from .formal_partial_launch import validate_partial_runtime
        validate_partial_runtime(value, contract, protocol=protocol, matrix=matrix)
        return r
    budget._startup_floor(value, contract)
    from .formal_input_interruption import validate_binding_scope
    # v1/v2 remain historical readers only. New preparation/worker/callbacks
    # cannot fall back to the cheaper zero-read interruption.
    validate_binding_scope(unpack(r['predecessor']))
    _absolute(unpack(r['predecessor'])['ledger_directory'])
    return r


def initial_floor_record(runtime_manifest, *, ledger_root_sha256):
    """Constant-size immutable first-event linkage, also for closed consumers."""
    from . import formal_budget as budget, formal_carryover as costs
    runtime = unpack(runtime_manifest)
    record, checksum = costs._read_bound(under(runtime['ledger_directory'],'events/000000.json'),costs.MAX_RECORD_BYTES)
    p = unpack(record)
    kind = ('resource_predecessor' if runtime['schema_version'] == RESOURCE_VERSION+'-runtime'
            else 'partial_predecessor' if runtime['schema_version'] == PARTIAL_VERSION+'-runtime'
            else 'startup_predecessor')
    expected = dict(schema_version=budget.VERSION+'-event',index=0,root_sha256=ledger_root_sha256,
        previous_sha256=ledger_root_sha256,type=kind,row=dict(runtime_manifest=runtime_manifest))
    if canonical(p) != canonical(expected):
        raise ValueError('exact first predecessor debit is missing or replaced')
    return record, checksum


def verify_initial_floor(runtime_manifest, *, contract, ledger_root_sha256, require_first_pending=False):
    """Verify the actual anchored startup prefix, never recover a writer/head.

    Called before Controller, callbacks' first environment, and worker factory.
    Not per forecast: live result/closed consumers need only the immutable first
    event linkage after the accepted ledger's budget replay has validated it.
    """
    from . import formal_budget as budget, formal_carryover as costs
    directory = Path(unpack(runtime_manifest)['ledger_directory'])
    root, root_bytes = costs._read_bound(under(directory,'ledger.json'),costs.MAX_ROOT_BYTES)
    actual = unpack(root, expected_sha256=ledger_root_sha256)
    if canonical(actual) != canonical(contract):
        raise ValueError('floor belongs to a different actual ledger contract')
    first, first_bytes = initial_floor_record(runtime_manifest,ledger_root_sha256=ledger_root_sha256)
    head, head_bytes = costs._read_bound(under(directory,'head.json'),costs.MAX_RECORD_BYTES)
    h = unpack(head)
    budget._fields(h,('root_sha256','event_count','last_event_sha256'))
    count = budget._integer(h['event_count'],positive=True)
    if h['root_sha256'] != ledger_root_sha256 or count > costs.MAX_EVENTS:
        raise ValueError('anchored bounded startup floor required')
    state = budget._State(contract,ledger_root_sha256)
    for index in range(count):
        record = first if index == 0 else costs._read_bound(under(directory,f'events/{index:06d}.json'),costs.MAX_RECORD_BYTES)[0]
        event = unpack(record)
        if event['root_sha256'] != ledger_root_sha256:
            raise ValueError('startup prefix root changed')
        state.apply(event,record['sha256'])
    if state.tip != h['last_event_sha256'] or state.predecessor_floor is None:
        raise ValueError('head does not anchor the mandatory predecessor debit')
    if state.halted is not None:
        raise ValueError('halted startup prefix cannot initialize a replacement worker')
    if require_first_pending:
        if state.pending is None or state.pending['work_id'] != contract['workloads'][0]['work_id']:
            raise ValueError('actual head must anchor the pending first input work, not a rolled-back floor-only prefix')
    elif state.count != 1 or state.pending is not None or state.controls:
        raise ValueError('controller construction requires the sole initial debit, not pre-existing work/control')
    for name,bound,checksum in [('ledger.json',costs.MAX_ROOT_BYTES,root_bytes),
            ('head.json',costs.MAX_RECORD_BYTES,head_bytes),('events/000000.json',costs.MAX_RECORD_BYTES,first_bytes)]:
        if costs._read_bound(under(directory,name),bound)[1] != checksum:
            raise ValueError('startup floor/root/head changed during verification')
    return first['sha256']


def validate_bundle(record):
    """Check actual full source/matrix/runtime links; no research data access."""
    from . import formal_budget as budget
    from .formal_matrix import PROTOCOL_SHA256, validate_matrix
    from .protocol import validate_execution, validate_protocol
    value = unpack(record)
    _fields(value, ('schema_version', 'protocol', 'execution', 'matrix', 'runtime'))
    version = _runtime_version(unpack(value['runtime']))
    if value['schema_version'] != version+'-bundle':
        raise ValueError('formal launch bundle version differs')
    p, e, m, r = (value[k] for k in ('protocol', 'execution', 'matrix', 'runtime'))
    validate_protocol(p, expected_sha256=PROTOCOL_SHA256)
    execution = validate_execution(e, p)
    validate_matrix(m, p)
    runtime = validate_runtime_binding(r, p, m)
    if (execution['entrypoint'] != ENTRYPOINT or execution['matrix_sha256'] != m['sha256']
            or execution['runtime_manifest_sha256'] != r['sha256']
            or execution['budget_ledger_schema_version'] != budget.VERSION):
        raise ValueError('exact executable, complete matrix and concrete runtime binding required')
    return value, runtime


def authority_paths(*, approval_path, approval_sha256, test_path, review_path, ledger_directory):
    return dict(approval_path=str(Path(approval_path).resolve()), approval_sha256=sha256(approval_sha256),
        test_path=str(Path(test_path).resolve()), review_path=str(Path(review_path).resolve()),
        journal_directory=str(under(ledger_directory, 'access')))


def _approval(authority, protocol, execution):
    from .final_eval_guard import validate_approval
    approval, test, review = (read_json(authority[k]) for k in ('approval_path', 'test_path', 'review_path'))
    validate_approval(approval, expected_sha256=authority['approval_sha256'],
        protocol=protocol, execution=execution, test=test, review=review)


def worker_command(bundle_path, bundle_sha256, authority, runtime):
    return [unpack(unpack(runtime)['environment'])['python_executable']['path'], '-u', '-m', MODULE,
        'worker', '--bundle', str(Path(bundle_path).resolve()), '--bundle-sha256', sha256(bundle_sha256),
        '--approval', authority['approval_path'], '--approval-sha256', authority['approval_sha256'],
        '--test', authority['test_path'], '--review', authority['review_path']]


def launch_directory(ledger_directory):
    ledger = Path(_absolute(str(Path(ledger_directory).resolve())))
    return under(ledger.parent, ledger.name+'.launch')


def prepare(*, output_directory, ledger_directory, input_paths, predecessor_ledger):
    """Seal actual code, environment and full matrix, without acquiring data."""
    from . import formal_budget as budget
    from .formal_environment import ConfiguredRuntime
    from .formal_matrix import build_matrix, load_protocol
    from .protocol import EXECUTION_VERSION, source_catalog
    from .formal_input_interruption import inspect_input_interruption
    ledger = Path(ledger_directory).resolve()
    claim = launch_directory(ledger)
    output = Path(output_directory).resolve()
    if (ledger.exists() or claim.exists() or output.is_relative_to(ledger)
            or output.is_relative_to(claim)):
        raise ValueError('prepare requires an unused distinct ledger and launch location')
    protocol = load_protocol()
    matrix = build_matrix(protocol)
    predecessor = inspect_input_interruption(predecessor_ledger)
    environment = ConfiguredRuntime(protocol).identity()
    runtime = runtime_binding(protocol, matrix, environment, ledger_directory=ledger, input_paths=input_paths,
                              predecessor=predecessor)
    p = unpack(protocol)
    execution = envelope(dict(schema_version=EXECUTION_VERSION, protocol_sha256=protocol['sha256'],
        source_sha256=source_catalog(), matrix_sha256=matrix['sha256'], runtime_manifest_sha256=runtime['sha256'],
        budget_ledger_schema_version=budget.VERSION, entrypoint=ENTRYPOINT,
        phase_caps_seconds=p['resource_contract']['phase_caps_seconds'], max_generated_forecasts=11513,
        scientific_forecasts=11020, method_required_slots=sorted(k for k, v in
            p['components']['method_mechanisms']['slots'].items() if v['disposition'] == 'REQUIRED'),
        terrain_configurations=sorted(p['components']['terrain_configurations']), final_eval_authorized=False))
    record = envelope(dict(schema_version=VERSION+'-bundle', protocol=protocol,
        execution=execution, matrix=matrix, runtime=runtime))
    validate_bundle(record)
    return publish(output, unpack(record))


def run(*, bundle_path, bundle_sha256, approval_path, approval_sha256, test_path, review_path,
        started_ns=None):
    """One concrete launch; failures are terminal, not an implicit resume.

    Missing CLI inputs are rejected before claiming a run. Once the pinned
    bundle locates its ledger, the exclusive sibling claim is never removed.
    Before transfer to a controller span, the remaining original input cap is
    retained conservatively IN ADDITION to the registered predecessor debit.
    Neither charge disappears on a pre-ledger or disk failure. On a live
    successful transfer, that same startup interval is included exactly once
    by run_all(started_ns=...). No research adapter precedes real approval.
    """
    started_ns = time.monotonic_ns() if started_ns is None else started_ns
    from . import formal_budget as budget, formal_controller as control
    from .formal_input_interruption import REGISTERED, REGISTERED_CHARGED_NS
    budget._integer(started_ns)
    if started_ns > time.monotonic_ns():
        raise ValueError('launch start cannot be in the future')
    for path in (approval_path, test_path, review_path): read_json(path)
    record = read_json(bundle_path)
    candidate = unpack(record, expected_sha256=bundle_sha256)
    runtime = unpack(candidate['runtime'])
    partial = is_restoration(runtime)
    if partial:
        # Do not consume another claim, replay old ledgers, restore all old
        # arrays or start the legacy controller/credit watchdog at restart.
        # The old preparation/receipt readers remain for historical evidence.
        raise ValueError('legacy restoration run is retired; use experiments.pirc17.checkpoint_resume run '
                         '--directory <existing-checkpoint-directory>; do not repeat init')
    version = _runtime_version(runtime)
    directory = Path(_absolute(runtime['ledger_directory']))
    authority = authority_paths(approval_path=approval_path, approval_sha256=approval_sha256,
        test_path=test_path, review_path=review_path, ledger_directory=directory)
    if directory.exists():
        raise FileExistsError('formal ledger already exists; no reset or resume')
    claim = launch_directory(directory)
    input_cap = 3600*budget.NANOSECONDS  # Original phase unless an exact approved resource overlay reallocates it.
    if partial:
        from .formal_partial_launch import launch_floor
        predecessor_charges, predecessor_root = launch_floor(candidate, approval_sha256=approval_sha256)
        if version == RESOURCE_VERSION:
            # The complete contract validates the bound resource policy and
            # original work inventory. This is a cumulative phase cap, NOT
            # fresh startup time: retain every predecessor charge below.
            input_cap = budget_contract(candidate, approval_sha256=approval_sha256)['phase_caps_ns'][PHASE_ORDER[0]]
    else:
        predecessor_charges = {p: REGISTERED_CHARGED_NS if p == PHASE_ORDER[0] else 0 for p in PHASE_ORDER}
        predecessor_root = REGISTERED['expected_root_sha256']
    predecessor_total = sum(predecessor_charges.values())
    prefix_cap = input_cap-predecessor_charges[PHASE_ORDER[0]]
    budget._integer(prefix_cap,positive=True)
    claim.mkdir()  # Exclusive, before live source/environment/authority checks.
    launch = budget._publish(claim/'start.json', dict(schema_version=version+'-launch',
        bundle_path=str(Path(bundle_path).resolve()), bundle_sha256=record['sha256'],
        bundle_file_sha256=file_hash(bundle_path), ledger_directory=str(directory),
        authority=authority, started_ns=started_ns, launcher_pid=os.getpid(),
        startup_phase=PHASE_ORDER[0], startup_reservation_ns=prefix_cap,
        registered_predecessor_root_sha256=predecessor_root,
        predecessor_charged_ns_by_phase=predecessor_charges,
        incomplete_claim_is_terminal=True, final_eval_authorized=False))
    ledger = runner = None
    failure = None
    candidate_complete = False
    tree_closed = False
    approval_verified = False
    terminal = None
    try:
        bundle, runtime = validate_bundle(record)
        if (str(Path.cwd().resolve()) != runtime['working_directory']
                or str(Path(sys.executable).resolve()) !=
                    unpack(runtime['environment'])['python_executable']['path']):
            raise ValueError('run must use the sealed working directory and interpreter')
        _approval(authority, bundle['protocol'], bundle['execution'])
        approval_verified = True
        contract = budget_contract(bundle, approval_sha256=authority['approval_sha256'])
        if contract['phase_caps_ns'][PHASE_ORDER[0]] != input_cap:
            raise ValueError('launch reservation differs from approved effective input phase cap')
        ledger = budget.Ledger.create(directory, contract)
        if version == RESOURCE_VERSION: ledger.import_resource_floor(bundle['runtime'])
        elif partial: ledger.import_partial_floor(bundle['runtime'])
        else: ledger.import_startup_floor(bundle['runtime'])
        verify_initial_floor(bundle['runtime'],contract=contract,ledger_root_sha256=ledger.root_sha256)
        callbacks = RuntimeCallbacks(ledger, bundle=record, authority=authority)
        runner = control.Controller(ledger, worker_command(bundle_path, bundle_sha256, authority, bundle['runtime']),
                                    authorize_work=callbacks.authorize, validate_result=callbacks.validate,
                                    **(dict(validate_bootstrap=callbacks.bootstrap) if partial else {}))
        callbacks.session = runner.session
        runner.run_all(runtime['phase_order'], started_ns=started_ns)
        candidate_complete = set(ledger.dispositions().values()) == {'success'}
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if runner is not None:
                runner.session.request_termination('supervision_error' if failure else 'operator_stop')
                closure = runner.session.close()
                tree_closed = closure.get('process_tree_closed') is True
                if not tree_closed:
                    raise control.native.UnclosedTree('launch did not close its actual owned tree')
            else:
                tree_closed = True  # No Controller/OwnedSession was ever constructed.
            # A poisoned writer may have applied an event in memory without
            # publishing it. It cannot prove transfer of the launch debit.
            floor_transferred = (ledger is not None and not ledger._poisoned
                and ledger.summary()['predecessor_floor'] is not None)
            transferred = floor_transferred and bool(ledger._state.controls)
            elapsed = time.monotonic_ns()-started_ns
            new_startup_charge = 0 if transferred else max(prefix_cap,elapsed)
            retained = predecessor_total if ledger is None else sum(ledger.summary()['charged_ns_by_phase'].values())
            cumulative_charge = retained if transferred else max(retained, predecessor_total+new_startup_charge)
            terminal = budget._publish(claim/'terminal.json', dict(schema_version=version+'-launch-terminal',
                launch_sha256=launch['sha256'], bundle_sha256=record['sha256'],
                ledger_root_sha256=None if ledger is None else ledger.root_sha256,
                ledger_tip_before_terminal=None if ledger is None else ledger.tip,
                candidate_complete=candidate_complete and failure is None, error_type=None if failure is None else type(failure).__name__,
                process_tree_closed=tree_closed, approval_verified=approval_verified,
                elapsed_through_receipt_ns=elapsed, startup_transferred_to_ledger=transferred,
                predecessor_charged_ns_by_phase=predecessor_charges,
                predecessor_floor_transferred_to_ledger=floor_transferred,
                predecessor_charge_location='ledger' if floor_transferred else 'claim',
                startup_conservative_charge_ns=new_startup_charge,
                cumulative_charge_lower_bound_ns=cumulative_charge,
                accounting_complete=False, human_accepted=False, scientific_claim_authorized=False))
            # This digest binds both meaning and bytes of the actual terminal
            # receipt. A receipt alone never proves the subsequent accounting.
            proof = digest(dict(content_sha256=terminal['sha256'], file_sha256=file_hash(claim/'terminal.json')))
            if (runner is not None and runner._terminal_meter is not None and not ledger._poisoned
                    and ledger._state.pending is None and ledger._state.active_control is None):
                runner.finish_terminal(proof)
            elif failure is None:
                raise control.ControllerStopped('terminal accounting unavailable')
            if ledger is not None and ledger.summary()['halted_reason'] is not None and failure is None:
                raise control.ControllerStopped('terminal ledger is halted')
        except BaseException as exc:
            if failure is None: failure = exc
        finally:
            if ledger is not None: ledger.close()
    if failure is not None:
        raise failure
    if not candidate_complete or terminal is None or not tree_closed:
        raise control.ControllerStopped('full formal inventory did not complete')
    return dict(terminal_path=str(claim/'terminal.json'), terminal_sha256=terminal['sha256'],
        ledger_root_sha256=ledger.root_sha256, ledger_tip=ledger.tip,
        execution_complete=True, human_accepted=False, scientific_claim_authorized=False)


class RuntimeWorker:
    """Actual FormalWorker plus effective-runtime checks inside reservations."""
    def __init__(self, session, contract, *, bundle_path, bundle_sha256, authority):
        from . import formal_budget as budget, formal_session as native
        # Absent authority fails BEFORE environment setup or any raw adapter.
        for key in ('approval_path', 'test_path', 'review_path'):
            read_json(authority[key])
        bundle = read_json(bundle_path)
        unpack(bundle, expected_sha256=bundle_sha256)
        self.bundle, runtime = validate_bundle(bundle)
        self.partial = is_restoration(runtime)
        self.version = _runtime_version(runtime)
        p, e, m = (self.bundle[k] for k in ('protocol', 'execution', 'matrix'))
        _approval(authority, p, e)
        expected = budget_contract(self.bundle, approval_sha256=authority['approval_sha256'])
        if (canonical(expected) != canonical(contract) or session['ledger_directory'] != runtime['ledger_directory']
                or authority['journal_directory'] != str(under(runtime['ledger_directory'], 'access'))
                or session['worker_command'] != worker_command(bundle_path, bundle_sha256, authority, self.bundle['runtime'])
                or str(Path.cwd().resolve()) != runtime['working_directory']):
            raise ValueError('worker command, real ledger or current directory differs from sealed runtime')
        if self.partial:
            # serve already checked membership and this actual bootstrap;
            # rebind its prefix without repeating the parent's full old-science
            # scan or demanding a NEW input reservation for completed work.
            request = native._read(under(session['directory'], 'bootstrap-request.json'))
            native._bootstrap_request(request, session_sha256=digest(session), session=session, contract=contract)
            self.floor_event_sha256 = initial_floor_record(self.bundle['runtime'],
                ledger_root_sha256=session['ledger_root_sha256'])[0]['sha256']
            options = dict(partial_predecessor_reference=runtime['predecessor_reference'])
        else:
            from .formal_input_interruption import verify_input_interruption
            verify_input_interruption(runtime['predecessor'])
            self.floor_event_sha256 = verify_initial_floor(self.bundle['runtime'],contract=contract,
                ledger_root_sha256=session['ledger_root_sha256'],require_first_pending=True)
            options = dict(continuation_binding=runtime['predecessor'])
        from .formal_environment import ConfiguredRuntime
        self.environment = ConfiguredRuntime(p)
        self.environment.verify(runtime['environment'])
        from .formal_worker import FormalWorker
        self.worker = FormalWorker(session, contract, input_options=dict(protocol=p, execution=e, matrix=m,
            authority=authority, **options, **runtime['input_paths']))
        self.directory = under(session['directory'], 'runtime')
        self.directory.mkdir()
        native._write(self.directory/'binding.json', runtime)
        self.start = native._write(self.directory/'start.json', dict(schema_version=self.version+'-worker-runtime',
            session_sha256=digest(session), ledger_root_sha256=session['ledger_root_sha256'],
            execution_sha256=e['sha256'], runtime_manifest_sha256=self.bundle['runtime']['sha256'],
            worker_pid=os.getpid(), environment=runtime['environment'],predecessor_floor_event_sha256=self.floor_event_sha256))
        self.phases = set()

    def bootstrap(self, output_directory):
        from . import formal_session as native
        if not self.partial:
            raise ValueError('partial runtime required for saved-science restoration')
        try:
            before = self.environment.check()
            result = self.worker.bootstrap(output_directory)
            after = self.environment.check()
            request = native._read(self.directory.parent/'bootstrap-request.json')
            native._write(self.directory/'bootstrap.json', dict(schema_version=self.version+'-bootstrap-runtime',
                startup_sha256=self.start['sha256'], request_sha256=request['sha256'], worker_pid=os.getpid(),
                before_sha256=digest(before), after_sha256=digest(after)))
            return result
        except BaseException:
            self.worker.close()
            raise

    def __call__(self, work, output_directory):
        from . import formal_session as native
        try:
            before = self.environment.check()
            result = self.worker(work, output_directory)
            after = self.environment.check()
            if work['phase'] not in self.phases:
                native._write(self.directory/(digest(work['phase'])+'.json'), dict(
                    schema_version=self.version+'-phase-runtime', startup_sha256=self.start['sha256'],
                    phase=work['phase'], first_work_id=work['work_id'], worker_pid=os.getpid(),
                    before_sha256=digest(before), after_sha256=digest(after)))
                self.phases.add(work['phase'])
            return result
        except BaseException:
            self.worker.close()
            raise


def read_runtime_evidence(session_directory, *, contract, session_sha256, ledger_root_sha256,
                          worker_pid, phase, first_work_id):
    """Reread immutable runtime bytes, without measuring historical hardware.

    Used by the live controller and the independent closed-output consumer.
    The accepted ledger pins the complete runtime binding, not a caller's
    declaration of the expected environment. No interpreter/DLL/data reload.
    """
    from . import formal_session as native
    from .formal_environment import VERSION as ENVIRONMENT_VERSION
    root = under(session_directory, 'runtime')
    binding_path, start_path = root/'binding.json', root/'start.json'
    phase_path = root/('bootstrap.json' if phase is None else digest(phase)+'.json')
    binding = native._read(binding_path, contract['runtime_manifest_sha256'])
    runtime = unpack(binding)
    # Historical v1/v2 evidence remains readable, never runnable. Both v2/v3
    # saved evidence bind their actual first debit, not today's source version.
    from .formal_predecessor import HISTORICAL_ENTRY
    historical = runtime['schema_version'] == HISTORICAL_ENTRY+'-runtime'
    fields = ('schema_version','protocol_sha256','matrix_sha256','environment','ledger_directory',
        'input_paths','working_directory','worker_module','phase_order','final_eval_authorized')
    partial = is_restoration(runtime)
    _fields(runtime,fields if historical else (*fields,'predecessor',*(('predecessor_reference',) if partial else ())))
    from .formal_input_interruption import HISTORICAL_ENTRY as INPUT_HISTORICAL_ENTRY
    saved_version = runtime['schema_version'].removesuffix('-runtime')
    if saved_version not in {HISTORICAL_ENTRY, INPUT_HISTORICAL_ENTRY, VERSION, PARTIAL_VERSION, RESOURCE_VERSION}:
        raise ValueError('unsupported saved formal runtime version')
    environment = unpack(runtime['environment'])
    if (runtime['schema_version'] != saved_version+'-runtime' or runtime['worker_module'] != MODULE
            or runtime['protocol_sha256'] != contract['protocol_sha256']
            or runtime['matrix_sha256'] != contract['matrix_sha256']
            or runtime['ledger_directory'] != contract['ledger_directory']
            or runtime['final_eval_authorized'] is not False
            or not isinstance(runtime['phase_order'], list)
            or len(runtime['phase_order']) != len(set(runtime['phase_order']))
            or set(runtime['phase_order']) != set(contract['phase_caps_ns'])
            or environment['schema_version'] != ENVIRONMENT_VERSION
            or environment['protocol_sha256'] != contract['protocol_sha256']
            or environment['final_eval_authorized'] is not False):
        raise ValueError('saved runtime binding differs from actual accepted ledger scope')
    _fields(runtime['input_paths'], INPUT_PATHS)
    for path in (*runtime['input_paths'].values(), runtime['working_directory'], runtime['ledger_directory']):
        _absolute(path)
    start = native._read(start_path)
    p = unpack(start)
    expected = dict(schema_version=saved_version+'-worker-runtime', session_sha256=session_sha256,
        ledger_root_sha256=ledger_root_sha256, execution_sha256=contract['execution_sha256'],
        runtime_manifest_sha256=contract['runtime_manifest_sha256'], worker_pid=worker_pid,
        environment=runtime['environment'])
    if not historical:
        floor,_ = initial_floor_record(binding,ledger_root_sha256=ledger_root_sha256)
        expected['predecessor_floor_event_sha256'] = floor['sha256']
    if canonical(p) != canonical(expected):
        raise ValueError('worker startup runtime does not bind the real owned session')
    record = native._read(phase_path)
    r = unpack(record)
    hardware = unpack(expected['environment'])['hardware']
    if phase is None:
        request = native._read(under(session_directory,'bootstrap-request.json'))
        expected_marker = dict(schema_version=saved_version+'-bootstrap-runtime',startup_sha256=start['sha256'],
            request_sha256=request['sha256'],worker_pid=worker_pid,before_sha256=digest(hardware),after_sha256=digest(hardware))
        if not partial or first_work_id is not None or canonical(r) != canonical(expected_marker):
            raise ValueError('bootstrap runtime differs from actual saved-science restoration')
    elif first_work_id is None or canonical(r) != canonical(dict(schema_version=saved_version+'-phase-runtime',
            startup_sha256=start['sha256'], phase=phase, first_work_id=first_work_id,
            worker_pid=worker_pid, before_sha256=digest(hardware), after_sha256=digest(hardware))):
        raise ValueError('phase runtime differs from effective sealed hardware or registered first work')
    return {'binding': dict(content_sha256=binding['sha256'], file_sha256=file_hash(binding_path)),
            'startup': dict(content_sha256=start['sha256'], file_sha256=file_hash(start_path)),
            'phase': dict(content_sha256=record['sha256'], file_sha256=file_hash(phase_path))}


def runtime_evidence(session_directory, *, bundle, session_sha256, ledger_root_sha256, worker_pid, phase,
                     first_work_id=None):
    """Controller checks the same actual files against its validated bundle."""
    runtime = unpack(bundle['runtime'])
    if first_work_id is None:
        works = [w for w in unpack(bundle['matrix'])['workloads'] if w['phase'] == phase]
        first_work_id = works[0]['work_id'] if works else None
    contract = dict(protocol_sha256=bundle['protocol']['sha256'], execution_sha256=bundle['execution']['sha256'],
        matrix_sha256=bundle['matrix']['sha256'], runtime_manifest_sha256=bundle['runtime']['sha256'],
        ledger_directory=runtime['ledger_directory'], phase_caps_ns={p: None for p in runtime['phase_order']})
    return read_runtime_evidence(session_directory, contract=contract, session_sha256=session_sha256,
        ledger_root_sha256=ledger_root_sha256, worker_pid=worker_pid, phase=phase, first_work_id=first_work_id)


class RuntimeCallbacks:
    """Lazy real controller callbacks; heavy setup is in the first reservation.

    The concrete launcher attaches its actual OwnedSession after constructing
    Controller. Effective runtime is checked on both sides of each result, and
    saved worker evidence is bound into the durable domain-verification proof.
    """
    def __init__(self, ledger, *, bundle, authority):
        self.ledger, self.bundle, self.authority = ledger, bundle, authority
        self.results = self.environment = self.session = None
        self.failed = False

    def _start(self):
        if self.failed:
            raise ValueError('failed concrete callbacks cannot restart')
        if self.results is not None: return
        from . import formal_session as native
        from .formal_environment import ConfiguredRuntime
        from .formal_results import ScientificResults
        bundle, runtime = validate_bundle(self.bundle)
        if is_restoration(runtime):
            raise ValueError('partial callbacks require actual metered bootstrap before unfinished science')
        if (not isinstance(self.session, native.OwnedSession) or self.session.ledger is not self.ledger
                or runtime['ledger_directory'] != str(self.ledger.directory)):
            raise ValueError('actual owned session and exact sealed ledger required')
        _approval(self.authority, bundle['protocol'], bundle['execution'])
        from .formal_input_interruption import verify_input_interruption
        verify_input_interruption(runtime['predecessor'])
        verify_initial_floor(bundle['runtime'],contract=self.ledger._state.contract,
                             ledger_root_sha256=self.ledger.root_sha256,require_first_pending=True)
        self.environment = ConfiguredRuntime(bundle['protocol'])
        self.environment.verify(runtime['environment'])
        self.bound = bundle
        self.first_work = {}
        for work in unpack(bundle['matrix'])['workloads']:
            self.first_work.setdefault(work['phase'], work['work_id'])
        self.results = ScientificResults(self.ledger, protocol=bundle['protocol'],
            execution=bundle['execution'], matrix=bundle['matrix'], authority=self.authority)

    def bootstrap(self, observation, output, tick):
        try:
            from .formal_partial_launch import bootstrap_callbacks
            return bootstrap_callbacks(self, observation, output, tick)
        except BaseException:
            self.failed = True
            raise

    def authorize(self, work):
        try:
            self._start()
            self.environment.check()
            return self.results.authorize(work)
        except BaseException:
            self.failed = True
            raise

    def validate(self, work, manifest, directory):
        try:
            self._start()
            self.environment.check()
            evidence = runtime_evidence(self.session.directory, bundle=self.bound,
                session_sha256=self.session.session['sha256'], ledger_root_sha256=self.ledger.root_sha256,
                worker_pid=self.session.worker_pid, phase=work['phase'], first_work_id=self.first_work[work['phase']])
            value = self.results.validate(work, manifest, directory)
            if work['phase'] == PHASE_ORDER[0]:
                from .formal_input_interruption import verify_continued_manifest
                value['details']['input_continuation'] = verify_continued_manifest(manifest, directory)
            self.environment.check()
            value['details']['effective_runtime'] = evidence
            return value
        except BaseException:
            self.failed = True
            raise


def main(argv=None, *, _started_ns=None):
    started = time.monotonic_ns() if _started_ns is None else _started_ns
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('environment', help='seal effective local environment; no research inputs')
    inspect.add_argument('--output-directory', type=Path, required=True)
    preparation = sub.add_parser('prepare', help='seal full local execution; no approval or research input reads')
    for name in ('output-directory', 'ledger-directory', 'predecessor-ledger', *(k.replace('_', '-') for k in INPUT_PATHS)):
        preparation.add_argument('--'+name, type=Path, required=True)
    partial_preparation = sub.add_parser('prepare-partial', help='seal saved-science continuation; no generation or approval')
    for name in ('output-directory','ledger-directory','predecessor-reference'):
        partial_preparation.add_argument('--'+name, type=Path, required=True)
    resource_preparation = sub.add_parser('prepare-resource', help='seal approved closed-cap continuation; no generation or approval')
    for name in ('output-directory','ledger-directory','predecessor-reference'):
        resource_preparation.add_argument('--'+name, type=Path, required=True)
    launch = sub.add_parser('run', help='consume exactly one human-authorized formal launch')
    for name in ('bundle', 'bundle-sha256', 'approval', 'approval-sha256', 'test', 'review'):
        launch.add_argument('--'+name, required=True)
    worker = sub.add_parser('worker', help='internal owned-session entrypoint only')
    for name in ('bundle', 'bundle-sha256', 'approval', 'approval-sha256', 'test', 'review',
                 'session-directory', 'session-sha256'):
        worker.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    if args.command == 'environment':
        from .formal_matrix import load_protocol
        from .formal_environment import ConfiguredRuntime
        environment = ConfiguredRuntime(load_protocol()).identity()
        path, record = publish(args.output_directory, unpack(environment))
        print(json.dumps(dict(path=str(path), environment_sha256=record['sha256'],
            final_eval_reads=0, final_eval_authorized=False)))
        return 0
    if args.command == 'prepare':
        path, record = prepare(output_directory=args.output_directory, ledger_directory=args.ledger_directory,
                                input_paths={k: getattr(args, k) for k in INPUT_PATHS},predecessor_ledger=args.predecessor_ledger)
        print(json.dumps(dict(path=str(path), bundle_sha256=record['sha256'],
            execution_sha256=unpack(record)['execution']['sha256'], final_eval_reads=0, final_eval_authorized=False)))
        return 0
    if args.command in {'prepare-partial', 'prepare-resource'}:
        from .formal_partial_launch import prepare_partial
        path, record = prepare_partial(output_directory=args.output_directory,ledger_directory=args.ledger_directory,
            predecessor_reference_path=args.predecessor_reference, resource=args.command == 'prepare-resource')
        print(json.dumps(dict(path=str(path),bundle_sha256=record['sha256'],
            execution_sha256=unpack(record)['execution']['sha256'],final_eval_reads=0,final_eval_authorized=False)))
        return 0
    if args.command == 'run':
        result = run(bundle_path=args.bundle, bundle_sha256=args.bundle_sha256, approval_path=args.approval,
            approval_sha256=args.approval_sha256, test_path=args.test, review_path=args.review, started_ns=started)
        print(json.dumps(result))
        return 0
    from . import formal_session as native
    # serve verifies actual native Job membership before constructing the
    # handler. Calling this command directly cannot create a research worker.
    def factory(session, contract):
        authority = authority_paths(approval_path=args.approval, approval_sha256=args.approval_sha256,
            test_path=args.test, review_path=args.review, ledger_directory=session['ledger_directory'])
        return RuntimeWorker(session, contract, bundle_path=args.bundle,
                             bundle_sha256=args.bundle_sha256, authority=authority)
    return native.serve(args.session_directory, args.session_sha256, factory)


if __name__ == '__main__':
    raise SystemExit(main(_started_ns=_MODULE_STARTED_NS))
