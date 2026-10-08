"""Exact CLI/binding and native denial, not full formal-run qualification.

Positive routing uses explicitly replaced authority/environment/scientific
handler. The native negative test uses the real CLI with absent approval.
"""
from copy import deepcopy
import os
from pathlib import Path
import sys
import subprocess

import pytest

from experiments.pirc17 import formal_budget as budget, formal_controller as control, formal_entrypoint as entry
from experiments.pirc17 import formal_session as native
from experiments.pirc17.formal_matrix import build_matrix, load_protocol
from experiments.pirc17.protocol import EXECUTION_VERSION, source_catalog
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, publish, read_json, unpack
from tests.test_pirc17_formal_controller import authority, verification, Clock, FakeSession
from tests.test_pirc17_formal_environment import actual_environment
from tests.pirc17_predecessor_fixture import software_input_interruption as software_predecessor,permit_software_eligibility,software_input_pins,software_frozen_metadata


@pytest.fixture(scope='module')
def scientific_contract():
    protocol = load_protocol()
    return protocol, build_matrix(protocol)


def bundle_for(tmp_path, scientific_contract, actual_environment):
    protocol, matrix = scientific_contract
    paths = {k: tmp_path/k for k in entry.INPUT_PATHS}  # Absent: never acquired by these tests.
    runtime = entry.runtime_binding(protocol, matrix, actual_environment,
        ledger_directory=tmp_path/'ledger',input_paths=paths,
        predecessor=software_predecessor(protocol,matrix,directory=tmp_path/'ABSENT-SOFTWARE-PREDECESSOR'))
    p = unpack(protocol)
    execution = envelope(dict(schema_version=EXECUTION_VERSION, protocol_sha256=protocol['sha256'],
        source_sha256=source_catalog(), matrix_sha256=matrix['sha256'], runtime_manifest_sha256=runtime['sha256'],
        budget_ledger_schema_version=budget.VERSION, entrypoint=entry.ENTRYPOINT,
        phase_caps_seconds=p['resource_contract']['phase_caps_seconds'], max_generated_forecasts=11513,
        scientific_forecasts=11020, method_required_slots=sorted(k for k, v in
            p['components']['method_mechanisms']['slots'].items() if v['disposition'] == 'REQUIRED'),
        terrain_configurations=sorted(p['components']['terrain_configurations']), final_eval_authorized=False))
    record = envelope(dict(schema_version=entry.VERSION+'-bundle', protocol=protocol,
        execution=execution, matrix=matrix, runtime=runtime))
    path, record = publish(tmp_path/'binding', unpack(record))
    auth_path, receipt = publish(tmp_path/'NOT-HUMAN-AUTHORITY', dict(software_fixture_only=True))
    auth = entry.authority_paths(approval_path=auth_path, approval_sha256=receipt['sha256'],
        test_path=auth_path, review_path=auth_path, ledger_directory=tmp_path/'ledger')
    return path, record, auth


def test_bundle_requires_all_original_axes_and_exact_location_binding(tmp_path, scientific_contract, actual_environment):
    path, record, auth = bundle_for(tmp_path, scientific_contract, actual_environment)
    bundle, runtime = entry.validate_bundle(record)
    assert len(unpack(bundle['matrix'])['workloads']) == 11659
    assert runtime['phase_order'] == list(entry.PHASE_ORDER)
    assert runtime['ledger_directory'] == str((tmp_path/'ledger').resolve())
    command = entry.worker_command(path, record['sha256'], auth, bundle['runtime'])
    assert command[:5] == [str(Path(sys.executable).resolve()), '-u', '-m', entry.MODULE, 'worker']
    assert '--particles' not in command and '--step' not in command and '--resume' not in command


@pytest.mark.parametrize('fault', ['new_ledger', 'phase_order', 'entrypoint', 'matrix', 'extra'])
def test_rehashed_wrong_launch_still_rejected(tmp_path, scientific_contract, actual_environment, fault):
    _, record, _ = bundle_for(tmp_path, scientific_contract, actual_environment)
    value = deepcopy(unpack(record))
    if fault in {'new_ledger', 'phase_order'}:
        r = unpack(value['runtime'])
        if fault == 'new_ledger': r['ledger_directory'] = str((tmp_path/'fresh-budget').resolve())
        else: r['phase_order'] = list(reversed(r['phase_order']))
        value['runtime'] = envelope(r)
    elif fault == 'entrypoint':
        e = unpack(value['execution']); e['entrypoint'] = 'PSDE-SDE/experiments/pirc17/formal_worker.py'
        value['execution'] = envelope(e)
    elif fault == 'matrix':
        m = unpack(value['matrix']); m['workloads'] = m['workloads'][:-1]; value['matrix'] = envelope(m)
    else: value['caller_declared_approved'] = True
    with pytest.raises(ValueError): entry.validate_bundle(envelope(value))


def test_runtime_wrapper_routes_and_saves_exact_phase_evidence_with_explicit_software_seams(
        tmp_path, scientific_contract, actual_environment, monkeypatch):
    from experiments.pirc17 import formal_environment, formal_worker
    path, record, auth = bundle_for(tmp_path, scientific_contract, actual_environment)
    b, runtime = entry.validate_bundle(record)
    contract = budget.contract_for_matrix(b['matrix'], protocol_sha256=b['protocol']['sha256'],
        execution_sha256=b['execution']['sha256'], runtime_manifest_sha256=b['runtime']['sha256'],
        approval_sha256=auth['approval_sha256'], ledger_directory=tmp_path/'ledger')
    session_dir = tmp_path/'ledger'/'session'
    session = dict(directory=str(session_dir), ledger_directory=str(tmp_path/'ledger'),
        ledger_root_sha256=digest(contract), worker_command=entry.worker_command(path, record['sha256'], auth, b['runtime']))
    seen = []
    class Environment:
        def __init__(self, protocol): self.hardware = deepcopy(unpack(actual_environment)['hardware'])
        def verify(self, value): assert value == actual_environment
        def check(self): return self.hardware
    class Handler:
        def __init__(self, actual_session, actual_contract, *, input_options):
            assert actual_session == session and actual_contract == contract
            assert input_options['authority'] == auth
            assert input_options['protocol'] == b['protocol']
            self.closed = False
        def __call__(self, work, output): seen.append(work); return {'software_fixture': True}
        def close(self): self.closed = True
    monkeypatch.setattr(entry, '_approval', lambda *args: None)  # Explicitly NOT human authorization.
    monkeypatch.setattr(formal_environment, 'ConfiguredRuntime', Environment)
    monkeypatch.setattr(formal_worker, 'FormalWorker', Handler)
    permit_software_eligibility(monkeypatch)  # Explicitly NOT real predecessor proof.
    with budget.Ledger.create(tmp_path/'ledger',contract) as ledger:
        ledger.import_startup_floor(b['runtime'])
        first = unpack(b['matrix'])['workloads'][0]
        ledger.reserve(first['work_id'])
        session_dir.mkdir()
        wrapper = entry.RuntimeWorker(session,contract,bundle_path=path,bundle_sha256=record['sha256'],authority=auth)
        assert wrapper(first,session_dir/'outputs') == {'software_fixture': True}
        assert seen == [first]
        proof = entry.runtime_evidence(session_dir,bundle=b,session_sha256=digest(session),
            ledger_root_sha256=digest(contract), worker_pid=os.getpid(), phase=first['phase'])
        assert proof['startup']['content_sha256'] == wrapper.start['sha256']
        with pytest.raises(ValueError,match='real owned session'):
            entry.runtime_evidence(session_dir,bundle=b,session_sha256=digest('wrong-session'),
                ledger_root_sha256=digest(contract),worker_pid=os.getpid(),phase=first['phase'])
        monkeypatch.setattr(wrapper.environment,'check',lambda: (_ for _ in ()).throw(ValueError('runtime drift')))
        with pytest.raises(ValueError,match='runtime drift'): wrapper(first,session_dir/'outputs')
        assert wrapper.worker.closed and seen == [first]


def test_actual_native_cli_denies_absent_approval_before_bundle_or_raw_input(tmp_path):
    if os.name != 'nt': pytest.skip('native Windows owned-session CLI')
    directory = tmp_path/'ledger'
    work_id = digest('METADATA-ONLY NATIVE DENIAL')
    contract = dict(schema_version=budget.VERSION, protocol_sha256=digest('fixture-protocol'),
        execution_sha256=digest('fixture-execution'), matrix_sha256=digest('fixture-matrix'),
        runtime_manifest_sha256=digest('fixture-runtime'), approval_sha256=digest('NO-APPROVAL'),
        ledger_directory=str(directory.resolve()), phase_caps_ns={'software': 60_000_000_000},
        total_cap_ns=60_000_000_000, max_generated_forecasts=0, max_attempts_per_item=1,
        workloads=[dict(work_id=work_id, phase='software', generated_forecasts=0, max_active_ns=30_000_000_000)])
    command = [sys.executable, '-u', '-m', entry.MODULE, 'worker', '--bundle', str(tmp_path/'ABSENT-BUNDLE'),
        '--bundle-sha256', digest('not-a-bundle'), '--approval', str(tmp_path/'ABSENT-APPROVAL'),
        '--approval-sha256', contract['approval_sha256'], '--test', str(tmp_path/'ABSENT-TEST'),
        '--review', str(tmp_path/'ABSENT-REVIEW')]
    with budget.Ledger.create(directory, contract) as ledger:
        runner = control.Controller(ledger, command, authorize_work=lambda w: authority(ledger, w),
                                    validate_result=verification)  # Controller seam; real child guard.
        with pytest.raises(control.ControllerStopped): runner.run_all(['software'])
        assert ledger.dispositions()[work_id] == 'failure'
        barrier = unpack(read_json(runner.session.directory/'barriers/000000.json'))
        assert barrier['status'] == 'failure' and barrier['error_type'] == 'ValueError'
        assert not (directory/'access').exists()
        assert not (runner.session.directory/'runtime').exists()
        assert not (runner.session.directory/'outputs/000000').exists()
        assert native._query_job(runner.session.job_name)['exists'] is False


def test_actual_controller_callback_denies_fixture_authority_before_environment_or_data(
        tmp_path, scientific_contract, actual_environment, monkeypatch):
    from experiments.pirc17 import formal_environment
    path, record, auth = bundle_for(tmp_path, scientific_contract, actual_environment)
    b = unpack(record)
    directory = tmp_path/'ledger'
    contract = budget.contract_for_matrix(b['matrix'], protocol_sha256=b['protocol']['sha256'],
        execution_sha256=b['execution']['sha256'], runtime_manifest_sha256=b['runtime']['sha256'],
        approval_sha256=auth['approval_sha256'], ledger_directory=directory)
    monkeypatch.setattr(formal_environment, 'ConfiguredRuntime',
        lambda *a: (_ for _ in ()).throw(AssertionError('environment setup before real authority')))
    with budget.Ledger.create(directory, contract) as ledger:
        callbacks = entry.RuntimeCallbacks(ledger, bundle=record, authority=auth)
        callbacks.session = native.OwnedSession(directory/'session', ledger,
            entry.worker_command(path, record['sha256'], auth, b['runtime']))
        work = contract['workloads'][0]
        ledger.reserve(work['work_id'])
        with pytest.raises(ValueError, match='receipt fields'): callbacks.authorize(work)
        assert callbacks.failed and callbacks.results is None and callbacks.environment is None
        assert not (directory/'access').exists()


@pytest.mark.parametrize('reject_continuation',[False,True])
def test_controller_routes_real_runtime_proof_into_domain_result_with_explicit_software_seams(
        tmp_path, scientific_contract, actual_environment, monkeypatch,reject_continuation):
    from experiments.pirc17 import formal_environment, formal_results,formal_input_interruption
    path, record, auth = bundle_for(tmp_path, scientific_contract, actual_environment)
    b = unpack(record); directory = tmp_path/'ledger'
    contract = budget.contract_for_matrix(b['matrix'], protocol_sha256=b['protocol']['sha256'],
        execution_sha256=b['execution']['sha256'], runtime_manifest_sha256=b['runtime']['sha256'],
        approval_sha256=auth['approval_sha256'], ledger_directory=directory)
    seen = []
    class Environment:
        def __init__(self, p): seen.append('configure')
        def verify(self, r): assert r == actual_environment; seen.append('verify')
        def check(self): seen.append('check')
    class Results:
        def __init__(self, actual_ledger, **scope):
            assert actual_ledger is ledger and scope['authority'] == auth
        def authorize(self, work): seen.append('authorize'); return authority(ledger, work)
        def validate(self, work, manifest, output):
            seen.append('validate'); return {'details': {'software_fixture_only': True}}
    proof = dict(software_fixture_only=True)
    def evidence(directory, **kwargs):
        assert kwargs['first_work_id'] == contract['workloads'][0]['work_id']
        seen.append('worker-proof'); return proof
    monkeypatch.setattr(entry, '_approval', lambda *a: None)
    monkeypatch.setattr(formal_environment, 'ConfiguredRuntime', Environment)
    monkeypatch.setattr(formal_results, 'ScientificResults', Results)
    monkeypatch.setattr(entry, 'runtime_evidence', evidence)
    def continuation(manifest,directory):
        seen.append('input-proof')
        assert manifest == {}  # Explicit transport fixture,not saved input evidence.
        if reject_continuation:raise ValueError('synthetic changed continuation input')
        return dict(software_fixture_only=True)
    monkeypatch.setattr(formal_input_interruption,'verify_continued_manifest',continuation)
    permit_software_eligibility(monkeypatch)
    with budget.Ledger.create(directory, contract) as ledger:
        ledger.import_startup_floor(b['runtime'])
        callbacks = entry.RuntimeCallbacks(ledger, bundle=record, authority=auth)
        callbacks.session = native.OwnedSession(directory/'session', ledger,
            entry.worker_command(path, record['sha256'], auth, b['runtime']))
        callbacks.session.session = envelope(dict(software_fixture_only=True))
        callbacks.session.worker_pid = 123
        assert seen == []  # Heavy initialization must not precede first reservation.
        work = contract['workloads'][0]; ledger.reserve(work['work_id'])
        assert callbacks.authorize(work) == authority(ledger, work)
        if reject_continuation:
            with pytest.raises(ValueError,match='changed continuation input'):
                callbacks.validate(work, {}, directory/'software-output')
            assert callbacks.failed
        else:
            value = callbacks.validate(work, {}, directory/'software-output')
            assert value['details']['effective_runtime'] == proof
            assert value['details']['input_continuation']['software_fixture_only'] is True
    assert seen == ['configure', 'verify', 'check', 'authorize', 'check', 'worker-proof', 'validate', 'input-proof']+([] if reject_continuation else ['check'])


def launch_kwargs(path, record, auth):
    return dict(bundle_path=path, bundle_sha256=record['sha256'], approval_path=auth['approval_path'],
        approval_sha256=auth['approval_sha256'], test_path=auth['test_path'], review_path=auth['review_path'])


def test_actual_run_denies_nonhuman_authority_durably_before_environment_or_science(
        tmp_path, scientific_contract, actual_environment, monkeypatch):
    from experiments.pirc17 import formal_environment
    path, record, auth = bundle_for(tmp_path, scientific_contract, actual_environment)
    monkeypatch.setattr(formal_environment, 'ConfiguredRuntime',
        lambda *a: (_ for _ in ()).throw(AssertionError('environment before authority')))
    kwargs = launch_kwargs(path, record, auth)
    with pytest.raises(ValueError, match='receipt fields'): entry.run(**kwargs)
    claim = entry.launch_directory(tmp_path/'ledger')
    terminal = unpack(read_json(claim/'terminal.json'))
    assert terminal['error_type'] == 'ValueError' and terminal['approval_verified'] is False
    assert terminal['candidate_complete'] is False and terminal['process_tree_closed'] is True
    assert terminal['startup_transferred_to_ledger'] is False
    old_charge = sum(terminal['predecessor_charged_ns_by_phase'].values())
    assert old_charge == 570_736_000_000
    assert terminal['startup_conservative_charge_ns'] >= 3600*budget.NANOSECONDS-old_charge
    assert terminal['cumulative_charge_lower_bound_ns'] >= 3600*budget.NANOSECONDS
    assert terminal['predecessor_charge_location'] == 'claim'
    assert 0 < terminal['elapsed_through_receipt_ns'] < terminal['startup_conservative_charge_ns']
    assert not (tmp_path/'ledger').exists()
    before = file_hash(claim/'terminal.json')
    with pytest.raises(FileExistsError): entry.run(**kwargs)
    assert file_hash(claim/'terminal.json') == before


def test_missing_authority_does_not_even_read_bundle_or_claim_a_formal_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'validate_bundle', lambda *a: pytest.fail('no authority present'))
    with pytest.raises(ValueError, match='bounded regular JSON'):
        entry.run(bundle_path=tmp_path/'ABSENT', bundle_sha256=digest('none'),
            approval_path=tmp_path/'NO-HUMAN', approval_sha256=digest('none'),
            test_path=tmp_path/'NO-TEST', review_path=tmp_path/'NO-REVIEW')
    assert list(tmp_path.iterdir()) == []


def test_incomplete_exclusive_launch_cannot_be_restarted(tmp_path, scientific_contract, actual_environment):
    path, record, auth = bundle_for(tmp_path, scientific_contract, actual_environment)
    claim = entry.launch_directory(tmp_path/'ledger')
    claim.mkdir()  # Simulate death immediately after the exclusive claim.
    with pytest.raises(FileExistsError): entry.run(**launch_kwargs(path, record, auth))
    assert list(claim.iterdir()) == [] and not (tmp_path/'ledger').exists()


def test_actual_prepare_cli_denies_missing_predecessor_before_environment_or_data(tmp_path):
    command = [sys.executable, '-m', entry.MODULE, 'prepare', '--output-directory', str(tmp_path/'bundle'),
               '--ledger-directory', str(tmp_path/'ledger')]
    for name in entry.INPUT_PATHS: command += ['--'+name.replace('_', '-'), str(tmp_path/name)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and '--predecessor-ledger' in result.stderr
    assert not (tmp_path/'bundle').exists() and not (tmp_path/'ledger').exists()


def test_prepare_cli_seals_all_axes_with_explicit_metadata_eligibility_and_environment_seams(
        tmp_path,scientific_contract,actual_environment,monkeypatch,capsys):
    from experiments.pirc17 import formal_input_interruption,formal_environment
    protocol,matrix = scientific_contract
    binding = software_predecessor(protocol,matrix,directory=tmp_path/'ABSENT-SOFTWARE-PREDECESSOR')
    monkeypatch.setattr(formal_input_interruption,'inspect_input_interruption',lambda directory: binding)
    class Environment:
        def __init__(self,p): assert p == protocol
        def identity(self): return actual_environment
    monkeypatch.setattr(formal_environment,'ConfiguredRuntime',Environment)
    command = ['prepare','--output-directory',str(tmp_path/'bundle'),'--ledger-directory',str(tmp_path/'ledger'),
               '--predecessor-ledger',str(tmp_path/'ABSENT-SOFTWARE-PREDECESSOR')]
    for name in entry.INPUT_PATHS: command += ['--'+name.replace('_','-'),str(tmp_path/name)]
    assert entry.main(command) == 0
    import json
    summary = json.loads(capsys.readouterr().out)
    record = read_json(summary['path'])
    b, runtime = entry.validate_bundle(record)
    assert record['sha256'] == summary['bundle_sha256']
    assert len(unpack(b['matrix'])['workloads']) == 11659
    assert unpack(b['execution'])['source_sha256'] == source_catalog()
    assert summary['final_eval_authorized'] is False and summary['final_eval_reads'] == 0
    assert runtime['phase_order'] == list(entry.PHASE_ORDER)
    assert runtime['predecessor'] == binding
    assert not (tmp_path/'ledger').exists() and not entry.launch_directory(tmp_path/'ledger').exists()
    assert all(not (tmp_path/k).exists() for k in entry.INPUT_PATHS)


@pytest.mark.parametrize('flag', ['--particles', '--step', '--seed', '--horizon', '--resume'])
def test_public_run_cannot_override_science_or_resume(flag):
    with pytest.raises(SystemExit) as error:
        entry.main(['run', '--bundle', 'unused', '--bundle-sha256', digest('unused'),
            '--approval', 'unused', '--approval-sha256', digest('unused'), '--test', 'unused', '--review', 'unused', flag, '1'])
    assert error.value.code == 2


def software_launch(tmp_path, scientific_contract, actual_environment, monkeypatch):
    """Replace authority/science/predecessor eligibility; real accounting/wiring.

    Ten one-item metadata phases replace the formal scientific denominator in
    this explicit test seam. This is not full-matrix empirical qualification.
    """
    path, record, auth = bundle_for(tmp_path, scientific_contract, actual_environment)
    b = unpack(record)
    matrix = unpack(b['matrix'])
    works = [next(w for w in matrix['workloads'] if w['phase'] == p) for p in entry.PHASE_ORDER]
    contract = dict(schema_version=budget.VERSION, protocol_sha256=b['protocol']['sha256'],
        execution_sha256=b['execution']['sha256'], matrix_sha256=b['matrix']['sha256'],
        runtime_manifest_sha256=b['runtime']['sha256'], approval_sha256=auth['approval_sha256'],
        ledger_directory=str((tmp_path/'ledger').resolve()),
        phase_caps_ns={p: (3600 if p == entry.PHASE_ORDER[0] else 60)*budget.NANOSECONDS for p in entry.PHASE_ORDER},
        total_cap_ns=(3600+9*60)*budget.NANOSECONDS, max_generated_forecasts=0, max_attempts_per_item=1,
        workloads=[dict(work_id=w['work_id'], phase=w['phase'], max_active_ns=30*budget.NANOSECONDS,
                        generated_forecasts=0) for w in works])
    # Explicit ten-phase software accounting seam, never full-matrix science.
    r = deepcopy(unpack(b['runtime']))
    r['predecessor'] = software_predecessor(b['protocol'],b['matrix'],directory=tmp_path/'ABSENT-SOFTWARE-PREDECESSOR',
        phase_caps_ns=contract['phase_caps_ns'],workloads=contract['workloads'])
    b['runtime'] = envelope(r)
    e = deepcopy(unpack(b['execution'])); e['runtime_manifest_sha256'] = b['runtime']['sha256']; b['execution'] = envelope(e)
    path,record = publish(tmp_path/'software-binding',b)
    contract['runtime_manifest_sha256'] = b['runtime']['sha256']; contract['execution_sha256'] = b['execution']['sha256']
    clock = Clock()
    seen = {}
    class Callbacks:
        def __init__(self, ledger, **kwargs): self.ledger = ledger; self.session = None
        def authorize(self, work):
            assert self.session is seen['controller'].session
            return authority(self.ledger, work)
        def validate(self, *args): return verification(*args)
    actual_controller = control.Controller
    def controller(ledger, command, **kwargs):
        assert command == entry.worker_command(path, record['sha256'], auth, b['runtime'])
        result = actual_controller(ledger, command, _clock=clock, _memory=lambda: 8*1024**3,
                                   _session_factory=FakeSession, **kwargs)
        seen['controller'] = result
        return result
    monkeypatch.setattr(entry, '_approval', lambda *args: None)  # NOT human authority.
    permit_software_eligibility(monkeypatch)
    monkeypatch.setattr(entry, 'RuntimeCallbacks', Callbacks)
    monkeypatch.setattr(budget, 'contract_for_matrix', lambda *a, **kw: contract)
    monkeypatch.setattr(control, 'Controller', controller)
    monkeypatch.setattr(entry.time, 'monotonic_ns', clock)
    return launch_kwargs(path, record, auth), clock, seen


def test_real_run_wiring_transfers_startup_once_and_seals_terminal_tail(
        tmp_path, scientific_contract, actual_environment, monkeypatch):
    kwargs, clock, seen = software_launch(tmp_path, scientific_contract, actual_environment, monkeypatch)
    result = entry.run(**kwargs, started_ns=clock()-7*budget.NANOSECONDS)
    runner = seen['controller']
    summary = runner.ledger.summary()
    terminal = unpack(read_json(result['terminal_path']))
    assert len(runner.session.calls) == 10 and runner.session.closed
    assert summary['control_charged_ns_by_phase'][entry.PHASE_ORDER[0]] == 34_532_000_000+12*budget.NANOSECONDS
    assert summary['predecessor_floor']['charged_total_ns'] == 570_736_000_000
    assert summary['control_charged_ns_by_phase'][entry.PHASE_ORDER[-1]] == 5*budget.NANOSECONDS
    assert terminal['startup_transferred_to_ledger'] is True and terminal['startup_conservative_charge_ns'] == 0
    assert terminal['predecessor_floor_transferred_to_ledger'] is True and terminal['predecessor_charge_location'] == 'ledger'
    assert terminal['accounting_complete'] is False  # Receipt precedes final ledger event.
    last = summary['terminal_control_sha256']
    assert summary['control_spans'][last]['terminal_evidence_sha256'] == digest(dict(
        content_sha256=result['terminal_sha256'], file_sha256=file_hash(result['terminal_path'])))
    assert result['execution_complete'] is True and result['scientific_claim_authorized'] is False
    assert runner.ledger._lock.stream.closed
    with budget.Ledger.open(tmp_path/'ledger', expected_root_sha256=result['ledger_root_sha256']) as reopened:
        assert reopened.summary() == summary
        assert reopened.tip == result['ledger_tip']
    with pytest.raises(FileExistsError): entry.run(**kwargs)


def test_slow_terminal_file_cannot_return_success_or_new_budget(
        tmp_path, scientific_contract, actual_environment, monkeypatch):
    kwargs, clock, seen = software_launch(tmp_path, scientific_contract, actual_environment, monkeypatch)
    actual_publish = budget._publish
    def slow(path, *a, **kw):
        record = actual_publish(path, *a, **kw)
        if Path(path).name == 'terminal.json': clock.advance(6*budget.NANOSECONDS)
        return record
    monkeypatch.setattr(budget, '_publish', slow)
    with pytest.raises(control.ControllerStopped): entry.run(**kwargs)
    runner = seen['controller']
    assert runner.ledger.summary()['halted_reason'] == 'control_credit_exhausted'
    assert runner.ledger.summary()['control_charged_ns_by_phase'][entry.PHASE_ORDER[-1]] >= 6*budget.NANOSECONDS
    assert runner.session.closed and runner.ledger._lock.stream.closed
    assert unpack(read_json(entry.launch_directory(tmp_path/'ledger')/'terminal.json'))['accounting_complete'] is False


@pytest.mark.parametrize('fault', ['ledger_create', 'predecessor_event', 'predecessor_head',
                                 'controller_construct', 'control_event', 'terminal_file'])
def test_initialization_and_disk_failures_keep_exclusive_claim_and_nonfree_startup(
        tmp_path, scientific_contract, actual_environment, monkeypatch, fault):
    kwargs, clock, seen = software_launch(tmp_path, scientific_contract, actual_environment, monkeypatch)
    def fail(*args, **kwargs): raise OSError('explicit software disk/startup failure')
    if fault == 'ledger_create':
        monkeypatch.setattr(budget.Ledger, 'create', fail)
    elif fault == 'controller_construct':
        monkeypatch.setattr(control, 'Controller', fail)
    else:
        actual_publish = budget._publish
        def broken(path, payload, **kw):
            if ((fault == 'control_event' and payload.get('type') == 'control_open')
                    or (fault == 'predecessor_event' and payload.get('type') == 'startup_predecessor')
                    or (fault == 'predecessor_head' and Path(path).name == 'head.json' and payload.get('event_count') == 1)
                    or (fault == 'terminal_file' and Path(path).name == 'terminal.json')):
                fail()
            return actual_publish(path, payload, **kw)
        monkeypatch.setattr(budget, '_publish', broken)
    with pytest.raises(OSError): entry.run(**kwargs)
    claim = entry.launch_directory(tmp_path/'ledger')
    assert (claim/'start.json').is_file()
    if fault == 'terminal_file':
        assert not (claim/'terminal.json').exists()
        assert seen['controller'].ledger.summary()['terminal_control_sha256'] is None
    else:
        terminal = unpack(read_json(claim/'terminal.json'))
        assert not terminal['candidate_complete'] and not terminal['accounting_complete']
        assert terminal['startup_transferred_to_ledger'] is False
        assert terminal['startup_conservative_charge_ns'] >= 3600*budget.NANOSECONDS-570_736_000_000
        assert terminal['cumulative_charge_lower_bound_ns'] >= 3600*budget.NANOSECONDS
        if fault in {'ledger_create','predecessor_event','predecessor_head','control_event'}:
            # A failed control publication poisons the writer after its valid
            # floor. The terminal must retain the cumulative claim rather
            # than trust the writer's in-memory, unpublished control state.
            assert terminal['predecessor_floor_transferred_to_ledger'] is False
            assert terminal['predecessor_charge_location'] == 'claim'
        else:
            assert terminal['predecessor_floor_transferred_to_ledger'] is True
        assert sum(terminal['predecessor_charged_ns_by_phase'].values()) == 570_736_000_000
    if 'controller' in seen:
        runner = seen['controller']
        assert runner.session.closed and runner.ledger._lock.stream.closed
        if fault == 'control_event':
            assert runner.ledger._poisoned and runner.session.calls == []
            assert [p.name for p in (tmp_path/'ledger'/'events').glob('*.json')] == ['000000.json']
            floor = read_json(tmp_path/'ledger'/'events'/'000000.json')
            assert unpack(floor)['type'] == 'startup_predecessor'
            assert unpack(read_json(tmp_path/'ledger'/'head.json')) == dict(
                root_sha256=runner.ledger.root_sha256,event_count=1,last_event_sha256=floor['sha256'])
    with pytest.raises(FileExistsError): entry.run(**kwargs)
