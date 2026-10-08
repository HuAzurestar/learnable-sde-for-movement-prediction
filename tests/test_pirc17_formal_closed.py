"""Read-only audits of actual short native fixtures, not empirical forecasts."""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import sys

import pytest

from experiments.pirc17 import formal_budget as budget, formal_controller as control, formal_session as native
from experiments.pirc17.formal_closed import ClosedOutputs
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, read_json, unpack
from tests.test_pirc17_formal_controller import authority, contract, verification


@contextmanager
def changed(path, value):
    """Mutate only a pytest-owned fixture, restoring it even on test failure."""
    original = path.read_bytes() if path.exists() else None
    path.write_bytes(value if isinstance(value, bytes) else canonical(value))
    try:
        yield
    finally:
        if original is None:
            path.unlink()
        else:
            path.write_bytes(original)


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    if os.name != 'nt':
        pytest.skip('real native Windows Job fixture')
    directory = tmp_path_factory.mktemp('closed-software-outputs')/'ledger'
    binding = contract(directory)
    binding['max_generated_forecasts'] = 0
    for work in binding['workloads']:
        work['generated_forecasts'] = 0
    prefixes, reads_while_running = [], []
    with budget.Ledger.create(directory, binding) as ledger:
        settle = ledger.settle
        def observed_settle(*args, **kwargs):
            result = settle(*args, **kwargs)
            prefixes.append(deepcopy(ledger.tip))
            # The writer still holds its lock and the actual worker is alive.
            view = ClosedOutputs(directory, contract=binding, tip=ledger.tip)
            reads_while_running.append(view.read(binding['workloads'][len(prefixes)-1]))
            return result
        ledger.settle = observed_settle
        runner = control.Controller(ledger,
            [sys.executable, '-u', str((Path(__file__).parent/'fixtures/pirc17_session_worker.py').resolve()),
             '--mode', 'artifact'], authorize_work=lambda w: authority(ledger, w), validate_result=verification)
        result = runner.run_all(['first', 'second'])
        tip = deepcopy(ledger.tip)
        assert runner.session.closed and runner.meter is None
        assert native._query_job(runner.session.job_name)['exists'] is False
        assert set(result['dispositions'].values()) == {'success'}
    return dict(directory=directory, contract=binding, tip=tip, prefixes=prefixes,
                running=reads_while_running, result=result, session=runner.session.directory)


def view(case, tip=None):
    return ClosedOutputs(case['directory'], contract=case['contract'], tip=tip or case['tip'])


@pytest.fixture(scope='module')
def runtime_completed(tmp_path_factory):
    """Actual native ledger/files; explicitly synthetic hardware observations."""
    if os.name != 'nt': pytest.skip('real native Windows Job fixture')
    from experiments.pirc17 import formal_entrypoint as entry, formal_environment as environment
    directory = tmp_path_factory.mktemp('closed-runtime-proof')/'ledger'
    binding = contract(directory)
    binding['max_generated_forecasts'] = 0
    for work in binding['workloads']: work['generated_forecasts'] = 0
    observed_environment = envelope(dict(schema_version=environment.VERSION,
        protocol_sha256=binding['protocol_sha256'], hardware={'software_fixture_only': True},
        final_eval_authorized=False))
    # Retained read-only v1 evidence fixture. This is not executable by v2.
    historical = 'pirc17-concrete-formal-entrypoint-v1'
    runtime = envelope(dict(schema_version=historical+'-runtime', protocol_sha256=binding['protocol_sha256'],
        matrix_sha256=binding['matrix_sha256'], environment=observed_environment,
        ledger_directory=str(directory), input_paths={k: str(directory.parent/k) for k in entry.INPUT_PATHS},
        working_directory=str(Path.cwd()), worker_module=entry.MODULE,
        phase_order=['first', 'second'], final_eval_authorized=False))
    binding['runtime_manifest_sha256'] = runtime['sha256']
    first_ids = {phase: next(w['work_id'] for w in binding['workloads'] if w['phase'] == phase)
                 for phase in binding['phase_caps_ns']}
    with budget.Ledger.create(directory, binding) as ledger:
        def validate(work, manifest, output):
            scope = runner.session
            runtime_dir = scope.directory/'runtime'
            if not runtime_dir.exists():
                runtime_dir.mkdir()
                native._write(runtime_dir/'binding.json', unpack(runtime))
                native._write(runtime_dir/'start.json', dict(schema_version=historical+'-worker-runtime',
                    session_sha256=scope.session['sha256'], ledger_root_sha256=ledger.root_sha256,
                    execution_sha256=binding['execution_sha256'], runtime_manifest_sha256=runtime['sha256'],
                    worker_pid=scope.worker_pid, environment=observed_environment))
            phase_path = runtime_dir/(digest(work['phase'])+'.json')
            if not phase_path.exists():
                native._write(phase_path, dict(schema_version=historical+'-phase-runtime',
                    startup_sha256=read_json(runtime_dir/'start.json')['sha256'], phase=work['phase'],
                    first_work_id=first_ids[work['phase']], worker_pid=scope.worker_pid,
                    before_sha256=digest(unpack(observed_environment)['hardware']),
                    after_sha256=digest(unpack(observed_environment)['hardware'])))
            value = verification(work, manifest, output)
            value['details']['effective_runtime'] = entry.read_runtime_evidence(scope.directory, contract=binding,
                session_sha256=scope.session['sha256'], ledger_root_sha256=ledger.root_sha256,
                worker_pid=scope.worker_pid, phase=work['phase'], first_work_id=first_ids[work['phase']])
            return value
        runner = control.Controller(ledger,
            [sys.executable, '-u', str((Path(__file__).parent/'fixtures/pirc17_session_worker.py').resolve()),
             '--mode', 'artifact'], authorize_work=lambda w: authority(ledger, w), validate_result=validate)
        result = runner.run_all(['first', 'second'])
        assert set(result['dispositions'].values()) == {'success'}
        tip = deepcopy(ledger.tip)
    return dict(directory=directory, contract=binding, tip=tip, session=runner.session.directory)


def test_closed_audit_rereads_actual_bound_runtime_files_without_hardware_access(runtime_completed, monkeypatch):
    from experiments.pirc17 import formal_environment
    case = runtime_completed
    before = {str(p): file_hash(p) for p in case['directory'].rglob('*') if p.is_file()}
    def forbidden(*a, **kw): pytest.fail('saved runtime audit must not remeasure hardware or mutate files')
    monkeypatch.setattr(formal_environment, 'ConfiguredRuntime', forbidden)
    monkeypatch.setattr(budget, '_publish', forbidden)
    closed = view(case)
    for work in case['contract']['workloads']:
        assert closed.read(work)['manifest']['fixture'] is True
    assert before == {str(p): file_hash(p) for p in case['directory'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault', ['binding', 'startup', 'phase_hash', 'phase_first_work', 'new_environment', 'missing'])
def test_runtime_proof_does_not_trust_cached_pass_or_rehashed_replacements(runtime_completed, fault):
    case = runtime_completed
    closed = view(case)
    work = case['contract']['workloads'][0]
    assert closed.read(work) is not None
    runtime_dir = case['session']/'runtime'
    target = runtime_dir/('binding.json' if fault == 'binding' else 'start.json' if fault in
                         {'startup', 'new_environment'} else digest(work['phase'])+'.json')
    payload = unpack(read_json(target))
    if fault == 'binding': payload['ledger_directory'] = str(case['directory']/'different')
    elif fault == 'startup': payload['worker_pid'] += 1
    elif fault == 'new_environment': payload['environment'] = envelope(dict(software_fixture_only='different'))
    elif fault == 'phase_hash': payload['after_sha256'] = digest('changed-threads')
    elif fault == 'phase_first_work': payload['first_work_id'] = digest('different-work')
    replacement = b'' if fault == 'missing' else envelope(payload)
    with changed(target, replacement), pytest.raises(ValueError): closed.read(work)


def test_runtime_proof_pins_exact_bytes_not_only_equivalent_json(runtime_completed):
    case = runtime_completed
    closed = view(case)
    target = case['session']/'runtime'/'start.json'
    with changed(target, target.read_bytes()+b' '), pytest.raises(ValueError, match='runtime bytes'):
        closed.read(case['contract']['workloads'][0])


def test_closed_batch_reuses_historical_runtime_without_skipping_native_items(runtime_completed,monkeypatch):
    from experiments.pirc17 import formal_entrypoint as entry
    from experiments.pirc17.formal_metadata_batch import metadata_batch
    case=runtime_completed
    closed=view(case)
    checks=[]
    original=entry.read_runtime_evidence
    def counted(*a,**kw):
        checks.append(kw['phase'])
        return original(*a,**kw)
    monkeypatch.setattr(entry,'read_runtime_evidence',counted)
    with metadata_batch(closed):
        for _ in range(5):
            for work in case['contract']['workloads']:
                assert closed.read(work)['manifest']['fixture'] is True
        assert checks==['first','second']
        work=case['contract']['workloads'][0]
        item=closed.read(work)
        artifact=item['directory']/'fixture.bin'
        with changed(artifact,b'wrong owned bytes'),pytest.raises(ValueError,match='artifact bytes'):
            closed.read(work)
    assert checks==['first','second','first','second']
    closed.read(work)
    assert checks[-1]=='first' and len(checks)==5


def test_closed_batch_cannot_handoff_changed_runtime_files(runtime_completed):
    from experiments.pirc17.formal_metadata_batch import metadata_batch
    case=runtime_completed
    closed=view(case)
    target=case['session']/'runtime/start.json'
    with changed(target,target.read_bytes()) as _:
        with pytest.raises(ValueError,match='bytes changed'):
            with metadata_batch(closed):
                assert closed.read(case['contract']['workloads'][0]) is not None
                target.write_bytes(target.read_bytes()+b' ')
        assert closed._metadata_batch is None
        with pytest.raises(ValueError,match='runtime bytes'):
            closed.read(case['contract']['workloads'][0])


def test_reads_actual_closed_files_without_writer_lock_native_checks_or_writes(completed, monkeypatch):
    case = completed
    before = {str(p): file_hash(p) for p in case['directory'].rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail('historical file audit must not acquire, dispatch, measure, or repair')
    monkeypatch.setattr(budget.Ledger, 'open', forbidden)
    monkeypatch.setattr(budget, '_publish', forbidden)
    monkeypatch.setattr(native, '_query_job', forbidden)
    monkeypatch.setattr(native.OwnedSession, 'run', forbidden)
    closed = view(case)
    for index, work in enumerate(case['contract']['workloads']):
        item = closed.read(work)
        assert item == case['running'][index]
        assert item['manifest']['count'] == index+1
        assert item['directory'].joinpath('fixture.bin').read_bytes() == b'bounded software-only artifact'
        assert item['artifacts']['fixture.bin']['file_sha256'] == file_hash(item['directory']/'fixture.bin')
    identity = unpack(closed.identity)
    summary = case['result']['summary']
    assert identity['work_dispositions'] == case['result']['dispositions']
    for field in ('charged_ns_by_phase', 'measured_ns_by_phase', 'conservatively_charged_ns_by_phase'):
        assert identity[field] == summary[field]
    assert identity['generated_forecasts_reserved'] == 0
    assert identity['historical_observations_remeasured'] is False
    assert before == {str(p): file_hash(p) for p in case['directory'].rglob('*') if p.is_file()}


def test_prefix_does_not_adopt_later_completed_work(completed):
    case = completed
    closed = view(case, case['prefixes'][0])
    works = case['contract']['workloads']
    assert closed.read(works[0])['manifest']['count'] == 1
    assert closed.disposition(works[1]['work_id']) == 'unattempted'
    assert closed.read(works[1]) is None
    assert closed.tip == case['prefixes'][0]
    assert closed.read(works[0]) == case['running'][0]


def test_uncommitted_tail_is_not_repaired_or_adopted(completed):
    case = completed
    extra = case['directory']/f"events/{case['tip']['event_count']:06d}.json"
    with changed(extra, b'not a committed ledger event'):
        assert view(case).tip == case['tip']
        assert extra.read_bytes() == b'not a committed ledger event'


@pytest.mark.parametrize('fault', ['prefix_hash', 'prefix_count', 'prefix_root', 'head_hash', 'head_rewind',
                                  'event_bytes', 'suffix_hash', 'contract', 'work_id', 'work_phase'])
def test_closed_prefix_identity_and_commitment_are_checked(completed, fault):
    case = completed
    tip = deepcopy(case['prefixes'][0])
    binding = deepcopy(case['contract'])
    target = None
    if fault == 'prefix_hash': tip['last_event_sha256'] = '0'*64
    if fault == 'prefix_count': tip['event_count'] = True
    if fault == 'prefix_root': tip['root_sha256'] = '0'*64
    if fault == 'contract': binding['approval_sha256'] = digest('different fake approval')
    if fault.startswith('head_'):
        target = case['directory']/'head.json'
        payload = unpack(read_json(target))
        if fault == 'head_hash': payload['last_event_sha256'] = '0'*64
        else: payload['event_count'] = 0
        replacement = envelope(payload)
    if fault in {'event_bytes', 'suffix_hash'}:
        index = 0 if fault == 'event_bytes' else tip['event_count']
        target = case['directory']/f'events/{index:06d}.json'
        if fault == 'event_bytes': replacement = b'invalid JSON'
        else:
            payload = unpack(read_json(target))
            payload['previous_sha256'] = '0'*64
            replacement = envelope(payload)
    def audit():
        closed = ClosedOutputs(case['directory'], contract=binding, tip=tip)
        if fault in {'work_id', 'work_phase'}:
            work = deepcopy(binding['workloads'][0])
            work['work_id' if fault == 'work_id' else 'phase'] = 'unregistered'
            closed.read(work)
    if target is None:
        with pytest.raises((ValueError, PermissionError)):
            audit()
    else:
        with changed(target, replacement), pytest.raises(ValueError):
            audit()


@pytest.mark.parametrize('relative', ['session.json', 'ready.json', 'requests/000000.json',
    'barriers/000000.json', 'outputs/000000/result.json', 'outputs/000000/fixture.bin'])
def test_actual_output_and_transport_file_corruption_is_rejected(completed, relative):
    closed = view(completed)
    path = completed['session']/relative
    with changed(path, b'corrupted software fixture'), pytest.raises(ValueError):
        closed.read(completed['contract']['workloads'][0])


def test_missing_extra_files_and_forged_manifest_are_not_accepted(completed):
    closed = view(completed)
    work = completed['contract']['workloads'][0]
    with changed(completed['session']/'outputs/000000/extra.bin', b'unlisted'), pytest.raises(ValueError):
        closed.read(work)
    path = completed['session']/'outputs/000000/result.json'
    payload = unpack(read_json(path))
    payload['value']['fixture'] = False
    with changed(path, envelope(payload)), pytest.raises(ValueError):
        closed.read(work)


@pytest.mark.parametrize('target', ['dispatch', 'control', 'observation'])
def test_actual_completion_evidence_not_only_result_file_is_required(completed, target):
    closed = view(completed)
    work = completed['contract']['workloads'][0]
    row = closed.entries[work['work_id']]
    if target == 'dispatch':
        path = completed['directory']/f"dispatches/{row['reservation']['reservation_sha256']}.json"
    else:
        key = row['control_sha256'] if target == 'control' else row['settlement']['completion_evidence_sha256']
        path = completed['directory']/f'controls/{key}.json'
    payload = unpack(read_json(path))
    payload['work_id'] = '0'*64
    with changed(path, envelope(payload)), pytest.raises(ValueError):
        closed.read(work)


def test_reserved_and_failed_work_never_become_closed_success(tmp_path):
    binding = contract(tmp_path/'ledger')
    with budget.Ledger.create(tmp_path/'ledger', binding) as ledger:
        work = binding['workloads'][0]
        reservation = ledger.reserve(work['work_id'])
        reserved_tip = deepcopy(ledger.tip)
        pending = ClosedOutputs(ledger.directory, contract=binding, tip=reserved_tip)
        assert pending.disposition(work['work_id']) == 'reserved'
        assert pending.read(work) is None
        # Deliberately no native success evidence: ledger failure remains only
        # a failure disposition, never a verified output or OS remeasurement.
        ledger.settle(reservation['reservation_sha256'], status='failure', elapsed_ns=1,
            completion_evidence_sha256=digest('synthetic failure only'), result_sha256=None, reason='software fixture')
        failed = ClosedOutputs(ledger.directory, contract=binding, tip=ledger.tip)
        assert failed.disposition(work['work_id']) == 'failure'
        assert failed.read(work) is None
        assert ClosedOutputs(ledger.directory, contract=binding, tip=reserved_tip).read(work) is None
