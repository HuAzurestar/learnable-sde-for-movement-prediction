"""Bounded recovery lifecycle fixtures, not data/approval/scientific evidence.

The predecessor is deliberately synthetic. Native cases use the real Windows
Job/mailbox/barriers and run only tiny software handlers with no model/kernel.
"""
from copy import deepcopy
import os
from pathlib import Path
import sys
import time

import pytest

from tests.test_pirc17_partial_predecessor import fixture
from tests.test_pirc17_formal_controller import Clock, FakeSession, authority, verification
from experiments.pirc17 import formal_budget as budget, formal_controller as control, formal_session as native
from experiments.pirc17.protocol_core import envelope, unpack, read_json, digest

NS = budget.NANOSECONDS
ORDER = ['input', 'method_training', 'terrain_training', 'method_forecasts']
WORKER = Path(__file__).parent/'fixtures/pirc17_session_worker.py'


@pytest.fixture
def imported(fixture):
    with budget.Ledger.create(fixture['new'], fixture['target']) as ledger:
        ledger.import_partial_floor(fixture['runtime'])
        yield ledger


class RestoringSession(FakeSession):
    def run(self, reservation, *, started_ns):
        self.calls.append(reservation['work_id'])
        self.clock.advance(NS//2)  # Below this fixture's original 1s work cap.
        output = self.directory/'outputs'/f'{self.sequence:06d}'
        output.mkdir(parents=True)
        result = native._write(output/'result.json', dict(value=dict(software_fixture_only=True)))
        observed = envelope(dict(schema_version=native.VERSION+'-observation',
            ledger_root_sha256=self.ledger.root_sha256, reservation_sha256=reservation['reservation_sha256'],
            work_id=reservation['work_id'], status='success', sequence=self.sequence, result_sha256=result['sha256'],
            started_ns=started_ns, ended_ns=self.clock(), elapsed_ns=self.clock()-started_ns,
            deadline_ns=started_ns+reservation['reserved_ns']))
        self.sequence += 1
        return observed

    def bootstrap(self, key, *, started_ns, deadline_ns, tick):
        self.bootstrap_calls = getattr(self, 'bootstrap_calls', 0)+1
        assert self.ledger.summary()['pending'] is None
        assert self.ledger.summary()['active_control_sha256'] == key
        self.before = self.ledger.summary()
        for _ in range(8):
            self.clock.advance(NS//2)
            tick()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory/'bootstrap').mkdir()
        return envelope(dict(schema_version=native.VERSION+'-bootstrap-observation',
            ledger_root_sha256=self.ledger.root_sha256, control_sha256=key,
            started_ns=started_ns, deadline_ns=deadline_ns, ended_ns=self.clock(),
            elapsed_ns=self.clock()-started_ns, software_fixture_only=True))


def fake_runner(ledger, clock, validator, **kwargs):
    return control.Controller(ledger, ['SOFTWARE-ONLY'], authorize_work=lambda w: authority(ledger, w),
        validate_result=verification, validate_bootstrap=validator, _clock=clock,
        _memory=lambda: 8*1024**3, _session_factory=kwargs.pop('factory', RestoringSession), **kwargs)


def test_restore_and_validation_are_input_control_time_not_forecast_or_new_generation(imported):
    clock = Clock()
    old = imported.summary()
    def validate(observation, output, tick):
        assert output.name == 'bootstrap'
        for _ in range(2):
            clock.advance(NS//2); tick()
        return dict(software_only=True, saved_models_verified=True)
    runner = fake_runner(imported, clock, validate)
    result = runner.run_all(ORDER)
    new = result['summary']
    assert runner.session.bootstrap_calls == 1
    assert len(runner.session.calls) == 1 and len(result['settlements']) == 1
    assert new['imported_success_count'] == 28 and new['generated_forecasts_reserved'] == 2
    assert runner.session.before['generated_forecasts_reserved'] == old['generated_forecasts_reserved']
    assert runner.session.before['attempted_work_items'] == old['attempted_work_items']
    assert new['measured_ns_by_phase']['input'] == old['measured_ns_by_phase']['input']
    assert new['control_observed_ns_by_phase']['input'] >= old['control_observed_ns_by_phase']['input']+5*NS
    assert new['charged_ns_by_phase']['input'] > old['charged_ns_by_phase']['input']+5*NS
    assert runner.bootstrap_receipt is not None and new['halted_reason'] is None
    assert len(new['control_spans']) == 2
    assert all(v == 'success' for v in result['dispositions'].values())


def test_failure_keeps_prepaid_input_cost_and_all_old_successes_without_dispatch(imported):
    clock = Clock()
    old = imported.summary()
    def fail(*args): raise ValueError('synthetic domain restoration rejection')
    runner = fake_runner(imported, clock, fail)
    with pytest.raises(ValueError, match='restoration rejection'): runner.run_all(ORDER)
    state = imported.summary()
    assert not runner.session.calls and runner.session.closed
    assert state['attempted_work_items'] == old['attempted_work_items']
    assert state['generated_forecasts_reserved'] == old['generated_forecasts_reserved']
    assert state['charged_ns_by_phase']['input'] >= old['charged_ns_by_phase']['input']+5*NS
    assert state['pending'] is None and state['active_control_sha256'] is None
    assert state['halted_reason'] == 'supervision_error'
    assert imported._state.imported_success
    with pytest.raises(control.ControllerStopped, match='one-shot'): runner.run_all(ORDER)


def test_input_expiry_cannot_borrow_another_phase_or_reset_old_cost(imported):
    clock = Clock()
    class Slow(RestoringSession):
        def bootstrap(self, key, *, started_ns, deadline_ns, tick):
            for _ in range(20):
                clock.advance(NS//2); tick()
    old = imported.summary()
    runner = fake_runner(imported, clock, lambda *a: dict(fixture=True), factory=Slow)
    with pytest.raises(control.ControllerStopped): runner.run_all(ORDER)
    state = imported.summary()
    assert not runner.session.calls and runner.session.closed
    assert state['generated_forecasts_reserved'] == old['generated_forecasts_reserved']
    assert state['charged_ns_by_phase']['method_forecasts'] == old['charged_ns_by_phase']['method_forecasts']
    assert state['halted_reason'] is not None


def test_imported_success_cannot_skip_bootstrap(imported):
    runner = fake_runner(imported, Clock(), None)
    with pytest.raises(control.ControllerStopped, match='metered restoration'): runner.run_all(ORDER)
    assert not runner.session.calls and imported.summary()['active_control_sha256'] is None


def test_controller_publishes_domain_receipt_and_binds_before_unfinished_dispatch(imported,monkeypatch):
    from experiments.pirc17.formal_partial_imports import VERSION as IMPORT_VERSION
    clock=Clock()
    reference=dict(path=str(imported.directory/'SOFTWARE-ONLY.json'),content_sha256=digest('software manifest'),
                   file_sha256=digest('software bytes'))
    admitted=[]
    def bind(ref,*,bootstrap_observation_sha256):
        # Explicit ledger-domain seam: full native/domain binding is exercised
        # by partial_worker; here check controller routing/ordering/accounting.
        proof=unpack(native._read(imported.directory/'controls'/(bootstrap_observation_sha256+'.json'),
                                  bootstrap_observation_sha256))
        assert proof['restoration_verification']['manifest_reference']==ref==reference
        assert not runner.session.calls and imported.summary()['pending'] is None
        assert imported.summary()['active_control_sha256']==proof['control_sha256']
        admitted.append(bootstrap_observation_sha256)
    monkeypatch.setattr(imported,'bind_partial_imports',bind)
    runner=fake_runner(imported,clock,lambda *a:dict(schema_version=IMPORT_VERSION,manifest_reference=reference))
    result=runner.run_all(ORDER)
    assert admitted==[runner.bootstrap_receipt['sha256']]
    assert len(result['settlements'])==1 and len(runner.session.calls)==1


@pytest.mark.partial_phase_seconds(60)
def test_concrete_domain_blocks_are_prepaid_inside_original_input_caps(imported,monkeypatch):
    from experiments.pirc17.formal_partial_imports import VERSION as IMPORT_VERSION
    from experiments.pirc17.formal_entrypoint import MODULE
    clock=Clock()
    def validate(observation,output,tick):
        # A synchronous metadata/domain block longer than the old 5s window.
        clock.advance(12*NS)
        tick()
        return dict(schema_version=IMPORT_VERSION,manifest_reference=dict(software_only=True))
    monkeypatch.setattr(imported,'bind_partial_imports',lambda *a,**kw:None) # Routing/timing ONLY, not domain PASS.
    runner=control.Controller(imported,[sys.executable,'-u','-m',MODULE,'worker'],
        authorize_work=lambda w:authority(imported,w),validate_result=verification,validate_bootstrap=validate,
        _clock=clock,_memory=lambda:8*1024**3,_session_factory=RestoringSession)
    old=imported.summary()
    result=runner.run_all(ORDER)
    new=result['summary']
    assert new['control_observed_ns_by_phase']['input']>=old['control_observed_ns_by_phase']['input']+16*NS
    assert new['charged_ns_by_phase']['input']<=imported._state.contract['phase_caps_ns']['input']
    assert new['conservatively_charged_ns_by_phase']['input']>old['conservatively_charged_ns_by_phase']['input']
    assert new['generated_forecasts_reserved']==2 and len(result['settlements'])==1


@pytest.mark.partial_phase_seconds(30)
def test_concrete_domain_window_cannot_borrow_or_reset_exhausted_input(imported):
    from experiments.pirc17.formal_entrypoint import MODULE
    called=[]
    runner=control.Controller(imported,[sys.executable,'-u','-m',MODULE,'worker'],
        authorize_work=lambda w:authority(imported,w),validate_result=verification,
        validate_bootstrap=lambda *a:called.append(1),_clock=Clock(),_memory=lambda:8*1024**3,
        _session_factory=RestoringSession)
    old=imported.summary()
    with pytest.raises(control.ControllerStopped):runner.run_all(ORDER)
    assert called==[] and runner.session.calls==[] and runner.session.closed
    state=imported.summary()
    assert state['charged_ns_by_phase']['input']==30*NS
    assert state['charged_ns_by_phase']['method_forecasts']==old['charged_ns_by_phase']['method_forecasts']
    assert state['generated_forecasts_reserved']==old['generated_forecasts_reserved']


def request_fixture(runner):
    c = unpack(runner.control)
    config = dict(directory=str(runner.session.directory), ledger_directory=str(runner.ledger.directory),
        ledger_root_sha256=runner.ledger.root_sha256, job_name=runner.session.job_name,
        worker_command=runner.session.command)
    key = digest('synthetic-session-descriptor-not-native-membership')
    request = envelope(dict(schema_version=native.VERSION+'-bootstrap-request', session_sha256=key,
        control_sha256=runner.control['sha256'], started_ns=c['started_ns'], deadline_ns=c['phase_deadline_ns']))
    return request, config, key


@pytest.mark.parametrize('mutation', ['none', 'session', 'deadline', 'clock', 'control', 'expired', 'wrong_root'])
def test_worker_admission_replays_original_budget_and_exact_input_control(imported, mutation, monkeypatch):
    runner = control.Controller(imported, ['SOFTWARE-ONLY'], authorize_work=lambda w: authority(imported, w),
        validate_result=verification, validate_bootstrap=lambda *a: dict(fixture=True))
    runner._open_phase('input')
    request, config, key = request_fixture(runner)
    try:
        if mutation == 'none':
            assert native._bootstrap_request(request, session_sha256=key, session=config,
                contract=imported._state.contract) == unpack(request)
        else:
            p = deepcopy(unpack(request))
            if mutation == 'session': p['session_sha256'] = digest('another session')
            elif mutation == 'deadline': p['deadline_ns'] += NS
            elif mutation == 'clock': p['started_ns'] += NS
            elif mutation == 'control': p['control_sha256'] = digest('missing control')
            elif mutation == 'expired':
                monkeypatch.setattr(native.time, 'monotonic_ns', lambda: p['deadline_ns'])
            elif mutation == 'wrong_root': config['ledger_root_sha256'] = digest('wrong root')
            with pytest.raises((ValueError, FileNotFoundError)):
                native._bootstrap_request(envelope(p), session_sha256=key, session=config,
                    contract=imported._state.contract)
    finally:
        monkeypatch.undo()
        runner._close_phase(last=True)


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows owned Job')
@pytest.mark.partial_phase_seconds(60)
@pytest.mark.parametrize('mode', ['bootstrap_counter', 'bootstrap_exception', 'bootstrap_child',
                                 'bootstrap_badbarrier', 'counter'])
def test_native_bootstrap_retains_worker_or_closes_failure_without_reserving_work(imported, mode):
    def validate(observation, output, tick):
        b = unpack(observation['barrier'])
        assert b['value'] == dict(software_fixture_only=True, bootstraps=1, prediction_calls=0)
        assert (output/'software-only.bin').read_bytes() == b'not a model or forecast'
        tick()
        return dict(software_fixture_only=True, exact_fixture_bytes=True)
    runner = control.Controller(imported, [sys.executable, '-u', str(WORKER.resolve()), '--mode', mode],
        authorize_work=lambda w: authority(imported, w), validate_result=verification, validate_bootstrap=validate)
    old = imported.summary()
    if mode == 'bootstrap_counter':
        result = runner.run_all(ORDER)
        assert len(result['settlements']) == 1
        assert result['summary']['generated_forecasts_reserved'] == old['generated_forecasts_reserved']
        observation = unpack(runner.bootstrap_receipt)['native_observation']
        b = unpack(observation)
        assert b['accounting']['active_processes'] >= 2
        q = unpack(read_json(runner.session.directory/'requests/000000.json'))
        assert q['previous_barrier_sha256'] == b['session_sha256'] and q['sequence'] == 0
        value = unpack(read_json(runner.session.directory/'outputs/000000/result.json'))['value']
        assert value['count'] == 1 and value['pid'] == b['worker_pid']
    else:
        with pytest.raises(native.SessionFailure): runner.run_all(ORDER)
        state = imported.summary()
        assert state['attempted_work_items'] == old['attempted_work_items']
        assert state['generated_forecasts_reserved'] == old['generated_forecasts_reserved']
        assert state['pending'] is None and state['active_control_sha256'] is None
        assert not list((imported.directory/'dispatches').glob('*.json'))
        assert state['halted_reason'] == 'supervision_error'
    closed = runner.session.close()
    assert closed['process_tree_closed']
    actual = native._query_job(runner.session.job_name)
    assert not actual['exists'] or actual['accounting']['active_processes'] == 0
    assert runner.session.sequence == (1 if mode == 'bootstrap_counter' else 0)
    with pytest.raises(ValueError, match='fresh healthy'):
        runner.session.bootstrap(digest('old control'), started_ns=time.monotonic_ns(),
            deadline_ns=time.monotonic_ns()+NS, tick=lambda: None)


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows owned Job')
@pytest.mark.partial_phase_seconds(14)
def test_native_expired_restoration_closes_tree_and_never_gets_a_forecast_token(imported):
    old = imported.summary()
    runner = control.Controller(imported,
        [sys.executable, '-u', str(WORKER.resolve()), '--mode', 'bootstrap_sleep'],
        authorize_work=lambda w: authority(imported, w), validate_result=verification,
        validate_bootstrap=lambda *a: dict(software_only=True))
    with pytest.raises((native.SessionFailure, control.ControllerStopped)): runner.run_all(ORDER)
    state = imported.summary()
    assert runner.session.closed and runner.session.failed
    assert state['attempted_work_items'] == old['attempted_work_items']
    assert state['generated_forecasts_reserved'] == old['generated_forecasts_reserved']
    assert state['charged_ns_by_phase']['input'] >= old['charged_ns_by_phase']['input']+5*NS
    assert state['charged_ns_by_phase']['method_forecasts'] == old['charged_ns_by_phase']['method_forecasts']
    assert state['pending'] is None and state['active_control_sha256'] is None
    actual = native._query_job(runner.session.job_name)
    assert not actual['exists'] or actual['accounting']['active_processes'] == 0
