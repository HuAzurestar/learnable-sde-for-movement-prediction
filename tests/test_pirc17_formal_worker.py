"""Real scientific routing on synthetic inputs and real file-ledger requests.

The in-process driver is NOT native OS closure proof or human authorization.
One separate Windows test invokes the actual retained handler in the owned
session and verifies that its real input guard denies before data acquisition.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import shutil
import sys
import time

import pytest

from experiments.pirc17 import formal_worker as module
from experiments.pirc17 import formal_budget as budget, formal_controller as control, formal_session as native
from experiments.pirc17 import formal_forecast_records as records
from experiments.pirc17.final_eval_guard import VerifiedAccess
from experiments.pirc17.formal_input_work import PreparedInputWork
from experiments.pirc17.protocol_core import canonical, digest, publish, read_json, unpack
from tests.test_pirc17_formal_controller import authority, verification
from tests.test_pirc17_formal_forecasts import prepared
from tests.test_pirc17_formal_input_work import case, orchestrator


class FileRequests:
    """No subprocesses/native-observation claim; actual reserve/result/settle IO."""
    def __init__(self, case, root):
        inputs = orchestrator(case, root)
        inputs.authority['journal_directory'] = case['args']['access_journal']
        self.options = dict(protocol=inputs.protocol, execution=inputs.execution, matrix=inputs.matrix,
                            authority=inputs.authority, **inputs.paths)
        scope = case['scope']
        self.contract = budget.contract_for_matrix(scope['matrix'], protocol_sha256=scope['protocol']['sha256'],
            execution_sha256=scope['execution']['sha256'], runtime_manifest_sha256=digest('SYNTHETIC RUNTIME'),
            approval_sha256=scope['approval_sha256'], ledger_directory=root/'ledger')
        self.ledger = budget.Ledger.create(root/'ledger', self.contract)
        self.directory = root/'ledger'/'test-session'
        for path in ('outputs', 'requests', 'barriers'): (self.directory/path).mkdir(parents=True, exist_ok=True)
        (self.ledger.directory/'dispatches').mkdir(exist_ok=True)
        self.session = dict(schema_version=native.VERSION, directory=str(self.directory.resolve()),
            ledger_directory=str(self.ledger.directory), ledger_root_sha256=self.ledger.root_sha256,
            execution_sha256=scope['execution']['sha256'], job_name='PIRC17-FORMAL-'+'a'*32,
            worker_command=['SOFTWARE FILE REQUESTS, NOT AN OS SESSION'])
        self.worker = module.FormalWorker(self.session, self.contract, input_options=self.options)
        self.previous, self.sequence = digest(self.session), 0

    def begin(self, full):
        self.started = time.monotonic_ns()
        self.reservation = self.ledger.reserve(full['work_id'])
        self.work = self.ledger._state.work[full['work_id']]
        self.output = self.directory/'outputs'/f'{self.sequence:06d}'; self.output.mkdir()
        self.request = native._write(self.directory/'requests'/f'{self.sequence:06d}.json', dict(
            schema_version=native.VERSION+'-request', session_sha256=digest(self.session), sequence=self.sequence,
            previous_barrier_sha256=self.previous, work_id=full['work_id'], reservation_sha256=self.reservation['reservation_sha256'],
            reservation_event_index=self.ledger.tip['event_count']-1, started_ns=self.started,
            deadline_ns=self.started+self.reservation['reserved_ns']))
        native._write(self.ledger.directory/'dispatches'/(self.reservation['reservation_sha256']+'.json'), dict(
            schema_version=native.VERSION+'-dispatch', ledger_root_sha256=self.ledger.root_sha256,
            reservation_sha256=self.reservation['reservation_sha256'], work_id=full['work_id'],
            session_directory=str(self.directory.resolve()), job_name=self.session['job_name']))

    def execute(self, full, *, settlement_status='success'):
        self.begin(full)
        manifest = self.worker(self.work, self.output)
        result = native._write(self.output/'result.json', dict(schema_version=native.VERSION+'-result',
            session_sha256=digest(self.session), sequence=self.sequence, reservation_sha256=self.reservation['reservation_sha256'],
            work_id=full['work_id'], value=manifest))
        barrier = native._write(self.directory/'barriers'/f'{self.sequence:06d}.json', dict(schema_version=native.VERSION+'-barrier',
            session_sha256=digest(self.session), sequence=self.sequence, previous_barrier_sha256=self.previous,
            request_sha256=self.request['sha256'], reservation_sha256=self.reservation['reservation_sha256'],
            work_id=full['work_id'], worker_pid=os.getpid(), status='success', result_sha256=result['sha256'], error_type=None))
        self.ledger.settle(self.reservation['reservation_sha256'], status=settlement_status,
            elapsed_ns=time.monotonic_ns()-self.started, completion_evidence_sha256=digest('NO NATIVE CLOSURE: SOFTWARE FILE DRIVER'),
            result_sha256=result['sha256'] if settlement_status == 'success' else None, reason='software request routing only')
        self.previous, self.sequence = barrier['sha256'], self.sequence+1
        return manifest

    def close(self):
        self.worker.close(); self.ledger.close()


@pytest.fixture
def driver(case, prepared, tmp_path, monkeypatch):
    # Replace acquisition ONLY. Fits, prediction kernels, scoring, typed domain
    # restoration and incremental committed-event consumption are real.
    calls = []
    def prepared_input(owner, work, *, output_directory):
        calls.append(work['work_id'])
        assert len(calls) == 1
        output = Path(output_directory)
        for path in case['source'].rglob('*'):
            if path.is_file():
                dest = output/path.relative_to(case['source']); dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        manifest = dict(case['manifest'])
        for key in ('context_path', 'qualification_path'):
            manifest[key] = str(output/Path(manifest[key]).relative_to(case['source']))
        q = unpack(case['qualification'])
        paths = dict(population_path=output/q['population_path'], population_sha256=q['population_sha256'],
                     eligibility_path=output/q['eligibility_path'])
        context = {k: v for k, v in case['args'].items() if k != 'access_journal'}
        training = replace(prepared['fits'].inputs, identity=context['input_identity'])
        case['maps'].catalog = context['map_catalog']
        return PreparedInputWork(context, training, case['maps'], paths, manifest)
    monkeypatch.setattr(module.InputWork, 'execute', prepared_input)
    value = FileRequests(case, tmp_path)
    value.input_calls = calls
    yield value
    value.close()


def select(worker, kind, *, subject='arm-01/full', rank=0, repetition=None):
    return next(w for w in worker.works.values() if w['kind'] == kind and w['origin_mode'] == 'causal_prefix'
        and w['origin_rank'] == rank and (w['seed'] is None or w['seed'] == 20260814)
        and w['repetition'] == repetition and (kind in {'common_scores', 'inertial_path'} or w['subject'] == subject))


def test_actual_fit_forecast_score_routes_reuse_one_worker_and_only_closed_dependencies(driver, case, monkeypatch):
    counts, events = Counter(), Counter()
    for name in ('FitConsumers', 'ForecastConsumers', 'SavedForecasts', 'ScoringConsumers'):
        actual = getattr(module, name)
        def counted(*args, _name=name, _actual=actual, **kwargs):
            counts[_name] += 1
            if _name == 'ForecastConsumers': assert set(kwargs) == {'fits', 'cases', 'population', 'maps'}
            return _actual(*args, **kwargs)
        monkeypatch.setattr(module, name, counted)
    original_read = native._read
    def read(path, *args, **kwargs):
        if Path(path).parent == driver.ledger.directory/'events': events[Path(path).name] += 1
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(native, '_read', read)
    def access(owner):
        # Synthetic metrics authorization adapter, not real ACCEPT01. Actual
        # source guards are independently tested and remain unmodified.
        scope = unpack(owner.saved.input_identity) if hasattr(owner, 'saved') else unpack(owner.args['input_identity'])
        _, event = publish(case['args']['access_journal'], dict(schema_version='pirc17-final-access-event-v1',
            event='started', access_kind='final_eval_metrics', attempt_id=f'{driver.sequence:032x}', at_utc='2026-09-30T00:00:00+00:00',
            **{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')}))
        return VerifiedAccess(access_kind='final_eval_metrics', legacy_cohort_ack='SYNTHETIC',
            access_started_sha256=event['sha256'], **{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')})
    def score(*, scorer, work, output_directory, **kwargs):
        return scorer._execute(access(scorer), work, output_directory)
    def analyze(*, analysis, work, output_directory, **kwargs):
        return analysis._execute(access(analysis), work, output_directory)
    def audit(*, audit, work, output_directory, **kwargs):
        return audit._execute(access(audit), work, output_directory)
    monkeypatch.setattr(module, 'score_formal_work', score)
    monkeypatch.setattr(module, 'analyze_formal_work', analyze)
    monkeypatch.setattr(module, 'reanalyze_formal_work', audit)
    worker = driver.worker
    driver.execute(case['work'])
    fits = [w for w in worker.works.values() if w['kind'] in {'method_fit', 'terrain_fit'}]
    for work in fits: driver.execute(work)
    assert len(fits) == 26 and len(worker.fits.receipts) == 26
    assert counts == {name: 1 for name in ('FitConsumers', 'ForecastConsumers', 'SavedForecasts', 'ScoringConsumers')}
    assert fits[-1]['work_id'] not in worker.completed  # Produced != admitted.
    first = select(worker, 'scientific_forecast')
    output = driver.execute(first)
    assert first['work_id'] not in worker.saved.index and output['status'] == 'success'
    driver.execute(select(worker, 'scientific_forecast', subject='base'))
    assert worker.saved.read(first).status == 'success'
    driver.execute(select(worker, 'inertial_path'))
    score_output = driver.execute(select(worker, 'common_scores'))
    payload = unpack(read_json(score_output['artifact_path']))
    assert payload['counts_by_status'] == {'success': 3, 'unavailable': 193}
    assert payload['new_forecasts'] == payload['new_fits'] == 0
    # Later runtime records must not invalidate an earlier score dependency set.
    for kind, repetition in [('forecast_replay', None), ('runtime_cold', 0), ('runtime_warmup', 0), ('runtime_warm', 0)]:
        assert driver.execute(select(worker, kind, repetition=repetition))['status'] == 'success'
    shortfall = driver.execute(select(worker, 'common_scores', rank=1))
    assert unpack(read_json(shortfall['artifact_path']))['counts_by_status'] == {'NOT_ADMITTED': 196}
    worker.scorer.verify(read_json(score_output['artifact_path']), work=select(worker, 'common_scores'),
                         access_started_sha256=payload['metrics_access_started_sha256'])
    analysis_work = next(w for w in worker.works.values() if w['kind'] == 'mechanisms_and_inference')
    analysis = unpack(read_json(driver.execute(analysis_work)['artifact_path']))
    assert len(analysis['method_ledger']) == 36 and len(analysis['factor_conclusions']) == 4
    assert analysis['new_forecasts'] == analysis['new_fits'] == 0
    assert counts == {name: 1 for name in counts} and driver.input_calls == [case['work']['work_id']]
    assert max(events.values()) <= 2  # Reserve checked once by replay and once by request; no full rescans.
    assert len(events) == worker.state.count
    assert not case['maps'].closed
    # These files are not real native supervision evidence. The actual audit
    # route must refuse them rather than certify the in-process test driver.
    audit_work = next(w for w in worker.works.values() if w['kind'] == 'independent_reanalysis')
    driver.begin(audit_work)
    with pytest.raises(ValueError, match='bounded regular JSON'):
        worker(driver.work, driver.output)
    assert worker.failed and worker.closed and case['maps'].closed


@pytest.mark.parametrize('fault', ['failed_settlement', 'result', 'barrier', 'current_head', 'event_root', 'work', 'output'])
def test_handler_rejects_unsettled_or_forged_prior_output_before_next_computation(driver, case, fault):
    driver.execute(case['work'], settlement_status='failure' if fault == 'failed_settlement' else 'success')
    next_work = next(w for w in driver.worker.works.values() if w['kind'] == 'method_fit')
    driver.begin(next_work)
    if fault in {'result', 'barrier', 'current_head', 'event_root'}:
        path = (driver.directory/'outputs/000000/result.json' if fault == 'result' else driver.directory/'barriers/000000.json'
            if fault == 'barrier' else driver.ledger.directory/'head.json' if fault == 'current_head'
            else driver.ledger.directory/'events'/f"{driver.ledger.tip['event_count']-1:06d}.json")
        value = deepcopy(unpack(read_json(path)))
        if fault == 'result': value['value']['context_sha256'] = '0'*64
        elif fault == 'barrier': value['worker_pid'] += 1
        elif fault == 'current_head': value['last_event_sha256'] = '0'*64
        else: value['root_sha256'] = '0'*64
        path.write_bytes(canonical({'payload': value, 'sha256': digest(value)}))
    work = dict(driver.work, generated_forecasts=99) if fault == 'work' else driver.work
    output = driver.output
    if fault == 'output': (output/'unexpected.bin').write_bytes(b'NOT EMPTY')
    with pytest.raises(ValueError): driver.worker(work, output)
    assert driver.worker.fits.receipts == {} and driver.worker.failed and driver.worker.closed
    assert case['maps'].closed
    with pytest.raises(ValueError, match='cannot restart'): driver.worker(driver.work, driver.output)


@pytest.mark.parametrize('skip', ['input', 'fits'])
def test_input_must_start_session_and_missing_fit_cannot_trigger_implicit_training(driver, case, skip):
    if skip == 'input':
        work = next(w for w in driver.worker.works.values() if w['kind'] == 'method_fit')
        message = 'input work must be successfully settled'
    else:
        driver.execute(case['work'])
        work = select(driver.worker, 'scientific_forecast')
        message = 'fit owners must settle'
    driver.begin(work)
    with pytest.raises(ValueError, match=message):
        driver.worker(driver.work, driver.output)
    assert driver.worker.fits is None or driver.worker.fits.receipts == {}
    assert driver.worker.forecasts is None


def test_actual_native_retained_handler_denies_missing_approval_before_any_raw_input(case, tmp_path):
    if os.name != 'nt': pytest.skip('native Windows handler')
    # No acquisition/metric adapters here: the child runs actual InputWork.
    inputs = orchestrator(case, tmp_path)
    config = dict(protocol=inputs.protocol, execution=inputs.execution, matrix=inputs.matrix,
                  authority=inputs.authority, **inputs.paths)
    config['authority'] = {k: str(v) if isinstance(v, Path) else v for k, v in config['authority'].items()}
    config.update({k: str(v) for k, v in inputs.paths.items()})
    config_path = tmp_path/'native-input-options.json'; config_path.write_bytes(canonical(config))
    scope = case['scope']
    contract = budget.contract_for_matrix(scope['matrix'], protocol_sha256=scope['protocol']['sha256'],
        execution_sha256=scope['execution']['sha256'], runtime_manifest_sha256=digest('SYNTHETIC RUNTIME'),
        approval_sha256=scope['approval_sha256'], ledger_directory=tmp_path/'native-ledger')
    with budget.Ledger.create(tmp_path/'native-ledger', contract) as ledger:
        runner = control.Controller(ledger, [sys.executable, '-u', str((Path(__file__).parent/'fixtures/pirc17_session_worker.py').resolve()),
            '--mode', 'formal_input_denial', '--fixture-directory', str(config_path)],
            authorize_work=lambda w: authority(ledger, w), validate_result=verification)
        try:
            runner._open_phase(case['work']['phase'])
            with pytest.raises(control.ControllerStopped): runner._work(ledger._state.work[case['work']['work_id']])
            assert ledger.dispositions()[case['work']['work_id']] == 'failure'
            barrier = unpack(read_json(runner.session.directory/'barriers/000000.json'))
            assert barrier['status'] == 'failure' and barrier['error_type'] == 'ValueError'
            assert not (runner.session.directory/'outputs/000000/qualification').exists()
            assert not inputs.authority['journal_directory'].exists()
        finally:
            runner._close_phase(last=True, stopped=True)
            runner.session.close()
        assert native._query_job(runner.session.job_name)['exists'] is False
