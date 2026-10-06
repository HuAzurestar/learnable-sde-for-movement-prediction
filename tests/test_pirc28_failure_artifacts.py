"""Disposable failure provenance controls, never scientific study evidence."""

import copy
import json
import sys
import time

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import validate_failure
from application.research_reuse import verified_failure, verified_reuse
from application.research_supervisor import ResearchSupervisor
from experiments.pirc28.evidence import collect_qualification
from experiments.pirc28.failure_artifacts import failure_result
from experiments.pirc28.failures import NumericalCellFailure
from experiments.pirc28.pipeline import execute_recipe
from experiments.pirc28.runtime import switching_plugin
from infrastructure.research_store import ResearchError, digest
from models.switching import SwitchingError
from tests import test_pirc28_evidence as evidence_tests
from tests.test_pirc28_runtime import recipe
from tests.test_pirc28_synthetic import fixture_poll, model


@pytest.fixture(scope='module')
def failed_cell(tmp_path_factory):
    value = recipe()
    value['initializations']['2'] = [model(rate=8).to_dict(), model(rate=9).to_dict()]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evidence_tests, 'recipe', lambda: copy.deepcopy(value))
        store, study, definition, grant, runner = evidence_tests.prepared(
            tmp_path_factory.mktemp('pirc28-failure')/'fixture')
    cell = next(cell for cell in study['cells'] if cell['fit_K'] == 2)
    result = runner.run_cell(study['study_id'], digest(cell), budget=BudgetSpec(90))
    if result['state'] != 'FAILED' or result.get('artifact_id') is None:
        logs = list((store.path/'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(result)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))
    frame = json.loads(store.read_artifact(result['artifact_id'], purpose='evaluate', authorization=grant))
    return store, study, definition, grant, runner, cell, result, frame


def collect(data, **options):
    store, study, definition, grant, *_ = data
    return collect_qualification(store, definition, [study['study_id']], {study['study_id']: grant},
        check_budget=fixture_poll, allow_fixture=True, **options)


def test_actual_failed_worker_publishes_bounded_bound_diagnostics(failed_cell):
    store, study, _, _, _, cell, result, frame = failed_cell
    attempt = store.attempts()[result['attempt_id']]
    metadata = store.manifest('artifact-'+result['artifact_id'])
    assert attempt['artifact_manifest_hash'] == digest(metadata)
    assert metadata['role'] == 'failure' and metadata['study_id'] == study['study_id']
    assert metadata['block_ids'] == [cell['block_id']]
    assert frame['status'] == 'FAILED' and frame['error_code'] == 'INFERENCE_INAPPLICABLE'
    assert frame['qualified_identifiability'] is frame['real_data_authorized'] is False
    assert 'metrics' not in frame and 'forecast' not in frame
    assert [item['state'] for item in frame['failure']['initializations']] == ['FAILED', 'FAILED']
    receipt = store.manifest('admission-'+frame['admission_hash'])
    AdmissionGate(store).failure_validator(receipt, study, cell, switching_plugin())(copy.deepcopy(frame))
    report = collect(failed_cell)
    row = next(row for row in report['rows'] if row['artifact_hash'] == result['artifact_id'])
    assert row['state'] == 'INFERENCE_INAPPLICABLE' and row['input_hash']
    assert row['metrics'] is row['posterior_occupancy'] is row['collapsed'] is None
    assert report['state'] == 'FIXTURE_ONLY' and report['gate']['state'] == 'INCOMPLETE'
    history = next(row for row in report['lineage'] if row['selected_attempt_id'] == result['attempt_id'])['attempts']
    assert history[0]['failure'] == frame['failure']
    assert history[0]['cost']['charged_ms'] > 0
    assert len([event for event in store.events() if event['event_kind'] == 'RESERVE']) == 1


@pytest.mark.parametrize('damage', ['status', 'metrics', 'spec', 'protocol', 'input', 'recipe',
    'stage', 'missing-init', 'promoted-init', 'role', 'cause', 'qualified', 'real', 'admission'])
def test_failure_validator_refuses_rehashed_forgery(failed_cell, damage):
    store, study, _, _, _, cell, _, original = failed_cell
    frame = copy.deepcopy(original)
    body = frame['failure']
    if damage == 'status': frame['status'] = 'SUCCEEDED'
    elif damage == 'metrics': frame['metrics'] = {'position_energy_score_d2': 0.0}
    elif damage == 'spec': frame['spec_hash'] = digest('other spec')
    elif damage == 'protocol': frame['protocol_hash'] = digest('other protocol')
    elif damage == 'input': frame['input_hash'] = digest('other data')
    elif damage == 'recipe': body['recipe_hash'] = digest('other recipe')
    elif damage == 'stage': body['stage'] = 'forecast'
    elif damage == 'missing-init': body['initializations'].pop()
    elif damage == 'promoted-init': body['initializations'][0]['state'] = 'SUCCEEDED'
    elif damage == 'role': body['generated_inputs'][0]['role'] = 'test'
    elif damage == 'cause': body['cause'] = {'error_code': 'TIMEOUT'}
    elif damage == 'qualified': frame['qualified_identifiability'] = True
    elif damage == 'real': frame['real_data_authorized'] = True
    else: frame['admission_hash'] = digest('other admission')
    frame['output_hash'] = digest({'failure': body, 'source_schema': frame['source_schema']})
    receipt = store.manifest('admission-'+original['admission_hash'])
    with pytest.raises(ResearchError):
        AdmissionGate(store).failure_validator(receipt, study, cell, switching_plugin())(frame)


def test_failed_artifact_is_verified_but_never_success_reuse(failed_cell):
    store, study, _, _, _, cell, result, _ = failed_cell
    attempt = store.attempts()[result['attempt_id']]
    checked = verified_failure(store, attempt, study, cell, switching_plugin())
    assert checked['state'] == 'FAILED' and checked['reused'] is False and checked['exit_code'] != 0
    with pytest.raises(ResearchError):
        verified_reuse(store, attempt, study, cell, switching_plugin())


@pytest.mark.parametrize('damage', ['missing-stop', 'wrong-stop-time', 'wrong-confirmation',
                                  'wrong-arm', 'wrong-charge', 'wrong-outcome'])
def test_failure_provenance_requires_actual_stop_and_original_cost(failed_cell, monkeypatch, damage):
    store, study, _, _, _, cell, result, _ = failed_cell
    attempt = store.attempts()[result['attempt_id']]
    events = copy.deepcopy(store.events())
    if damage == 'missing-stop':
        events = [event for event in events if event['event_kind'] != 'WORKER_TREE_STOPPED']
    elif damage.startswith('wrong-stop') or damage == 'wrong-confirmation':
        stop = next(event for event in events if event['event_kind'] == 'WORKER_TREE_STOPPED')['payload']
        if damage == 'wrong-stop-time': stop['observed_elapsed_ms'] += 1
        else: stop['confirmation'] = 'worker self-report'
    else:
        cost = next(event for event in events if event['event_kind'] == 'SETTLE')['payload']
        if damage == 'wrong-arm': cost['arm_id'] = 'fresh-budget-alias'
        elif damage == 'wrong-charge': cost['charged_ms'] += 1
        else: cost['outcome'] = 'SUCCEEDED'
    monkeypatch.setattr(store, '_events', lambda: events)
    # A forged read view must not append into the real fixture journal.
    monkeypatch.setattr(store, '_append', lambda *args, **kwargs: None)
    with pytest.raises(ResearchError, match='UNQUALIFIED'):
        verified_failure(store, attempt, study, cell, switching_plugin())


def test_failed_diagnostics_require_grant_before_any_read(failed_cell):
    store, study, definition, grant, *_ = failed_cell
    start = len(store.events())
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        collect_qualification(store, definition, [study['study_id']],
            {study['study_id']: {**grant, 'purposes': ['preview']}},
            check_budget=fixture_poll, allow_fixture=True)
    assert not any(event['event_kind'] == 'READ_STARTED' for event in store.events()[start:])


def test_failed_diagnostics_are_subject_to_total_byte_quota(failed_cell):
    with pytest.raises(ResearchError, match='RESOURCE_LIMIT'):
        collect(failed_cell, maximum_total_bytes=1)


def test_explicit_retry_retains_both_failures_and_original_arm_charge(failed_cell):
    store, study, _, _, runner, cell, first, _ = failed_cell
    before = BudgetLedger(store).balance('pirc28-k2')['committed_ms']
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
        runner.run_cell(study['study_id'], digest(cell), budget=BudgetSpec(90))
    second = runner.run_cell(study['study_id'], digest(cell), budget=BudgetSpec(90),
        parent_attempt_id=first['attempt_id'], reason='explicit whole-cell failure control retry')
    assert second['state'] == 'FAILED' and second['artifact_id'] != first['artifact_id']
    report = collect(failed_cell)
    lineage = next(row for row in report['lineage'] if row['selected_attempt_id'] == second['attempt_id'])
    assert len(lineage['attempts']) == 2 and all(item['failure'] for item in lineage['attempts'])
    assert BudgetLedger(store).balance('pirc28-k2')['committed_ms'] > before
    reserves = [event['payload'] for event in store.events() if event['event_kind'] == 'RESERVE']
    assert len(reserves) == 2 and all(item['arm_id'] == 'pirc28-k2' for item in reserves)


@pytest.mark.parametrize('stage,target', [('generation', 'generate_scenario'),
    ('test-inference', 'filter_switching'), ('forecast', 'forecast_switching'),
    ('evaluation', 'energy_score')])
def test_whole_cell_numerical_stages_retain_complete_history(monkeypatch, stage, target):
    def fail(*args, **kwargs):
        raise SwitchingError('NONFINITE', 'private raw numerical exception')
    monkeypatch.setattr('experiments.pirc28.pipeline.'+target, fail)
    value = recipe()
    with pytest.raises(NumericalCellFailure) as caught:
        execute_recipe(value, 2, 19, check_budget=fixture_poll)
    diagnostic = caught.value.diagnostic
    assert diagnostic['stage'] == stage and diagnostic['error_code'] == 'NONFINITE'
    assert 'private raw numerical exception' not in str(diagnostic)
    study = {'switching_recipe': value, 'protocol_hash': digest('protocol'), 'data_hash': digest('input')}
    cell = {'fit_K': 2, 'seed': 19}
    plugin = switching_plugin()
    frame = failure_result(diagnostic, study, cell, plugin)
    validate_failure(frame, spec=study, cell=cell, plugin=plugin)
    if stage == 'generation':
        assert all(item['state'] == 'NOT_RUN' for item in diagnostic['initializations'])
    else:
        assert diagnostic['selection']['selected_initialization'] is not None
        assert len(diagnostic['generated_inputs']) == 3


def test_legacy_adapter_cannot_publish_exit86_as_failure_evidence(tmp_path):
    from tests.test_research_budget import registered
    store, attempts = registered(tmp_path)
    command = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"error_code\":\"NONFINITE\"}'); sys.exit(86)"
    result = ResearchSupervisor(store).run(attempts[0],
        lambda output: [sys.executable, '-c', command, str(output)], BudgetSpec(5))
    assert result['state'] == 'FAILED' and result['artifact_id'] is None
    assert store.attempts()[attempts[0]]['error_code'] == 'WORKER_FAILED'


def test_all_selection_failures_publish_complete_fit_and_null_score_history(monkeypatch):
    import experiments.pirc28.pipeline as pipeline
    select = pipeline.select_initialization
    def failed_selection(*args, **kwargs):
        def fail(*args, **kwargs):
            raise FloatingPointError('private raw selection error')
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr('estimation.switching._filters', fail)
            return select(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'select_initialization', failed_selection)
    value = recipe()
    with pytest.raises(NumericalCellFailure) as caught:
        execute_recipe(value, 2, 19, check_budget=fixture_poll)
    diagnostic = caught.value.diagnostic
    assert diagnostic['stage'] == 'selection' and diagnostic['cause'] is None
    assert diagnostic['selection']['selected_initialization'] is None
    assert all(item['state'] == 'SUCCEEDED' for item in diagnostic['initializations'])
    assert all(item['score'] is None for item in diagnostic['selection']['candidates'])
    study = {'switching_recipe': value, 'protocol_hash': digest('protocol'), 'data_hash': digest('input')}
    cell = {'fit_K': 2, 'seed': 19}
    plugin = switching_plugin()
    validate_failure(failure_result(diagnostic, study, cell, plugin), spec=study, cell=cell, plugin=plugin)


@pytest.mark.parametrize('error', [ResearchError('TIMEOUT', 'owner refusal'),
    SwitchingError('UNAUTHORIZED_DATA', 'not numeric'), SwitchingError('CONTRACT_MISMATCH', 'bad contract'),
    RuntimeError('unknown bug')])
def test_whole_cell_wrapper_preserves_non_numerical_generation_errors(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr('experiments.pirc28.pipeline.generate_scenario', fail)
    with pytest.raises(type(error)) as caught:
        execute_recipe(recipe(), 2, 19, check_budget=fixture_poll)
    assert caught.value is error


def test_delayed_failure_validation_cannot_publish_after_deadline(tmp_path):
    from tests.test_research_budget import registered
    store, attempts = registered(tmp_path)
    command = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"error_code\":\"NONFINITE\"}'); sys.exit(86)"
    calls = []
    def delayed(frame):
        calls.append(frame)
        time.sleep(3.2)
    result = ResearchSupervisor(store).run(attempts[0],
        lambda output: [sys.executable, '-c', command, str(output)], BudgetSpec(3),
        failure_validator=delayed)
    assert calls, 'test must cross the deadline inside actual failure validation'
    assert result['state'] == 'TIMEOUT' and result['artifact_id'] is None
    assert store.attempts()[attempts[0]]['state'] == 'TIMEOUT'
    assert BudgetLedger(store).balance('affine')['closed']
