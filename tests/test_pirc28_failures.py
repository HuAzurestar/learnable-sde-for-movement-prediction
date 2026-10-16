"""Temporary numerical controls, not independent scientific study replicates."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from estimation.switching import SwitchingFit, select_initialization
from experiments.pirc28.evidence import _success_row
from experiments.pirc28.failures import NumericalCellFailure
from experiments.pirc28.pipeline import execute_recipe
from inference.switching import InferenceSpec
from infrastructure.research_store import ResearchError, digest, encode
from models.switching import SwitchingError
from tests.test_pirc28_qualification import spec
from tests.test_pirc28_runtime import recipe
from tests.test_pirc28_synthetic import fixture_poll, model, generate_scenario, scenario
from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc28.evidence import collect_qualification


def partial_recipe():
    value = recipe()
    value['initializations']['2'] = [model(rate=8).to_dict(), model().to_dict()]
    original = spec(2)
    qualification = replace(original, scenario_axes={value['scenario']['scenario_id']: original.scenario_axes['scenario']})
    value['identifiability_spec'] = qualification.to_dict()
    return value, qualification


@pytest.fixture(scope='module')
def partial():
    value, qualification = partial_recipe()
    snapshots = []
    result = execute_recipe(value, 2, 19, check_budget=fixture_poll, progress_callback=snapshots.append)
    return value, qualification, result, snapshots


def test_real_quota_failure_retains_both_initializations_without_zero_scores(partial):
    _, _, result, snapshots = partial
    failed, successful = result['fit']['initializations']
    assert failed['state'] == 'FAILED' and failed['error_code'] == 'INFERENCE_INAPPLICABLE'
    assert failed['last_accepted_state'] is None and 'model' not in failed
    assert successful['state'] == 'SUCCEEDED'
    assert result['fit']['selection']['selected_initialization'] == 'init-1'
    assert result['fit']['selection']['candidates'][0]['score'] is None
    assert result['fit']['selection']['candidates'][1]['state'] == 'SUCCEEDED'
    assert snapshots[-1]['fit_failures'][0] == failed
    assert digest(result) == digest(json.loads(encode(result)))


def test_recovery_preserves_failure_and_does_not_retry_or_reset_it(partial, monkeypatch):
    value, _, result, snapshots = partial
    saved = next(state for state in snapshots if state['fit_failures'][0] is not None)
    import experiments.pirc28.pipeline as pipeline
    fit = pipeline.fit_switching
    called = []
    def checked(*args, **kwargs):
        called.append(kwargs['initialization_id'])
        assert kwargs['initialization_id'] != 'init-0'
        return fit(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'fit_switching', checked)
    assert execute_recipe(value, 2, 19, check_budget=fixture_poll, resume_state=saved) == result
    assert called == ['init-1']


@pytest.mark.parametrize('damage', ['code', 'model', 'count', 'state'])
def test_recovery_refuses_changed_failure_identity(partial, damage):
    value, _, _, snapshots = partial
    saved = copy.deepcopy(next(state for state in snapshots if state['fit_failures'][0] is not None))
    if damage == 'code':
        saved['fit_failures'][0]['error_code'] = 'UNAUTHORIZED_DATA'
    elif damage == 'model':
        saved['fit_failures'][0]['initial_model_hash'] = digest('another initializer')
    elif damage == 'count':
        saved['fit_failures'].pop()
    else:
        saved['fit_failures'][0]['last_accepted_state'] = {'not': 'an accepted EM state'}
    with pytest.raises(SwitchingError):
        execute_recipe(value, 2, 19, check_budget=fixture_poll, resume_state=saved)


def test_all_fits_fail_with_complete_typed_cell_diagnostic():
    value = recipe()
    value['initializations']['2'] = [model(rate=8).to_dict(), model(rate=9).to_dict()]
    with pytest.raises(NumericalCellFailure) as failure:
        execute_recipe(value, 2, 19, check_budget=fixture_poll)
    diagnostic = failure.value.diagnostic
    assert diagnostic['stage'] == 'fit' and diagnostic['error_code'] == 'INFERENCE_INAPPLICABLE'
    assert [value['initialization_id'] for value in diagnostic['initializations']] == ['init-0', 'init-1']
    assert len(diagnostic['generated_inputs']) == 3
    assert 'metrics' not in diagnostic and all(value['state'] == 'FAILED' for value in diagnostic['initializations'])


@pytest.mark.parametrize('error', [ResearchError('TIMEOUT', 'stop'), SwitchingError('IDENTITY_CONFLICT', 'stop'),
    SwitchingError('UNAUTHORIZED_DATA', 'stop'), SwitchingError('CONTRACT_MISMATCH', 'stop'), RuntimeError('bug')])
def test_fit_does_not_downgrade_authority_contract_or_unknown_errors(monkeypatch, error):
    calls = []
    def refuse(*args, **kwargs):
        calls.append(kwargs['initialization_id'])
        raise error
    monkeypatch.setattr('experiments.pirc28.pipeline.fit_switching', refuse)
    with pytest.raises(type(error)) as caught:
        execute_recipe(partial_recipe()[0], 2, 19, check_budget=fixture_poll)
    assert caught.value is error and calls == ['init-0']


def selection_inputs():
    obs, _, _ = generate_scenario(scenario(k=1, paths=1), check_budget=fixture_poll)
    inference = InferenceSpec(max_iterations=1)
    fits = [SwitchingFit(model(k=1), (0.0,), 'MAX_ITERATIONS', np.ones(1), False, (),
        'init-'+str(index), inference.to_dict(), ()) for index in range(2)]
    return obs, fits, inference


@pytest.mark.parametrize('error', [SwitchingError('NONFINITE', 'private payload'),
    np.linalg.LinAlgError('private payload'), FloatingPointError('private payload'), OverflowError('private payload')])
def test_selection_skips_only_typed_numerical_failures_and_retains_null_scores(monkeypatch, error):
    obs, fits, inference = selection_inputs()
    calls = []
    def selected_filters(*args):
        calls.append(args[1])
        if len(calls) == 1:
            raise error
        return [SimpleNamespace(log_likelihood=-7.0)]
    monkeypatch.setattr('estimation.switching._filters', selected_filters)
    chosen, receipt = select_initialization(fits, obs, [1.0], np.zeros((1, 4)), np.eye(4)[None],
        inference, split_role='selection', check_budget=fixture_poll)
    assert chosen is fits[1] and len(calls) == 2
    assert receipt['candidates'][0]['score'] is None and receipt['candidates'][0]['state'] == 'FAILED'
    assert 'private payload' not in str(receipt)


def test_nonfinite_selection_objectives_are_failures_not_winners(monkeypatch):
    obs, fits, inference = selection_inputs()
    monkeypatch.setattr('estimation.switching._filters', lambda *args: [SimpleNamespace(log_likelihood=float('nan'))])
    with pytest.raises(NumericalCellFailure) as caught:
        select_initialization(fits, obs, [1.0], np.zeros((1, 4)), np.eye(4)[None],
            inference, split_role='selection', check_budget=fixture_poll)
    assert caught.value.diagnostic['stage'] == 'selection'
    assert all(value['error_code'] == 'NONFINITE' and value['score'] is None
               for value in caught.value.diagnostic['selection']['candidates'])


@pytest.mark.parametrize('error', [ResearchError('BUDGET_EXHAUSTED', 'stop'),
    SwitchingError('IDENTITY_CONFLICT', 'stop'), SwitchingError('UNAUTHORIZED_DATA', 'stop'), RuntimeError('bug')])
def test_selection_does_not_downgrade_authority_or_unknown_failures(monkeypatch, error):
    obs, fits, inference = selection_inputs()
    calls = []
    def refuse(*args):
        calls.append(args)
        raise error
    monkeypatch.setattr('estimation.switching._filters', refuse)
    with pytest.raises(type(error)) as caught:
        select_initialization(fits, obs, [1.0], np.zeros((1, 4)), np.eye(4)[None],
            inference, split_role='selection', check_budget=fixture_poll)
    assert caught.value is error and len(calls) == 1


def test_failure_after_accepted_em_progress_survives_recovery(monkeypatch):
    import estimation.switching as estimation
    proposal = estimation._proposal
    calls = []
    def fail_second(*args):
        calls.append(args)
        if len(calls) == 2:
            raise SwitchingError('NONFINITE', 'numerical proposal failure')
        return proposal(*args)
    monkeypatch.setattr(estimation, '_proposal', fail_second)
    value = recipe()
    first = model().to_dict()
    first['a'] = [[0.0, 0.0], [0.0, 0.0]]
    value['initializations']['2'] = [first, model().to_dict()]
    snapshots = []
    expected = execute_recipe(value, 2, 19, check_budget=fixture_poll, progress_callback=snapshots.append)
    failure = expected['fit']['initializations'][0]
    assert failure['state'] == 'FAILED'
    assert failure['last_accepted_state']['completed_iterations'] == 1
    assert len(failure['last_accepted_state']['objective_trace']) == 2
    saved = next(state for state in snapshots if state['fit_failures'][0] is not None)
    monkeypatch.setattr(estimation, '_proposal', proposal)
    assert execute_recipe(value, 2, 19, check_budget=fixture_poll, resume_state=saved) == expected
    damaged = copy.deepcopy(saved)
    damaged['fit_failures'][0]['last_accepted_state']['identity']['data_hash'] = digest('different train data')
    with pytest.raises(SwitchingError) as caught:
        execute_recipe(value, 2, 19, check_budget=fixture_poll, resume_state=damaged)
    assert caught.value.code == 'IDENTITY_CONFLICT'


def test_actual_managed_partial_failure_keeps_original_arm_and_discloses_lineage(tmp_path, monkeypatch):
    from tests import test_pirc28_evidence as evidence_tests
    monkeypatch.setattr(evidence_tests, 'recipe', lambda: partial_recipe()[0])
    store, value, qualification, grant, runner = evidence_tests.prepared(tmp_path/'managed-partial')
    cell = next(cell for cell in value['cells'] if cell['fit_K'] == 2)
    result = runner.run_cell(value['study_id'], digest(cell), budget=BudgetSpec(90))
    if result['state'] != 'SUCCEEDED':
        logs = list((store.path/'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(result)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))
    report = collect_qualification(store, qualification, [value['study_id']], {value['study_id']: grant},
        check_budget=fixture_poll, allow_fixture=True)
    assert report['state'] == 'FIXTURE_ONLY' and report['gate']['state'] == 'INCOMPLETE'
    assert report['qualified_identifiability'] is report['real_data_authorized'] is False
    lineage = next(row for row in report['lineage'] if row['selected_attempt_id'] == result['attempt_id'])
    failed, successful = lineage['initialization_diagnostics']
    assert failed['fit_error_code'] == 'INFERENCE_INAPPLICABLE' and failed['selection_score'] is None
    assert failed['collapsed'] is None and successful['fit_state'] == 'SUCCEEDED'
    assert len(store.attempts()) == 1
    assert BudgetLedger(store).balance('pirc28-k2')['committed_ms'] > 0
    reserves = [event['payload'] for event in store.events() if event['event_kind'] == 'RESERVE']
    assert len(reserves) == 1 and reserves[0]['arm_id'] == 'pirc28-k2'


def test_source_collector_accepts_complete_partial_failure_history(partial):
    value, qualification, result, _ = partial
    row = _success_row(qualification, {'switching_recipe': value},
        {'fit_K': 2, 'seed': 19, 'replicate_id': 'rep-0'}, result, digest(result))
    assert row['state'] == 'SUCCEEDED'


@pytest.mark.parametrize('damage', ['hidden-failure', 'failed-winner', 'zero-score', 'initial-model',
                                  'false-state', 'success-model', 'success-trace'])
def test_source_collector_refuses_suppressed_or_promoted_initializer_failures(partial, damage):
    value, qualification, original, _ = partial
    result = copy.deepcopy(original)
    if damage == 'hidden-failure':
        result['fit']['initializations'].pop(0)
    elif damage == 'failed-winner':
        result['fit']['selection']['selected_initialization'] = 'init-0'
    elif damage == 'zero-score':
        result['fit']['selection']['candidates'][0]['score'] = 0.0
    elif damage == 'initial-model':
        result['fit']['initializations'][0]['initial_model_hash'] = digest('wrong')
    elif damage == 'false-state':
        result['fit']['selection']['candidates'][0]['state'] = 'SUCCEEDED'
    elif damage == 'success-model':
        result['fit']['initializations'][1]['model'] = None
        result['fit']['initializations'][1]['model_hash'] = digest(None)
    else:
        result['fit']['initializations'][1]['objective_trace'] = [1.0, -1.0]
    with pytest.raises(ResearchError, match='UNQUALIFIED'):
        _success_row(qualification, {'switching_recipe': value},
            {'fit_K': 2, 'seed': 19, 'replicate_id': 'rep-0'}, result, digest(result))
