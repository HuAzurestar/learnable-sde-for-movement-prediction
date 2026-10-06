"""Disposable independent selection controls, never scientific executions."""

import copy

import pytest

from experiments.pirc28 import selection_evidence as verifier
from experiments.pirc28.pipeline import execute_recipe
from infrastructure.research_store import ResearchError, digest
from models.switching import SwitchingError
from tests.test_pirc28_approximation import v2_recipe
from tests.test_pirc28_failures import partial_recipe
from tests.test_pirc28_runtime import recipe
from tests.test_pirc28_synthetic import fixture_poll


@pytest.fixture(scope='module')
def control():
    value = recipe()
    value['initializations']['2'] *= 2
    result = execute_recipe(value, 2, 19, check_budget=fixture_poll)
    return value, result['fit']


@pytest.fixture(scope='module')
def partial():
    value, _ = partial_recipe()
    result = execute_recipe(value, 2, 19, check_budget=fixture_poll)
    return value, result['fit']


def verify(control, fit=None, **kwargs):
    value, original = control
    return verifier.validate_selected_fit(original if fit is None else fit, value, 19, k=2,
        check_budget=kwargs.pop('check_budget', fixture_poll), **kwargs)


def test_all_candidates_scores_and_stable_tie_are_recomputed_without_authority(control):
    model, receipt = verify(control)
    assert model.to_dict() == control[1]['model']
    assert receipt['selected_initialization'] == 'init-0'
    assert len(receipt['candidates']) == 2
    assert receipt['candidates'][0]['score'] == receipt['candidates'][1]['score']
    assert receipt['selection_input_replayed'] is receipt['selection_scores_recomputed'] is True
    assert receipt['selected_model_hash'] == digest(model.to_dict())
    for name in ('fit_training_execution_verified', 'fit_failures_execution_verified',
                 'full_generated_input_hashes_verified', 'reference_sampling_verified',
                 'managed_evidence_bound', 'qualified_identifiability', 'real_data_authorized'):
        assert receipt[name] is False


def test_partial_fit_failure_is_retained_not_retried_or_given_zero_score(partial, monkeypatch):
    monkeypatch.setattr('estimation.switching.fit_switching', lambda *a, **k: pytest.fail('fit failure retried'))
    _, receipt = verify(partial)
    assert receipt['selected_initialization'] == 'init-1'
    assert receipt['candidates'][0] == {'initialization_id': 'init-0', 'state': 'FIT_FAILURE_RETAINED', 'score': None}
    assert receipt['fit_failures_execution_verified'] is False


def test_producer_selection_and_filter_aggregate_helpers_are_not_called(control, monkeypatch):
    for name in ('select_initialization', '_filters'):
        monkeypatch.setattr('estimation.switching.'+name, lambda *a, **k: pytest.fail('producer selection reused'))
    assert verify(control)[1]['selection_scores_recomputed'] is True


def test_only_selection_role_is_generated_and_no_truth_diagnostic_is_read(control, monkeypatch):
    fit = copy.deepcopy(control[1])
    fit['mode_diagnostics'] = object()
    fit['recovery_diagnostics'] = object()
    fit['unavailable_metrics'] = object()
    original = verifier.generate_scenario
    seen = []
    def selection_only(spec, **kwargs):
        assert spec.scenario_id.endswith('-selection')
        assert spec.sample_count == control[0]['selection_paths']
        seen.append(spec)
        return original(spec, **kwargs)
    monkeypatch.setattr(verifier, 'generate_scenario', selection_only)
    assert verify(control, fit)[1] == verify(control)[1]
    assert len(seen) == 2


def test_coherently_forged_scores_and_winner_are_not_numeric_proof(control):
    fit = copy.deepcopy(control[1])
    fit['selection']['candidates'][1]['score'] += 1
    fit['selection']['selected_initialization'] = 'init-1'
    # Metadata-only max-score validation would accept this consistent forgery.
    with pytest.raises(ResearchError, match='arithmetic differs'):
        verify(control, fit)


def test_an_unselected_candidate_score_is_also_checked(control):
    fit = copy.deepcopy(control[1])
    fit['selection']['candidates'][1]['score'] -= 1
    with pytest.raises(ResearchError, match='arithmetic differs'):
        verify(control, fit)


@pytest.mark.parametrize('damage', ['recipe-hash', 'authority', 'role-order', 'scenario-hash', 'source-roster',
    'source-field', 'short-hash', 'missing-init', 'reordered-init', 'unknown-history', 'model-rate-bounds',
    'model-unknown-field', 'model-ragged', 'model-boolean', 'model-huge-integer', 'occupied-boolean',
    'occupancy', 'accuracy', 'objective', 'recovery-level', 'long-trace', 'nonfinite-trace', 'huge-trace',
    'missing-candidate', 'unknown-candidate', 'future-selection', 'selection-version', 'selected-model',
    'inference', 'boolean-inference', 'boolean-collapse', 'huge-score', 'boolean-score'])
def test_unbounded_changed_or_unregistered_metadata_refuses_before_generation(control, monkeypatch, damage):
    fit = copy.deepcopy(control[1])
    history = fit['initializations'][0]
    if damage == 'recipe-hash': fit['recipe_hash'] = '0'*64
    elif damage == 'authority': fit['qualified_identifiability'] = True
    elif damage == 'role-order': fit['generated_inputs'].reverse()
    elif damage == 'scenario-hash': fit['generated_inputs'][0]['scenario_hash'] = '0'*64
    elif damage == 'source-roster': fit['generated_inputs'].pop()
    elif damage == 'source-field': fit['generated_inputs'][0]['truth'] = []
    elif damage == 'short-hash': fit['generated_inputs'][0]['observations_hash'] = '0'
    elif damage == 'missing-init': fit['initializations'].pop()
    elif damage == 'reordered-init': fit['initializations'].reverse()
    elif damage == 'unknown-history': history['truth'] = []
    elif damage == 'model-rate-bounds':
        history['model']['rate_bounds'][1] += 1
        history['model_hash'] = digest(history['model'])
    elif damage == 'model-unknown-field': history['model']['truth'] = []
    elif damage == 'model-ragged': history['model']['a'][0].pop()
    elif damage == 'model-boolean': history['model']['a'][0][0] = True
    elif damage == 'model-huge-integer': history['model']['a'][0][0] = 10**400
    elif damage == 'occupied-boolean': history['occupied_K'] = True
    elif damage == 'occupancy': history['posterior_occupancy'] = [1, 1]
    elif damage == 'accuracy': history['accuracy'] = 'exact'
    elif damage == 'objective': history['objective'] = 'true-joint-likelihood'
    elif damage == 'recovery-level': history['resume_level'] = 'exact'
    elif damage == 'long-trace': history['objective_trace'] = [0]*100
    elif damage == 'nonfinite-trace': history['objective_trace'][0] = float('nan')
    elif damage == 'huge-trace': history['objective_trace'][0] = 10**400
    elif damage == 'missing-candidate': fit['selection']['candidates'].pop()
    elif damage == 'unknown-candidate': fit['selection']['candidates'][0]['truth'] = []
    elif damage == 'future-selection': fit['selection']['split_role'] = 'test'
    elif damage == 'selection-version': fit['selection']['schema_version'] = 'another-version'
    elif damage == 'selected-model': fit['model'] = copy.deepcopy(fit['model']); fit['model']['a'][0][0] += 1
    elif damage == 'inference': history['inference_spec']['max_step_seconds'] /= 2
    elif damage == 'boolean-inference': history['inference_spec']['max_iterations'] = True
    elif damage == 'boolean-collapse': fit['selection']['candidates'][0]['collapsed'] = int(history['collapsed'])
    elif damage == 'huge-score': fit['selection']['candidates'][0]['score'] = 10**400
    else: fit['selection']['candidates'][0]['score'] = True
    monkeypatch.setattr(verifier, 'generate_scenario', lambda *a, **k: pytest.fail('generation before metadata refusal'))
    with pytest.raises((ResearchError, SwitchingError)):
        verify(control, fit)


@pytest.mark.parametrize('field', ['observations_hash', 'truth_hash'])
def test_selection_source_hash_is_replayed_not_just_well_formed(control, field):
    fit = copy.deepcopy(control[1])
    fit['generated_inputs'][1][field] = '0'*64
    if field == 'observations_hash': fit['selection']['selection_hash'] = '0'*64
    with pytest.raises(ResearchError, match='replayed selection source'):
        verify(control, fit)


@pytest.mark.parametrize('damage', ['drop-failure', 'unknown-failure', 'promote-failed-fit', 'zero-failure-score',
    'accepted-shape', 'accepted-trace-quota'])
def test_fit_failure_history_cannot_be_dropped_promoted_or_unbounded(partial, monkeypatch, damage):
    fit = copy.deepcopy(partial[1])
    if damage == 'drop-failure': fit['initializations'].pop(0)
    elif damage == 'unknown-failure': fit['initializations'][0]['private'] = []
    elif damage == 'promote-failed-fit': fit['selection']['selected_initialization'] = 'init-0'
    elif damage == 'zero-failure-score': fit['selection']['candidates'][0]['score'] = 0
    elif damage == 'accepted-shape': fit['initializations'][0]['last_accepted_state'] = {'not': 'registered state'}
    else:
        history = fit['initializations'][0]
        history['last_accepted_state'] = {'schema_version': 'pirc28-em-checkpoint-v1',
            'identity': dict(data_hash='0'*64, spec_hash='0'*64, initial_distribution_hash='0'*64,
                            initialization_id='init-0', initial_model_hash='0'*64),
            'model': partial[0]['initializations']['2'][0], 'objective_trace': [0]*100,
            'rejected_objectives': [], 'completed_iterations': 0, 'complete_reason': None}
        fit['selection']['candidates'][0]['last_accepted_state'] = copy.deepcopy(history['last_accepted_state'])
    monkeypatch.setattr(verifier, 'generate_scenario', lambda *a, **k: pytest.fail('generation before failure-record refusal'))
    with pytest.raises((ResearchError, SwitchingError)):
        verify(partial, fit)


def test_replayed_typed_selection_failure_is_retained_and_cannot_win(control, monkeypatch):
    fit = copy.deepcopy(control[1])
    row = fit['selection']['candidates'][0]
    row.update(schema_version='pirc28-initialization-failure-v1', state='FAILED', stage='selection',
        error_code='INFERENCE_INAPPLICABLE', exception_type='SwitchingError', score=None)
    fit['selection']['selected_initialization'] = 'init-1'
    original = verifier.filter_switching
    called = []
    def bounded(*a, **kw):
        called.append(1)
        if len(called) == 1:
            raise SwitchingError('INFERENCE_INAPPLICABLE', 'explicit disposable numeric cutoff')
        return original(*a, **kw)
    monkeypatch.setattr(verifier, 'filter_switching', bounded)
    _, receipt = verify(control, fit)
    assert receipt['selected_initialization'] == 'init-1'
    assert receipt['candidates'][0]['state'] == 'FAILED'
    assert receipt['candidates'][0]['score'] is None
    assert len(called) == 2


def test_declared_selection_failure_cannot_hide_a_successful_candidate(control):
    fit = copy.deepcopy(control[1])
    fit['selection']['candidates'][0].update(schema_version='pirc28-initialization-failure-v1',
        state='FAILED', stage='selection', error_code='INFERENCE_INAPPLICABLE', exception_type='SwitchingError', score=None)
    fit['selection']['selected_initialization'] = 'init-1'
    with pytest.raises(ResearchError, match='cannot be censored'):
        verify(control, fit)


@pytest.mark.parametrize('error', [ResearchError('TIMEOUT', 'explicit cutoff'),
    ResearchError('BUDGET_EXHAUSTED', 'explicit fuse'), SwitchingError('IDENTITY_CONFLICT', 'explicit conflict'),
    SwitchingError('UNAUTHORIZED_DATA', 'explicit no grant'), RuntimeError('unexpected implementation failure')])
def test_authority_and_unknown_errors_propagate_without_numeric_downgrade(control, monkeypatch, error):
    def stop(*a, **kw): raise error
    monkeypatch.setattr(verifier, 'filter_switching', stop)
    with pytest.raises(type(error)) as caught:
        verify(control)
    assert caught.value is error


def test_budget_poll_is_required_and_cutoff_propagates_before_generation(control, monkeypatch):
    with pytest.raises(ResearchError, match='budget poll'):
        verify(control, check_budget=None)
    calls = []
    def poll():
        calls.append(1)
        if len(calls) == 13: raise ResearchError('TIMEOUT', 'explicit verifier cutoff')
    monkeypatch.setattr(verifier, 'generate_scenario', lambda *a, **k: pytest.fail('generation after cutoff'))
    with pytest.raises(ResearchError, match='TIMEOUT'):
        verify(control, check_budget=poll)
    assert len(calls) == 13


@pytest.mark.parametrize('seed,k', [(True, 2), (-1, 2), (2**64, 2), (19, True), (19, 0), (19, 3)])
def test_seed_and_arm_identity_are_bounded_nonboolean(control, seed, k):
    with pytest.raises(ResearchError):
        verifier.validate_selected_fit(control[1], control[0], seed, k=k, check_budget=fixture_poll)


def test_combined_over_cap_refuses_before_selection_generation(control, monkeypatch):
    value = copy.deepcopy(control[0]); value['inference']['max_substeps'] = 4000
    monkeypatch.setattr(verifier, 'generate_scenario', lambda *a, **k: pytest.fail('generation before combined-cap refusal'))
    with pytest.raises(ResearchError, match='combined tensors'):
        verifier.validate_selected_fit(control[1], value, 19, k=2, check_budget=fixture_poll)


def test_k1_choice_replay_keeps_single_mode_boundary_without_science():
    value = recipe()
    fit = execute_recipe(value, 1, 19, check_budget=fixture_poll)['fit']
    model, receipt = verifier.validate_selected_fit(fit, value, 19, k=1, check_budget=fixture_poll)
    assert model.n_modes == 1 and receipt['selected_initialization'] == 'init-0'
    assert receipt['qualified_identifiability'] is False


def test_prospective_v2_disclosed_model_choice_does_not_lift_generation_stop():
    # Explicit engineering projection, not a claim that V2 fitting executed.
    value = v2_recipe(); value['train_paths'] = 1
    legacy = copy.deepcopy(value); legacy['schema_version'] = 'pirc28-synthetic-recipe-v1'; legacy.pop('approximation_spec')
    fit = execute_recipe(legacy, 2, 19, check_budget=fixture_poll)['fit']
    fit['recipe_hash'] = digest({'recipe': value, 'K': 2, 'seed': 19})
    _, receipt = verifier.validate_selected_fit(fit, value, 19, k=2, check_budget=fixture_poll)
    assert receipt['selection_scores_recomputed'] is True
    assert receipt['fit_training_execution_verified'] is receipt['managed_evidence_bound'] is False
    with pytest.raises(ResearchError, match='NEEDS_IMPLEMENTATION'):
        execute_recipe(value, 2, 19, check_budget=fixture_poll)
