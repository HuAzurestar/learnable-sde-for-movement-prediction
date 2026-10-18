"""Complete deterministic input replay controls, not scientific execution."""

import copy
from dataclasses import replace

import pytest

from experiments.pirc28 import input_evidence as verifier
from experiments.pirc28.pipeline import execute_recipe
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc28_runtime import recipe
from tests.test_pirc28_synthetic import fixture_poll


@pytest.fixture(scope='module')
def control():
    value = recipe()
    return value, execute_recipe(value, 2, 19, check_budget=fixture_poll)['fit']['generated_inputs']


def verify(control, generated=None, **kwargs):
    value, original = control
    return verifier.validate_generated_inputs(original if generated is None else generated, value, 19,
        k=2, check_budget=kwargs.pop('check_budget', fixture_poll), **kwargs)


def test_complete_roles_replayed_without_fit_filter_forecast_or_reference(control, monkeypatch):
    for target in ('estimation.switching.fit_switching', 'inference.switching.filter_switching',
                   'inference.switching.forecast_switching',
                   'experiments.pirc28.approximation_checks.check_prefix_approximation'):
        monkeypatch.setattr(target, lambda *a, **k: pytest.fail('input replay used inference'))
    original, seen = verifier.generate_scenario, []
    def replay(source, **kwargs):
        seen.append(source)
        return original(source, **kwargs)
    monkeypatch.setattr(verifier, 'generate_scenario', replay)
    receipt = verify(control)
    assert [source.scenario_id.rsplit('-', 1)[1] for source in seen] == ['train', 'selection', 'test']
    assert all(tuple(source.times) == tuple(control[0]['scenario']['times']) for source in seen)
    assert receipt['generated_inputs_hash'] == digest(control[1])
    assert receipt['full_generated_input_hashes_verified'] is True
    for name in ('source_authority_verified', 'fit_training_execution_verified',
                 'reference_sampling_verified', 'managed_evidence_bound',
                 'qualified_identifiability', 'real_data_authorized'):
        assert receipt[name] is False
    assert receipt == verifier.expected_input_receipt(control[1], control[0], 19, k=2)


@pytest.mark.parametrize('role', range(3))
@pytest.mark.parametrize('field', ['observations_hash', 'truth_hash'])
def test_well_formed_forged_hash_for_every_role_is_recomputed(control, role, field):
    generated = copy.deepcopy(control[1])
    generated[role][field] = '0'*64
    with pytest.raises(ResearchError, match='complete generated source hashes differ'):
        verify(control, generated)


@pytest.mark.parametrize('damage', ['missing-role', 'duplicate-role', 'role-order', 'unknown-field',
    'scenario', 'short-hash', 'huge-hash', 'non-string-hash', 'boolean-seed', 'boolean-k'])
def test_metadata_refuses_before_generation(control, monkeypatch, damage):
    generated = copy.deepcopy(control[1])
    value = control[0]
    seed, k = 19, 2
    if damage == 'missing-role': generated.pop()
    elif damage == 'duplicate-role': generated[1] = copy.deepcopy(generated[0])
    elif damage == 'role-order': generated.reverse()
    elif damage == 'unknown-field': generated[0]['truth'] = []
    elif damage == 'scenario': generated[0]['scenario_hash'] = '0'*64
    elif damage == 'short-hash': generated[2]['truth_hash'] = '0'
    elif damage == 'huge-hash': generated[2]['truth_hash'] = '0'*100000
    elif damage == 'non-string-hash': generated[2]['truth_hash'] = True
    elif damage == 'boolean-seed': seed = True
    else: k = True
    monkeypatch.setattr(verifier, 'generate_scenario', lambda *a, **k: pytest.fail('early generation'))
    with pytest.raises(ResearchError):
        verifier.validate_generated_inputs(generated, value, seed, k=k, check_budget=fixture_poll)


def test_resource_refusal_precedes_generation(control, monkeypatch):
    def refuse(*args, **kwargs):
        raise ResearchError('RESOURCE_PLAN_REJECTED', 'original source envelope refused')
    monkeypatch.setattr(verifier, 'plan_resources', refuse)
    monkeypatch.setattr(verifier, 'generate_scenario', lambda *a, **k: pytest.fail('early generation'))
    with pytest.raises(ResearchError, match='original source envelope refused'):
        verify(control)


def test_truth_or_future_substitution_cannot_hide_behind_valid_prefix(control, monkeypatch):
    original = verifier.generate_scenario
    def altered(source, **kwargs):
        observations, truth, manifest = original(source, **kwargs)
        if source.scenario_id.endswith('-test'):
            observations = replace(observations, values=observations.values.copy())
            observations.values[:, -1, 0] += 1
        return observations, truth, manifest
    monkeypatch.setattr(verifier, 'generate_scenario', altered)
    with pytest.raises(ResearchError, match='complete generated source hashes differ'):
        verify(control)


def test_deadline_and_unknown_errors_propagate_without_receipt(control):
    class Sentinel(Exception):
        pass
    def stop():
        raise Sentinel('original funded deadline')
    with pytest.raises(Sentinel):
        verify(control, check_budget=stop)
    with pytest.raises(ResearchError, match='explicit input replay budget poll required'):
        verify(control, check_budget=None)


def test_receipt_does_not_hash_or_inspect_unrelated_evaluator_fields(control):
    generated = copy.deepcopy(control[1])
    assert verify(control, generated)['generated_inputs_hash'] == digest(generated)
