"""Independent diagnostic/replayed-input controls, never managed authority."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from experiments.pirc28 import approximation_evidence as verifier
from experiments.pirc28.approximation_checks import check_prefix_approximation, observation_prefix
from experiments.pirc28.approximation import ApproximationSpec
from experiments.pirc28.pipeline import validate_recipe
from experiments.pirc28.synthetic import generate_scenario
from inference.switching import InferenceSpec
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc28_approximation import v2_recipe
from tests.test_pirc28_approximation_checks import observations, run, settings
from tests.test_pirc28_inference import initial
from tests.test_pirc28_synthetic import fixture_poll


@pytest.fixture(scope='module')
def control():
    batch, model = observations(paths=2)
    return batch, model, run(batch, model)


def verify(control, report=None, **kwargs):
    batch, model, original = control
    return verifier.validate_prefix_numbers(original if report is None else report, batch, model,
        *initial(model.n_modes), InferenceSpec(), settings(), origin_index=2, seed=123,
        check_budget=kwargs.pop('check_budget', fixture_poll), **kwargs)


def test_honest_all_path_k1_report_is_consistent_but_not_authorized(control):
    result = verify(control)
    assert result['state'] == 'NUMERICAL_CHECKS_PASSED'
    assert result['report_hash'] == digest(control[2])
    assert result['input_identity_bound'] is result['numerical_diagnostics_consistent'] is True
    for key in ('reference_sampling_verified', 'selected_fit_selection_verified',
                'managed_evidence_bound', 'qualified_identifiability', 'real_data_authorized'):
        assert result[key] is False


def test_k2_actual_fixed_batches_keep_inapplicable_disposition_and_all_paths():
    batch, model = observations(k=2, paths=2, rate=4.0)
    report = run(batch, model)
    result = verify((batch, model, report))
    assert result['state'] == report['state']
    assert report['state'] == 'INFERENCE_INAPPLICABLE'
    assert sum(path['reference_jump_count'] for path in report['paths']) > 0
    assert result['qualified_identifiability'] is False


def test_verifier_never_calls_producer_aggregate_sampler_or_boundary_helpers(control, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('producer reference helper used by independent checker')
    for name in ('reference_summary', '_errors', '_known_mode_checks', '_reference_batch'):
        monkeypatch.setattr('experiments.pirc28.approximation_checks.'+name, forbidden)
    assert verify(control)['numerical_diagnostics_consistent'] is True


def test_future_nan_values_and_times_are_not_read(control):
    batch, model, report = control
    values, times = batch.values.copy(), batch.times.copy()
    values[:, 3:] = np.nan
    times[3:] = np.nan
    changed = replace(batch, values=values, times=times)
    assert verify((changed, model, report)) == verify(control)


@pytest.mark.parametrize('damage', ['prefix', 'model', 'prior', 'inference', 'spec'])
def test_caller_owned_expected_inputs_cannot_be_replaced_by_report_claims(control, monkeypatch, damage):
    batch, model, report = control
    prior, inference, spec = initial(1), InferenceSpec(), settings()
    if damage == 'prefix':
        values = batch.values.copy(); values[0, 1, 0] += 1
        batch = replace(batch, values=values)
    elif damage == 'model':
        from models.switching import SwitchingSDEModel
        value = model.to_dict(); value['a'][0][0] += 1
        model = SwitchingSDEModel.from_dict(value)
    elif damage == 'prior': prior[1][0, 0] += 1
    elif damage == 'inference': inference = replace(inference, max_rate_step=0.025)
    else:
        value = spec.to_dict(); value['step_halving']['thresholds']['position_mean_max_abs_m'] /= 2
        spec = ApproximationSpec.from_dict(value)
    monkeypatch.setattr(verifier, '_filter', lambda *a, **k: pytest.fail('filter before expected-input binding refusal'))
    with pytest.raises(ResearchError):
        verifier.validate_prefix_numbers(report, batch, model, *prior, inference, spec,
            origin_index=2, seed=123, check_budget=fixture_poll)


def test_unequal_initial_gaussian_mixture_is_independently_bound():
    batch, model = observations(k=2)
    p, m, c = initial(2)
    p[:] = [0.8, 0.2]; m[1, :2] = [1.0, -0.5]; c[1] *= 2
    inference, spec = InferenceSpec(), settings()
    report = check_prefix_approximation(batch, model, p, m, c, inference, spec,
        origin_index=2, seed=123, check_budget=fixture_poll)
    result = verifier.validate_prefix_numbers(report, batch, model, p, m, c, inference, spec,
        origin_index=2, seed=123, check_budget=fixture_poll)
    assert result['numerical_diagnostics_consistent'] is True
    assert result['reference_sampling_verified'] is False


def test_nonpositive_likelihood_interval_is_unknown_not_zero_and_not_execution_proof(control):
    # Deliberately synthetic batch bodies: consistency cannot authenticate their
    # execution, which is why every returned authority/sampling flag stays false.
    batch, model, report = control
    path = copy.deepcopy(report['paths'][0])
    for index, value in enumerate(path['reference_batch_summaries']):
        value['log_likelihood_trace'][-1] = [100.0, 0.0, -100.0, -100.0][index]
    prefix = observation_prefix(batch, 2)
    p, m, c = initial(1)
    initial_summary = verifier._filter(prefix, model, p, m, c, InferenceSpec(), 0, fixture_poll)
    initial_summary['initial_log_likelihood'] = path['reference_batch_summaries'][0]['log_likelihood_trace'][0]
    summary, widths, _, _ = verifier._batches(path['reference_batch_summaries'],
        settings().to_dict()['event_driven_reference'], 3, 1, 0, 123, initial_summary, fixture_poll)
    assert np.isclose(summary['log_likelihood'], 100-np.log(4))
    assert not np.isclose(summary['log_likelihood'], -25)
    assert widths['log_likelihood_per_observation_abs'] is None
    with pytest.raises(ResearchError, match='unavailable'):
        verifier._close(0, widths['log_likelihood_per_observation_abs'])


@pytest.mark.parametrize('field', ['spec_hash', 'model_hash', 'observation_prefix_hash',
    'initial_state_hash', 'inference_hash', 'reference_seed', 'origin_index', 'reference_interval_method',
    'reference_initial_conditioning', 'reference_resampling', 'accuracy', 'comparison',
    'managed_evidence_bound', 'qualified_identifiability', 'real_data_authorized'])
def test_expected_identity_and_no_authority_are_enforced_before_filtering(control, monkeypatch, field):
    report = copy.deepcopy(control[2])
    old = report[field]
    report[field] = not old if type(old) is bool else old+1 if type(old) is int else old+'-forged'
    monkeypatch.setattr(verifier, '_filter', lambda *a, **k: pytest.fail('filter before identity refusal'))
    with pytest.raises(ResearchError):
        verify(control, report)


@pytest.mark.parametrize('damage', ['unknown-field', 'missing-path', 'reordered-paths', 'path-index',
    'prefix-hash', 'missing-batch', 'reordered-batches', 'batch-index', 'seed-entropy', 'boolean-entropy',
    'ragged-array', 'nan-array', 'boolean-array', 'huge-integer', 'negative-probability', 'initial-mean',
    'initial-likelihood', 'coarse-mean', 'fine-mean', 'reference-mean-rehashed', 'reference-hash',
    'halving-error', 'reference-error', 'mc-width', 'minimum-ess', 'batch-ess', 'batch-jumps',
    'total-jumps', 'batch-count', 'boundary-deleted', 'boundary-boolean', 'boundary-error',
    'path-state', 'reasons', 'whole-state', 'false-as-zero', 'resource-integer-as-boolean',
    'resource-nested-object'])
def test_missing_tampered_or_untyped_diagnostics_are_refused(control, damage):
    report = copy.deepcopy(control[2])
    path = report['paths'][0]
    batch = path['reference_batch_summaries'][0]
    if damage == 'unknown-field': report['truth'] = []
    elif damage == 'missing-path': report['paths'].pop()
    elif damage == 'reordered-paths': report['paths'].reverse()
    elif damage == 'path-index': path['path_index'] = True
    elif damage == 'prefix-hash': path['prefix_hash'] = '0'*64
    elif damage == 'missing-batch': path['reference_batch_summaries'].pop()
    elif damage == 'reordered-batches': path['reference_batch_summaries'].reverse()
    elif damage == 'batch-index': batch['batch_index'] = True
    elif damage == 'seed-entropy': batch['seed_entropy'][0] += 1
    elif damage == 'boolean-entropy': batch['seed_entropy'][1] = False
    elif damage == 'ragged-array': batch['means'][1].pop()
    elif damage == 'nan-array': batch['means'][1][0] = float('nan')
    elif damage == 'boolean-array': batch['means'][1][0] = True
    elif damage == 'huge-integer': batch['means'][1][0] = 10**400
    elif damage == 'negative-probability': batch['probabilities'][1][0] = -1
    elif damage == 'initial-mean': batch['means'][0][0] += 1
    elif damage == 'initial-likelihood': batch['log_likelihood_trace'][0] += 1
    elif damage == 'coarse-mean': path['coarse_summary']['means'][1][0] += 1
    elif damage == 'fine-mean': path['fine_summary']['means'][1][0] += 1
    elif damage == 'reference-mean-rehashed':
        path['reference_summary']['means'][1][0] += 1
        path['reference_summary_hash'] = digest(path['reference_summary'])
    elif damage == 'reference-hash': path['reference_summary_hash'] = '0'*64
    elif damage == 'halving-error': path['step_halving_errors']['position_mean_max_abs_m'] += 1
    elif damage == 'reference-error': path['reference_errors']['position_mean_max_abs_m'] += 1
    elif damage == 'mc-width': path['reference_half_widths']['position_mean_max_abs_m'] += 1
    elif damage == 'minimum-ess': path['reference_minimum_ess_fraction'] = 0
    elif damage == 'batch-ess': batch['minimum_ess_fraction'] = 0
    elif damage == 'batch-jumps': batch['jump_count'] = True
    elif damage == 'total-jumps': path['reference_jump_count'] += 1
    elif damage == 'batch-count': path['reference_batches'] += 1
    elif damage == 'boundary-deleted': path['known_mode_boundary'].pop()
    elif damage == 'boundary-boolean': path['known_mode_boundary'][0]['passed'] = 1
    elif damage == 'boundary-error': path['known_mode_boundary'][0]['covariance_max_abs'] += 1
    elif damage == 'path-state': path['state'] = 'INFERENCE_INAPPLICABLE'
    elif damage == 'reasons': path['reasons'] = ['REFERENCE_MC_PRECISION_FAILED']
    elif damage == 'whole-state': report['state'] = 'INFERENCE_INAPPLICABLE'
    elif damage == 'false-as-zero': report['managed_evidence_bound'] = 0
    elif damage == 'resource-integer-as-boolean': report['resource_plan']['independent_batches'] = True
    else: report['resource_plan']['tensor_elements'] = {'forged': [1]*1000}
    with pytest.raises(ResearchError):
        verify(control, report)


def test_timeout_propagates_without_partial_receipt_or_retry(control):
    calls = []
    def poll():
        calls.append(1)
        if len(calls) == 13:
            raise ResearchError('TIMEOUT', 'explicit disposable verifier cutoff')
    with pytest.raises(ResearchError, match='TIMEOUT'):
        verify(control, check_budget=poll)
    assert len(calls) == 13


def test_report_byte_quota_is_enforced_without_receipt(control, monkeypatch):
    monkeypatch.setattr(verifier, 'MAXIMUM_REPORT_BYTES', 10)
    with pytest.raises(ResearchError, match='byte quota'):
        verify(control)


def test_replayed_observation_prefix_equals_full_generation_without_future_generation(monkeypatch):
    value = v2_recipe()
    scenario, _, _, _, _ = validate_recipe(value, 2)
    split = np.random.default_rng(123).integers(0, 2**63, size=3, dtype=np.int64)
    full = replace(scenario, scenario_id=scenario.scenario_id+'-test', seed=int(split[2]), sample_count=value['test_paths'])
    expected, _, _ = generate_scenario(full, check_budget=fixture_poll)
    expected = observation_prefix(expected, value['forecast_origin_index'])
    seen = []
    def prefix_only(spec, **kwargs):
        seen.append(spec)
        assert tuple(spec.times) == tuple(full.times[:value['forecast_origin_index']+1])
        return generate_scenario(spec, **kwargs)
    monkeypatch.setattr(verifier, 'generate_scenario', prefix_only)
    actual, receipt = verifier.replay_test_prefix(value, 123, check_budget=fixture_poll)
    assert actual.to_dict() == expected.to_dict()
    assert len(seen) == 1
    assert receipt['test_split_seed'] == int(split[2])
    assert receipt['test_scenario_hash'] == full.scenario_hash
    assert receipt['observation_prefix_hash'] == digest(expected.to_dict())
    assert receipt['reference_seed'] == verifier.reference_seed(value, 123)
    assert receipt['prefix_inputs_replayed'] is True
    for name in ('full_generated_input_hashes_verified', 'source_authority_verified', 'managed_evidence_bound'):
        assert receipt[name] is False


def test_reference_seed_is_cell_recipe_domain_bound_not_selected_model_or_retry():
    value = v2_recipe()
    identity = digest({'recipe': value, 'K': 2, 'seed': 123})
    expected = int(digest([identity, 'pirc28-prefix-reference-v1'])[:16], 16)
    assert verifier.reference_seed(value, 123) == expected
    assert verifier.reference_seed(copy.deepcopy(value), 123) == expected
    assert verifier.reference_seed(value, 124) != expected
    changed = copy.deepcopy(value)
    changed['prior_mean'][0] += 1
    assert verifier.reference_seed(changed, 123) != expected


@pytest.mark.parametrize('seed', [-1, 2**64, True])
def test_reference_seed_refuses_unsigned_identity_violations(seed):
    with pytest.raises(ResearchError):
        verifier.reference_seed(v2_recipe(), seed)


def test_replay_refuses_additive_over_cap_before_generating(monkeypatch):
    value = v2_recipe()
    value['train_paths'] = 3
    monkeypatch.setattr(verifier, 'generate_scenario', lambda *a, **k: pytest.fail('generation before combined-cap refusal'))
    with pytest.raises(ResearchError, match='combined tensors'):
        verifier.replay_test_prefix(value, 123, check_budget=fixture_poll)


def test_replay_and_verifier_require_explicit_budget_poll(control):
    with pytest.raises(ResearchError, match='budget poll'):
        verifier.replay_test_prefix(v2_recipe(), 123, check_budget=None)
    with pytest.raises(ResearchError, match='budget poll'):
        verify(control, check_budget=None)


def test_replay_budget_cutoff_precedes_generation_and_propagates(monkeypatch):
    monkeypatch.setattr(verifier, 'generate_scenario', lambda *a, **k: pytest.fail('generation after cutoff'))
    def poll():
        raise ResearchError('BUDGET_EXHAUSTED', 'explicit disposable original-arm cutoff')
    with pytest.raises(ResearchError, match='BUDGET_EXHAUSTED'):
        verifier.replay_test_prefix(v2_recipe(), 123, check_budget=poll)


def test_legacy_replay_is_not_silently_upgraded_to_scientific_contract():
    value = v2_recipe()
    value['schema_version'] = 'pirc28-synthetic-recipe-v1'
    value.pop('approximation_spec')
    with pytest.raises(ResearchError, match='V2'):
        verifier.replay_test_prefix(value, 123, check_budget=fixture_poll)
