"""Disposable bounded numerical controls, not managed scientific qualification."""

from dataclasses import replace
import math

import numpy as np
import pytest

from experiments.pirc28.approximation import ApproximationSpec
from experiments.pirc28.approximation_checks import (MC_METHOD, _event_predict, _known_mode_checks, _reference_batch,
    check_prefix_approximation, numerical_plan, observation_prefix, reference_design_preflight,
    reference_summary, schedule_plan, student_t_critical)
from experiments.pirc28.plan import load_plan, REFERENCE_PLAN
from experiments.pirc28.synthetic import generate_scenario
from inference.switching import InferenceSpec
from infrastructure.research_store import ResearchError, digest, encode
from models.switching import SwitchingError, SwitchingSDEModel
from tests.test_pirc28_inference import initial
from tests.test_pirc28_synthetic import fixture_poll, model, scenario


def settings():
    value = load_plan(REFERENCE_PLAN)['approximation_spec']
    # Explicit small disposable reference, NEVER a production-plan default.
    # These legacy test models permit fitted rates to 100, not production 5.
    value['step_halving']['maximum_substeps'] = 4000
    value['event_driven_reference']['particles'] = 64
    value['event_driven_reference']['independent_batches'] = 4
    value['event_driven_reference']['maximum_jumps_per_particle_interval'] = 256
    return ApproximationSpec.from_dict(value)


def observations(k=1, paths=1, rate=0.4):
    source = replace(scenario(k=k, paths=paths, rate=rate), times=(0.0, 0.5, 1.0, 2.0))
    batch, _, _ = generate_scenario(source, check_budget=fixture_poll)
    return batch, SwitchingSDEModel.from_dict(source.model)


def run(batch, fitted, **kwargs):
    return check_prefix_approximation(batch, fitted, *initial(fitted.n_modes), InferenceSpec(), settings(),
        origin_index=kwargs.pop('origin_index', 2), seed=kwargs.pop('seed', 123),
        check_budget=kwargs.pop('check_budget', fixture_poll), **kwargs)


@pytest.fixture(scope='module')
def k1_control():
    batch, fitted = observations(paths=2)
    return batch, fitted, run(batch, fitted)


def test_k1_affine_boundary_reference_and_halving_pass_without_scientific_claim(k1_control):
    _, _, result = k1_control
    assert result['state'] == 'NUMERICAL_CHECKS_PASSED'
    assert len(result['paths']) == 2
    for path in result['paths']:
        assert path['reasons'] == [] and path['reference_batches'] == 4
        assert path['reference_jump_count'] == 0
        assert np.isclose(path['reference_minimum_ess_fraction'], 1)
        assert max(path['step_halving_errors'].values()) < 1e-10
        assert max(path['reference_errors'].values()) < 1e-10
        assert path['known_mode_boundary'][0]['passed'] is True
        assert path['known_mode_boundary'][0]['covariance_max_abs'] < 1e-10
    assert result['managed_evidence_bound'] is result['qualified_identifiability'] is result['real_data_authorized'] is False
    assert result['resource_plan']['resource_admitted'] is False
    assert result['reference_interval_method'] == MC_METHOD
    assert 'not-exact' in result['accuracy']
    encode(result)


def test_future_values_times_and_truth_cannot_change_any_prefix_result(k1_control):
    batch, fitted, original = k1_control
    values, times = batch.values.copy(), batch.times.copy()
    values[:, 3:] = np.nan
    times[3:] = np.nan
    changed = replace(batch, values=values, times=times)
    assert run(changed, fitted) == original
    assert not np.shares_memory(observation_prefix(batch, 2).values, batch.values)


@pytest.mark.parametrize('damage', ['origin-zero', 'origin-boolean', 'origin-last', 'prefix-nan', 'prefix-time',
    'H', 'R', 'truth-type', 'path-quota'])
def test_prefix_contract_refuses_invalid_or_vacuous_inputs(k1_control, damage):
    batch = k1_control[0]
    origin = 2
    if damage == 'origin-zero': origin = 0
    elif damage == 'origin-boolean': origin = True
    elif damage == 'origin-last': origin = 4
    elif damage == 'prefix-nan':
        values = batch.values.copy(); values[0, 0, 0] = np.nan
        batch = replace(batch, values=values)
    elif damage == 'prefix-time': batch = replace(batch, times=np.array([0, 0, 1, 2]))
    elif damage == 'H': batch = replace(batch, H=np.eye(3))
    elif damage == 'R': batch = replace(batch, R=-np.eye(2))
    elif damage == 'truth-type': batch = {'states': [], 'modes': []}
    else: batch = replace(batch, values=np.zeros((65, 4, 2)))
    with pytest.raises(SwitchingError):
        observation_prefix(batch, origin)


@pytest.mark.parametrize('seed', [-1, 2**64, True])
def test_invalid_reference_seed_is_refused_before_work(k1_control, seed):
    with pytest.raises(SwitchingError, match='unsigned'):
        run(k1_control[0], k1_control[1], seed=seed)


def test_budget_timeout_propagates_without_partial_pass_or_retry(k1_control):
    calls = []
    def timeout():
        calls.append(1)
        if len(calls) > 12:
            raise ResearchError('TIMEOUT', 'explicit fixture supervisor cutoff')
    with pytest.raises(ResearchError, match='TIMEOUT'):
        run(k1_control[0], k1_control[1], check_budget=timeout)
    assert len(calls) == 13


@pytest.mark.parametrize('degrees,confidence,expected', [(3, 0.95, 3.182446305284263),
    (3, 0.99, 5.840909309733352), (9, 0.95, 2.262157162854099), (31, 0.95, 2.039513446396408)])
def test_integer_df_student_quantile_matches_standard_reference_values(degrees, confidence, expected):
    assert math.isclose(student_t_critical(degrees, confidence, check_budget=fixture_poll), expected, rel_tol=1e-10)


def fake_batch(log_likelihood=0.0, probability=0.5):
    return {'probabilities': np.tile([probability, 1-probability], (3, 1)),
        'means': np.zeros((3, 4)), 'log_likelihood': np.full(3, log_likelihood),
        'minimum_ess_fraction': 1.0, 'jump_count': 0}


def test_likelihood_aggregation_is_log_mean_not_mean_log_and_refuses_uncertain_lower_bound():
    batches = [fake_batch(value) for value in (-100, -100, 0, 0)]
    summary = reference_summary(batches, 0.95, check_budget=fixture_poll)
    assert abs(summary['log_likelihood']+math.log(2)) < 1e-10
    assert summary['log_likelihood'] != np.mean([-100, -100, 0, 0])
    assert summary['half_widths']['log_likelihood_per_observation_abs'] is None
    assert summary['interval_method'] == MC_METHOD


@pytest.mark.parametrize('damage', ['missing-batch', 'nan', 'probability', 'empty', 'ess', 'jumps'])
def test_reference_summary_cannot_delete_failure_or_accept_invalid_batch(damage):
    batches = [fake_batch() for _ in range(4)]
    if damage == 'missing-batch': batches.pop()
    elif damage == 'nan': batches[0]['log_likelihood'][0] = np.nan
    elif damage == 'probability': batches[0]['probabilities'][0, 0] = -0.5
    elif damage == 'empty':
        for value in batches:
            value.update(probabilities=np.zeros((0, 2)), means=np.zeros((0, 4)), log_likelihood=np.zeros(0))
    elif damage == 'ess': batches[0]['minimum_ess_fraction'] = 0
    else: batches[0]['jump_count'] = True
    with pytest.raises(SwitchingError):
        reference_summary(batches, 0.95, check_budget=fixture_poll)


def test_high_reference_mc_uncertainty_remains_failed_without_extra_batches(monkeypatch):
    batch, fitted = observations(k=2, rate=0)
    calls = []
    def reference(*args):
        index = args[7]
        calls.append(index)
        return fake_batch(probability=[0.1, 0.3, 0.7, 0.9][index])
    monkeypatch.setattr('experiments.pirc28.approximation_checks._reference_batch', reference)
    result = run(batch, fitted)
    assert calls == list(range(4))
    assert result['state'] == 'INFERENCE_INAPPLICABLE'
    assert 'REFERENCE_MC_PRECISION_FAILED' in result['paths'][0]['reasons']
    assert result['qualified_identifiability'] is False


def test_resource_limits_refuse_before_filter_or_reference(monkeypatch):
    batch, fitted = observations(k=2)
    value = settings().to_dict()
    value['step_halving']['maximum_substeps'] = 100_000
    value['event_driven_reference'].update(particles=4096, independent_batches=32, maximum_jumps_per_particle_interval=10_000)
    monkeypatch.setattr('experiments.pirc28.approximation_checks.filter_switching', lambda *a, **k: pytest.fail('work before resource refusal'))
    with pytest.raises(SwitchingError, match='core quota'):
        check_prefix_approximation(batch, fitted, *initial(2), InferenceSpec(), ApproximationSpec.from_dict(value),
            origin_index=2, seed=123, check_budget=fixture_poll)


def test_event_driven_reference_counts_many_jumps_not_just_endpoint_changes():
    batch, fitted = observations(k=2, rate=8)
    prefix = observation_prefix(batch, 2)
    result = _reference_batch(prefix, fitted, *initial(2), settings().to_dict()['event_driven_reference'],
                             456, 0, 0, fixture_poll)
    assert result['jump_count'] > 64  # Multiple real events per particle prefix.
    assert np.allclose(result['probabilities'].sum(axis=1), 1)
    assert np.isfinite(result['means']).all()


def test_reference_jump_limit_is_failure_not_a_truncated_success():
    class AlwaysJump:
        def exponential(self, scale): return 0.01
        def choice(self, k, p): return int(np.argmax(p))
    fitted = model(rate=8)
    kernels = [[value.numpy() for value in fitted.conditional_affine_transition(mode, 1)] for mode in range(2)]
    with pytest.raises(SwitchingError, match='jump quota exceeded'):
        _event_predict(fitted, 0, np.zeros(4), np.eye(4), 1, AlwaysJump(), 1, kernels, fixture_poll)


def test_absorbing_label_permutation_retains_same_continuous_reference():
    batch, fitted = observations(k=2, rate=0)
    prefix = observation_prefix(batch, 2)
    _, means, covariances = initial(2)
    settings_value = settings().to_dict()['event_driven_reference']
    left = _reference_batch(prefix, fitted, np.array([1.0, 0]), means, covariances, settings_value, 123, 0, 0, fixture_poll)
    right = _reference_batch(prefix, fitted.permuted([1, 0]), np.array([0, 1.0]), means[::-1], covariances[::-1], settings_value, 123, 0, 0, fixture_poll)
    assert np.array_equal(left['probabilities'], right['probabilities'][:, ::-1])
    assert np.allclose(left['means'], right['means'])
    assert np.allclose(left['log_likelihood'], right['log_likelihood'])


def test_reference_seed_and_every_prefix_path_are_bound(k1_control):
    batch, fitted, result = k1_control
    assert result['reference_seed'] == 123
    assert result['observation_prefix_hash'] == digest(observation_prefix(batch, 2).to_dict())
    assert result['model_hash'] == digest(fitted.to_dict())
    assert [row['path_index'] for row in result['paths']] == [0, 1]
    assert result['paths'][0]['prefix_hash'] != result['paths'][1]['prefix_hash']
    for path in result['paths']:
        assert path['reference_summary_hash'] == digest(path['reference_summary'])
        assert [value['batch_index'] for value in path['reference_batch_summaries']] == [0, 1, 2, 3]
        assert all(value['seed_entropy'] == [123, path['path_index'], value['batch_index']]
                   for value in path['reference_batch_summaries'])
    assert numerical_plan(settings(), InferenceSpec(), observation_prefix(batch, 2), fitted)['resource_admitted'] is False


def test_full_k2_numerical_check_retains_all_actual_reference_and_boundary_diagnostics():
    batch, fitted = observations(k=2, rate=8)
    result = run(batch, fitted)
    path = result['paths'][0]
    assert path['reference_jump_count'] > 64 and path['reference_batches'] == 4
    assert len(path['known_mode_boundary']) == 2
    assert all(value['passed'] for value in path['known_mode_boundary'])
    assert all(math.isfinite(value) for value in path['step_halving_errors'].values())
    assert all(math.isfinite(value) for value in path['reference_errors'].values())
    # A finite reference is not automatically sufficiently precise or applicable.
    assert result['state'] == ('INFERENCE_INAPPLICABLE' if path['reasons'] else 'NUMERICAL_CHECKS_PASSED')
    assert result['qualified_identifiability'] is False


def test_identical_dynamics_multi_jump_reference_matches_k1_continuous_law():
    batch, fitted = observations(k=2, rate=8)
    arrays = {name: getattr(fitted, name).numpy().copy() for name in ('A', 'B', 'a', 'L')}
    for value in arrays.values(): value[1] = value[0]
    identical = SwitchingSDEModel(fitted.Q.numpy(), **arrays, rate_bounds=fitted.rate_bounds)
    one = SwitchingSDEModel([[0]], **{name: value[:1] for name, value in arrays.items()}, rate_bounds=fitted.rate_bounds)
    prefix = observation_prefix(batch, 2)
    settings_value = settings().to_dict()['event_driven_reference']
    switching = _reference_batch(prefix, identical, *initial(2), settings_value, 123, 0, 0, fixture_poll)
    constant = _reference_batch(prefix, one, *initial(1), settings_value, 123, 0, 0, fixture_poll)
    assert switching['jump_count'] > 64
    assert np.allclose(switching['means'], constant['means'], atol=1e-10, rtol=0)
    assert np.allclose(switching['log_likelihood'], constant['log_likelihood'], atol=1e-10, rtol=0)


def test_known_mode_control_independently_detects_covariance_corruption(monkeypatch):
    import experiments.pirc28.approximation_checks as core
    original = core.filter_switching
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result.posteriors = tuple(replace(p, covariances=p.covariances+0.01*np.eye(4)) for p in result.posteriors)
        return result
    monkeypatch.setattr(core, 'filter_switching', corrupt)
    batch, fitted = observations()
    _, means, covariances = initial(1)
    checks = _known_mode_checks(observation_prefix(batch, 2), fitted, means, covariances, InferenceSpec(),
        settings().to_dict()['known_mode_boundary'], 0, fixture_poll)
    assert checks[0]['passed'] is False and checks[0]['covariance_max_abs'] >= 0.009


@pytest.mark.parametrize('seed,particles', [(12, 64), (901, 128)])
def test_initial_mixture_conditioning_is_exact_with_unequal_priors_and_covariances(seed, particles):
    batch, fitted = observations(k=2, rate=0)
    values = np.full((1, 4, 1), 0.75)
    batch = replace(batch, values=values, H=np.eye(4)[:1], R=np.array([[0.5]]))
    means = np.zeros((2, 4)); means[1, 0] = 2.0
    covariances = np.array([np.eye(4), 2*np.eye(4)])
    prior = np.array([0.3, 0.7])
    innovation = np.array([1.5, 2.5])
    residual = 0.75-means[:, 0]
    weighted_density = prior*np.exp(-residual**2/(2*innovation))/np.sqrt(2*math.pi*innovation)
    expected_p = weighted_density/weighted_density.sum()
    expected_mean = means.copy()
    expected_mean[:, 0] += np.array([1.0, 2.0])/innovation*residual
    config = settings().to_dict()['event_driven_reference']; config['particles'] = particles
    result = _reference_batch(observation_prefix(batch, 2), fitted, prior, means, covariances,
                              config, seed, 0, 0, fixture_poll)
    assert np.allclose(result['probabilities'][0], expected_p, atol=1e-14, rtol=0)
    assert np.allclose(result['means'][0], expected_p@expected_mean, atol=1e-14, rtol=0)
    assert math.isclose(result['log_likelihood'][0], math.log(weighted_density.sum()), abs_tol=1e-14)


def test_initial_zero_probability_singular_component_is_not_conditioned():
    batch, fitted = observations(k=2, rate=0)
    batch = replace(batch, H=np.eye(4)[:1], R=np.zeros((1, 1)), values=batch.values[:, :, :1])
    means = np.zeros((2, 4)); covariances = np.array([np.zeros((4, 4)), np.eye(4)])
    result = _reference_batch(observation_prefix(batch, 2), fitted, np.array([0.0, 1.0]), means, covariances,
                              settings().to_dict()['event_driven_reference'], 123, 0, 0, fixture_poll)
    assert np.array_equal(result['probabilities'][0], [0, 1])
    assert np.isfinite(result['log_likelihood']).all()


def test_exact_initial_posterior_is_sampled_once_without_redundant_resampling(monkeypatch):
    batch, fitted = observations(k=2, rate=0)
    original = np.random.default_rng
    calls = {role: [] for role in range(3)}
    class Recorded:
        def __init__(self, entropy):
            self.role = entropy.entropy[-1]; self.rng = original(entropy)
        def choice(self, a, size=None, p=None):
            calls[self.role].append(np.array(p))
            return self.rng.choice(a, size=size, p=p)
        def exponential(self, scale): return self.rng.exponential(scale)
    monkeypatch.setattr(np.random, 'default_rng', Recorded)
    prior, means, covariances = initial(2)
    result = _reference_batch(observation_prefix(batch, 2), fitted, prior, means, covariances,
                              settings().to_dict()['event_driven_reference'], 123, 0, 0, fixture_poll)
    assert len(calls[0]) == 1 and np.array_equal(calls[0][0], result['probabilities'][0])
    assert len(calls[2]) == 1  # Three observations: only propagation toward the third resamples.


def test_metadata_and_observation_core_work_plans_are_identical():
    batch, fitted = observations(k=2)
    prefix = observation_prefix(batch, 2)
    assert numerical_plan(settings(), InferenceSpec(), prefix, fitted) == schedule_plan(
        settings(), InferenceSpec(), batch.times, 2, 1, 2, fitted.rate_bounds[1], 2)


def test_reference_design_rejects_old_balanced_mc_floor_without_data_or_fit():
    spec = load_plan(REFERENCE_PLAN)['approximation_spec']
    spec['event_driven_reference'].update(particles=512, independent_batches=4, maximum_jumps_per_particle_interval=256)
    with pytest.raises(SwitchingError, match='balanced IID'):
        reference_design_preflight(ApproximationSpec.from_dict(spec), InferenceSpec(), [0, 0.5, 1, 1.5, 2], 4, 8, 5, 2)


def test_reference_design_exceeding_segment_quota_is_refused_even_with_more_particles():
    spec = load_plan(REFERENCE_PLAN)['approximation_spec']
    spec['event_driven_reference'].update(particles=4096, independent_batches=32)
    with pytest.raises(SwitchingError, match='core quota'):
        reference_design_preflight(ApproximationSpec.from_dict(spec), InferenceSpec(), [0, 0.5, 1, 1.5, 2], 4, 8, 5, 2)


@pytest.mark.parametrize('rate,expected', [(0, 0.0), (1e6, 1.0)])
def test_jump_tail_bound_discloses_absorbing_or_nonrare_quota_risk(rate, expected):
    spec = load_plan(REFERENCE_PLAN)['approximation_spec']
    spec['step_halving']['maximum_substeps'] = 100_000
    # Tiny time step makes the fine grid feasible even at rate 1e6; its large
    # rate-step setting cannot be relaxed. A low jump quota gives nonrare risk.
    spec['event_driven_reference']['maximum_jumps_per_particle_interval'] = 1
    inference = InferenceSpec(max_substeps=100_000)
    report = reference_design_preflight(ApproximationSpec.from_dict(spec), inference, [0, 1e-5], 1, 8, rate, 2)
    assert report['particle_interval_jump_quota_exceedance_union_bound'] == expected
    assert report['actual_mc_precision_proven'] is report['finite_particle_bias_bounded'] is False


def test_positive_rate_jump_tail_underflow_is_not_reported_as_exact_zero():
    spec = ApproximationSpec.from_dict(load_plan(REFERENCE_PLAN)['approximation_spec'])
    report = reference_design_preflight(spec, InferenceSpec(), [0, 1e-308], 1, 1, 1e-308, 2)
    assert report['particle_interval_jump_quota_exceedance_union_bound'] > 0


@pytest.mark.parametrize('damage', ['paths', 'modes', 'dimension', 'rate', 'origin', 'schedule'])
def test_schedule_metadata_refuses_invalid_dimensions_and_nonfinite_prefix(damage):
    arguments = [settings(), InferenceSpec(), [0, 0.5, 1], 2, 1, 2, 100, 2]
    if damage == 'paths': arguments[4] = True
    elif damage == 'modes': arguments[5] = 3
    elif damage == 'dimension': arguments[7] = 0
    elif damage == 'rate': arguments[6] = math.nan
    elif damage == 'origin': arguments[3] = 0
    else: arguments[2] = [0, math.nan, 1]
    with pytest.raises(SwitchingError, match='reference'):
        schedule_plan(*arguments)
