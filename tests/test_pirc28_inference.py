"""Numerical, causality and inference-boundary fixtures for PIRC-28."""

from dataclasses import replace

import numpy as np
import pytest

from estimation.switching import ctmc_bridge_moments, fit_switching, select_initialization
from experiments.pirc28.synthetic import ObservationBatch, generate_scenario
from inference.switching import (FilteringPosterior, InferenceSpec, filter_switching,
    forecast_switching, measurement_update, smooth_switching)
from infrastructure.research_store import digest
from models.switching import SwitchingError, SwitchingSDEModel
from tests.test_pirc28_synthetic import fixture_poll, model, scenario


def initial(k):
    return np.ones(k)/k, np.zeros((k, 4)), np.tile(np.eye(4), (k, 1, 1))


def filter_fixture(k=2, paths=1, **kwargs):
    s = scenario(k=k, paths=paths)
    obs, _, _ = generate_scenario(s, check_budget=fixture_poll)
    m = SwitchingSDEModel.from_dict(s.model)
    result = filter_switching(obs, m, *initial(k), InferenceSpec(**kwargs), check_budget=fixture_poll)
    return obs, m, result


def test_k1_filter_matches_exact_kalman_at_observation_times():
    obs, m, result = filter_fixture(k=1)
    _, means, covariance = initial(1)
    mean, cov = means[0], covariance[0]
    objective = 0
    for index, time in enumerate(obs.times):
        if index:
            f, c, s = [item.numpy() for item in m.conditional_affine_transition(0, time - obs.times[index-1])]
            mean, cov = f @ mean + c, f @ cov @ f.T + s
        mean, cov, score = measurement_update(mean, cov, obs.values[0, index], obs.H, obs.R)
        objective += score
        assert np.allclose(result.posteriors[index].means[0], mean, atol=1e-10)
        assert np.allclose(result.posteriors[index].covariances[0], cov, atol=1e-10)
    assert np.isclose(result.log_likelihood, objective)
    assert result.inference_spec['accuracy'] == 'approximate'


def test_known_mode_degeneracy_and_normalization():
    m = model(rate=0)
    obs, _, _ = generate_scenario(replace(scenario(rate=0), initial_mode_probability=(1.0, 0.0)), check_budget=fixture_poll)
    p, means, covariance = initial(2)
    result = filter_switching(obs, m, [1.0, 0], means, covariance, InferenceSpec(), check_budget=fixture_poll)
    assert np.all(result.probabilities[:, 0] == 1)
    assert np.all(result.probabilities[:, 1] == 0)
    smoothed = smooth_switching(result, check_budget=fixture_poll)
    assert np.allclose(smoothed.probabilities.sum(axis=1), 1)
    assert np.all(smoothed.probabilities[:, 0] == 1)
    assert np.allclose(smoothed.pair_probabilities.sum(axis=(1, 2)), 1)


def test_unused_absorbing_mode_needs_no_singular_observation_likelihood():
    # Separately declared observed-state reference fixture. Its velocities are
    # direct synthetic measurements, not future-derived velocity estimates.
    s = replace(scenario(rate=0, paths=1), H=tuple(np.eye(4).tolist()),
                R=tuple(np.zeros((4, 4)).tolist()), initial_mode_probability=(1.0, 0.0))
    obs, _, _ = generate_scenario(s, check_budget=fixture_poll)
    _, means, covariance = initial(2)
    covariance[1] = 0
    result = filter_switching(obs, model(rate=0), [1.0, 0.0], means, covariance,
                             InferenceSpec(), check_budget=fixture_poll)
    assert np.all(result.probabilities[:, 0] == 1)
    assert np.allclose(result.posteriors[-1].means[0], obs.values[0, -1])


def test_filter_origin_and_prefix_hash_ignore_future_observations():
    obs, m, result = filter_fixture()
    values = obs.values.copy()
    values[0, -1] += 30
    changed = replace(obs, values=values)
    result2 = filter_switching(changed, m, *initial(2), InferenceSpec(), check_budget=fixture_poll)
    assert result.posteriors[1].to_dict() == result2.posteriors[1].to_dict()
    assert result.posteriors[-1].prefix_hash != result2.posteriors[-1].prefix_hash
    forecast = forecast_switching(m, result.posteriors[1], [1, 2], sample_count=8, seed=12, check_budget=fixture_poll)
    forecast2 = forecast_switching(m, result2.posteriors[1], [1, 2], sample_count=8, seed=12, check_budget=fixture_poll)
    assert forecast == forecast2
    smooth = smooth_switching(result, check_budget=fixture_poll)
    smooth2 = smooth_switching(result2, check_budget=fixture_poll)
    assert not np.allclose(smooth.means[0], smooth2.means[0])
    with pytest.raises(SwitchingError) as error:
        forecast_switching(m, replace(result.posteriors[1], role='smoothing-diagnostic-only'), [1],
                          sample_count=2, seed=1, check_budget=fixture_poll)
    assert error.value.code == 'FUTURE_INFORMATION'


def test_ctmc_bridge_counts_include_return_jumps():
    m, dt = model(rate=0.4), 2.0
    dwell, counts = ctmc_bridge_moments(m, dt)
    transition = m.mode_transition(dt).numpy()
    assert np.allclose(dwell.sum(axis=0), dt)
    assert counts[0, 1, 0, 0] > 0  # Endpoint 0->0 includes positive expected return jumps.
    assert counts[1, 0, 0, 0] > 0
    for origin in range(2):
        assert np.isclose(np.sum(counts[:, :, origin, :] * transition[origin]), 0.4 * dt)


def test_joint_mode_state_forecast_retains_correlation_and_bimodality():
    m = SwitchingSDEModel(np.zeros((2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2)),
                          np.zeros((2, 2)), np.zeros((2, 2, 2)))
    means = np.zeros((2, 4))
    means[:, 0] = [-100, 100]
    posterior = FilteringPosterior(2.0, np.array([0.5, 0.5]), means, np.zeros((2, 4, 4)),
                                   digest(m.to_dict()), digest('past-only'))
    result = forecast_switching(m, posterior, [1, 3], sample_count=80, seed=45, check_budget=fixture_poll)
    samples, modes = np.asarray(result['samples']), np.asarray(result['mode_paths'])
    assert np.array_equal(samples[:, :, 0], 200*modes - 100)
    assert set(samples[:, -1, 0]) == {-100, 100}
    assert result['probability_any_jump'] == [0, 0]
    assert result['mean_dwell_seconds'] == [None, None]
    assert result['mean_is_summary_only'] is True


def test_any_jump_and_different_endpoint_probabilities_are_distinct():
    _, m, filtered = filter_fixture()
    forecast = forecast_switching(m, filtered.posteriors[-1], [2, 5], sample_count=8,
                                 seed=8, check_budget=fixture_poll)
    assert np.all(np.array(forecast['probability_any_jump']) > forecast['probability_different_endpoint_mode'])
    assert np.allclose(np.asarray(forecast['mode_marginal']).sum(axis=1), 1)


def test_label_permuted_filter_is_equivariant():
    obs, m, filtered = filter_fixture()
    p, means, covariance = initial(2)
    other = filter_switching(obs, m.permuted([1, 0]), p[::-1], means[::-1], covariance[::-1],
                            InferenceSpec(), check_budget=fixture_poll)
    assert np.isclose(filtered.log_likelihood, other.log_likelihood)
    assert np.allclose(filtered.probabilities, other.probabilities[:, ::-1])
    assert np.allclose(filtered.means, other.means[:, ::-1])
    smooth = smooth_switching(filtered, check_budget=fixture_poll)
    smooth2 = smooth_switching(other, check_budget=fixture_poll)
    assert np.allclose(smooth.probabilities, smooth2.probabilities[:, ::-1])


def test_halving_substep_converges_toward_fine_grid_on_multi_jump_fixture():
    s = replace(scenario(paths=1), times=(0.0, 2.0, 4.0))
    obs, truth, _ = generate_scenario(s, check_budget=fixture_poll)
    m = SwitchingSDEModel.from_dict(s.model)
    estimates = [filter_switching(obs, m, *initial(2), InferenceSpec(max_rate_step=value),
                                 check_budget=fixture_poll) for value in (0.05, 0.025, 0.00625)]
    coarse, fine, reference = [result.posteriors[-1] for result in estimates]
    coarse_error = np.linalg.norm(coarse.means - reference.means) + np.linalg.norm(coarse.probability-reference.probability)
    fine_error = np.linalg.norm(fine.means-reference.means) + np.linalg.norm(fine.probability-reference.probability)
    assert fine_error <= coarse_error * 1.1 + 1e-9
    assert any(len(row) > 1 for row in truth.jump_times)


def test_generalized_em_accepts_only_nondecreasing_approximate_objective():
    obs, m, _ = filter_fixture(k=1, paths=3)
    wrong = m.to_dict()
    wrong['a'] = [[0.0, 0.0]]
    fit = fit_switching(obs, SwitchingSDEModel.from_dict(wrong), *initial(1), InferenceSpec(max_iterations=3),
                        initialization_id='unit-fixture-1', check_budget=fixture_poll)
    assert np.all(np.diff(fit.objective_trace) >= 0)
    assert len(fit.objective_trace) >= 2
    assert fit.objective_trace[-1] > fit.objective_trace[0]
    assert fit.model.a[0, 0] > 0.1
    assert fit.collapsed is False
    assert fit.to_dict()['accuracy'] == 'approximate'


def test_selection_retains_failed_or_collapsed_candidate_diagnostics():
    obs, m, _ = filter_fixture(k=1, paths=1)
    spec = InferenceSpec(max_iterations=1)
    fit = fit_switching(obs, m, *initial(1), spec, initialization_id='unit-fit', check_budget=fixture_poll)
    selected, receipt = select_initialization([fit], obs, *initial(1), spec, split_role='selection', check_budget=fixture_poll)
    assert selected is fit
    assert len(receipt['candidates']) == 1
    assert 'stopping_reason' in receipt['candidates'][0]
    with pytest.raises(SwitchingError) as error:
        select_initialization([fit], obs, *initial(1), spec, split_role='test', check_budget=fixture_poll)
    assert error.value.code == 'UNAUTHORIZED_DATA'


def test_quota_contract_and_poll_refusals():
    obs, m, _ = filter_fixture()
    with pytest.raises(SwitchingError) as error:
        filter_switching(obs, m, *initial(2), InferenceSpec(max_substeps=1), check_budget=fixture_poll)
    assert error.value.code == 'INFERENCE_INAPPLICABLE'
    with pytest.raises(SwitchingError):
        InferenceSpec(max_rate_step=0.06).validate()
    def stop():
        raise TimeoutError('shared supervisor deadline')
    with pytest.raises(TimeoutError):
        filter_switching(obs, m, *initial(2), InferenceSpec(), check_budget=stop)


def test_fit_rejects_truth_instead_of_extracting_observations_from_it():
    _, truth, _ = generate_scenario(scenario(), check_budget=fixture_poll)
    with pytest.raises(SwitchingError):
        fit_switching(truth, model(), *initial(2), InferenceSpec(), initialization_id='rejected', check_budget=fixture_poll)


def test_em_checkpoint_continues_at_accepted_iteration_without_new_iteration_budget():
    obs, m, _ = filter_fixture(k=1, paths=2)
    saved = []
    spec = InferenceSpec(max_iterations=3)
    expected = fit_switching(obs, m, *initial(1), spec, initialization_id='checkpoint-fixture',
        check_budget=fixture_poll, progress_callback=saved.append)
    intermediate = next(state for state in saved if state['completed_iterations'] == 1 and state['complete_reason'] is None)
    actual = fit_switching(obs, m, *initial(1), spec, initialization_id='checkpoint-fixture',
        check_budget=fixture_poll, resume_state=intermediate)
    assert actual.model.to_dict() == expected.model.to_dict()
    assert actual.objective_trace == expected.objective_trace
    assert len(actual.objective_trace) <= spec.max_iterations + 1
    changed = replace(obs, values=obs.values + 0.1)
    with pytest.raises(SwitchingError) as error:
        fit_switching(changed, m, *initial(1), spec, initialization_id='checkpoint-fixture',
            check_budget=fixture_poll, resume_state=intermediate)
    assert error.value.code == 'IDENTITY_CONFLICT'


def test_k2_generalized_em_reports_occupancy_and_all_objectives():
    obs, m, _ = filter_fixture(k=2, paths=3)
    result = fit_switching(obs, m, *initial(2), InferenceSpec(max_iterations=2),
        initialization_id='k2-fixture', check_budget=fixture_poll)
    assert np.all(np.diff(result.objective_trace) >= 0)
    assert np.isclose(result.occupancy.sum(), 1)
    assert result.to_dict()['configured_K'] == 2
    assert np.isfinite(result.model.Q.numpy()).all()


def test_identical_modes_are_flagged_even_with_balanced_occupancy():
    obs, m, _ = filter_fixture(k=2, paths=2)
    identical = m.to_dict()
    identical['a'][1] = identical['a'][0]
    result = fit_switching(obs, SwitchingSDEModel.from_dict(identical), *initial(2),
        InferenceSpec(max_iterations=1), initialization_id='nonidentifiable-fixture', check_budget=fixture_poll)
    assert np.allclose(result.occupancy, [0.5, 0.5])
    assert result.indistinguishable_modes is True
    assert result.collapsed is True
