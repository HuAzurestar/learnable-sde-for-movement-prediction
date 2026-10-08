"""Review regression oracles: synthetic inputs, no empirical data or fits.

Tiny four-second software rollouts do not consume the formal science ledger.
These tests establish implementation contracts, not empirical qualification.
"""
from dataclasses import replace

import numpy as np
import pytest

from experiments.pirc17 import decision_policy as policy
from experiments.pirc17.formal_paired import describe_paired
from experiments.pirc17.inference import infer
from experiments.pirc17.method_rollout import causal_features
from experiments.pirc17.origins import causal_prefix
from tests.test_pirc17_decision_policy import rows_for
from tests.test_pirc17_method_rollout import bound, fitted, forecast


ORDINARY = ('seg_constant_mode', 'pointwise_mixture', 'single_gaussian', 'gmm_kernel')


def prefixes(speed):
    endpoint = np.array([2., 1.])
    velocity = np.array([speed, .5 * speed])
    return causal_prefix([endpoint - 2 * velocity, endpoint - velocity, endpoint], [-2., -1., 0.])


def nontrivial(kind):
    model = fitted(kind, names=('solar',), linear=[[-.04, .02], [.01, -.03]])
    weights = []
    for index, value in enumerate(model.weights):
        value = value.copy()
        value[0] += np.array([index * .4, -index * .2])
        value[-1] = [.02, -.01]
        if kind == 'explicit_decomp':
            value[3] = [.7, .2]  # Deliberately active speed coefficient.
        weights.append(value)
    probabilities = np.array([.2, .3, .5]) if len(weights) == 3 else np.ones(1)
    return bound(replace(model, weights=tuple(weights), mode_probabilities=probabilities))


@pytest.mark.parametrize('kind', ORDINARY)
@pytest.mark.parametrize('propagation', ['fp', 'mc', 'crn'])
def test_ordinary_methods_ignore_velocity_given_same_position_clock_parameters_and_stream(kind, propagation):
    dynamics = nontrivial(kind)
    fixed_probabilities = dynamics.model.mode_probabilities.copy()
    settings = dict(propagation=propagation, horizons=(.25, 1., 2., 4.),
                    condition_names=('solar',),
                    condition_at=lambda positions, time: positions[:, :1] * .01 + time * .1,
                    crn_pair_id='software-fixed-pair' if propagation == 'crn' else None)
    original = forecast(dynamics, prefixes(1.), **settings)
    for speed in (-1., 3.):
        changed = forecast(dynamics, prefixes(speed), **settings)
        np.testing.assert_array_equal(original.forecast.positions_m, changed.forecast.positions_m)
        np.testing.assert_array_equal(original.conditional_means_m, changed.conditional_means_m)
        np.testing.assert_array_equal(original.conditional_covariances_m2, changed.conditional_covariances_m2)
    np.testing.assert_array_equal(dynamics.model.mode_probabilities, fixed_probabilities)
    assert not original.diagnostics['scientific_claim_authorized']


@pytest.mark.parametrize('propagation', ['fp', 'mc', 'crn'])
def test_explicit_features_use_speed_not_signed_initial_heading(propagation):
    dynamics = nontrivial('explicit_decomp')
    settings = dict(propagation=propagation, horizons=(.25, 1., 2., 4.),
                    condition_names=('solar',), condition_at=lambda positions, time: np.zeros((len(positions), 1)),
                    crn_pair_id='software-fixed-pair' if propagation == 'crn' else None)
    original = forecast(dynamics, prefixes(1.), **settings)
    reversed_heading = forecast(dynamics, prefixes(-1.), **settings)
    faster = forecast(dynamics, prefixes(3.), **settings)
    np.testing.assert_array_equal(original.forecast.positions_m, reversed_heading.forecast.positions_m)
    assert not np.array_equal(original.forecast.positions_m[:, 0], faster.forecast.positions_m[:, 0])


def test_explicit_radial_features_have_zero_origin_branch_and_are_position_not_velocity_direction():
    model = fitted('explicit_decomp')
    positions = np.array([[0., 0.], [3., 4.]])
    velocity = np.array([[4., 3.], [-4., -3.]])
    features = causal_features(model, positions, velocity, np.empty((2, 0)))
    np.testing.assert_array_equal(features[:, 3], [5., 5.])
    np.testing.assert_allclose(features[:, 4:6], [[0., 0.], [.6, .8]], rtol=0, atol=1e-15)


@pytest.mark.parametrize('family', ['weighted-es-primary', 'weighted-es-lio'])
@pytest.mark.parametrize('delta,expected', [(-60., 'beneficial'), (60., 'harmful'),
                                         (0., 'equivalent'), (policy.CONFIG.delta_m, 'inconclusive')])
def test_actual_terrain_roles_seed_signs_interval_reversal_and_verdict(family, delta, expected):
    contrasts = policy.registered_contrasts(family)
    if family == 'weighted-es-primary':
        assert contrasts['all-vs-base'] == ('all-terrain', 'base')
        assert all(contrasts[g] == ('all-terrain', 'loo-' + g) for g in policy.GROUPS)
    else:
        assert all(contrasts[g] == ('lio-' + g, 'base') for g in policy.GROUPS)
    rows, _ = rows_for(family, delta=delta)
    # Nonzero block spread exercises real endpoint order, not only a point interval.
    candidates = {candidate for candidate, control in contrasts.values()}
    for row in rows:
        if row['configuration'] in candidates:
            row['score_m'] += (int(row['independent_block_id'].split('-')[1]) % 5 - 2) * .5
    result = infer(rows, config=policy.CONFIG, family=contrasts,
                   mechanism_passed={name: True for name in contrasts})
    for value in result['results'].values():
        assert value['verdict'] == expected
        lower, upper = value['simultaneous_interval_m']
        assert lower < upper
        assert value['improvement_simultaneous_interval_m'] == [-upper, -lower]
        assert value['improvement_m'] == -value['delta_estimate_m']
        if delta:
            assert all(np.sign(v) == np.sign(delta) for v in value['seed_delta_m'])
        else:
            assert all(abs(v) < policy.CONFIG.delta_m for v in value['seed_delta_m'])
            assert lower < 0 < upper
        if expected == 'inconclusive':
            assert lower < policy.CONFIG.delta_m < upper


def test_named_loo_100_160_example_has_positive_factor_value_without_changing_roles():
    rows, _ = rows_for('weighted-es-primary', blocks=2)
    for row in rows:
        row['score_m'] = 100. if row['configuration'] == 'all-terrain' else 160.
    result = describe_paired(rows, 'weighted-es-primary')['results']['road']
    assert (result['candidate'], result['control']) == ('all-terrain', 'loo-road')
    assert result['delta_estimate_m'] == -60.
    assert result['improvement_m'] == 60.
    assert result['seed_delta_m'] == [-60.] * 5
