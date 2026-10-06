"""Deterministic evaluator fixtures, not statistical research qualification."""

from dataclasses import replace
import copy

import numpy as np
import pytest

from experiments.pirc28.qualification import (AXES, METRICS, PARAMETERS, IdentifiabilitySpec,
    forecast_coverage, jump_diagnostics, mode_diagnostics, parameter_recovery, phase_scenarios, qualify)
from infrastructure.research_store import digest
from models.switching import SwitchingError, SwitchingSDEModel
from tests.test_pirc28_synthetic import fixture_poll, model, scenario


def axes():
    return dict(zip(AXES, ([1.0], [2.0], [0.5], [0.1], [4.0], [0.5])))


def spec(count=8):
    return IdentifiabilitySpec({ 'scenario': {name: values[0] for name, values in axes().items()}},
        tuple('rep-'+str(index) for index in range(count)), {name: 0.25 for name in METRICS},
        {name: 1.0 for name in PARAMETERS}, minimum_occupancy=0.05,
        maximum_collapse_fraction=0.25, minimum_recovery_fraction=0.5,
        minimum_effective_replicates=2, bootstrap_draws=100)


def rows(value):
    return [{'scenario_id': 'scenario', 'replicate_id': replicate, 'K': k, 'state': 'SUCCEEDED',
        'input_hash': digest(replicate), 'artifact_hash': digest([replicate, k]),
        'metrics': {**{name: 0.01 for name in METRICS}, 'position_energy_score_d2': 1/k},
        'posterior_occupancy': [1.0] if k == 1 else [0.5, 0.5], 'collapsed': False}
        for replicate in value.replicate_ids for k in (1, 2)]


def run(value, result):
    return qualify(value, result, digest(result), check_budget=fixture_poll)


def test_phase_matrix_is_cartesian_bounded_and_replayable():
    point = axes()
    point['mode_separation'] = [0.0, 2.0]
    point['observation_interval_s'] = [0.5, 1.0]
    result = phase_scenarios(scenario(paths=1), point)
    assert len(result) == 4
    assert result == phase_scenarios(scenario(paths=1), point)
    assert len({row['scenario_hash'] for row in result}) == 4
    for row in result:
        s = row['scenario']
        q = np.asarray(s['model']['Q'])
        pi = np.asarray(s['initial_mode_probability'])
        assert np.allclose(pi @ q, 0)
        assert np.isclose(1/np.sum(pi * -q.diagonal()), row['axes']['dwell_time_s'])
        assert s['times'][-1] == 4
    with pytest.raises(SwitchingError):
        phase_scenarios(scenario(paths=1), point, maximum_scenarios=3)


@pytest.mark.parametrize('name,value', [('mode_occupancy', 1.0), ('dwell_time_s', 0),
    ('observation_interval_s', True), ('trajectory_length_s', 0.1), ('observation_noise_m', -1)])
def test_phase_axes_reject_invalid_values(name, value):
    point = axes()
    point[name] = [value]
    with pytest.raises(SwitchingError):
        phase_scenarios(scenario(paths=1), point)


def test_parameter_recovery_is_label_and_noise_rotation_invariant():
    truth = model()
    parameters = {name: getattr(truth, name).numpy()[[1, 0]].copy() for name in ('A', 'B', 'a', 'L')}
    parameters['L'] = parameters['L'] @ np.array([[0, -1], [1, 0]])
    estimate = SwitchingSDEModel(truth.Q.numpy()[[1, 0]][:, [1, 0]], **parameters)
    result = parameter_recovery(estimate, truth, {name: 1.0 for name in PARAMETERS})
    assert result['label_order'] == [1, 0]
    assert all(value < 1e-12 for value in result['metrics'].values())
    estimate = SwitchingSDEModel(truth.Q.numpy()*2, **{name: getattr(truth, name).numpy() for name in ('A', 'B', 'a', 'L')})
    assert parameter_recovery(estimate, truth, {name: 1.0 for name in PARAMETERS})['metrics']['rate_rmse_s_inv'] > 0


def test_mode_calibration_uses_parameter_alignment_and_continuous_dwell():
    labels = np.array([[0, 1], [1, 0]])
    p = np.eye(2)[labels]
    first = mode_diagnostics(p, labels, [0, 1], [[1.0, 1.0], [1.0, 1.0]], 2)
    second = mode_diagnostics(p[..., [1, 0]], labels, [1, 0], [[1.0, 1.0], [1.0, 1.0]], 2)
    assert first == second
    assert first['mode_ece'] == first['mode_brier'] == first['occupancy_error'] == 0
    assert first['within_path_points_are_not_replicates']
    with pytest.raises(SwitchingError):
        mode_diagnostics(p, labels, [0, 1], [[1, 0], [1, 1]], 2)


def test_coverage_is_explicitly_marginal_not_joint():
    samples = np.zeros((100, 2, 4))
    samples[..., :2] = np.linspace(-1, 1, 100)[:, None, None]
    result = forecast_coverage(samples, np.zeros((2, 4)))
    assert result['position_marginal_coverage'] == 1
    assert result['position_marginal_coverage_error'] == pytest.approx(0.1)
    assert 'not joint' in result['definition']


def test_return_jumps_and_endpoint_changes_have_separate_proper_scores():
    forecast = {'role': 'filtering-origin', 'origin': 0.0, 'horizons': [1.0],
                'probability_any_jump': [0.9], 'probability_different_endpoint_mode': [0.1]}
    result = jump_diagnostics([forecast], [[0.25, 0.75]], np.array([[0, 0]]), [0, 1], 0, [1])
    assert result['any_jump_brier'] == pytest.approx(0.01)
    assert result['different_endpoint_mode_brier'] == pytest.approx(0.01)
    assert 'within-path-events-are-not-replicates' in result['sampling_unit']
    with pytest.raises(SwitchingError):
        jump_diagnostics([{**forecast, 'role': 'smoothing'}], [[0.25]], np.array([[0, 1]]), [0, 1], 0, [1])


def test_missing_thresholds_cannot_qualify_or_authorize_real_data():
    value = spec()
    value = replace(value, minimum_occupancy=None, thresholds={**value.thresholds, 'mode_ece': None})
    result = run(value, rows(value))
    assert result['state'] == 'NEEDS_PREREGISTRATION'
    assert set(result['missing_fields']) == {'mode_ece', 'minimum_occupancy'}
    assert result['real_data_authorized'] is False


def test_recovery_intervals_use_registered_replicates_and_replay():
    value = spec()
    result = run(value, rows(value))
    assert result == run(value, rows(value))
    assert result['state'] == 'PASSED'
    point = result['scenarios']['scenario']
    assert point['registered_replicates'] == point['effective_replicates'] == 8
    assert point['paired_energy_difference_interval'] == [-0.5, -0.5]
    assert 0.5 < point['recovery_fraction_interval'][0] < 1
    assert result['real_data_authorized'] is False
    assert run(spec(2), rows(spec(2)))['state'] == 'UNIDENTIFIABLE'


def test_missing_and_budget_exhausted_rows_remain_distinct():
    value = spec()
    data = rows(value)
    missing = run(value, data[:-1])
    assert missing['state'] == 'INCOMPLETE'
    assert missing['scenarios']['scenario']['registered_replicates'] == 8
    exhausted = copy.deepcopy(data)
    exhausted[-1].update(state='BUDGET_EXHAUSTED', metrics=None, posterior_occupancy=None, collapsed=None)
    result = run(value, exhausted)
    assert result['state'] == 'BUDGET_EXHAUSTED'
    point = result['scenarios']['scenario']
    assert point['recovery_fraction'] == 7/8
    assert point['failures'][0]['K2'] == 'BUDGET_EXHAUSTED'


def test_collapse_and_bad_calibration_do_not_disappear_from_denominator():
    value = spec()
    data = rows(value)
    for row in data:
        if row['K'] == 2:
            row['collapsed'] = True
            row['metrics']['mode_ece'] = 0.5
    result = run(value, data)
    assert result['state'] == 'UNIDENTIFIABLE'
    point = result['scenarios']['scenario']
    assert point['recovery_fraction'] == 0
    assert point['observed_collapse_fraction'] == 1
    assert point['successful_fit_metric_mean_intervals']['mode_ece'] == [0.5, 0.5]


def test_k1_failure_does_not_censor_successful_k2_collapse_or_metrics():
    value, data = spec(), rows(spec())
    for row in data:
        if row['K'] == 1:
            row.update(state='FAILED', metrics=None, posterior_occupancy=None, collapsed=None)
        else:
            row['collapsed'] = True
    result = run(value, data)
    point = result['scenarios']['scenario']
    assert result['state'] == 'UNIDENTIFIABLE'
    assert point['effective_replicates'] == point['recovering_replicates'] == 0
    assert point['successful_K2_fits'] == point['metric_effective_replicates'] == 8
    assert point['known_K2_collapse_count'] == 8
    assert point['observed_collapse_fraction'] == point['registered_collapse_lower_bound'] == 1
    assert point['paired_energy_difference_interval'] is None
    assert point['successful_fit_metric_mean_intervals']['mode_ece'] == [0.01, 0.01]
    assert 'COLLAPSE_GATE_FAILED' in point['reasons']


def test_no_successful_k2_fit_has_unknown_collapse_not_observed_zero():
    value, data = spec(), rows(spec())
    for row in data:
        if row['K'] == 2:
            row.update(state='FAILED', metrics=None, posterior_occupancy=None, collapsed=None)
    point = run(value, data)['scenarios']['scenario']
    assert point['observed_collapse_fraction'] is None
    assert point['registered_collapse_lower_bound'] == 0
    assert point['unknown_K2_collapse_replicates'] == list(value.replicate_ids)
    assert point['metric_effective_replicates'] == 0
    assert point['successful_fit_metric_mean_intervals'] == {}


def test_partial_collapse_separates_observed_share_from_registered_lower_bound():
    value, data = spec(), rows(spec())
    for row in data:
        if row['K'] == 2:
            if row['replicate_id'] == value.replicate_ids[0]:
                row['collapsed'] = True
            else:
                row.update(state='NOT_RUN', metrics=None, posterior_occupancy=None, collapsed=None)
    result = run(value, data)
    point = result['scenarios']['scenario']
    assert result['state'] == 'INCOMPLETE'
    assert point['observed_collapse_fraction'] == 1
    assert point['registered_collapse_lower_bound'] == 1/8
    assert point['collapse_threshold_denominator'] == 'all-registered-replicates'
    assert point['unknown_K2_collapse_replicates'] == list(value.replicate_ids[1:])


@pytest.mark.parametrize('change', ['duplicate', 'repeated-input', 'unpaired-input', 'hash', 'negative', 'nan'])
def test_invalid_aggregate_refuses_qualification(change):
    value, data = spec(), rows(spec())
    aggregate = None
    if change == 'duplicate':
        data.append(data[0])
    elif change == 'repeated-input':
        data[2]['input_hash'] = data[0]['input_hash']
    elif change == 'unpaired-input':
        data[1]['input_hash'] = digest('different-block')
    elif change == 'hash':
        aggregate = digest('wrong')
    elif change == 'negative':
        data[1]['metrics']['mode_ece'] = -0.1
    else:
        data[1]['metrics']['mode_ece'] = float('nan')
    with pytest.raises((SwitchingError, ValueError)):
        qualify(value, data, digest(data) if aggregate is None else aggregate, check_budget=fixture_poll)


def test_qualification_polls_in_bounded_bootstrap_batches():
    value, calls = spec(), []
    def poll():
        calls.append(1)
        if len(calls) == 20:
            raise RuntimeError('shared supervisor refusal')
    with pytest.raises(RuntimeError, match='supervisor refusal'):
        qualify(value, rows(value), digest(rows(value)), check_budget=poll)


def test_spec_rejects_missing_axis_and_initialization_as_replicate():
    value = spec()
    with pytest.raises(SwitchingError):
        replace(value, scenario_axes={'scenario': {'mode_separation': 1}}).validate()
    with pytest.raises(SwitchingError):
        replace(value, replicate_ids=('init-0', 'init-0')).validate()


def test_identifiability_spec_round_trip_refuses_unbound_fields():
    value = spec().to_dict()
    assert IdentifiabilitySpec.from_dict(value).to_dict() == value
    value['unregistered_threshold_override'] = 0
    with pytest.raises(SwitchingError):
        IdentifiabilitySpec.from_dict(value)


def test_frozen_recipe_emits_evaluator_recovery_without_promoting_cell():
    from experiments.pirc28.pipeline import execute_recipe
    from tests.test_pirc28_runtime import recipe
    value = recipe()
    identifier = value['scenario']['scenario_id']
    qualification = replace(spec(), scenario_axes={identifier: spec().scenario_axes['scenario']})
    value['identifiability_spec'] = qualification.to_dict()
    output = execute_recipe(value, 2, 19, check_budget=fixture_poll)
    diagnostics = output['fit']['recovery_diagnostics']
    assert diagnostics['state'] == 'UNQUALIFIED_CELL_DIAGNOSTICS'
    assert set(diagnostics['parameter_recovery']['metrics']) == set(PARAMETERS.values())
    assert diagnostics['mode_diagnostics']['calibration_bins'] == qualification.calibration_bins
    assert set(('any_jump_brier', 'different_endpoint_mode_brier')) <= diagnostics['jump_diagnostics'].keys()
    assert output['fit']['qualified_identifiability'] is False
    relabeled = copy.deepcopy(value)
    relabeled['scenario']['label_permutation'] = [1, 0]
    relabeled_output = execute_recipe(relabeled, 2, 19, check_budget=fixture_poll)
    relabeled_diagnostics = relabeled_output['fit']['recovery_diagnostics']
    assert relabeled_diagnostics['parameter_recovery']['metrics'] == diagnostics['parameter_recovery']['metrics']
    for name in ('mode_brier', 'mode_ece', 'occupancy_error'):
        assert relabeled_diagnostics['mode_diagnostics'][name] == pytest.approx(diagnostics['mode_diagnostics'][name])
    one = execute_recipe(value, 1, 19, check_budget=fixture_poll)
    assert 'unavailable' in one['fit']['recovery_diagnostics']
    assert output['fit']['generated_inputs'] == one['fit']['generated_inputs']
    value['identifiability_spec'] = replace(qualification, minimum_occupancy=None).to_dict()
    output = execute_recipe(value, 2, 19, check_budget=fixture_poll)
    assert output['fit']['recovery_diagnostics']['state'] == 'NEEDS_PREREGISTRATION'
