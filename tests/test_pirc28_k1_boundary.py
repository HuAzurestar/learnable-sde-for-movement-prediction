"""Disposable analytic controls; no managed/scientific qualification."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from experiments.pirc28 import k1_boundary as boundary
from experiments.pirc28.input_evidence import validate_generated_inputs
from experiments.pirc28.pipeline import execute_recipe, recipe_allocation
from experiments.pirc28.selection_evidence import validate_selected_fit
from inference.switching import InferenceSpec
from infrastructure.research_store import ResearchError
from models.switching import SwitchingError
from tests.test_pirc28_approximation import v2_recipe
from tests.test_pirc28_approximation_checks import observations, settings
from tests.test_pirc28_inference import initial
from tests.test_pirc28_runtime import recipe
from tests.test_pirc28_synthetic import fixture_poll


@pytest.fixture(scope='module')
def control():
    batch, model = observations(paths=2)
    report = boundary.check_k1_boundary(batch, model, *initial(1), InferenceSpec(), settings(),
        origin_index=2, check_budget=fixture_poll)
    return batch, model, report


def verify(control, report=None, **kwargs):
    batch, model, original = control
    return boundary.validate_k1_boundary(original if report is None else report, batch, model,
        *initial(1), InferenceSpec(), settings(), origin_index=2,
        check_budget=kwargs.pop('check_budget', fixture_poll), **kwargs)


def test_all_paths_analytic_only_and_no_scientific_authority(control, monkeypatch):
    monkeypatch.setattr('experiments.pirc28.approximation_checks._reference_batch',
                        lambda *a, **k: pytest.fail('K1 sampled Monte Carlo'))
    batch, model, report = control
    assert boundary.check_k1_boundary(batch, model, *initial(1), InferenceSpec(), settings(),
        origin_index=2, check_budget=fixture_poll) == report
    receipt = verify(control)
    assert receipt['state'] == 'NUMERICAL_CHECKS_PASSED'
    assert len(report['paths']) == 2
    assert report['resource_plan']['reference_particles'] == 0
    assert report['resource_plan']['independent_batches'] == 0
    assert report['resource_plan']['owner_report_copy_elements'] > 0
    assert all(receipt[name] is False for name in boundary.FLAGS)


def test_verifier_does_not_use_producer_update_or_boundary(control, monkeypatch):
    monkeypatch.setattr(boundary, '_known_mode_checks', lambda *a, **k: pytest.fail('producer boundary'))
    monkeypatch.setattr('experiments.pirc28.approximation_checks._kalman_update',
                        lambda *a, **k: pytest.fail('producer update'))
    assert verify(control)['numerical_diagnostics_consistent'] is True


@pytest.mark.parametrize('damage', ['model_hash', 'observation_prefix_hash', 'inference_hash',
    'spec_hash', 'initial_state_hash', 'origin_index', 'resource_plan', 'managed_evidence_bound',
    'qualified_identifiability', 'real_data_authorized', 'reference_sampling_verified', 'state',
    'missing-path', 'path-order', 'boolean-path', 'passed', 'mode', 'mean', 'covariance',
    'likelihood', 'extra-field', 'boolean-plan', 'huge-plan'])
def test_forged_report_refused(control, damage):
    report = copy.deepcopy(control[2])
    if damage.endswith('_hash'): report[damage] = '0'*64
    elif damage in boundary.FLAGS: report[damage] = True
    elif damage == 'origin_index': report[damage] = True
    elif damage == 'resource_plan': report[damage]['tensor_elements'] = 1
    elif damage == 'boolean-plan': report['resource_plan']['reference_particles'] = False
    elif damage == 'huge-plan': report['resource_plan']['tensor_elements'] = [0]*100000
    elif damage == 'state': report[damage] = 'INFERENCE_INAPPLICABLE'
    elif damage == 'missing-path': report['paths'].pop()
    elif damage == 'path-order': report['paths'].reverse()
    elif damage == 'boolean-path': report['paths'][0]['path_index'] = False
    elif damage == 'passed': report['paths'][0]['boundary']['passed'] = False
    elif damage == 'mode': report['paths'][0]['boundary']['mode'] = False
    elif damage == 'extra-field': report['truth'] = []
    else:
        field = {'mean': 'mean_max_abs', 'covariance': 'covariance_max_abs',
                 'likelihood': 'log_likelihood_abs'}[damage]
        report['paths'][0]['boundary'][field] = 1.0
    with pytest.raises(ResearchError):
        verify(control, report)


def test_future_values_and_times_never_used(control):
    batch, model, report = control
    values, times = batch.values.copy(), batch.times.copy()
    values[:, 3:] = np.nan
    times[3:] = np.nan
    assert verify((replace(batch, values=values, times=times), model, report)) == verify(control)


def test_actual_mismatch_keeps_inapplicable_not_pass(control, monkeypatch):
    original = boundary._known_mode_checks
    def failed(*args, **kwargs):
        rows = original(*args, **kwargs)
        rows[0].update(passed=False, mean_max_abs=1.0)
        return rows
    monkeypatch.setattr(boundary, '_known_mode_checks', failed)
    batch, model, _ = control
    report = boundary.check_k1_boundary(batch, model, *initial(1), InferenceSpec(), settings(),
        origin_index=2, check_budget=fixture_poll)
    assert report['state'] == 'INFERENCE_INAPPLICABLE'
    # An invented failure also must not substitute for independently computed evidence.
    with pytest.raises(ResearchError):
        verify(control, report)


def test_independent_verifier_preserves_real_failed_filter_boundary(control, monkeypatch):
    from experiments.pirc28 import approximation_checks, approximation_evidence
    original = approximation_checks.filter_switching
    def altered_filter(*args, **kwargs):
        filtered = original(*args, **kwargs)
        # Simulate a genuine implementation error in the filtered output, not
        # a forged producer PASS/FAIL or a modification of original observations.
        filtered.posteriors[0].means[0, 0] += 1.0
        return filtered
    monkeypatch.setattr(approximation_checks, 'filter_switching', altered_filter)
    monkeypatch.setattr(approximation_evidence, 'filter_switching', altered_filter)
    batch, model, _ = control
    report = boundary.check_k1_boundary(batch, model, *initial(1), InferenceSpec(), settings(),
        origin_index=2, check_budget=fixture_poll)
    receipt = verify(control, report)
    assert report['state'] == receipt['state'] == 'INFERENCE_INAPPLICABLE'
    assert all(row['boundary']['passed'] is False for row in report['paths'])
    assert receipt['numerical_diagnostics_consistent'] is True
    assert receipt['qualified_identifiability'] is False


def test_deadline_or_invalid_model_stops_before_filter(control, monkeypatch):
    class Deadline(Exception): pass
    def stop(): raise Deadline('original arm deadline')
    monkeypatch.setattr(boundary, '_known_mode_checks', lambda *a, **k: pytest.fail('early filter'))
    with pytest.raises(Deadline): verify(control, check_budget=stop)
    with pytest.raises(ResearchError): verify(control, check_budget=None)
    batch, model = observations(k=2)
    with pytest.raises(ResearchError, match='one-mode'):
        boundary.check_k1_boundary(batch, model, *initial(2), InferenceSpec(), settings(),
            origin_index=2, check_budget=fixture_poll)


def test_resource_refusal_before_prefix_copy(control, monkeypatch):
    monkeypatch.setattr(boundary, 'observation_prefix', lambda *a: pytest.fail('early copy'))
    batch, model, _ = control
    with pytest.raises(SwitchingError, match='original step quota'):
        boundary.check_k1_boundary(batch, model, *initial(1),
            InferenceSpec(max_substeps=1), settings(), origin_index=2, check_budget=fixture_poll)


def test_k1_full_generated_roles_are_independently_replayed():
    value = recipe()
    fit = execute_recipe(value, 1, 19, check_budget=fixture_poll)['fit']
    generated = fit['generated_inputs']
    model, receipt = validate_selected_fit(fit, value, 19, k=1, check_budget=fixture_poll)
    assert model.n_modes == 1
    assert receipt['selection_input_replayed'] is receipt['selection_scores_recomputed'] is True
    assert validate_generated_inputs(generated, value, 19, k=1,
        check_budget=fixture_poll)['full_generated_input_hashes_verified'] is True
    for role in range(3):
        changed = copy.deepcopy(generated)
        changed[role]['observations_hash'] = '0'*64
        with pytest.raises(ResearchError, match='source hashes differ'):
            validate_generated_inputs(changed, value, 19, k=1, check_budget=fixture_poll)


def test_v2_k1_additive_boundary_without_mc_and_legacy_unchanged():
    allocation = recipe_allocation(v2_recipe(), 1)
    assert allocation['reference_elements'] == 0 and allocation['reference_core_plan'] is None
    assert allocation['k1_boundary_elements'] == allocation['k1_boundary_plan']['tensor_elements'] > 0
    raw = sum(allocation[name] for name in ('source_data_elements', 'fit_metadata_elements',
        'retained_forecast_elements', 'reference_elements', 'k1_boundary_elements'))
    assert 0 <= allocation['padded_pool_elements']-raw <= 3
    assert recipe_allocation(recipe(), 1)['k1_boundary_elements'] == 0
