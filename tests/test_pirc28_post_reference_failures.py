"""Post-reference retention controls; metadata is NOT a fabricated native failure."""

import copy

import pytest

from experiments.pirc28 import pipeline, managed_evidence
from experiments.pirc28.failure_artifacts import failure_result, validate_cell_failure
from experiments.pirc28.failures import NumericalCellFailure, numerical_diagnostic
from experiments.pirc28.reference_execution import attach_reference_failure, validate_reference_execution
from experiments.pirc28.runtime import switching_plugin
from infrastructure.research_store import ResearchError, digest
from models.switching import SwitchingError
from tests.test_pirc28_managed_approximation import reference_execution
from tests.test_pirc28_managed_evidence import computed_prefix
from tests.test_pirc28_reference_execution import source
from tests.test_pirc28_synthetic import fixture_poll


@pytest.mark.parametrize('stage,target', [('test-inference', 'filter_switching'),
    ('forecast', 'forecast_switching'), ('evaluation', 'energy_score')])
def test_library_later_failure_retains_complete_actual_report_without_success_metrics(reference_execution, monkeypatch,
                                                                                       stage, target):
    recipe, expected, snapshots = reference_execution
    saved = next(state for state in reversed(snapshots) if all(
        fit is not None and fit['complete_reason'] is not None for fit in state['fit_states']))
    report = expected['fit']['approximation_checks']
    def fail(*args, **kwargs):
        raise SwitchingError('NUMERICAL_FAILURE', 'registered later stage failed')
    monkeypatch.setattr(pipeline, target, fail)
    with pytest.raises(NumericalCellFailure) as caught:
        pipeline.execute_recipe(recipe, 2, 19, check_budget=fixture_poll, resume_state=saved,
            fixture_approximation_executor=lambda *a, **k: copy.deepcopy(report))
    diagnostic = caught.value.diagnostic
    assert diagnostic['stage'] == stage and diagnostic['approximation_checks'] == report
    assert 'reference_execution' not in diagnostic  # no native authority from a library hook
    study = {'switching_recipe': recipe, 'protocol_hash': 'p', 'data_hash': 'd', 'admission': {'mode': 'fixture'}}
    result = failure_result(diagnostic, study, {'fit_K': 2, 'seed': 19}, switching_plugin())
    assert result['status'] == 'FAILED' and 'metrics' not in result and 'forecast' not in result
    validate_cell_failure(result, study, {'fit_K': 2, 'seed': 19})


def later_diagnostic(fit, stage):
    cause = numerical_diagnostic(SwitchingError('NUMERICAL_FAILURE', 'unit later-stage error'), stage, None)
    error = NumericalCellFailure(stage, copy.deepcopy(fit['initializations']), copy.deepcopy(fit['selection']),
                                 error_code='NUMERICAL_FAILURE', cause=cause)
    error.diagnostic.update(recipe_hash=fit['recipe_hash'], generated_inputs=copy.deepcopy(fit['generated_inputs']),
                            approximation_checks=copy.deepcopy(fit['approximation_checks']))
    return error.diagnostic


@pytest.mark.parametrize('stage', ['test-inference', 'forecast', 'evaluation'])
def test_worker_attachment_binds_retained_report_but_cannot_reclassify_actual_success(computed_prefix, stage):
    store, study, cell, attempt, result, value, _ = source(computed_prefix)
    fit = result['fit']
    diagnostic = later_diagnostic(fit, stage)
    attach_reference_failure(diagnostic, study, cell, attempt, fit['reference_execution']['completed_batches'])
    failure = failure_result(diagnostic, study, cell, switching_plugin())
    validate_cell_failure(failure, study, cell)
    assert diagnostic['reference_execution']['complete'] is True
    assert validate_reference_execution(diagnostic['reference_execution'], study, cell, attempt, fit['model'],
                                         diagnostic['approximation_checks']) is True
    assert failure['source_schema'] == 'pirc28-synthetic-failure-v3' and failure['status'] == 'FAILED'
    assert 'metrics' not in failure and 'forecast' not in failure
    # This is a unit diagnostic, not an actual failed attempt. Do NOT publish it.
    # Borrowed valid native metadata cannot change canonical actual source bytes.
    failure['admission_hash'] = result['admission_hash']
    history = next(row['attempts'] for row in value['report']['lineage']
                   if row['selected_attempt_id'] == attempt['attempt_id'])
    metadata = store.manifest('artifact-'+attempt['artifact_id'])
    with pytest.raises(ResearchError, match='checked source artifact projection changed'):
        managed_evidence.check_cell_prefix(store, study, cell, attempt, failure, metadata,
            computed_prefix[3], history, check_budget=fixture_poll)
    assert store.attempts()[attempt['attempt_id']]['state'] == 'SUCCEEDED'


@pytest.mark.parametrize('damage', ['no-report', 'failed-report', 'missing-batch', 'boolean-attempt',
    'missing-attempt', 'short-run', 'boolean-source', 'non-string-root', 'empty-store'])
def test_later_failure_and_native_identity_cannot_be_forged(computed_prefix, damage):
    _, study, cell, attempt, result, *_ = source(computed_prefix)
    fit = result['fit']
    diagnostic = later_diagnostic(fit, 'forecast')
    attach_reference_failure(diagnostic, study, cell, attempt, fit['reference_execution']['completed_batches'])
    record = diagnostic['reference_execution']
    if damage == 'no-report': diagnostic['approximation_checks'] = None
    elif damage == 'failed-report': diagnostic['approximation_checks']['state'] = 'INFERENCE_INAPPLICABLE'
    elif damage == 'missing-batch': record['completed_batches'] = record['completed_batches'][:-1]
    elif damage == 'boolean-attempt': record['attempt_id'] = True
    elif damage == 'missing-attempt': record['attempt_id'] = None
    elif damage == 'short-run': record['run_id'] = '0'
    elif damage == 'boolean-source': record['source_code_hash'] = True
    elif damage == 'non-string-root': record['runtime_binding'] = {'root': True, 'store_id': 'fixture'}
    else: record['runtime_binding'] = {'root': study['runtime_binding']['root'], 'store_id': ''}
    value = failure_result(diagnostic, study, cell, switching_plugin())
    with pytest.raises(ResearchError):
        validate_cell_failure(value, study, cell)


@pytest.mark.parametrize('stage', ['test-inference', 'forecast', 'evaluation'])
@pytest.mark.parametrize('error', [ResearchError('UNAUTHORIZED_DATA', 'original authority unavailable'),
                                 RuntimeError('unknown later-stage error')])
def test_later_authority_or_unknown_errors_do_not_become_numerical_failure(reference_execution, monkeypatch, stage, error):
    recipe, _, _ = reference_execution
    target = {'test-inference': 'filter_switching', 'forecast': 'forecast_switching', 'evaluation': 'energy_score'}[stage]
    def stop(*args, **kwargs):
        raise error
    monkeypatch.setattr(pipeline, target, stop)
    with pytest.raises(type(error), match=str(error)):
        pipeline.execute_recipe(recipe, 2, 19, check_budget=fixture_poll,
            fixture_approximation_executor=lambda *a, **k: reference_execution[1]['fit']['approximation_checks'])
