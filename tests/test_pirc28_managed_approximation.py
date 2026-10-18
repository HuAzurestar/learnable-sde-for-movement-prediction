"""Actual reference execution in disposable engineering stores, never SCI evidence."""

import copy
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc28.approximation_checks import check_prefix_approximation as check_prefix
from experiments.pirc28.approximation_evidence import reference_seed
from experiments.pirc28.failures import NumericalCellFailure
from experiments.pirc28.evidence import collect_qualification
from experiments.pirc28.failure_artifacts import failure_result, validate_cell_failure
from experiments.pirc28.pipeline import execute_recipe
from experiments.pirc28.runtime import switching_plugin
from infrastructure.research_store import ResearchError, digest, encode
from models.switching import SwitchingError
from tests import test_pirc28_evidence as evidence_tests
from tests.test_pirc28_approximation import v2_recipe
from tests.test_pirc28_synthetic import fixture_poll


def engineering_recipe(*, distinguishable=False):
    value = v2_recipe()
    # Explicit disposable fixture design; production plan bytes remain unchanged.
    value['approximation_spec']['event_driven_reference']['particles'] = 1024
    if not distinguishable:
        for model in [value['scenario']['model'], *value['initializations']['2']]:
            model['a'] = [[0.0, 0.0], [0.0, 0.0]]
    return value


def executed(value, seed=19, **options):
    return execute_recipe(value, 2, seed, check_budget=fixture_poll,
                          fixture_approximation_executor=check_prefix, **options)


@pytest.fixture(scope='module')
def reference_execution():
    value = engineering_recipe()
    snapshots = []
    return value, executed(value, progress_callback=snapshots.append), snapshots


def test_actual_reference_is_consistent_without_promoting_authority(reference_execution):
    value, result, _ = reference_execution
    fit = result['fit']
    report = fit['approximation_checks']
    assert report['state'] == 'NUMERICAL_CHECKS_PASSED'
    assert report['reference_seed'] == reference_seed(value, 19)
    assert report['model_hash'] == digest(fit['model'])
    assert len(report['paths']) == value['test_paths']
    assert all(path['reference_batches'] == 8 for path in report['paths'])
    assert fit['approximation_consistency']['numerical_diagnostics_consistent'] is True
    for document in (report, fit['approximation_consistency']):
        assert document['managed_evidence_bound'] is False
        assert document['qualified_identifiability'] is False
        assert document['real_data_authorized'] is False
    assert fit['approximation_execution_scope'] == 'engineering-only; not-managed-authority-proof'


def test_complete_report_and_two_owner_copies_are_additively_declared(reference_execution):
    report = reference_execution[1]['fit']['approximation_checks']
    def numeric_slots(value):
        if type(value) in {int, float}:
            return 1
        if type(value) is dict:
            return sum(numeric_slots(item) for item in value.values())
        if type(value) is list:
            return sum(numeric_slots(item) for item in value)
        return 0
    plan = report['resource_plan']
    count = numeric_slots(report)
    assert 0 < count <= plan['retained_report_numeric_elements']
    assert plan['owner_report_copy_elements'] == 2 * plan['retained_report_numeric_elements']
    assert plan['tensor_elements'] > 3 * count


def test_completed_fit_resume_restarts_full_fixed_reference(reference_execution):
    value, expected, snapshots = reference_execution
    saved = next(state for state in reversed(snapshots)
                 if all(item is not None for item in state['fit_states']))
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs['seed'])
        return check_prefix(*args, **kwargs)
    actual = execute_recipe(value, 2, 19, check_budget=fixture_poll, resume_state=saved,
                           fixture_approximation_executor=run)
    assert calls == [reference_seed(value, 19)]
    assert actual == expected


@pytest.mark.parametrize('exception', [TimeoutError('shared fuse'), RuntimeError('unknown reference failure')])
def test_reference_authority_and_unknown_failures_are_not_numerical_results(exception):
    def refuse(*args, **kwargs):
        raise exception
    with pytest.raises(type(exception), match=str(exception)):
        execute_recipe(engineering_recipe(), 2, 19, check_budget=fixture_poll,
                       fixture_approximation_executor=refuse)


def test_hook_is_not_a_v1_or_scientific_default_bypass():
    value = engineering_recipe()
    with pytest.raises(ResearchError, match='NEEDS_IMPLEMENTATION'):
        execute_recipe(value, 2, 19, check_budget=fixture_poll)
    value['schema_version'] = 'pirc28-synthetic-recipe-v1'
    value.pop('approximation_spec')
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
        executed(value)


def test_k1_does_not_spend_k2_reference_allocation():
    def forbidden(*args, **kwargs):
        pytest.fail('K1 ran the K2 reference')
    result = execute_recipe(engineering_recipe(), 1, 19, check_budget=fixture_poll,
                            fixture_approximation_executor=forbidden)
    assert result['fit']['approximation_checks'] is None


def test_actual_managed_original_arm_reference_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_tests, 'recipe', engineering_recipe)
    store, value, definition, grant, runner = evidence_tests.prepared(tmp_path / 'fixture')
    cell = next(item for item in value['cells'] if item['fit_K'] == 2)
    result = runner.run_cell(value['study_id'], digest(cell), budget=BudgetSpec(120))
    if result['state'] != 'SUCCEEDED':
        logs = list((store.path / 'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(result)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))
    payload = json.loads((store.path / 'artifacts' / result['artifact_id']).read_bytes())
    assert payload['qualification'] == 'fixture'
    assert payload['fit']['approximation_checks']['state'] == 'NUMERICAL_CHECKS_PASSED'
    assert payload['fit']['approximation_consistency']['managed_evidence_bound'] is False
    events = store.events()
    reservations = [event for event in events if event['event_kind'] == 'RESERVE']
    assert len(reservations) == 1
    assert reservations[0]['payload']['arm_id'] == 'pirc28-k2'
    previous = BudgetLedger(store).balance('pirc28-k2')['committed_ms']
    assert previous > 0
    reused = runner.run_cell(value['study_id'], digest(cell), budget=BudgetSpec(120))
    assert reused['artifact_id'] == result['artifact_id']
    assert BudgetLedger(store).balance('pirc28-k2')['committed_ms'] == previous
    assert len(store.attempts()) == 1
    collected = collect_qualification(store, definition, [value['study_id']],
        {value['study_id']: grant}, check_budget=fixture_poll, allow_fixture=True)
    assert collected['state'] == 'FIXTURE_ONLY'
    assert collected['qualified_identifiability'] is collected['real_data_authorized'] is False


def test_actual_managed_failed_reference_retains_original_cost_and_all_batches(tmp_path, monkeypatch):
    value = engineering_recipe(distinguishable=True)
    value['approximation_spec']['event_driven_reference']['minimum_ess_fraction'] = 1.0
    monkeypatch.setattr(evidence_tests, 'recipe', lambda: copy.deepcopy(value))
    store, study, definition, grant, runner = evidence_tests.prepared(tmp_path / 'failed-reference')
    cell = next(item for item in study['cells'] if item['fit_K'] == 2)
    result = runner.run_cell(study['study_id'], digest(cell), budget=BudgetSpec(120))
    assert result['state'] == 'FAILED' and result.get('artifact_id') is not None
    payload = json.loads(store.read_artifact(result['artifact_id'], purpose='evaluate', authorization=grant))
    assert payload['source_schema'] == 'pirc28-synthetic-failure-v3'
    assert payload['failure']['reference_execution']['complete'] is True
    assert 'metrics' not in payload and 'forecast' not in payload
    report = payload['failure']['approximation_checks']
    assert report['state'] == 'INFERENCE_INAPPLICABLE'
    assert all(path['reference_batches'] == 8 for path in report['paths'])
    assert BudgetLedger(store).balance('pirc28-k2')['committed_ms'] > 0
    assert len([event for event in store.events() if event['event_kind'] == 'RESERVE']) == 1
    collected = collect_qualification(store, definition, [study['study_id']],
        {study['study_id']: grant}, check_budget=fixture_poll, allow_fixture=True)
    lineage = next(item for item in collected['lineage'] if item['selected_attempt_id'] == result['attempt_id'])
    summary = lineage['attempts'][0]['failure']
    # V2 aggregate disclosure is bounded; the complete failed reference and
    # every original batch remain in the authorized canonical source artifact.
    assert summary['schema_version'] == 'pirc28-bounded-failure-disclosure-v1'
    assert summary['source_artifact_id'] == result['artifact_id'] == digest(payload)
    assert summary['source_schema'] == payload['source_schema']
    assert summary['complete_failure_hash'] == digest(payload['failure'])
    for name in ('stage', 'error_code', 'recipe_hash', 'generated_inputs'):
        assert summary[name] == payload['failure'][name]
    assert summary['selection_hash'] == digest(payload['failure']['selection'])
    assert summary['native_execution_hashes']['reference_execution'] == digest(
        payload['failure']['reference_execution'])
    disclosed = summary['reports']['approximation_checks']
    assert disclosed['report_hash'] == digest(report)
    assert disclosed['state'] == report['state'] == 'INFERENCE_INAPPLICABLE'
    assert len(disclosed['paths']) == len(report['paths'])
    omitted = {'coarse_summary', 'fine_summary', 'reference_summary', 'reference_batch_summaries'}
    for compact, original in zip(disclosed['paths'], report['paths']):
        assert compact['complete_source_path_hash'] == digest(original)
        assert compact['reference_batches'] == original['reference_batches'] == 8
        assert not omitted.intersection(compact)
        assert {key: item for key, item in compact.items() if key != 'complete_source_path_hash'} == {
            key: item for key, item in original.items() if key not in omitted}
    assert len(encode(summary)) <= 64 * 1024
    assert summary['successful_metrics_available'] is False
    assert summary['qualified_identifiability'] is summary['real_data_authorized'] is False
    assert json.loads(store.read_artifact(result['artifact_id'], purpose='evaluate', authorization=grant)) == payload
    assert collected['state'] == 'FIXTURE_ONLY'


def test_actual_pilot_admission_cannot_enable_engineering_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_tests, 'recipe', engineering_recipe)
    store, study, _, _, runner = evidence_tests.prepared(tmp_path / 'pilot', pilot=True)
    cell = next(item for item in study['cells'] if item['fit_K'] == 2)
    result = runner.run_cell(study['study_id'], digest(cell), budget=BudgetSpec(45))
    assert result['state'] == 'FAILED' and result.get('artifact_id') is None
    logs = list((store.path / 'artifacts').glob('.attempt-*/worker.log'))
    assert any('NEEDS_PREREGISTRATION: complete pre-fit master required' in path.read_text(errors='replace') for path in logs)
    assert BudgetLedger(store).balance('pirc28-k2')['committed_ms'] > 0


def test_actual_failed_reference_keeps_diagnostics_and_bound_selected_model():
    value = engineering_recipe(distinguishable=True)
    # Frozen beforehand: any nonuniform particle weights fail this engineering gate.
    value['approximation_spec']['event_driven_reference']['minimum_ess_fraction'] = 1.0
    with pytest.raises(NumericalCellFailure) as caught:
        executed(value)
    diagnostic = caught.value.diagnostic
    assert diagnostic['stage'] == 'approximation'
    assert diagnostic['approximation_checks']['state'] == 'INFERENCE_INAPPLICABLE'
    spec = {'switching_recipe': value, 'protocol_hash': 'p', 'data_hash': 'd',
            'admission': {'mode': 'fixture'}}
    cell = {'fit_K': 2, 'seed': 19}
    result = failure_result(diagnostic, spec, cell, switching_plugin())
    validate_cell_failure(result, spec, cell)
    for name in ('model_hash', 'inference_hash', 'initial_state_hash'):
        damaged = copy.deepcopy(result)
        damaged['failure']['approximation_checks'][name] = '0' * 64
        with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
            validate_cell_failure(damaged, spec, cell)


@pytest.mark.parametrize('exception', [SwitchingError('NUMERICAL_FAILURE', 'reference fixture'),
                                     FloatingPointError('reference fixture')])
def test_typed_reference_exception_has_no_invented_partial_report(exception):
    value = engineering_recipe()
    def fail(*args, **kwargs):
        raise exception
    with pytest.raises(NumericalCellFailure) as caught:
        execute_recipe(value, 2, 19, check_budget=fixture_poll, fixture_approximation_executor=fail)
    diagnostic = caught.value.diagnostic
    assert diagnostic['stage'] == 'approximation'
    assert diagnostic['approximation_checks'] is None
    spec = {'switching_recipe': value, 'protocol_hash': 'p', 'data_hash': 'd',
            'admission': {'mode': 'fixture'}}
    cell = {'fit_K': 2, 'seed': 19}
    validate_cell_failure(failure_result(diagnostic, spec, cell, switching_plugin()), spec, cell)
