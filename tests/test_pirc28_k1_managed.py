"""Native original-arm K1 controls in disposable SIMULATED operator stores.

The tight-boundary negative control is a separate engineering definition, not
an alteration of the real scientific plan or a production qualification run.
"""

import copy
import json

import pytest
import numpy as np

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.runner import SharedRunner
from experiments.pirc25.affine import code_hash
from experiments.pirc28 import pipeline, managed_evidence
from experiments.pirc28.computation import QualificationRunner, validate_computation_result
from experiments.pirc28.failure_artifacts import failure_result, validate_cell_failure
from experiments.pirc28.failures import NumericalCellFailure
from experiments.pirc28.k1_execution import attach_failure, validate_execution
from experiments.pirc28.qualification import IdentifiabilitySpec
from experiments.pirc28.runtime import registries, switching_plugin
from infrastructure.research_store import ResearchError, digest
from models.switching import SwitchingError
from tests import test_pirc28_study as study_tests
from tests.test_pirc28_approximation import v2_master, v2_recipe
from tests.test_pirc28_synthetic import fixture_poll


@pytest.fixture(scope='module', params=['normal', 'tight-boundary'])
def native_k1(tmp_path_factory, request):
    kind = request.param
    def definition():
        value = v2_master()
        if kind == 'tight-boundary':
            for spec in [value['approximation_spec'], *(recipe['approximation_spec'] for recipe in value['recipes'].values())]:
                spec['known_mode_boundary'].update(absolute_tolerance=1e-30, relative_tolerance=1e-30)
        return value
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study_tests, 'definition_value', definition)
        store, master, grants, _, registrations = study_tests.prepared(
            tmp_path_factory.mktemp('native-k1-'+kind)/'store')
    study = store.manifest('study-'+registrations[0]['study_id'])['spec']
    cell = next(cell for cell in study['cells'] if cell['fit_K'] == 1)
    registry, recovery = registries()
    outcome = SharedRunner(store, registry, recovery).run_cell(study['study_id'], digest(cell),
        budget=BudgetSpec(120, category='pilot'))
    if outcome['state'] not in ('SUCCEEDED', 'FAILED') or not outcome.get('artifact_id'):
        logs = list((store.path/'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(outcome)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))
    attempt = store.attempts()[outcome['attempt_id']]
    payload = json.loads(store.read_artifact(outcome['artifact_id'], purpose='evaluate',
                                          authorization=grants[study['study_id']]))
    body = payload['fit'] if payload['status'] == 'SUCCEEDED' else payload['failure']
    if payload['status'] == 'SUCCEEDED':
        model = body['model']
    else:
        chosen = body['selection']['selected_initialization']
        model = next(value['model'] for value in body['initializations'] if value['initialization_id'] == chosen)
    return kind, store, master, grants, study, cell, attempt, payload, body, model


def test_actual_worker_success_and_numerical_failure_bound_to_original_k1(native_k1):
    kind, store, master, _, study, cell, attempt, payload, body, model = native_k1
    report, record = body['k1_boundary_checks'], body['k1_boundary_execution']
    assert payload['status'] == ('SUCCEEDED' if kind == 'normal' else 'FAILED')
    assert validate_execution(record, study, cell, attempt, model, report,
        admission_hash=payload['admission_hash']) is True
    assert record['source_preflight']['study_definition_hash'] == master.definition_hash
    assert record['source_preflight']['admission_hash'] == payload['admission_hash']
    assert record['reference_sampling_verified'] is record['qualified_identifiability'] is False
    assert report['resource_plan']['reference_particles'] == report['resource_plan']['independent_batches'] == 0
    events = [value for value in store.events() if value['payload'].get('attempt_id') == attempt['attempt_id']]
    assert sum(value['event_kind'] == 'WORKER_STARTED' for value in events) == 1
    assert sum(value['event_kind'] == 'WORKER_TREE_STOPPED' for value in events) == 1
    settlement, = [value['payload'] for value in events if value['event_kind'] == 'SETTLE']
    assert settlement['arm_id'] == cell['arm_id'] and settlement['reserved_ms'] == 120_000
    assert settlement['charged_ms'] > 0
    assert BudgetLedger(store).balance(cell['arm_id'])['committed_ms'] == settlement['charged_ms']
    if kind == 'tight-boundary':
        assert payload['source_schema'] == 'pirc28-synthetic-failure-v4'
        assert payload['failure']['stage'] == 'k1-boundary'
        assert report['state'] == 'INFERENCE_INAPPLICABLE'
        assert 'metrics' not in payload and 'forecast' not in payload
        validate_cell_failure(payload, study, cell)


@pytest.fixture(scope='module')
def funded_k1(native_k1):
    _, store, master, grants, *_ = native_k1
    definition = IdentifiabilitySpec.from_dict(master.to_dict()['identifiability_spec'])
    before = BudgetLedger(store).balance('pirc28-k2')['committed_ms']
    outcome = QualificationRunner(store).run(definition, master.study_ids, grants,
                                            budget=BudgetSpec(120, category='pilot'))
    if outcome['state'] != 'SUCCEEDED':
        log = store.path/'artifacts'/('.attempt-'+outcome['attempt_id'])/'worker.log'
        pytest.fail(str(outcome)+'\n'+log.read_text(errors='replace'))
    with store.lock():
        value = json.loads(store._verified_artifact_content(store._manifest('artifact-'+outcome['artifact_id'])))
    request = store.manifest('pirc28-qualification-request-'+outcome['attempt_id'])
    return native_k1, outcome, value, request, before


def test_actual_funded_collector_binds_complete_k1_inputs_selection_boundary_and_costs(funded_k1):
    native, outcome, value, _, before = funded_k1
    kind, store, master, _, _, _, attempt, _, body, _ = native
    report = value['report']
    receipt, = report['cell_approximation_evidence']
    assert receipt['attempt_id'] == attempt['attempt_id']
    assert receipt['k1_boundary_execution_hash'] == digest(body['k1_boundary_execution'])
    assert receipt['k1_boundary_verified'] is (kind == 'normal')
    assert receipt['full_generated_input_hashes_verified'] is receipt['selection_scores_recomputed'] is True
    assert receipt['prefix_inputs_replayed'] is receipt['prefix_diagnostics_consistent'] is True
    assert receipt['authorized_native_source_and_cost_bound'] is True
    assert receipt['reference_sampling_verified'] is False and receipt['reference_seed'] is None
    assert receipt['reference_execution_hash'] is None
    assert report['qualified_identifiability'] is report['real_data_authorized'] is False
    assert report['study_definition_hash'] == master.definition_hash
    assert any(row['state'] == 'NOT_RUN' for row in report['rows'])
    if kind == 'tight-boundary':
        row = next(row for row in report['rows'] if row['artifact_hash'] == attempt['artifact_id'])
        assert row['state'] == 'INFERENCE_INAPPLICABLE' and row['metrics'] is None
        assert any(history['failure']['stage'] == 'k1-boundary'
                   for lineage in report['lineage'] for history in lineage['attempts'] if 'failure' in history)
    cost = store.manifest(outcome['receipt_id'])['cost']
    assert cost['arm_id'] == 'pirc28-k2' and cost['charged_ms'] > 0
    assert BudgetLedger(store).balance('pirc28-k2')['committed_ms'] == before+cost['charged_ms']


def test_owner_repeats_only_bounded_original_metadata_not_numerics(funded_k1, monkeypatch):
    native, _, value, request, _ = funded_k1
    store = native[1]
    def forbidden(*args, **kwargs): pytest.fail('unfunded owner numerics')
    for name in ('validate_selected_fit', 'replay_test_prefix', 'validate_prefix_numbers', 'validate_generated_inputs'):
        monkeypatch.setattr(managed_evidence, name, forbidden)
    monkeypatch.setattr('experiments.pirc28.k1_boundary.validate_k1_boundary', forbidden)
    monkeypatch.setattr('experiments.pirc28.k1_boundary.check_k1_boundary', forbidden)
    validate_computation_result(store, value, request)


@pytest.mark.parametrize('field', ['attempt_id', 'admission_hash', 'authorization_hash',
    'all_attempt_costs_hash', 'k1_boundary_execution_hash', 'k1_boundary_verified',
    'reference_sampling_verified', 'full_generated_input_hashes_verified', 'reference_seed', 'qualified_identifiability'])
def test_owner_refuses_rehashed_k1_receipt_forgery(funded_k1, field):
    native, _, original, request, _ = funded_k1
    value = copy.deepcopy(original)
    proof = value['report']['cell_approximation_evidence'][0]
    proof[field] = not proof[field] if type(proof[field]) is bool else 'forged'
    value['report']['cell_approximation_evidence_hash'] = digest(value['report']['cell_approximation_evidence'])
    value['report_hash'] = digest(value['report'])
    with pytest.raises(ResearchError):
        validate_computation_result(native[1], value, request)


@pytest.mark.parametrize('damage', ['master', 'admission', 'attempt', 'scope', 'missing-preflight', 'model', 'code'])
def test_borrowed_original_k1_execution_metadata_refused(native_k1, damage):
    _, _, _, _, study, cell, attempt, payload, body, model = native_k1
    record = copy.deepcopy(body['k1_boundary_execution'])
    if damage in ('master', 'admission', 'attempt'):
        field = {'master': 'study_definition_hash', 'admission': 'admission_hash', 'attempt': 'attempt_id'}[damage]
        record['source_preflight'][field] = '0'*(32 if field == 'attempt_id' else 64)
        record['source_preflight_hash'] = digest(record['source_preflight'])
    elif damage == 'scope': record['scope'] = 'fixture-native-k1-boundary-only; not-scientific-qualification'
    elif damage == 'missing-preflight': record['source_preflight'] = None
    elif damage == 'model': record['selected_model_hash'] = '0'*64
    else: record['source_code_hash'] = '0'*64
    with pytest.raises(ResearchError):
        validate_execution(record, study, cell, attempt, model, body['k1_boundary_checks'],
                           admission_hash=payload['admission_hash'])


@pytest.fixture(scope='module')
def library_k1():
    recipe, snapshots = v2_recipe(), []
    def no_reference(*args, **kwargs): pytest.fail('K1 used MC reference executor')
    result = pipeline.execute_recipe(recipe, 1, 19, check_budget=fixture_poll,
        progress_callback=lambda state: snapshots.append(copy.deepcopy(state)),
        fixture_approximation_executor=no_reference)
    saved = next(value for value in reversed(snapshots) if all(
        fit is not None and fit['complete_reason'] is not None for fit in value['fit_states']))
    return recipe, result, saved, no_reference


@pytest.mark.parametrize('stage,target', [('test-inference', 'filter_switching'),
    ('forecast', 'forecast_switching'), ('evaluation', 'energy_score')])
def test_later_k1_numerical_failure_retains_report_not_success(library_k1, monkeypatch, tmp_path, stage, target):
    recipe, expected, saved, no_reference = library_k1
    def fail(*args, **kwargs): raise SwitchingError('NUMERICAL_FAILURE', 'engineering later failure')
    monkeypatch.setattr(pipeline, target, fail)
    with pytest.raises(NumericalCellFailure) as caught:
        pipeline.execute_recipe(recipe, 1, 19, check_budget=fixture_poll, resume_state=saved,
                                fixture_approximation_executor=no_reference)
    body = caught.value.diagnostic
    assert body['stage'] == stage and body['k1_boundary_checks'] == expected['fit']['k1_boundary_checks']
    assert 'k1_boundary_execution' not in body  # no native authority from library execution
    study = {'study_id': 'metadata-only-control', 'switching_recipe': recipe, 'protocol_hash': 'p',
        'data_hash': 'd', 'code_hash': code_hash(), 'runtime_binding': {'root': str(tmp_path), 'store_id': 'unit-only'},
        'admission': {'mode': 'fixture'}}
    cell, attempt = {'fit_K': 1, 'seed': 19}, {'attempt_id': 'a'*32, 'run_id': 'b'*64}
    attach_failure(body, study, cell, attempt)
    payload = failure_result(body, study, cell, switching_plugin())
    validate_cell_failure(payload, study, cell)
    assert payload['source_schema'] == 'pirc28-synthetic-failure-v4' and 'metrics' not in payload
    forged = copy.deepcopy(payload)
    forged['failure']['k1_boundary_checks'] = None
    with pytest.raises(ResearchError): validate_cell_failure(forged, study, cell)
    # No attempt/artifact is fabricated or published by this metadata-only control.


def test_k1_boundary_numerical_exception_retains_partial_failure(library_k1, monkeypatch):
    recipe, _, saved, no_reference = library_k1
    def fail(*args, **kwargs): raise np.linalg.LinAlgError('engineering singular innovation')
    monkeypatch.setattr('experiments.pirc28.k1_boundary.check_k1_boundary', fail)
    with pytest.raises(NumericalCellFailure) as caught:
        pipeline.execute_recipe(recipe, 1, 19, check_budget=fixture_poll, resume_state=saved,
                                fixture_approximation_executor=no_reference)
    assert caught.value.diagnostic['stage'] == 'k1-boundary'
    assert caught.value.diagnostic['k1_boundary_checks'] is None
    assert caught.value.diagnostic['cause']['error_code'] == 'NUMERICAL_FAILURE'


@pytest.mark.parametrize('error', [ResearchError('TIMEOUT', 'original deadline'),
                                 ResearchError('UNAUTHORIZED_DATA', 'original grant'), RuntimeError('unknown')])
def test_k1_boundary_authority_and_unknown_errors_never_become_numerical(library_k1, monkeypatch, error):
    recipe, _, saved, no_reference = library_k1
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr('experiments.pirc28.k1_boundary.check_k1_boundary', fail)
    with pytest.raises(type(error), match=str(error)):
        pipeline.execute_recipe(recipe, 1, 19, check_budget=fixture_poll, resume_state=saved,
                                fixture_approximation_executor=no_reference)


def test_missing_k1_boundary_executor_remains_refused_before_generation(monkeypatch):
    monkeypatch.setattr(pipeline, 'generate_scenario', lambda *a, **k: pytest.fail('early generation'))
    with pytest.raises(ResearchError, match='NEEDS_IMPLEMENTATION'):
        pipeline.execute_recipe(v2_recipe(), 1, 19, check_budget=fixture_poll)


@pytest.mark.parametrize('k', [1, 2])
def test_checkpoint_recovery_discloses_original_k1_or_k2_work_not_invented_sampling(k):
    from experiments.pirc28.worker import _diagnostic_recovery
    value = _diagnostic_recovery(k, 'running; whole-fixed-reference-restart')
    if k == 1:
        assert value['reference_phase'] == value['reference_recovery'] == 'not-applicable; K1-without-MC'
        assert value['k1_boundary_recovery'] == 'restart-complete-original-K1-boundary-on-each-original-arm-attempt'
    else:
        assert value['reference_phase'] == 'running; whole-fixed-reference-restart'
        assert value['reference_recovery'] == 'restart-all-fixed-batches-from-frozen-recipe-cell-seed; no-partial-reuse'
