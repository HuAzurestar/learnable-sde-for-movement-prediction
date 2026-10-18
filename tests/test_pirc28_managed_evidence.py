"""Actual authorized prefix collection and owner forgery/refusal controls, not SCI."""

import copy
from dataclasses import replace
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc28.computation import QualificationRunner, computation_plan, validate_computation_result
from experiments.pirc28 import managed_evidence
from infrastructure.research_store import ResearchError, digest
from tests import test_pirc28_evidence as evidence_tests
from tests.test_pirc28_managed_approximation import engineering_recipe
from tests.test_pirc28_qualification import spec as definition_spec
from tests.test_pirc28_synthetic import fixture_poll


@pytest.fixture(scope='module')
def computed_prefix(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evidence_tests, 'recipe', engineering_recipe)
        store, study, definition, grant, runner = evidence_tests.prepared(
            tmp_path_factory.mktemp('pirc28-prefix-owner') / 'fixture')
    cell = next(cell for cell in study['cells'] if cell['fit_K'] == 2)
    source = runner.run_cell(study['study_id'], digest(cell), budget=BudgetSpec(120))
    assert source['state'] == 'SUCCEEDED'
    before = BudgetLedger(store).balance('pirc28-k2')['committed_ms']
    result = QualificationRunner(store).run(definition, [study['study_id']], {study['study_id']: grant},
        budget=BudgetSpec(90, category='smoke'), allow_fixture=True)
    if result['state'] != 'SUCCEEDED':
        path = store.path / 'artifacts' / ('.attempt-'+result['attempt_id']) / 'worker.log'
        pytest.fail(str(result)+'\n'+path.read_text(errors='replace'))
    with store.lock():
        value = json.loads(store._verified_artifact_content(store._manifest('artifact-'+result['artifact_id'])))
    request = store.manifest('pirc28-qualification-request-'+result['attempt_id'])
    return store, study, definition, grant, cell, source, result, value, request, before


def test_real_funded_collection_recomputes_and_binds_prefix_without_promotion(computed_prefix):
    store, study, _, grant, cell, source, result, value, request, before = computed_prefix
    proof, = value['report']['cell_approximation_evidence']
    assert proof['artifact_id'] == source['artifact_id']
    assert proof['attempt_id'] == source['attempt_id'] and proof['cell_hash'] == digest(cell)
    assert proof['authorization_hash'] == digest(grant)
    assert proof['selection_scores_recomputed'] is proof['prefix_inputs_replayed'] is True
    assert proof['prefix_diagnostics_consistent'] is proof['authorized_native_source_and_cost_bound'] is True
    assert proof['full_generated_input_hashes_verified'] is True
    assert proof['reference_sampling_verified'] is True
    assert len(proof['generated_input_replay_hash']) == 64
    for name in ('managed_evidence_bound', 'qualified_identifiability', 'real_data_authorized'):
        assert proof[name] is False
    assert value['report']['state'] == 'FIXTURE_ONLY'
    assert value['report']['qualified_identifiability'] is False
    assert request['frozen']['resource_plan']['maximum_prefix_replay_elements'] > 0
    assert BudgetLedger(store).balance('pirc28-k2')['committed_ms'] > before
    receipt = store.manifest(result['receipt_id'])
    assert receipt['cost']['arm_id'] == 'pirc28-k2' and receipt['cost']['charged_ms'] > 0


def test_owner_never_repeats_numerical_replay(computed_prefix, monkeypatch):
    store, *_, value, request, _ = computed_prefix
    def forbidden(*args, **kwargs):
        pytest.fail('owner performed numerical replay outside qualification worker')
    for name in ('validate_selected_fit', 'replay_test_prefix', 'validate_prefix_numbers', 'validate_generated_inputs'):
        monkeypatch.setattr(managed_evidence, name, forbidden)
    validate_computation_result(store, value, request)


@pytest.mark.parametrize('damage', ['approved-master', 'check-count', 'check-state', 'check-hash',
                                  'lineage-hash', 'roster-hash', 'qualified-type'])
def test_owner_refuses_rehashed_aggregate_qualification_forgery(computed_prefix, damage):
    store, *_, original, request, _ = computed_prefix
    value = copy.deepcopy(original)
    approximation = value['report']['approximation_qualification']
    if damage == 'approved-master':
        approximation['approved_master_binding']['state'] = 'REGISTERED_MASTER_MATCHED'
    elif damage == 'check-count':
        approximation['selected_check_counts']['NUMERICAL_CHECKS_PASSED'] = 2048
    elif damage == 'check-state':
        approximation['cell_dispositions'][0]['check_state'] = 'NUMERICAL_CHECKS_PASSED'
    elif damage == 'check-hash':
        approximation['cell_dispositions'][0]['selected_check_hash'] = digest('borrowed-proof')
    elif damage == 'lineage-hash':
        approximation['complete_lineage_hash'] = digest('omit-failed-costs')
    elif damage == 'roster-hash':
        approximation['all_attempt_proof_roster_hash'] = digest('omit-missing-checks')
    else:
        approximation['qualified'] = 0
    assert digest(approximation) != digest(original['report']['approximation_qualification'])
    value['report_hash'] = digest(value['report'])
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
        validate_computation_result(store, value, request)


@pytest.mark.parametrize('damage', ['missing-list', 'missing-hash', 'missing-cell', 'duplicate-cell',
    'state', 'recipe', 'attempt', 'cost', 'admission', 'grant', 'model', 'prefix', 'report', 'seed',
    'sampling', 'qualified', 'native-source', 'selection', 'prefix-type', 'unknown', 'runtime-root',
    'generated-hash', 'generated-type'])
def test_owner_refuses_rehashed_prefix_roster_or_identity_forgery(computed_prefix, damage):
    store, *_, original, request, _ = computed_prefix
    value = copy.deepcopy(original)
    report = value['report']
    proofs = report['cell_approximation_evidence']
    proof = proofs[0]
    if damage == 'missing-list':
        report.pop('cell_approximation_evidence')
    elif damage == 'missing-hash':
        report.pop('cell_approximation_evidence_hash')
    elif damage == 'missing-cell':
        proofs.clear()
    elif damage == 'duplicate-cell':
        proofs.append(copy.deepcopy(proof))
    elif damage == 'runtime-root':
        proof['runtime_binding']['root'] += '-fresh-budget-store'
    elif damage == 'unknown':
        proof['scientific'] = True
    else:
        field = {'state': 'state', 'recipe': 'recipe_cell_hash', 'attempt': 'attempt_id',
            'cost': 'all_attempt_costs_hash', 'admission': 'admission_hash', 'grant': 'authorization_hash',
            'model': 'selected_model_hash', 'prefix': 'observation_prefix_hash', 'report': 'reference_report_hash',
            'seed': 'reference_seed', 'sampling': 'reference_sampling_verified', 'qualified': 'qualified_identifiability',
            'native-source': 'authorized_native_source_and_cost_bound', 'selection': 'selection_scores_recomputed',
            'prefix-type': 'prefix_inputs_replayed', 'generated-hash': 'generated_input_replay_hash',
            'generated-type': 'full_generated_input_hashes_verified'}[damage]
        proof[field] = 0 if damage in ('prefix-type', 'generated-type') else False if damage in ('native-source', 'selection', 'sampling') \
            else True if damage in ('sampling', 'qualified') else 0 if damage == 'seed' else 'forged'
    if damage != 'missing-hash':
        report['cell_approximation_evidence_hash'] = digest(report.get('cell_approximation_evidence'))
    value['report_hash'] = digest(report)
    with pytest.raises(ResearchError):
        validate_computation_result(store, value, request)


def test_source_guard_refuses_unauthorized_or_substituted_bytes_before_numerics(computed_prefix, monkeypatch):
    store, study, _, grant, cell, source, _, value, _, _ = computed_prefix
    attempt = store.attempts()[source['attempt_id']]
    metadata = store.manifest('artifact-'+source['artifact_id'])
    payload = json.loads(store.read_artifact(source['artifact_id'], purpose='evaluate', authorization=grant))
    history = next(item['attempts'] for item in value['report']['lineage']
                   if item['selected_attempt_id'] == source['attempt_id'])
    monkeypatch.setattr(managed_evidence, 'validate_selected_fit', lambda *a, **k: pytest.fail('numeric bypass'))
    bad = copy.deepcopy(grant)
    bad['purposes'] = ['preview']
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        managed_evidence.check_cell_prefix(store, study, cell, attempt, payload, metadata, bad, history,
                                          check_budget=fixture_poll)
    payload['fit']['model']['a'][0][0] += 1
    with pytest.raises(ResearchError, match='projection changed'):
        managed_evidence.check_cell_prefix(store, study, cell, attempt, payload, metadata, grant, history,
                                          check_budget=fixture_poll)


@pytest.mark.parametrize('damage', ['missing-stop', 'wrong-stop-cost', 'wrong-stop-confirmation'])
def test_success_prefix_owner_requires_actual_native_stop(computed_prefix, monkeypatch, damage):
    store, *_, original, request, _ = computed_prefix
    source_attempt = original['report']['cell_approximation_evidence'][0]['attempt_id']
    events = copy.deepcopy(store.events())
    if damage == 'missing-stop':
        events = [event for event in events if not (event['event_kind'] == 'WORKER_TREE_STOPPED'
                  and event['payload'].get('attempt_id') == source_attempt)]
    else:
        stop = next(event['payload'] for event in events if event['event_kind'] == 'WORKER_TREE_STOPPED'
                    and event['payload'].get('attempt_id') == source_attempt)
        if damage == 'wrong-stop-cost':
            stop['observed_elapsed_ms'] += 1
        else:
            stop['confirmation'] = 'worker self-report'
    monkeypatch.setattr(store, 'events', lambda: events)
    with pytest.raises(ResearchError, match='native whole-tree'):
        validate_computation_result(store, original, request)


def test_additive_replay_preflight_is_frozen_before_budget_and_keeps_original_caps():
    definition = definition_spec(2)
    baseline = computation_plan(definition, 512*1024*1024)
    value = computation_plan(definition, 512*1024*1024, source_recipes=[engineering_recipe()])
    assert baseline['maximum_prefix_replay_elements'] == 0 < value['maximum_prefix_replay_elements'] < 20_000_000
    assert value['maximum_bootstrap_elements'] == baseline['maximum_bootstrap_elements']
    assert value['maximum_combined_numeric_bytes'] == 8*(value['maximum_bootstrap_elements']
                                                       +value['maximum_prefix_replay_elements'])
    assert value['maximum_combined_numeric_bytes'] <= 512*1024*1024
    definition = replace(definition_spec(410), bootstrap_draws=10000)
    assert computation_plan(definition, 512*1024*1024)['maximum_bootstrap_elements'] < 50_000_000
    with pytest.raises(ResearchError, match='additive bootstrap/prefix'):
        computation_plan(definition, 512*1024*1024, source_recipes=[engineering_recipe()])
