"""Native pilot-path engineering controls with simulated operator provenance.

Disposable stores only: this is NOT production scientific qualification.
Actual worker, sampler, canonical artifacts, admission and costs are required.
"""

import copy
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.runner import SharedRunner
from experiments.pirc28.evidence import collect_qualification
from experiments.pirc28.reference_execution import SOURCE_SCHEMA, validate_reference_execution
from experiments.pirc28.runtime import registries
from infrastructure.research_store import ResearchError, digest
from tests import test_pirc28_study as study_tests
from tests.test_pirc28_approximation import v2_master
from tests.test_pirc28_synthetic import fixture_poll


@pytest.fixture(scope='module')
def native_pilot(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study_tests, 'definition_value', v2_master)
        store, master, grants, _, registrations = study_tests.prepared(tmp_path_factory.mktemp('pilot-reference')/'store')
    study = store.manifest('study-'+registrations[0]['study_id'])['spec']
    cell = next(cell for cell in study['cells'] if cell['fit_K'] == 2)
    registry, recovery = registries()
    result = SharedRunner(store, registry, recovery).run_cell(study['study_id'], digest(cell),
        budget=BudgetSpec(120, category='pilot'))
    if result['state'] not in ('SUCCEEDED', 'FAILED') or result.get('artifact_id') is None:
        logs = list((store.path/'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(result)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))
    attempt = store.attempts()[result['attempt_id']]
    grant = grants[study['study_id']]
    payload = json.loads(store.read_artifact(result['artifact_id'], purpose='evaluate', authorization=grant))
    body = payload['fit'] if payload['status'] == 'SUCCEEDED' else payload['failure']
    record = body['reference_execution']
    if payload['status'] == 'SUCCEEDED':
        model = body['model']
    else:
        chosen = body['selection']['selected_initialization']
        model = next(fit['model'] for fit in body['initializations'] if fit['initialization_id'] == chosen)
    return store, master, grants, study, cell, attempt, payload, record, model, body['approximation_checks']


def test_actual_original_pilot_sampling_retains_failure_or_success_without_qualification(native_pilot):
    store, master, _, study, cell, attempt, payload, record, model, report = native_pilot
    assert record['schema_version'] == SOURCE_SCHEMA
    assert record['source_preflight']['study_definition_hash'] == master.definition_hash
    assert record['source_preflight']['admission_hash'] == payload['admission_hash']
    assert validate_reference_execution(record, study, cell, attempt, model, report,
        admission_hash=payload['admission_hash']) is True
    assert record['complete'] is True
    assert len(record['completed_batches']) == study['switching_recipe']['test_paths']*8
    assert record['qualified_identifiability'] is record['managed_evidence_bound'] is False
    assert record['real_data_authorized'] is False
    events = [event for event in store.events() if event['payload'].get('attempt_id') == attempt['attempt_id']]
    assert sum(event['event_kind'] == 'WORKER_STARTED' for event in events) == 1
    assert sum(event['event_kind'] == 'WORKER_TREE_STOPPED' for event in events) == 1
    costs = [event['payload'] for event in events if event['event_kind'] == 'SETTLE']
    assert len(costs) == 1 and costs[0]['arm_id'] == cell['arm_id']
    assert costs[0]['charged_ms'] > 0 and costs[0]['reserved_ms'] == 120_000
    assert BudgetLedger(store).balance(cell['arm_id'])['committed_ms'] == costs[0]['charged_ms']
    if payload['status'] == 'FAILED':
        assert payload['source_schema'] == 'pirc28-synthetic-failure-v3'
        assert payload['error_code'] == 'INFERENCE_INAPPLICABLE'
        assert 'metrics' not in payload


@pytest.mark.parametrize('damage', ['master', 'admission', 'attempt', 'scope', 'missing-preflight'])
def test_borrowed_pilot_metadata_cannot_change_original_binding(native_pilot, damage):
    _, _, _, study, cell, attempt, payload, original, model, report = native_pilot
    record = copy.deepcopy(original)
    if damage == 'scope':
        record['scope'] = 'fixture-native-execution-only; not-scientific-qualification'
    elif damage == 'missing-preflight':
        record.pop('source_preflight')
    else:
        field = {'master': 'study_definition_hash', 'admission': 'admission_hash', 'attempt': 'attempt_id'}[damage]
        record['source_preflight'][field] = '0'*(32 if field == 'attempt_id' else 64)
        record['source_preflight_hash'] = digest(record['source_preflight'])
    with pytest.raises(ResearchError):
        validate_reference_execution(record, study, cell, attempt, model, report,
            admission_hash=payload['admission_hash'])


def test_authorized_pilot_collection_keeps_missing_cells_and_scientific_withholding(native_pilot):
    store, master, grants, study, _, attempt, _, record, _, _ = native_pilot
    from experiments.pirc28.qualification import IdentifiabilitySpec
    definition = IdentifiabilitySpec.from_dict(master.to_dict()['identifiability_spec'])
    result = collect_qualification(store, definition, master.study_ids, grants, check_budget=fixture_poll)
    receipts = result['cell_approximation_evidence']
    receipt = next(value for value in receipts if value['attempt_id'] == attempt['attempt_id'])
    assert receipt['reference_execution_hash'] == digest(record)
    assert receipt['reference_sampling_verified'] is True
    assert result['qualified_identifiability'] is False
    assert result['real_data_authorized'] is False
    assert result['study_definition_hash'] == master.definition_hash
    assert len(result['rows']) == len(master.to_dict()['recipes'])*len(definition.replicate_ids)*2
    assert any(row['state'] == 'NOT_RUN' for row in result['rows'])
    assert receipt['all_attempt_costs_hash']
