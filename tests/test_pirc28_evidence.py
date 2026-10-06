"""Disposable synthetic stores/attestations; not a production research study."""

from dataclasses import replace
import copy
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.runner import SharedRunner
from experiments.pirc28.evidence import SELECTION_RULE, _success_row, collect_qualification
from experiments.pirc28.pipeline import recipe_resources
from experiments.pirc28.qualification import phase_manifest
from experiments.pirc28.runtime import bind_recipe_execution, registries, resume_command, switching_plugin
from experiments.pirc28.synthetic import ScenarioSpec
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_pirc28_qualification import spec as definition_spec
from tests.test_pirc28_runtime import recipe
from tests.test_pirc28_synthetic import fixture_poll
from tests.test_research_store import spec


def prepared(root, *, pilot=False, explicit_scope=False):
    root.mkdir()
    store = ResearchStore(root, 'pirc28-evidence-test-only', initialize=True)
    value = spec()
    value['study_id'] = 'pirc28-evidence-fixture'
    value['runtime_binding'] = {'root': str(root), 'store_id': store.store_id}
    value['switching_recipe'] = recipe()
    original = definition_spec(2)
    matrix = None
    if pilot:
        matrix = phase_manifest(ScenarioSpec.from_dict(value['switching_recipe']['scenario']),
            {name: [point] for name, point in original.scenario_axes['scenario'].items()})
        value['phase_matrix_hash'] = store.publish('pirc28-phase-matrix-'+digest(matrix), matrix)
        value['switching_recipe']['scenario'] = matrix['scenarios'][0]['scenario']
    identifier = value['switching_recipe']['scenario']['scenario_id']
    definition = replace(original, scenario_axes={identifier: original.scenario_axes['scenario']})
    if explicit_scope:
        definition = replace(definition, scenario_roles={identifier: 'required-qualified'})
    value['switching_recipe']['identifiability_spec'] = definition.to_dict()
    value['arms'] = [{'arm_id': 'pirc28-k'+str(k), 'model_family_id': 'pirc28-affine-k'+str(k),
        'method_family_id': 'pirc28-approximate-gem', 'objective_id': 'position-energy',
        'budget_seconds': 86400} for k in (1, 2)]
    value['cells'] = [{'arm_id': 'pirc28-k'+str(k), 'fit_K': k, 'replicate_id': replicate,
        'seed': 19+index, 'block_id': replicate} for index, replicate in enumerate(definition.replicate_ids) for k in (1, 2)]
    bind_recipe_execution(value)
    grant = admit_fixture(store, value, switching_plugin(), root,
        execution_config=lambda cell: recipe_resources(value['switching_recipe'], cell['fit_K'])[0],
        execution_inputs=lambda cell: recipe_resources(value['switching_recipe'], cell['fit_K'])[1],
        recovery_command_builder=resume_command, pilot=pilot,
        source_content=encode(value['switching_recipe']) if pilot else None,
        dataset_id='synthetic-pirc28' if pilot else 'synthetic',
        preregistration_extra={'pirc28_identifiability_spec_hash': digest(definition.to_dict()),
                               'pirc28_phase_matrix_hash': None if matrix is None else digest(matrix),
                               'primary_metrics': ['position_energy_score_d2'],
                               'result_selection_rule': SELECTION_RULE})
    store.register(value, digest(value))
    registry, recovery = registries()
    return store, value, definition, grant, SharedRunner(store, registry, recovery)


@pytest.fixture(scope='module')
def completed_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp('pirc28-evidence') / 'fixture'
    store, value, definition, grant, runner = prepared(root)
    # Explicit owner-side failure fixture, not a claim of an actual crash.
    cell = value['cells'][0]
    run_id = store.register_run(value['study_id'], cell)
    parent = store.new_attempt(run_id)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(parent, BudgetSpec(45))
    store.transition(parent, 'RUNNING')
    ledger.settle(reservation['reservation_id'], 2000, outcome='FAILED')
    store.transition(parent, 'FAILED', error_code='OWNER_TEST_INTERRUPTION')
    results = []
    for index, cell in enumerate(value['cells'][:2]):
        result = runner.run_cell(value['study_id'], digest(cell), budget=BudgetSpec(45),
            parent_attempt_id=parent if index == 0 else None,
            reason='explicit qualification fixture retry' if index == 0 else None)
        if result['state'] != 'SUCCEEDED':
            logs = list((store.path / 'artifacts').glob('.attempt-*/worker.log'))
            pytest.fail(str(result)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))
        results.append(result)
    return store, value, definition, grant, results


def collect(data, **options):
    store, value, definition, grant, _ = data
    return collect_qualification(store, definition, [value['study_id']], {value['study_id']: grant},
                                check_budget=fixture_poll, allow_fixture=True, **options)


def test_real_worker_results_collect_with_failures_and_costs_but_stay_fixture(completed_pair):
    store, _, _, _, results = completed_pair
    before = {k: BudgetLedger(store).balance('pirc28-k'+str(k)) for k in (1, 2)}
    result = collect(completed_pair)
    assert result['state'] == 'FIXTURE_ONLY'
    assert result['gate']['state'] == 'INCOMPLETE'
    assert result['qualified_identifiability'] is result['real_data_authorized'] is False
    assert len(result['rows']) == 4
    assert len([row for row in result['rows'] if row['state'] == 'SUCCEEDED']) == 2
    successes = [row for row in result['rows'] if row['state'] == 'SUCCEEDED']
    assert successes[0]['input_hash'] == successes[1]['input_hash']
    retried = next(row for row in result['lineage'] if row['K'] == 1 and row['replicate_id'] == 'rep-0')
    assert [attempt['state'] for attempt in retried['attempts']] == ['FAILED', 'SUCCEEDED']
    assert retried['attempts'][0]['cost']['charged_ms'] == 2000
    assert retried['selected_attempt_id'] == results[0]['attempt_id']
    assert {k: BudgetLedger(store).balance('pirc28-k'+str(k)) for k in (1, 2)} == before
    assert any(event['event_kind'] == 'READ_COMPLETED' and event['payload'].get('artifact_id') == results[0]['artifact_id'] for event in store.events())


def test_fixture_cannot_be_promoted_by_default(completed_pair):
    store, value, definition, grant, _ = completed_pair
    with pytest.raises(ResearchError, match='UNQUALIFIED'):
        collect_qualification(store, definition, [value['study_id']], {value['study_id']: grant}, check_budget=fixture_poll)


def test_collection_requires_actual_evaluate_grant_before_bytes(completed_pair):
    store, value, definition, grant, _ = completed_pair
    forged = {**grant, 'purposes': ['preview']}
    start = len(store.events())
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        collect_qualification(store, definition, [value['study_id']], {value['study_id']: forged},
            check_budget=fixture_poll, allow_fixture=True)
    recent = store.events()[start:]
    assert recent[-1]['event_kind'] in ('EXPOSURE_DENIED', 'DISCLOSURE_DENIED')
    assert not any(event['event_kind'] == 'READ_STARTED' for event in recent)


def test_collection_has_per_matrix_bytes_and_poll_refusals(completed_pair):
    with pytest.raises(ResearchError, match='RESOURCE_LIMIT'):
        collect(completed_pair, maximum_total_bytes=1)
    store, value, definition, grant, _ = completed_pair
    def refuse():
        raise RuntimeError('shared supervisor refusal')
    with pytest.raises(RuntimeError, match='supervisor refusal'):
        collect_qualification(store, definition, [value['study_id']], {value['study_id']: grant},
            check_budget=refuse, allow_fixture=True)


@pytest.mark.parametrize('damage', ['missing-settlement', 'wrong-arm', 'wrong-charge', 'unknown-cost', 'wrong-outcome'])
def test_cost_history_must_match_authoritative_attempts(completed_pair, monkeypatch, damage):
    store = completed_pair[0]
    events = copy.deepcopy(store.events())
    if damage == 'missing-settlement':
        events = [event for event in events if event['event_kind'] != 'SETTLE']
    else:
        cost = next(event for event in events if event['event_kind'] == 'SETTLE')['payload']
        if damage == 'wrong-arm':
            cost['arm_id'] = 'unrelated-arm'
        elif damage == 'wrong-charge':
            cost['charged_ms'] += 1
        elif damage == 'unknown-cost':
            cost['monotonic_elapsed_ms'] = None
        else:
            cost['outcome'] = 'SUCCEEDED'
    monkeypatch.setattr(store, 'events', lambda: events)
    with pytest.raises(ResearchError, match='UNQUALIFIED'):
        collect(completed_pair)


def test_legacy_authorized_reader_cannot_substitute_artifact_bytes(completed_pair, monkeypatch):
    store = completed_pair[0]
    monkeypatch.setattr(store, 'read_artifact', lambda *args, **kwargs: encode({'fake': 'result'}))
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        collect(completed_pair)


@pytest.mark.parametrize('damage', ['seed', 'missing-cell', 'same-arm'])
def test_registered_matrix_cannot_change_independence_or_family(completed_pair, monkeypatch, damage):
    store, value, _, _, _ = completed_pair
    resolve = store.manifest
    def changed(object_id):
        document = resolve(object_id)
        if object_id != 'study-'+value['study_id']:
            return document
        document = copy.deepcopy(document)
        study = document['spec']
        if damage == 'seed':
            study['cells'][2]['seed'] = study['cells'][0]['seed']
            study['cells'][3]['seed'] = study['cells'][1]['seed']
        elif damage == 'missing-cell':
            study['cells'].pop()
        else:
            for cell in study['cells']:
                cell['arm_id'] = study['arms'][0]['arm_id']
        document['spec_hash'] = digest(study)
        return document
    monkeypatch.setattr(store, 'manifest', changed)
    with pytest.raises(ResearchError, match='IDENTITY_CONFLICT|UNQUALIFIED'):
        collect(completed_pair)


@pytest.mark.parametrize('damage', ['definition', 'source-role', 'source-seed', 'metric', 'selected-fit', 'future-selection', 'missing-init'])
def test_source_row_refuses_unbound_domain_diagnostics(completed_pair, damage):
    store, value, definition, grant, results = completed_pair
    result = json.loads(store.read_artifact(results[1]['artifact_id'], purpose='evaluate', authorization=grant))
    if damage == 'definition':
        result['fit']['recovery_diagnostics']['identifiability_spec_hash'] = '0'*64
    elif damage == 'source-role':
        result['fit']['generated_inputs'][0]['role'] = 'test'
    elif damage == 'source-seed':
        result['fit']['generated_inputs'][0]['scenario_hash'] = '0'*64
    elif damage == 'metric':
        result['fit']['recovery_diagnostics']['mode_diagnostics'].pop('mode_ece')
    elif damage == 'selected-fit':
        result['fit']['selection']['selected_initialization'] = 'undeclared-init'
    elif damage == 'future-selection':
        result['fit']['selection']['split_role'] = 'test'
    else:
        result['fit']['initializations'].clear()
    with pytest.raises(ResearchError, match='IDENTITY_CONFLICT|UNQUALIFIED'):
        _success_row(definition, value, value['cells'][1], result, results[1]['artifact_id'])


def test_disclosure_rechecks_grant_after_extra_processing(completed_pair, monkeypatch):
    store = completed_pair[0]
    calls = []
    def denied(*args, **kwargs):
        calls.append(1)
        raise ResearchError('UNAUTHORIZED_DATA', 'grant expired after result processing')
    monkeypatch.setattr(store, 'verify_artifact_read', denied)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        collect(completed_pair)
    assert calls


@pytest.mark.parametrize('explicit_scope', [False, True])
def test_pilot_source_plan_is_validated_without_formal_blind_test_prerequisite(tmp_path, explicit_scope):
    store, value, definition, grant, runner = prepared(tmp_path / 'pilot-fixture', pilot=True, explicit_scope=explicit_scope)
    executed = runner.run_cell(value['study_id'], digest(value['cells'][1]), budget=BudgetSpec(45, category='pilot'))
    if executed['state'] != 'SUCCEEDED':
        logs = list((store.path / 'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(executed)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))
    result = collect_qualification(store, definition, [value['study_id']], {value['study_id']: grant}, check_budget=fixture_poll)
    assert result['state'] == 'INCOMPLETE'
    assert result['qualified_identifiability'] is result['real_data_authorized'] is False
    assert result['preregistration_hashes']
    assert any(row['state'] == 'SUCCEEDED' for row in result['rows'])
    if explicit_scope:
        assert result['gate']['scenario_roles'] == definition.scenario_roles
        assert result['gate']['applicable_ranges'] == []
        assert result['gate']['schema_version'] == 'pirc28-identifiability-gate-v2'


def test_changing_the_frozen_scope_definition_requires_new_preregistration(tmp_path):
    from experiments.pirc28.evidence import _preregistration
    store, study, definition, _, _ = prepared(tmp_path/'scope-identity', pilot=True)
    changed = replace(definition, scenario_roles={next(iter(definition.scenario_axes)): 'required-qualified'})
    assert digest(changed.to_dict()) != digest(definition.to_dict())
    with pytest.raises(ResearchError, match='NEEDS_PREREGISTRATION'):
        _preregistration(store, study, changed, study['cells'][0], store.events())


def test_empty_scientific_matrix_still_requires_scope_authorization(tmp_path):
    store, value, definition, grant, _ = prepared(tmp_path / 'empty-scope', pilot=True)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        collect_qualification(store, definition, [value['study_id']],
            {value['study_id']: {**grant, 'block_ids': []}}, check_budget=fixture_poll)


def test_source_plan_cannot_be_registered_after_an_attempt_is_admitted(tmp_path, monkeypatch):
    store, value, definition, grant, _ = prepared(tmp_path / 'late-plan', pilot=True)
    events = copy.deepcopy(store.events())
    publication = next(event for event in events if event['event_kind'] == 'MANIFEST'
        and event['payload'].get('object_id', '').startswith('preregistration-'))
    from experiments.pirc28.evidence import _run_id
    events.append({'event_kind': 'ADMISSION', 'sequence': publication['sequence']-1,
                   'payload': {'run_id': _run_id(value, value['cells'][0])}})
    monkeypatch.setattr(store, 'events', lambda: events)
    with pytest.raises(ResearchError, match='NEEDS_PREREGISTRATION'):
        collect_qualification(store, definition, [value['study_id']], {value['study_id']: grant}, check_budget=fixture_poll)


@pytest.mark.parametrize('damage', ['axis-source', 'recipe-source', 'missing-matrix'])
def test_scientific_axis_ranges_are_bound_to_reconstructed_sources(tmp_path, monkeypatch, damage):
    store, value, definition, grant, _ = prepared(tmp_path / 'physical-phase', pilot=True)
    resolve = store.manifest
    def changed(object_id):
        document = copy.deepcopy(resolve(object_id))
        if damage == 'axis-source' and object_id.startswith('pirc28-phase-matrix-'):
            document['axes']['observation_noise_m'] = [5.0]
        elif object_id == 'study-'+value['study_id']:
            if damage == 'missing-matrix':
                document['spec'].pop('phase_matrix_hash')
            elif damage == 'recipe-source':
                document['spec']['switching_recipe']['scenario']['model']['a'][0][0] += 2
            document['spec_hash'] = digest(document['spec'])
        return document
    monkeypatch.setattr(store, 'manifest', changed)
    with pytest.raises(ResearchError, match='UNQUALIFIED|NEEDS_PREREGISTRATION'):
        collect_qualification(store, definition, [value['study_id']], {value['study_id']: grant}, check_budget=fixture_poll)
