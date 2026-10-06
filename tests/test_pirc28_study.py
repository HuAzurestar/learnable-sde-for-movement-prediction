"""Disposable scientific-definition controls; never production qualification."""

from dataclasses import replace, FrozenInstanceError
import copy
import hashlib
import json

import pytest

from application.research_budget import BudgetSpec
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc28.computation import QualificationRunner, validate_computation_result
from experiments.pirc28.evidence import collect_qualification
from experiments.pirc28.pipeline import recipe_resources
from experiments.pirc28.qualification import IdentifiabilitySpec, phase_manifest
from experiments.pirc28.runtime import registries, resume_command, switching_plugin
from experiments.pirc28.study import (SCHEMA, SyntheticStudyDefinition, arm_bindings,
                                     register_definition_studies, stage_definition)
from experiments.pirc28.synthetic import ScenarioSpec
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_pirc28_qualification import axes, spec
from tests.test_pirc28_runtime import recipe
from tests.test_pirc28_synthetic import fixture_poll


def definition_value():
    template = recipe()
    base = copy.deepcopy(template['scenario'])
    base['model']['L'][1] = copy.deepcopy(base['model']['L'][0])
    points = axes()
    points['mode_separation'] = [0, 1]
    matrix = phase_manifest(ScenarioSpec.from_dict(base), points)
    rows = {row['scenario']['scenario_id']: row for row in matrix['scenarios']}
    qualification = replace(spec(2), scenario_axes={name: row['axes'] for name, row in rows.items()},
        scenario_roles={name: 'required-unidentifiable' if row['axes']['mode_separation'] == 0 else 'required-qualified'
                        for name, row in rows.items()})
    recipes, seeds = {}, {}
    for index, (name, row) in enumerate(sorted(rows.items())):
        changed = copy.deepcopy(template)
        changed['scenario'] = row['scenario']
        second = copy.deepcopy(changed['initializations']['2'][0])
        second['a'][0][0] += 0.2
        changed['initializations']['2'].append(second)
        recipes[name] = changed
        seeds[name] = {replicate: 100+2*index+i for i, replicate in enumerate(qualification.replicate_ids)}
    return {'schema_version': SCHEMA, 'runtime_code_hash': code_hash(), 'phase_matrix': matrix,
        'identifiability_spec': qualification.to_dict(), 'recipes': recipes, 'replicate_seeds': seeds,
        'arms': arm_bindings('pirc28-k1', 'pirc28-k2'), 'forecast_origin_s': 1.0, 'forecast_horizons_s': [0.5, 1.0]}


def protocol_inputs(definition, binding):
    protocols = {}
    for value in definition.proposals(binding):
        protocols[value['study_id']] = {'schema_version': 'pirc25-data-protocol-v1',
            'protocol_id': 'input-'+value['study_id'], 'study_id': value['study_id'], 'blocks': [
                {'block_id': replicate, 'dataset_id': 'synthetic-pirc28', 'release_id': 'v1',
                 'source_block_id': replicate, 'sha256': hashlib.sha256(encode(value['switching_recipe'])).hexdigest(),
                 'path': 'plugin-input.bin', 'split_role': 'train', 'fit_scope': True}
                for replicate in sorted({cell['block_id'] for cell in value['cells']})]}
    return protocols


def prepared(root):
    root.mkdir()
    store = ResearchStore(root, 'pirc28-study-test-only', initialize=True)
    definition = SyntheticStudyDefinition.from_dict(definition_value())
    stage_definition(store, definition)
    binding = {'root': str(root), 'store_id': store.store_id}
    prototypes = definition.proposals(binding)
    protocols = protocol_inputs(definition, binding)
    plan = definition.preregistration_fields(protocols)
    grants, admissions = {}, {}
    for value in prototypes:
        source = root/('source-'+value['study_id'])
        source.mkdir()
        value['schema_version'] = store.SCHEMA
        grant = admit_fixture(store, value, switching_plugin(), source,
            execution_config=lambda cell: recipe_resources(value['switching_recipe'], cell['fit_K'])[0],
            execution_inputs=lambda cell: recipe_resources(value['switching_recipe'], cell['fit_K'])[1],
            recovery_command_builder=resume_command, pilot=True,
            source_content=encode(value['switching_recipe']), dataset_id='synthetic-pirc28',
            preregistration_extra=plan, protocol_id=protocols[value['study_id']]['protocol_id'],
            authorization_id='execute-'+value['study_id'])
        grants[value['study_id']], admissions[value['study_id']] = grant, value['admission']
    registered = register_definition_studies(store, definition, admissions)
    return store, definition, grants, admissions, registered


def test_definition_owns_canonical_nested_state_and_detached_proposals(tmp_path):
    value = definition_value()
    definition = SyntheticStudyDefinition.from_dict(value)
    original = definition.definition_hash
    assert original == digest(value)
    value['forecast_horizons_s'][0] = 0.25
    detached = definition.to_dict()
    detached['arms']['2']['budget_seconds'] = 999999
    assert definition.definition_hash == original
    assert definition.to_dict()['arms']['2']['budget_seconds'] == 86400
    with pytest.raises(FrozenInstanceError):
        definition.content = b'{}'
    binding = {'root': str(tmp_path), 'store_id': 'explicit-test-identity'}
    proposals = definition.proposals(binding)
    assert len(proposals) == 2
    assert all('schema_version' not in proposal and 'admission' not in proposal for proposal in proposals)
    assert proposals[0]['arms'] == proposals[1]['arms']
    proposals[0]['cells'][0]['seed'] = 999
    assert definition.proposals(binding)[0]['cells'][0]['seed'] != 999
    assert SyntheticStudyDefinition(definition.content).to_dict() == definition.to_dict()


@pytest.mark.parametrize('damage', ['missing-threshold', 'missing-control', 'bad-control-covariance',
    'duplicate-seed', 'missing-seed', 'renamed-family', 'duplicate-initialization', 'no-fit',
    'horizon-mismatch', 'physical-source', 'override', 'unknown-field', 'nonfinite'])
def test_invalid_or_unfrozen_definitions_are_refused(damage):
    value = definition_value()
    name = next(iter(value['recipes']))
    if damage == 'missing-threshold':
        value['identifiability_spec']['minimum_occupancy'] = None
    elif damage == 'missing-control':
        value['identifiability_spec']['scenario_roles'] = {name: 'required-qualified' for name in value['recipes']}
    elif damage == 'bad-control-covariance':
        base = value['phase_matrix']['baseline_scenario']
        base['model']['L'][1][0][0] += 0.2
        matrix = phase_manifest(ScenarioSpec.from_dict(base), value['phase_matrix']['axes'])
        value['phase_matrix'] = matrix
        for row in matrix['scenarios']:
            value['recipes'][row['scenario']['scenario_id']]['scenario'] = row['scenario']
    elif damage == 'duplicate-seed':
        value['replicate_seeds'][name]['rep-1'] = value['replicate_seeds'][name]['rep-0']
    elif damage == 'missing-seed':
        value['replicate_seeds'][name].pop('rep-1')
    elif damage == 'renamed-family':
        value['arms']['2']['method_family_id'] = 'new-budget-alias'
    elif damage == 'duplicate-initialization':
        value['recipes'][name]['initializations']['2'][1] = value['recipes'][name]['initializations']['2'][0]
    elif damage == 'no-fit':
        value['recipes'][name]['inference']['max_iterations'] = 0
    elif damage == 'horizon-mismatch':
        value['forecast_horizons_s'][0] = 0.25
    elif damage == 'physical-source':
        value['recipes'][name]['scenario']['times'][-1] += 0.1
    elif damage == 'override':
        value['recipes'][name]['identifiability_spec'] = value['identifiability_spec']
    elif damage == 'unknown-field':
        value['posthoc_seed_override'] = {}
    else:
        value['forecast_origin_s'] = float('nan')
    with pytest.raises(ResearchError):
        SyntheticStudyDefinition.from_dict(value)


def test_stage_only_metadata_has_no_study_grant_attempt_or_reservation(tmp_path):
    root = tmp_path/'metadata-only'
    root.mkdir()
    store = ResearchStore(root, 'pirc28-stage-test-only', initialize=True)
    definition = SyntheticStudyDefinition.from_dict(definition_value())
    result = stage_definition(store, definition)
    assert result['registered_studies'] is result['real_data_authorized'] is False
    assert store.manifest('pirc28-study-definition-'+definition.definition_hash) == definition.to_dict()
    assert not store.attempts()
    assert not any(event['event_kind'] in ('RESERVE', 'ADMISSION', 'WORKER_STARTED') for event in store.events())
    assert not any(event['event_kind'] == 'MANIFEST' and event['payload']['object_id'].startswith(('study-', 'authorization-'))
                   for event in store.events())


def test_preregistration_freezes_complete_physical_source_matrix(tmp_path):
    definition = SyntheticStudyDefinition.from_dict(definition_value())
    protocols = protocol_inputs(definition, {'root': str(tmp_path), 'store_id': 'test-only'})
    fields = definition.preregistration_fields(protocols)
    assert fields['study_ids'] == definition.study_ids
    assert fields['pirc28_study_definition_hash'] == definition.definition_hash
    assert len(fields['protocol_bindings']) == 2
    changed = copy.deepcopy(protocols)
    next(iter(changed.values()))['blocks'][0]['sha256'] = digest('another recipe')
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        definition.preregistration_fields(changed)
    changed.pop(next(iter(changed)))
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
        definition.preregistration_fields(changed)


def test_existing_operator_inputs_register_complete_matrix_without_budget(tmp_path):
    store, definition, grants, _, registered = prepared(tmp_path/'registration')
    assert {record['study_id'] for record in registered} == set(definition.study_ids)
    assert not store.attempts()
    assert not any(event['event_kind'] == 'RESERVE' for event in store.events())
    qualification = IdentifiabilitySpec.from_dict(definition.to_dict()['identifiability_spec'])
    result = collect_qualification(store, qualification, definition.study_ids, grants, check_budget=fixture_poll)
    assert result['study_definition_bound'] is True
    assert result['study_definition_hash'] == definition.definition_hash
    assert result['state'] == 'INCOMPLETE'
    assert result['qualified_identifiability'] is result['real_data_authorized'] is False
    assert len(result['rows']) == 8 and all(row['state'] == 'NOT_RUN' for row in result['rows'])


def test_managed_master_bound_computation_and_owner_tampering_refusals(tmp_path, monkeypatch):
    store, definition, grants, _, _ = prepared(tmp_path/'managed-master')
    qualification = IdentifiabilitySpec.from_dict(definition.to_dict()['identifiability_spec'])
    result = QualificationRunner(store).run(qualification, definition.study_ids, grants,
        budget=BudgetSpec(90, category='pilot'))
    if result['state'] != 'SUCCEEDED':
        path = store.path/'artifacts'/('.attempt-'+result['attempt_id'])/'worker.log'
        pytest.fail(str(result)+'\n'+path.read_text(errors='replace'))
    with store.lock():
        value = json.loads(store._verified_artifact_content(store._manifest('artifact-'+result['artifact_id'])))
    request = store.manifest('pirc28-qualification-request-'+result['attempt_id'])
    assert value['report']['state'] == 'INCOMPLETE'
    assert value['report']['study_definition_hash'] == definition.definition_hash
    assert value['report']['study_definition_bound'] is True
    assert value['report']['qualified_identifiability'] is value['report']['real_data_authorized'] is False
    # Owner validation remains metadata-only; no second unbudgeted aggregation.
    monkeypatch.setattr('experiments.pirc28.evidence.qualify',
        lambda *args, **kwargs: pytest.fail('owner repeated numerical aggregation'))
    validate_computation_result(store, value, request)
    for field, replacement in [('study_definition_hash', None),
            ('study_definition_hash', digest('different master')), ('study_definition_bound', False)]:
        forged = copy.deepcopy(value)
        forged['report'][field] = replacement
        forged['report_hash'] = digest(forged['report'])
        with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
            validate_computation_result(store, forged, request)
    manifest = store.manifest
    changed = definition.to_dict()
    changed['replicate_seeds'][next(iter(changed['recipes']))]['rep-0'] = 9999
    monkeypatch.setattr(store, 'manifest', lambda name: changed
        if name == 'pirc28-study-definition-'+definition.definition_hash else manifest(name))
    with pytest.raises(ResearchError, match='IDENTITY_CONFLICT'):
        validate_computation_result(store, value, request)


@pytest.fixture(scope='module')
def completed_source(tmp_path_factory):
    store, definition, grants, admissions, _ = prepared(tmp_path_factory.mktemp('pirc28-study')/'workers')
    plan = definition.to_dict()
    name = next(name for name, axes in plan['identifiability_spec']['scenario_axes'].items() if axes['mode_separation'] == 1)
    proposal = next(value for value in definition.proposals({'root': str(store.path.parent), 'store_id': store.store_id})
                    if value['switching_recipe']['scenario']['scenario_id'] == name)
    cell = next(cell for cell in proposal['cells'] if cell['fit_K'] == 2)
    registry, recovery = registries()
    result = SharedRunner(store, registry, recovery).run_cell(proposal['study_id'], digest(cell), budget=BudgetSpec(90, category='pilot'))
    if result['state'] != 'SUCCEEDED':
        logs = list((store.path/'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(result)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))
    return store, definition, grants, admissions, result


def test_actual_managed_defined_source_keeps_other_cells_incomplete(completed_source):
    store, definition, grants, _, result = completed_source
    qualification = IdentifiabilitySpec.from_dict(definition.to_dict()['identifiability_spec'])
    report = collect_qualification(store, qualification, definition.study_ids, grants, check_budget=fixture_poll)
    assert report['state'] == 'INCOMPLETE' and report['study_definition_bound'] is True
    assert sum(row['state'] == 'SUCCEEDED' for row in report['rows']) == 1
    assert next(row for row in report['lineage'] if row['selected_attempt_id'] == result['attempt_id'])['K'] == 2
    assert report['qualified_identifiability'] is False
    with pytest.raises(ResearchError, match='NEEDS_PREREGISTRATION'):
        stage_definition(store, definition)


def test_cell_seed_cannot_change_after_complete_definition_freeze(completed_source, monkeypatch):
    store, definition, grants, _, _ = completed_source
    resolve = store.manifest
    target = definition.study_ids[0]
    def altered(object_id):
        value = resolve(object_id)
        if object_id == 'study-'+target:
            value = copy.deepcopy(value)
            for cell in value['spec']['cells'][:2]:
                cell['seed'] = 999
            value['spec_hash'] = digest(value['spec'])
        return value
    monkeypatch.setattr(store, 'manifest', altered)
    qualification = IdentifiabilitySpec.from_dict(definition.to_dict()['identifiability_spec'])
    with pytest.raises(ResearchError, match='IDENTITY_CONFLICT'):
        collect_qualification(store, qualification, definition.study_ids, grants, check_budget=fixture_poll)


def test_complete_definition_must_precede_actual_admission(completed_source, monkeypatch):
    store, definition, grants, _, _ = completed_source
    events = copy.deepcopy(store.events())
    publication = next(event for event in events if event['event_kind'] == 'MANIFEST'
        and event['payload'].get('object_id') == 'pirc28-study-definition-'+definition.definition_hash)
    publication['sequence'] = max(event['sequence'] for event in events)+1
    monkeypatch.setattr(store, 'events', lambda: events)
    qualification = IdentifiabilitySpec.from_dict(definition.to_dict()['identifiability_spec'])
    with pytest.raises(ResearchError, match='NEEDS_PREREGISTRATION'):
        collect_qualification(store, qualification, definition.study_ids, grants, check_budget=fixture_poll)


@pytest.mark.parametrize('damage', ['missing-study', 'wrong-stage', 'wrong-purpose', 'missing-package'])
def test_registration_requires_complete_original_pilot_inputs(completed_source, damage):
    store, definition, _, original, _ = completed_source
    admissions = copy.deepcopy(original)
    first = definition.study_ids[0]
    if damage == 'missing-study':
        admissions.pop(first)
    elif damage == 'wrong-stage':
        admissions[first]['mode'] = 'formal'
    elif damage == 'wrong-purpose':
        admissions[first]['purpose'] = 'evaluate'
    else:
        admissions[first].pop('package_hash')
    with pytest.raises(ResearchError):
        register_definition_studies(store, definition, admissions)


def test_operator_preregistration_cannot_drop_master_freeze(completed_source, monkeypatch):
    store, definition, _, admissions, _ = completed_source
    resolve = store.manifest
    def changed(object_id):
        value = resolve(object_id)
        if object_id.startswith('preregistration-'):
            value = copy.deepcopy(value)
            value.pop('pirc28_study_definition_hash')
        return value
    monkeypatch.setattr(store, 'manifest', changed)
    with pytest.raises(ResearchError, match='NEEDS_PREREGISTRATION'):
        register_definition_studies(store, definition, admissions)


def test_mathematical_pass_without_whole_study_freeze_is_not_scientific(tmp_path, monkeypatch):
    from tests.test_pirc28_evidence import prepared as legacy_prepared
    from experiments.pirc28 import evidence
    store, study, qualification, grant, _ = legacy_prepared(tmp_path/'legacy', pilot=True)
    monkeypatch.setattr(evidence, 'qualify', lambda *args, **kwargs: {'state': 'PASSED', 'missing_fields': []})
    report = evidence.collect_qualification(store, qualification, [study['study_id']], {study['study_id']: grant}, check_budget=fixture_poll)
    assert report['state'] == 'NEEDS_PREREGISTRATION'
    assert report['qualified_identifiability'] is report['study_definition_bound'] is False
    assert report['gate']['missing_fields'] == ['pirc28_study_definition_hash']


def test_staging_cli_never_initializes_or_registers_studies(tmp_path, capsys):
    from experiments.pirc28.__main__ import main
    root = tmp_path/'cli'
    root.mkdir()
    store = ResearchStore(root, 'pirc28-cli-stage-test-only', initialize=True)
    definition = SyntheticStudyDefinition.from_dict(definition_value())
    path = tmp_path/'definition.json'
    path.write_bytes(definition.content)
    result = main(['--root', str(root), '--store-id', store.store_id, 'stage-definition', str(path)])
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output['definition_hash'] == definition.definition_hash and output['registered_studies'] is False
