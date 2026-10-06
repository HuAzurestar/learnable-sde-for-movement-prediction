"""Scientific-plan metadata controls, never actual scientific replicates."""

import copy
import itertools
import json
import math

import numpy as np
import pytest

from application.research_registry import plan_resources
from experiments.pirc25.affine import code_hash
from experiments.pirc28.__main__ import main as runtime_main
from experiments.pirc28.plan import (MAXIMUM_PLAN_BYTES, REFERENCE_PLAN, describe_plan,
                                    load_plan, main, prepare_definition)
from experiments.pirc28.pipeline import recipe_resources
from experiments.pirc28.qualification import AXES, IdentifiabilitySpec, _wilson
from experiments.pirc28.runtime import switching_plugin
from experiments.pirc28.study import SyntheticStudyDefinition
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode


@pytest.fixture(scope='module')
def reference():
    plan = load_plan(REFERENCE_PLAN)
    return plan, prepare_definition(plan)


def test_real_reference_plan_covers_full_six_axis_cartesian_matrix(reference):
    plan, definition = reference
    value = definition.to_dict()
    assert value['runtime_code_hash'] == code_hash()
    assert len(value['recipes']) == 64 and len(definition.content) < 4*1024*1024
    expected = {tuple(point) for point in itertools.product(*(plan['axes'][name] for name in AXES))}
    actual = {tuple(row['axes'][name] for name in AXES) for row in value['phase_matrix']['scenarios']}
    assert actual == expected
    qualification = IdentifiabilitySpec.from_dict(value['identifiability_spec'])
    assert not qualification.missing_fields and len(qualification.replicate_ids) == 16
    roles = list(qualification.scenario_roles.values())
    assert roles.count('required-qualified') == 2 and roles.count('required-unidentifiable') == 32
    assert roles.count('phase-diagram-only') == 30
    assert _wilson(16, 16, 0.95)[0] > 0.7 and _wilson(15, 16, 0.95)[0] > 0.7
    assert _wilson(14, 16, 0.95)[0] < 0.7


def test_independent_seeds_and_initializers_are_fixed_without_truth_adaptation(reference, tmp_path):
    plan, definition = reference
    value = definition.to_dict()
    seeds = [seed for scenario in value['replicate_seeds'].values() for seed in scenario.values()]
    assert len(seeds) == len(set(seeds)) == 1024
    initializations = [recipe['initializations'] for recipe in value['recipes'].values()]
    assert all(value == initializations[0] for value in initializations)
    assert len(initializations[0]['2']) == 2 and len({digest(item) for item in initializations[0]['2']}) == 2
    assert all(digest(item) != digest(recipe['scenario']['model'])
        for recipe in value['recipes'].values() for item in recipe['initializations']['2'])
    proposals = definition.proposals({'root': str(tmp_path), 'store_id': 'explicit-test-only'})
    assert sum(len(proposal['cells']) for proposal in proposals) == 2048
    for proposal in proposals:
        for index in range(0, len(proposal['cells']), 2):
            left, right = proposal['cells'][index:index+2]
            assert left['seed'] == right['seed'] and left['replicate_id'] == right['replicate_id']
            assert left['fit_K'] == 1 and right['fit_K'] == 2
            assert 'admission' not in proposal
    assert plan == load_plan(REFERENCE_PLAN)


def test_every_declared_source_fits_existing_resources_and_substep_rate_family(reference):
    _, definition = reference
    value = definition.to_dict()
    plugin = switching_plugin()
    for recipe in value['recipes'].values():
        times = recipe['scenario']['times']
        origin = times[recipe['forecast_origin_index']]
        assert origin == 2.0
        assert [times[index]-origin for index in recipe['horizon_indices']] == [0.5, 1.0]
        assert recipe['train_paths'] == 3 and recipe['test_paths'] == 8
        for k in (1, 2):
            config, inputs = recipe_resources(recipe, k)
            plan = plan_resources(plugin.registry_entry, config, inputs, matrix_cells=32)
            assert plan['tensor_elements'] <= 20_000_000 and plan['tensor_bytes'] <= 512*1024*1024
            for model in recipe['initializations'][str(k)]:
                rate = model['rate_bounds'][1]
                delta = min(recipe['inference']['max_step_seconds'], recipe['inference']['max_rate_step']/rate)
                assert sum(math.ceil(float(dt)/delta) for dt in np.diff(times)) <= recipe['inference']['max_substeps']


def test_required_negative_controls_have_actual_common_observation_laws(reference):
    _, definition = reference
    value = definition.to_dict()
    roles = value['identifiability_spec']['scenario_roles']
    for name, recipe in value['recipes'].items():
        if roles[name] != 'required-unidentifiable':
            continue
        scenario = recipe['scenario']
        for field in ('A', 'B', 'a', 'L'):
            assert scenario['model'][field][0] == scenario['model'][field][1]
        assert scenario['initial_means'][0] == scenario['initial_means'][1]
        assert scenario['initial_covariances'][0] == scenario['initial_covariances'][1]


@pytest.mark.parametrize('damage', ['missing-threshold', 'missing-inference', 'missing-axis', 'constant-axis',
    'unknown-field', 'few-replicates', 'impossible-power', 'vacuous-recovery', 'vacuous-collapse',
    'vacuous-occupancy', 'duplicate-required', 'outside-required', 'required-negative', 'false-control',
    'boolean-control', 'different-diffusion', 'duplicate-init', 'too-many-init', 'initializer-rate-family',
    'insufficient-substeps', 'allocation', 'horizon-grid', 'no-forecast-paths', 'too-many-cells', 'nonfinite'])
def test_profile_refuses_unfrozen_infeasible_or_reduced_science(reference, damage):
    value = copy.deepcopy(reference[0])
    if damage == 'missing-threshold': value['qualification']['thresholds'].pop('mode_brier')
    elif damage == 'missing-inference': value['inference'].pop('max_rate_step')
    elif damage == 'missing-axis': value['axes'].pop('mode_occupancy')
    elif damage == 'constant-axis': value['axes']['trajectory_length_s'] = [8.0]
    elif damage == 'unknown-field': value['qualified_identifiability'] = True
    elif damage == 'few-replicates': value['sampling']['replicate_count'] = 1
    elif damage == 'impossible-power': value['qualification']['minimum_recovery_fraction'] = 0.95
    elif damage == 'vacuous-recovery': value['qualification']['minimum_recovery_fraction'] = 0
    elif damage == 'vacuous-collapse': value['qualification']['maximum_collapse_fraction'] = 1
    elif damage == 'vacuous-occupancy': value['qualification']['minimum_occupancy'] = 0
    elif damage == 'duplicate-required': value['scope']['required_qualified_points'].append(value['scope']['required_qualified_points'][0])
    elif damage == 'outside-required': value['scope']['required_qualified_points'][0]['mode_separation'] = 3
    elif damage == 'required-negative': value['scope']['required_qualified_points'][0]['mode_separation'] = 0
    elif damage == 'false-control': value['scope']['negative_control_separation'] = 2
    elif damage == 'boolean-control': value['scope']['negative_control_separation'] = False
    elif damage == 'different-diffusion': value['truth_parameters']['L'][1][0][0] = 0.3
    elif damage == 'duplicate-init': value['initializations']['2'][1] = copy.deepcopy(value['initializations']['2'][0])
    elif damage == 'too-many-init': value['initializations']['2'].append(copy.deepcopy(value['initializations']['2'][0]))
    elif damage == 'initializer-rate-family': value['initializations']['1'][0]['rate_bounds'] = [0, 8]
    elif damage == 'insufficient-substeps': value['inference']['max_substeps'] = 1599
    elif damage == 'allocation': value['sampling']['train_paths'] = 1000
    elif damage == 'horizon-grid': value['forecast_horizons_s'] = [0.3, 1.0]
    elif damage == 'no-forecast-paths': value['sampling']['forecast_paths'] = 0
    elif damage == 'too-many-cells': value['sampling']['replicate_count'] = 1000
    else: value['qualification']['thresholds']['mode_ece'] = float('nan')
    with pytest.raises(ResearchError):
        prepare_definition(value)


def test_plan_and_definition_are_deterministic_detached_and_content_bound(reference):
    plan, definition = reference
    changed = copy.deepcopy(plan)
    changed['qualification']['thresholds']['mode_ece'] = 0.09
    newer = prepare_definition(changed)
    assert newer.definition_hash != definition.definition_hash
    assert newer.to_dict()['replicate_seeds'] != definition.to_dict()['replicate_seeds']
    assert prepare_definition(plan).definition_hash == definition.definition_hash
    changed['truth_parameters']['a'][0][0] = -99
    assert newer.to_dict()['phase_matrix']['baseline_scenario']['model']['a'][0][0] == -0.8
    assert SyntheticStudyDefinition(definition.content).definition_hash == definition.definition_hash


def test_read_only_cli_cannot_fit_open_store_or_authorize(reference, monkeypatch, capsys):
    import experiments.pirc28.study as study
    monkeypatch.setattr(study, 'stage_definition', lambda *args: pytest.fail('summary staged metadata'))
    monkeypatch.setattr(ResearchStore, '__init__', lambda *args, **kwargs: pytest.fail('summary opened store'))
    monkeypatch.setattr('experiments.pirc28.pipeline.execute_recipe', lambda *args, **kwargs: pytest.fail('summary ran science'))
    assert main([str(REFERENCE_PLAN)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['definition_hash'] == reference[1].definition_hash
    assert output['paired_cell_count'] == 2048 and output['prepared_only'] is True
    assert output['scientific_groups_executed'] == 0
    assert output['qualified_identifiability'] is output['real_data_authorized'] is output['registered_studies'] is False


def test_stage_plan_cli_publishes_only_metadata_into_explicit_existing_store(reference, tmp_path, capsys):
    root = tmp_path/'explicit-disposable-runtime'
    store = ResearchStore(root, 'pirc28-plan-test-only', initialize=True)
    assert runtime_main(['--root', str(root), '--store-id', store.store_id, 'stage-plan', str(REFERENCE_PLAN)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['definition_hash'] == reference[1].definition_hash
    assert output['registered_studies'] is output['real_data_authorized'] is False
    assert store.manifest('pirc28-study-definition-'+output['definition_hash']) == reference[1].to_dict()
    assert not store.attempts()
    assert not any(event['event_kind'] in ('RESERVE', 'ADMISSION', 'WORKER_STARTED') for event in store.events())
    assert not any(event['event_kind'] == 'MANIFEST' and event['payload']['object_id'].startswith(('study-', 'authorization-'))
                   for event in store.events())
    missing = tmp_path/'not-a-runtime'
    assert runtime_main(['--root', str(missing), '--store-id', 'original-id', 'stage-plan', str(REFERENCE_PLAN)]) == 2
    assert not missing.exists()


@pytest.mark.parametrize('kind', ['duplicate', 'nonfinite', 'oversized'])
def test_profile_file_refuses_ambiguous_or_oversized_input(tmp_path, kind):
    path = tmp_path/'profile.json'
    if kind == 'duplicate': path.write_text('{"plan_id":"one","plan_id":"two"}')
    elif kind == 'nonfinite': path.write_text('{"value":NaN}')
    else: path.write_bytes(b' '* (MAXIMUM_PLAN_BYTES+1))
    with pytest.raises(ResearchError):
        load_plan(path)


def test_brier_thresholds_are_not_impossible_zero_noise_targets(reference):
    plan, definition = reference
    thresholds = definition.to_dict()['identifiability_spec']['thresholds']
    # Two-component mode Brier has uniform-label risk 0.5. Each future
    # Bernoulli event has Bayes risk p(1-p), up to 0.25 even for oracle rates.
    assert thresholds['mode_brier'] < 0.5
    assert thresholds['any_jump_brier'] > 0.25 and thresholds['different_endpoint_mode_brier'] > 0.25
    assert plan['qualification']['coverage_nominal'] == 0.9
