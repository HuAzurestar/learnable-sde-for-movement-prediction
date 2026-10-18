"""Synthetic qualification stores; no production ledger or study is created."""

from dataclasses import replace
import json

import numpy as np
import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_recovery import SharedRecovery
from experiments.pirc25.runner import SharedRunner
from experiments.pirc28.__main__ import main
from experiments.pirc28.pipeline import execute_recipe, recipe_resources
from experiments.pirc28.runtime import bind_recipe_execution, registries, resume_command, switching_plugin
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from models.switching import SwitchingError
from tests.research_admission_fixtures import admit_fixture
from tests.test_pirc28_synthetic import fixture_poll, model, scenario
from tests.test_research_store import spec


def recipe():
    s = replace(scenario(paths=2), times=(0.0, 0.5, 1.0, 1.5, 2.0), max_jumps=16)
    return {'schema_version': 'pirc28-synthetic-recipe-v1', 'scenario': s.to_dict(),
        'initializations': {'1': [model(k=1).to_dict()], '2': [model().to_dict()]},
        'inference': {'max_iterations': 2, 'max_substeps': 64, 'max_initializations': 2},
        'train_paths': 2, 'selection_paths': 1, 'test_paths': 1, 'forecast_paths': 6,
        'forecast_origin_index': 2, 'horizon_indices': [3, 4],
        'prior_mean': [0.0, 0.0, 0.0, 0.0], 'prior_covariance': np.eye(4).tolist()}


def prepare(root):
    root.mkdir()
    store = ResearchStore(root, 'pirc28-qualification-fixture', initialize=True)
    value = spec()
    value['study_id'] = 'pirc28-unit-qualification'
    value['runtime_binding'] = {'root': str(root), 'store_id': store.store_id}
    value['switching_recipe'] = recipe()
    value['cells'][0]['fit_K'] = 2
    bind_recipe_execution(value)
    plugin = switching_plugin()
    config, inputs = recipe_resources(value['switching_recipe'], 2)
    grant = admit_fixture(store, value, plugin, root, execution_config=config, execution_inputs=inputs,
                          recovery_command_builder=resume_command)
    store.register(value, digest(value))
    registry, recovery = registries()
    return store, value, registry, recovery, grant


def test_frozen_recipe_replay_and_separate_split_identities():
    first = execute_recipe(recipe(), 2, 19, check_budget=fixture_poll)
    second = execute_recipe(recipe(), 2, 19, check_budget=fixture_poll)
    assert first == second
    assert json.loads(encode(first)) == first
    assert len({item['scenario_hash'] for item in first['fit']['generated_inputs']}) == 3
    assert first['fit']['qualified_identifiability'] is False
    assert first['fit']['interpretation_level'] == 'synthetic-statistical-only'
    assert 'permutation_invariant_mode_brier' in first['fit']['mode_diagnostics']
    assert first['forecast']['role'] == 'filtering-origin'
    assert all(fit['accuracy'] == 'approximate' for fit in first['fit']['initializations'])
    k1 = execute_recipe(recipe(), 1, 19, check_budget=fixture_poll)
    assert k1['fit']['generated_inputs'] == first['fit']['generated_inputs']
    assert set(k1['metrics']) == set(first['metrics'])
    assert 'permutation_invariant_mode_brier' in k1['fit']['unavailable_metrics']


def test_pipeline_resume_matches_continuous_and_rejects_changed_cell():
    snapshots = []
    expected = execute_recipe(recipe(), 2, 19, check_budget=fixture_poll, progress_callback=snapshots.append)
    saved = next(state for state in snapshots if state['fit_states'][0] is not None
                 and state['fit_states'][0]['complete_reason'] is None)
    actual = execute_recipe(recipe(), 2, 19, check_budget=fixture_poll, resume_state=saved)
    assert actual == expected
    with pytest.raises(SwitchingError):
        execute_recipe(recipe(), 2, 20, check_budget=fixture_poll, resume_state=saved)


def test_registered_worker_uses_shared_budget_and_reuses_success(tmp_path):
    store, value, registry, recovery, _ = prepare(tmp_path / 'fixture')
    runner = SharedRunner(store, registry, recovery)
    result = runner.run_cell(value['study_id'], digest(value['cells'][0]), budget=BudgetSpec(45))
    if result['state'] != 'SUCCEEDED':
        logs = list((store.path / 'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(result) + '\n' + '\n'.join(path.read_text(errors='replace') for path in logs))
    payload = json.loads((store.path / 'artifacts' / result['artifact_id']).read_bytes())
    assert payload['source_schema'] == 'pirc28-synthetic-result-v1'
    assert payload['qualification'] == 'fixture'
    assert payload['resume_level'] == 'numerical-tolerance'
    assert payload['admission_hash']
    assert BudgetLedger(store).balance('affine')['committed_ms'] > 0
    reused = runner.run_cell(value['study_id'], digest(value['cells'][0]), budget=BudgetSpec(45))
    assert reused['artifact_id'] == result['artifact_id']
    assert len(store.attempts()) == 1


def test_shared_owner_checkpoint_and_real_worker_resume_preserve_prior_cost(tmp_path):
    store, value, registry, recovery, grant = prepare(tmp_path / 'resume-fixture')
    snapshots = []
    expected = execute_recipe(value['switching_recipe'], 2, value['cells'][0]['seed'],
        check_budget=fixture_poll, progress_callback=snapshots.append)
    state = next(item for item in snapshots if item['fit_states'][0] is not None
                 and item['fit_states'][0]['complete_reason'] is None)
    run_id = store.register_run(value['study_id'], value['cells'][0])
    parent = store.new_attempt(run_id)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(parent, BudgetSpec(45))
    store.transition(parent, 'RUNNING')
    # Owner-side checkpoint integration fixture, not a claim of process crash.
    checkpoint = SharedRecovery(store, registry, recovery).checkpoint(parent,
        {'step': 1, 'data_position': {'accepted_EM_steps': 1}, 'method_state': state, 'rng_state': state['rng_state']})
    ledger.settle(reservation['reservation_id'], 2000, outcome='FAILED')
    store.transition(parent, 'FAILED', error_code='QUALIFICATION_FIXTURE_INTERRUPTION')
    before = ledger.balance('affine')['committed_ms']
    actual = SharedRecovery(store, registry, recovery).resume(parent, checkpoint, authorization=grant, budget=BudgetSpec(45))
    if actual['state'] != 'SUCCEEDED':
        logs = list((store.path / 'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(actual) + '\n' + '\n'.join(path.read_text(errors='replace') for path in logs))
    payload = json.loads((store.path / 'artifacts' / actual['artifact_id']).read_bytes())
    assert payload['metrics'] == expected['metrics']
    assert payload['fit'] == expected['fit']
    assert payload['forecast'] == expected['forecast']
    assert ledger.balance('affine')['committed_ms'] > before >= 2000
    assert store.attempts()[parent]['state'] == 'FAILED'
    assert store.attempts()[actual['attempt_id']]['parent_attempt_id'] == parent


def test_cli_requires_existing_shared_store_and_valid_budget(tmp_path, monkeypatch):
    monkeypatch.delenv('SDE_RUNTIME_ROOT', raising=False)
    with pytest.raises(SystemExit):
        main(['--store-id', 'pirc28', 'run', 'study'])
    assert main(['--root', str(tmp_path / 'missing'), '--store-id', 'pirc28', 'run', 'study']) == 2
    assert not (tmp_path / 'missing' / 'pirc25' / 'store.json').exists()


def test_resource_binding_refuses_oversized_matrix_before_execution():
    value = spec()
    value['switching_recipe'] = recipe()
    value['switching_recipe']['inference']['max_substeps'] = 100_000
    value['cells'][0]['fit_K'] = 2
    with pytest.raises(ResearchError) as error:
        bind_recipe_execution(value)
    assert error.value.code == 'RESOURCE_PLAN_REJECTED'
