"""Additive registered allocation controls; no science or production authority."""

import copy
import sys

import pytest

from application.research_execution import execution_binding
from application.research_registry import GLOBAL_LIMITS, plan_resources
from experiments.pirc25.affine import code_hash
from experiments.pirc28.pipeline import recipe_allocation, recipe_resources
from experiments.pirc28.plan import REFERENCE_PLAN, load_plan, prepare_definition
from experiments.pirc28.runtime import bind_recipe_execution, switching_plugin
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc28_runtime import recipe
from tests.test_research_store import spec


@pytest.fixture(scope='module')
def master():
    return prepare_definition(load_plan(REFERENCE_PLAN))


def test_full_master_registered_pool_covers_additive_fit_forecast_source_and_reference(master):
    plugin = switching_plugin()
    maximum = 0
    for value in master.to_dict()['recipes'].values():
        for k in (1, 2):
            allocation = recipe_allocation(value, k)
            config, inputs = recipe_resources(value, k)
            registered = plan_resources(plugin.registry_entry, config, inputs, matrix_cells=32)
            assert registered['tensor_elements'] == allocation['combined_tensor_elements']
            assert registered['tensor_bytes'] == allocation['combined_tensor_bytes']
            assert registered['counts']['observations'] == inputs['workspace_vectors']
            assert inputs['observations'] == sum(value[name] for name in (
                'train_paths', 'selection_paths', 'test_paths'))*len(value['scenario']['times'])
            assert config['components'] == max(value['train_paths'], value['selection_paths'])*len(value['initializations'][str(k)])
            assert allocation['resource_admitted'] is allocation['managed_executor_implemented'] is False
            assert allocation['combination_rule'] == 'additive-concurrent-envelope; no-phase-reuse'
            raw = sum(allocation[name] for name in ('source_data_elements', 'fit_metadata_elements',
                'retained_forecast_elements', 'reference_elements', 'k1_boundary_elements'))
            assert 0 <= allocation['padded_pool_elements']-raw <= 3
            assert registered['tensor_elements'] == (allocation['fit_workspace_elements']
                +allocation['forecast_workspace_elements']+allocation['padded_pool_elements'])
            if k == 2:
                assert allocation['reference_elements'] == allocation['reference_core_plan']['tensor_elements'] > 0
                core = allocation['reference_core_plan']
                assert core['completed_batch_trace_elements'] == 32+16*value['test_paths']*value[
                    'approximation_spec']['event_driven_reference']['independent_batches']
                assert core['owner_trace_copy_elements'] == 2*core['completed_batch_trace_elements']
            else:
                assert allocation['reference_elements'] == 0 and allocation['reference_core_plan'] is None
                assert allocation['k1_boundary_elements'] == allocation['k1_boundary_plan']['tensor_elements'] > 0
                assert allocation['k1_boundary_plan']['reference_particles'] == 0
            assert registered['limits'] == {**dict(GLOBAL_LIMITS), 'matrix_cells': 10000,
                                           'result_bytes': 64*1024*1024}
            maximum = max(maximum, registered['tensor_elements'])
    assert maximum == 19_719_396 < GLOBAL_LIMITS['tensor_elements']
    assert plugin.registry_entry.version == '1.1.0'


@pytest.mark.parametrize('k', [1, 2])
def test_legacy_recipe_reserves_source_and_retained_forecasts_without_inventing_reference(k):
    value = recipe()
    allocation = recipe_allocation(value, k)
    assert allocation['reference_elements'] == 0 and allocation['reference_core_plan'] is None
    assert allocation['source_data_elements'] > 0 and allocation['retained_forecast_elements'] > 0
    assert allocation['inputs']['observations'] == 20
    assert allocation['inputs']['workspace_vectors']*4 >= allocation['source_data_elements']
    plan_resources(switching_plugin().registry_entry, allocation['config'], allocation['inputs'], matrix_cells=1)


def test_individually_small_allocations_cannot_bypass_combined_global_cap(master, monkeypatch):
    value = copy.deepcopy(max(master.to_dict()['recipes'].values(), key=lambda r: len(r['scenario']['times'])))
    value['inference']['max_substeps'] = 1700
    allocation = recipe_allocation(value, 2)
    assert allocation['reference_elements'] < GLOBAL_LIMITS['tensor_elements']
    assert allocation['fit_workspace_elements'] < GLOBAL_LIMITS['tensor_elements']
    assert allocation['combined_tensor_elements'] > GLOBAL_LIMITS['tensor_elements']
    monkeypatch.setattr('experiments.pirc28.pipeline.generate_scenario',
                        lambda *a, **k: pytest.fail('data generation before admission'))
    with pytest.raises(ResearchError, match='combined tensors'):
        bind_recipe_execution({'switching_recipe': value, 'cells': [{'fit_K': 2}]})


def test_compiler_refuses_combined_peak_before_master_or_scientific_runs(monkeypatch):
    plan = load_plan(REFERENCE_PLAN)
    plan['inference']['max_substeps'] = 1700
    monkeypatch.setattr('experiments.pirc28.pipeline.generate_scenario',
                        lambda *a, **k: pytest.fail('data generation during metadata preparation'))
    with pytest.raises(ResearchError, match='combined tensors'):
        prepare_definition(plan)


@pytest.mark.parametrize('damage', ['absent', 'zero', 'boolean'])
def test_registered_storage_input_is_required_bounded_and_not_a_boolean(damage):
    config, inputs = recipe_resources(recipe(), 2)
    if damage == 'absent': inputs.pop('workspace_vectors')
    else: inputs['workspace_vectors'] = 0 if damage == 'zero' else True
    with pytest.raises(ResearchError):
        plan_resources(switching_plugin().registry_entry, config, inputs, matrix_cells=1)


@pytest.mark.parametrize('field', ['workspace_vectors', 'observations'])
def test_actual_worker_independently_refuses_forged_counts_before_control_or_numerics(tmp_path, monkeypatch, field):
    from experiments.pirc28 import worker
    value = spec()
    root, store_id = str(tmp_path), 'explicit-memory-only-test-store'
    value.update(runtime_binding={'root': root, 'store_id': store_id}, switching_recipe=recipe(), code_hash=code_hash())
    value['cells'][0]['fit_K'] = 2
    bind_recipe_execution(value)
    cell = value['cells'][0]
    config, inputs = recipe_resources(value['switching_recipe'], 2)
    inputs[field] = 1
    # Forgery even recomputes a valid generic registered plan/hash. The real
    # worker must derive recipe counts rather than trust that self-consistency.
    cell['execution'] = execution_binding(switching_plugin().registry_entry, config, inputs, matrix_cells=1)
    class MemoryStore:
        def manifest(self, key):
            assert key == 'study-'+value['study_id']
            return {'spec': value}
    monkeypatch.setattr(worker, 'ResearchStore', lambda *a: MemoryStore())
    monkeypatch.setattr(worker.WorkerControl, 'from_environment', lambda: pytest.fail('control reached after count forgery'))
    monkeypatch.setattr(worker, 'execute_recipe', lambda *a, **k: pytest.fail('numerical execution after count forgery'))
    monkeypatch.setattr(sys, 'argv', ['worker', root, store_id, value['study_id'], digest(cell), str(tmp_path/'unused.json')])
    with pytest.raises(ResearchError, match='declared resources differ'):
        worker.main()
    assert not (tmp_path/'pirc25').exists() and not (tmp_path/'unused.json').exists()


def test_plugin_resource_revision_keeps_original_method_budget_identities(master, tmp_path):
    value = master.to_dict()
    proposals = master.proposals({'root': str(tmp_path), 'store_id': 'explicit-metadata-only-test'})
    assert all(proposal['arms'] == list(value['arms'].values()) for proposal in proposals)
    assert value['arms']['2']['arm_id'] == 'pirc28-k2'
    assert value['arms']['2']['budget_seconds'] == 86400
    assert all(cell['execution']['component_version'] == '1.1.0' for proposal in proposals for cell in proposal['cells'])
    assert not list(tmp_path.iterdir())
