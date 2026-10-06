"""PIRC-28 plugin declarations; every execution uses the shared supervisor."""

from pathlib import Path
import sys

from application.research_contracts import CapabilityRegistry, ExecutionPlugin
from application.research_execution import execution_binding
from application.research_recovery import RecoveryPlugin, RecoveryRegistry
from application.research_registry import GLOBAL_LIMITS, RegistryEntry, implementation_hash
from infrastructure.research_store import ResearchError, atomic_write, digest, encode
from .pipeline import recipe_resources

PLUGIN_ID = 'pirc28-switching'
VERSION = '1.0.0'


def command(output, spec, cell):
    binding = spec.get('runtime_binding')
    if (type(binding) is not dict or set(binding) != {'root', 'store_id'}
            or type(binding['root']) is not str or not Path(binding['root']).is_absolute()
            or type(binding['store_id']) is not str or not binding['store_id']):
        raise ResearchError('MISSING_INPUT', 'explicit shared runtime root and store identity are required')
    return [sys.executable, '-m', 'experiments.pirc28.worker', binding['root'], binding['store_id'],
            spec['study_id'], digest(cell), str(output)]


def resume_command(output, spec, cell, state):
    target = Path(output).with_name('pirc28-resume-state.json')
    atomic_write(target, encode(state))
    return command(output, spec, cell) + [str(target), digest(state)]


def switching_plugin():
    def counts_schema(values):
        return {'type': 'object', 'properties': values, 'required': sorted(values), 'additionalProperties': False}
    integer = lambda lower, upper: {'type': 'integer', 'minimum': lower, 'maximum': upper}
    config = counts_schema({**{name: integer(1, limit) for name, limit in
        (('paths', 65536), ('steps', 100_000), ('mixtures', 2), ('components', 4096))},
        'dtype': {'type': 'string', 'enum': ['float64']}, 'device': {'type': 'string', 'enum': ['cpu']},
        'noise_dim': {'type': 'integer', 'enum': [2]}})
    inputs = counts_schema({'observations': integer(1, 10_000_000), 'state_dim': {'type': 'integer', 'enum': [4]}})
    entry = RegistryEntry(component_id=PLUGIN_ID, component_kind='execution-adapter', version=VERSION,
        code_hash=implementation_hash(command), config_schema=config, input_schema=inputs,
        output_schema={'type': 'object', 'properties': {}, 'additionalProperties': True},
        state_order=('x', 'y', 'vx', 'vy'), units=('m', 'm', 'm/s', 'm/s'),
        capabilities=frozenset({'switching-transition'}), resource_class='cpu', resume_level='numerical-tolerance',
        resource_contract={'schema_version': 'pirc25-resource-contract-v1', 'counts': {
            **{name: {'config': [name]} for name in ('paths', 'steps', 'mixtures', 'components')},
            'observations': {'input': ['observations']}, 'state_dim': {'constant': 4}},
            'tensors': [{'name': 'EM-workspace-' + str(index),
                'axes': ['components', 'steps', 'mixtures', 'mixtures', 'state_dim', 'state_dim'],
                'item_bytes': 8} for index in range(24)] + [
                {'name': 'observations', 'axes': ['observations', 'state_dim'], 'item_bytes': 8},
                {'name': 'forecast', 'axes': ['paths', 'steps', 'state_dim'], 'item_bytes': 8}],
            'limits': {**dict(GLOBAL_LIMITS), 'matrix_cells': 10000, 'result_bytes': 64*1024*1024}})
    return ExecutionPlugin(PLUGIN_ID, entry.capabilities, entry.state_order, entry.units,
                           entry.resume_level, command, entry)


def registries():
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(switching_plugin())
    recovery.register(RecoveryPlugin(PLUGIN_ID, 'numerical-tolerance', resume_command, VERSION))
    return registry, recovery


def bind_recipe_execution(spec):
    plugin = switching_plugin()
    for cell in spec['cells']:
        config, inputs = recipe_resources(spec['switching_recipe'], cell['fit_K'])
        cell.update(plugin_id=PLUGIN_ID, capability='switching-transition', resource_class='cpu', visibility='synthetic')
        cell['execution'] = execution_binding(plugin.registry_entry, config, inputs, matrix_cells=len(spec['cells']))
    return spec
