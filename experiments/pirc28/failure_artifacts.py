"""Canonical numerical failures bound to ordinary managed admission and costs."""

from dataclasses import replace

import numpy as np

from application.research_preregistration import hash_reference
from infrastructure.research_store import ResearchError, digest
from .failures import NUMERICAL_CODES, valid_numerical_diagnostic
from .pipeline import validate_recipe


def failure_result(diagnostic, spec, cell, plugin):
    payload = {'failure': diagnostic, 'source_schema': 'pirc28-synthetic-failure-v1'}
    return {'schema_version': 'pirc25-failure-v1', 'status': 'FAILED', 'qualification': 'failed',
        'spec_hash': digest(spec), 'cell_hash': digest(cell), 'protocol_hash': spec['protocol_hash'],
        'input_hash': spec['data_hash'], 'output_hash': digest(payload), 'error_code': diagnostic['error_code'],
        'state_order': list(plugin.state_order), 'units': list(plugin.units), 'time_unit': 's',
        'resume_level': plugin.resume_level, 'qualified_identifiability': False, 'real_data_authorized': False,
        **payload}


def validate_cell_failure(result, spec, cell):
    """Bounded metadata only; no fitting, forecasting or numerical aggregation."""
    from .evidence import validate_initialization_history
    value = result.get('failure')
    keys = {'schema_version', 'stage', 'error_code', 'initializations', 'selection', 'cause',
            'recipe_hash', 'generated_inputs'}
    stages = {'generation', 'fit', 'selection', 'test-inference', 'forecast', 'evaluation'}
    if (result.get('source_schema') != 'pirc28-synthetic-failure-v1' or type(value) is not dict
            or set(value) != keys or value['schema_version'] != 'pirc28-numerical-cell-failure-v1'
            or type(value['stage']) is not str or value['stage'] not in stages
            or type(value['error_code']) is not str or value['error_code'] not in NUMERICAL_CODES
            or value['error_code'] != result['error_code']
            or value['recipe_hash'] != digest({'recipe': spec['switching_recipe'], 'K': cell['fit_K'], 'seed': cell['seed']})):
        raise ResearchError('CONTRACT_MISMATCH', 'numerical failure recipe/stage/schema differs')
    scenario, models, _, _, _ = validate_recipe(spec['switching_recipe'], cell['fit_K'])
    generated = value['generated_inputs']
    if (type(generated) is not list or any(type(item) is not dict for item in generated)
            or not 0 <= len(generated) <= 3 or (value['stage'] != 'generation' and len(generated) != 3)
            or (value['stage'] == 'generation' and len(generated) == 3)):
        raise ResearchError('CONTRACT_MISMATCH', 'failure generated-source history differs')
    seeds = np.random.default_rng(cell['seed']).integers(0, 2**63, size=3, dtype=np.int64)
    for index, role in enumerate(('train', 'selection', 'test')[:len(generated)]):
        frozen = replace(scenario, seed=int(seeds[index]), sample_count=spec['switching_recipe'][role+'_paths'],
                         scenario_id=scenario.scenario_id+'-'+role)
        if (set(generated[index]) != {'role', 'scenario_hash', 'observations_hash', 'truth_hash'}
                or generated[index]['role'] != role or generated[index]['scenario_hash'] != frozen.scenario_hash
                or not hash_reference(generated[index]['observations_hash'])
                or not hash_reference(generated[index]['truth_hash'])):
            raise ResearchError('IDENTITY_CONFLICT', 'failure roles differ from registered source streams')
    histories = value['initializations']
    if value['stage'] == 'generation':
        expected = [{'initialization_id': 'init-'+str(index), 'state': 'NOT_RUN',
                     'initial_model_hash': digest(model.to_dict())} for index, model in enumerate(models)]
        if histories != expected or value['selection'] is not None:
            raise ResearchError('CONTRACT_MISMATCH', 'generation failure cannot invent fitted initializations')
    else:
        chosen = validate_initialization_history(spec, cell, value, generated,
            allow_failed_selection=value['stage'] in {'fit', 'selection'})
        if (value['stage'] == 'fit' and (value['selection'] is not None or chosen is not None)
                or value['stage'] == 'selection' and (value['selection'] is None or chosen is not None
                    or not any(history['state'] == 'SUCCEEDED' for history in histories))):
            raise ResearchError('CONTRACT_MISMATCH', 'failed initialization stage/disposition differs')
    cause = value['cause']
    if value['stage'] in {'fit', 'selection'}:
        if cause is not None or value['error_code'] != 'INFERENCE_INAPPLICABLE':
            raise ResearchError('CONTRACT_MISMATCH', 'all-initializer failure must retain its individual causes')
    elif (not valid_numerical_diagnostic(cause, value['stage'], None)
            or set(cause) != {'schema_version', 'state', 'initialization_id', 'stage', 'error_code', 'exception_type'}
            or cause['error_code'] != value['error_code']):
        raise ResearchError('CONTRACT_MISMATCH', 'whole-cell numerical cause differs')
