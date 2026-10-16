"""K1 worker diagnostic provenance, not standalone execution/SCI authority.

The collector must verify the canonical artifact, actual admission, grants,
whole-tree stop and original costs. The owner performs metadata checks only.
"""

from pathlib import Path
import re

import numpy as np

from infrastructure.research_store import digest, encode
from models.switching import SwitchingSDEModel
from .approximation import ApproximationSpec, RECIPE_SCHEMA
from .approximation_evidence import _keys, _number, _require, _same_declaration
from .k1_boundary import SCHEMA as REPORT_SCHEMA, FLAGS, boundary_plan
from .reference_execution import PREFLIGHT_FIELDS

SCHEMA = 'pirc28-native-k1-boundary-execution-v1'


def validate_report_metadata(report, recipe, model):
    """Original expected metadata only; NEVER filtering or reference sampling."""
    if report is None:
        return False
    from .pipeline import validate_recipe
    scenario, _, inference, mean, covariance = validate_recipe(recipe, 1)
    expected = {'schema_version': REPORT_SCHEMA, 'model_hash': digest(model),
        'inference_hash': digest(inference.to_dict()),
        'spec_hash': ApproximationSpec.from_dict(recipe['approximation_spec']).spec_hash,
        'origin_index': recipe['forecast_origin_index'],
        'initial_state_hash': digest({'probability': [1.0], 'means': np.tile(mean, (1, 1)).tolist(),
                                     'covariances': np.tile(covariance, (1, 1, 1)).tolist()}),
        'resource_plan': boundary_plan(inference, scenario.times, recipe['forecast_origin_index'],
                                     recipe['test_paths'], len(scenario.H)),
        'accuracy': 'one-mode-conditional-affine-Kalman-boundary; not-switching-accuracy', **FLAGS}
    _keys(report, {*expected, 'observation_prefix_hash', 'state', 'paths'}, 'native K1 report')
    for name, value in expected.items():
        _same_declaration(report[name], value, lambda: None)
    _require(type(report['observation_prefix_hash']) is str
             and re.fullmatch('[0-9a-f]{64}', report['observation_prefix_hash']), 'bounded K1 prefix hash required')
    _require(type(report['paths']) is list and len(report['paths']) == recipe['test_paths'],
             'complete original K1 report paths required')
    for index, path in enumerate(report['paths']):
        _keys(path, {'path_index', 'boundary'}, 'native K1 path')
        _require(type(path['path_index']) is int and path['path_index'] == index, 'original K1 path order differs')
        row = path['boundary']
        _keys(row, {'mode', 'passed', 'mean_max_abs', 'covariance_max_abs', 'log_likelihood_abs'}, 'native K1 boundary')
        _require(type(row['mode']) is int and row['mode'] == 0 and type(row['passed']) is bool,
                 'typed K1 boundary disposition required')
        for name in ('mean_max_abs', 'covariance_max_abs', 'log_likelihood_abs'):
            _require(_number(row[name]) >= 0, 'nonnegative finite K1 boundary error required')
    state = 'NUMERICAL_CHECKS_PASSED' if all(row['boundary']['passed'] for row in report['paths']) else 'INFERENCE_INAPPLICABLE'
    _require(type(report['state']) is str and report['state'] == state, 'K1 report disposition differs')
    _require(len(encode(report)) <= 16*1024+1024*recipe['test_paths'], 'native K1 report byte quota exceeded')
    return True


def execution_record(study, cell, attempt, model, report, *, source_preflight=None):
    recipe = study['switching_recipe']
    _require(recipe['schema_version'] == RECIPE_SCHEMA and cell['fit_K'] == 1
             and study['admission']['mode'] in ('fixture', 'pilot'), 'original K1 native scope differs')
    _require(SwitchingSDEModel.from_dict(model).n_modes == 1, 'original selected K1 model required')
    for name, size in (('attempt_id', 32), ('run_id', 64)):
        value = attempt.get(name)
        _require(type(value) is str and re.fullmatch('[0-9a-f]{'+str(size)+'}', value),
                 'typed original K1 native identity required')
    binding = study.get('runtime_binding')
    _keys(binding, {'root', 'store_id'}, 'original K1 runtime')
    _require(type(binding['root']) is str and 0 < len(binding['root']) <= 4096
             and Path(binding['root']).is_absolute() and type(binding['store_id']) is str
             and 0 < len(binding['store_id']) <= 128, 'bounded original K1 runtime required')
    _require(type(study.get('code_hash')) is str and re.fullmatch('[0-9a-f]{64}', study['code_hash']),
             'original K1 native code identity required')
    if study['admission']['mode'] == 'fixture':
        _require(source_preflight is None, 'fixture K1 cannot borrow source preflight')
    else:
        _keys(source_preflight, PREFLIGHT_FIELDS, 'original K1 preflight')
        _require(type(source_preflight['admission_hash']) is str
                 and re.fullmatch('[0-9a-f]{64}', source_preflight['admission_hash']), 'original K1 admission required')
        expected = {'study_definition_hash': study.get('pirc28_study_definition_hash'),
                    'attempt_id': attempt['attempt_id'], 'run_id': attempt['run_id'],
                    'admission_hash': source_preflight['admission_hash'],
                    'qualified_identifiability': False, 'real_data_authorized': False}
        _require(type(expected['study_definition_hash']) is str
                 and re.fullmatch('[0-9a-f]{64}', expected['study_definition_hash']), 'original K1 master required')
        _same_declaration(source_preflight, expected, lambda: None)
    complete = validate_report_metadata(report, recipe, model)
    value = {'schema_version': SCHEMA, 'study_id': study['study_id'], 'spec_hash': digest(study),
        'cell_hash': digest(cell), 'recipe_cell_hash': digest({'recipe': recipe, 'K': 1, 'seed': cell['seed']}),
        'attempt_id': attempt['attempt_id'], 'run_id': attempt['run_id'],
        'runtime_binding': binding, 'source_code_hash': study['code_hash'], 'selected_model_hash': digest(model),
        'boundary_report_hash': digest(report) if complete else None, 'complete': complete,
        'scope': study['admission']['mode']+'-native-k1-boundary-only; not-scientific-qualification',
        'source_preflight': source_preflight,
        'source_preflight_hash': digest(source_preflight) if source_preflight is not None else None,
        'recovery': 'restart-complete-original-K1-boundary-on-each-original-arm-attempt', **FLAGS}
    _require(len(encode(value)) <= 16*1024, 'native K1 execution byte quota exceeded')
    return value


def validate_execution(record, study, cell, attempt, model, report, *, admission_hash=None):
    _require(type(record) is dict, 'actual native K1 boundary record missing')
    expected = execution_record(study, cell, attempt, model, report,
                                source_preflight=record.get('source_preflight'))
    if study['admission']['mode'] == 'pilot' and admission_hash is not None:
        _require(expected['source_preflight']['admission_hash'] == admission_hash,
                 'K1 preflight belongs to a different actual admission')
    _same_declaration(record, expected, lambda: None)
    return expected['complete']


def attach_failure(diagnostic, study, cell, attempt, *, source_preflight=None):
    from .evidence import validate_initialization_history
    chosen = validate_initialization_history(study, cell, diagnostic, diagnostic['generated_inputs'])
    _require(chosen is not None, 'K1 boundary failure has no selected fit')
    diagnostic['k1_boundary_execution'] = execution_record(study, cell, attempt, chosen['model'],
        diagnostic['k1_boundary_checks'], source_preflight=source_preflight)
