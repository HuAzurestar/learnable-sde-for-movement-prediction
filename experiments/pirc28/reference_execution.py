"""Bound actual native worker batch completions, not scientific accuracy.

Metadata made by these helpers is NOT standalone proof of execution. It gains
execution provenance only inside the canonical authorized source artifact,
checked against original admission, command, code, native stop and every cost.
No reference resampling or fitting is performed by the owner checker.
"""

import re
from pathlib import Path

from infrastructure.research_store import digest, encode
from .approximation import ApproximationSpec, RECIPE_SCHEMA
from .approximation_evidence import _integer, _keys, _require, reference_seed

SCHEMA = 'pirc28-native-reference-execution-v1'
SOURCE_SCHEMA = 'pirc28-native-reference-execution-v2'
PREFLIGHT_FIELDS = {'study_definition_hash', 'attempt_id', 'run_id', 'admission_hash',
                    'qualified_identifiability', 'real_data_authorized'}
BATCH_FIELDS = {'batch_index', 'seed_entropy', 'probabilities', 'means',
                'log_likelihood_trace', 'minimum_ess_fraction', 'jump_count'}
COMPLETION_FIELDS = {'path_index', 'batch_index', 'seed_entropy', 'summary_hash'}


def batch_completion_record(path_index, published_batch):
    """Digest the exact public batch projection AFTER actual sampler return."""
    _integer(path_index, 0, 63)
    _keys(published_batch, BATCH_FIELDS, 'completed reference batch')
    _integer(published_batch['batch_index'], 0, 31)
    seed = published_batch['seed_entropy']
    _require(type(seed) is list and len(seed) == 3 and all(type(item) is int for item in seed),
             'typed completed batch entropy required')
    return {'path_index': path_index, 'batch_index': published_batch['batch_index'],
            'seed_entropy': seed, 'summary_hash': digest(published_batch)}


def _validate_completions(completed, recipe, seed):
    batches = recipe['approximation_spec']['event_driven_reference']['independent_batches']
    _require(type(completed) is list and len(completed) <= recipe['test_paths']*batches,
             'bounded completed batch roster required')
    for index, record in enumerate(completed):
        _keys(record, COMPLETION_FIELDS, 'completion record')
        path, batch = divmod(index, batches)
        _integer(record['path_index'], path, path)
        _integer(record['batch_index'], batch, batch)
        entropy = record['seed_entropy']
        _require(type(entropy) is list and len(entropy) == 3
            and all(type(item) is int for item in entropy) and entropy == [seed, path, batch],
            'original fixed path/batch seed differs')
        value = record['summary_hash']
        _require(type(value) is str and len(value) == 64 and re.fullmatch('[0-9a-f]{64}', value),
                 'bounded completion summary hash required')


def _report_completions(report, recipe):
    paths = report.get('paths')
    count = recipe['approximation_spec']['event_driven_reference']['independent_batches']
    _require(type(paths) is list and len(paths) == recipe['test_paths'], 'complete reference path roster required')
    records = []
    for index, path in enumerate(paths):
        _require(type(path) is dict, 'reference path object required')
        _integer(path.get('path_index'), index, index)
        batches = path.get('reference_batch_summaries')
        _require(type(batches) is list and len(batches) == count, 'complete reference batch roster required')
        records.extend(batch_completion_record(index, batch) for batch in batches)
    return records


def execution_record(study, cell, attempt, model, report, completed, *, source_preflight=None):
    """Build worker evidence / owner's expected metadata; not grant authority."""
    recipe = study['switching_recipe']
    for name, size in (('attempt_id', 32), ('run_id', 64)):
        value = attempt.get(name)
        _require(type(value) is str and len(value) == size and re.fullmatch('[0-9a-f]+', value),
                 'typed original native '+name+' required')
    source = study.get('code_hash')
    _require(type(source) is str and len(source) == 64 and re.fullmatch('[0-9a-f]{64}', source),
             'typed original native source hash required')
    binding = study.get('runtime_binding')
    _keys(binding, {'root', 'store_id'}, 'original native runtime binding')
    _require(type(binding['root']) is str and 0 < len(binding['root']) <= 4096
        and Path(binding['root']).is_absolute() and type(binding['store_id']) is str
        and 0 < len(binding['store_id']) <= 128, 'typed original native runtime identity required')
    mode = study['admission']['mode']
    _require(recipe['schema_version'] == RECIPE_SCHEMA and cell['fit_K'] == 2
        and mode in ('fixture', 'pilot'), 'native reference source scope differs')
    if mode == 'fixture':
        _require(source_preflight is None, 'engineering reference cannot borrow scientific preflight')
    else:
        _keys(source_preflight, PREFLIGHT_FIELDS, 'original source preflight')
        _require(source_preflight['study_definition_hash'] == study.get('pirc28_study_definition_hash')
            and source_preflight['attempt_id'] == attempt['attempt_id']
            and source_preflight['run_id'] == attempt['run_id']
            and all(type(source_preflight[name]) is str and re.fullmatch('[0-9a-f]{64}', source_preflight[name])
                    for name in ('study_definition_hash', 'admission_hash'))
            and source_preflight['qualified_identifiability'] is False
            and source_preflight['real_data_authorized'] is False,
            'original scientific source preflight identity differs')
    spec = ApproximationSpec.from_dict(recipe['approximation_spec'])
    seed = reference_seed(recipe, cell['seed'])
    _validate_completions(completed, recipe, seed)
    complete = report is not None
    if complete:
        _require(type(report) is dict and report.get('state') in ('NUMERICAL_CHECKS_PASSED', 'INFERENCE_INAPPLICABLE')
            and report.get('spec_hash') == spec.spec_hash and report.get('reference_seed') == seed
            and report.get('model_hash') == digest(model)
            and report.get('origin_index') == recipe['forecast_origin_index'],
            'native reference report identity differs')
        _require(digest(completed) == digest(_report_completions(report, recipe)),
                 'actual completed batches differ from complete published report')
    record = {'schema_version': SCHEMA, 'scope': 'fixture-native-execution-only; not-scientific-qualification',
        'study_id': study['study_id'], 'spec_hash': digest(study), 'cell_hash': digest(cell),
        'attempt_id': attempt['attempt_id'], 'run_id': attempt['run_id'],
        'runtime_binding': study['runtime_binding'], 'source_code_hash': study['code_hash'],
        'recipe_cell_hash': digest({'recipe': recipe, 'K': 2, 'seed': cell['seed']}),
        'selected_model_hash': digest(model), 'approximation_spec_hash': spec.spec_hash,
        'reference_seed': seed, 'origin_index': recipe['forecast_origin_index'],
        'reference_report_hash': None if report is None else digest(report),
        'completed_batches': completed, 'complete': complete,
        'recovery': 'restart-all-fixed-batches-on-each-original-arm-attempt; no-partial-reuse',
        'managed_evidence_bound': False, 'qualified_identifiability': False, 'real_data_authorized': False}
    if mode == 'pilot':
        record.update(schema_version=SOURCE_SCHEMA,
            scope='preregistered-synthetic-pilot-native-execution-only; not-scientific-qualification',
            source_preflight=source_preflight, source_preflight_hash=digest(source_preflight))
    limit = 16384+256*recipe['test_paths']*spec.to_dict()['event_driven_reference']['independent_batches']
    _require(len(encode(record)) <= limit, 'native reference completion bytes exceed quota')
    return record


def validate_reference_execution(record, study, cell, attempt, model, report, *, admission_hash=None):
    """Bounded owner metadata; never repeat sampling or trust complete=true."""
    _require(type(record) is dict, 'actual native reference execution record missing')
    expected = execution_record(study, cell, attempt, model, report, record.get('completed_batches'),
                                source_preflight=record.get('source_preflight'))
    if study['admission']['mode'] == 'pilot' and admission_hash is not None:
        _require(expected['source_preflight']['admission_hash'] == admission_hash,
                 'native source preflight belongs to a different actual admission')
    _require(digest(record) == digest(expected), 'native reference execution/source binding differs')
    return expected['complete']


def attach_reference_failure(diagnostic, study, cell, attempt, completed, *, source_preflight=None):
    """Worker-only metadata attachment; caller still needs actual native provenance."""
    from .evidence import validate_initialization_history
    chosen = validate_initialization_history(study, cell, diagnostic, diagnostic['generated_inputs'])
    _require(chosen is not None, 'reference failure has no selected fit')
    diagnostic['reference_execution'] = execution_record(study, cell, attempt, chosen['model'],
        diagnostic['approximation_checks'], completed, source_preflight=source_preflight)
