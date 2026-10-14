"""Actual guarded input work and independent saved-context restoration.

Called inside the registered input reservation, not a new qualification run or
approval entrypoint. Training values and maps remain with the retained worker;
the controller restores small typed causal/scoring inputs from owned files.
Historical guarded source reads are bound, not claimed to be independently
repeated by deserializing their output. No forecast or model fit occurs here.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import final_eval_guard as guard
from . import formal_eligibility, formal_inputs, formal_maps, formal_training
from .features import EARTH_RADIUS_M, LocalFrame
from .formal_origins import RegisteredVelocityPrior, validate_prior
from .formal_reanalysis import CONTEXT_VERSION, _access, input_context, qualification_evidence
from .method_inputs import SolarConditionField
from .origins import VelocityPrior, causal_prefix, frozen_array
from .protocol_core import canonical, digest, file_hash, publish, read_json, sha256, under, unpack


def restore_input_context(record, *, protocol, execution, matrix, approval_sha256):
    """Restore immutable typed objects and require an exact semantic roundtrip."""
    value = unpack(record)
    guard._fields(value, ('schema_version', 'protocol_sha256', 'execution_sha256', 'matrix_sha256',
        'input_identity', 'population', 'prior_evidence', 'map_catalog', 'causal_prefixes', 'scoring_inputs',
        'truth_passed_to_predictors', 'raw_sources_independently_reloaded'))
    if (value['schema_version'] != CONTEXT_VERSION or value['protocol_sha256'] != protocol['sha256']
            or value['execution_sha256'] != execution['sha256'] or value['matrix_sha256'] != matrix['sha256']
            or value['truth_passed_to_predictors'] is not False or value['raw_sources_independently_reloaded'] is not False):
        raise ValueError('saved input context differs from expected execution/scope')
    p, scope = unpack(protocol), unpack(value['input_identity'])
    contract = p['forecast_contract']['training']
    if (scope['schema_version'] != formal_training.INPUT_VERSION or scope['approval_sha256'] != sha256(approval_sha256)
            or scope['method_input_sha256'] != p['dataset_inputs']['development']['method_input_sha256']
            or scope['sample_counts'] != contract['method_roles']
            or scope['outer_train_windows'] != contract['outer_train_windows']
            or scope['validation_windows'] != contract['validation_windows']
            or scope['terrain_training_policy_sha256'] != contract['terrain']['training_policy_sha256']
            or any(type(scope[k]) is not int or scope[k] != 0 for k in ('final_eval_numeric_training_rows', 'new_predictive_model_fits'))):
        raise ValueError('saved input training population/policy/approval differs')
    evidence = value['prior_evidence']
    prior_data = deepcopy(unpack(evidence)); prior_data.pop('prior_identity')
    rows = prior_data['training_rows']
    if (prior_data['population_identity'] != contract['population_identity']
            or prior_data['outer_train_windows'] != contract['outer_train_windows']
            or len(rows) != contract['outer_train_windows']
            or len({r['sample_id'] for r in rows}) != len(rows)
            or any(r['outer_split'] != 'train' or r['method_role'] not in {'train', 'adapt'} for r in rows)):
        raise ValueError('saved velocity prior changed its outer-train population')
    prior = RegisteredVelocityPrior(VelocityPrior([r['velocity_east_north_mps'] for r in rows], digest(prior_data)), deepcopy(evidence))
    validate_prior(prior)
    prefixes = []
    if not isinstance(value['causal_prefixes'], list) or len(value['causal_prefixes']) > 46:
        raise ValueError('bounded original causal population required')
    for row in value['causal_prefixes']:
        terrain = causal_prefix(row['terrain_prefix_positions_m'], row['prefix_times_seconds'])
        method = causal_prefix(row['method_prefix_positions_m'], row['prefix_times_seconds'])
        frame = LocalFrame(*row['scoring_frame'])
        solar = SolarConditionField(LocalFrame(row['solar']['frame_longitude'], row['solar']['frame_latitude']), row['origin_epoch_ns'])
        if (len(terrain.history_times_seconds) != 3 or terrain.epoch_seconds != 0.
                or not np.array_equal(terrain.position_m, [0., 0.])
                or not np.allclose(frame.from_lonlat(solar.frame.to_lonlat(method.history_positions_m)),
                    terrain.history_positions_m, rtol=0., atol=128*np.finfo(float).eps*EARTH_RADIUS_M)):
            raise ValueError('saved prefix changed original origin, three-point history or coordinate frames')
        prefix = formal_inputs.FinalPrefix(row['sample_id'], row['independent_block_id'], row['split'],
            sha256(row['window_sha256']), sha256(row['population_sha256']), sha256(row['condition_sha256']),
            row['origin_epoch_ns'], terrain, method, frame, solar, frozen_array(row['score_seconds'], (4,)))
        if canonical(prefix.identity()) != canonical(row):
            raise ValueError('saved prefix does not reproduce its causal/UTC/solar identity')
        prefixes.append(prefix)
    scoring = unpack(value['scoring_inputs'])
    targets = tuple(formal_inputs.ScoringTargets(row['sample_id'], sha256(row['window_sha256']),
        frozen_array(row['elapsed_seconds'], (4,)), frozen_array(row['positions_m'], (4, 2))) for row in scoring['targets'])
    positions = formal_inputs.FinalPositionInputs(tuple(prefixes), targets, value['population']['sha256'],
        sha256(scoring['positions_access_started_sha256']))
    result = dict(protocol=deepcopy(protocol), execution=deepcopy(execution), matrix=deepcopy(matrix),
        input_identity=deepcopy(value['input_identity']), population=deepcopy(value['population']),
        positions=positions, prior=prior, map_catalog=deepcopy(value['map_catalog']))
    if canonical(input_context(**result)) != canonical(record):
        raise ValueError('saved input context does not reproduce its complete typed projection')
    return result


def _owned_record(manifest, directory, path_key, hash_key, *, max_bytes=32*1024*1024):
    directory, path = Path(directory).resolve(), Path(manifest[path_key])
    if not path.is_absolute() or not path.resolve().is_relative_to(directory):
        raise ValueError('input manifest artifact must stay in its own output directory')
    relative = path.relative_to(directory).as_posix()
    if str(under(directory, relative)) != str(path):
        raise ValueError('canonical owned input path required')
    record = read_json(path, max_bytes=max_bytes)
    unpack(record, expected_sha256=sha256(manifest[hash_key]))
    return path, record


def read_input_work(manifest, *, directory, protocol, execution, matrix, approval_sha256, access_journal):
    """Controller domain callback: actual saved files, no raw-data reload."""
    guard._fields(manifest, ('context_path', 'context_sha256', 'qualification_path', 'qualification_sha256'))
    _, context = _owned_record(manifest, directory, 'context_path', 'context_sha256')
    qualified_path, qualification = _owned_record(manifest, directory, 'qualification_path', 'qualification_sha256')
    args = restore_input_context(context, protocol=protocol, execution=execution, matrix=matrix, approval_sha256=approval_sha256)
    evidence = qualification_evidence(qualification, directory=qualified_path.parent, protocol=protocol, execution=execution,
        approval_sha256=approval_sha256, population=args['population'], prefixes=args['positions'].prefixes)
    scope, q = unpack(args['input_identity']), unpack(qualification)
    _access(access_journal, q['access_started_sha256'], scope=scope, kind='final_eval_eligibility', population_required=False)
    _access(access_journal, args['positions'].access_started_sha256, scope=scope, kind='final_eval_positions')
    _access(access_journal, scope['access_started_sha256'], scope=scope, kind='final_eval_features')
    _access(access_journal, unpack(args['map_catalog'])['access_started_sha256'], scope=scope, kind='final_eval_features')
    # Prefix timestamps must also match the ORIGINAL selected source metadata,
    # not just be self-consistent with a freely edited serialized secant.
    for prefix in args['positions'].prefixes:
        window = formal_inputs.validate_window(read_json(under(qualified_path.parent, q['windows'][prefix.sample_id])),
                                              expected_sha256=prefix.window_sha256)
        expected_times = [(r['absolute_epoch_ns']-window['origin_epoch_ns'])/1_000_000_000 for r in window['prefix'][-3:]]
        if prefix.terrain_origin.history_times_seconds.tolist() != expected_times:
            raise ValueError('saved prefix history clock differs from original qualified metadata')
    population_paths = dict(population_path=under(qualified_path.parent, q['population_path']),
        population_sha256=q['population_sha256'], eligibility_path=under(qualified_path.parent, q['eligibility_path']))
    return args, population_paths, dict(context_sha256=context['sha256'], qualification=evidence,
        typed_inputs_restored=True, actual_access_events_verified=True, raw_sources_independently_reloaded=False)


def input_verification(work, manifest, directory, *, protocol, execution, matrix, approval_sha256, access_journal):
    """Real domain validator plus the controller's complete byte inventory."""
    from .formal_controller import VERSION, verify_artifacts
    matches = [w for w in unpack(matrix)['workloads'] if w['work_id'] == work['work_id']]
    if len(matches) != 1 or matches[0]['kind'] != 'input_qualification_and_population':
        raise ValueError('registered input work required')
    full = matches[0]
    budget_work = dict(work_id=full['work_id'], phase=full['phase'], generated_forecasts=full['generated_forecasts'],
                       max_active_ns=full['max_active_seconds']*1_000_000_000)
    if canonical(work) not in (canonical(full), canonical(budget_work)):
        raise ValueError('exact registered input work or its budget projection required')
    args, paths, details = read_input_work(manifest, directory=directory, protocol=protocol, execution=execution,
        matrix=matrix, approval_sha256=approval_sha256, access_journal=access_journal)
    directory = Path(directory).resolve()
    qualified_path, qualified = _owned_record(manifest, directory, 'qualification_path', 'qualification_sha256')
    q = unpack(qualified)
    expected = {Path(manifest['context_path']), qualified_path}
    expected.update(under(qualified_path.parent, name) for name in (q['eligibility_path'], q['population_path'], *q['windows'].values()))
    expected = {path.relative_to(directory).as_posix() for path in expected}
    artifacts = {}
    for path in directory.rglob('*'):
        relative = path.relative_to(directory).as_posix()
        checked = under(directory, relative)
        if checked.is_file() and relative != 'result.json':
            artifacts[relative] = dict(file_sha256=file_hash(checked), bytes=checked.stat().st_size)
    if set(artifacts) != expected:
        raise ValueError('input work contains missing/extra files outside its registered context and qualification')
    validation = dict(schema_version=VERSION+'-artifact-verification', work_id=work['work_id'],
        verified=True, artifacts=artifacts, details=details)
    verify_artifacts(validation, work, directory)
    return validation, args, paths


@dataclass
class PreparedInputWork:
    """Worker-only retained state; only the small manifest crosses transport."""
    context: dict
    training: formal_training.RegisteredTrainingInputs
    maps: formal_maps.RegisteredMaps
    population_paths: dict
    manifest: dict
    source_context_reference: dict | None = None

    def close(self):
        self.maps.close()


class InputWork:
    def __init__(self, *, protocol, execution, matrix, authority, release, snapshot, data_root,
                 trajectory_path, development_eligibility_path, continuation_binding=None,
                 partial_predecessor_reference=None):
        self.protocol, self.execution, self.matrix = (deepcopy(x) for x in (protocol, execution, matrix))
        self.authority = deepcopy(authority)
        guard._fields(self.authority, ('approval_path', 'approval_sha256', 'test_path', 'review_path', 'journal_directory'))
        sha256(authority['approval_sha256'])
        self.continuation_binding = deepcopy(continuation_binding)
        self.partial_predecessor_reference = deepcopy(partial_predecessor_reference)
        if partial_predecessor_reference is not None and continuation_binding is not None:
            raise ValueError('partial and input-only predecessors are distinct and exclusive')
        if continuation_binding is not None:
            from .formal_input_interruption import validate_binding_scope
            validate_binding_scope(unpack(continuation_binding))
        self.paths = dict(release=Path(release), snapshot=Path(snapshot), data_root=Path(data_root),
            trajectory_path=Path(trajectory_path), development_eligibility_path=Path(development_eligibility_path))
        self.work = {w['work_id']: w for w in unpack(matrix)['workloads'] if w['kind'] == 'input_qualification_and_population'}
        if len(self.work) != 1 or unpack(matrix)['protocol_sha256'] != protocol['sha256'] or unpack(execution)['matrix_sha256'] != matrix['sha256']:
            raise ValueError('one exact registered input work and execution required')
        self.attempted = set()

    def execute(self, work, *, output_directory):
        if self.partial_predecessor_reference is not None:
            raise ValueError('partial input is already consumed; metered restoration required')
        if self.work.get(work.get('work_id')) != work:
            raise ValueError('exact input work required')
        if work['work_id'] in self.attempted:
            raise ValueError('input work already attempted; no new population or retry')
        self.attempted.add(work['work_id'])
        base = dict(protocol=self.protocol, execution=self.execution, **self.authority)
        output = Path(output_directory).resolve()
        qualified = formal_eligibility.qualify_final_inputs(**base, release=self.paths['release'],
            snapshot=self.paths['snapshot'], output_directory=output/'qualification')
        q = unpack(read_json(qualified['result_path']), expected_sha256=qualified['result_sha256'])
        qdir = Path(qualified['result_path']).parent
        population_paths = dict(population_path=under(qdir, q['population_path']), population_sha256=q['population_sha256'],
                                eligibility_path=under(qdir, q['eligibility_path']))
        population = read_json(population_paths['population_path'])
        if self.continuation_binding is not None:
            from .formal_input_interruption import verify_continued_population
            verify_continued_population(read_json(population_paths['eligibility_path']), population)
        positions = formal_inputs.load_final_positions(**base, **population_paths, qualification_path=qualified['result_path'],
            qualification_sha256=qualified['result_sha256'], release=self.paths['release'], data_root=self.paths['data_root'])
        training = formal_training.prepare_formal_training(**base,
            population_path=population_paths['population_path'], population_sha256=population_paths['population_sha256'],
            final_eligibility_path=population_paths['eligibility_path'], **self.paths)
        maps = formal_maps.prepare_formal_maps(**base, **population_paths, snapshot=self.paths['snapshot'], data_root=self.paths['data_root'])
        try:
            args = dict(protocol=self.protocol, execution=self.execution, matrix=self.matrix, input_identity=training.identity,
                population=population, positions=positions, prior=training.prior, map_catalog=maps.catalog)
            context = input_context(**args)
            # Validate the durable roundtrip before returning the worker result.
            # This is repeated from saved bytes by the independent controller.
            restore_input_context(context, protocol=self.protocol, execution=self.execution, matrix=self.matrix,
                                  approval_sha256=self.authority['approval_sha256'])
            if self.continuation_binding is not None:
                from .formal_input_interruption import verify_continued_context
                verify_continued_context(context)
            path, context = publish(output, unpack(context))
            manifest = dict(context_path=str(path), context_sha256=context['sha256'],
                qualification_path=qualified['result_path'], qualification_sha256=qualified['result_sha256'])
            return PreparedInputWork(args, training, maps, population_paths, manifest)
        except BaseException:
            maps.close()
            raise

    def restore_partial(self, binding, *, output_directory):
        """Distinct recovery path: original input work is already consumed."""
        if self.attempted or self.continuation_binding is not None:
            raise ValueError('partial restoration cannot retry or use the old input-only continuation')
        self.attempted.update(self.work)
        from .formal_partial_inputs import restore_partial_inputs
        return restore_partial_inputs(inputs=self, binding=binding, output_directory=output_directory)
