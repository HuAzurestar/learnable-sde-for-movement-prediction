"""Guarded restoration of the original partial run's complete saved inputs.

No raw trajectory/development reload, new eligibility selection, fit or
prediction. Current authority and durable access starts are compulsory even
when values come from saved files. This is called inside metered bootstrap.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from . import final_eval_guard as guard, formal_partial_predecessor as predecessor
from . import formal_maps, formal_training
from .configurations import configuration_encoder, terrain_configurations
from .features import CanonicalEncoder
from .formal_closed import ClosedOutputs
from .formal_input_interruption import context_semantic_identity, eligibility_semantic_identity
from .formal_input_work import PreparedInputWork, _owned_record, read_input_work, restore_input_context
from .formal_reanalysis import _access, input_context
from .formal_pinned_metadata import OwnedRecord, record_bytes, record_payload
from .protocol_core import canonical, envelope, file_hash, publish, read_json, under, unpack


def _source_input(binding, protocol, matrix):
    b, costs, science, original, contract = predecessor._history(binding)
    if protocol != original['protocol'] or matrix != original['matrix']:
        raise ValueError('partial input restoration must preserve original protocol/matrix')
    works = [w for w in unpack(matrix)['workloads'] if w['kind'] == 'input_qualification_and_population']
    if len(works) != 1:
        raise ValueError('one original input owner required')
    item = ClosedOutputs(b['ledger_directory'], contract=contract, tip=science['committed_tip']).read(works[0])
    completed = science['completed_sources'][works[0]['work_id']]
    if (item is None or any(item[k] != completed[k] for k in
            ('result_sha256', 'settlement_sha256', 'observation_sha256', 'controller_elapsed_ns'))
            or item['manifest']['context_sha256'] != science['context_sha256']):
        raise ValueError('saved input must have its independently verified original completion')
    return dict(item=item, original=original, costs=costs, science=science, root=Path(b['ledger_directory']))


def _qualification(access, *, source, protocol, execution, output):
    """Metadata only, before position/feature value restoration."""
    if access.access_kind != 'final_eval_eligibility' or access.population_sha256 is not None:
        raise ValueError('eligibility-only access required')
    item, original = source['item'], source['original']
    qpath, qualified = _owned_record(item['manifest'], item['directory'], 'qualification_path', 'qualification_sha256')
    q = unpack(qualified)
    report_path = under(qpath.parent, q['eligibility_path'])
    report_file_hash = file_hash(report_path)
    old_report = read_json(report_path, expected_file_sha256=report_file_hash)
    old_population = read_json(under(qpath.parent, q['population_path']))
    unpack(old_report, expected_sha256=q['eligibility_sha256'])
    guard.validate_population(old_population, old_report, expected_sha256=q['population_sha256'],
        protocol=protocol, execution=original['execution'])
    if (q['protocol_sha256'] != protocol['sha256'] or q['execution_sha256'] != original['execution']['sha256']
            or q['approval_sha256'] != source['costs']['approval_sha256']
            or q['position_feature_value_prediction_metric_reads'] != 0):
        raise ValueError('saved full qualification differs from original scope')
    scope = dict(protocol_sha256=protocol['sha256'], execution_sha256=original['execution']['sha256'],
                 approval_sha256=q['approval_sha256'])
    _access(under(source['root'], 'access'), q['access_started_sha256'], scope=scope,
            kind='final_eval_eligibility', population_required=False)
    report = envelope(dict(unpack(old_report), execution_sha256=execution['sha256']))
    population = envelope(guard.population_contract(report, protocol=protocol, execution=execution))
    if eligibility_semantic_identity(report, population) != eligibility_semantic_identity(old_report, old_population):
        raise ValueError('restoration changed original denominator, dispositions or ordered selection')
    # Private, complete immutable meanings, not a caller-provided PASS or hash.
    # Avoid rehashing the entire denominator for each eligible window. Original
    # protocol/report bytes are checked again before any successful handoff.
    fixed_protocol, fixed_report = OwnedRecord(protocol), OwnedRecord(old_report)
    input_binding = record_payload(fixed_protocol)['dataset_inputs']['sha256']
    fixed_report_payload = record_payload(fixed_report)
    rule = fixed_report_payload['eligibility_rule_sha256']
    rows = fixed_report_payload['rows']
    windows = {r['sample_id']: 'windows/'+r['window_sha256']+'.json' for r in rows if r['eligible']}
    if q['windows'] != windows or q['denominator_samples'] != len(rows) or q['eligible_samples'] != len(windows):
        raise ValueError('complete original qualification/window inventory required')
    from .formal_inputs import validate_window
    for row in rows:
        if not row['eligible']:
            continue
        record = read_json(under(qpath.parent, windows[row['sample_id']]))
        w = validate_window(record, expected_sha256=row['window_sha256'])
        if (any(w['sample'][k] != row[k] for k in ('sample_id', 'segment_id', 'independent_block_id', 'split'))
                or w['input_binding_sha256'] != input_binding
                or w['eligibility_rule_sha256'] != rule):
            raise ValueError('original source window changed')
        publish(output/'windows', unpack(record))
    if (record_bytes(protocol) != record_bytes(fixed_protocol)
            or record_bytes(old_report) != record_bytes(fixed_report)
            or file_hash(report_path) != report_file_hash):
        raise ValueError('qualification protocol/report changed before handoff')
    rpath, report = publish(output/'eligibility', unpack(report))
    ppath, population = publish(output/'population', unpack(population))
    current = dict(q, execution_sha256=execution['sha256'], approval_sha256=access.approval_sha256,
        access_started_sha256=access.access_started_sha256,
        eligibility_path=rpath.relative_to(output).as_posix(), eligibility_sha256=report['sha256'],
        population_path=ppath.relative_to(output).as_posix(), population_sha256=population['sha256'])
    path, record = publish(output, current)
    return dict(population=population, qualification=record, qualification_path=path,
        population_paths=dict(population_path=ppath, population_sha256=population['sha256'], eligibility_path=rpath))


def _positions(access, *, source, protocol, matrix, qualified):
    if access.access_kind != 'final_eval_positions' or access.population_sha256 != qualified['population']['sha256']:
        raise ValueError('current population-bound position access required')
    item = source['item']
    args, _, _ = read_input_work(item['manifest'], directory=item['directory'], protocol=protocol,
        execution=source['original']['execution'], matrix=matrix,
        approval_sha256=source['costs']['approval_sha256'], access_journal=under(source['root'], 'access'))
    _, context = _owned_record(item['manifest'], item['directory'], 'context_path', 'context_sha256')
    population = qualified['population']
    positions = replace(args['positions'],
        prefixes=tuple(replace(p, population_sha256=population['sha256']) for p in args['positions'].prefixes),
        population_sha256=population['sha256'], access_started_sha256=access.access_started_sha256)
    return args, context, positions


def _training(access, *, original, protocol, execution, population, snapshot):
    if access.access_kind != 'final_eval_features' or access.population_sha256 != population['sha256']:
        raise ValueError('current population-bound feature access required')
    # Unfitted canonical transforms/metadata only; no 485 development values,
    # fitting, dummy development object or final truth reaches fit consumers.
    encoder = CanonicalEncoder.frozen_pirc22(Path(snapshot))
    encoders = {name: configuration_encoder(encoder, name) for name in terrain_configurations()}
    old = unpack(original['input_identity'])
    if old['configuration_columns'] != {k:list(e.columns) for k,e in encoders.items()}:
        raise ValueError('recovered canonical configuration columns changed')
    identity = envelope(dict(old, execution_sha256=execution['sha256'], approval_sha256=access.approval_sha256,
        population_sha256=population['sha256'], access_started_sha256=access.access_started_sha256))
    return formal_training.RestoredFitInputs(encoders, original['prior'], identity)


def _maps(access, *, original, protocol, execution, population, data_root):
    if access.access_kind != 'final_eval_features' or access.population_sha256 != population['sha256']:
        raise ValueError('current population-bound map access required')
    old = unpack(original['map_catalog'])
    if (old['protocol_sha256'] != protocol['sha256']
            or old['snapshot_inventory_sha256'] != unpack(protocol)['dataset_inputs']['snapshot']['content_inventory_sha256']
            or old['parent_columns'] != list(formal_maps.PARENT_COLUMNS)):
        raise ValueError('saved entire-snapshot map catalog differs')
    root = Path(data_root).resolve()
    policy = unpack(protocol)['dataset_inputs']['online_maps']
    parents = sorted(old['registered_parent_witnesses'])
    query, sources = formal_maps._open_query(root, policy, parents, execution)
    catalog = envelope(dict(old, execution_sha256=execution['sha256'], approval_sha256=access.approval_sha256,
        population_sha256=population['sha256'], access_started_sha256=access.access_started_sha256))
    maps = formal_maps.RegisteredMaps(query, catalog, root=root, policy=policy, parents=parents, execution=execution)
    try:
        if sources != old['source_sha256']:
            raise ValueError('map backend source identity changed')
        maps.observation()  # Full static policy/asset identity, zero value queries.
        return maps
    except BaseException:
        maps.close()
        raise


def restore_partial_inputs(*, inputs, binding, output_directory):
    """Actual metered bootstrap calls this; no new authority is synthesized."""
    output = Path(output_directory).resolve()
    output.mkdir(exist_ok=True)
    base = dict(protocol=inputs.protocol, execution=inputs.execution, **inputs.authority)
    source = guard.guarded_call(access_kind='final_eval_eligibility', **base,
        operation=lambda access: _restore_qualification(access, inputs, binding, output))
    original, old_context, positions = guard.guarded_call(access_kind='final_eval_positions', **base,
        **source['qualified']['population_paths'], operation=lambda access: _positions(access,
            source=source, protocol=inputs.protocol, matrix=inputs.matrix, qualified=source['qualified']))
    population = source['qualified']['population']
    paths = source['qualified']['population_paths']
    training = guard.guarded_call(access_kind='final_eval_features', **base, **paths,
        operation=lambda access: _training(access, original=original, protocol=inputs.protocol,
            execution=inputs.execution, population=population, snapshot=inputs.paths['snapshot']))
    maps = guard.guarded_call(access_kind='final_eval_features', **base, **paths,
        operation=lambda access: _maps(access, original=original, protocol=inputs.protocol,
            execution=inputs.execution, population=population, data_root=inputs.paths['data_root']))
    try:
        args = dict(protocol=inputs.protocol, execution=inputs.execution, matrix=inputs.matrix,
            input_identity=training.identity, population=population, positions=positions,
            prior=training.prior, map_catalog=maps.catalog)
        context = input_context(**args)
        restore_input_context(context, protocol=inputs.protocol, execution=inputs.execution,
            matrix=inputs.matrix, approval_sha256=inputs.authority['approval_sha256'])
        if context_semantic_identity(context) != context_semantic_identity(old_context):
            raise ValueError('current guarded context changed original scientific inputs')
        path, context = publish(output, unpack(context))
        manifest = dict(context_path=str(path), context_sha256=context['sha256'],
            qualification_path=str(source['qualified']['qualification_path']),
            qualification_sha256=source['qualified']['qualification']['sha256'])
        source_ref = predecessor.reference(Path(source['item']['manifest']['context_path']),
                                           expected_sha256=old_context['sha256'])
        return PreparedInputWork(args, training, maps, paths, manifest, source_ref)
    except BaseException:
        maps.close()
        raise


def _restore_qualification(access, inputs, binding, output):
    source = _source_input(binding, inputs.protocol, inputs.matrix)
    if inputs.execution['sha256'] == source['original']['execution']['sha256']:
        raise ValueError('distinct successor execution required')
    source['qualified'] = _qualification(access, source=source, protocol=inputs.protocol,
        execution=inputs.execution, output=output/'qualification')
    return source
