"""Public aggregates AFTER final checkpoint audit, never a restart gate.

Reuse original public projection/bootstrap/Holm kernels without reviving the
retired ClosedOutputs restoration pipeline. Load the saved context once; no
new fits/predictions/map queries, forecast-array reopening, metric recomputation
or fabricated legacy admission/human acceptance.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

from .checkpoint_audit import (COUNTS, VERSION as AUDIT_VERSION, QUERY_ACCOUNTING_SCOPE,
    validate_query_accounting, require_complete, sources)
from .checkpoint_resume import load
from .checkpoint_scoring import CheckpointScores, start_access
from .checkpoint_state import atomic_json, single_writer
from .comparison_registry import ORIGIN_MODES
from .formal_export import (VERSION, MAX_BYTES, LIMITS, _rename, _statistical_view,
    _public_scores, _public_mechanisms, _diagnostic_projection, _runtime_projection, replay_public)
from .formal_forecast_records import origin_stream_id
from .formal_paired import FAMILIES
from .protocol_core import canonical, digest, file_hash, publish, read_json, under, unpack


def bound_record(binding, root):
    path = Path(binding['path'])
    path = path if path.is_absolute() else under(root, binding['path'])
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('checkpoint output pointer escapes its result root')
    record = read_json(path, expected_file_sha256=binding['file_sha256'], max_bytes=MAX_BYTES)
    unpack(record, expected_sha256=binding['content_sha256'])
    return record


def verify_sources(saved, state, analysis_record, audit_record, *, score_index_sha256):
    """Bind export to completed fresh replay, not a producer PASS flag alone."""
    require_complete(saved, state)
    analysis, audit = unpack(analysis_record), unpack(audit_record)
    settings = saved.checkpoint_settings
    scope = dict(protocol_sha256=saved.protocol['sha256'], execution_sha256=saved.execution['sha256'],
        matrix_sha256=saved.matrix['sha256'], population_sha256=saved.population['sha256'])
    if any(analysis.get(k) != v or audit.get(k) != v for k,v in scope.items()):
        raise ValueError('export source scientific scope changed')
    if (audit.get('schema_version') != AUDIT_VERSION
            or audit.get('independent_saved_output_reanalysis') is not True
            or audit.get('producer_cache_used_for_numbers') is not False
            or audit.get('raw_score_rows_recomputed') != 11368
            or audit.get('common_blocks_recomputed') != 58
            or audit.get('counts_by_kind') != COUNTS
            or any(audit.get(k) is not False for k in
                   ('retired_bootstrap_gate_asserted','scientific_claim_authorized','human_accepted'))):
        raise ValueError('full fresh independent checkpoint audit required')
    if (audit['input_context_sha256'] != settings['context_sha256']
            or audit['analysis_sha256'] != analysis_record['sha256']
            or audit['saved_score_index_sha256'] != score_index_sha256
            or analysis['saved_score_index_sha256'] != score_index_sha256
            or audit['checkpoint_inventory_sha256'] != digest(dict(index=saved.index,
                failures=saved.failures, settings_id=settings['settings_id']))):
        raise ValueError('checkpoint/analysis/score sources changed after independent audit')
    raw_analysis = {k:v for k,v in analysis.items() if k not in
        {'checkpoint_cache_sha256','metrics_access_started_sha256','independent_raw_output_replay_completed'}}
    if (audit['recomputed_analysis_payload_sha256'] != digest(raw_analysis)
            or audit['factor_conclusions'] != analysis['factor_conclusions']):
        raise ValueError('analysis is not the freshly replayed numeric source')
    if audit.get('feature_query_accounting_scope') != QUERY_ACCOUNTING_SCOPE:
        raise ValueError('recorded query accounting cannot become per-column or robustness evidence')
    rows = audit['full_work_inventory']
    inventory = {r['work_id']:r for r in rows}
    if (len(inventory) != len(rows) or set(inventory) != set(saved.work)
            or Counter(r['kind'] for r in rows) != COUNTS
            or dict(Counter(r['status'] for r in rows)) != audit['counts_by_status']):
        raise ValueError('complete11659-work audit inventory required')
    for wid,row in inventory.items():
        work = saved.work[wid]
        if row['kind'] != work['kind'] or row['phase'] != work['phase']:
            raise ValueError('audited work descriptor differs from registered matrix')
        if wid in saved.forecasts:
            expected = saved.index.get(wid, {}).get('content_sha256')
            if row['evidence_sha256'] != expected:
                raise ValueError('forecast source differs from final audit inventory')
            axes = {k:work[k] for k in ('matrix','subject','origin_mode','origin_rank','seed','repetition')}
            if row.get('forecast_axes') != axes:
                raise ValueError('forecast cost axes differ from registered matrix')
            seconds = row.get('kernel_prediction_seconds')
            if seconds is not None and (type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0):
                raise ValueError('finite nonnegative kernel prediction time required')
            validate_query_accounting(row.get('feature_query_accounting'))
        elif work['kind'] == 'common_scores':
            if row['evidence_sha256'] != state['blocks'][wid]['content_sha256']:
                raise ValueError('score source differs from final audit inventory')
    return inventory


def project(saved, cache, state, analysis_record, audit_record, *, progress, access_sha256):
    scores = CheckpointScores(scorer=cache, root=cache.cache_root, index=state['blocks'],
        access_journal=cache.cache_root.parent.parent/'metric-access')
    inventory = verify_sources(saved, state, analysis_record, audit_record,
                               score_index_sha256=scores.identity['sha256'])
    analysis, audit = unpack(analysis_record), unpack(audit_record)
    if (analysis['checkpoint_cache_sha256'] != cache.cache_identity['sha256']
            or state['cache_sha256'] != cache.cache_identity['sha256']):
        raise ValueError('export cache identity differs from audited analysis')
    populations = {m:analysis['modes'][m]['expected_origins_by_block'] for m in ORIGIN_MODES}
    actual = {m:{c.independent_block_id:[origin_stream_id(c)] for c in saved.cases[m]} for m in ORIGIN_MODES}
    if populations != actual:
        raise ValueError('public projection cannot change the registered origin population')
    names = {b:f'block-{i+1:03d}' for i,b in enumerate(sorted({b for p in populations.values() for b in p}))}
    for mode,population in populations.items():
        for block,origins in population.items():
            for i,origin in enumerate(origins):
                names[origin] = f'{mode}:{names[block]}:origin-{i+1:03d}'
    dispositions = {r['work_id']:r for r in analysis['score_work_dispositions']}
    if len(dispositions) != 58 or set(dispositions) != set(cache.work):
        raise ValueError('all58 audited score dispositions required')
    rows, checked_access = [], set()
    for wid,work in cache.work.items():
        binding = state['blocks'][wid]
        key = binding['metrics_access_started_sha256']
        if key not in checked_access:
            scores._access(key); checked_access.add(key)
        record = bound_record(binding, cache.cache_root)
        block = unpack(record)
        if (record['sha256'] != dispositions[wid]['score_artifact_sha256']
                or record['sha256'] != inventory[wid]['evidence_sha256']
                or block['work_id'] != wid or block['expected_rows'] != 196 or len(block['rows']) != 196
                or block['checkpoint_cache_sha256'] != cache.cache_identity['sha256']
                or block['metrics_access_started_sha256'] != key):
            raise ValueError('complete score block differs from independently audited source')
        dependencies = {w['work_id']:w for w in cache.dependencies[wid]}
        ids = [r['forecast_work_id'] for r in block['rows']]
        if len(ids) != len(set(ids)) or set(ids) != set(dependencies):
            raise ValueError('all196 unique registered score dependencies required')
        for r in block['rows']:
            dependency = dependencies[r['forecast_work_id']]
            fields = {'matrix':'matrix','configuration':'subject','seed':'seed','origin_mode':'origin_mode',
                      'origin_rank':'origin_rank','scientific':'scientific'}
            if any(r[k] != dependency[v] for k,v in fields.items()) or r['partition'] != 'final_eval':
                raise ValueError('score row changed its registered scientific axes')
            case = saved.case(dependency)
            if (r['origin_id'] != (None if case is None else origin_stream_id(case))
                    or r['independent_block_id'] != (None if case is None else case.independent_block_id)):
                raise ValueError('score row changed its registered origin identity')
            row = {k:deepcopy(r[k]) for k in ('forecast_work_id','matrix','configuration','seed','origin_mode',
                'origin_rank','partition','scientific','status','score_m','saved_forecast_sha256')}
            row.update(origin_id=None if r['origin_id'] is None else names[r['origin_id']],
                independent_block_id=None if r['independent_block_id'] is None else names[r['independent_block_id']],
                source_score_sha256=record['sha256'], source_reason_sha256=digest(r['reason']),
                scores=_public_scores(r['scores']))
            rows.append(row)
    if len(rows) != 11368 or len({r['forecast_work_id'] for r in rows}) != 11368:
        raise ValueError('full11368 public score denominator required')
    inputs = dict(decision_policy_sha256=analysis['decision_policy_sha256'], populations=_rename(populations,names),
        mechanisms=_public_mechanisms(analysis), mechanism_details=_diagnostic_projection(analysis,names),
        terrain_ownership=deepcopy(analysis['terrain_ownership']), score_rows=rows)
    replayed = replay_public(inputs)
    for mode in ORIGIN_MODES:
        for family in FAMILIES:
            original = _rename(_statistical_view(analysis['modes'][mode]['families'][family]),names)
            actual = _statistical_view(replayed['modes'][mode]['families'][family])
            for key in ('missing_rows','failed_rows'):
                original[key] = sorted(original[key])
            if canonical(original) != canonical(actual):
                raise ValueError('public aggregates differ from audited statistical inference')
    if replayed['factor_conclusions'] != analysis['factor_conclusions']:
        raise ValueError('public factor verdicts differ from independent audit')
    works = [w for w in saved.work.values() if w['kind']=='aggregate_export']
    result = dict(schema_version=VERSION, work_id=works[0]['work_id'], protocol_sha256=saved.protocol['sha256'],
        execution_sha256=saved.execution['sha256'], matrix_sha256=saved.matrix['sha256'],
        population_sha256=saved.population['sha256'], approval_sha256=saved.checkpoint_settings['approval_sha256'],
        audit_sha256=audit_record['sha256'], analysis_sha256=analysis_record['sha256'],
        scope='Full checkpoint public aggregates; ordinal origin labels, not raw trajectories or new empirical replication.',
        inputs=inputs, recomputed=replayed, method_ledger=deepcopy(analysis['method_ledger']),
        history=dict(role='baseline-only',terrain_verdict=None), runtime_replay=_runtime_projection(audit['runtime_replay']),
        audit_summary=dict(full_work_count=11659,counts_by_kind=deepcopy(audit['counts_by_kind']),
            feature_query_accounting_scope=deepcopy(audit['feature_query_accounting_scope']),
            counts_by_status=deepcopy(audit['counts_by_status']),raw_score_rows_recomputed=11368,
            common_blocks_recomputed=58,producer_cache_used_for_numbers=False,
            work_inventory=[dict({k:deepcopy(r[k]) for k in ('work_id','kind','phase','status','role','evidence_sha256')},
                forecast_axes=deepcopy(r.get('forecast_axes')),kernel_prediction_seconds=r.get('kernel_prediction_seconds'),
                feature_query_accounting=deepcopy(r.get('feature_query_accounting')))
                            for r in audit['full_work_inventory']]),
        score_work_dispositions=deepcopy(analysis['score_work_dispositions']),
        budget_snapshot=dict(charged_ns_by_phase=deepcopy(progress['charged_ns_by_phase']),
            generated_forecasts_reserved=progress['generated'],includes_export_completion=False,
            auxiliary_scoring_analysis_audit_costs_included=False,
            scope='Recorded forecast-runner costs including historical stops; not total project cost.'),
        metrics_access_started_sha256=access_sha256,limitations=list(LIMITS),
        retired_bootstrap_gate_asserted=False,raw_trajectories_exported=False,original_identifiers_exported=False,
        scientific_claim_authorized=False,numerically_qualified=False,human_accepted=False,new_forecasts=0,new_fits=0)
    if len(canonical(result)) > MAX_BYTES-1024:
        raise ValueError('public aggregate exceeds fixed128MiB bound')
    return result


def export(directory):
    directory = Path(directory).resolve()
    settings,_ = load(directory)
    access = start_access(directory,settings)
    saved,cache,state = sources(directory)
    analysis_record = bound_record(read_json(cache.cache_root/'analysis.json'),cache.cache_root)
    audit_record = bound_record(read_json(cache.cache_root/'audit.json'),cache.cache_root)
    with single_writer(cache.cache_root/'export-owner'):
        value = project(saved,cache,state,analysis_record,audit_record,
            progress=read_json(directory/'progress.json'),access_sha256=access['sha256'])
        pointer = cache.cache_root/'export.json'
        if pointer.is_file():
            binding = read_json(pointer)
            existing = unpack(bound_record(binding,cache.cache_root))
            # An immutable public snapshot remains its ORIGINAL before-export
            # cost snapshot if a supervisor records later closure overhead.
            # Do not rewrite it or recompute science because a cost increased.
            snapshot = existing['budget_snapshot']
            if (progress := read_json(directory/'progress.json'))['generated'] < snapshot['generated_forecasts_reserved']:
                raise ValueError('checkpoint generation history decreased')
            if any(progress['charged_ns_by_phase'][k] < v for k,v in snapshot['charged_ns_by_phase'].items()):
                raise ValueError('checkpoint cost history decreased')
            expected = dict(value,metrics_access_started_sha256=existing['metrics_access_started_sha256'],
                            budget_snapshot=snapshot)
            if canonical(existing) != canonical(expected):
                raise ValueError('existing export differs from current audited sources')
            return binding
        path,record = publish(cache.cache_root/'public-aggregates',value)
        binding = dict(path=str(path),content_sha256=record['sha256'],file_sha256=file_hash(path))
        atomic_json(pointer,binding)
        return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.directory)),flush=True)


if __name__ == '__main__':
    main()
