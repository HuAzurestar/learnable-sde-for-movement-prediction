"""Public-safe execution census over captured checkpoint metadata only.

Never restores models, queries maps, loads arrays, rescores, bootstraps or
changes producer state. Index completeness is NOT score-payload validation,
the independent output audit, numerical qualification or final acceptance.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .checkpoint_resume import load
from .protocol_core import digest, file_hash, read_json

POLICY = Path(__file__).with_name('plans')/'pre-evaluation-decision-policy-v1.json'
MODES = {'causal_prefix': 46, 'known_velocity': 6, 'point_only': 6}
FORECAST_KINDS = {'scientific_forecast', 'same_grid_reference', 'inertial_path',
                  'forecast_replay', 'runtime_cold', 'runtime_warmup', 'runtime_warm'}


def registered_design(policy):
    if digest({k:v for k,v in policy.items() if k!='sha256'}) != policy['sha256']:
        raise ValueError('frozen decision policy changed')
    families = {name: rule['contrasts'] for name,rule in policy['family_rules'].items()}
    methods = {c for name,pairs in families.items() if name.startswith('method-')
               for pair in pairs.values() for c in pair} | {'arm-16/full','arm-06/dt60'}
    terrain = {c for name,pairs in families.items() if name.startswith('weighted-es-')
               for pair in pairs.values() for c in pair}
    seeds = tuple(policy['parameters']['seeds']['value'])
    if (len(families)!=7 or len(methods)!=28 or len(terrain)!=10 or
            seeds!=tuple(range(20260814,20260819))):
        raise ValueError('original seven families, 38 configurations and five seeds required')
    return families, {'NEX326-methods': methods, 'terrain': terrain}, seeds


def source_binding(wid, imported, progress):
    binding = None
    if wid in imported:
        row = imported[wid]['manifest']
        binding = dict(path=row['artifact_path'],content_sha256=row['artifact_sha256'])
    for state in (progress['failures'],progress['completed']):
        row = state.get(wid)
        if isinstance(row,dict) and row.get('artifact_path'):
            binding = dict(path=row['artifact_path'],content_sha256=row['artifact_sha256'])
    return digest(dict(binding=binding,failure=progress['failures'].get(wid)))


def counts(rows):
    forecasts = Counter(r['forecast'] for r in rows)
    scores = Counter(r['score_index'] for r in rows)
    failed = forecasts['failed'] or scores['failed'] or scores['unavailable'] or scores['invalid']
    complete = forecasts['success']==len(rows) and scores['success']==len(rows)
    return dict(expected=len(rows),forecast_success=forecasts['success'],
        forecast_failed=forecasts['failed'],forecast_runnable=forecasts['runnable'],
        score_index_success=scores['success'],score_index_failed=scores['failed'],
        score_index_unavailable=scores['unavailable'],score_index_invalid=scores['invalid'],
        score_index_missing=scores['missing'],
        disposition=('unavailable_failed' if failed else
                     'index_complete_validation_pending' if complete else 'pending_missing'),
        score_payloads_independently_validated=False,scientific_claim_authorized=False)


def census(settings, imported, progress, index, policy):
    """Pure captured-metadata reduction, with the original Cartesian denominator."""
    families, configurations, seeds = registered_design(policy)
    if progress['settings_id']!=settings['settings_id']:
        raise ValueError('predictor settings identity differs')
    completed = set(imported)|set(progress['completed'])
    failed = set(progress['failures'])
    if completed & failed:
        raise ValueError('a forecast cannot be both successful and failed')
    works = [w for w in settings['workloads'] if w['kind']=='scientific_forecast']
    expected = {(matrix,c,mode,rank,seed) for matrix,configs in configurations.items()
                for c in configs for mode,n in MODES.items() for rank in range(n) for seed in seeds}
    keys = [(w['matrix'],w['subject'],w['origin_mode'],w['origin_rank'],w['seed']) for w in works]
    if len(works)!=11020 or len(set(w['work_id'] for w in works))!=11020 or set(keys)!=expected or len(set(keys))!=11020:
        raise ValueError('full original configuration/origin/seed inventory required; no subset')
    rows=[]
    for w in works:
        wid=w['work_id'];entry=index['rows'].get(wid)
        forecast='success' if wid in completed else 'failed' if wid in failed else 'runnable'
        score='missing'
        if entry is not None:
            score=entry.get('status') if isinstance(entry,dict) else 'invalid'
            if (score not in {'success','failed','unavailable'} or
                    entry.get('source_sha256')!=source_binding(wid,imported,progress) or
                    forecast=='runnable' or (score=='success' and forecast!='success')):
                score='invalid'
        rows.append(dict(matrix=w['matrix'],origin_mode=w['origin_mode'],
                         configuration=w['subject'],forecast=forecast,score_index=score))
    groups=[dict(matrix=matrix,origin_mode=mode,independent_blocks=MODES[mode],
        **counts([r for r in rows if r['matrix']==matrix and r['origin_mode']==mode]))
        for matrix in configurations for mode in MODES]
    config_rows=[dict(matrix=matrix,origin_mode=mode,configuration=c,
        **counts([r for r in rows if (r['matrix'],r['origin_mode'],r['configuration'])==(matrix,mode,c)]))
        for matrix,configs in configurations.items() for mode in MODES for c in sorted(configs)]
    family_rows=[]
    for mode in MODES:
        for name,pairs in families.items():
            matrix='terrain' if name.startswith('weighted-es-') else 'NEX326-methods'
            configs={c for pair in pairs.values() for c in pair}
            family_rows.append(dict(family_id=name,origin_mode=mode,
                candidate_contrasts=len(pairs),unique_configurations=len(configs),
                independent_blocks=MODES[mode],
                **counts([r for r in rows if r['matrix']==matrix and r['origin_mode']==mode and r['configuration'] in configs])))
    ancillary=[]
    for kind in sorted(FORECAST_KINDS-{'scientific_forecast'}):
        ids={w['work_id'] for w in settings['workloads'] if w['kind']==kind}
        ancillary.append(dict(kind=kind,expected=len(ids),success=len(ids&completed),
                              failed=len(ids&failed),runnable=len(ids-completed-failed)))
    return dict(schema_version='pirc17-public-safe-execution-coverage-v1',
        captured_utc=datetime.now(timezone.utc).isoformat(),
        settings_id=settings['settings_id'],predictor_index_sha256=digest(progress),
        score_index_sha256=digest(index),decision_policy_sha256=policy['sha256'],
        scientific=counts(rows),matrix_modes=groups,configurations=config_rows,
        comparison_families=family_rows,ancillary_forecasts=ancillary,
        family_rows_overlap_and_cannot_be_summed=True,
        unique_independent_primary_blocks=46,origin_mode_cases=58,
        secondary_modes_reuse_first_six_primary_blocks=True,
        score_payloads_read=0,model_or_forecast_arrays_read=0,
        new_forecasts=0,new_fits=0,new_particle_scores=0,new_inference_tests=0,
        independent_saved_output_audit_completed=False,scientific_claim_authorized=False,
        registered_sixteen_table_figure_inventory_replaced=False,
        interpretation='Captured index/source-binding census, not score-payload validation or qualification. Any required failed/invalid entry makes the whole registered family unavailable; missing entries remain pending. Never zero-fill or use a successful intersection. Incomplete families have no effect estimate here.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','cache','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('report must remain outside the live producer directory')
    settings,imported=load(args.root)
    index=read_json(args.cache/'progress.json')
    if index['cache_sha256']!=args.cache.name:
        raise ValueError('original score-cache identity required')
    progress=read_json(args.root/'progress.json')
    result=census(settings,imported,progress,index,read_json(POLICY))
    result.update(generator_sha256=file_hash(__file__),wall_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('captured_utc','scientific','matrix_modes','ancillary_forecasts','wall_seconds')}))


if __name__=='__main__':main()
