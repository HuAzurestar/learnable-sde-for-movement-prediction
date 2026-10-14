"""Frozen-origin environment descriptors and descriptive cached method means.

Reads exactly the 46 admitted origin rows, not full trajectories or map values.
Categories are original WorldCover codes; no error-selected thresholds or new
hypothesis tests. Missing/error scores fail the entire descriptive projection.
No fitted models, forecasts, particle scores, bootstrap or final audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time

import pyarrow.parquet as pq

from .case_review import check_output
from .checkpoint_preview import read_cached_row
from .checkpoint_resume import load
from .protocol_core import digest, file_hash, read_json, under, unpack

VERSION='pirc17-origin-environment-description-v1'
STAGE_SHA='8900f3dee68d5fc9cafd5ced2e2394503432d2493251601c0ba9cb771cbefba1'
SUBJECTS=('arm-01/full','arm-04/gmm_kernel','arm-06/dt300')
SEEDS=tuple(range(20260814,20260819))
CLASSES={10:'Trees',20:'Shrub',30:'Grass',40:'Cropland',50:'Built-up',60:'Bare',
         70:'Snow/ice',80:'Water',90:'Wetland',95:'Mangrove',100:'Moss/lichen'}
FACTORS={'worldcover_class':'worldcover_status','slope_radians':'dem_surface_status',
         'road_distance_m':'overture_road_status','river_distance_m':'hydrorivers_river_status'}
PARENTS=('worldcover_parent_asset_id','dem_surface_parent_asset_id',
         'overture_road_parent_asset_id','hydrorivers_river_parent_asset_id')


def describe_origins(rows):
    if len(rows)!=46 or len({r['independent_block_id'] for r in rows})!=46:
        raise ValueError('all 46 unique primary origins required; no complete-case subset')
    for r in rows:
        if any(r[s]!='valid' for s in FACTORS.values()):
            raise ValueError('missing origin descriptor; no imputation or subset')
        if any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) for k in FACTORS):
            raise ValueError('nonfinite origin descriptor')
        if r['worldcover_class'] not in CLASSES or r['road_distance_m']<0 or r['river_distance_m']<0:
            raise ValueError('unsupported cover/negative distance')
        if not 0<=r['slope_radians']<math.pi/2:
            raise ValueError('unsupported slope angle')
    counts=Counter(r['worldcover_class'] for r in rows)
    continuous={}
    for name,key,scale,units in [('slope_degrees','slope_radians',180/math.pi,'degrees'),
                               ('road_distance_m','road_distance_m',1,'m'),
                               ('river_distance_m','river_distance_m',1,'m')]:
        values=[r[key]*scale for r in rows]
        continuous[name]=dict(units=units,n=46,minimum=min(values),median=statistics.median(values),maximum=max(values))
    return dict(independent_blocks=46,worldcover_at_origin=[dict(code=k,label=v,blocks=counts[k]) for k,v in CLASSES.items()],
                continuous_at_origin=continuous,all_four_descriptors_valid=46,
                origin_class_is_whole_route_environment=False)


def read_origins(settings,context,protocol):
    binding=unpack(protocol)['dataset_inputs'];frozen=binding['snapshot']
    root=Path(settings['input_paths']['snapshot']);release=Path(settings['input_paths']['release'])
    manifest=read_json(root/'manifest.json',expected_file_sha256=frozen['manifest_sha256'])
    spec=read_json(root/'feature_spec.json',expected_file_sha256=frozen['feature_spec_file_sha256'])
    if (manifest['status']!='valid' or manifest['dataset_id']!=binding['dataset_id']
        or manifest['snapshot_id']!=frozen['snapshot_id'] or len(manifest['files'])!=frozen['files']
        or digest(spec)!=frozen['feature_spec_content_sha256']):
        raise ValueError('registered snapshot metadata differs')
    path=release/'samples.jsonl'
    if file_hash(path)!=binding['release_artifact_sha256']['samples.jsonl']:
        raise ValueError('registered release sample identity differs')
    selected=unpack(context['population'])['selection']['selected']
    samples={};wanted={r['sample_id'] for r in selected}
    for line in path.open(encoding='utf-8'):
        r=json.loads(line)
        if r['sample_id'] in wanted:
            if r['sample_id'] in samples:raise ValueError('duplicate selected release sample')
            samples[r['sample_id']]=r
    prefixes={r['sample_id']:r for r in context['causal_prefixes']}
    entries={}
    for e in manifest['files']:
        key=e['file_id'],e['split']
        if key in entries:raise ValueError('ambiguous registered feature file')
        entries[key]=e
    columns=['dataset_version','file_id','segment_id','split','independent_block_id','absolute_epoch_ns']+list(FACTORS)+list(FACTORS.values())+list(PARENTS)
    rows=[];sources={}
    for rank,admitted in enumerate(selected):
        sid=admitted['sample_id'];sample=samples[sid];prefix=prefixes[sid]
        entry=entries[(sample['file_id'],admitted['split'])]
        if (entry['condition_sha256']!=prefix['condition_sha256'] or
            sample['independent_block_id']!=admitted['independent_block_id']):
            raise ValueError('origin feature/condition binding differs')
        feature_path=under(root,entry['path'])
        if file_hash(feature_path)!=entry['sha256']:raise ValueError('selected feature file bytes differ')
        source=pq.ParquetFile(feature_path)
        if source.metadata.num_rows!=entry['row_count']:raise ValueError('selected feature row count differs')
        table=pq.read_table(feature_path,columns=columns,filters=[('absolute_epoch_ns','=',prefix['origin_epoch_ns'])])
        if table.num_rows!=1:raise ValueError('origin row missing/ambiguous; no nearest-time substitution')
        row=table.to_pylist()[0]
        if any(row[k]!=v for k,v in dict(dataset_version=binding['dataset_id'],file_id=sample['file_id'],
            segment_id=sample['segment_id'],split=admitted['split'],independent_block_id=admitted['independent_block_id']).items()):
            raise ValueError('source row differs from exact selected origin identity')
        metadata=table.schema.metadata or {}
        if metadata.get(b'pirc21.feature_spec_sha256') not in (None,frozen['feature_spec_content_sha256'].encode()):
            raise ValueError('origin row schema specification differs')
        if any(not isinstance(row[k],str) or not row[k] for k in PARENTS):raise ValueError('origin factor lacks source parent')
        row.update(rank=rank,sample_id=sid);rows.append(row)
        sources[sid]=dict(feature_path=entry['path'],feature_sha256=entry['sha256'],condition_sha256=entry['condition_sha256'])
    describe_origins(rows)
    return rows,sources


def descriptive_scores(rows,origins):
    expected={(rank,seed,subject) for rank in range(46) for seed in SEEDS for subject in SUBJECTS}
    by_rank={r['rank']:r for r in origins};seen=set()
    for row in rows:
        axes=row['origin_rank'],row['seed'],row['configuration']
        if axes not in expected or axes in seen:raise ValueError('duplicate/unregistered descriptive score')
        origin=by_rank[row['origin_rank']]
        if (row['status']!='success' or row['sample_id']!=origin['sample_id'] or
            row['independent_block_id']!=origin['independent_block_id'] or
            row['origin_mode']!='causal_prefix' or row['matrix']!='NEX326-methods' or
            any(not math.isfinite(v) for v in (row['score_m'],row['scores']['fde_m']))):
            raise ValueError('failed/wrong/missing descriptive score; no success-only subset')
        seen.add(axes)
    if seen!=expected:raise ValueError('complete original 690-row grid required')
    result=[]
    for code,label in CLASSES.items():
        ranks={r['rank'] for r in origins if r['worldcover_class']==code}
        subjects=[]
        for subject in SUBJECTS:
            # First average five simulation seeds within each original block.
            block_means=[]
            for rank in sorted(ranks):
                group=[r for r in rows if r['origin_rank']==rank and r['configuration']==subject]
                if len(group)!=5:raise ValueError('incomplete within-block seed grid')
                block_means.append((statistics.mean(r['score_m'] for r in group),statistics.mean(r['scores']['fde_m'] for r in group)))
            subjects.append(dict(configuration=subject,forecast_rows=5*len(ranks),
                weighted_es_m=statistics.mean(v[0] for v in block_means) if block_means else None,
                fde_m=statistics.mean(v[1] for v in block_means) if block_means else None))
        result.append(dict(code=code,label=label,independent_blocks=len(ranks),
            status='descriptive_complete' if ranks else 'not_represented',subjects=subjects))
    overall=[dict(configuration=subject,weighted_es_m=statistics.mean(r['score_m'] for r in rows if r['configuration']==subject),
                  fde_m=statistics.mean(r['scores']['fde_m'] for r in rows if r['configuration']==subject)) for subject in SUBJECTS]
    return dict(cached_forecast_rows=690,independent_blocks=46,forecast_rng_seeds=5,
        group_means=result,overall=overall,hypothesis_tests=0,confidence_intervals=None,
        categories_defined_after_outcome_inspection=True,scientific_verdict_authorized=False)


def run(runtime,cache,stage_path,output):
    started=time.perf_counter();runtime,cache=Path(runtime).resolve(),Path(cache).resolve();output=check_output(output)
    if cache.parent!=runtime/'offline-scores' or file_hash(stage_path)!=STAGE_SHA:
        raise ValueError('unchanged original cache and stage required')
    settings,imported=load(runtime);progress=read_json(runtime/'progress.json');index=read_json(cache/'progress.json')
    if progress['settings_id']!=settings['settings_id'] or index['cache_sha256']!=cache.name:raise ValueError('changed runtime/cache')
    context=unpack(read_json(settings['context']),expected_sha256=settings['context_sha256'])
    bundle=unpack(read_json(settings['bundle']),expected_sha256=settings['bundle_sha256'])
    origins,sources=read_origins(settings,context,bundle['protocol']);descriptors=describe_origins(origins)
    rows=[];score_sources={}
    works=[w for w in settings['workloads'] if w['kind']=='scientific_forecast' and w['matrix']=='NEX326-methods'
        and w['origin_mode']=='causal_prefix' and w['subject'] in SUBJECTS]
    for work in works:
        wid=work['work_id'];completion=imported.get(wid,{}).get('manifest') or progress['completed'].get(wid)
        binding=None if completion is None else dict(path=completion['artifact_path'],content_sha256=completion['artifact_sha256'])
        failure=progress['failures'].get(wid);entry=index['rows'].get(wid)
        if failure or binding is None or entry is None:raise ValueError('statistical failure: missing/error original forecast or score')
        source=digest(dict(binding=binding,failure=failure));row,value=read_cached_row(cache,work,entry,source)
        rows.append(row);score_sources[wid]=digest(value)
    scores=descriptive_scores(rows,origins)
    stage=read_json(stage_path)
    for overall in scores['overall']:
        original=next(c for c in stage['configs'] if c['configuration']==overall['configuration'])
        for name in ('weighted_es_m','fde_m'):
            if not math.isclose(overall[name],original[name],rel_tol=1e-12,abs_tol=1e-8):
                raise ValueError('descriptive total differs from original complete-primary stage')
    summary=dict(schema_version=VERSION,descriptor_scope='one frozen offline snapshot row per admitted forecast origin',
        environment_scope='original WorldCover origin pixel, not full route, predicted positions or city type',
        online_feature_value_equality_independently_verified=False,
        **descriptors,method_descriptive_scores=scores,
        provenance=dict(context_sha256=settings['context_sha256'],population_sha256=context['population']['sha256'],
            protocol_sha256=bundle['protocol']['sha256'],snapshot_manifest_sha256=file_hash(Path(settings['input_paths']['snapshot'])/'manifest.json'),
            origin_source_records_sha256=digest(sources),cached_score_records_sha256=digest(score_sources),
            stage_statistics_sha256=STAGE_SHA,renderer_sha256=file_hash(__file__)),
        private_coordinates_timestamps_or_identifiers_included=False,new_forecasts=0,new_fits=0,
        new_particle_scores=0,new_inference_tests=0,independent_saved_output_audit=False,
        interpretation='Post-outcome descriptive means only. Small/unequal origin categories cannot establish environmental effects, whole-route classes, causal mechanisms or global applicability.')
    private=dict(schema_version=VERSION,origins=origins,origin_sources=sources,score_sources=score_sources)
    output.mkdir(parents=True)
    for name,value in [('environment-summary.json',summary),('private-origin-bindings.json',private)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='descriptive_environment_saved',wall_seconds=time.perf_counter()-started,
        feature_files_read=46,cached_scores_read=690,worldcover_counts={r['label']:r['blocks'] for r in descriptors['worldcover_at_origin']},
        continuous_at_origin=descriptors['continuous_at_origin'],group_means=scores['group_means']),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('runtime','cache','stage','output'):parser.add_argument('--'+arg,type=Path,required=True)
    a=parser.parse_args();run(a.runtime,a.cache,a.stage,a.output)


if __name__=='__main__':main()
