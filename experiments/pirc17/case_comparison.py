"""Private matched-case comparison of existing arrays and score-cache records.

Reuses the three source-bound, outcome-independent cases from case_review.
No prediction, refitting, scoring, bootstrap, geographic inference or audit.
Never substitutes missing/failed subjects or ranks. Only the original first
RNG seed is shown; example winners are not population-level conclusions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .case_review import check_output, map_layers, observed_route, saved_arrays
from .checkpoint_preview import read_cached_row
from .checkpoint_resume import load
from .features import LocalFrame
from .protocol_core import digest, file_hash, read_json, unpack

VERSION = 'pirc17-private-matched-cases-v1'
CASE_SHA = 'ae135a0e455894f3be11a2fcdb2bedb8f74f9f0203a94b6240f4e8e966a6c208'
STAGE_SHA = '8900f3dee68d5fc9cafd5ced2e2394503432d2493251601c0ba9cb771cbefba1'
SUBJECTS = ('arm-01/full','arm-04/gmm_kernel','arm-06/dt300')
FAMILIES = {'arm-04/gmm_kernel':'method-model-structure',
            'arm-06/dt300':'method-observation-interval'}


def case_score(cache, index, settings, imported, progress, evidence):
    work=next(w for w in settings['workloads'] if w['work_id']==evidence['work_id'])
    completion=imported.get(work['work_id'],{}).get('manifest') or progress['completed'].get(work['work_id'])
    binding=None if completion is None else dict(path=completion['artifact_path'],content_sha256=completion['artifact_sha256'])
    source=digest(dict(binding=binding,failure=progress['failures'].get(work['work_id'])))
    entry=index['rows'].get(work['work_id'])
    if entry is None:return None,dict(status='unavailable',reason='no existing score; no implicit scoring')
    row,value=read_cached_row(cache,work,entry,source)
    if (row['sample_id']!=evidence['sample_id'] or row['independent_block_id']!=evidence['independent_block_id']
            or row['saved_forecast_sha256']!=evidence['record_sha256']):
        raise ValueError('cached score is not this exact selected forecast')
    if row['status']!='success':return None,dict(status=row['status'],reason=row['reason'])
    return row,dict(status='success',cached_row_identity=digest(value),source_sha256=source,
        weighted_es_m=row['score_m'],fde_m=row['scores']['fde_m'],
        es_by_time_m=[t['energy_score_m'] for t in row['scores']['by_time']])


def validate_case(positions,seconds,target,row):
    truth=np.asarray(target['positions_m'])
    if not np.array_equal(seconds,target['elapsed_seconds']):
        raise ValueError('candidate clock differs from registered same-case truth')
    fde=float(np.linalg.norm(positions[:,-1].mean(axis=0)-truth[-1]))
    if not math.isclose(fde,row['scores']['fde_m'],rel_tol=1e-12,abs_tol=1e-8):
        raise ValueError('cached FDE does not describe saved candidate arrays')
    by_time=row['scores']['by_time']
    if not np.array_equal([t['elapsed_seconds'] for t in by_time],seconds):
        raise ValueError('cached score clock differs')
    final=by_time[-1]['region']
    if final['region']!='ensemble_mean_radial_quantile_disk':
        raise ValueError('unexpected saved region estimator')
    if not np.allclose(final['center_m'],positions[:,-1].mean(axis=0),rtol=0,atol=1e-8):
        raise ValueError('cached region center differs from saved particle mean')
    levels=[x for x in final['levels'] if x['level']==.9]
    if len(levels)!=1 or not np.isfinite(levels[0]['radius_m']) or levels[0]['radius_m']<=0:
        raise ValueError('missing/invalid original 90% region; no radius substitute')
    if type(levels[0]['covered']) is not bool or levels[0]['covered']!=(fde<=levels[0]['radius_m']):
        raise ValueError('saved disk coverage contradicts selected target/center/radius')
    return dict(center_m=final['center_m'],radius_m=levels[0]['radius_m'],
                covered=levels[0]['covered'],empirical_mass=levels[0]['empirical_mass'])


def comparison_bounds(route,models):
    parts=[np.asarray(route)]
    for model in models:
        if model.get('status')!='success':continue
        parts.append(model['positions'].reshape(-1,2))
        center=np.asarray(model['region']['center_m']);radius=model['region']['radius_m']
        parts.extend([(center-radius)[None,:],(center+radius)[None,:]])
    cloud=np.concatenate(parts)
    if not np.isfinite(cloud).all():raise ValueError('nonfinite display bounds')
    low=cloud.min(axis=0)-150;high=cloud.max(axis=0)+150
    return [float(low[0]),float(low[1]),float(high[0]),float(high[1])]


def render(path,geography,prefix,target,route,models,rasters,vectors,bounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm,ListedColormap
    from matplotlib.patches import Circle
    colors=['#006400','#ffbb22','#ffff4c','#f096ff','#fa0000','#b4b4b4','#f0f0f0','#0064c8','#0096a0','#00cf75','#fae6a0']
    cmap=ListedColormap(colors);norm=BoundaryNorm([5,15,25,35,45,55,65,75,85,92.5,97.5,105],len(colors))
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    truth=np.asarray(target['positions_m']);history=np.asarray(prefix['terrain_prefix_positions_m'])
    for ax,model in zip(axes,models):
        if 'worldcover' in rasters:
            data,extent=rasters['worldcover'];ax.imshow(data,extent=extent,origin='upper',cmap=cmap,norm=norm,alpha=.4)
        for key,color in [('roads','#555555'),('rivers','#087aca')]:
            ax.add_collection(LineCollection(vectors.get(key,[]),colors=color,linewidths=.6,alpha=.5))
        ax.plot(route[:,0],route[:,1],color='black',linewidth=2,label='Observed future route')
        ax.plot(history[:,0],history[:,1],color='#008f39',linewidth=3)
        ax.scatter(*history[-1],marker='*',s=140,color='#008f39',zorder=7)
        ax.scatter(*truth[-1],marker='X',s=100,color='black',zorder=7,label='Observed last target')
        if model['status']=='success':
            positions=model['positions'];means=positions.mean(axis=0)
            ax.scatter(positions[:,-1,0],positions[:,-1,1],s=4,color='#b34000',alpha=.28,label='512 particles at last target')
            ax.plot(np.r_[0,means[:,0]],np.r_[0,means[:,1]],'--',color='#b34000',linewidth=1,label='Four-mean guide, not continuous path')
            ax.scatter(means[:,0],means[:,1],marker='D',s=35,color='#b34000',zorder=8)
            region=model['region'];ax.add_patch(Circle(region['center_m'],region['radius_m'],fill=False,
                edgecolor='#0051ad',linewidth=1.8,label='Original cached 90% disk'))
            ax.set_title(f'{model["subject"]}\nES={model["score"]["weighted_es_m"]:.1f} m; FDE={model["score"]["fde_m"]:.1f} m\n'
                f'Last-target 90% disk: {"covered" if region["covered"] else "NOT covered"}',fontsize=11)
        else:
            ax.set_title(model['subject']+'\n'+model['status']);ax.text(.5,.5,'Unavailable; not replaced',transform=ax.transAxes,ha='center')
        ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('East from origin (m)');ax.set_ylabel('North from origin (m)')
        ax.legend(loc='lower left',fontsize=7)
    fig.suptitle(f'Private matched case | rank {geography["rank"]}: {geography["harvest_area"]}, {geography["country"]}\n'
        f'Same case and first RNG seed 20260814; N=512; max integration step=5 s; last actual target={target["elapsed_seconds"][-1]:.0f} s\n'
        'GMM: model-structure question; dt300: observation-interval question; shared registered Full control. Single examples do NOT establish population superiority.',fontsize=11)
    fig.savefig(path,dpi=140);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


def run(runtime,cache,cases,stage_path,output):
    started=time.perf_counter();output=check_output(output)
    runtime,cache,cases=Path(runtime).resolve(),Path(cache).resolve(),Path(cases).resolve()
    if cache.parent!=runtime/'offline-scores':raise ValueError('original runtime cache required')
    if file_hash(cases/'case-manifest.json')!=CASE_SHA:raise ValueError('original fixed review case manifest required')
    if file_hash(stage_path)!=STAGE_SHA:raise ValueError('original complete-primary stage statistics required')
    previous=read_json(cases/'case-manifest.json');geography=read_json(cases/'geography-summary.json')
    settings,imported=load(runtime);progress=read_json(runtime/'progress.json');index=read_json(cache/'progress.json')
    if index['cache_sha256']!=cache.name or progress['settings_id']!=settings['settings_id']:
        raise ValueError('changed runtime/cache identity')
    if geography['provenance']['context_sha256']!=settings['context_sha256'] or previous['selected_ranks']!=[0,3,4]:
        raise ValueError('original three-case selection changed')
    context=unpack(read_json(settings['context']),expected_sha256=settings['context_sha256'])
    stage=read_json(stage_path)
    for subject,family in FAMILIES.items():
        matches=[c for c in stage['comparisons'] if c['candidate']==subject]
        if len(matches)!=1 or matches[0]['control']!='arm-01/full' or matches[0]['family_id']!=family:
            raise ValueError('display candidates are not registered comparisons to this control')
    output.mkdir(parents=True)
    manifest=dict(schema_version=VERSION,private_local_review_only=True,public_route_release_authorized=False,
        original_case_manifest_sha256=CASE_SHA,context_sha256=settings['context_sha256'],
        cache_sha256=cache.name,stage_statistics_sha256=file_hash(stage_path),renderer_sha256=file_hash(__file__),
        cases=[],new_forecasts=0,new_fits=0,new_particle_scores=0,new_inference_tests=0,
        independent_saved_output_audit=False,case_effects_are_population_conclusions=False)
    prefixes={r['sample_id']:r for r in context['causal_prefixes']}
    targets={r['sample_id']:r for r in unpack(context['scoring_inputs'])['targets']}
    for old_case in previous['cases']:
        geo=old_case['geography'];prefix=prefixes[geo['sample_id']];target=targets[geo['sample_id']]
        route,route_evidence=observed_route(settings,prefix,target,geo)
        models=[]
        for subject in SUBJECTS:
            arrays,evidence=saved_arrays(settings,imported,progress,geo['rank'],subject=subject)
            model=dict(subject=subject,status=evidence['status'],forecast=evidence);models.append(model)
            if arrays is None:continue
            if evidence['sample_id']!=geo['sample_id'] or evidence['independent_block_id']!=prefix['independent_block_id']:
                raise ValueError('candidate belongs to another independent case')
            if subject=='arm-01/full' and evidence['record_sha256']!=old_case['record_sha256']:
                raise ValueError('original case control changed')
            row,score=case_score(cache,index,settings,imported,progress,evidence)
            model['score']=score
            if row is None:model['status']=score['status'];continue
            positions,seconds=arrays;region=validate_case(positions,seconds,target,row)
            model.update(positions=positions,seconds=seconds,region=region)
        bounds=comparison_bounds(route,models)
        rasters,vectors,maps=map_layers(settings,unpack(context['map_catalog']),LocalFrame(*prefix['scoring_frame']),bounds)
        name=f'comparison-rank-{geo["rank"]:02d}.png'
        render(output/name,geo,prefix,target,route,models,rasters,vectors,bounds)
        manifest['cases'].append(dict(rank=geo['rank'],geography=geo,route=route_evidence,maps=maps,
            figure=name,figure_sha256=file_hash(output/name),
            models=[{k:v for k,v in model.items() if k not in ('positions','seconds')} for model in models]))
        print(json.dumps(dict(event='private_matched_case_rendered',rank=geo['rank'],statuses=[m['status'] for m in models])),flush=True)
    manifest['wall_seconds']=time.perf_counter()-started
    (output/'comparison-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='matched_cases_saved',cases=3,wall_seconds=manifest['wall_seconds'],new_forecasts=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','cache','cases','stage','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();run(a.runtime,a.cache,a.cases,a.stage,a.output)


if __name__=='__main__':main()
