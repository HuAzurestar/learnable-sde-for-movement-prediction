"""Private matched terrain maps from six existing forecasts and cached scores.

Keep the outcome-independent three cases and first seed already illustrated.
Never predict, fit, score particles, bootstrap, qualify, retry or publish routes.
Append real terrain-vs-base maps to the unchanged bilingual private review.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import fitz
import numpy as np

from .case_comparison import CASE_SHA, case_score, comparison_bounds, validate_case
from .case_review import check_output, map_layers, observed_route, saved_arrays
from .case_supplement import Writer, check_body
from .checkpoint_resume import load
from .features import LocalFrame
from .inference import PRIMARY_FAMILY
from .protocol_core import file_hash, read_json, unpack

RANKS = (0, 3, 4)
SUBJECTS = ('base', 'all-terrain')
REVIEW_SHA = {
    'en': '6f5be890568057d4375ae1efbd78b536e743a7e38f98125ab76a3ee912a74317',
    'zh': '0da64e30279aa005c4cd3dcf2e4e380ae381f47bea38be3e50b22fca93fc5115',
}


def validate_pair(models, original):
    """A matched example, not a population inference or qualification."""
    if tuple(m['subject'] for m in models) != SUBJECTS:
        raise ValueError('both original terrain subjects in fixed order required')
    if PRIMARY_FAMILY['all-vs-base'] != ('all-terrain', 'base'):
        raise ValueError('registered terrain control changed')
    for model in models:
        if model['status'] != 'success':
            continue  # Preserve missing/failed panel; never select a replacement.
        f = model['forecast']; s = model['score']
        if (f['matrix'] != 'terrain' or f['configuration'] != model['subject'] or
                f['fit_identity'] != 'terrain-fit:'+model['subject'] or
                f['rank'] != original['rank'] or f['seed'] != 20260814 or
                f['sample_id'] != original['sample_id'] or
                f['independent_block_id'] != original['independent_block_id'] or
                f['maximum_step_seconds'] != 5. or f['history_step_seconds'] != 5. or
                f['newly_generated_forecasts'] != 0 or s['status'] != 'success'):
            raise ValueError('different terrain model, case, fit or simulation seed')
        if (type(f['raw_map_query_rows']) is not int or
                (model['subject'] == 'base' and f['raw_map_query_rows'] != 0) or
                (model['subject'] == 'all-terrain' and f['raw_map_query_rows'] <= 0)):
            raise ValueError('actual terrain map demand contradicts subject')
        numbers = [f['prediction_seconds'], s['weighted_es_m'], s['fde_m'], *s['es_by_time_m']]
        if (len(s['es_by_time_m']) != 4 or
                any(not math.isfinite(n) or n < 0 for n in numbers) or
                not math.isclose(sum(s['es_by_time_m'])/4, s['weighted_es_m'], abs_tol=1e-8)):
            raise ValueError('original case metrics are invalid or inconsistent')
    if any(m['status'] != 'success' for m in models):
        return None
    if models[0]['forecast']['brownian_identity'] != models[1]['forecast']['brownian_identity']:
        raise ValueError('paired terrain examples must use the same original Brownian stream')
    return models[1]['score']['weighted_es_m']-models[0]['score']['weighted_es_m']


def render(path, geo, prefix, target, route, models, rasters, vectors, bounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Circle, Patch
    classes = [10,20,30,40,50,60,70,80,90,95,100]
    names = ['Trees','Shrub','Grass','Cropland','Built-up','Bare','Snow/ice',
             'Water','Wetland','Mangrove','Moss/lichen']
    colors = ['#006400','#ffbb22','#ffff4c','#f096ff','#fa0000','#b4b4b4',
              '#f0f0f0','#0064c8','#0096a0','#00cf75','#fae6a0']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([5,15,25,35,45,55,65,75,85,92.5,97.5,105],len(colors))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    history = np.asarray(prefix['terrain_prefix_positions_m'])
    truth = np.asarray(target['positions_m'])
    for row, kind in enumerate(('worldcover', 'dem')):
        for col, model in enumerate(models):
            ax = axes[row, col]
            if kind in rasters:
                data, extent = rasters[kind]
                kwargs = dict(cmap=cmap,norm=norm,alpha=.45) if kind=='worldcover' else dict(cmap='terrain',alpha=.75)
                artist = ax.imshow(data, extent=extent, origin='upper', **kwargs)
                if kind=='dem':fig.colorbar(artist,ax=ax,shrink=.7,label='Elevation (m, catalog DEM)')
                else:
                    present = set(np.unique(np.ma.asarray(data).compressed()))
                    cover_legend = ax.legend(handles=[Patch(facecolor=c,alpha=.45,label=n)
                        for v,c,n in zip(classes,colors,names) if v in present],
                        loc='upper right',fontsize=6,title='WorldCover classes',title_fontsize=6)
                    ax.add_artist(cover_legend)
            else:
                ax.text(.5,.98,kind+' unavailable; no substitute map',transform=ax.transAxes,ha='center',va='top')
            for key, color in [('roads','#555555'),('rivers','#087aca')]:
                if vectors.get(key):
                    ax.add_collection(LineCollection(vectors[key],colors=color,linewidths=.6,alpha=.5,
                        label='Roads (Overture)' if key=='roads' else 'Rivers (HydroRIVERS)'))
            ax.plot(route[:,0],route[:,1],color='black',linewidth=2,label='Observed future route')
            ax.plot(history[:,0],history[:,1],color='#008f39',linewidth=3,label='Visible prefix')
            ax.scatter(*history[-1],marker='*',s=140,color='#008f39',zorder=8)
            ax.scatter(truth[:,0],truth[:,1],marker='X',s=60,color='black',zorder=7,label='Four observed targets')
            if model['status']=='success':
                positions = model['positions']; means = positions.mean(axis=0); region = model['region']
                ax.scatter(positions[:,-1,0],positions[:,-1,1],s=4,color='#b34000',alpha=.3,label='512 final particles')
                ax.plot(np.r_[0,means[:,0]],np.r_[0,means[:,1]],'--',color='#b34000',label='Mean guide, NOT a continuous path')
                ax.scatter(means[:,0],means[:,1],marker='D',s=35,color='#b34000',zorder=8)
                ax.add_patch(Circle(region['center_m'],region['radius_m'],fill=False,edgecolor='#0051ad',linewidth=1.8,label='Original cached 90% disk'))
                s = model['score']; f = model['forecast']
                ax.set_title(f'{model["subject"]} | {kind}\nES={s["weighted_es_m"]:.1f} m; FDE={s["fde_m"]:.1f} m; '
                             f'90%: {"covered" if region["covered"] else "NOT covered"}\n'
                             f'Original core time={f["prediction_seconds"]:.2f} s; actual map rows={f["raw_map_query_rows"]}',fontsize=10)
            else:
                ax.set_title(model['subject']+' | '+kind+' | '+model['status'])
                ax.text(.5,.5,'Unavailable; not replaced',transform=ax.transAxes,ha='center')
            ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[1],bounds[3]); ax.set_aspect('equal',adjustable='box')
            ax.set_xlabel('East from origin (m)'); ax.set_ylabel('North from origin (m)')
            ax.legend(loc='lower left',fontsize=6)
    fig.suptitle(f'PRIVATE matched terrain case | {geo["harvest_area"]}, {geo["country"]} | rank {geo["rank"]}\n'
                 'Same case, first seed 20260814 and Brownian stream; N=512; max step=5 s; '
                 'target times (s): '+', '.join(f'{t:.0f}' for t in target['elapsed_seconds'])+'\n'
                 'base: history only; all-terrain: history + roads/rivers/WorldCover/surface. '
                 'Background shown for BOTH; three cases do NOT establish a population terrain effect.',fontsize=10)
    fig.savefig(path,dpi=140); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)


def case_note(case, language):
    zh = language=='zh'; delta=case['case_es_difference_m']
    if delta is None:
        return '必需结果缺失/失败，此例不计算效应，不替换案例。' if zh else 'Required output missing/failed: no case effect or replacement.'
    base,terrain=case['models']; fde_delta=terrain['score']['fde_m']-base['score']['fde_m']
    ratio=terrain['forecast']['prediction_seconds']/base['forecast']['prediction_seconds'] if base['forecast']['prediction_seconds']>0 else None
    time_text='N/A' if ratio is None else f'{ratio:.1f}'
    covered=terrain['region']['covered']
    if zh:
        return (f'此例全地形相对 base：ES {delta:+.2f}m，终点误差 {fde_delta:+.2f}m（负值为降低）；'
                f'原核心耗时比 {time_text} 倍。全地形真实终点{"在" if covered else "不在"}原 90% 圆内。'
                '这不是总体地形效应、显著性或运行时间基准结论。')
    return (f'Case all-terrain minus base: ES {delta:+.2f} m, FDE {fde_delta:+.2f} m (negative means lower); '
            f'original core-time ratio {time_text}. All-terrain final target is {"inside" if covered else "outside"} its original 90% disk. '
            'Not a population terrain effect, significance finding or runtime benchmark.')


def marginal_regions(positions, seconds, target, row):
    """Check all four original cached circles without estimating a new region."""
    validate_case(positions,seconds,target,row)
    regions=[]
    for k,t in enumerate(seconds):
        original=row['scores']['by_time'][k]['region']
        mean=positions[:,k].mean(axis=0)
        error=float(np.linalg.norm(mean-np.asarray(target['positions_m'])[k]))
        levels=[v for v in original['levels'] if v['level']==.9]
        if (original['region']!='ensemble_mean_radial_quantile_disk' or len(levels)!=1 or
                not np.allclose(original['center_m'],mean,rtol=0,atol=1e-8)):
            raise ValueError('original marginal circle differs from saved distribution')
        level=levels[0]
        if (not math.isfinite(level['radius_m']) or level['radius_m']<=0 or
                type(level['covered']) is not bool or level['covered']!=(error<=level['radius_m'])):
            raise ValueError('invalid original marginal radius or coverage')
        regions.append(dict(center_m=original['center_m'],radius_m=level['radius_m'],
                            covered=level['covered'],mean_error_m=error))
    return regions


def slot_bounds(models, slot, history, route):
    parts=[np.asarray(history),np.asarray(route)]
    for model in models:
        if model['status']!='success':continue
        parts.append(model['positions'][:,slot])
        r=model['regions'][slot]; center=np.asarray(r['center_m'])
        parts.extend([(center-r['radius_m'])[None,:],(center+r['radius_m'])[None,:]])
    cloud=np.concatenate(parts)
    if not np.isfinite(cloud).all():raise ValueError('nonfinite horizon display bounds')
    lo=cloud.min(axis=0)-50;hi=cloud.max(axis=0)+50
    return [float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])]


def render_horizons(path, geo, prefix, target, route, route_seconds, models, rasters, vectors, slots):
    """Two times by two terrain subjects; no joins between predicted means."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm,ListedColormap
    from matplotlib.patches import Circle,Patch
    classes=[10,20,30,40,50,60,70,80,90,95,100]
    colors=['#006400','#ffbb22','#ffff4c','#f096ff','#fa0000','#b4b4b4',
            '#f0f0f0','#0064c8','#0096a0','#00cf75','#fae6a0']
    names=['Trees','Shrub','Grass','Cropland','Built-up','Bare','Snow/ice','Water','Wetland','Mangrove','Moss/lichen']
    cmap=ListedColormap(colors);norm=BoundaryNorm([5,15,25,35,45,55,65,75,85,92.5,97.5,105],len(colors))
    fig,axes=plt.subplots(2,2,figsize=(10,10),layout='constrained')
    history=np.asarray(prefix['terrain_prefix_positions_m']);truth=np.asarray(target['positions_m'])
    for row,k in enumerate(slots):
        observed=route[np.asarray(route_seconds)<=target['elapsed_seconds'][k]]
        bounds=slot_bounds(models,k,history,observed)
        for col,model in enumerate(models):
            ax=axes[row,col]
            if 'worldcover' in rasters:
                data,extent=rasters['worldcover'];ax.imshow(data,extent=extent,origin='upper',cmap=cmap,norm=norm,alpha=.45)
                present=set(np.unique(np.ma.asarray(data).compressed()))
                legend=ax.legend(handles=[Patch(facecolor=c,alpha=.45,label=n) for v,c,n in zip(classes,colors,names) if v in present],
                    loc='upper right',fontsize=6,title='WorldCover',title_fontsize=6)
                ax.add_artist(legend)
            else:ax.text(.5,.98,'WorldCover unavailable',transform=ax.transAxes,ha='center',va='top')
            for key,color in [('roads','#555555'),('rivers','#087aca')]:
                if vectors.get(key):ax.add_collection(LineCollection(vectors[key],colors=color,linewidths=.6,alpha=.5))
            ax.plot(observed[:,0],observed[:,1],color='black',linewidth=2,label='Observed route up to this time')
            ax.plot(history[:,0],history[:,1],color='#008f39',linewidth=2,label='Visible history')
            ax.scatter(*history[-1],marker='*',s=90,color='#008f39',zorder=8)
            ax.scatter(*truth[k],marker='X',s=80,color='black',zorder=9,label='Actual target')
            if model['status']=='success':
                positions=model['positions'][:,k];r=model['regions'][k]
                ax.scatter(positions[:,0],positions[:,1],s=5,color='#b34000',alpha=.3,label='512 predicted positions')
                ax.scatter(*r['center_m'],marker='D',s=45,color='#b34000',zorder=8,label='Predicted mean')
                ax.add_patch(Circle(r['center_m'],r['radius_m'],fill=False,edgecolor='#0051ad',linewidth=1.7,label='Original 90% prediction disk'))
                ax.set_title(f'{model["subject"]} | nominal {int((60,300,900,1800)[k]/60)} min; actual {target["elapsed_seconds"][k]:.0f} s\n'
                    f'Mean error={r["mean_error_m"]:.1f} m; R90={r["radius_m"]:.1f} m\n'
                    f'Target {"inside" if r["covered"] else "OUTSIDE"} original 90% disk',fontsize=9)
            else:
                ax.set_title(f'{model["subject"]} | {target["elapsed_seconds"][k]:.0f} s | {model["status"]}',fontsize=9)
                ax.text(.5,.5,'Unavailable; not replaced',transform=ax.transAxes,ha='center')
            ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal',adjustable='box')
            ax.set_xlabel('East from origin (m)',fontsize=8);ax.set_ylabel('North from origin (m)',fontsize=8)
            ax.tick_params(labelsize=7);ax.legend(loc='lower left',fontsize=5.7)
    fig.suptitle(f'{geo["harvest_area"]}, {geo["country"]} | fixed first seed 20260814 | saved position distributions\n'
        'Left: terrain base (history only); right: all-terrain. Same axes within each time; different time panels may zoom.\n'
        'Grey roads: Overture; blue rivers: HydroRIVERS. No predicted trajectory or interpolation is plotted.',fontsize=9)
    fig.savefig(path,dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


def manuscript_projection(manifest):
    """Authorized case diagnostics and image bindings, no coordinate arrays/IDs."""
    if (tuple(c['rank'] for c in manifest['cases'])!=RANKS or
            any(tuple(m['subject'] for m in c['models'])!=SUBJECTS for c in manifest['cases'])):
        raise ValueError('all three fixed cases and both subjects required for manuscript')
    output=dict(schema_version='pirc17-local-manuscript-horizon-cases-v1',
        local_manuscript_inclusion_requested=True,public_distribution_authorized=False,final_human_accepted=False,
        saved_case_source_sha256=CASE_SHA,renderer_sha256=manifest['renderer_sha256'],
        independent_saved_output_audit=False,population_effects_established=False,
        original_science_denominator=11020,primary_independent_blocks=46,selected_cases=3,
        selected_scientific_slots=6,
        reused_scientific_forecasts=sum(m['status']=='success' for c in manifest['cases'] for m in c['models']),
        seed=20260814,particles=512,maximum_step_seconds=5,
        nominal_scoring_seconds=[60,300,900,1800],new_forecasts=0,new_fits=0,new_particle_scores=0,new_inference_tests=0,
        selection='Original outcome-independent first block of each of first three encountered source countries; no replacement',cases=[])
    for case in manifest['cases']:
        models=[]
        for m in case['models']:
            value=dict(subject=m['subject'],status=m['status'])
            if m['status']=='success':
                value.update(weighted_es_m=m['score']['weighted_es_m'],fde_m=m['score']['fde_m'],
                    original_core_seconds=m['forecast']['prediction_seconds'],raw_map_query_rows=m['forecast']['raw_map_query_rows'],
                    marginal_es_m=m['score']['es_by_time_m'],
                    horizons=[{k:v for k,v in r.items() if k!='center_m'} for r in m['regions']])
            models.append(value)
        output['cases'].append(dict(label=case['geography']['harvest_area'],country=case['geography']['country'],
            actual_scoring_seconds=case['target_elapsed_seconds'],models=models,figures=case['horizon_figures'],
            interpretation=dict(en=case_note(case,'en'),zh=case_note(case,'zh'))))
    return output


def append_reviews(output, review, font, manifest):
    for language in ('en','zh'):
        zh = language=='zh'; appendix = fitz.open(); writer = Writer(appendix,language,font)
        p = writer.page('PIRC-17 全地形与无地形：实际匹配案例' if zh else 'PIRC-17 matched terrain-vs-base real cases',landscape=True)
        intro = ('保持原固定三个案例和首 seed，不依据结果选例。每例对比地形矩阵自己的 base 与 all-terrain，'
                 '不是将方法 Full 当成地形对照。base 保留历史；all-terrain 加入道路、河流、WorldCover 和坡面特征。'
                 '两者独立拟合，共享原布朗流、512 粒子、最大积分步长 5s 和四个原评分目标。' if zh else
                 'Keep the original three outcome-independent cases and first seed. Compare the terrain matrix\'s own base and all-terrain, not method Full. '
                 'base retains history; all-terrain adds roads, rivers, WorldCover and surface features. Separately fitted models share the original Brownian stream, '
                 '512 particles, maximum 5 s integration step and four original scoring targets.')
        writer.text(p,intro,85,size=12,height=130)
        rows = []
        for case in manifest['cases']:
            for model in case['models']:
                if model['status']!='success':
                    rows.append([case['geography']['harvest_area']+' / '+model['subject'],model['status'],'N/A','N/A','N/A','N/A']); continue
                s=model['score']; r=model['region']; f=model['forecast']
                rows.append([case['geography']['harvest_area']+' / '+model['subject'],f'{s["weighted_es_m"]:.2f}',
                    f'{s["fde_m"]:.2f}',f'{r["radius_m"]:.2f}',str(r['covered']),f'{f["prediction_seconds"]:.2f}'])
        writer.table(p,['案例 / 配置','ES (m)','FDE (m)','R90 (m)','覆盖 / covered','core (s)'] if zh else
                     ['Case / configuration','ES (m)','FDE (m)','R90 (m)','Covered','Core (s)'],rows,245,[240,150,150,150,150,150])
        notes = ('ES 衡量概率分布预测，FDE 为最后时刻粒子均值距离，均越小越好。R90 是原 90% 预测圆半径，不是越大越好；'
                 'coverage 只表示该例真实终点是否在圆内，不是总体覆盖率。单例 ES 差定义为 all-terrain 减 base，负值才表示此例 ES 较低。'
                 '\ncore(s) 来自原预测记录，仅为当次核心预测计时，含预测期间原地图查询；不是新测的冷启动/热启动基准，'
                 '不含训练、输入准备、评分、输出验证或这次渲染。正式运行时间实验仍待原队列执行。'
                 '\n地图背景对两者都显示，并不意味着 base 使用地图。只保存四时刻位置分布；虚线不是连续预测 GPX。'
                 '三例只有一个模拟 seed，无置信区间或总体优越结论。原地形族仍存在缺项/失败，不能用这六份成功记录替代完整族分析。' if zh else
                 'ES scores the predictive distribution; FDE is final particle-mean distance. Lower is better for both. R90 is the original 90% disk radius: larger is not automatically better. '
                 'Covered is a case indicator, not population coverage. Case ES difference is all-terrain minus base; negative means lower ES here.\n'
                 'Core seconds are original measured prediction times including prediction-time map queries, not newly measured cold/warm benchmarks. They exclude training, input preparation, '
                 'scoring, output validation and this rendering. Original formal runtime trials remain pending.\n'
                 'Both panels show map context; base does not query maps. Four saved distributions are not continuous predicted GPX. Three first-seed examples have no confidence intervals or '
                 'population superiority conclusion. Missing/failed required terrain rows still invalidate their whole registered family; these successful examples cannot replace that analysis.')
        writer.text(p,notes,480,size=12,height=290)
        for case in manifest['cases']:
            geo=case['geography']; p=writer.page(f'{geo["harvest_area"]}, {geo["country"]}: base / all-terrain',landscape=True)
            with fitz.open(output/Path(case['figure']).with_suffix('.pdf')) as figure:
                p.show_pdf_page(fitz.Rect(38,80,p.rect.width-38,685),figure,0,keep_proportion=True)
            delta=case['case_es_difference_m']; delta_text='N/A' if delta is None else f'{delta:+.2f} m'
            writer.text(p,('该例 ES 差（all-terrain - base）：' if zh else 'Case ES difference (all-terrain - base): ')+delta_text+
                ('。黑线为保留的真实观测；橙点为预测终点分布；蓝圆为原 90% 预测区域。来源区域标签未核实为城市/街道，保留时钟未独立核实为传感器时间。' if zh else
                 '. Black: retained actual observations; orange: forecast final particles; blue: original 90% disk. Harvest label is not a verified city/street; retained clock is not independently sensor-verified.'),693,size=10,height=35)
            writer.text(p,case_note(case,language),735,size=10,height=55)
        appendix_path=output/f'terrain-appendix-{language}.pdf'; appendix.save(appendix_path,garbage=3,deflate=True)
        with fitz.open(review/f'private-review-{language}.pdf') as old:
            combined=fitz.open(); combined.insert_pdf(old); combined.insert_pdf(appendix)
            combined.set_toc(old.get_toc()+[[1,'PRIVATE matched terrain examples',len(old)+1]])
            combined.set_metadata(dict(title='PIRC-17 private review with matched terrain examples',author='',subject='Not accepted; private routes are not publicly releasable'))
            path=output/f'private-review-{language}.pdf'; combined.save(path,garbage=3,deflate=True); combined.close()
            with fitz.open(path) as saved:
                check_body(old,saved)
                if len(saved)!=len(old)+4:raise ValueError('all four terrain appendix pages required')
                manifest['documents'].append(dict(language=language,original_pages=len(old),appendix_pages=4,pages=len(saved),
                    pdf=path.name,pdf_sha256=file_hash(path),appendix=appendix_path.name,appendix_sha256=file_hash(appendix_path)))
        appendix.close()


def run(runtime, cache, cases, review, font, output, horizons=False):
    started=time.perf_counter(); output=check_output(output)
    runtime,cache,cases,review = [Path(p).resolve() for p in (runtime,cache,cases,review)]
    if cache.parent!=runtime/'offline-scores':raise ValueError('original runtime score cache required')
    if file_hash(cases/'case-manifest.json')!=CASE_SHA:raise ValueError('fixed original case manifest required')
    for lang,sha in REVIEW_SHA.items():
        if file_hash(review/f'private-review-{lang}.pdf')!=sha:raise ValueError('original bilingual private review required')
    font=Path(font)
    if not font.is_file():raise ValueError('existing CJK font required; no download')
    previous=read_json(cases/'case-manifest.json'); geography=read_json(cases/'geography-summary.json')
    settings,imported=load(runtime); progress=read_json(runtime/'progress.json'); index=read_json(cache/'progress.json')
    if (tuple(previous['selected_ranks'])!=RANKS or tuple(c['rank'] for c in previous['cases'])!=RANKS or
            progress['settings_id']!=settings['settings_id'] or index['cache_sha256']!=cache.name or
            geography['provenance']['context_sha256']!=settings['context_sha256']):
        raise ValueError('original cases, runtime or cache identity changed')
    context=unpack(read_json(settings['context']),expected_sha256=settings['context_sha256'])
    prefixes={p['sample_id']:p for p in context['causal_prefixes']}
    targets={t['sample_id']:t for t in unpack(context['scoring_inputs'])['targets']}
    output.mkdir(parents=True)
    manifest=dict(schema_version='pirc17-private-terrain-cases-v1',private_local_review_only=True,
        public_route_release_authorized=False,final_human_accepted=False,independent_saved_output_audit=False,
        case_effects_are_population_conclusions=False,new_forecasts=0,new_fits=0,new_particle_scores=0,new_inference_tests=0,
        original_case_manifest_sha256=CASE_SHA,original_review_sha256=REVIEW_SHA,
        context_sha256=settings['context_sha256'],matrix_sha256=settings['matrix_sha256'],cache_sha256=cache.name,
        renderer_sha256=file_hash(__file__),selected_ranks=list(RANKS),cases=[],documents=[])
    for original in previous['cases']:
        geo=original['geography']; prefix=prefixes[geo['sample_id']]; target=targets[geo['sample_id']]
        route,route_evidence=observed_route(settings,prefix,target,geo); models=[]
        for subject in SUBJECTS:
            arrays,evidence=saved_arrays(settings,imported,progress,geo['rank'],subject=subject,matrix='terrain')
            model=dict(subject=subject,status=evidence['status'],forecast=evidence); models.append(model)
            if arrays is None:continue
            row,score=case_score(cache,index,settings,imported,progress,evidence); model['score']=score
            if row is None:model['status']=score['status'];continue
            positions,seconds=arrays
            model.update(positions=positions,region=validate_case(positions,seconds,target,row),
                         regions=marginal_regions(positions,seconds,target,row))
        delta=validate_pair(models,original); bounds=comparison_bounds(route,models)
        rasters,vectors,maps=map_layers(settings,unpack(context['map_catalog']),LocalFrame(*prefix['scoring_frame']),bounds)
        name=f'terrain-rank-{geo["rank"]:02d}.png'
        render(output/name,geo,prefix,target,route,models,rasters,vectors,bounds)
        horizon_figures=[]
        if horizons:
            for slots,suffix in [((0,1),'01-05min'),((2,3),'15-30min')]:
                image=f'case-{geo["rank"]:02d}-{suffix}.png'
                render_horizons(output/image,geo,prefix,target,route,route_evidence['elapsed_seconds'],models,rasters,vectors,slots)
                horizon_figures.append(dict(png=image,png_sha256=file_hash(output/image),
                    pdf=Path(image).with_suffix('.pdf').name,pdf_sha256=file_hash(output/Path(image).with_suffix('.pdf'))))
        manifest['cases'].append(dict(rank=geo['rank'],geography=geo,route=route_evidence,maps=maps,
            figure=name,figure_sha256=file_hash(output/name),figure_pdf_sha256=file_hash(output/Path(name).with_suffix('.pdf')),
            target_elapsed_seconds=target['elapsed_seconds'],case_es_difference_m=delta,horizon_figures=horizon_figures,
            models=[{k:v for k,v in m.items() if k!='positions'} for m in models]))
        print(json.dumps(dict(event='private_terrain_case_rendered',rank=geo['rank'],statuses=[m['status'] for m in models])),flush=True)
    append_reviews(output,review,font,manifest)
    manifest['wall_seconds']=time.perf_counter()-started
    (output/'terrain-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if horizons:
        (output/'case-horizons.json').write_text(json.dumps(manuscript_projection(manifest),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='private_terrain_review_saved',documents=manifest['documents'],wall_seconds=manifest['wall_seconds'],new_forecasts=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','cache','cases','review','font','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--horizons',action='store_true',help='Render the explicitly requested four-time case manuscript illustrations')
    run(**vars(parser.parse_args()))


if __name__=='__main__':main()
