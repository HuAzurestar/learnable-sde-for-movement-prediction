"""Assemble private bilingual paper review copies from existing case figures.

No runtime, model, raw route, map query, array score or statistical estimator
is constructed. Original public manuscript pages are copied without editing.
Private case figures/notes are appended, never written into the public repo.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import fitz

from .case_review import check_output
from .origin_environment import describe_origins
from .protocol_core import file_hash, read_json, under

RANKS = (0, 3, 4)
SUBJECTS = ('arm-01/full', 'arm-04/gmm_kernel', 'arm-06/dt300')
INPUT_SHA = {
    'case-manifest.json': 'ae135a0e455894f3be11a2fcdb2bedb8f74f9f0203a94b6240f4e8e966a6c208',
    'comparison-manifest.json': 'e42fa0d6e191b98bdac4dfd02016182c09d8b05833e4f5e2f88f44fbc79bc6cc',
    'private-origin-bindings.json': '64ec590c296689ee17f4a8ae4f6e5bec5253c509d8f3fb09e6d3eafc4f836277',
    'en/main.pdf': 'db7254cf2ac67ecab3583783c98aad42e6196d0933970ab898e7c9cb8f17be0a',
    'zh/main.pdf': '2feba237d2b3e6aaf3dc2e8c9679e84d2ffa7ff513f252f9e454f92ea4bd9a6a',
    'map-source-acknowledgements-v1.json': 'e4590547ada23209f2efd27d9cc127e9efacfc22910bf6346810049719502b1c',
}
FIGURE_SHA = {
    'case-rank-00.pdf': '4f51ff203672e697d256d4be886549cb2386f59f54e1acb15cd7b386e87723a2',
    'case-rank-03.pdf': '872105b881c2bd6137c31b68b869b8df25f859a79552a0e2288a7bab98f53bd3',
    'case-rank-04.pdf': 'a3a6307e2260b732dfcb76c3c757803e92233119b46c0b7c22a332cf6882a43f',
    'comparison-rank-00.pdf': '89e43cc4154950f3065c13cac69d3b139e70038836636541d66b3d73de7798c0',
    'comparison-rank-03.pdf': 'b6ba68c0c96a4c2f9c991000bda29445c2e1fb9439ff77d759a8b778703a000e',
    'comparison-rank-04.pdf': '6bc5c6f1c39310d6a766be1b214a5fdc29cac8d090727227f0e65559cfab6197',
}
INTERPRETATION = {
    'en': {
        0: 'GMM has lower ES and endpoint error than Full in this example. dt300 has slightly lower ES but a higher endpoint error. All three final disks cover the target; coverage alone does not show equal precision.',
        3: 'All three models miss the final target with their original 90% disks. dt300 has lower ES and endpoint error here, but it does not repair coverage. This is an explicit failure example, not an omitted outlier.',
        4: 'GMM has lower ES and endpoint error here; dt300 has higher values for both. All disks cover the target, but dt300 uses a larger disk. A larger covered region is not automatically a better forecast.',
    },
    'zh': {
        0: '此例 GMM 的 ES 和终点误差都低于 Full。dt300 的 ES 略低，但终点误差较高。三个终点预测圆均覆盖目标；仅有覆盖不能说明它们同样精确。',
        3: '三种模型的真实终点均落在各自原始 90% 预测圆之外。dt300 的 ES 和终点误差在此例较低，但没有修复覆盖。这里明确展示失败案例，而不是把它当异常值删掉。',
        4: '此例 GMM 的 ES 和终点误差较低，dt300 两者都较高。三个预测圆均覆盖目标，但 dt300 的圆更大。更大的覆盖区域不自动代表更好的预测。',
    },
}
COVER_NAMES = {10: ('Trees', '树木'), 30: ('Grass', '草地'),
               40: ('Cropland', '耕地'), 50: ('Built-up', '建筑用地')}


def validate_sources(cases, comparisons, environment):
    """Require the whole fixed descriptive set; never select on outcome."""
    for manifest in (cases, comparisons):
        if (manifest.get('private_local_review_only') is not True or
                manifest.get('public_route_release_authorized') is not False or
                manifest.get('independent_saved_output_audit') is not False):
            raise ValueError('original private, non-audit evidence required')
        if tuple(c['rank'] for c in manifest['cases']) != RANKS:
            raise ValueError('fixed three cases required; no replacement or subset')
    describe_origins(environment['origins'])
    origins = {r['rank']: r for r in environment['origins']}
    selected = []
    for original, compared in zip(cases['cases'], comparisons['cases']):
        rank = original['rank']; origin = origins[rank]
        if (original['status'] != 'success' or original['seed'] != 20260814 or
                original['newly_generated_forecasts'] != 0 or
                original['geography'] != compared['geography'] or
                origin['sample_id'] != original['sample_id'] or
                origin['independent_block_id'] != original['independent_block_id']):
            raise ValueError('selected origin/figure source differs or failed')
        if tuple(m['subject'] for m in compared['models']) != SUBJECTS:
            raise ValueError('all three original comparison subjects required')
        seconds = original['target_elapsed_seconds']
        if (len(seconds) != 4 or any(not math.isfinite(t) for t in seconds) or
                any(a>=b for a,b in zip(seconds,seconds[1:]))):
            raise ValueError('four original finite target times required')
        for i, model in enumerate(compared['models']):
            forecast = model['forecast']; score = model['score']; region = model['region']
            if (model['status'] != 'success' or score['status'] != 'success' or
                    forecast['status'] != 'success' or forecast['rank'] != rank or
                    forecast['seed'] != 20260814 or forecast['subject'] != SUBJECTS[i] or
                    forecast['newly_generated_forecasts'] != 0 or
                    forecast['sample_id'] != original['sample_id'] or
                    forecast['independent_block_id'] != original['independent_block_id']):
                raise ValueError('failed, replaced or differently seeded case score')
            numbers = [score['weighted_es_m'], score['fde_m'], region['radius_m'],
                       forecast['prediction_seconds'], *score['es_by_time_m']]
            if len(score['es_by_time_m']) != 4 or any(not math.isfinite(v) or v < 0 for v in numbers):
                raise ValueError('finite nonnegative case metrics required')
            if not math.isclose(sum(score['es_by_time_m']) / 4, score['weighted_es_m'], abs_tol=1e-8):
                raise ValueError('original four-time ES mean differs')
            if type(region['covered']) is not bool or region['covered'] != (score['fde_m'] <= region['radius_m']):
                raise ValueError('original disk coverage contradicts metric')
            if i == 0 and (forecast['record_sha256'] != original['record_sha256'] or
                    not math.isclose(score['fde_m'], original['point_mean_errors_m'][-1], abs_tol=1e-8)):
                raise ValueError('original Full case and comparison differ')
        selected.append(dict(original=original, compared=compared, origin=origin))
    return selected


class Writer:
    def __init__(self, document, language, font):
        self.document = document; self.language = language; self.font = str(font)

    def page(self, title, landscape=False):
        # Large landscape pages preserve original vector-map labels at usable scale.
        width, height = (1190.55, 841.89) if landscape else (595.28, 841.89)
        page = self.document.new_page(width=width, height=height)
        if self.language=='zh':page.insert_font(fontname='casecjk', fontfile=self.font)
        self.text(page, title, 38, size=17)
        footer = ('PRIVATE LOCAL REVIEW - not accepted or publicly releasable' if self.language == 'en'
                  else '仅供本地私有审阅：未经最终验收，不可作为公开发布版本')
        self.text(page, footer, height-30, size=9, color=(.5,.12,.12))
        return page

    def text(self, page, text, top, size=11, color=(0,0,0), left=38, right=None, height=180):
        right = page.rect.width-38 if right is None else right
        spare = page.insert_textbox(fitz.Rect(left,top,right,min(top+height,page.rect.height-8)),
                                   text, fontname='casecjk' if self.language=='zh' else 'helv',
                                   fontsize=size, lineheight=1.4, color=color)
        if spare < 0:
            raise ValueError('case review text overflow; do not clip prose')

    def table(self, page, headers, rows, top, widths):
        if sum(widths)>page.rect.width-76:raise ValueError('table wider than review page')
        for n, row in enumerate([headers, *rows]):
            left=38
            if len(row)!=len(widths):raise ValueError('table column count differs')
            for cell,width in zip(row,widths):
                self.text(page,str(cell),top+n*28,size=9.5,left=left+4,right=left+width-4,height=26)
                left+=width
            page.draw_line((38,top+n*28+24),(left,top+n*28+24),color=(.65,.65,.65),width=.4)


def cover(writer):
    zh=writer.language=='zh'
    p=writer.page('PIRC-17 真实案例审阅附录' if zh else 'PIRC-17 real-case review appendix')
    sections = [
        (90, '用途与边界' if zh else 'Purpose and boundary',
         '把原工作稿与六张既有真实地图案例放在同一私有 PDF 中，方便从解释、数值到图像顺序阅读。公开工作稿原页保持不变；这里不是最终论文验收或公开路线许可。' if zh else
         'This private copy joins the unchanged reader manuscript to six existing real-map figures. Notes connect technology, numbers and pictures. It is neither final paper acceptance nor permission to publish routes.'),
        (230, '如何选例' if zh else 'How cases were selected',
         '按冻结主人口顺序，取首次遇到的三个来源国家各自第一块，固定 rank 0、3、4，使用原第一随机种子 20260814。未按误差、完成情况或图像好看程度挑选。这三例不是国家代表样本。' if zh else
         'In sealed primary-population order, take the first block of each of the first three encountered source countries: ranks 0, 3 and 4, first RNG seed 20260814. No filtering by error, completion or attractive fit. These are not representative national samples.'),
        (390, '预测与地图' if zh else 'Forecast and map meaning',
         '每例展示原 N=512、最大积分步长 5s 的四时刻预测，名义 1/5/15/30 分钟，实际时刻列于各例。Full 的 WorldCover 路线放大与 SRTM 高程面板，以及 Full/GMM/dt300 的同尺度比较，均为既有图。灰线道路、蓝线河流来自原 Overture/HydroRIVERS 来源；地图背景不是因果地形效应证据。' if zh else
         'Original N=512, maximum integration step 5 s, four nominal horizons 1/5/15/30 min; actual clocks are listed per case. Existing Full panels show a WorldCover route zoom and SRTM elevation. Matched Full/GMM/dt300 panels share axes. Grey roads and blue rivers use original Overture/HydroRIVERS sources. Map context is not causal terrain-effect evidence.'),
        (590, '不能从三例推出什么' if zh else 'What these cases do not establish',
         '每例仅一个模拟种子，不是五次训练或五个受试者。总体方法比较仍使用完整 46 块；地形效应依照单独矩阵。来源标签不是已核实街道；保留 condition 时钟也尚未独立核实为传感器时间。未新增预测、训练、粒子评分、检验或正式输出审计。' if zh else
         'One simulation seed per case is not five fits or five participants. Formal method comparisons remain on all 46 blocks; terrain effects require the separate matrix. Harvest labels are not verified streets, and retained condition clocks are not independently verified sensor timestamps. No new predictions, fits, particle scores, tests or formal output audit.'),
    ]
    for top,heading,body in sections:
        writer.text(p,heading,top,size=13,height=28)
        writer.text(p,body,top+32,size=11,height=140)


def case_notes(writer, selected):
    zh=writer.language=='zh';original=selected['original'];compared=selected['compared'];origin=selected['origin']
    geo=original['geography'];rank=original['rank']
    p=writer.page(f'{"案例" if zh else "Case"} {rank}: {geo["harvest_area"]}, {geo["country"]}')
    label=COVER_NAMES[origin['worldcover_class']][1 if zh else 0]
    text=(f'来源采集标签：{geo["country"]} / {geo["region"]} / {geo["harvest_area"]}，未核实逐点城市/街道归属。起点像素：{label}；坡度 {math.degrees(origin["slope_radians"]):.2f} 度；道路距离 {origin["road_distance_m"]:.2f}m；河流距离 {origin["river_distance_m"]:.2f}m。仅为起点离线描述，不是整条路线分类。' if zh else
          f'Source harvest label: {geo["country"]} / {geo["region"]} / {geo["harvest_area"]}, not a verified street. Origin pixel: {label}; slope {math.degrees(origin["slope_radians"]):.2f} degrees; road distance {origin["road_distance_m"]:.2f} m; river distance {origin["river_distance_m"]:.2f} m. Origin-only offline descriptors, not whole-route classification.')
    writer.text(p,text,80,size=11,height=128)
    seconds=original['target_elapsed_seconds']
    writer.text(p,('原目标时刻(s)：' if zh else 'Original target times (s): ')+', '.join(f'{x:.0f}' for x in seconds)+
                (f'；真实未来路线含 {original["route"]["observed_future_points"]} 个来源观测点。' if zh else
                 f'; source future route contains {original["route"]["observed_future_points"]} observations.'),195,size=10,height=36)
    rows=[]
    for model in compared['models']:
        s=model['score'];r=model['region']
        rows.append([model['subject'].split('/')[-1],f'{s["weighted_es_m"]:.2f}',f'{s["fde_m"]:.2f}',
                     f'{r["radius_m"]:.2f}',('是' if zh else 'yes') if r['covered'] else ('否' if zh else 'NO'),
                     f'{model["forecast"]["prediction_seconds"]:.3f}'])
    headers=['模型' if zh else 'Model','ES(m)','FDE(m)','R90(m)','覆盖' if zh else 'Covered','核心时长(s)' if zh else 'Core time(s)']
    writer.table(p,headers,rows,240,[100,70,75,85,60,125])
    writer.text(p,'原缓存的各时刻 ES(m)，不是各时刻位置误差：' if zh else
                'Original cached ES(m) per target, not point-position error:',368,size=10,height=30)
    writer.table(p,['t(s)','Full','GMM','dt300'],
                 [[f'{t:.0f}']+[f'{m["score"]["es_by_time_m"][i]:.2f}' for m in compared['models']]
                  for i,t in enumerate(seconds)],402,[85,140,140,150])
    writer.text(p,INTERPRETATION[writer.language][rank],552,size=11,height=90)
    writer.text(p,('读数：ES 是四时刻分布评分平均，FDE 是最后时刻粒子均值到真实目标的距离；两者越低越好。R90 是原经验径向预测圆半径，不能单独按越小越好排名。核心时长为原已就绪输入后的预测记录，不含加载、序列化、冷启动或整批成本，不是完整速度基准。' if zh else
                  'ES averages four distribution scores; FDE is the last particle mean-to-target distance; lower is better for both. R90 is the original radial disk radius, not a smaller-is-always-better metric. Original core timings exclude input loading, serialization, cold start and batch cost; they are not a complete runtime benchmark.'),665,size=10,height=112)


def figure_page(writer, selected, source_pdf, matched):
    zh=writer.language=='zh';rank=selected['original']['rank']
    title=('同例模型比较' if matched else 'Full 的实际路线与高程时域') if zh else ('Matched model comparison' if matched else 'Full: actual route and elevation horizons')
    p=writer.page(f'{title} | rank {rank}',landscape=True)
    source=fitz.open(source_pdf)
    if len(source)!=1:raise ValueError('original one-page vector case figure required')
    # Whole-page placement; do not crop particles, targets, disks or original legends.
    p.show_pdf_page(fitz.Rect(38,85,p.rect.width-38,485),source,0,keep_proportion=True)
    source.close()
    common=('黑线：真实未来路线；绿色：预测可见的三点历史与起点；菱形：粒子均值。虚线只连接四个保存均值，不是已保存的连续预测路线。坐标是以起点为原点的东/北米制，不是街道地址。' if zh else
            'Black: source observed future route; green: visible three-point history and origin; diamonds: particle means. Dashed joins guide four saved means, not a saved continuous forecast path. Axes are east/north metres from the origin, not street addresses.')
    specific=(('橙色点为原 512 粒子的最终分布，蓝圆为原评分缓存的 90% 圆，没有事后扩大。三个候选共享坐标范围。GMM 与 dt300 属于不同技术问题，分别对照 Full01；不能把三例视为总体模型排名。' if zh else
               'Orange points are all 512 final particles; blue circles are original cached 90% disks, not enlarged after inspection. All three panels share axes. GMM and dt300 answer distinct registered questions against Full01; these cases are not a population ranking.') if matched else
              ('左：WorldCover 与真实路线局部放大；中：高程背景下约 1/5 分钟粒子；右：约 15/30 分钟粒子。中/右时间颜色见原图例，左图的显示尺度不同。高程与地图叠图说明案例环境，不证明对应特征导致了改善或失败。' if zh else
               'Left: WorldCover and actual route zoom; middle: elevation and near-1/5-minute particles; right: near-15/30-minute particles. Original legends give time colours; the left panel has a different zoom. Map overlays explain context, not causal feature effects.'))
    writer.text(p,common,525,size=12,height=80)
    writer.text(p,specific,640,size=12,height=92)
    writer.text(p,('来源署名：ESA WorldCover 2021 v200；SRTM（目录收据）；Overture / OpenStreetMap / TomTom；HydroRIVERS（Lehner、Grill 2013）。完整许可要求见原稿来源节及署名记录；本图仍不具备公开路线发布授权。' if zh else
                  'Source credits: ESA WorldCover 2021 v200; SRTM (catalog receipts); Overture / OpenStreetMap / TomTom; HydroRIVERS (Lehner and Grill 2013). See the manuscript provenance section and acknowledgement record for full requirements; private-route release is not authorized.'),748,size=10,height=46)


def check_body(source, combined):
    """Check copied original pages, not experimental outputs or final acceptance."""
    for i,page in enumerate(source):
        copied=combined[i]
        if page.rect!=copied.rect or page.get_text()!=copied.get_text():
            raise ValueError('original manuscript page content changed')
        if page.get_pixmap().samples!=copied.get_pixmap().samples:
            raise ValueError('original manuscript page rendering changed')


def run(cases, comparisons, origins, paper, font, output):
    started=time.perf_counter();output=check_output(output)
    paths={
        'case-manifest.json':Path(cases)/'case-manifest.json',
        'comparison-manifest.json':Path(comparisons)/'comparison-manifest.json',
        'private-origin-bindings.json':Path(origins)/'private-origin-bindings.json',
        'map-source-acknowledgements-v1.json':Path(paper)/'map-source-acknowledgements-v1.json',
        **{f'{lang}/main.pdf':Path(paper)/lang/'main.pdf' for lang in ('en','zh')},
    }
    for key,path in paths.items():
        if file_hash(path)!=INPUT_SHA[key]:raise ValueError('original supplement input bytes differ: '+key)
    case=read_json(paths['case-manifest.json']);comparison=read_json(paths['comparison-manifest.json'])
    selected=validate_sources(case,comparison,read_json(paths['private-origin-bindings.json']))
    figures={}
    for item in selected:
        for key,root in [('original',cases),('compared',comparisons)]:
            row=item[key];png=under(root,row['figure']);pdf=png.with_suffix('.pdf')
            if file_hash(png)!=row['figure_sha256'] or file_hash(pdf)!=FIGURE_SHA[pdf.name]:
                raise ValueError('original case figure bytes differ')
            figures[(item['original']['rank'],key)]=pdf
    font=Path(font)
    if not font.is_file():raise ValueError('existing local CJK font required; no font download')
    output.mkdir(parents=True)
    manifest=dict(schema_version='pirc17-private-case-supplement-v1',private_local_review_only=True,
                  public_route_release_authorized=False,final_human_accepted=False,
                  source_sha256={**INPUT_SHA,**FIGURE_SHA},font_sha256=file_hash(font),
                  generator_sha256=file_hash(__file__),selected_ranks=list(RANKS),
                  new_forecasts=0,new_fits=0,new_particle_scores=0,new_inference_tests=0,
                  independent_saved_output_audit=False,original_manuscript_pages_unchanged=True,documents=[])
    for language in ('en','zh'):
        supplement=fitz.open();writer=Writer(supplement,language,font);cover(writer)
        for item in selected:
            case_notes(writer,item)
            for key in ('original','compared'):
                figure_page(writer,item,figures[(item['original']['rank'],key)],key=='compared')
        appendix_path=output/f'case-appendix-{language}.pdf'
        supplement.save(appendix_path,garbage=3,deflate=True)
        body=fitz.open(paths[f'{language}/main.pdf']);combined=fitz.open();combined.insert_pdf(body);combined.insert_pdf(supplement)
        bookmarks=body.get_toc()+[[1,'PRIVATE real-case appendix',len(body)+1]]
        bookmarks += [[2,f'Case {r}: notes, Full map and matched models',len(body)+2+3*i] for i,r in enumerate(RANKS)]
        combined.set_toc(bookmarks)
        combined.set_metadata(dict(title='PIRC-17 private review with real-case appendix',author='',
                                   subject='Not accepted; private routes are not publicly releasable'))
        path=output/f'private-review-{language}.pdf';combined.save(path,garbage=3,deflate=True)
        saved=fitz.open(path);check_body(body,saved)
        if len(saved)!=len(body)+10:raise ValueError('complete ten-page real-case appendix required')
        manifest['documents'].append(dict(language=language,body_pages=len(body),appendix_pages=10,
            pages=len(saved),pdf=path.name,pdf_sha256=file_hash(path),appendix=appendix_path.name,
            appendix_sha256=file_hash(appendix_path)))
        saved.close();body.close();combined.close();supplement.close()
    manifest['wall_seconds']=time.perf_counter()-started
    (output/'supplement-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='private_review_saved',documents=manifest['documents'],wall_seconds=manifest['wall_seconds'],new_forecasts=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cases','comparisons','origins','paper','font','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();run(**vars(args))


if __name__=='__main__':main()
