"""Local, source-bound case illustrations. Never forecasts, fits or qualifies.

Outputs contain private routes and are restricted to an ignored .local folder.
Only the geography-summary.json projection omits routes and recording IDs.
Cases are the first admitted block from each of the first three encountered
countries, in sealed population order, using the first registered RNG seed.
Selection never consults prediction errors or completion status.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import pyarrow.parquet as pq

from .checkpoint_resume import load
from .checkpoint_saved import source_leaf
from .features import LocalFrame
from .protocol_core import file_hash, read_json, under, unpack

VERSION = 'pirc17-private-case-review-v1'


def select_cases(rows, limit=3):
    """First per encountered country; no outcome/status fields inspected."""
    seen, selected = set(), []
    for row in rows:
        if row['country'] not in seen:
            seen.add(row['country'])
            selected.append(row)
        if len(selected) == limit:
            break
    return selected


def geographical_summary(rows):
    if len(rows) != 46 or len({r['sample_id'] for r in rows}) != 46:
        raise ValueError('original 46-block population required; no subset summary')
    countries = Counter(r['country'] for r in rows)
    areas = Counter((r['country'], r['region'], r['harvest_area']) for r in rows)
    return dict(schema_version=VERSION, independent_blocks=46,
        country_count=len(countries), countries=dict(sorted(countries.items())),
        harvest_areas=[dict(country=k[0], region=k[1], harvest_area=k[2], blocks=n)
                       for k, n in sorted(areas.items())],
        location_semantics='source cluster label, not verified point-level city/street address',
        route_coordinates_included=False, recording_ids_included=False,
        geographical_performance_inference=False)


def check_output(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[2]
    private = root/'.local'
    if private not in path.parents or path == private:
        raise ValueError('private case output must be a new child of PSDE .local')
    if path.exists():
        raise FileExistsError('retain earlier evidence; choose a new output directory')
    return path


def bind_geography(settings, context, raw_path, metadata_path):
    release = Path(settings['input_paths']['release'])
    dataset = read_json(release/'dataset.json')
    if file_hash(raw_path) != dataset['source']['raw_harvest']['sha256']:
        raise ValueError('raw recording-to-cluster input differs from release source')
    # Read only two small columns, not raw coordinate/speed strings.
    mapping = pq.read_table(raw_path, columns=['file_id', 'cluster_A']).to_pylist()
    ids = [r['file_id'] for r in mapping]
    if len(ids) != len(set(ids)):
        raise ValueError('ambiguous raw recording mapping')
    clusters = {r['file_id']:r['cluster_A'] for r in mapping}
    metadata = read_json(metadata_path)
    samples = {}
    with (release/'samples.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            samples[row['sample_id']] = row
    result = []
    for rank, admitted in enumerate(unpack(context['population'])['selection']['selected']):
        sample = samples[admitted['sample_id']]
        if (sample['split'] != admitted['split'] or
                sample['independent_block_id'] != admitted['independent_block_id']):
            raise ValueError('release sample differs from sealed admission')
        label = metadata[str(clusters[sample['file_id']])]
        if len(label) != 3 or any(not isinstance(v, str) or not v for v in label):
            raise ValueError('unavailable source location; no geocoding replacement')
        result.append(dict(rank=rank, sample_id=sample['sample_id'],
            file_id=sample['file_id'], country=label[0], region=label[1],
            harvest_area=label[2].split('_', 1)[0], original_cluster_label=label))
    summary = geographical_summary(result)
    summary['provenance'] = dict(context_sha256=settings['context_sha256'],
        population_sha256=context['population']['sha256'],
        raw_harvest_sha256=dataset['source']['raw_harvest']['sha256'],
        samples_sha256=file_hash(release/'samples.jsonl'),
        cluster_metadata_sha256=file_hash(metadata_path),
        cluster_metadata_in_original_release_binding=False,
        label_normalization='remove suffix beginning at first underscore; no external geocoding')
    return result, summary


def saved_arrays(settings, imported, progress, rank, *, subject='arm-01/full', matrix='NEX326-methods'):
    if matrix not in ('NEX326-methods', 'terrain'):
        raise ValueError('original method or terrain matrix required')
    works = [w for w in settings['workloads'] if w['kind']=='scientific_forecast'
        and w['matrix']==matrix and w['subject']==subject
        and w['origin_mode']=='causal_prefix' and w['origin_rank']==rank]
    if len(works) != 5 or sorted(w['seed'] for w in works) != list(range(20260814, 20260819)):
        raise ValueError('registered case requires original five-seed design')
    work = min(works, key=lambda w:w['seed'])
    binding = imported.get(work['work_id'], {}).get('manifest') or progress['completed'].get(work['work_id'])
    if not binding:
        return None, dict(status='failed' if work['work_id'] in progress['failures'] else 'unavailable',
                          rank=rank, subject=subject, seed=work['seed'], work_id=work['work_id'])
    record = read_json(binding['artifact_path'])
    value = unpack(record, expected_sha256=binding['artifact_sha256'])
    if value['status'] != 'success' or value['work_id'] != work['work_id']:
        raise ValueError('selected completion is not this successful registered work')
    leaf, leaf_path = source_leaf(record, binding['artifact_path'], scientific=True)
    arrays = unpack(leaf)['arrays']
    path = under(leaf_path.parent, arrays['path'])
    if file_hash(path) != arrays['sha256'] or path.stat().st_size != arrays['bytes']:
        raise ValueError('selected saved array bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        positions = archive['positions_m'].copy()
        seconds = archive['elapsed_seconds'].copy()
    if (positions.shape != (512,4,2) or seconds.shape != (4,)
            or not np.isfinite(positions).all() or not np.isfinite(seconds).all()):
        raise ValueError('invalid saved case arrays; no replacement')
    evidence = dict(status='success', rank=rank, seed=work['seed'],
        work_id=work['work_id'], subject=subject, record_sha256=record['sha256'],
        sample_id=value['sample_id'], independent_block_id=value['independent_block_id'],
        leaf_record_sha256=leaf['sha256'], array_sha256=arrays['sha256'],
        prediction_seconds=value['prediction_seconds'], newly_generated_forecasts=0)
    if matrix == 'terrain':
        if value['scientific_work'] != work['scientific'] or value['origin_mode'] != 'causal_prefix':
            raise ValueError('terrain scientific routing differs from the registered case')
        evidence.update(matrix=matrix, fit_identity=value['fit_identity'],
            configuration=value['diagnostics']['configuration'],
            brownian_identity=value['diagnostics']['brownian_identity'],
            maximum_step_seconds=value['diagnostics']['maximum_step_seconds'],
            history_step_seconds=value['diagnostics']['history_step_seconds'],
            raw_map_query_rows=value['maps']['query_rows_this_forecast'])
    return (positions, seconds), evidence


def observed_route(settings, prefix, target, geography):
    source = None
    release = Path(settings['input_paths']['release'])
    with (release/'condition_file_manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['file_id'] == geography['file_id']:
                if source is not None:
                    raise ValueError('ambiguous condition source')
                source = row
    if source is None or source['sha256'] != prefix['condition_sha256']:
        raise ValueError('source condition differs from sealed prefix')
    path = under(Path(settings['input_paths']['data_root'])/'cond_slices', source['relative_path'])
    if file_hash(path) != source['sha256']:
        raise ValueError('selected condition source bytes changed')
    table = pq.read_table(path, columns=['file_id','t','lon','lat'])
    if set(table['file_id'].to_pylist()) != {geography['file_id']}:
        raise ValueError('selected condition contains a different recording')
    epochs = table['t'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    times = (epochs-prefix['origin_epoch_ns'])/1e9
    frame = LocalFrame(*prefix['scoring_frame'])
    xy = frame.from_lonlat(np.column_stack((table['lon'].to_numpy(),table['lat'].to_numpy())))
    for second, position in zip(target['elapsed_seconds'], target['positions_m']):
        indexes = np.flatnonzero(times == second)
        if len(indexes) != 1 or not np.allclose(xy[indexes[0]],position,rtol=0,atol=1e-6):
            raise ValueError('source target join differs; no interpolation/filling')
    mask = (times >= 0) & (times <= target['elapsed_seconds'][-1])
    route, seconds = xy[mask], times[mask]
    if len(route)<4 or not np.isfinite(route).all() or np.any(np.diff(seconds)<=0):
        raise ValueError('invalid original future observation sequence')
    return route, dict(condition_sha256=source['sha256'], observed_future_points=len(route),
        elapsed_seconds=seconds.tolist(), time_semantics='retained condition clock; not independently verified sensor timestamps')


def tile_name(latitude, longitude):
    return f'{"N" if latitude>=0 else "S"}{abs(latitude):02d}{"E" if longitude>=0 else "W"}{abs(longitude):03d}'


def map_layers(settings, catalog, frame, bounds):
    """Bounded raster windows and spatially filtered original vectors only."""
    import duckdb
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window, from_bounds
    import shapely
    root = Path(settings['input_paths']['data_root'])
    assets = catalog['asset_sha256']
    lonlat = frame.to_lonlat([[bounds[0],bounds[1]],[bounds[2],bounds[3]]])
    west,south = lonlat[0]; east,north = lonlat[1]
    cell = tile_name(3*math.floor(frame.latitude/3),3*math.floor(frame.longitude/3))
    dem_tile = tile_name(math.floor(frame.latitude),math.floor(frame.longitude))
    candidates = dict(worldcover=f'map_data/worldcover_priority_2021_v200/ESA_WorldCover_10m_2021_v200_{cell}_Map.tif',
        dem=next((k for k in sorted(assets,key=lambda p:('priority' not in p,p))
                  if 'srtm' in k and k.endswith(dem_tile+'.hgt.gz')),None),
        roads=f'map_data/overture_priority_2026_08_19/{cell}.parquet',
        rivers='map_data/hydrology/hydrorivers_v1/HydroRIVERS_v10_eu_shp.zip')
    evidence, rasters, vectors = {}, {}, {}
    for kind,relative in candidates.items():
        if relative is None or relative not in assets:
            evidence[kind] = dict(status='unavailable', reason='no selected origin tile in saved catalog')
            continue
        path = under(root,relative)
        if file_hash(path) != assets[relative]:
            raise ValueError('selected map asset bytes differ from saved catalog')
        evidence[kind] = dict(status='read', asset=relative, sha256=assets[relative])
        if kind not in ('dem','worldcover'):
            continue
        raster_path = '/vsigzip/'+path.as_posix() if relative.endswith('.gz') else str(path)
        with rasterio.open(raster_path) as src:
            if src.crs is None or src.crs.to_epsg()!=4326:
                raise ValueError('unhandled raster CRS; no assumed alignment')
            b=src.bounds
            clip=(max(west,b.left),max(south,b.bottom),min(east,b.right),min(north,b.top))
            if clip[0]>=clip[2] or clip[1]>=clip[3]:
                evidence[kind]['status']='no_overlap';continue
            window=from_bounds(*clip,src.transform).intersection(Window(0,0,src.width,src.height))
            height=min(512,max(1,math.ceil(window.height)))
            width=min(512,max(1,math.ceil(window.width)))
            data=src.read(1,window=window,out_shape=(height,width),masked=True,resampling=Resampling.nearest)
            actual=rasterio.windows.bounds(window,src.transform)
            corners=frame.from_lonlat([[actual[0],actual[1]],[actual[2],actual[3]]])
            extent=(corners[0,0],corners[1,0],corners[0,1],corners[1,1])
            rasters[kind]=(data,extent)
            evidence[kind].update(crs='EPSG:4326', shape=[height,width],
                display_sampling='nearest, at most 512 pixels per dimension',
                nodata_fraction=float(np.ma.getmaskarray(data).mean()),
                whole_view_covered=bool(b.left<=west and b.right>=east and b.bottom<=south and b.top>=north))
    db=duckdb.connect()
    try:
        db.execute('SET threads=1');db.execute("SET memory_limit='512MB'");db.execute('LOAD spatial')
        for kind in ('roads','rivers'):
            if evidence[kind]['status']!='read':continue
            path=under(root,candidates[kind])
            if kind=='roads':
                sql="SELECT ST_AsWKB(geometry) FROM read_parquet(?) WHERE subtype='road' AND ST_Intersects(geometry, ST_MakeEnvelope(?,?,?,?))"
                query_path=str(path)
            else:
                with zipfile.ZipFile(path) as archive:
                    members=sorted(n for n in archive.namelist() if n.endswith('.shp'))
                if len(members)!=1:raise ValueError('ambiguous HydroRIVERS source')
                query_path='/vsizip/'+path.as_posix()+'/'+members[0]
                sql='SELECT ST_AsWKB(geom) FROM ST_Read(?) WHERE ST_Intersects(geom, ST_MakeEnvelope(?,?,?,?))'
            values=db.execute(sql,[query_path,float(west),float(south),float(east),float(north)]).fetchall()
            lines=[]
            rectangle=shapely.box(west,south,east,north)
            for (value,) in values:
                geometry=shapely.from_wkb(bytes(value)).intersection(rectangle)
                parts=list(geometry.geoms) if geometry.geom_type=='MultiLineString' else [geometry]
                for part in parts:
                    if part.geom_type=='LineString' and not part.is_empty:
                        lines.append(frame.from_lonlat(np.asarray(part.coords)[:,:2]))
            vectors[kind]=lines;evidence[kind]['visible_line_parts']=len(lines)
    finally:
        db.close()
    return rasters,vectors,evidence


def render_case(path, geography, prefix, target, positions, route, rasters, vectors, bounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    classes=[10,20,30,40,50,60,70,80,90,95,100]
    colors=['#006400','#ffbb22','#ffff4c','#f096ff','#fa0000','#b4b4b4','#f0f0f0','#0064c8','#0096a0','#00cf75','#fae6a0']
    labels=['Trees','Shrub','Grass','Cropland','Built-up','Bare','Snow/ice','Water','Wetland','Mangrove','Moss/lichen']
    cmap=ListedColormap(colors);norm=BoundaryNorm([5,15,25,35,45,55,65,75,85,92.5,97.5,105],len(colors))
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    truth=np.asarray(target['positions_m']);prefix_xy=np.asarray(prefix['terrain_prefix_positions_m'])
    means=positions.mean(axis=0)
    for j,ax in enumerate(axes):
        kind='worldcover' if j==0 else 'dem'
        if kind in rasters:
            data,extent=rasters[kind]
            im=ax.imshow(data,extent=extent,origin='upper',alpha=.75,
                cmap=cmap if kind=='worldcover' else 'terrain',norm=norm if kind=='worldcover' else None)
            if kind=='dem':fig.colorbar(im,ax=ax,shrink=.65,label='Elevation (m, SRTM)')
        else:ax.text(.02,.98,kind+' unavailable',va='top',transform=ax.transAxes)
        for key,color in [('roads','#555555'),('rivers','#087aca')]:
            ax.add_collection(LineCollection(vectors.get(key,[]),colors=color,linewidths=.7,alpha=.55))
        ax.plot(route[:,0],route[:,1],color='black',linewidth=2,label='Observed future route')
        ax.plot(prefix_xy[:,0],prefix_xy[:,1],color='#008f39',linewidth=3)
        ax.scatter(*prefix_xy[-1],marker='*',s=140,color='#008f39',zorder=6)
        ax.scatter(truth[:,0],truth[:,1],marker='X',s=75,color='black',zorder=6)
        if j>0:
            indexes=[0,1] if j==1 else [2,3]
            guide=np.vstack(([0.,0.],means[indexes]))
            ax.plot(guide[:,0],guide[:,1],'--',color='#e46411',linewidth=1.5)
            for k in indexes:
                color=['#9a4fca','#0055bb','#dd5555','#e09c00'][k]
                ax.scatter(positions[:,k,0],positions[:,k,1],s=3,color=color,alpha=.3)
                ax.scatter(*means[k],marker='D',s=55,color=color,zorder=7,
                    label=f'{target["elapsed_seconds"][k]:.0f} s: particle mean')
            ax.legend(loc='lower left',fontsize=8)
        if j==0:
            lo=route.min(axis=0)-150;hi=route.max(axis=0)+150
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
            present=set(np.unique(rasters['worldcover'][0].compressed())) if 'worldcover' in rasters else set()
            ax.legend(handles=[Line2D([0],[0],color=c,lw=5,label=l) for v,c,l in zip(classes,colors,labels) if v in present],loc='best',fontsize=8)
        else:ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3])
        ax.set_aspect('equal',adjustable='box');ax.set_xlabel('East from origin (m)');ax.set_ylabel('North from origin (m)')
        ax.set_title(['WorldCover + actual route (zoom)','DEM + Full: near 1 and 5 min','DEM + Full: near 15 and 30 min'][j])
    errors=np.linalg.norm(means-truth,axis=1)
    fig.suptitle(f'Private review | rank {geography["rank"]}: {geography["harvest_area"]}, {geography["region"]}, {geography["country"]}\n'
        f'Full; seed 20260814; N=512; saved 4 times; endpoint mean error {errors[-1]:.1f} m\n'
        'Black: actual future; green: visible 3-point prefix; colored dots: particles; diamonds: means; dashed joins: guides, NOT saved continuous forecast paths',fontsize=11)
    fig.savefig(path,dpi=140);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
    return errors.tolist()


def run(runtime,raw,metadata,output):
    started=time.perf_counter();output=check_output(output)
    settings,imported=load(runtime);progress=read_json(Path(runtime)/'progress.json')
    context=unpack(read_json(settings['context']),expected_sha256=settings['context_sha256'])
    rows,summary=bind_geography(settings,context,raw,metadata)
    selected=select_cases(rows)
    prefixes={r['sample_id']:r for r in context['causal_prefixes']}
    targets={r['sample_id']:r for r in unpack(context['scoring_inputs'])['targets']}
    output.mkdir(parents=True)
    (output/'geography-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest=dict(schema_version=VERSION,private_local_review_only=True,
        public_route_release_authorized=False,independent_saved_output_audit=False,
        selection_rule='first block per first three encountered source countries in sealed primary order; first seed; no outcome filtering',
        selected_ranks=[r['rank'] for r in selected],new_forecasts=0,new_fits=0,cases=[])
    for row in selected:
        arrays,evidence=saved_arrays(settings,imported,progress,row['rank'])
        evidence['geography']=row
        manifest['cases'].append(evidence)
        if arrays is None:continue
        positions,seconds=arrays;prefix=prefixes[row['sample_id']];target=targets[row['sample_id']]
        if (evidence['sample_id']!=row['sample_id'] or
                evidence['independent_block_id']!=prefix['independent_block_id']):
            raise ValueError('selected forecast does not bind the admitted observation block')
        if evidence['status']=='success' and not np.array_equal(seconds,target['elapsed_seconds']):
            raise ValueError('selected forecast clock differs from original target clock')
        route,route_evidence=observed_route(settings,prefix,target,row)
        cloud=np.concatenate((positions.reshape(-1,2),route,np.asarray(prefix['terrain_prefix_positions_m'])))
        low=cloud.min(axis=0)-150;high=cloud.max(axis=0)+150
        bounds=[low[0],low[1],high[0],high[1]]
        rasters,vectors,maps=map_layers(settings,unpack(context['map_catalog']),LocalFrame(*prefix['scoring_frame']),bounds)
        name=f'case-rank-{row["rank"]:02d}.png'
        errors=render_case(output/name,row,prefix,target,positions,route,rasters,vectors,bounds)
        evidence.update(route=route_evidence,maps=maps,figure=name,figure_sha256=file_hash(output/name),
            target_elapsed_seconds=seconds.tolist(),point_mean_errors_m=errors,
            point_error_semantics='case description only; not new primary statistics or geographical inference')
        print(json.dumps(dict(event='private_case_rendered',rank=row['rank'],country=row['country'],future_points=route_evidence['observed_future_points']),ensure_ascii=False),flush=True)
    manifest['review_wall_seconds']=time.perf_counter()-started
    (output/'case-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='local_review_saved',cases=len(manifest['cases']),countries=summary['country_count'],wall_seconds=manifest['review_wall_seconds']),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('runtime','raw','metadata','output'):parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args();run(args.runtime,args.raw,args.metadata,args.output)


if __name__=='__main__':main()
