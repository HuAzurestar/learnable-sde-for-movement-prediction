"""Bounded IO tests using temporary software fixtures, never research data."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.pirc17 import method_development as module
from experiments.pirc17.method_development import PreparationBudget, load_method_development
from tests.test_pirc20_nex326_adapter import _write_release


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path):
    cohort_path, trajectory, conditions, dataset_id = _write_release(tmp_path)
    release = cohort_path.parent
    snapshot = tmp_path/"snapshot"
    snapshot.mkdir()
    samples = [json.loads(line) for line in (release/"samples.jsonl").read_text().splitlines()]
    entries, feature_entries = [], []
    for s in samples:
        # Offset source point indices, including non-segment points in each file.
        size = 13
        times = np.datetime64('2023-11-14T22:13:20', 'ns') + np.arange(size).astype('timedelta64[s]')
        path = conditions/f"{s['file_id']}_cond.parquet"
        pd.DataFrame({'file_id':[s['file_id']]*size,'t':times,
            'lon':114+np.arange(size)*1e-5,'lat':22+np.arange(size)*1e-5,
            'solar_elev':np.full(size,999999.)}).to_parquet(path,index=False)
        entries.append({'file_id':s['file_id'],'relative_path':path.name,'sha256':digest(path)})
        feature = snapshot/f"{s['file_id']}.parquet"
        pd.DataFrame({'file_id':[s['file_id']]*6,'segment_id':[s['segment_id']]*6,
            'point_index':np.arange(7,13),'absolute_epoch_ns':times[7:].astype(np.int64),
            'split':[s['split']]*6,'independent_block_id':[s['independent_block_id']]*6}).to_parquet(feature,index=False)
        feature_entries.append({'file_id':s['file_id'],'split':s['split'],'path':feature.name,
                                'sha256':digest(feature),'row_count':6})
    (release/'condition_file_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in entries),encoding='utf-8')
    dataset = json.loads((release/'dataset.json').read_text())
    dataset['artifacts']['condition_file_manifest.jsonl']['sha256'] = digest(release/'condition_file_manifest.jsonl')
    dataset['artifacts']['samples.jsonl'] = {'sha256':digest(release/'samples.jsonl')}
    save(release/'dataset.json',dataset)
    manifest={'dataset_id':dataset_id,'files':feature_entries}
    save(snapshot/'manifest.json',manifest)
    inventory=hashlib.sha256()
    for row in sorted(feature_entries,key=lambda r:r['path']):
        inventory.update(f"{row['path']}\0{row['sha256']}\0{row['row_count']}\n".encode())
    eligible={'dataset_id':dataset_id,'inputs_sha256':{'samples.jsonl':digest(release/'samples.jsonl')},
        'snapshot':{'content_inventory_sha256':inventory.hexdigest()},
        'eligibility':{'rows':[s for s in samples if s['split']!='final_eval']}}
    eligibility=tmp_path/'eligible.json'
    save(eligibility,eligible)
    return dict(eligibility_path=eligibility,eligibility_sha256=digest(eligibility),release=release,
        snapshot=snapshot,condition_root=conditions,trajectory_path=trajectory,
        sample_ids=[s['sample_id'] for s in eligible['eligibility']['rows']],
        budget=PreparationBudget(3,18,1_000_000,100,30.))


def rewrite_feature_binding(kwargs, mutation):
    manifest=json.loads((kwargs['snapshot']/'manifest.json').read_text())
    entry=next(e for e in manifest['files'] if e['split']=='train')
    path=kwargs['snapshot']/entry['path']
    frame=pd.read_parquet(path)
    mutation(frame)
    frame.to_parquet(path,index=False)
    entry['sha256']=digest(path)
    save(kwargs['snapshot']/'manifest.json',manifest)
    h=hashlib.sha256()
    for e in sorted(manifest['files'],key=lambda e:e['path']):
        h.update(f"{e['path']}\0{e['sha256']}\0{e['row_count']}\n".encode())
    eligible=json.loads(kwargs['eligibility_path'].read_text())
    eligible['snapshot']['content_inventory_sha256']=h.hexdigest()
    save(kwargs['eligibility_path'],eligible)
    kwargs['eligibility_sha256']=digest(kwargs['eligibility_path'])


def test_real_io_join_preserves_roles_regions_source_indexes_and_new_conditions(tmp_path):
    kwargs=fixture(tmp_path)
    # Missing held-out files must not be opened or be required by development IO.
    (kwargs['condition_root']/'evaluation-a_cond.parquet').unlink()
    (kwargs['snapshot']/'evaluation-a.parquet').unlink()
    prepared=load_method_development(**kwargs)
    assert prepared.identity['sample_counts']=={'train':1,'adapt':1,'validation':1}
    assert prepared.identity['observed_points']==18
    assert prepared.identity['final_eval_label_prediction_metric_reads']==0
    assert not prepared.identity['formal_training_accepted']
    for prefix,segment in zip(prepared.prefixes,prepared.segments):
        assert segment.region.startswith('fixture-city-')
        assert prefix.condition_at.frame.longitude==114+7e-5
        np.testing.assert_array_equal(segment.time,[-2,-1,0,1,2,3])
        np.testing.assert_array_equal(segment.state[:3],prefix.visible_positions_m)
        assert np.max(np.abs(segment.conditions['solar_elev']))<=90
    roles,report=prepared.resample(2.)
    assert all(len(v)==1 for v in roles.values())
    assert report['transitions_by_role']=={'train':3,'adapt':3,'validation':3}
    assert report['short_tails_by_role']=={'train':1,'adapt':1,'validation':1}
    assert report['minimum_interval_seconds']==1
    assert not report['noise_embedding_qualified']


@pytest.mark.parametrize('cap,value',[('max_samples',2),('max_total_points',17)])
def test_population_caps_fail_before_trajectory_metadata_or_point_loading(tmp_path,monkeypatch,cap,value):
    kwargs=fixture(tmp_path)
    kwargs['budget']=replace(kwargs['budget'],**{cap:value})
    monkeypatch.setattr(module,'load_source_regions',lambda *a:pytest.fail('must not open regions after cap failure'))
    with pytest.raises(ValueError,match='sample/point caps'):
        load_method_development(**kwargs)


def test_final_eval_selection_fails_before_any_point_loading(tmp_path,monkeypatch):
    kwargs=fixture(tmp_path)
    samples=[json.loads(r) for r in (kwargs['release']/'samples.jsonl').read_text().splitlines()]
    kwargs['sample_ids']=[next(s['sample_id'] for s in samples if s['split']=='final_eval')]
    monkeypatch.setattr(module,'load_source_regions',lambda *a:pytest.fail('must not load sealed selection'))
    with pytest.raises(ValueError,match='forbid final-eval'):
        load_method_development(**kwargs)


@pytest.mark.parametrize('which',['eligibility','condition','feature','trajectory'])
def test_input_corruption_is_not_silently_rebound(tmp_path,which):
    kwargs=fixture(tmp_path)
    path={'eligibility':kwargs['eligibility_path'],'condition':kwargs['condition_root']/'train-a_cond.parquet',
          'feature':kwargs['snapshot']/'train-a.parquet','trajectory':kwargs['trajectory_path']}[which]
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_method_development(**kwargs)


@pytest.mark.parametrize('kind',['time','index','role','block'])
def test_bound_but_inconsistent_source_join_is_rejected(tmp_path,kind):
    kwargs=fixture(tmp_path)
    column,value={'time':('absolute_epoch_ns',1),'index':('point_index',100),
                  'role':('split','final_eval'),'block':('independent_block_id','wrong')}[kind]
    rewrite_feature_binding(kwargs,lambda frame:frame.__setitem__(column,value))
    with pytest.raises(ValueError):
        load_method_development(**kwargs)


@pytest.mark.parametrize('kind',['ambiguous','missing','absent'])
def test_region_metadata_must_be_real_unique_and_present(tmp_path,kind):
    kwargs=fixture(tmp_path)
    frame=pd.read_parquet(kwargs['trajectory_path'])
    selected=frame.file_id=='train-a'
    if kind=='ambiguous': frame.loc[frame.index[selected][0],'city']='another-city'
    elif kind=='missing': frame.loc[selected,['city','region']]=None
    else: frame=frame[~selected]
    frame.to_parquet(kwargs['trajectory_path'],index=False)
    dataset=json.loads((kwargs['release']/'dataset.json').read_text())
    dataset['source']['trajectory']['sha256']=digest(kwargs['trajectory_path'])
    save(kwargs['release']/'dataset.json',dataset)
    with pytest.raises(ValueError,match='region|absent'):
        load_method_development(**kwargs)


@pytest.mark.parametrize('which',['max_file_bytes','max_file_rows'])
def test_individual_file_caps_are_enforced(tmp_path,which):
    kwargs=fixture(tmp_path)
    kwargs['budget']=replace(kwargs['budget'],**{which:1})
    with pytest.raises(ValueError,match='cap'):
        load_method_development(**kwargs)


def test_elapsed_budget_fails_without_retry(tmp_path):
    kwargs=fixture(tmp_path)
    kwargs['budget']=replace(kwargs['budget'],wall_seconds=1e-12)
    with pytest.raises(TimeoutError,match='no automatic retry'):
        load_method_development(**kwargs)


@pytest.mark.parametrize('field,value',[('max_samples',True),('max_total_points',0),
                                      ('max_file_rows',1.5),('wall_seconds',float('nan'))])
def test_preparation_caps_must_be_explicit_valid_values(field,value):
    with pytest.raises(ValueError):
        replace(PreparationBudget(3,18,1_000_000,100,30.),**{field:value})
