"""Synthetic software checks, never additional scientific cases or predictions."""
import copy
import json

import numpy as np
import pytest

from experiments.pirc17.case_review import saved_arrays
from experiments.pirc17.case_terrain import (
    validate_pair, render, case_note, marginal_regions, slot_bounds,
    manuscript_projection, render_horizons,
)


def fixture():
    original=dict(rank=0,sample_id='synthetic-case',independent_block_id='synthetic-block')
    models=[]
    for subject,es in [('base',2.),('all-terrain',1.)]:
        forecast=dict(matrix='terrain',configuration=subject,fit_identity='terrain-fit:'+subject,
            **original,seed=20260814,newly_generated_forecasts=0,maximum_step_seconds=5.,history_step_seconds=5.,
            raw_map_query_rows=0 if subject=='base' else 100,
            brownian_identity=dict(path_sha256='synthetic-stream'),prediction_seconds=1.)
        score=dict(status='success',weighted_es_m=es,fde_m=1.,es_by_time_m=[es]*4)
        models.append(dict(subject=subject,status='success',forecast=forecast,score=score,
            positions=np.zeros((512,4,2)),region=dict(center_m=[0.,0.],radius_m=2.,covered=True)))
    return models,original


def test_delta_direction_uses_terrain_own_control():
    models,original=fixture()
    assert validate_pair(models,original)==-1.
    models[1]['score'].update(weighted_es_m=3.,es_by_time_m=[3.]*4)
    assert validate_pair(models,original)==1.


def test_case_explanation_reports_direction_and_keeps_inference_boundary():
    models,original=fixture();models[1]['region']['covered']=False
    case=dict(models=models,case_es_difference_m=-1.)
    assert 'ES -1.00' in case_note(case,'en')
    assert 'outside' in case_note(case,'en') and 'Not a population' in case_note(case,'en')
    assert '不在' in case_note(case,'zh') and '不是总体' in case_note(case,'zh')
    case['case_es_difference_m']=None
    assert 'no case effect' in case_note(case,'en') and '不计算效应' in case_note(case,'zh')


@pytest.mark.parametrize('field,value',[
    ('matrix','NEX326-methods'),('configuration','loo-road'),('fit_identity','method-fit:full'),
    ('rank',3),('seed',20260815),('sample_id','other'),('independent_block_id','other'),
    ('maximum_step_seconds',10.),('history_step_seconds',2.),('newly_generated_forecasts',1),
])
def test_no_different_matrix_model_case_seed_or_clock(field,value):
    models,original=fixture();models[1]['forecast'][field]=value
    with pytest.raises(ValueError):validate_pair(models,original)


def test_shared_noise_and_actual_map_demand_are_checked():
    models,original=fixture();models[1]['forecast']['brownian_identity']={'path_sha256':'different'}
    with pytest.raises(ValueError):validate_pair(models,original)
    for subject_index,value in [(0,1),(1,0),(1,True)]:
        models,original=fixture();models[subject_index]['forecast']['raw_map_query_rows']=value
        with pytest.raises(ValueError):validate_pair(models,original)


def test_missing_case_is_not_dropped_replaced_or_given_an_effect():
    models,original=fixture();models[0]=dict(subject='base',status='failed')
    assert validate_pair(models,original) is None
    assert len(models)==2 and models[0]['status']=='failed'
    with pytest.raises(ValueError):validate_pair(models[1:],original)
    with pytest.raises(ValueError):validate_pair(list(reversed(models)),original)


def test_existing_score_is_not_zero_filled_or_averaged_differently():
    models,original=fixture();models[1]['score']['weighted_es_m']=0.
    with pytest.raises(ValueError):validate_pair(models,original)
    models,original=fixture();models[1]['score']['fde_m']=float('nan')
    with pytest.raises(ValueError):validate_pair(models,original)


def test_saved_terrain_selection_keeps_fixed_seed_and_failure():
    works=[dict(kind='scientific_forecast',matrix='terrain',subject='base',origin_mode='causal_prefix',
                origin_rank=0,seed=20260814+i,work_id=str(i)) for i in range(5)]
    settings=dict(workloads=works);progress=dict(completed={},failures={'0':'interrupted'})
    before=copy.deepcopy((settings,progress))
    arrays,evidence=saved_arrays(settings,{},progress,0,subject='base',matrix='terrain')
    assert arrays is None and evidence['status']=='failed' and evidence['seed']==20260814
    assert (settings,progress)==before
    with pytest.raises(ValueError):saved_arrays(settings,{},progress,0,subject='base')
    works[1]['seed']=20260814
    with pytest.raises(ValueError):saved_arrays(settings,{},progress,0,subject='base',matrix='terrain')


def test_render_preserves_failed_panel_and_writes_only_display(tmp_path):
    models,original=fixture();models[1]=dict(subject='all-terrain',status='failed')
    path=tmp_path/'synthetic-map.png'
    geo=dict(rank=0,harvest_area='Synthetic',country='XX')
    prefix=dict(terrain_prefix_positions_m=[[-2.,0.],[-1.,0.],[0.,0.]])
    target=dict(positions_m=[[1.,0.]]*4,elapsed_seconds=[60.,300.,900.,1800.])
    rasters=dict(worldcover=(np.array([[10,30],[40,50]]),[-3.,3.,-3.,3.]))
    vectors=dict(roads=[np.array([[-1.,0.],[1.,0.]])],rivers=[np.array([[0.,-1.],[0.,1.]])])
    render(path,geo,prefix,target,np.array([[0.,0.],[1.,0.]]),models,rasters,vectors,[-3.,-3.,3.,3.])
    assert path.is_file() and path.with_suffix('.pdf').is_file()
    import fitz
    with fitz.open(path.with_suffix('.pdf')) as doc:
        text=doc[0].get_text()
        assert 'Unavailable; not replaced' in text and 'all-terrain' in text
        assert 'NOT a continuous path' in text and 'three cases do NOT' in text
        assert 'WorldCover classes' in text and 'Cropland' in text and 'Built-up' in text
        assert 'Roads (Overture)' in text and 'Rivers (HydroRIVERS)' in text


def marginal_fixture():
    positions=np.zeros((512,4,2));seconds=np.array([60.,300.,900.,1800.])
    target=dict(positions_m=[[1.,0.]]*4,elapsed_seconds=seconds.tolist())
    row=dict(scores=dict(fde_m=1.,by_time=[dict(elapsed_seconds=t,region=dict(
        region='ensemble_mean_radial_quantile_disk',center_m=[0.,0.],
        levels=[dict(level=.9,radius_m=2.,covered=True,empirical_mass=.900390625)])) for t in seconds]))
    return positions,seconds,target,row


def test_all_four_original_regions_are_checked_not_only_the_last():
    positions,seconds,target,row=marginal_fixture()
    regions=marginal_regions(positions,seconds,target,row)
    assert len(regions)==4 and all(r['mean_error_m']==1. for r in regions)
    row['scores']['by_time'][0]['region']['center_m']=[99.,0.]
    with pytest.raises(ValueError):marginal_regions(positions,seconds,target,row)


@pytest.mark.parametrize('field,value',[('radius_m',0.),('radius_m',float('nan')),('covered',False)])
def test_early_slot_coverage_or_radius_cannot_be_adjusted(field,value):
    positions,seconds,target,row=marginal_fixture()
    row['scores']['by_time'][1]['region']['levels'][0][field]=value
    with pytest.raises(ValueError):marginal_regions(positions,seconds,target,row)


def test_each_slot_bounds_include_both_original_disks_and_all_particles():
    models,original=fixture()
    for m in models:m['regions']=[dict(center_m=[0.,0.],radius_m=2.,covered=True)]*4
    models[1]['positions'][0,0]=[1000.,-1000.]
    assert slot_bounds(models,0,[[0.,0.]],[[3.,0.]])==[-52.,-1050.,1050.,52.]


def test_manuscript_projection_has_metrics_and_figure_bindings_not_private_coordinates():
    models,original=fixture()
    for m in models:
        m.pop('positions')
        m['regions']=[dict(center_m=[123.,456.],radius_m=2.,covered=True,mean_error_m=1.)]*4
    case=dict(geography=dict(harvest_area='Synthetic',country='XX'),models=models,
        target_elapsed_seconds=[60.,300.,900.,1800.],case_es_difference_m=-1.,horizon_figures=[])
    cases=[dict(copy.deepcopy(case),rank=rank) for rank in (0,3,4)]
    manifest=dict(renderer_sha256='synthetic-hash',cases=cases)
    result=manuscript_projection(manifest)
    text=json.dumps(result)
    for private in ('center_m','positions','synthetic-case','synthetic-block','sample_id','work_id'):
        assert private not in text
    assert result['new_forecasts']==0 and result['public_distribution_authorized'] is False
    assert len(result['cases'][0]['models'][0]['horizons'])==4
    with pytest.raises(ValueError):manuscript_projection(dict(manifest,cases=cases[:-1]))


def test_four_time_plot_is_not_a_predicted_trajectory(tmp_path):
    models,original=fixture();positions,seconds,target,row=marginal_fixture()
    for m in models:m['regions']=marginal_regions(positions,seconds,target,row)
    models[1]=dict(subject='all-terrain',status='failed')
    path=tmp_path/'synthetic-horizons.png'
    render_horizons(path,dict(harvest_area='Synthetic',country='XX'),
        dict(terrain_prefix_positions_m=[[-2.,0.],[-1.,0.],[0.,0.]]),target,
        np.array([[0.,0.],[1.,0.]]),[0.,60.],models,{}, {},(0,1))
    import fitz
    with fitz.open(path.with_suffix('.pdf')) as doc:
        text=doc[0].get_text()
        assert 'nominal 1 min' in text and 'nominal 5 min' in text
        assert 'No predicted trajectory' in text and 'Unavailable; not replaced' in text
        assert '512 predicted positions' in text and 'Actual target' in text
