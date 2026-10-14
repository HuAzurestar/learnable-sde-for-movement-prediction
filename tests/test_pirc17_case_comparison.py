"""Small synthetic software fixtures, never new PIRC17 scientific experiments."""
import numpy as np
import pytest

from experiments.pirc17.case_comparison import comparison_bounds, validate_case
from experiments.pirc17.case_review import saved_arrays


def fixture():
    positions=np.zeros((512,4,2));seconds=np.array([60.,300.,900.,1800.])
    target=dict(positions_m=[[1.,0.]]*4,elapsed_seconds=seconds.tolist())
    row=dict(scores=dict(fde_m=1.,by_time=[dict(elapsed_seconds=t,region=dict(
        region='ensemble_mean_radial_quantile_disk',center_m=[0.,0.],
        levels=[dict(level=.9,radius_m=2.,covered=True,empirical_mass=.900390625)])) for t in seconds]))
    return positions,seconds,target,row


def test_case_validates_original_arrays_clock_fde_and_region():
    positions,seconds,target,row=fixture()
    assert validate_case(positions,seconds,target,row)['radius_m']==2.
    row['scores']['fde_m']=2.
    with pytest.raises(ValueError):validate_case(positions,seconds,target,row)


def test_clock_and_circle_cannot_be_replaced():
    positions,seconds,target,row=fixture()
    with pytest.raises(ValueError):validate_case(positions,seconds+1,target,row)
    row['scores']['by_time'][-1]['region']['center_m']=[5.,0.]
    with pytest.raises(ValueError):validate_case(positions,seconds,target,row)


def test_cached_coverage_must_describe_the_original_target():
    positions,seconds,target,row=fixture()
    row['scores']['by_time'][-1]['region']['levels'][0]['covered']=False
    with pytest.raises(ValueError):validate_case(positions,seconds,target,row)


def test_bounds_include_all_particle_extremes_truth_and_original_disks():
    positions,seconds,target,row=fixture();positions[0,0]=[1000.,-1200.]
    model=dict(status='success',positions=positions,region=dict(center_m=[0.,0.],radius_m=2000.))
    assert comparison_bounds([[3000.,0.]], [model])==[-2150.,-2150.,3150.,2150.]


def test_missing_selected_candidate_keeps_its_axes_and_is_not_substituted():
    works=[dict(kind='scientific_forecast',matrix='NEX326-methods',subject='arm-06/dt300',
                origin_mode='causal_prefix',origin_rank=3,seed=20260814+i,work_id=str(i)) for i in range(5)]
    arrays,evidence=saved_arrays(dict(workloads=works),{},dict(completed={},failures={'0':'failed'}),3,subject='arm-06/dt300')
    assert arrays is None and evidence['status']=='failed' and evidence['seed']==20260814
    with pytest.raises(ValueError):saved_arrays(dict(workloads=works),{},dict(completed={},failures={}),3,subject='invented-model')
