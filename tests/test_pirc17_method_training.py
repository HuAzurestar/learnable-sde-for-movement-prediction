"""Small generated software fixtures only; no real research fits or outcomes."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.nex326.cohort import _generated_segment
from experiments.pirc17 import method_training as module
from experiments.pirc17.method_training import (fit_development_method, prepare_method_training,
                                               required_slot, uniform_training_segment)


def fixture():
    roles = {}
    for role in ("train", "adapt", "validation"):
        roles[role] = []
        for i in range(6):
            source = _generated_segment(role, i, 8, 60., {})
            times = source.time.copy()
            times[-1] = times[-2]+17.
            roles[role].append(replace(source, time=times))
    return SimpleNamespace(identity={"sha256":"software-only-input", "solar_policy":"software-fixture",
        "frame_policy":"software-fixture"}, resample=lambda dt: (roles, {"nominal_interval_seconds":dt})), roles


def test_only_last_partial_interval_is_removed_without_mutating_original():
    _, roles = fixture()
    source = roles['train'][0]
    before = source.time.copy(), source.state.copy()
    result, report = uniform_training_segment(source, 60.)
    np.testing.assert_array_equal(result.time, [0,60,120,180,240,300,360])
    np.testing.assert_array_equal(source.time, before[0])
    np.testing.assert_array_equal(source.state, before[1])
    assert report['removed_tail_seconds']==17.
    assert not report['scoring_observations_changed']
    assert not result.state.flags.writeable


@pytest.mark.parametrize('kind',['internal','long','too_short'])
def test_invalid_grid_cannot_be_silently_repaired_or_sample_dropped(kind):
    _, roles=fixture()
    source=roles['train'][0]
    t=source.time.copy()
    if kind=='internal': t[2]-=1
    elif kind=='long': t[-1]=t[-2]+80
    else:
        source=replace(source,time=t[:4],state=source.state[:4],conditions={k:v[:4] for k,v in source.conditions.items()})
        t=source.time.copy(); t[-1]=t[-2]+17
    with pytest.raises(ValueError): uniform_training_segment(replace(source,time=t),60.)


def test_exact_regular_grid_retains_all_points():
    _,roles=fixture()
    source=roles['train'][0]
    source=replace(source,time=np.arange(8)*60.)
    result,report=uniform_training_segment(source,60.)
    assert len(result.time)==8 and not report['removed_partial_tail']


@pytest.mark.parametrize('slot',['arm-01/full','arm-02/pointwise','arm-03/single_gaussian',
                                  'arm-04/gmm_kernel','arm-05/explicit_decomp',
                                  'arm-08/qmle','arm-09/mixed','arm-09/pure_es',
                                  'arm-12/scratch','arm-14/reptile',
                                  'arm-15/drift_only','arm-15/two_step'])
def test_real_fitter_mechanisms_bind_fixed_reference_units_on_software_data(slot):
    prepared,_=fixture()
    result=fit_development_method(prepared,slot,max_transitions=200)
    components=required_slot(slot)['components']
    assert result.dynamics.model.model_kind==components['model']
    assert result.dynamics.model.estimator_method==components['estimator']
    assert result.dynamics.model.transfer_method==components['transfer']
    assert result.dynamics.model.finetune_method==components['finetune']
    assert result.training['transitions_by_role']=={'train':36,'adapt':36,'validation':36}
    assert result.training['removed_tails_by_role']=={'train':6,'adapt':6,'validation':6}
    assert result.dynamics.reference_interval_seconds==60.
    assert not result.training['formal_training_accepted']
    assert not result.dynamics.identity()['embedding_is_continuous_time_fit_qualification']
    if components['transfer']=='meta_reptile': assert result.dynamics.model.meta_task_count==2


def test_equivalent_fit_components_share_identity_but_do_not_merge_slots():
    prepared,_=fixture()
    left=fit_development_method(prepared,'arm-01/full',max_transitions=200)
    right=fit_development_method(prepared,'arm-07/full',max_transitions=200)
    assert left.dynamics.fit_identity==right.dynamics.fit_identity
    assert left.training['training_identity_sha256']==right.training['training_identity_sha256']
    assert left.slot_id!=right.slot_id


def test_total_transition_cap_applies_before_any_fitting(monkeypatch):
    prepared,_=fixture()
    monkeypatch.setattr(module,'train_model',lambda *a:pytest.fail('must not fit over budget'))
    with pytest.raises(ValueError,match='budget exceeded'):
        fit_development_method(prepared,'arm-01/full',max_transitions=1)


@pytest.mark.parametrize('slot',['arm-13/animal_pretrain','unknown'])
def test_excluded_or_unknown_slot_cannot_trigger_fit(slot,monkeypatch):
    prepared,_=fixture()
    monkeypatch.setattr(module,'train_model',lambda *a:pytest.fail('must not fit excluded slot'))
    with pytest.raises(ValueError,match='REQUIRED'):
        fit_development_method(prepared,slot,max_transitions=200)
