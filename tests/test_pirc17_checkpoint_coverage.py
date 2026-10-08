"""Synthetic metadata fixtures, not new PIRC17 forecasts or statistical tests."""
import copy

import pytest

from experiments.pirc17.checkpoint_coverage import POLICY, MODES, census, registered_design, source_binding
from experiments.pirc17.protocol_core import digest, read_json


def fixture():
    policy=read_json(POLICY);_,configs,seeds=registered_design(policy)
    works=[]
    for matrix,subjects in configs.items():
        for subject in sorted(subjects):
            for mode,n in MODES.items():
                for rank in range(n):
                    for seed in seeds:
                        row=dict(kind='scientific_forecast',matrix=matrix,subject=subject,
                                 origin_mode=mode,origin_rank=rank,seed=seed)
                        row['work_id']=digest(row);works.append(row)
    settings=dict(settings_id='fixture',workloads=works)
    progress=dict(settings_id='fixture',completed={},failures={})
    return settings,{},progress,dict(rows={}),policy


def test_original_counts_and_whole_family_pending():
    args=fixture();report=census(*args)
    assert report['scientific']['expected']==11020
    assert report['scientific']['forecast_runnable']==11020
    assert report['scientific']['disposition']=='pending_missing'
    assert len(report['configurations'])==114 and len(report['comparison_families'])==21
    assert all(not f['scientific_claim_authorized'] for f in report['comparison_families'])
    primary={f['family_id']:f for f in report['comparison_families'] if f['origin_mode']=='causal_prefix'}
    assert primary['weighted-es-primary']['expected']==1380
    assert primary['weighted-es-lio']['expected']==1150
    assert report['family_rows_overlap_and_cannot_be_summed']


@pytest.mark.parametrize('change',['drop','duplicate','unknown_mode','settings','policy'])
def test_no_successful_subset_or_changed_design(change):
    settings,imported,progress,index,policy=fixture()
    if change=='drop':settings['workloads'].pop()
    if change=='duplicate':settings['workloads'][-1]=copy.deepcopy(settings['workloads'][0])
    if change=='unknown_mode':settings['workloads'][0]['origin_mode']='other'
    if change=='settings':progress['settings_id']='other'
    if change=='policy':policy['parameters']['seeds']['value'].pop()
    with pytest.raises(ValueError):census(settings,imported,progress,index,policy)


def test_failed_required_row_is_not_zero_or_dropped():
    settings,imported,progress,index,policy=fixture();w=settings['workloads'][0]
    progress['failures'][w['work_id']]='synthetic failure'
    index['rows'][w['work_id']]=dict(status='failed',source_sha256=source_binding(w['work_id'],imported,progress))
    report=census(settings,imported,progress,index,policy)
    assert report['scientific']['forecast_failed']==1 and report['scientific']['expected']==11020
    assert report['scientific']['score_index_failed']==1
    containing=[f for f in report['comparison_families'] if f['origin_mode']==w['origin_mode'] and f['family_id'] in {'method-model-structure','method-observation-interval'}]
    assert all(f['disposition']=='unavailable_failed' for f in containing)


@pytest.mark.parametrize('entry',[{},'empty',dict(status='success',source_sha256='wrong')])
def test_empty_or_bad_index_does_not_become_a_valid_score(entry):
    settings,imported,progress,index,policy=fixture();w=settings['workloads'][0]
    progress['completed'][w['work_id']]=dict(artifact_path='fixture',artifact_sha256='a'*64)
    index['rows'][w['work_id']]=entry
    report=census(settings,imported,progress,index,policy)
    assert report['scientific']['score_index_invalid']==1
    assert report['scientific']['disposition']=='unavailable_failed'


def test_successful_index_is_not_payload_validation_and_inputs_unchanged():
    settings,imported,progress,index,policy=fixture();w=settings['workloads'][0]
    imported[w['work_id']]=dict(manifest=dict(artifact_path='fixture',artifact_sha256='a'*64))
    index['rows'][w['work_id']]=dict(status='success',source_sha256=source_binding(w['work_id'],imported,progress))
    before=copy.deepcopy((settings,imported,progress,index,policy))
    report=census(settings,imported,progress,index,policy)
    assert (settings,imported,progress,index,policy)==before
    assert report['scientific']['score_index_success']==1
    assert not report['scientific']['score_payloads_independently_validated']
    assert report['new_forecasts']==report['new_particle_scores']==report['new_inference_tests']==0


def test_complete_indices_still_do_not_authorize_a_scientific_claim():
    settings,imported,progress,index,policy=fixture()
    for w in settings['workloads']:
        progress['completed'][w['work_id']]=dict(artifact_path='fixture',artifact_sha256='a'*64)
        index['rows'][w['work_id']]=dict(status='success',source_sha256=source_binding(w['work_id'],imported,progress))
    report=census(settings,imported,progress,index,policy)
    assert report['scientific']['forecast_success']==11020
    assert report['scientific']['disposition']=='index_complete_validation_pending'
    assert not report['scientific_claim_authorized']
    assert all(f['disposition']=='index_complete_validation_pending' and not f['scientific_claim_authorized']
               for f in report['comparison_families'])


def test_conflicting_terminal_states_are_rejected():
    settings,imported,progress,index,policy=fixture();wid=settings['workloads'][0]['work_id']
    progress['completed'][wid]=dict(artifact_path='fixture',artifact_sha256='a'*64)
    progress['failures'][wid]='synthetic failure'
    with pytest.raises(ValueError,match='both successful and failed'):
        census(settings,imported,progress,index,policy)
