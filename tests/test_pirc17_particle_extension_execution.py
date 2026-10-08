"""Actual missing-only core and real-kernel equivalence, using software data."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pirc17 import particle_extension_execution as execution
from experiments.pirc17.rollout import rollout as real_rollout
from tests.test_pirc17_particle_extension import fixture_reference, generate_new
from tests.test_pirc17_particle_extension_lineage import (
    candidate_bytes, evidence, inputs, reference, case, reserve, seal)

lineage = execution.lineage


def fixture_runtime(source,locations,monkeypatch):
    # Only this fixture's nonexistent eligibility file uses its synthetic
    # identity. All model/ledger/particle/source/journal hashes remain real.
    sha = lineage.science.load_plan(source['plan'],source['plan_sha256'])['eligibility_sha256']
    original = execution.native._hash
    monkeypatch.setattr(execution.native,'_hash',lambda p:sha if Path(p)==locations['eligibility'] else original(p))
    monkeypatch.setattr(execution.native.physical_memory,'available_physical_bytes',lambda:execution.engine.MINIMUM_FREE_BYTES)


@pytest.fixture
def ready(case,monkeypatch):
    fixture_runtime(case[0],case[1],monkeypatch)
    return case


def run(reservation,**kwargs):
    return execution.run(**{k:reservation[k] for k in ('directory','root_sha256','index','start_sha256')},**kwargs)


def interrupt_after(count):
    def stop(event):
        if event['phase']=='run_committed' and event['new_committed']==count:
            raise KeyboardInterrupt('software user interruption')
    return stop


def assert_exact_result(case,reservation,result):
    _,_,prepared,path,baseline,_=case
    work=reservation['attempt']/'work'
    saved=execution.journal.read(work/'journal',result['journal_tip']['manifest_sha256'],expected_tip=result['journal_tip'])
    rows=execution.assembled(work,saved)
    for row,expected in zip(rows,baseline):
        assert all(np.array_equal(a,b) for a,b in zip(execution.load_particle_evidence(row,work),
            execution.load_particle_evidence(expected,path.parent)))
        for key in ('scores','particle_precision','brownian_identity','model_identity_sha256','feature_query_rows',
                    'invalid_feature_rows','actual_horizons_seconds','independent_block_id'):
            assert row[key]==expected[key]
    science=json.loads((work/'whole-science.json').read_text())
    assert science['reference_runs_covered']==len(prepared.reference)-3
    assert science['new_run_count']==len(prepared.keys)==len(rows)
    assert science['reference_numerical']['out_of_tolerance']>0
    assert result['production_resume_ready'] is False and result['resources']['outer_cap_enforced'] is False
    assert all(result[k] is False for k in ('certified','numerically_qualified','formal_training_accepted'))
    assert result['final_eval_label_prediction_metric_reads']==0
    assert not list(work.glob('*.jsonl'))


def test_three_attempts_integrate_only_missing_suffix_with_whole_exact_result(ready):
    _,_,prepared,_,_,tracker=ready
    protected={p:p.read_bytes() for p in prepared.bound}
    before=len(tracker['calls'])
    for count in (5,7):
        attempt=reserve(ready)
        with pytest.raises(KeyboardInterrupt):run(attempt,progress=interrupt_after(count))
        assert tracker['maps'][-1].closed
        assert not (attempt['attempt']/'work/core-result.json').exists()
        assert seal(attempt)['continuable']
    attempt=reserve(ready)
    result=run(attempt)
    assert result['status']=='computed_pending_supervisory_closure',result
    assert result['inherited_success_count']==12 and result['new_committed_count']==12
    assert result['missing_completed_record_count']==result['new_failure_count']==0
    assert len(tracker['calls'])-before==24
    assert all(c['particles']==16 for c in tracker['calls'][before:])
    assert_exact_result(ready,attempt,result)
    closed=seal(attempt,worker_returncode=0,interrupted=False)
    assert closed['status']=='computed_closed' and not closed['continuable']
    assert len(lineage.inherit(prepared,attempt['directory'],attempt['root_sha256'])[0])==24
    with pytest.raises(ValueError,match='no automatic retry'):reserve(ready)
    assert all(p.read_bytes()==value for p,value in protected.items())


def test_interrupt_after_all_records_only_reruns_audit_not_forecasts(ready):
    before=len(ready[-1]['calls'])
    first=reserve(ready)
    with pytest.raises(KeyboardInterrupt):run(first,progress=interrupt_after(24))
    assert seal(first)['continuable']
    second=reserve(ready)
    result=run(second)
    assert result['status']=='computed_pending_supervisory_closure',result
    assert result['inherited_success_count']==24 and result['new_committed_count']==0
    assert len(ready[-1]['calls'])-before==24
    assert_exact_result(ready,second,result)


@pytest.mark.parametrize('exception',[ValueError,MemoryError])
def test_prediction_failure_keeps_complete_denominator_and_cannot_retry(ready,monkeypatch,exception):
    original,calls=execution.engine.rollout,[]
    def fail_once(*a,**k):
        calls.append(1)
        if len(calls)==1:raise exception('synthetic failed extension forecast')
        return original(*a,**k)
    monkeypatch.setattr(execution.engine,'rollout',fail_once)
    first=reserve(ready)
    result=run(first)
    assert result['status']=='failed' and result['new_failure_count']==1
    assert result['expected_run_count']==24
    assert result['new_committed_count']==(1 if exception is MemoryError else 24)
    assert result['missing_completed_record_count']==(23 if exception is MemoryError else 0)
    assert not (first['attempt']/'work/whole-science.json').exists()
    assert ready[-1]['maps'][-1].closed
    assert not seal(first,worker_returncode=1,interrupted=False)['continuable']


@pytest.mark.parametrize('fault',['target','time','block','runtime','reservation'])
def test_changed_actual_inputs_stop_before_any_new_forecast(ready,monkeypatch,fault):
    first=reserve(ready)
    tracker=ready[-1]
    before=len(tracker['calls'])
    window=tracker['windows']['validation'][0]
    if fault=='target':window.target_positions_m=window.target_positions_m+100
    if fault=='time':window.horizon_seconds=window.horizon_seconds+1
    if fault=='block':window.block_id='other'
    if fault=='runtime':
        real=execution.science.runtime_identity()
        monkeypatch.setattr(execution.science,'runtime_identity',lambda:{**real,'torch_intraop_threads':999})
    if fault=='reservation':first['start_sha256']='f'*64
    with pytest.raises(ValueError):run(first)
    assert len(tracker['calls'])==before and not (first['attempt']/'work').exists()


def test_new_worker_never_reopens_same_output(ready):
    first=reserve(ready)
    with pytest.raises(KeyboardInterrupt):run(first,progress=interrupt_after(1))
    with pytest.raises(FileExistsError,match='never reopens'):run(first)
    with pytest.raises(FileExistsError,match='unclosed'):reserve(ready)


def test_fresh_low_memory_between_records_preserves_progress_and_stops_remaining_work(ready,monkeypatch):
    first=reserve(ready)
    def low_after_two(event):
        if event['phase']=='run_committed' and event['new_committed']==2:
            monkeypatch.setattr(execution.native.physical_memory,'available_physical_bytes',lambda:0)
    result=run(first,progress=low_after_two)
    assert result['status']=='failed' and result['new_committed_count']==2
    assert result['missing_completed_record_count']==22 and result['new_failure_count']==0
    assert result['observer']['minimum_observed_available_bytes']==0
    assert any(e['error_type']=='MemoryError' for e in result['errors'])
    assert not (first['attempt']/'work/whole-science.json').exists()


def test_startup_time_consumes_original_cooperative_cap_before_preflight(ready,monkeypatch):
    first=reserve(ready)
    real_datetime=lineage.datetime
    class Later(real_datetime):
        @classmethod
        def now(cls,tz=None): return real_datetime.now(tz)+timedelta(seconds=1001)
    monkeypatch.setattr(lineage,'datetime',Later)
    before=len(ready[-1]['calls'])
    with pytest.raises(TimeoutError,match='no renewal'):run(first)
    assert len(ready[-1]['calls'])==before and not (first['attempt']/'work').exists()


def test_late_reference_mutation_retains_rows_but_forbids_whole_result(ready):
    first=reserve(ready)
    path=Path(ready[0]['reference_audit'])
    def change(event):
        if event['phase']=='run_committed' and event['new_committed']==24:
            path.write_bytes(path.read_bytes()+b' ')
    result=run(first,progress=change)
    assert result['status']=='failed' and result['new_committed_count']==24 and not result['new_failure_count']
    assert not (first['attempt']/'work/whole-science.json').exists()
    assert any('changed' in error['error_message'] for error in result['errors'])


def test_real_kernel_interrupted_extension_matches_uninterrupted_every_array(inputs,evidence,monkeypatch):
    args,tracker=inputs
    args=dict(args,configurations=['base','all-terrain','loo-road','loo-river','loo-worldcover','loo-surface'],
        seeds=[execution.engine.SEEDS[0]],particles=[4,8],steps=[5.,2.5],limit_origins=1,wall_seconds=1000,
        selection_policy='lexical_independent_blocks',map_backend='multicell')
    calls=[]
    def tracked(*a,**k):
        calls.append((k['particles'],k['max_step_seconds']))
        return real_rollout(*a,**k)
    monkeypatch.setattr(execution.engine,'rollout',tracked)
    source=fixture_reference(args,evidence)
    prepared,path,baseline=generate_new(source,args)
    locations={k:args[k] for k in lineage.LOCATION_ARGUMENTS}
    fixture_runtime(source,locations,monkeypatch)
    actual=(source,locations,prepared,path,baseline,tracker)
    assert len(calls)==24+12
    first=reserve(actual)
    with pytest.raises(KeyboardInterrupt):run(first,progress=interrupt_after(5))
    assert seal(first)['continuable']
    second=reserve(actual)
    result=run(second)
    assert result['status']=='computed_pending_supervisory_closure',result
    assert result['inherited_success_count']==5 and result['new_committed_count']==7
    assert len(calls)==24+12+12 and all(p==16 for p,_ in calls[36:])
    assert_exact_result(actual,second,result)
