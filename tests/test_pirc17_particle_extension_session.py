"""Whole consumers on synthetic science; actual native jobs only for software children."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from experiments.pirc17 import particle_extension_session as session
from tests.test_pirc17_particle_extension_execution import (
    candidate_bytes,evidence,inputs,reference,case,ready,reserve,run,seal,interrupt_after)

lineage=session.lineage


def test_whole_consumer_reproduces_two_attempts_without_new_forecasts(ready):
    first=reserve(ready)
    with pytest.raises(KeyboardInterrupt):run(first,progress=interrupt_after(4))
    seal(first)
    second=reserve(ready)
    result=run(second)
    assert result['status']=='computed_pending_supervisory_closure',result
    seal(second,worker_returncode=0,interrupted=False)
    before=lineage.storage.inventory(second['directory'])
    calls=len(ready[-1]['calls'])
    report=session.inspect_closed(second['directory'],second['root_sha256'])
    assert report['status']=='verified_closed_whole_workload' and report['attempt_count']==2
    assert report['expected_run_count']==report['new_committed_count']==24
    assert report['reference_runs_covered']==report['exact_reference_prefix_checks']==48
    assert report['reference_numerical']['out_of_tolerance']>0
    assert not report['resource_certified'] and not report['numerically_qualified']
    assert not report['certified'] and not report['formal_training_accepted']
    assert report['final_eval_label_prediction_metric_reads']==0
    assert before==lineage.storage.inventory(second['directory']) and len(ready[-1]['calls'])==calls


def test_whole_consumer_requires_terminal_success_not_saved_progress(ready):
    first=reserve(ready)
    with pytest.raises(FileExistsError,match='unclosed'):session.inspect_closed(first['directory'],first['root_sha256'])
    with pytest.raises(KeyboardInterrupt):run(first,progress=interrupt_after(2))
    seal(first)
    with pytest.raises(ValueError,match='complete owned closure'):
        session.inspect_closed(first['directory'],first['root_sha256'])


@pytest.mark.parametrize('fault',['science','inherited_root','memory','resources','map_asset'])
def test_independent_replay_rejects_internally_rehashed_but_false_completed_results(ready,fault):
    first=reserve(ready)
    result=run(first)
    assert result['status']=='computed_pending_supervisory_closure',result
    work=first['attempt']/'work'
    if fault=='science':
        path=work/'whole-science.json'
        value=json.loads(path.read_text())
        value['reference_numerical']['out_of_tolerance']+=1
        path.write_bytes(lineage.journal._encode(value))
        result['whole_science_sha256']=lineage.native._hash(path)
    if fault=='inherited_root':
        path=work/'inherited.json'
        value=json.loads(path.read_text())
        value['root_sha256']='f'*64
        path.write_bytes(lineage.journal._encode(value))
    if fault=='memory':result['observer']['minimum_observed_available_bytes']=0
    if fault=='resources':result['resources']['cooperative_remaining_seconds']+=1
    if fault=='map_asset':
        path=work/'fake-map.bin'
        path.write_bytes(b'unchanged synthetic asset')
        result['new_attempt_maps']['receipt_sha256']={str(path):'f'*64}
    (work/'core-result.json').write_bytes(lineage.journal._encode(result))
    closed=seal(first,worker_returncode=0,interrupted=False)
    assert closed['status']=='computed_closed'  # Storage closure is not scientific proof.
    with pytest.raises(ValueError):session.inspect_closed(first['directory'],first['root_sha256'])


def test_stop_is_exact_attempt_request_not_termination_claim(ready):
    first=reserve(ready)
    assert not session._stop_requested(first)
    session.request_stop(first['directory'],first['root_sha256'])
    assert session._stop_requested(first)
    assert not (first['attempt']/'closure.json').exists()
    with pytest.raises(FileExistsError):session.request_stop(first['directory'],first['root_sha256'])
    seal(first)
    second=reserve(ready)
    assert not session._stop_requested(second)
    path=second['attempt']/'stop.json'
    lineage.journal._publish_json(path,{'schema_version':session.VERSION+'-stop','start_sha256':first['start_sha256']})
    with pytest.raises(ValueError,match='exact extension attempt'):session._stop_requested(second)


def test_late_map_asset_mutation_outside_result_tree_fails_whole_inspection(ready,monkeypatch):
    first=reserve(ready)
    result=run(first)
    assert result['status']=='computed_pending_supervisory_closure',result
    asset=Path(ready[0]['plan']).parent/'external-software-map.bin'
    asset.write_bytes(b'synthetic map bytes')
    result['new_attempt_maps']['receipt_sha256']={str(asset):lineage.native._hash(asset)}
    (first['attempt']/'work/core-result.json').write_bytes(lineage.journal._encode(result))
    seal(first,worker_returncode=0,interrupted=False)
    original=session.execution.science.whole_audit
    def mutate(*a,**kw):
        checked=original(*a,**kw)
        asset.write_bytes(b'changed after initial verification')
        return checked
    monkeypatch.setattr(session.execution.science,'whole_audit',mutate)
    with pytest.raises(ValueError,match='map receipt or asset changed during whole replay'):
        session.inspect_closed(first['directory'],first['root_sha256'])


@pytest.mark.skipif(os.name!='nt',reason='native Windows job membership contract')
def test_direct_worker_entry_cannot_bypass_live_owned_job(ready):
    first=reserve(ready)
    calls=len(ready[-1]['calls'])
    with pytest.raises(PermissionError,match='live reserved process job'):
        session.worker(**{k:first[k] for k in ('directory','root_sha256','index','start_sha256')})
    assert len(ready[-1]['calls'])==calls and not (first['attempt']/'work').exists()


@pytest.mark.skipif(os.name!='nt',reason='native Windows job closure contract')
def test_actual_job_reaps_descendants_on_timeout_and_explicit_stop():
    ok=session.run_owned([sys.executable,'-c','raise SystemExit(0)'],timeout=10)
    assert ok['worker_returncode']==0 and ok['process_tree_closed']
    assert ok['accounting']['active_processes']==0 and ok['accounting']['total_processes']>=2
    command="import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],creationflags=subprocess.CREATE_NO_WINDOW)"
    timed=session.run_owned([sys.executable,'-c',command],timeout=2)
    assert timed['timed_out'] and timed['process_tree_closed'] and timed['accounting']['active_processes']==0
    assert timed['accounting']['total_processes']>=3
    begun=time.perf_counter()
    stopped=session.run_owned([sys.executable,'-c','import time; time.sleep(30)'],timeout=10,
        stop_requested=lambda:time.perf_counter()-begun>1)
    assert stopped['interrupted'] and not stopped['timed_out'] and stopped['process_tree_closed']
    assert stopped['accounting']['active_processes']==0


@pytest.mark.skipif(os.name!='nt',reason='native Windows supervised CLI contract')
def test_actual_cold_cli_bad_reference_closes_job_without_any_forecast(ready):
    source,locations,*_=ready
    source=deepcopy(source)
    plan=json.loads(source['plan'].read_text())
    plan.update(reference_audit_sha256='f'*64,wall_seconds=30,outer_wall_seconds=45)
    source['plan'].write_bytes(lineage.journal._encode(plan))
    source['plan_sha256']=lineage.native._hash(source['plan'])
    command=[sys.executable,'-u','-m','experiments.pirc17.particle_extension_session','run']
    for k,v in {**source,**locations}.items():command.extend(['--'+k.replace('_','-'),str(v)])
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=90,
        creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==1,(result.stdout,result.stderr)
    assert 'closed evidence hash mismatch' in result.stderr,result.stderr
    directory=lineage.science.directory_for(source,plan)
    sha=lineage.native._hash(directory/'root.json')
    _,closed,_,_=lineage.history(directory,sha)
    assert len(closed)==1 and closed[0]['closure']['status']=='failed'
    closure=closed[0]['closure']
    assert closure['process_tree_closed'] and closure['new_committed_count']==closure['new_failure_count']==0
    assert not closure['continuable'] and not closure['timed_out']
    process=json.loads((directory/'attempts/000000/work-process.json').read_text())
    assert process['accounting']['active_processes']==0 and process['accounting']['total_processes']>=2
    assert process['worker_returncode']!=0 and not (directory/'attempts/000000/work').exists()
    with pytest.raises(ValueError,match='no automatic retry'):
        lineage.reserve(source=source,locations=locations)
