"""Software-only lane partition, atomic recovery, concurrency and Job tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

from experiments.pirc17 import checkpoint_parallel as parallel
from experiments.pirc17.checkpoint_handoff import MainThreadPause, handoff, inflight
from experiments.pirc17.checkpoint_resume import INPUT, KINDS
from experiments.pirc17.checkpoint_state import NS, Progress, atomic_json
from experiments.pirc17.protocol_core import digest, envelope, read_json


def setup(directory, count=6, auxiliary=False):
    works = [dict(work_id=digest(['parallel-software',i]), kind='scientific_forecast',
        matrix='terrain', phase='terrain_forecasts', generated_forecasts=1,
        max_active_seconds=1) for i in range(count)]
    if auxiliary:
        works += [dict(work_id=digest(['aux',i]), kind=kind, matrix='terrain',
            phase='runtime_and_forecast_replay', generated_forecasts=1, max_active_seconds=1)
            for i,kind in enumerate(('runtime_warmup','runtime_warm'))]
    phases = [INPUT,'terrain_forecasts','runtime_and_forecast_replay']
    manifest = envelope(dict(execution_sha256='e'*64, imported_entries={works[0]['work_id']: {}}))
    atomic_json(directory/'imports.json', manifest)
    settings = dict(workloads=works, phase_order=phases, charged_ns_by_phase=dict.fromkeys(phases,0),
        phase_caps_ns=dict.fromkeys(phases,0), total_cap_ns=0, generated=1, generation_limit=len(works),
        execution_sha256='e'*64, import_manifest=str(directory/'imports.json'), import_manifest_sha256=manifest['sha256'])
    settings['settings_id'] = digest(settings)
    atomic_json(directory/'settings.json', settings)
    return Progress(directory, settings=settings, imported_ids=[works[0]['work_id']])


def reserve(p, lane, work, reply):
    p.reserve_lane(lane, phase=work['phase'], maximum_ns=NS, work_id=work['work_id'],
                   reply=str(reply), ignore_time_budgets=True)


def test_static_split_excludes_imports_successes_and_failures(tmp_path):
    p = setup(tmp_path)
    missing = p.missing(kinds=KINDS)
    reserve(p,'0',missing[0],tmp_path/'0.json')
    p.settle_lane('0', elapsed_ns=NS//2, result={'status':'success'})
    reserve(p,'1',missing[1],tmp_path/'1.json')
    p.settle_lane('1', elapsed_ns=NS//2, failure='old interruption')
    plan = parallel.assignment(p)
    assert plan['lanes']['0'] == [w['work_id'] for w in missing[2:4]]
    assert plan['lanes']['1'] == [missing[4]['work_id']]
    assert not set(plan['lanes']['0']) & set(plan['lanes']['1'])
    assert plan == parallel.assignment(p)


def test_two_pending_results_settle_without_lost_updates_and_legacy_recovery(tmp_path):
    p = setup(tmp_path)
    a,b = p.missing(kinds=KINDS)[:2]
    reserve(p,'0',a,tmp_path/'0.json')
    reserve(p,'1',b,tmp_path/'1.json')
    assert len(read_json(p.path)['parallel_pending']) == 2
    with pytest.raises(RuntimeError):
        p.settle(result={'status':'success'})
    atomic_json(tmp_path/'1.json', dict(work_id=b['work_id'], result={'status':'success'}))
    restored = Progress(tmp_path, settings=p.settings, imported_ids=p.imported_ids)
    restored.recover_pending()
    assert restored.value['pending'] is None and not restored.value['parallel_pending']
    assert b['work_id'] in restored.value['completed'] and a['work_id'] in restored.value['failures']
    assert len(restored.missing(kinds=KINDS)) == 3
    assert restored.value['charged_ns_by_phase'][a['phase']] == 2*NS


def test_duplicate_lane_work_and_generation_scope_rejected(tmp_path):
    p = setup(tmp_path)
    a,b = p.missing(kinds=KINDS)[:2]
    reserve(p,'0',a,tmp_path/'0.json')
    with pytest.raises(RuntimeError):
        reserve(p,'0',b,tmp_path/'1.json')
    with pytest.raises(ValueError):
        reserve(p,'1',a,tmp_path/'1.json')
    p.settings['generation_limit'] = p.value['generated']
    with pytest.raises(TimeoutError):
        reserve(p,'1',b,tmp_path/'1.json')


def test_clone_shares_fitted_readonly_state_but_owns_all_mutable_state():
    shared = SimpleNamespace(models={'never':'refit'}, encoders={}, cases={}, attempted={'imported'},
        maps=SimpleNamespace(fresh_provider=lambda:object()), drivers={'old':object()}, driver_origin='old',
        runtime_hardware=object(),runtime_maps=object(),runtime_state=object(),runtime_session='old')
    cloned = parallel.clone_consumer(shared)
    assert cloned.models is shared.models and cloned.cases is shared.cases and cloned.encoders is shared.encoders
    assert cloned.maps is not shared.maps and cloned.attempted is not shared.attempted
    assert cloned.attempted == shared.attempted and cloned.drivers == {} and cloned.drivers is not shared.drivers
    assert cloned.driver_origin is None and cloned.runtime_state is None and cloned.runtime_session != 'old'


def test_lane_loops_overlap_and_keep_one_dedicated_thread_per_provider(tmp_path):
    stop = threading.Event()
    barrier = threading.Barrier(2)
    ids = {}
    closed = []
    def consumer(lane):
        def execute(work, *, output_directory):
            ids[lane] = threading.get_native_id()
            barrier.wait(timeout=3)
            Path(output_directory).mkdir()
            return {'status':'success'}
        return SimpleNamespace(execute=execute, work={lane:{'work_id':lane}}, close_all=lambda:closed.append(lane))
    for lane in ('0','1'):
        request = tmp_path/'requests'/lane
        request.mkdir(parents=True)
        atomic_json(request/'000001.json', dict(work_id=lane, output_directory=str(tmp_path/lane)))
    threads = [threading.Thread(target=parallel.lane_loop,args=(consumer(lane),tmp_path,lane,stop)) for lane in ('0','1')]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic()+5
        while not all((tmp_path/lane/'reply.json').exists() for lane in ('0','1')):
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert ids['0'] != ids['1']
        assert all(read_json(tmp_path/lane/'reply.json')['result']['status'] == 'success' for lane in ('0','1'))
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
    assert sorted(closed) == ['0','1']


STUB = r'''
import os,sys,time,threading
from pathlib import Path
from experiments.pirc17.checkpoint_state import atomic_json
from experiments.pirc17.protocol_core import read_json
root,session=map(Path,sys.argv[1:]);ends={};guard=threading.Lock()
def lane_loop(lane):
    sequence=1
    while True:
        path=session/'requests'/lane/f'{sequence:06d}.json'
        if path.is_file():
            request=read_json(path);wid=request['work_id'];sequence+=1
            work=next(w for w in read_json(root/'settings.json')['workloads'] if w['work_id']==wid)
            if work['kind'].startswith('runtime_'):
                assert len(ends)>=5, 'runtime before science completion'
            began=time.time();time.sleep(.2)
            output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=True)
            atomic_json(output/'reply.json',dict(work_id=wid,result=dict(status='success',began=began,ended=time.time(),lane=lane)))
            if work['kind']=='scientific_forecast':
                with guard: ends[wid]=time.time()
        time.sleep(.01)
for lane in ('0','1'):
    threading.Thread(target=lane_loop,args=(lane,),daemon=True).start()
atomic_json(session/'ready.json',dict(models=26,pid=os.getpid()))
while True: time.sleep(.1)
'''


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows Job lifecycle')
def test_actual_parallel_controller_no_duplicates_and_auxiliary_serial_after_science(tmp_path):
    p = setup(tmp_path, auxiliary=True)
    command = lambda directory,session:[sys.executable,'-u','-c',STUB,str(directory),str(session)]
    result = parallel.run(tmp_path, report_seconds=.1, _worker_command=command)
    assert result['completed_scientific'] == 6 and result['failures'] == 0 and result['generated'] == 8
    assert parallel.run(tmp_path, _worker_command=lambda *a:pytest.fail('must not load again')) == result
    session = next((tmp_path/'sessions').iterdir())
    closed = read_json(session/'closed.json')
    assert closed['process_tree_closed'] and closed['accounting']['active_processes'] == 0
    progress = read_json(tmp_path/'progress.json')
    assert len(progress['completed']) == 7 and progress['pending'] is None
    scientific = [r for wid,r in progress['completed'].items() if p.work[wid]['kind']=='scientific_forecast']
    assert any(a['began']<b['ended'] and b['began']<a['ended'] for a in scientific for b in scientific if a['lane']!=b['lane'])
    auxiliary = [r for wid,r in progress['completed'].items() if p.work[wid]['kind']!='scientific_forecast']
    assert all(r['lane']=='0' and r['began']>=max(v['ended'] for v in scientific) for r in auxiliary)
    assert not (tmp_path/'writer.json').exists()


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows thread identity')
def test_main_thread_pause_actual_process_always_resumes():
    child = subprocess.Popen([sys.executable,'-I','-S','-c','import time; time.sleep(30)'],
        creationflags=subprocess.CREATE_NO_WINDOW)
    pause = None
    try:
        process = psutil.Process(child.pid)
        # Venv exe may be a launcher: use the actual interpreter process.
        time.sleep(.2)
        descendants = process.children(recursive=True)
        target = descendants[-1] if descendants else process
        pause = MainThreadPause(target)
        pause.suspend()
        assert pause.suspended
        pause.resume()
        assert not pause.suspended and target.is_running()
    finally:
        if pause:
            pause.resume();pause.close()
        for process in psutil.Process(child.pid).children(recursive=True):
            process.kill()
        child.kill();child.wait(timeout=10)


def test_unsafe_handoff_point_is_rejected_without_writing_progress(tmp_path):
    p = setup(tmp_path)
    session = tmp_path/'sessions'/'old'
    (session/'requests').mkdir(parents=True)
    owner = dict(pid=1,created=1,token='owned')
    atomic_json(tmp_path/'writer.json',owner)
    original = p.path.read_bytes()
    with pytest.raises(RuntimeError):
        inflight(tmp_path,session,owner)
    assert original == p.path.read_bytes()


LEGACY = r'''
import sys
from pathlib import Path
from experiments.pirc17.checkpoint_resume import run
worker=r"""
import sys,time
from pathlib import Path
from experiments.pirc17.checkpoint_state import atomic_json
from experiments.pirc17.protocol_core import read_json
session=Path(sys.argv[1]);atomic_json(session/'ready.json',{'models':26})
sequence=1
while True:
    path=session/'requests'/f'{sequence:06d}.json'
    if path.exists():
        item=read_json(path);sequence+=1
        time.sleep(2)
        out=Path(item['output_directory']);out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'reply.json',{'work_id':item['work_id'],'result':{'status':'success'}})
    time.sleep(.01)
"""
run(Path(sys.argv[-1]), ignore_time_budgets=True,
    _worker_command=lambda root,session:[sys.executable,'-u','-c',worker,str(session)])
'''


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows owned handoff')
@pytest.mark.parametrize('timeout', [10, .05])
def test_actual_handoff_retains_finished_item_or_resumes_legacy_on_timeout(tmp_path, timeout):
    setup(tmp_path, count=3)
    child = subprocess.Popen([sys.executable,'-u','-c',LEGACY,
        'experiments.pirc17.checkpoint_resume','run','--directory',str(tmp_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    owner, session = None, None
    try:
        deadline = time.monotonic()+10
        while True:
            assert time.monotonic()<deadline
            if (tmp_path/'writer.json').is_file() and (tmp_path/'progress.json').is_file():
                owner=read_json(tmp_path/'writer.json')
                progress=read_json(tmp_path/'progress.json')
                sessions=list((tmp_path/'sessions').iterdir()) if (tmp_path/'sessions').exists() else []
                if sessions and progress['pending'] and progress['pending']['work_id']:
                    session=sessions[0]
                    if (session/'requests'/'000001.json').is_file():
                        break
            time.sleep(.01)
        if timeout < 1:
            with pytest.raises(TimeoutError):
                handoff(tmp_path,expected_pid=owner['pid'],session=session,timeout_seconds=timeout)
            child.wait(timeout=10)  # Proves the suspended original main thread was resumed.
            assert len(read_json(tmp_path/'progress.json')['completed']) == 2
            assert not (session/'parallel-handoff-request.json').exists()
        else:
            receipt=handoff(tmp_path,expected_pid=owner['pid'],session=session,timeout_seconds=timeout)
            child.wait(timeout=10)
            assert receipt['old_process_tree_closed'] and receipt['status']['completed_scientific']==2
            progress=read_json(tmp_path/'progress.json')
            assert len(progress['attempted']) == 1 and not progress['failures'] and progress['pending'] is None
            assert progress['charged_ns_by_phase']['terrain_forecasts'] == NS
            assert len(parallel.assignment(Progress(tmp_path, settings=read_json(tmp_path/'settings.json'),
                imported_ids=[read_json(tmp_path/'settings.json')['workloads'][0]['work_id']]))['lanes']['0']) == 1
    finally:
        if child.poll() is None:
            if owner:
                try:
                    psutil.Process(owner['pid']).kill()  # fixture owner only; Job closes descendants.
                except psutil.NoSuchProcess:
                    pass
            child.kill();child.wait(timeout=10)
