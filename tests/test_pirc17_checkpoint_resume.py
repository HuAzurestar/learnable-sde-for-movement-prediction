"""Restart/interruption tests: synthetic data, plus actual Windows Job closure."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pirc17 import checkpoint_resume as runner
from experiments.pirc17.checkpoint_state import NS, Progress, atomic_json, single_writer
from experiments.pirc17.protocol_core import digest, envelope, file_hash, read_json


def settings():
    work = [dict(work_id=digest(['work',i]), kind='scientific_forecast', phase='method_forecasts', matrix='NEX326-methods',
                 max_active_seconds=1, generated_forecasts=1) for i in range(3)]
    value = dict(workloads=work, phase_order=[runner.INPUT,'method_forecasts'],
        charged_ns_by_phase={runner.INPUT:2*NS, 'method_forecasts':3*NS},
        phase_caps_ns={runner.INPUT:20*NS, 'method_forecasts':10*NS}, total_cap_ns=30*NS,
        generated=1, generation_limit=4)
    value['settings_id'] = digest(value)
    return value


def progress(tmp_path):
    s = settings()
    return Progress(tmp_path, settings=s, imported_ids=[s['workloads'][0]['work_id']])


def test_completed_items_are_skipped_without_opening_their_outputs(tmp_path):
    p = progress(tmp_path)
    first = p.missing(kinds=runner.KINDS)[0]
    p.reserve(phase=first['phase'], maximum_ns=NS, work_id=first['work_id'])
    p.settle(elapsed_ns=NS//4, result=dict(status='success', artifact_path='never/open/this/old/file'))
    restored = Progress(tmp_path, settings=p.settings, imported_ids=p.imported_ids)
    assert len(restored.missing(kinds=runner.KINDS)) == 1
    assert restored.value['generated'] == 2
    assert restored.value['charged_ns_by_phase']['method_forecasts'] == 3*NS+NS//4


def test_resource_shortage_is_advisory_not_an_admission_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1))
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda root: SimpleNamespace(free=1))
    runner.resource_observation(tmp_path)
    report = __import__('json').loads(capsys.readouterr().out)
    assert report['advisory_only'] and {w['resource'] for w in report['warnings']} == {'ram','disk'}


def test_waiting_for_worker_never_polls_ram_or_aborts_for_a_machine_threshold(tmp_path, monkeypatch):
    def forbidden():
        pytest.fail('machine-wide RAM observation inside rollout wait')
    monkeypatch.setattr(runner.psutil, 'virtual_memory', forbidden)
    reply = tmp_path/'reply.json'
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: atomic_json(reply, {'work_id':'saved'}))
    result = runner.wait_reply(reply, child=SimpleNamespace(poll=lambda: None),
                               deadline_ns=None, stop_requested=lambda: False)
    assert result == {'work_id':'saved'}


def test_failed_science_is_not_runnable_or_covered_by_a_fake_eta(tmp_path):
    p = progress(tmp_path)
    work = p.missing(kinds=runner.KINDS)[0]
    p.reserve(phase=work['phase'], maximum_ns=NS, work_id=work['work_id'])
    p.settle(elapsed_ns=NS//4, failure='interrupted')
    report = runner.status(p)
    assert report['completed_scientific'] == 1 and report['remaining_scientific'] == 2
    assert report['failed_scientific'] == 1 and report['runnable_scientific'] == 1


def test_interrupted_unpublished_attempt_not_success_or_silent_retry(tmp_path):
    p = progress(tmp_path)
    work = p.missing(kinds=runner.KINDS)[0]
    p.reserve(phase=work['phase'], maximum_ns=NS, work_id=work['work_id'], reply=str(tmp_path/'absent.json'))
    restored = Progress(tmp_path, settings=p.settings, imported_ids=p.imported_ids)
    restored.recover_pending()
    assert not restored.value['completed']
    assert work['work_id'] in restored.value['failures']
    assert restored.value['charged_ns_by_phase']['method_forecasts'] == 4*NS
    assert work not in restored.missing(kinds=runner.KINDS)


def test_atomic_reply_recovers_only_pending_item_without_forecasting(tmp_path):
    p = progress(tmp_path)
    work = p.missing(kinds=runner.KINDS)[0]
    reply = tmp_path/'reply.json'
    p.reserve(phase=work['phase'], maximum_ns=NS, work_id=work['work_id'], reply=str(reply))
    atomic_json(reply, dict(work_id=work['work_id'], result=dict(status='success', artifact_path='saved.json')))
    p.recover_pending()
    assert work['work_id'] in p.value['completed']
    assert p.value['charged_ns_by_phase']['method_forecasts'] == 4*NS  # Unknown elapsed is NOT refunded.


def test_atomic_snapshot_failure_keeps_previous_complete_checkpoint(tmp_path, monkeypatch):
    path = tmp_path/'progress.json'
    atomic_json(path, {'completed':['a']})
    def interrupted(*args):
        raise OSError('simulated replace failure')
    monkeypatch.setattr(os, 'replace', interrupted)
    with pytest.raises(OSError):
        atomic_json(path, {'completed':['a','b']})
    assert read_json(path) == {'completed':['a']}
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('persistent', [False, True])
def test_windows_read_sharing_retry_is_bounded_and_preserves_checkpoint(tmp_path, monkeypatch, persistent):
    from experiments.pirc17 import checkpoint_state as state
    path = tmp_path/'progress.json'
    atomic_json(path, {'completed':['a']})
    replace, calls, delays = os.replace, [], []
    def busy_reader(source, target):
        calls.append(1)
        if persistent or len(calls) == 1:
            error = PermissionError('Windows reader denies replacement')
            error.winerror = 32
            raise error
        return replace(source, target)
    monkeypatch.setattr(os, 'replace', busy_reader)
    monkeypatch.setattr(state.time, 'sleep', delays.append)
    if persistent:
        with pytest.raises(PermissionError):
            atomic_json(path, {'completed':['a','b']})
        assert len(calls) == 21 and len(delays) == 20
        assert read_json(path) == {'completed':['a']}
    else:
        atomic_json(path, {'completed':['a','b']})
        assert len(calls) == 2 and delays == [.01]
        assert read_json(path) == {'completed':['a','b']}
    assert not list(tmp_path.glob('*.tmp'))


def test_live_duplicate_rejected_without_wait_and_pid_reuse_reclaimed(tmp_path):
    with single_writer(tmp_path) as owner:
        with pytest.raises(RuntimeError, match='live runner'):
            with single_writer(tmp_path):
                pytest.fail('second runner entered')
    # SAME still-live PID, DIFFERENT creation time: don't kill it or block.
    atomic_json(tmp_path/'writer.json', dict(owner, created=owner['created']-100))
    with single_writer(tmp_path):
        pass
    assert not (tmp_path/'writer.json').exists()


def test_budgets_and_generation_are_cumulative_and_never_borrowed(tmp_path):
    p = progress(tmp_path)
    work = p.missing(kinds=runner.KINDS)[0]
    with pytest.raises(TimeoutError):
        p.reserve(phase=work['phase'], maximum_ns=8*NS, work_id=work['work_id'])
    p.reserve(phase=work['phase'], maximum_ns=NS, work_id=work['work_id'])
    p.settle(elapsed_ns=2*NS, failure='timeout overrun')
    assert p.value['charged_ns_by_phase']['method_forecasts'] == 5*NS
    assert p.value['generated'] == 2
    with pytest.raises(ValueError, match='already'):
        p.reserve(phase=work['phase'], maximum_ns=NS, work_id=work['work_id'])


def test_explicit_time_override_keeps_charges_and_generation_scope(tmp_path):
    p = progress(tmp_path)
    p.settings['phase_caps_ns']['method_forecasts'] = NS
    p.settings['total_cap_ns'] = NS
    work = p.missing(kinds=runner.KINDS)[0]
    p.reserve(phase=work['phase'], maximum_ns=NS, work_id=work['work_id'], ignore_time_budgets=True)
    p.settle(elapsed_ns=3*NS, result=dict(status='success'))
    assert p.value['charged_ns_by_phase']['method_forecasts'] == 6*NS
    assert p.value['generated'] == 2
    assert p.remaining_ns(work['phase']) == 0
    p.settings['generation_limit'] = 2
    other = p.missing(kinds=runner.KINDS)[0]
    with pytest.raises(TimeoutError, match='generation'):
        p.reserve(phase=other['phase'], maximum_ns=NS, work_id=other['work_id'], ignore_time_budgets=True)


def test_exhausted_budget_does_not_pay_for_another_worker_startup(tmp_path):
    s = settings()
    s['phase_caps_ns']['method_forecasts'] = s['charged_ns_by_phase']['method_forecasts']
    s['execution_sha256'] = 'e'*64
    manifest = tmp_path/'imports.json'
    record = envelope(dict(execution_sha256=s['execution_sha256'], imported_entries={s['workloads'][0]['work_id']:{}}))
    atomic_json(manifest, record)
    s['import_manifest'], s['import_manifest_sha256'] = str(manifest), record['sha256']
    s['settings_id'] = digest({k:v for k,v in s.items() if k != 'settings_id'})
    atomic_json(tmp_path/'settings.json', s)
    result = runner.run(tmp_path, _worker_command=lambda *args: pytest.fail('worker started without runnable budget'))
    assert result['charged_seconds'][runner.INPUT] == 2 and result['generated'] == 1
    assert not (tmp_path/'sessions').exists()


def test_online_metadata_transforms_match_original_without_snapshot_scan(tmp_path, monkeypatch):
    from experiments.pirc17.checkpoint_inputs import OnlineTransforms
    from experiments.pirc17.features import CanonicalEncoder
    from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
    from tests.test_pirc21_adapter import _write_snapshot, _rows
    root = _write_snapshot(tmp_path)
    selection = FeatureSelection(variant_ids=('road.distance_log1p','road.direction','worldcover.grouped'),
        composition_ids=('road.log_distance_x_direction','worldcover.grouped_x_road_distance'))
    original = FeatureSnapshotAdapter(root, selection)
    expected = CanonicalEncoder(original).encode(_rows('train', [0,2,4,6])).model_matrix(4)
    def forbidden(*args):
        pytest.fail('restart scanned all snapshot data')
    monkeypatch.setattr(FeatureSnapshotAdapter, '_validate_snapshot', forbidden)
    restored = OnlineTransforms(root, selection, manifest_sha256=file_hash(root/'manifest.json'),
        spec_sha256=file_hash(root/'feature_spec.json'))
    actual = CanonicalEncoder(restored).encode(_rows('train', [0,2,4,6])).model_matrix(4)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(RuntimeError, match='cannot load'):
        restored._load('train')


STUB = r'''
import json,os,sys,time
from pathlib import Path
from experiments.pirc17.checkpoint_state import atomic_json
directory,session,behavior=map(str,sys.argv[1:])
session=Path(session)
atomic_json(session/'ready.json',{'models':26})
sequence=1
held=[]
while True:
    request=session/'requests'/f'{sequence:06d}.json'
    if request.exists():
        item=json.loads(request.read_text())
        wid=item['work_id']
        sequence+=1
        if behavior=='immutable':
            held.append(request.open('rb'))  # Denies Windows replacement until process exit.
        output=Path(item['output_directory']);output.mkdir(parents=True,exist_ok=True)
        if behavior=='hang':
            (output/'partial.npz').write_bytes(b'partial')
            time.sleep(20)
        if behavior=='crash':
            sys.exit(9)
        if behavior=='override':
            time.sleep(1.1)  # Longer than the ORIGINAL one-second item cap.
        if behavior=='stop':
            owner=json.loads((Path(directory)/'writer.json').read_text())
            atomic_json(Path(directory)/'stop.json',{'token':owner['token']})
            time.sleep(20)
        atomic_json(output/'reply.json',{'work_id':wid,'result':{'status':'success','artifact_path':'software-only'}})
    time.sleep(.01)
'''


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows Job lifecycle')
@pytest.mark.parametrize('behavior', ['complete','hang','crash','stop','override','immutable','lowresources'])
def test_actual_process_timeout_closure_and_restart_no_duplicate(tmp_path, behavior, monkeypatch):
    if behavior == 'lowresources':
        monkeypatch.setattr(runner.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1))
        monkeypatch.setattr(runner.shutil, 'disk_usage', lambda root: SimpleNamespace(free=1))
    s = settings()
    s['execution_sha256'] = 'e'*64
    manifest = tmp_path/'imports.json'
    imported = {s['workloads'][0]['work_id']: {'no_actual_arrays':'must never open'}}
    manifest_record = envelope(dict(execution_sha256=s['execution_sha256'], imported_entries=imported))
    atomic_json(manifest, manifest_record)
    s['import_manifest'] = str(manifest)
    s['import_manifest_sha256'] = manifest_record['sha256']
    if behavior == 'override':
        s['phase_caps_ns'] = {runner.INPUT:0, 'method_forecasts':0}
        s['total_cap_ns'] = 0
    s['settings_id'] = digest({k:v for k,v in s.items() if k != 'settings_id'})
    atomic_json(tmp_path/'settings.json', s)
    command = lambda directory,session: [sys.executable,'-u','-c',STUB,str(directory),str(session),behavior]
    if behavior in {'complete','override','immutable','lowresources'}:
        kwargs = dict(ignore_time_budgets=behavior == 'override', report_seconds=.05 if behavior == 'override' else None)
        result = runner.run(tmp_path, maximum_items=2 if behavior == 'immutable' else 1, _worker_command=command, **kwargs)
        assert result['completed_scientific'] == (3 if behavior == 'immutable' else 2)
        result = runner.run(tmp_path, maximum_items=1, _worker_command=command, **kwargs)
        assert result['completed_scientific'] == 3 and result['generated'] == 3
    else:
        with pytest.raises((TimeoutError,RuntimeError,KeyboardInterrupt)):
            runner.run(tmp_path, maximum_items=1, _worker_command=command)
        p = Progress(tmp_path, settings=s, imported_ids=imported)
        assert len(p.value['attempted']) == 1 and not p.value['completed'] and p.value['pending'] is None
        assert len(p.missing(kinds=runner.KINDS)) == 1
    sessions = list((tmp_path/'sessions').iterdir())
    for session in sessions:
        assert not (session/'request.json').exists()
        if behavior == 'immutable':
            assert sorted(p.name for p in (session/'requests').iterdir()) == ['000001.json','000002.json']
        closed = read_json(session/'closed.json')
        assert closed['process_tree_closed'] and closed['accounting']['active_processes'] == 0
        if behavior == 'override':
            assert read_json(session/'run.json')['ignore_time_budgets']
            report = read_json(session/'progress-report.json')
            assert report['ignore_time_budgets'] and report['by_matrix']['NEX326-methods']['recent_mean_seconds'] > 1
    assert not (tmp_path/'writer.json').exists()
