"""Synthetic journal/closure tests only: no production forecast or OS claims."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
import threading

import numpy as np
import pytest

from experiments.pirc17 import particle_extension_lineage as lineage
from tests.test_pirc17_particle_extension import (
    candidate_bytes, evidence, inputs, reference, generate_new)


@pytest.fixture
def case(reference):
    source, args, tracker = reference
    prepared, path, rows = generate_new(source, args)
    locations = {key: args[key] for key in lineage.LOCATION_ARGUMENTS}
    return source, locations, prepared, path, rows, tracker


def reserve(case):
    return lineage.reserve(source=case[0], locations=case[1])


def append(case, reservation, count, *, corrupt=None):
    _, _, prepared, path, baseline, _ = case
    root, _, _, _ = lineage.history(reservation['directory'], reservation['root_sha256'], current_index=reservation['index'])
    rows, folders, ancestry, start, digest = lineage.inherit(prepared, reservation['directory'],
        reservation['root_sha256'], current_index=reservation['index'])
    work = reservation['attempt']/'work'
    work.mkdir()
    writer = lineage.journal.Writer(work/'journal', workloads=[dict(zip(lineage.KEYS,k)) for k in prepared.keys],
        inherited_count=len(rows), ancestry=ancestry,
        contract=lineage.journal_contract(root, reservation['root_sha256'], reservation['index'], digest, prepared))
    for original in baseline[len(rows):len(rows)+count]:
        arrays = lineage.science.load_particle_evidence(original,path.parent)
        row = deepcopy(original)
        row.pop('particle_artifact')
        if corrupt:
            corrupt(row)
        writer.append(row,arrays if row['status']=='success' else None)
    return writer


def publish_process(reservation, **changes):
    # Deliberately simulated records. Native job ownership must be tested in
    # the actual supervising layer before any empirical extension launch.
    process = dict(worker_pid=os.getpid(),worker_returncode=130,process_tree_closed=True,
        accounting={'active_processes':0,'total_processes':1},timed_out=False,interrupted=True,
        supervisor_error=None,elapsed_seconds=0.,job_name=reservation['start']['job_name'])
    process.update(changes)
    lineage.journal._publish_json(reservation['attempt']/'work-process.json',process)


def seal(reservation, **changes):
    publish_process(reservation,**changes)
    return lineage.seal(reservation)


def test_three_attempts_reuse_all_ordered_new_rows_without_reference_reexecution(case):
    source, _, prepared, path, baseline, tracker = case
    before = len(tracker['calls'])
    saved_reference = {p:p.read_bytes() for p in prepared.bound}
    closures = []
    for count in (5,7,12):
        current = reserve(case)
        assert current['start']['inherited_count'] == sum(c['new_committed_count'] for c in closures)
        assert current['start']['prior_elapsed_seconds'] == math.fsum(c['elapsed_seconds'] for c in closures)
        append(case,current,count)
        closed = seal(current)
        assert closed['continuable'] and closed['status']=='interrupted'
        closures.append(closed)
    root, history, _, _ = lineage.history(current['directory'],current['root_sha256'])
    actual, directories, ancestry, _, _ = lineage.inherit(prepared,current['directory'],current['root_sha256'])
    assert len(actual)==24 and len(history)==3 and len(set(directories))==3
    assert len(ancestry['prior_closures'])==3
    for row, folder, expected in zip(actual,directories,baseline):
        assert all(np.array_equal(a,b) for a,b in zip(lineage.science.load_particle_evidence(row,folder),
                                                     lineage.science.load_particle_evidence(expected,path.parent)))
        assert row['scores']==expected['scores'] and row['particle_precision']==expected['particle_precision']
    assert len(tracker['calls'])==before
    assert all(p.read_bytes()==value for p,value in saved_reference.items())
    final = reserve(case)  # Full saved prefix can still need its whole audit.
    assert final['start']['inherited_count']==24 and final['start']['cooperative_remaining_seconds'] < 1000
    assert root['plan']['particles']==[16] and prepared.reference[1]['particle_counts']==[4,8]


def test_unclosed_reservation_blocks_duplicate_launch_and_path_forks(case):
    first = reserve(case)
    with pytest.raises(FileExistsError,match='unclosed'):
        reserve(case)
    with pytest.raises(ValueError,match='sole latest'):
        lineage.history(first['directory'],first['root_sha256'],current_index=1)
    source, locations, *_ = case
    with pytest.raises(ValueError,match='cannot fork'):
        lineage.initialize(source=source,locations=dict(locations,snapshot=locations['snapshot']/'other'))
    assert [p.name for p in (first['directory']/'attempts').iterdir()]==['000000']


def test_empty_closed_interruption_charges_budget_and_preserves_zero_prefix(case):
    first = reserve(case)
    closed = seal(first)
    second = reserve(case)
    assert closed['continuable'] and closed['journal_tip'] is None
    assert second['start']['inherited_count']==0
    assert second['start']['prior_elapsed_seconds']==closed['elapsed_seconds'] > 0
    assert second['start']['outer_remaining_seconds'] < first['start']['outer_remaining_seconds']


@pytest.mark.parametrize('fault',['returncode','timeout','error','not_interrupted','failed_row','core_failure'])
def test_failures_and_unknown_exits_are_retained_not_silently_retried(case,fault):
    first = reserve(case)
    changes = {}
    if fault=='returncode': changes['worker_returncode']=0
    if fault=='timeout': changes['timed_out']=True
    if fault=='error': changes['supervisor_error']={'error_type':'SoftwareError'}
    if fault=='not_interrupted': changes['interrupted']=False
    if fault=='failed_row':
        def fail(row): row.update(status='failure',error_type='SoftwareError',error_message='synthetic failed run')
        append(case,first,1,corrupt=fail)
    if fault=='core_failure':
        (first['attempt']/'work').mkdir()
        lineage.journal._publish_json(first['attempt']/'work/core-result.json',{'status':'failed'})
    result = seal(first,**changes)
    assert result['status']=='failed' and result['continuable'] is False
    assert result['new_failure_count']==(1 if fault=='failed_row' else 0)
    with pytest.raises(ValueError,match='no automatic retry'):
        reserve(case)


@pytest.mark.parametrize('fault',['active','unclosed','job','undercharge'])
def test_live_or_mismatched_process_evidence_cannot_be_sealed(case,fault):
    first = reserve(case)
    changes = {'active':{'accounting':{'active_processes':1}},'unclosed':{'process_tree_closed':False},
        'job':{'job_name':'another-job'},'undercharge':{'elapsed_seconds':1000.}}[fault]
    publish_process(first,**changes)
    with pytest.raises(ValueError): lineage.seal(first)
    assert not (first['attempt']/'closure.json').exists()


def test_closed_tail_deletion_or_particle_mutation_is_detected(case):
    first = reserve(case)
    append(case,first,2)
    seal(first)
    path = first['attempt']/'work/journal/records/000001.json'
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError,match='inventory'): reserve(case)
    path.write_bytes(original)
    particle = first['attempt']/'work/journal/particles/000001.npz'
    particle.write_bytes(particle.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='inventory'): reserve(case)


def test_uncommitted_orphans_are_preserved_but_never_inherited(case):
    first = reserve(case)
    append(case,first,2)
    orphan = first['attempt']/'work/journal/particles/000002.npz'
    orphan.write_bytes(b'no committed record')
    closed = seal(first)
    assert closed['new_committed_count']==2
    second = reserve(case)
    rows, _, _, _, _ = lineage.inherit(case[2],second['directory'],second['root_sha256'],current_index=1)
    assert len(rows)==2 and orphan.read_bytes()==b'no committed record'


@pytest.mark.parametrize('fault',['score','block','manifest','workload'])
def test_closed_byte_integrity_is_not_a_substitute_for_scientific_admission(case,fault):
    first = reserve(case)
    def corrupt(row):
        if fault=='score': row['scores']['time_weighted_energy_score_m'] += 1
        if fault=='block': row['independent_block_id']='not-original'
    writer = append(case,first,1,corrupt=corrupt)
    if fault in ('manifest','workload'):
        path = writer.directory/'manifest.json'
        value = json.loads(path.read_text())
        if fault=='manifest': value['contract']['scientific_header']['input_validation_seconds'] += 1
        if fault=='workload': value['workloads'][-1]['seed'] += 1
        # Rebind the test record to the deliberately changed manifest so the
        # storage-only chain is valid; science/contract checks must still fail.
        path.write_bytes(lineage.journal._encode(value))
        sha = lineage.native._hash(path)
        record_path = writer.directory/'records/000000.json'
        record = json.loads(record_path.read_text())
        record.update(manifest_sha256=sha,previous_record_sha256=sha)
        record_path.write_bytes(lineage.journal._encode(record))
    seal(first)
    second = reserve(case)
    with pytest.raises(ValueError):
        lineage.inherit(case[2],second['directory'],second['root_sha256'],current_index=1)


def test_cumulative_cooperative_expiry_forbids_continuation_even_before_outer_cap(case,monkeypatch):
    first = reserve(case)
    publish_process(first)
    original_clock, real_datetime = lineage.time.perf_counter,lineage.datetime
    class Later(real_datetime):
        @classmethod
        def now(cls,tz=None): return real_datetime.now(tz)+timedelta(seconds=1001)
    monkeypatch.setattr(lineage.time,'perf_counter',lambda:original_clock()+1001)
    monkeypatch.setattr(lineage,'datetime',Later)
    closed = lineage.seal(first)
    assert 1000 < closed['elapsed_seconds'] < 1050
    assert not closed['continuable'] and closed['status']=='failed'
    with pytest.raises(ValueError,match='no automatic retry'): reserve(case)


def test_concurrent_reservation_has_one_winner(case,monkeypatch):
    source,locations,*_ = case
    directory,_,_ = lineage.initialize(source=source,locations=locations)
    original,barrier = lineage._attempts,threading.Barrier(2)
    def synchronized(path):
        result=original(path)
        barrier.wait(timeout=15)
        return result
    monkeypatch.setattr(lineage,'_attempts',synchronized)
    def attempt():
        try: return reserve(case)
        except FileExistsError: return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:attempt(),range(2)))
    assert sum(r is not None for r in results)==1
    assert [p.name for p in (directory/'attempts').iterdir()]==['000000']
