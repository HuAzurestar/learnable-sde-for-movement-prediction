"""Full-size software file inventory, no real eligibility/sample/forecast read."""
from copy import deepcopy
import os
from pathlib import Path
import sys

import pytest

from experiments.pirc17 import formal_budget as budget, formal_controller as control, formal_session as native
from experiments.pirc17.formal_closed import ClosedOutputs
from experiments.pirc17.formal_matrix import build_matrix, load_protocol
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, read_json, unpack
from tests.test_pirc17_formal_closed import changed
from tests.test_pirc17_formal_controller import authority, contract, verification


@pytest.fixture
def ledger(tmp_path):
    with budget.Ledger.create(tmp_path/'ledger', contract(tmp_path/'ledger')) as value: yield value


def test_small_verification_keeps_existing_inline_path_without_more_io(ledger):
    work = ledger._state.contract['workloads'][0]
    small = dict(schema_version=control.VERSION+'-artifact-verification', work_id=work['work_id'], verified=True,
                 artifacts={}, details={'software': True})
    assert control._save_verification(ledger, work, small) is small
    assert control.read_verification(ledger, work, small) is small
    assert not (ledger.directory/'verifications').exists()


@pytest.mark.parametrize('fault', [None, 'root', 'work', 'size', 'hash', 'content', 'changed-file', 'bound'])
def test_large_verification_reference_is_not_an_unchecked_success(ledger, fault):
    work = ledger._state.contract['workloads'][0]
    value = dict(schema_version=control.VERSION+'-artifact-verification', work_id=work['work_id'], verified=True,
        artifacts={}, details={'software_only_padding': 'x'*native.MAX_MESSAGE_BYTES})
    ref = control._save_verification(ledger, work, value)
    assert len(canonical(ref)) < native.MAX_MESSAGE_BYTES
    if fault is None:
        assert control.read_verification(ledger, work, ref) == value
        return
    if fault == 'root': ref['ledger_root_sha256'] = '0'*64
    elif fault == 'work': ref['work_id'] = '0'*64
    elif fault == 'size': ref['bytes'] += 1
    elif fault == 'hash': ref['file_sha256'] = '0'*64
    elif fault == 'content': ref['content_sha256'] = '0'*64
    elif fault == 'bound': ref['bytes'] = control.MAX_VERIFICATION_BYTES+1
    else:
        path = ledger.directory/'verifications'/(ref['content_sha256']+'.json')
        with changed(path, b'changed'), pytest.raises(ValueError): control.read_verification(ledger, work, ref)
        return
    with pytest.raises((ValueError, FileNotFoundError)): control.read_verification(ledger, work, ref)


def test_large_file_cannot_replace_scope_even_with_rehashed_outer_reference(ledger):
    work = ledger._state.contract['workloads'][0]
    value = dict(details={'software_padding': 'x'*native.MAX_MESSAGE_BYTES})
    ref = control._save_verification(ledger, work, value)
    path = ledger.directory/'verifications'/(ref['content_sha256']+'.json')
    payload = deepcopy(unpack(read_json(path))); payload['work_id'] = '0'*64
    forged = envelope(payload); target = path.parent/(forged['sha256']+'.json'); target.write_bytes(canonical(forged))
    ref.update(content_sha256=forged['sha256'], file_sha256=file_hash(target), bytes=target.stat().st_size)
    with pytest.raises(ValueError, match='another ledger/work'): control.read_verification(ledger, work, ref)


def test_unbounded_verification_cannot_publish(ledger):
    work = ledger._state.contract['workloads'][0]
    with pytest.raises(ValueError, match='bounded file size'):
        control._save_verification(ledger, work, {'padding': 'x'*control.MAX_VERIFICATION_BYTES})
    assert not (ledger.directory/'verifications').exists()


def test_actual_native_full_candidate_capacity_inventory_is_preserved_and_closed(tmp_path):
    if os.name != 'nt': pytest.skip('native Windows closed artifact inventory')
    protocol = load_protocol(); matrix = build_matrix(protocol)
    capacity = unpack(protocol)['dataset_inputs']['partitions']['counts']['final_eval']+4
    assert capacity == 12374  # Maximum qualified windows plus the four input records.
    binding = budget.contract_for_matrix(matrix, protocol_sha256=protocol['sha256'],
        execution_sha256=digest('SOFTWARE CAPACITY ONLY'), runtime_manifest_sha256=digest('SOFTWARE RUNTIME'),
        approval_sha256=digest('NO HUMAN AUTHORIZATION'), ledger_directory=tmp_path/'ledger')
    full = next(w for w in unpack(matrix)['workloads'] if w['kind'] == 'input_qualification_and_population')
    with budget.Ledger.create(tmp_path/'ledger', binding) as ledger:
        runner = control.Controller(ledger, [sys.executable, '-u', str((Path(__file__).parent/'fixtures/pirc17_session_worker.py').resolve()),
            '--mode', 'many_artifacts', '--artifact-count', str(capacity)],
            authorize_work=lambda w: authority(ledger, w), validate_result=verification)
        try:
            runner._open_phase(full['phase'])
            runner._work(ledger._state.work[full['work_id']])
            runner._close_phase(last=True)
        finally:
            if runner.meter is not None: runner._close_phase(last=True, stopped=True)
            runner.session.close()
        closed = ClosedOutputs(ledger.directory, contract=binding, tip=ledger.tip)
        item = closed.read(full)
        assert len(item['artifacts']) == capacity
        observer = unpack(read_json(ledger.directory/'controls'/(item['observation_sha256']+'.json')))
        ref = observer['scientific_manifest_validation']
        assert ref['schema_version'] == control.VERSION+'-verification-reference'
        assert ref['bytes'] > native.MAX_MESSAGE_BYTES
        assert len(canonical(envelope(observer))) < native.MAX_MESSAGE_BYTES
        assert native._query_job(runner.session.job_name)['exists'] is False
        # Actual closed reading still verifies all bytes, not just the sidecar.
        path = item['directory']/next(iter(item['artifacts']))
        with changed(path, b'different'), pytest.raises(ValueError, match='bytes/hash changed'): closed.read(full)
        proof = ledger.directory/'verifications'/(ref['content_sha256']+'.json')
        with changed(proof, b'damaged'), pytest.raises(ValueError): closed.read(full)
