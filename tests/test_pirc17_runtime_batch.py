"""Pure synthetic historical runtime/floor checks, NOT launch qualification."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.pirc17 import formal_budget as budget, formal_entrypoint as entry
from experiments.pirc17 import formal_environment as environment, formal_session as native
from experiments.pirc17 import formal_metadata_batch as batches, formal_runtime_batch as module
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, unpack
from tests.test_pirc17_formal_closed import changed


def fixture(tmp_path, *, historical=False, bootstrap=False):
    root = tmp_path/'ledger'; session = root/'session-000001'; runtime_dir = session/'runtime'
    runtime_dir.mkdir(parents=True)
    contract = dict(protocol_sha256=digest('SYNTHETIC protocol'), execution_sha256=digest('SYNTHETIC execution'),
        matrix_sha256=digest('SYNTHETIC matrix'), ledger_directory=str(root), phase_caps_ns={'first':1000000000})
    observed = envelope(dict(schema_version=environment.VERSION, protocol_sha256=contract['protocol_sha256'],
                            hardware={'software_fixture_only':True}, final_eval_authorized=False))
    version = 'pirc17-concrete-formal-entrypoint-v1' if historical else entry.RESOURCE_VERSION
    payload = dict(schema_version=version+'-runtime', protocol_sha256=contract['protocol_sha256'],
        matrix_sha256=contract['matrix_sha256'], environment=observed, ledger_directory=str(root),
        input_paths={k:str(root/k) for k in entry.INPUT_PATHS}, working_directory=str(tmp_path),
        worker_module=entry.MODULE, phase_order=['first'], final_eval_authorized=False)
    if not historical:
        payload.update(predecessor=envelope(dict(software_only=True)),
                       predecessor_reference=dict(software_only=True))
    runtime = native._write(runtime_dir/'binding.json', payload)
    contract['runtime_manifest_sha256'] = runtime['sha256']
    session_sha, ledger_sha, pid = digest('SYNTHETIC session'), digest('SYNTHETIC ledger'), 777
    start = dict(schema_version=version+'-worker-runtime', session_sha256=session_sha,
        ledger_root_sha256=ledger_sha, execution_sha256=contract['execution_sha256'],
        runtime_manifest_sha256=runtime['sha256'], worker_pid=pid, environment=observed)
    if not historical:
        (root/'events').mkdir()
        floor = native._write(root/'events/000000.json', dict(schema_version=budget.VERSION+'-event',
            index=0, root_sha256=ledger_sha, previous_sha256=ledger_sha,
            type='resource_predecessor', row=dict(runtime_manifest=runtime)))
        start['predecessor_floor_event_sha256'] = floor['sha256']
    start = native._write(runtime_dir/'start.json', start)
    phase, first = (None, None) if bootstrap else ('first', digest('SYNTHETIC first work'))
    marker = dict(schema_version=version+('-bootstrap-runtime' if bootstrap else '-phase-runtime'),
        startup_sha256=start['sha256'], worker_pid=pid,
        before_sha256=digest(unpack(observed)['hardware']), after_sha256=digest(unpack(observed)['hardware']))
    if bootstrap:
        request = native._write(session/'bootstrap-request.json', dict(software_only=True))
        marker['request_sha256'] = request['sha256']
    else: marker.update(phase=phase, first_work_id=first)
    native._write(runtime_dir/('bootstrap.json' if bootstrap else digest(phase)+'.json'), marker)
    args = dict(contract=contract, session_sha256=session_sha, ledger_root_sha256=ledger_sha,
                worker_pid=pid, phase=phase, first_work_id=first)
    return dict(root=root, session=session, runtime=runtime_dir, args=args, owner=SimpleNamespace())


@pytest.mark.parametrize('mode', ['historical', 'resource', 'bootstrap'])
def test_repeated_reads_use_original_validator_at_both_boundaries_only(tmp_path, monkeypatch, mode):
    case = fixture(tmp_path, historical=mode=='historical', bootstrap=mode=='bootstrap')
    calls=[]
    original=entry.read_runtime_evidence
    def checked(*a,**kw):
        calls.append(kw['phase'])
        return original(*a,**kw)
    monkeypatch.setattr(entry,'read_runtime_evidence',checked)
    expected=original(case['session'],**case['args'])
    with batches.metadata_batch(case['owner']) as batch:
        for _ in range(1000):
            value=module.read_closed_runtime(case['owner'],case['session'],**case['args'])
            assert value==expected
        assert calls==[case['args']['phase']] and len(batch.validations)==1
        assert len(batch.readers)==(3 if mode=='historical' else 5 if mode=='bootstrap' else 4)
        with pytest.raises(TypeError):value['binding']['file_sha256']=digest('mutated proof')
    assert calls==[case['args']['phase']]*2 and batch.validations=={}
    assert case['owner']._metadata_batch is None
    module.read_closed_runtime(case['owner'],case['session'],**case['args'])
    assert len(calls)==3  # No cold-reader shortcut persists.


@pytest.mark.parametrize('fault',['binding','startup','phase','floor','request'])
def test_changed_file_bytes_block_batch_handoff_even_when_json_meaning_is_equal(tmp_path,fault):
    case=fixture(tmp_path,bootstrap=fault=='request')
    path={'binding':case['runtime']/'binding.json','startup':case['runtime']/'start.json',
          'phase':case['runtime']/(digest('first')+'.json'),
          'floor':case['root']/'events/000000.json','request':case['session']/'bootstrap-request.json'}[fault]
    raw=path.read_bytes()
    handed_off=False
    try:
        with pytest.raises(ValueError,match='bytes changed'):
            with batches.metadata_batch(case['owner']):
                module.read_closed_runtime(case['owner'],case['session'],**case['args'])
                path.write_bytes(raw+b' ')
                module.read_closed_runtime(case['owner'],case['session'],**case['args'])
            handed_off=True
        assert not handed_off and case['owner']._metadata_batch is None
    finally:path.write_bytes(raw)


@pytest.mark.parametrize('fault',['pid','session','root','execution','first_work','phase'])
def test_changed_complete_scope_cannot_hit_previous_validation(tmp_path,fault):
    case=fixture(tmp_path)
    altered=deepcopy(case['args'])
    if fault=='pid':altered['worker_pid']+=1
    elif fault=='session':altered['session_sha256']=digest('other session')
    elif fault=='root':altered['ledger_root_sha256']=digest('other ledger')
    elif fault=='execution':altered['contract']['execution_sha256']=digest('other execution')
    elif fault=='first_work':altered['first_work_id']=digest('other first work')
    else:altered['phase']='other phase'
    with batches.metadata_batch(case['owner']):
        module.read_closed_runtime(case['owner'],case['session'],**case['args'])
        with pytest.raises((ValueError,FileNotFoundError)):
            module.read_closed_runtime(case['owner'],case['session'],**altered)


def test_closing_original_semantic_validator_is_mandatory(tmp_path,monkeypatch):
    case=fixture(tmp_path)
    original=entry.read_runtime_evidence
    calls=[]
    def checked(*a,**kw):
        calls.append(1)
        if len(calls)>1:raise ValueError('SOFTWARE original semantic/link failure')
        return original(*a,**kw)
    monkeypatch.setattr(entry,'read_runtime_evidence',checked)
    with pytest.raises(ValueError,match='semantic/link failure'):
        with batches.metadata_batch(case['owner']) as batch:
            module.read_closed_runtime(case['owner'],case['session'],**case['args'])
    assert batch.status=='failed' and batch.validations=={} and case['owner']._metadata_batch is None


def test_runtime_validation_inventory_is_finite_and_cleared(tmp_path,monkeypatch):
    case=fixture(tmp_path)
    monkeypatch.setattr(batches,'MAX_VALIDATIONS',0)
    with pytest.raises(ValueError,match='validation inventory'):
        with batches.metadata_batch(case['owner']) as batch:
            module.read_closed_runtime(case['owner'],case['session'],**case['args'])
    assert batch.status=='failed' and batch.validations=={}
