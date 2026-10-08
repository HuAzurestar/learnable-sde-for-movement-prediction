"""Synthetic historical metadata only; never real approval, process or science.

Every accepted synthetic predecessor explicitly replaces the source pins and
native query. Such a fixture cannot pass the production registered identities.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from experiments.pirc17 import formal_budget as budget, formal_entrypoint as entry
from experiments.pirc17 import formal_predecessor as module
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, read_json, unpack


def save(path, payload):
    record = envelope(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(record)+b'\n')
    return record


@pytest.fixture
def startup(tmp_path, monkeypatch):
    phase = 'input_qualification_and_binding'
    ledger_dir = tmp_path/'ledger'
    session_dir = ledger_dir/'session-000000'
    claim = tmp_path/'ledger.launch'
    work_id, science_id = digest('input-fixture'), digest('science-fixture')
    job = 'PIRC17-FORMAL-'+'f'*32
    protocol = envelope(dict(software_fixture_only=True))
    execution = envelope(dict(software_fixture_only=True, role='execution'))
    matrix = envelope(dict(protocol_sha256=protocol['sha256'],phase_caps_seconds={phase:10,'science':20},
        max_generated_forecasts=1,workloads=[dict(work_id=work_id,phase=phase,max_active_seconds=10,generated_forecasts=0),
            dict(work_id=science_id,phase='science',max_active_seconds=20,generated_forecasts=1)]))
    runtime = envelope(dict(ledger_directory=str(ledger_dir),environment=envelope(dict(
        python_executable=dict(path=str(Path(sys.executable).resolve()))))))
    checks = {}
    for name, task in [('test','TEST-01'),('review','REVIEW-01')]:
        checks[name] = save(tmp_path/(name+'.json'),dict(schema_version='pirc17-pre-eval-check-v1',task=task,
            result='PASS',protocol_sha256=protocol['sha256'],execution_sha256=execution['sha256'],
            software_fixture_only=True))
    approval = save(tmp_path/'NOT-HUMAN-APPROVAL.json',dict(schema_version='pirc17-human-accept01-v1',task='ACCEPT-01',
        decision='CONFIRMED',human_confirmation=dict(kind='user-message',reference='SOFTWARE FIXTURE NOT HUMAN AUTHORITY'),
        protocol_sha256=protocol['sha256'],execution_sha256=execution['sha256'],
        test_sha256=checks['test']['sha256'],review_sha256=checks['review']['sha256'],software_fixture_only=True))
    bundle = save(tmp_path/'bundle.json',dict(schema_version=module.HISTORICAL_ENTRY+'-bundle',
        protocol=protocol,execution=execution,matrix=matrix,runtime=runtime))
    authority = entry.authority_paths(approval_path=tmp_path/'NOT-HUMAN-APPROVAL.json',approval_sha256=approval['sha256'],
        test_path=tmp_path/'test.json',review_path=tmp_path/'review.json',ledger_directory=ledger_dir)
    command = entry.worker_command(tmp_path/'bundle.json',bundle['sha256'],authority,runtime)
    contract = budget.contract_for_matrix(matrix,protocol_sha256=protocol['sha256'],execution_sha256=execution['sha256'],
        runtime_manifest_sha256=runtime['sha256'],approval_sha256=approval['sha256'],ledger_directory=ledger_dir)
    with budget.Ledger.create(ledger_dir,contract) as ledger:
        root = ledger.root_sha256
        control = save(ledger_dir/'controls'/'TEMP.json',dict(schema_version='pirc17-cumulative-controller-v1-control',
            ledger_root_sha256=root,phase=phase,phase_deadline_ns=10_000_000_100,session_directory=str(session_dir),
            started_ns=100,worker_command=command,worker_job_name=job))
        (ledger_dir/'controls'/'TEMP.json').rename(ledger_dir/'controls'/(control['sha256']+'.json'))
        ledger.open_control(control['sha256'],phase=phase,credit_ns=1_000_000_000)
        reservation = ledger.reserve(work_id)
        session = save(session_dir/'session.json',dict(schema_version=module.native.VERSION,directory=str(session_dir),
            job_name=job,worker_command=command,ledger_directory=str(ledger_dir),ledger_root_sha256=root,execution_sha256=execution['sha256']))
        ready = save(session_dir/'ready.json',dict(schema_version=module.native.VERSION+'-ready',session_sha256=session['sha256'],worker_pid=123))
        (session_dir/'outputs').mkdir()
        request = save(session_dir/'requests/000000.json',dict(schema_version=module.native.VERSION+'-request',
            session_sha256=session['sha256'],previous_barrier_sha256=session['sha256'],sequence=0,work_id=work_id,
            reservation_sha256=reservation['reservation_sha256'],reservation_event_index=1,started_ns=200,
            deadline_ns=200+reservation['reserved_ns']))
        barrier = save(session_dir/'barriers/000000.json',dict(schema_version=module.native.VERSION+'-barrier',
            session_sha256=session['sha256'],previous_barrier_sha256=session['sha256'],request_sha256=request['sha256'],
            sequence=0,work_id=work_id,reservation_sha256=reservation['reservation_sha256'],worker_pid=123,
            status='failure',result_sha256=None,error_type='ValueError'))
        save(ledger_dir/'dispatches'/(reservation['reservation_sha256']+'.json'),dict(schema_version=module.native.VERSION+'-dispatch',
            ledger_root_sha256=root,reservation_sha256=reservation['reservation_sha256'],work_id=work_id,
            session_directory=str(session_dir),job_name=job))
        closure = dict(process_tree_closed=True,job_name=job,accounting=dict(active_processes=0,total_processes=2))
        native = envelope(dict(schema_version=module.native.VERSION+'-observation',ledger_root_sha256=root,
            reservation_sha256=reservation['reservation_sha256'],work_id=work_id,sequence=0,session_sha256=session['sha256'],
            barrier_sha256=barrier['sha256'],worker_pid=123,status='failure',result_sha256=None,started_ns=200,
            deadline_ns=200+reservation['reserved_ns'],elapsed_ns=3_000_000_000,ended_ns=3_000_000_200,closure=closure))
        observed = save(ledger_dir/'controls'/'WORK-TEMP.json',dict(schema_version='pirc17-cumulative-controller-v1-work-observation',
            ledger_root_sha256=root,control_sha256=control['sha256'],work_id=work_id,reservation_sha256=reservation['reservation_sha256'],
            candidate_status='failure',scientific_manifest_validation=None,native_observation=native,
            work_authority=dict(schema_version='pirc17-cumulative-controller-v1-work-authority',ledger_root_sha256=root,
                execution_sha256=execution['sha256'],approval_sha256=approval['sha256'],work_id=work_id,granted=True)))
        (ledger_dir/'controls'/'WORK-TEMP.json').rename(ledger_dir/'controls'/(observed['sha256']+'.json'))
        ledger.settle(reservation['reservation_sha256'],status='failure',elapsed_ns=3_000_000_000,
            completion_evidence_sha256=observed['sha256'],result_sha256=None,reason='SOFTWARE FIXTURE ONLY')
        ledger.halt('supervision_error',observed['sha256'])
        phase_receipt = save(ledger_dir/'controls'/'PHASE-TEMP.json',dict(schema_version='pirc17-cumulative-controller-v1-phase-observation',
            ledger_root_sha256=root,control_sha256=control['sha256'],control_credit_ns=1_000_000_000,
            control_observed_ns=800_000_000,closure=closure))
        (ledger_dir/'controls'/'PHASE-TEMP.json').rename(ledger_dir/'controls'/(phase_receipt['sha256']+'.json'))
        ledger.close_control(control['sha256'],observed_ns=800_000_000,evidence_sha256=phase_receipt['sha256'],reason='stopped')
        stopped = save(ledger_dir/'controls'/'STOP-TEMP.json',dict(schema_version='pirc17-cumulative-controller-v1-stopped',
            control_sha256=control['sha256'],control_start=control,error_type='ControllerStopped'))
        (ledger_dir/'controls'/'STOP-TEMP.json').rename(ledger_dir/'controls'/(stopped['sha256']+'.json'))
        launch = save(claim/'start.json',dict(schema_version=module.HISTORICAL_ENTRY+'-launch',bundle_path=str(tmp_path/'bundle.json'),
            bundle_sha256=bundle['sha256'],bundle_file_sha256=file_hash(tmp_path/'bundle.json'),ledger_directory=str(ledger_dir),
            authority=authority,started_ns=100,startup_phase=phase,startup_reservation_ns=10_000_000_000,
            incomplete_claim_is_terminal=True,final_eval_authorized=False))
        terminal = save(claim/'terminal.json',dict(schema_version=module.HISTORICAL_ENTRY+'-launch-terminal',launch_sha256=launch['sha256'],
            bundle_sha256=bundle['sha256'],ledger_root_sha256=root,ledger_tip_before_terminal=ledger.tip,process_tree_closed=True,
            candidate_complete=False,approval_verified=True,startup_transferred_to_ledger=True,startup_conservative_charge_ns=0,
            accounting_complete=False,human_accepted=False,scientific_claim_authorized=False))
        proof = digest(dict(content_sha256=terminal['sha256'],file_sha256=file_hash(claim/'terminal.json')))
        ledger.finish_control_tail(control['sha256'],observed_ns=900_000_000,evidence_sha256=proof)
    pins = dict(expected_root_sha256=root,expected_head_sha256=read_json(ledger_dir/'head.json')['sha256'],
        expected_terminal_proof_sha256=proof)
    monkeypatch.setattr(module,'REGISTERED',pins)
    monkeypatch.setattr(module,'REGISTERED_CHARGED_NS',4_000_000_000)
    queries = []
    def closed_query(name):
        queries.append(name)
        return dict(exists=False,accounting=None)
    monkeypatch.setattr(module.native,'_query_job',closed_query)
    return dict(directory=ledger_dir,claim=claim,job=job,queries=queries,root=tmp_path,
        control=control,observed=observed,phase_receipt=phase_receipt,stopped=stopped,reservation=reservation)


def inventory(root):
    return {p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*') if p.is_file()}


def test_complete_linkage_reconciles_costs_without_writer_data_or_mutation(startup,monkeypatch):
    before = inventory(startup['root'])
    def forbidden(*args,**kwargs):
        raise AssertionError('eligibility opened writer or science')
    monkeypatch.setattr(budget.Ledger,'open',forbidden)
    monkeypatch.setattr(budget.Ledger,'create',forbidden)
    binding = module.inspect_startup_predecessor(startup['directory'])
    value = unpack(binding)
    assert value['eligible_closed_input_startup'] and value['process_tree_closed']
    assert value['authorizes_execution'] is False and value['final_eval_reads'] == value['generated_forecasts'] == 0
    assert unpack(value['cost_snapshot'])['charged_total_ns'] == 4_000_000_000
    assert len(value['source_file_sha256']) == 24
    assert startup['queries'] == [startup['job'],startup['job']]
    assert module.verify_startup_predecessor(binding) == value
    assert inventory(startup['root']) == before


@pytest.mark.parametrize('where',['access','session-000000/runtime','session-000000/outputs/000000',
    'session-000001','session-000000/requests/000001.json','session-000000/barriers/000001.json','FOREIGN'])
def test_access_runtime_output_or_extra_work_is_ineligible_even_if_empty(startup,where):
    path = startup['directory']/where
    if path.suffix == '.json': path.write_bytes(b'{}')
    else: path.mkdir()
    before = inventory(startup['root'])
    with pytest.raises(ValueError): module.inspect_startup_predecessor(startup['directory'])
    assert inventory(startup['root']) == before


@pytest.mark.parametrize('where',['start.json','terminal.json','partial.pending'])
def test_missing_or_partial_claim_is_never_repaired(startup,where):
    path = startup['claim']/where
    if path.exists(): path.unlink()
    else: path.write_bytes(b'fixture residue')
    with pytest.raises(ValueError,match='consumed startup claim'):
        module.inspect_startup_predecessor(startup['directory'])


@pytest.mark.parametrize('where',['bundle.json','NOT-HUMAN-APPROVAL.json','test.json','review.json',
    'ledger/head.json','ledger/ledger.json','ledger.launch/start.json','ledger.launch/terminal.json',
    'ledger/session-000000/session.json','ledger/session-000000/requests/000000.json',
    'ledger/session-000000/barriers/000000.json','ledger/session-000000/ready.json'])
def test_rehashed_wrong_historical_records_are_rejected(startup,where):
    path = startup['root']/where
    payload = deepcopy(unpack(read_json(path)))
    payload['schema_version'] = 'INVALID SOFTWARE FIXTURE'
    path.write_bytes(canonical(envelope(payload)))
    with pytest.raises(ValueError): module.inspect_startup_predecessor(startup['directory'])


@pytest.mark.parametrize('name',['control','observed','phase_receipt','stopped'])
def test_each_controller_link_is_required(startup,name):
    path = startup['directory']/'controls'/(startup[name]['sha256']+'.json')
    path.unlink()
    with pytest.raises(ValueError): module.inspect_startup_predecessor(startup['directory'])


@pytest.mark.parametrize('observation',[dict(exists=True,accounting=dict(active_processes=1)),
    dict(exists=True,accounting=None),dict(exists=True,accounting=dict(active_processes=False)),
    dict(exists=False,accounting=dict(active_processes=0)),dict(exists=None,accounting=None)])
def test_saved_closure_never_overrides_live_or_unknown_native_group(startup,monkeypatch,observation):
    monkeypatch.setattr(module.native,'_query_job',lambda name: observation)
    with pytest.raises(ValueError): module.inspect_startup_predecessor(startup['directory'])


def test_live_group_appearing_after_metadata_read_is_rejected(startup,monkeypatch):
    observations = iter([dict(exists=False,accounting=None),dict(exists=True,accounting=dict(active_processes=1))])
    monkeypatch.setattr(module.native,'_query_job',lambda name: next(observations))
    with pytest.raises(ValueError,match='still active'): module.inspect_startup_predecessor(startup['directory'])


def test_native_query_permission_error_is_not_misclassified_as_absence(startup,monkeypatch):
    def denied(name): raise PermissionError('SOFTWARE QUERY DENIAL FIXTURE')
    monkeypatch.setattr(module.native,'_query_job',denied)
    with pytest.raises(PermissionError): module.inspect_startup_predecessor(startup['directory'])


def test_registered_binding_rechecks_bytes_and_actual_liveness(startup,monkeypatch):
    binding = module.inspect_startup_predecessor(startup['directory'])
    path = startup['directory']/'session-000000/session.json'
    path.write_bytes(path.read_bytes()+b'\n')  # Same content hash, different sealed bytes.
    with pytest.raises(ValueError,match='binding changed'): module.verify_startup_predecessor(binding)
    monkeypatch.setattr(module.native,'_query_job',lambda name: dict(exists=True,accounting=dict(active_processes=1)))
    with pytest.raises(ValueError,match='still active'): module.verify_startup_predecessor(binding)


def test_same_meaning_concurrent_replacement_is_detected(startup,monkeypatch):
    read = module._Reads.read
    changed = False
    def replace_after_read(self,path,*args,**kwargs):
        nonlocal changed
        record = read(self,path,*args,**kwargs)
        if path.name == 'ready.json' and not changed:
            path.write_bytes(path.read_bytes()+b'\n')
            changed = True
        return record
    monkeypatch.setattr(module._Reads,'read',replace_after_read)
    with pytest.raises(ValueError,match='bytes changed'): module.inspect_startup_predecessor(startup['directory'])


def test_caller_cannot_replace_source_pins_or_bind_a_free_cost(startup):
    with pytest.raises(TypeError): module.inspect_startup_predecessor(startup['directory'],expected_root_sha256=digest('CHEAPER'))
    binding = unpack(module.inspect_startup_predecessor(startup['directory']))
    cost = unpack(binding['cost_snapshot']); cost['charged_total_ns'] = 0
    binding['cost_snapshot'] = envelope(cost)
    with pytest.raises(ValueError,match='binding changed'): module.verify_startup_predecessor(envelope(binding))


def test_bounded_inventory_rejects_excess_without_traversing_outputs(startup,monkeypatch):
    monkeypatch.setattr(module,'MAX_METADATA_NODES',4)
    with pytest.raises(ValueError,match='bounded startup'): module.inspect_startup_predecessor(startup['directory'])


def test_relative_location_is_not_guessed(startup):
    with pytest.raises(ValueError,match='absolute predecessor'): module.inspect_startup_predecessor('ledger')
