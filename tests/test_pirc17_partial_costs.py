"""Actual immutable journal fixtures; explicit native-closure observation seam.

No empirical inputs, training, generation, human approval or live native Job.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from experiments.pirc17 import formal_budget as budget, formal_controller as control
from experiments.pirc17 import formal_partial_costs as module, formal_session as native
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, read_json, unpack


@pytest.fixture
def partial(tmp_path,monkeypatch):
    directory = (tmp_path/'ledger').resolve()
    phase = 'method_forecasts'
    contract = dict(schema_version=budget.VERSION,protocol_sha256=digest('fixture protocol'),
        execution_sha256=digest('fixture execution'),matrix_sha256=digest('fixture matrix'),
        runtime_manifest_sha256=digest('fixture runtime'),approval_sha256=digest('NOT APPROVAL'),
        ledger_directory=str(directory),phase_caps_ns={'input':1000,phase:1000},total_cap_ns=2000,
        max_generated_forecasts=2,max_attempts_per_item=1,
        workloads=[dict(work_id=digest(0),phase='input',max_active_ns=700,generated_forecasts=0),
                   dict(work_id=digest(1),phase=phase,max_active_ns=500,generated_forecasts=1),
                   dict(work_id=digest(2),phase=phase,max_active_ns=500,generated_forecasts=1)])
    ledger = budget.Ledger.create(directory,contract)
    session = directory/'session-000000'
    session.mkdir()
    (session/'requests').mkdir()
    job = 'PIRC17-FORMAL-'+'0'*32
    worker = [sys.executable,'-u','SOFTWARE-FIXTURE-NOT-ACTUAL-WORKER']
    native._write(session/'session.json',dict(schema_version=native.VERSION,directory=str(session),
        job_name=job,worker_command=worker,ledger_directory=str(directory),ledger_root_sha256=ledger.root_sha256,
        execution_sha256=contract['execution_sha256']))
    def start(phase,clock):
        record = envelope(dict(schema_version=control.VERSION+'-control',ledger_root_sha256=ledger.root_sha256,
            phase=phase,started_ns=clock,phase_deadline_ns=clock+1000,worker_job_name=job,
            session_directory=str(session),worker_command=worker))
        native._write(directory/'controls'/(record['sha256']+'.json'),unpack(record))
        return record['sha256']
    (directory/'controls').mkdir()
    first = start('input',1000)
    ledger.open_control(first,phase='input',credit_ns=100)
    pending = ledger.reserve(digest(0))
    ledger.settle(pending['reservation_sha256'],status='success',elapsed_ns=250,
        completion_evidence_sha256=digest('fixture completion0'),result_sha256=digest('fixture result0'),reason='fixture')
    ledger.close_control(first,observed_ns=90,evidence_sha256=digest('fixture phase0'),reason='phase_complete')
    second = start(phase,1350)
    ledger.open_control(second,phase=phase,credit_ns=20)
    pending = ledger.reserve(digest(1))
    ledger.settle(pending['reservation_sha256'],status='success',elapsed_ns=30,
        completion_evidence_sha256=digest('fixture completion1'),result_sha256=digest('fixture result1'),reason='fixture')
    publish = budget._publish
    def fail_head(path,payload,**kwargs):
        if path.name == 'head.json':
            raise PermissionError('software fixture head publication fault')
        return publish(path,payload,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(budget,'_publish',fail_head)
        with pytest.raises(PermissionError):
            ledger.reserve(digest(2))
    tip, root, pending = deepcopy(ledger.tip), ledger.root_sha256,deepcopy(ledger._state.pending)
    ledger.close()
    claim = Path(str(directory)+'.launch')
    claim.mkdir()
    predecessor = dict.fromkeys(contract['phase_caps_ns'],0)
    launch = native._write(claim/'start.json',dict(schema_version=module.ENTRY_VERSION+'-launch',
        ledger_directory=str(directory),bundle_sha256=digest('fixture bundle'),
        authority=dict(approval_sha256=contract['approval_sha256'],journal_directory=str(directory/'access')),
        predecessor_charged_ns_by_phase=predecessor,started_ns=1000))
    terminal = native._write(claim/'terminal.json',dict(schema_version=module.ENTRY_VERSION+'-launch-terminal',
        launch_sha256=launch['sha256'],bundle_sha256=digest('fixture bundle'),ledger_root_sha256=root,
        ledger_tip_before_terminal=tip,process_tree_closed=True,approval_verified=True,candidate_complete=False,
        accounting_complete=False,error_type='PermissionError',elapsed_through_receipt_ns=400,
        predecessor_charged_ns_by_phase=predecessor,cumulative_charge_lower_bound_ns=1100,
        predecessor_floor_transferred_to_ledger=False,startup_transferred_to_ledger=False))
    monkeypatch.setattr(module.native,'_query_job',lambda name:dict(exists=False,accounting=None))
    return dict(directory=directory,contract=contract,pending=pending,claim=claim,session=session,phase=phase,
        pins=dict(expected_root_sha256=root,expected_head_sha256=read_json(directory/'head.json')['sha256'],
                  expected_launch_sha256=launch['sha256'],expected_terminal_sha256=terminal['sha256']))


def inspect(case):
    return unpack(module.inspect_predispatch_costs(case['directory'],**case['pins']))


def changed_record(case,path,transform,pin=None):
    payload = deepcopy(unpack(read_json(path)))
    transform(payload)
    record = envelope(payload)
    path.write_bytes(canonical(record))
    if pin is not None:
        case['pins'][pin] = record['sha256']


def inventory(case):
    return {str(p):file_hash(p) for root in (case['directory'],case['claim']) for p in root.rglob('*') if p.is_file()}


def test_retains_all_costs_pending_generation_and_claim_hold_without_any_writer(partial,monkeypatch):
    before = inventory(partial)
    def forbidden(*args,**kwargs):
        raise AssertionError('cost inspection opened writer')
    monkeypatch.setattr(budget.Ledger,'open',forbidden)
    monkeypatch.setattr(budget.Ledger,'create',forbidden)
    value = inspect(partial)
    assert value['charged_ns_by_phase'] == {'input':350,'method_forecasts':750}
    assert value['measured_ns_by_phase'] == {'input':250,'method_forecasts':30}
    assert value['conservatively_charged_ns_by_phase'] == {'input':100,'method_forecasts':720}
    assert value['charged_total_ns'] == value['retained_terminal_total_hold_ns'] == 1100
    assert value['unknown_tail_hold_ns'] == 200
    assert value['control_charged_ns_by_phase'] == {'input':100,'method_forecasts':20}
    assert value['control_observed_ns_by_phase'] == {'input':90,'method_forecasts':20}
    assert value['active_phase_overhead_through_receipt_ns'] == 20
    assert value['additional_observed_control_charge_ns'] == 0
    assert value['generation_reservations_retained'] == 2 and value['generated_calls_released'] == 0
    assert value['pending_reservation'] == partial['pending']
    assert value['work_disposition_counts'] == {'success':2,'reserved':1}
    assert value['read_only'] and value['accounting_categories_reconciled']
    assert not value['authorizes_execution'] and not value['authorizes_generation_token_transfer']
    assert not value['old_ledger_modified'] and not value['unknown_tail_is_measured']
    assert inventory(partial) == before


@pytest.mark.parametrize('pin',['expected_root_sha256','expected_head_sha256','expected_launch_sha256','expected_terminal_sha256'])
def test_all_four_independent_pins_required(partial,pin):
    partial['pins'][pin] = digest('wrong pin')
    with pytest.raises(ValueError): inspect(partial)


def test_live_original_job_is_a_hard_recovery_failure(partial,monkeypatch):
    monkeypatch.setattr(module.native,'_query_job',lambda name:dict(exists=True,accounting=dict(active_processes=1)))
    with pytest.raises(native.UnclosedTree): inspect(partial)


def test_job_liveness_is_checked_again_before_return(partial,monkeypatch):
    calls = [0]
    def query(name):
        calls[0] += 1
        return dict(exists=calls[0]>2,accounting=dict(active_processes=1))
    monkeypatch.setattr(module.native,'_query_job',query)
    with pytest.raises(native.UnclosedTree): inspect(partial)


@pytest.mark.parametrize('kind',['dispatch','request'])
def test_any_pending_dispatch_or_request_blocks_token_eligibility(partial,kind):
    pending = partial['pending']
    path = (partial['directory']/'dispatches'/(pending['reservation_sha256']+'.json') if kind=='dispatch'
            else partial['session']/'requests/000000.json')
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(canonical(envelope(dict(reservation_sha256=pending['reservation_sha256']))))
    with pytest.raises(ValueError,match='dispatch|request'): inspect(partial)


@pytest.mark.parametrize('name',['000008.json','foreign.txt','.head.pending'])
def test_extra_unanchored_events_are_never_adopted(partial,name):
    (partial['directory']/'events'/name).write_bytes(b'fixture extra event')
    with pytest.raises(ValueError,match='contiguous reserve suffix'): inspect(partial)


def test_missing_suffix_does_not_become_free_credit(partial):
    (partial['directory']/'events/000007.json').unlink()
    with pytest.raises(ValueError,match='contiguous reserve suffix'): inspect(partial)


@pytest.mark.parametrize('change',[
    lambda p:p.update(ledger_root_sha256=digest('wrong root')),
    lambda p:p.update(launch_sha256=digest('wrong launch')),
    lambda p:p.update(process_tree_closed=False),
    lambda p:p.update(candidate_complete=True),
    lambda p:p.update(error_type='ValueError'),
    lambda p:p.update(ledger_tip_before_terminal=dict(event_count=9)),
    lambda p:p.update(predecessor_charged_ns_by_phase={'input':1,'method_forecasts':0}),
    lambda p:p.update(elapsed_through_receipt_ns=360),
])
def test_rehashed_false_terminal_semantics_are_rejected(partial,change):
    changed_record(partial,partial['claim']/'terminal.json',change,'expected_terminal_sha256')
    with pytest.raises(ValueError): inspect(partial)


def test_oversized_terminal_claim_cannot_borrow_input_phase_credit(partial):
    changed_record(partial,partial['claim']/'terminal.json',lambda p:p.update(cumulative_charge_lower_bound_ns=1500),
                   'expected_terminal_sha256')
    with pytest.raises(TimeoutError,match='phase capacity'): inspect(partial)


def test_newly_observed_control_overrun_is_added_before_terminal_hold(partial):
    changed_record(partial,partial['claim']/'terminal.json',lambda p:p.update(elapsed_through_receipt_ns=430),
                   'expected_terminal_sha256')
    value = inspect(partial)
    assert value['additional_observed_control_charge_ns'] == 30
    assert value['control_charged_ns_by_phase']['method_forecasts'] == 50
    assert value['control_observed_ns_by_phase']['method_forecasts'] == 50
    assert value['measured_ns_by_phase']['method_forecasts'] == 60
    assert value['unknown_tail_hold_ns'] == 170 and value['charged_total_ns'] == 1100


def test_a_smaller_terminal_claim_never_refunds_existing_charges(partial):
    changed_record(partial,partial['claim']/'terminal.json',lambda p:p.update(cumulative_charge_lower_bound_ns=100),
                   'expected_terminal_sha256')
    value = inspect(partial)
    assert value['charged_total_ns'] == 900 and value['unknown_tail_hold_ns'] == 0


def test_byte_change_during_inspection_is_detected_even_with_equal_payload(partial,monkeypatch):
    read = module._read_bound
    changed = [False]
    def bound(path,limit):
        result = read(path,limit)
        if path.name == '000000.json' and not changed[0]:
            path.write_bytes(path.read_bytes()+b'\n')
            changed[0] = True
        return result
    monkeypatch.setattr(module,'_read_bound',bound)
    with pytest.raises(ValueError,match='bytes changed'): inspect(partial)


def test_relative_directory_is_not_inferred(partial):
    with pytest.raises(ValueError,match='absolute'):
        module.inspect_predispatch_costs('ledger',**partial['pins'])
