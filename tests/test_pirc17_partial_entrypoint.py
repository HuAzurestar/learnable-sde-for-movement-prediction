"""Full frozen matrix, synthetic predecessor METADATA; no empirical authority.

Positive runtime routing replaces approval/environment/scientific handler
explicitly. No original source data, fitted model or prediction is acquired.
The real CLI/source/ledger/runtime readers and rejection paths are unmodified.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import os
import time

import pytest

from tests.test_pirc17_formal_entrypoint import scientific_contract
from tests.test_pirc17_formal_environment import actual_environment
from experiments.pirc17 import formal_entrypoint as entry,formal_partial_launch as launch
from experiments.pirc17 import formal_budget as budget,formal_session as native,formal_controller as control
from experiments.pirc17 import formal_partial_predecessor as predecessor,formal_partial_costs as costs,formal_partial_science as science
from experiments.pirc17.protocol_core import digest,envelope,unpack,publish,read_json


@pytest.fixture(scope='module')
def metadata(scientific_contract,tmp_path_factory):
    root=tmp_path_factory.mktemp('v4-SOFTWARE-metadata').resolve()
    old=root/'ABSENT-old-empirical-ledger'
    protocol,matrix=scientific_contract
    execution=envelope(dict(protocol_sha256=protocol['sha256'],matrix_sha256=matrix['sha256'],software_only=True))
    runtime=envelope(dict(protocol_sha256=protocol['sha256'],matrix_sha256=matrix['sha256'],ledger_directory=str(old),
        input_paths={k:str(root/('ABSENT-'+k)) for k in entry.INPUT_PATHS}))
    approval=digest('SOFTWARE not human predecessor approval')
    contract=budget.contract_for_matrix(matrix,protocol_sha256=protocol['sha256'],execution_sha256=execution['sha256'],
        runtime_manifest_sha256=runtime['sha256'],approval_sha256=approval,ledger_directory=old)
    works=unpack(matrix)['workloads']
    completed=[w for w in works if w['kind'] in {'input_qualification_and_population','method_fit','terrain_fit'}]
    predictions=[w for w in works if w['kind']=='scientific_forecast']
    completed.append(predictions[0]);pending=predictions[1]
    statuses={w['work_id']:'success' for w in completed};statuses[pending['work_id']]='reserved'
    tip=dict(root_sha256=digest(contract),event_count=60,last_event_sha256=digest('SOFTWARE native tip'))
    charges=dict.fromkeys(contract['phase_caps_ns'],0)
    charges[entry.PHASE_ORDER[0]]=100*budget.NANOSECONDS;charges['method_forecasts']=30*budget.NANOSECONDS
    cost=envelope(dict(schema_version=costs.VERSION,read_only=True,authorizes_execution=False,
        authorizes_generation_token_transfer=False,process_tree_closed=True,pending_dispatched=False,pending_requested=False,
        old_ledger_modified=False,accounting_categories_reconciled=True,ledger_directory=str(old),ledger_root_sha256=digest(contract),
        committed_tip=tip,protocol_sha256=protocol['sha256'],execution_sha256=execution['sha256'],matrix_sha256=matrix['sha256'],
        runtime_manifest_sha256=runtime['sha256'],approval_sha256=approval,work_dispositions=statuses,
        work_disposition_counts=dict(Counter(statuses.values())),pending_reservation=dict(work_id=pending['work_id'],phase=pending['phase'],
            reserved_ns=30*budget.NANOSECONDS,generated_forecasts=1,reservation_name='SOFTWARE never dispatched',
            reservation_sha256=digest('SOFTWARE pending')),
        generation_reservations_retained=2,generated_calls_released=0,phase_caps_ns=contract['phase_caps_ns'],total_cap_ns=contract['total_cap_ns'],
        charged_ns_by_phase=charges,measured_ns_by_phase=dict.fromkeys(charges,0),conservatively_charged_ns_by_phase=charges,
        control_charged_ns_by_phase=dict.fromkeys(charges,0),control_observed_ns_by_phase=dict.fromkeys(charges,0),
        charged_total_ns=sum(charges.values()),retained_terminal_total_hold_ns=sum(charges.values()),
        head_sha256=digest('SOFTWARE head'),terminal_sha256=digest('SOFTWARE terminal')))
    def artifact(w):
        if w['kind']=='input_qualification_and_population':return None
        return dict(path='ABSENT-files/'+w['work_id']+'.json',file_sha256=digest(['SOFTWARE bytes',w['work_id']]),
                    content_sha256=digest(['SOFTWARE record',w['work_id']]))
    sources={w['work_id']:dict(result_sha256=digest(['SOFTWARE result',w['work_id']]),
        settlement_sha256=digest(['SOFTWARE settlement',w['work_id']]),observation_sha256=digest(['SOFTWARE observation',w['work_id']]),
        controller_elapsed_ns=1,artifact=artifact(w)) for w in completed}
    saved=envelope(dict(schema_version=science.VERSION,read_only=True,authorizes_execution=False,cross_execution_admitted=False,
        new_fits=0,new_forecasts=0,ledger_root_sha256=digest(contract),committed_tip=tip,original_protocol_sha256=protocol['sha256'],
        original_execution_sha256=execution['sha256'],original_matrix_sha256=matrix['sha256'],original_approval_sha256=approval,
        completed_sources=sources,successful_work_counts=dict(Counter(w['kind'] for w in completed)),
        fit_bindings={w['fit_identity']:artifact(w) for w in completed if w['kind'] in {'method_fit','terrain_fit'}},
        forecast_index={predictions[0]['work_id']:artifact(predictions[0])},forecast_kind_counts={'scientific_forecast':1},
        forecast_status_counts={'success':1},context_sha256=digest('SOFTWARE absent context')))
    refs={}
    for name,record in [('cost',cost),('science',saved),('bundle',envelope(dict(protocol=protocol,execution=execution,matrix=matrix,runtime=runtime)))]:
        path,_=publish(root/name,unpack(record));refs[name]=predecessor.reference(path)
    binding=predecessor.build_binding(cost_reference=refs['cost'],science_reference=refs['science'],bundle_reference=refs['bundle'])
    path,_=publish(root/'binding',unpack(binding))
    return dict(root=root,protocol=protocol,matrix=matrix,binding_path=path,cost=cost,runtime=runtime,completed=completed,pending=pending)


def prepare(metadata,tmp_path,actual_environment,monkeypatch):
    from experiments.pirc17 import formal_environment
    class Environment:
        def __init__(self,p):self.hardware=deepcopy(unpack(actual_environment)['hardware'])
        def identity(self):return actual_environment
        def verify(self,value):assert value==actual_environment
        def check(self):return self.hardware
    monkeypatch.setattr(formal_environment,'ConfiguredRuntime',Environment)
    path,record=launch.prepare_partial(output_directory=tmp_path/'bundle',ledger_directory=tmp_path/'ledger',
        predecessor_reference_path=metadata['binding_path'])
    ap,approval=publish(tmp_path/'SOFTWARE-authority',dict(software_only=True))
    auth=entry.authority_paths(approval_path=ap,approval_sha256=approval['sha256'],test_path=ap,review_path=ap,
        ledger_directory=tmp_path/'ledger')
    bundle,runtime=entry.validate_bundle(record)
    return path,record,bundle,runtime,auth


def test_partial_preparation_preserves_full_science_without_reading_absent_data(metadata,tmp_path,actual_environment,monkeypatch):
    path,record,b,r,auth=prepare(metadata,tmp_path,actual_environment,monkeypatch)
    assert b['protocol']==metadata['protocol'] and b['matrix']==metadata['matrix']
    assert unpack(b['matrix'])['counts_by_kind']['scientific_forecast']==11020
    assert len(unpack(b['matrix'])['workloads'])==11659
    assert r['input_paths']==unpack(metadata['runtime'])['input_paths']
    assert not (tmp_path/'ledger').exists() and not entry.launch_directory(tmp_path/'ledger').exists()
    charges,root=launch.launch_floor(b,approval_sha256=auth['approval_sha256'])
    assert charges==unpack(metadata['cost'])['charged_ns_by_phase'] and sum(charges.values())==130*budget.NANOSECONDS
    assert entry.worker_command(path,record['sha256'],auth,b['runtime'])[1:5]==['-u','-m',entry.MODULE,'worker']


@pytest.mark.parametrize('fault',['path','reference','old_location','phase_order','generation','version'])
def test_rehashed_partial_scope_changes_reject(metadata,tmp_path,actual_environment,monkeypatch,fault):
    _,record,b,r,auth=prepare(metadata,tmp_path,actual_environment,monkeypatch)
    value=deepcopy(unpack(record));runtime=deepcopy(r)
    if fault=='path':runtime['input_paths']['trajectory_path']=str(tmp_path/'OTHER')
    elif fault=='reference':runtime['predecessor_reference']['content_sha256']=digest('other predecessor')
    elif fault=='old_location':runtime['ledger_directory']=unpack(metadata['runtime'])['ledger_directory']
    elif fault=='phase_order':runtime['phase_order'].reverse()
    elif fault=='version':runtime['schema_version']=entry.VERSION+'-runtime'
    else:
        m=deepcopy(unpack(value['matrix']));m['max_generated_forecasts']-=1;value['matrix']=envelope(m)
    value['runtime']=envelope(runtime)
    with pytest.raises(ValueError):entry.validate_bundle(envelope(value))


def test_partial_first_event_is_once_only_and_never_demands_new_input_reservation(metadata,tmp_path,actual_environment,monkeypatch):
    _,record,b,r,auth=prepare(metadata,tmp_path,actual_environment,monkeypatch)
    monkeypatch.setattr(predecessor,'verify_partial_predecessor',lambda b:b) # SOFTWARE metadata only.
    contract=budget.contract_for_matrix(b['matrix'],protocol_sha256=b['protocol']['sha256'],execution_sha256=b['execution']['sha256'],
        runtime_manifest_sha256=b['runtime']['sha256'],approval_sha256=auth['approval_sha256'],ledger_directory=tmp_path/'ledger')
    with budget.Ledger.create(tmp_path/'ledger',contract) as ledger:
        ledger.import_partial_floor(b['runtime'])
        first=entry.initial_floor_record(b['runtime'],ledger_root_sha256=ledger.root_sha256)[0]
        assert unpack(first)['type']=='partial_predecessor'
        assert entry.verify_initial_floor(b['runtime'],contract=contract,ledger_root_sha256=ledger.root_sha256)==first['sha256']
        with pytest.raises(ValueError,match='pending first input'):
            entry.verify_initial_floor(b['runtime'],contract=contract,ledger_root_sha256=ledger.root_sha256,require_first_pending=True)
        assert ledger.summary()['imported_success_count']==28 and ledger.summary()['generated_forecasts_reserved']==2
        with pytest.raises(ValueError):ledger.import_partial_floor(b['runtime'])


def test_retired_partial_launch_never_enters_restoration_or_consumes_claim(metadata,tmp_path,actual_environment,monkeypatch):
    path,record,b,r,auth=prepare(metadata,tmp_path,actual_environment,monkeypatch)
    monkeypatch.setattr(entry, 'validate_bundle', lambda *a: pytest.fail('legacy restoration validation entered'))
    monkeypatch.setattr(control, 'Controller', lambda *a, **kw: pytest.fail('legacy controller started'))
    with pytest.raises(ValueError, match='restoration run is retired; use.*checkpoint_resume'):entry.run(bundle_path=path,bundle_sha256=record['sha256'],
        **{k:auth[k] for k in ('approval_path','approval_sha256','test_path','review_path')})
    claim=entry.launch_directory(tmp_path/'ledger')
    assert not claim.exists() and not (tmp_path/'ledger').exists()
    with pytest.raises(ValueError, match='restoration run is retired'):entry.run(bundle_path=path,bundle_sha256=record['sha256'],
        **{k:auth[k] for k in ('approval_path','approval_sha256','test_path','review_path')})
    assert not claim.exists()


def test_runtime_bootstrap_routes_only_partial_reference_and_records_distinct_runtime(metadata,tmp_path,actual_environment,monkeypatch):
    from experiments.pirc17 import formal_worker
    path,record,b,r,auth=prepare(metadata,tmp_path,actual_environment,monkeypatch)
    monkeypatch.setattr(entry,'_approval',lambda *a:None) # Explicit SOFTWARE authority seam.
    monkeypatch.setattr(predecessor,'verify_partial_predecessor',lambda b:b)
    contract=budget.contract_for_matrix(b['matrix'],protocol_sha256=b['protocol']['sha256'],execution_sha256=b['execution']['sha256'],
        runtime_manifest_sha256=b['runtime']['sha256'],approval_sha256=auth['approval_sha256'],ledger_directory=tmp_path/'ledger')
    seen=[]
    class Handler:
        def __init__(self,session,contract,*,input_options):
            assert input_options['partial_predecessor_reference']==r['predecessor_reference']
            assert 'continuation_binding' not in input_options
            self.closed=False
        def bootstrap(self,output):seen.append('restore');return dict(software_only=True)
        def __call__(self,work,output):seen.append(work['work_id']);return dict(software_only=True)
        def close(self):self.closed=True
    monkeypatch.setattr(formal_worker,'FormalWorker',Handler)
    with budget.Ledger.create(tmp_path/'ledger',contract) as ledger:
        ledger.import_partial_floor(b['runtime'])
        directory=ledger.directory/'session-000001';directory.mkdir()
        session=dict(schema_version=native.VERSION,directory=str(directory),job_name='PIRC17-FORMAL-'+'a'*32,
            ledger_directory=str(ledger.directory),ledger_root_sha256=ledger.root_sha256,execution_sha256=b['execution']['sha256'],
            worker_command=entry.worker_command(path,record['sha256'],auth,b['runtime']))
        native._write(directory/'session.json',session)
        now=time.monotonic_ns()
        (ledger.directory/'controls').mkdir()
        c=control._publish(ledger.directory/'controls',dict(schema_version=control.VERSION+'-control',
            ledger_root_sha256=ledger.root_sha256,phase=entry.PHASE_ORDER[0],started_ns=now,
            phase_deadline_ns=now+3500*budget.NANOSECONDS,worker_job_name=session['job_name'],session_directory=str(directory),worker_command=session['worker_command']))
        ledger.open_control(c['sha256'],phase=entry.PHASE_ORDER[0],credit_ns=5*budget.NANOSECONDS)
        native._write(directory/'bootstrap-request.json',dict(schema_version=native.VERSION+'-bootstrap-request',
            session_sha256=digest(session),control_sha256=c['sha256'],started_ns=now,deadline_ns=now+3500*budget.NANOSECONDS))
        wrapper=entry.RuntimeWorker(session,contract,bundle_path=path,bundle_sha256=record['sha256'],authority=auth)
        before=ledger.summary()
        assert wrapper.bootstrap(directory/'bootstrap')==dict(software_only=True)
        proof=entry.read_runtime_evidence(directory,contract=contract,session_sha256=digest(session),
            ledger_root_sha256=ledger.root_sha256,worker_pid=os.getpid(),phase=None,first_work_id=None)
        assert proof['startup']['content_sha256']==wrapper.start['sha256'] and seen==['restore']
        assert ledger.summary()==before
        work=metadata['pending'];wrapper(work,directory/'outputs')
        proof=entry.read_runtime_evidence(directory,contract=contract,session_sha256=digest(session),
            ledger_root_sha256=ledger.root_sha256,worker_pid=os.getpid(),phase=work['phase'],first_work_id=work['work_id'])
        assert proof and seen==['restore',work['work_id']]
