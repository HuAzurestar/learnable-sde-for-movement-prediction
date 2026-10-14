"""Synthetic pinned evidence and actual journal replay; not launch authority.

Live source inspection is explicitly replaced for writer-unit fixtures. The
unmodified production import must invoke it before publishing any event.
"""
from copy import deepcopy
from collections import Counter

import pytest

from experiments.pirc17 import formal_budget as budget, formal_partial_predecessor as module
from experiments.pirc17 import formal_partial_costs as costs, formal_partial_science as science
from experiments.pirc17.protocol_core import canonical,digest,envelope,file_hash,publish,read_json,unpack

NS = budget.NANOSECONDS


@pytest.fixture
def fixture(tmp_path,monkeypatch,request):
    old = (tmp_path/'old-source-not-empirical').resolve()
    new = (tmp_path/'successor-software-journal').resolve()
    old.mkdir()
    protocol = envelope(dict(software_fixture_only=True))
    works = [dict(work_id=digest('input'),kind='input_qualification_and_population',phase='input',
                  max_active_seconds=1,generated_forecasts=0)]
    for i in range(26):
        works.append(dict(work_id=digest(['fit',i]),kind='method_fit' if i<16 else 'terrain_fit',
            phase='method_training' if i<16 else 'terrain_training',max_active_seconds=1,
            generated_forecasts=0,fit_identity=f'fixture-fit{i}'))
    for i in range(2):
        works.append(dict(work_id=digest(['forecast',i]),kind='scientific_forecast',phase='method_forecasts',
                          max_active_seconds=1,generated_forecasts=1))
    marker = request.node.get_closest_marker('partial_phase_seconds')
    seconds = marker.args[0] if marker is not None else 10
    matrix = envelope(dict(protocol_sha256=protocol['sha256'],workloads=works,max_generated_forecasts=2,
        phase_caps_seconds={k:seconds for k in ('input','method_training','terrain_training','method_forecasts')}))
    execution = envelope(dict(protocol_sha256=protocol['sha256'],matrix_sha256=matrix['sha256'],fixture_only=True))
    oldruntime = envelope(dict(protocol_sha256=protocol['sha256'],matrix_sha256=matrix['sha256'],ledger_directory=str(old)))
    oldapproval = digest('not-human-old-approval')
    contract = budget.contract_for_matrix(matrix,protocol_sha256=protocol['sha256'],execution_sha256=execution['sha256'],
        runtime_manifest_sha256=oldruntime['sha256'],approval_sha256=oldapproval,ledger_directory=old)
    original = envelope(dict(protocol=protocol,execution=execution,matrix=matrix,runtime=oldruntime))
    pending = dict(work_id=works[-1]['work_id'],phase='method_forecasts',reserved_ns=NS,
        generated_forecasts=1,reservation_name='SOFTWARE-ONLY-NOT-ACTUAL-DISPATCH',reservation_sha256=digest('old pending'))
    states = {w['work_id']:'success' for w in works[:-1]}
    states[pending['work_id']] = 'reserved'
    tip = dict(root_sha256=digest(contract),event_count=60,last_event_sha256=digest('fixture tip'))
    charges = dict(input=2*NS,method_training=NS,terrain_training=NS,method_forecasts=1500_000_000)
    measured = dict(input=NS,method_training=500_000_000,terrain_training=500_000_000,method_forecasts=200_000_000)
    cost = envelope(dict(schema_version=costs.VERSION,read_only=True,authorizes_execution=False,
        authorizes_generation_token_transfer=False,process_tree_closed=True,pending_dispatched=False,
        pending_requested=False,old_ledger_modified=False,accounting_categories_reconciled=True,
        ledger_directory=str(old),ledger_root_sha256=digest(contract),committed_tip=tip,
        protocol_sha256=protocol['sha256'],execution_sha256=execution['sha256'],matrix_sha256=matrix['sha256'],
        runtime_manifest_sha256=oldruntime['sha256'],approval_sha256=oldapproval,work_dispositions=states,
        work_disposition_counts=dict(Counter(states.values())),pending_reservation=pending,
        generation_reservations_retained=2,generated_calls_released=0,
        phase_caps_ns=contract['phase_caps_ns'],total_cap_ns=contract['total_cap_ns'],charged_ns_by_phase=charges,
        measured_ns_by_phase=measured,conservatively_charged_ns_by_phase={k:charges[k]-measured[k] for k in charges},
        control_charged_ns_by_phase=dict.fromkeys(charges,100_000_000),
        control_observed_ns_by_phase=dict.fromkeys(charges,50_000_000),charged_total_ns=sum(charges.values()),
        retained_terminal_total_hold_ns=sum(charges.values()),head_sha256=digest('fixture head'),
        terminal_sha256=digest('fixture terminal')))
    def artifact(w):
        return dict(path='fixture-artifacts/'+w['work_id']+'.json',file_sha256=digest(['bytes',w['work_id']]),
                    content_sha256=digest(['content',w['work_id']]))
    sources = {w['work_id']:dict(result_sha256=digest(['result',w['work_id']]),
        settlement_sha256=digest(['settlement',w['work_id']]),observation_sha256=digest(['observation',w['work_id']]),
        controller_elapsed_ns=1,artifact=None if w['kind']=='input_qualification_and_population' else artifact(w))
        for w in works[:-1]}
    scientific = envelope(dict(schema_version=science.VERSION,read_only=True,authorizes_execution=False,
        cross_execution_admitted=False,new_fits=0,new_forecasts=0,ledger_root_sha256=digest(contract),committed_tip=tip,
        original_protocol_sha256=protocol['sha256'],original_execution_sha256=execution['sha256'],
        original_matrix_sha256=matrix['sha256'],original_approval_sha256=oldapproval,completed_sources=sources,
        successful_work_counts=dict(Counter(w['kind'] for w in works[:-1])),
        fit_bindings={w['fit_identity']:artifact(w) for w in works[1:27]},
        forecast_index={works[-2]['work_id']:artifact(works[-2])},
        forecast_kind_counts={'scientific_forecast':1},forecast_status_counts={'success':1}))
    paths = {}
    for key,record in (('cost',cost),('science',scientific),('bundle',original)):
        path, _ = publish(tmp_path/key,unpack(record))
        paths[key] = path
    refs = dict(cost_reference=module.reference(paths['cost']),science_reference=module.reference(paths['science']),
                bundle_reference=module.reference(paths['bundle']))
    binding = module.build_binding(**refs)
    runtime = envelope(dict(protocol_sha256=protocol['sha256'],matrix_sha256=matrix['sha256'],
        ledger_directory=str(new),predecessor=binding,fixture_only=True))
    target = budget.contract_for_matrix(matrix,protocol_sha256=protocol['sha256'],execution_sha256=digest('new execution'),
        runtime_manifest_sha256=runtime['sha256'],approval_sha256=digest('NOT-HUMAN-NEW-APPROVAL'),ledger_directory=new)
    live_checks = []
    def live_check(value):
        live_checks.append(value['sha256'])
        return value  # Explicit synthetic-evidence seam, NOT real closure.
    monkeypatch.setattr(module,'verify_partial_predecessor',live_check)
    return dict(old=old,new=new,paths=paths,refs=refs,binding=binding,runtime=runtime,target=target,
                works=works,cost=cost,science=scientific,checks=live_checks)


def test_exact_once_import_preserves_costs_and_completed_work_without_new_science(fixture):
    with budget.Ledger.create(fixture['new'],fixture['target']) as ledger:
        ledger.import_partial_floor(fixture['runtime'])
        value = ledger.summary()
        assert value['imported_success_count'] == 28
        assert value['charged_ns_by_phase'] == unpack(fixture['cost'])['charged_ns_by_phase']
        assert value['generated_forecasts_reserved'] == 2
        assert value['attempted_work_items'] == 28 and value['unattempted_work_items'] == 1
        assert fixture['checks'] == [fixture['binding']['sha256']]
        assert ledger._state.imported_success == unpack(fixture['science'])['completed_sources']
        assert len((fixture['new']/'events/000000.json').read_bytes()) < 64*1024
        before = ledger.summary()
        with pytest.raises(ValueError,match='healthy empty'):
            ledger.import_partial_floor(fixture['runtime'])
        assert ledger.summary() == before and len(fixture['checks']) == 1
        with pytest.raises(ValueError,match='already attempted'):
            ledger.reserve(fixture['works'][0]['work_id'])


def test_first_actual_pending_dispatch_uses_retained_token_exactly_once_and_keeps_old_charge(fixture):
    with budget.Ledger.create(fixture['new'],fixture['target']) as ledger:
        ledger.import_partial_floor(fixture['runtime'])
        wid = fixture['works'][-1]['work_id']
        charge = ledger.summary()['charged_ns_by_phase']['method_forecasts']
        pending = ledger.reserve(wid)
        value = ledger.summary()
        assert value['generated_forecasts_reserved'] == 2  # Already at full cap; no free/refunded/new token.
        assert value['charged_ns_by_phase']['method_forecasts'] == charge+NS
        assert value['carried_generation_reservations'] == {}
        assert value['consumed_generation_transfers'][wid] == dict(
            predecessor_reservation_sha256=unpack(fixture['cost'])['pending_reservation']['reservation_sha256'],
            successor_reservation_sha256=pending['reservation_sha256'])
        ledger.settle(pending['reservation_sha256'],status='success',elapsed_ns=100_000_000,
            completion_evidence_sha256=digest('fixture completion'),result_sha256=digest('fixture result'),reason='fixture')
        assert ledger.summary()['charged_ns_by_phase']['method_forecasts'] == charge+100_000_000
        with pytest.raises(ValueError,match='already attempted'):
            ledger.reserve(wid)


@pytest.mark.parametrize('consume',[False,True])
def test_cold_replay_reconstructs_import_and_token_history_without_live_check(fixture,monkeypatch,consume):
    with budget.Ledger.create(fixture['new'],fixture['target']) as ledger:
        ledger.import_partial_floor(fixture['runtime'])
        if consume: ledger.reserve(fixture['works'][-1]['work_id'])
        root, tip, state = ledger.root_sha256,ledger.tip,ledger.summary()
    def forbidden(*args,**kwargs): raise AssertionError('pure replay invoked live source inspection')
    monkeypatch.setattr(module,'verify_partial_predecessor',forbidden)
    with budget.Ledger.open(fixture['new'],expected_root_sha256=root,expected_tip=tip) as reopened:
        assert reopened.summary() == state


def test_live_source_failure_precedes_any_event_or_debit(fixture,monkeypatch):
    def fail(*args): raise ValueError('real source reinspection failed')
    monkeypatch.setattr(module,'verify_partial_predecessor',fail)
    with budget.Ledger.create(fixture['new'],fixture['target']) as ledger:
        before = ledger.summary()
        with pytest.raises(ValueError,match='reinspection failed'): ledger.import_partial_floor(fixture['runtime'])
        assert ledger.summary() == before and list((fixture['new']/'events').iterdir()) == []


def test_changed_snapshot_bytes_fail_even_when_semantic_payload_is_equal(fixture):
    fixture['paths']['science'].write_bytes(fixture['paths']['science'].read_bytes()+b'\n')
    with budget.Ledger.create(fixture['new'],fixture['target']) as ledger:
        with pytest.raises(ValueError,match='byte identity'): ledger.import_partial_floor(fixture['runtime'])
        assert ledger.tip['event_count'] == 0 and not fixture['checks']


@pytest.mark.parametrize('key',['execution_sha256','approval_sha256','matrix_sha256','protocol_sha256',
                               'max_generated_forecasts','phase_caps_ns','workloads'])
def test_changed_or_reused_successor_scope_rejected_before_live_inspection(fixture,key):
    target = deepcopy(fixture['target'])
    if key=='execution_sha256': target[key] = unpack(fixture['cost'])[key]
    elif key=='approval_sha256': target[key] = unpack(fixture['cost'])[key]
    elif key in ('matrix_sha256','protocol_sha256'): target[key] = digest('wrong scope')
    elif key=='max_generated_forecasts': target[key] += 1
    elif key=='phase_caps_ns': target[key]['method_forecasts'] += NS
    else: target[key] = target[key][1:]
    with pytest.raises(ValueError): module.partial_floor(fixture['runtime'],target)
    assert not fixture['checks']


@pytest.mark.parametrize('change',[
    lambda p:p.update(pending_dispatched=True),
    lambda p:p.update(pending_requested=True),
    lambda p:p.update(process_tree_closed=False),
    lambda p:p.update(authorizes_execution=True),
    lambda p:p.update(generated_calls_released=1),
    lambda p:p.update(generation_reservations_retained=1),
    lambda p:p['pending_reservation'].update(generated_forecasts=2),
    lambda p:p['pending_reservation'].update(reserved_ns=2*NS),
    lambda p:p.update(charged_total_ns=1),
    lambda p:p['charged_ns_by_phase'].update(method_forecasts=11*NS),
])
def test_rehashed_false_source_claim_cannot_enter_runtime(fixture,tmp_path,change):
    value = deepcopy(unpack(fixture['cost']))
    change(value)
    path, _ = publish(tmp_path/'false-cost',value)
    refs = dict(fixture['refs'],cost_reference=module.reference(path))
    with pytest.raises(ValueError): module.build_binding(**refs)


@pytest.mark.parametrize('key',['fit_bindings','forecast_index','completed_sources'])
def test_no_successful_subset_or_missing_owner_import(fixture,tmp_path,key):
    value = deepcopy(unpack(fixture['science']))
    value[key].pop(next(iter(value[key])))
    path, _ = publish(tmp_path/'false-science',value)
    with pytest.raises(ValueError):
        module.build_binding(**dict(fixture['refs'],science_reference=module.reference(path)))


def test_source_record_loss_breaks_replay_instead_of_resetting_budget(fixture):
    with budget.Ledger.create(fixture['new'],fixture['target']) as ledger:
        ledger.import_partial_floor(fixture['runtime'])
        root = ledger.root_sha256
    fixture['paths']['cost'].unlink()
    with pytest.raises(ValueError): budget.Ledger.open(fixture['new'],expected_root_sha256=root)


def test_late_partial_event_cannot_replace_existing_science_or_credits(fixture):
    with budget.Ledger.create(fixture['new'],fixture['target']) as ledger:
        ledger.reserve(fixture['works'][0]['work_id'])
        before = ledger.summary()
        with pytest.raises(ValueError,match='once-only first'):
            ledger._append('partial_predecessor',dict(runtime_manifest=fixture['runtime']))
        assert ledger.summary() == before
