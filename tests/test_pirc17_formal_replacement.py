"""Mandatory v3 post-read wiring/denial; no human or scientific qualification."""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_budget as budget,formal_entrypoint as entry,formal_session as native
from experiments.pirc17 import formal_environment,formal_worker,formal_predecessor
from experiments.pirc17.protocol_core import canonical,digest,envelope,file_hash,read_json,unpack
from tests.test_pirc17_formal_entrypoint import bundle_for,scientific_contract,actual_environment,software_launch
from tests.pirc17_predecessor_fixture import permit_software_eligibility,software_input_pins,software_frozen_metadata


def scoped_contract(record,auth,directory):
    b = unpack(record)
    return budget.contract_for_matrix(b['matrix'],protocol_sha256=b['protocol']['sha256'],
        execution_sha256=b['execution']['sha256'],runtime_manifest_sha256=b['runtime']['sha256'],
        approval_sha256=auth['approval_sha256'],ledger_directory=directory)


@pytest.mark.parametrize('fault',['missing','legacy','free_total','free_columns','new_ledger','new_root','new_head','new_terminal'])
def test_rehashed_runtime_cannot_drop_or_change_registered_floor(tmp_path,scientific_contract,actual_environment,fault):
    _,record,_ = bundle_for(tmp_path,scientific_contract,actual_environment)
    b = unpack(record); r = deepcopy(unpack(b['runtime']))
    if fault == 'missing': del r['predecessor']
    elif fault == 'legacy': r['schema_version'] = formal_predecessor.HISTORICAL_ENTRY+'-runtime'; del r['predecessor']
    else:
        pred = unpack(r['predecessor']); cost = unpack(pred['cost_snapshot'])
        if fault == 'free_total': cost['charged_total_ns'] = 0
        elif fault == 'free_columns':
            for name in ('charged_ns_by_phase','measured_ns_by_phase','conservatively_charged_ns_by_phase',
                         'control_charged_ns_by_phase','control_observed_ns_by_phase'):
                cost[name] = dict.fromkeys(cost[name],0)
            cost['charged_total_ns'] = 0
        elif fault == 'new_ledger': cost['ledger_directory'] = r['ledger_directory']; pred['ledger_directory'] = r['ledger_directory']
        else: cost[{'new_root':'ledger_root_sha256','new_head':'head_sha256','new_terminal':'terminal_proof_sha256'}[fault]] = digest('CHEAPER HISTORY')
        pred['cost_snapshot'] = envelope(cost); r['predecessor'] = envelope(pred)
    with pytest.raises(ValueError): entry.validate_runtime_binding(envelope(r),b['protocol'],b['matrix'])
    assert not (tmp_path/'ledger').exists()


@pytest.mark.parametrize('version',['pirc17-concrete-formal-entrypoint-v1','pirc17-concrete-formal-entrypoint-v2'])
def test_legacy_bundle_is_readable_metadata_but_never_executable(tmp_path,scientific_contract,actual_environment,version):
    _,record,_ = bundle_for(tmp_path,scientific_contract,actual_environment)
    b = deepcopy(unpack(record)); b['schema_version'] = version+'-bundle'
    with pytest.raises(ValueError,match='version differs'): entry.validate_bundle(envelope(b))


def test_new_runtime_cannot_use_the_old_zero_read_cheaper_predecessor(tmp_path,scientific_contract,actual_environment):
    from tests.pirc17_predecessor_fixture import software_predecessor
    protocol,matrix=scientific_contract
    with pytest.raises(ValueError):
        entry.runtime_binding(protocol,matrix,actual_environment,ledger_directory=tmp_path/'ledger',
            input_paths={k:tmp_path/k for k in entry.INPUT_PATHS},
            predecessor=software_predecessor(protocol,matrix,directory=tmp_path/'CHEAPER-ABSENT-HISTORY'))
    assert not (tmp_path/'ledger').exists()


def test_prepare_missing_registered_predecessor_never_configures_environment_or_data(tmp_path,monkeypatch):
    def forbidden(*a,**kw): pytest.fail('environment before verified historical eligibility')
    monkeypatch.setattr(formal_environment,'ConfiguredRuntime',forbidden)
    with pytest.raises(FileNotFoundError):
        entry.prepare(output_directory=tmp_path/'bundle',ledger_directory=tmp_path/'ledger',
            predecessor_ledger=tmp_path/'ABSENT',input_paths={k:tmp_path/k for k in entry.INPUT_PATHS})
    assert list(tmp_path.iterdir()) == []


def test_parent_skipped_debit_stops_before_controller_and_retains_cumulative_claim(
        tmp_path,scientific_contract,actual_environment,monkeypatch):
    kwargs,clock,seen = software_launch(tmp_path,scientific_contract,actual_environment,monkeypatch)
    monkeypatch.setattr(budget.Ledger,'import_startup_floor',lambda *a: None)  # Explicit injected implementation fault.
    with pytest.raises(ValueError,match='bounded regular'): entry.run(**kwargs)
    assert 'controller' not in seen
    terminal = unpack(read_json(entry.launch_directory(tmp_path/'ledger')/'terminal.json'))
    assert terminal['predecessor_charge_location'] == 'claim' and terminal['predecessor_floor_transferred_to_ledger'] is False
    assert terminal['cumulative_charge_lower_bound_ns'] >= 3600*budget.NANOSECONDS
    assert unpack(read_json(tmp_path/'ledger/head.json'))['event_count'] == 0
    assert not (tmp_path/'ledger/access').exists()


def test_floor_io_is_additional_new_startup_time_not_hidden_by_old_charge(
        tmp_path,scientific_contract,actual_environment,monkeypatch):
    kwargs,clock,seen = software_launch(tmp_path,scientific_contract,actual_environment,monkeypatch)
    publish = budget._publish
    def delayed(path,payload,**kw):
        record = publish(path,payload,**kw)
        if payload.get('type') == 'startup_predecessor': clock.advance(7*budget.NANOSECONDS)
        return record
    monkeypatch.setattr(budget,'_publish',delayed)
    entry.run(**kwargs)
    s = seen['controller'].ledger.summary()
    assert s['predecessor_floor']['charged_total_ns'] == 570_736_000_000
    assert s['control_charged_ns_by_phase'][entry.PHASE_ORDER[0]] == 34_532_000_000+12*budget.NANOSECONDS


@pytest.mark.parametrize('fault',['no_debit','removed','rehashed_wrong_runtime','head_zero','orphan_debit','changed_root','floor_only_rollback'])
def test_worker_floor_fault_stops_before_environment_handler_or_runtime_files(
        tmp_path,scientific_contract,actual_environment,monkeypatch,fault):
    path,record,auth = bundle_for(tmp_path,scientific_contract,actual_environment)
    b = unpack(record); directory = tmp_path/'ledger'
    contract = scoped_contract(record,auth,directory)
    permit_software_eligibility(monkeypatch)
    monkeypatch.setattr(entry,'_approval',lambda *a: None)  # NOT human authority.
    def forbidden(*a,**kw): pytest.fail('environment/science before anchored initial debit')
    monkeypatch.setattr(formal_environment,'ConfiguredRuntime',forbidden)
    monkeypatch.setattr(formal_worker,'FormalWorker',forbidden)
    with budget.Ledger.create(directory,contract) as ledger:
        if fault != 'no_debit':
            ledger.import_startup_floor(b['runtime'])
            floor_head = read_json(directory/'head.json')
        ledger.reserve(contract['workloads'][0]['work_id'])
        first = directory/'events/000000.json'
        if fault == 'removed': first.unlink()
        elif fault == 'rehashed_wrong_runtime':
            value = deepcopy(unpack(read_json(first))); value['row']['runtime_manifest'] = envelope(dict(NOT_A_RUNTIME=True))
            first.write_bytes(canonical(envelope(value)))
        elif fault in {'head_zero','orphan_debit'}:
            (directory/'head.json').write_bytes(canonical(envelope(dict(root_sha256=ledger.root_sha256,event_count=0,last_event_sha256=ledger.root_sha256))))
        elif fault == 'floor_only_rollback':
            (directory/'head.json').write_bytes(canonical(floor_head))
        elif fault == 'changed_root':
            root = deepcopy(unpack(read_json(directory/'ledger.json'))); root['approval_sha256'] = digest('WRONG ACTUAL ROOT')
            (directory/'ledger.json').write_bytes(canonical(envelope(root)))
        session_dir = directory/'session'; session_dir.mkdir()
        session = dict(directory=str(session_dir),ledger_directory=str(directory),ledger_root_sha256=ledger.root_sha256,
            worker_command=entry.worker_command(path,record['sha256'],auth,b['runtime']))
        with pytest.raises(ValueError): entry.RuntimeWorker(session,contract,bundle_path=path,bundle_sha256=record['sha256'],authority=auth)
        assert not (session_dir/'runtime').exists() and not (directory/'access').exists()


def test_worker_saved_eligible_flag_does_not_replace_real_historical_verification(
        tmp_path,scientific_contract,actual_environment,monkeypatch):
    path,record,auth = bundle_for(tmp_path,scientific_contract,actual_environment)
    b = unpack(record); directory = tmp_path/'ledger'; contract = scoped_contract(record,auth,directory)
    monkeypatch.setattr(entry,'_approval',lambda *a: None)  # Deliberately isolate the next REQUIRED guard.
    def forbidden(*a,**kw): pytest.fail('runtime/science before real predecessor eligibility')
    monkeypatch.setattr(formal_environment,'ConfiguredRuntime',forbidden)
    monkeypatch.setattr(formal_worker,'FormalWorker',forbidden)
    with budget.Ledger.create(directory,contract) as ledger:
        session_dir = directory/'session'; session_dir.mkdir()
        session = dict(directory=str(session_dir),ledger_directory=str(directory),ledger_root_sha256=ledger.root_sha256,
            worker_command=entry.worker_command(path,record['sha256'],auth,b['runtime']))
        with pytest.raises(FileNotFoundError):
            entry.RuntimeWorker(session,contract,bundle_path=path,bundle_sha256=record['sha256'],authority=auth)
        assert not (session_dir/'runtime').exists() and not (directory/'access').exists()


def test_parent_callback_missing_debit_stops_before_environment_or_scientific_results(
        tmp_path,scientific_contract,actual_environment,monkeypatch):
    from experiments.pirc17 import formal_results
    path,record,auth = bundle_for(tmp_path,scientific_contract,actual_environment)
    b = unpack(record); directory = tmp_path/'ledger'; contract = scoped_contract(record,auth,directory)
    monkeypatch.setattr(entry,'_approval',lambda *a: None)
    permit_software_eligibility(monkeypatch)
    def forbidden(*a,**kw): pytest.fail('callback environment/science before initial debit')
    monkeypatch.setattr(formal_environment,'ConfiguredRuntime',forbidden)
    monkeypatch.setattr(formal_results,'ScientificResults',forbidden)
    with budget.Ledger.create(directory,contract) as ledger:
        callbacks = entry.RuntimeCallbacks(ledger,bundle=record,authority=auth)
        callbacks.session = native.OwnedSession(directory/'session',ledger,
            entry.worker_command(path,record['sha256'],auth,b['runtime']))
        work = contract['workloads'][0]; ledger.reserve(work['work_id'])
        with pytest.raises(ValueError,match='first predecessor debit'): callbacks.authorize(work)
        assert callbacks.failed and callbacks.results is None and callbacks.environment is None
        assert not (directory/'access').exists()
