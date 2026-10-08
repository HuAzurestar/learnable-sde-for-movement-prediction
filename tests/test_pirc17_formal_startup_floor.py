"""Software-only floor accounting; NOT replacement-run approval or execution."""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_budget as budget, formal_predecessor as predecessor
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, read_json, unpack
from tests.test_pirc17_formal_predecessor import startup, inventory


def held_writer_inventory(directory):
    # Windows correctly denies reading the one locked byte while this NEW
    # ledger's writer is held. Compare immutable records/head, not its lease.
    # The CLOSED historical ledger remains fully byte-hashed by inventory().
    return {p.relative_to(directory).as_posix():file_hash(p) for p in directory.rglob('*')
            if p.is_file() and p.name != 'writer.lock'}


@pytest.fixture
def replacement(startup):
    old = unpack(read_json(startup['directory']/'ledger.json'))
    binding = predecessor.inspect_startup_predecessor(startup['directory'])
    directory = startup['root']/'replacement-ledger'
    runtime = envelope(dict(software_fixture_only=True,protocol_sha256=old['protocol_sha256'],
        matrix_sha256=old['matrix_sha256'],ledger_directory=str(directory),predecessor=binding))
    contract = deepcopy(old)
    contract.update(execution_sha256=digest('REPLACEMENT SOFTWARE EXECUTION'),
        approval_sha256=digest('NOT HUMAN REPLACEMENT APPROVAL'),runtime_manifest_sha256=runtime['sha256'],
        ledger_directory=str(directory))
    return directory,contract,runtime


def test_import_debits_original_categories_before_admission_and_never_refunds_floor(startup,replacement):
    directory,contract,runtime = replacement
    before = inventory(startup['directory'])
    phase = 'input_qualification_and_binding'
    with budget.Ledger.create(directory,contract) as ledger:
        record = ledger.import_startup_floor(runtime)
        assert unpack(record)['type'] == 'startup_predecessor' and unpack(record)['index'] == 0
        s = ledger.summary()
        assert s['charged_ns_by_phase'][phase] == 4_000_000_000
        assert s['measured_ns_by_phase'][phase] == 3_000_000_000
        assert s['conservatively_charged_ns_by_phase'][phase] == s['control_charged_ns_by_phase'][phase] == 1_000_000_000
        assert s['control_observed_ns_by_phase'][phase] == 900_000_000
        assert s['remaining_ns_by_phase'][phase] == 6_000_000_000 and s['remaining_total_ns'] == 26_000_000_000
        assert s['attempted_work_items'] == s['generated_forecasts_reserved'] == 0 and s['halted_reason'] is None
        reservation = ledger.reserve(contract['workloads'][0]['work_id'])
        assert reservation['reserved_ns'] == 6_000_000_000  # Original per-work max was10s.
        ledger.settle(reservation['reservation_sha256'],status='failure',elapsed_ns=2_000_000_000,
            completion_evidence_sha256=digest('NEW SOFTWARE WORK'),result_sha256=None,reason='software fixture')
        assert ledger.summary()['charged_ns_by_phase'][phase] == 6_000_000_000
        assert ledger.summary()['measured_ns_by_phase'][phase] == 5_000_000_000
        assert ledger.summary()['remaining_total_ns'] == 24_000_000_000
        assert ledger.summary()['predecessor_floor']['charged_total_ns'] == 4_000_000_000
        assert contract['phase_caps_ns'][phase] == 10_000_000_000 and contract['total_cap_ns'] == 30_000_000_000
    assert inventory(startup['directory']) == before


def test_control_credit_and_phase_exhaustion_use_balance_after_old_cost(replacement):
    directory,contract,runtime = replacement
    phase = 'input_qualification_and_binding'
    with budget.Ledger.create(directory,contract) as ledger:
        ledger.import_startup_floor(runtime)
        before = ledger.summary()
        with pytest.raises(TimeoutError): ledger.open_control(digest('OVER CAP'),phase=phase,credit_ns=6_000_000_001)
        assert ledger.summary() == before
        ledger.open_control(digest('EXHAUST REMAINING'),phase=phase,credit_ns=6_000_000_000)
        assert ledger.summary()['halted_reason'] == 'phase_cap_reached'
        assert ledger.summary()['charged_ns_by_phase'][phase] == 10_000_000_000
        assert ledger.summary()['remaining_total_ns'] == 20_000_000_000
        with pytest.raises(TimeoutError): ledger.reserve(contract['workloads'][0]['work_id'])


def test_reopen_retains_floor_without_historical_queries_or_duplicate_charge(replacement,monkeypatch):
    directory,contract,runtime = replacement
    with budget.Ledger.create(directory,contract) as ledger:
        ledger.import_startup_floor(runtime)
        before = ledger.summary()
        root,tip = ledger.root_sha256,ledger.tip
    def forbidden(*args,**kwargs): raise AssertionError('budget replay queried old processes')
    monkeypatch.setattr(predecessor,'verify_startup_predecessor',forbidden)
    monkeypatch.setattr(predecessor.native,'_query_job',forbidden)
    with budget.Ledger.open(directory,expected_root_sha256=root,expected_tip=tip) as reopened:
        assert reopened.summary() == before
        with pytest.raises(ValueError,match='once-only'): reopened.import_startup_floor(runtime)
        assert reopened.summary() == before


@pytest.mark.parametrize('first',['floor','reservation','control'])
def test_duplicate_or_late_floor_rejected_without_mutation(replacement,first):
    directory,contract,runtime = replacement
    with budget.Ledger.create(directory,contract) as ledger:
        if first == 'floor': ledger.import_startup_floor(runtime)
        elif first == 'reservation': ledger.reserve(contract['workloads'][0]['work_id'])
        else: ledger.open_control(digest('EARLY CONTROL'),phase='input_qualification_and_binding',credit_ns=1)
        before = ledger.summary(); files = held_writer_inventory(directory)
        with pytest.raises(ValueError,match='once-only'): ledger.import_startup_floor(runtime)
        assert ledger.summary() == before and held_writer_inventory(directory) == files


@pytest.mark.parametrize('fault',['missing','runtime_hash','protocol','matrix','directory','old_execution','old_approval',
    'caps','total','negative','boolean','different_phase','wrong_sum','wrong_root','wrong_head','wrong_terminal','free_cost'])
def test_missing_mismatched_free_or_invalid_floor_is_never_a_credit(replacement,fault):
    directory,contract,runtime = replacement
    value = deepcopy(unpack(runtime)); b = unpack(value['predecessor']); c = unpack(b['cost_snapshot'])
    if fault == 'missing': del value['predecessor']
    elif fault in {'protocol','matrix'}: value[fault+'_sha256'] = digest('WRONG SCOPE')
    elif fault == 'directory': value['ledger_directory'] = str(directory/'different')
    elif fault == 'old_execution': contract['execution_sha256'] = c['execution_sha256']
    elif fault == 'old_approval': contract['approval_sha256'] = c['approval_sha256']
    elif fault == 'caps': c['phase_caps_ns']['input_qualification_and_binding'] += 1
    elif fault == 'total': c['total_cap_ns'] += 1
    elif fault == 'negative': c['measured_ns_by_phase']['input_qualification_and_binding'] = -1
    elif fault == 'boolean': c['measured_ns_by_phase']['input_qualification_and_binding'] = True
    elif fault == 'different_phase': c['control_observed_ns_by_phase']['science'] = 1
    elif fault == 'wrong_sum': c['charged_total_ns'] += 1
    elif fault in {'wrong_root','wrong_head','wrong_terminal'}:
        c[{'wrong_root':'ledger_root_sha256','wrong_head':'head_sha256','wrong_terminal':'terminal_proof_sha256'}[fault]] = digest('CHEAPER HISTORY')
    elif fault == 'free_cost':
        # Coherent-looking invented lower charge still fails actual reinspection.
        c['charged_total_ns'] -= 1
        c['charged_ns_by_phase']['input_qualification_and_binding'] -= 1
        c['measured_ns_by_phase']['input_qualification_and_binding'] -= 1
    b['cost_snapshot'] = envelope(c)
    if fault != 'missing': value['predecessor'] = envelope(b)
    runtime = envelope(value)
    contract['runtime_manifest_sha256'] = digest('WRONG RUNTIME') if fault == 'runtime_hash' else runtime['sha256']
    with budget.Ledger.create(directory,contract) as ledger:
        before = ledger.summary(); files = held_writer_inventory(directory)
        with pytest.raises((ValueError,KeyError)): ledger.import_startup_floor(runtime)
        assert ledger.summary() == before and held_writer_inventory(directory) == files


def test_live_predecessor_or_changed_bound_bytes_stop_before_debit(startup,replacement,monkeypatch):
    directory,contract,runtime = replacement
    with budget.Ledger.create(directory,contract) as ledger:
        before = ledger.summary()
        monkeypatch.setattr(predecessor.native,'_query_job',lambda name: dict(exists=True,accounting=dict(active_processes=1)))
        with pytest.raises(ValueError,match='still active'): ledger.import_startup_floor(runtime)
        assert ledger.summary() == before
        monkeypatch.setattr(predecessor.native,'_query_job',lambda name: dict(exists=False,accounting=None))
        old = startup['directory']/'writer.lock'; old.write_bytes(b'changed fixture lock')
        with pytest.raises(ValueError): ledger.import_startup_floor(runtime)
        assert ledger.summary() == before


def test_durable_floor_survives_failed_head_publication_without_refund(replacement,monkeypatch):
    directory,contract,runtime = replacement
    publish = budget._publish
    def fail_head(path,payload,**kwargs):
        if path.name == 'head.json' and kwargs.get('replace'): raise OSError('SOFTWARE IO FIXTURE')
        return publish(path,payload,**kwargs)
    with budget.Ledger.create(directory,contract) as ledger:
        root = ledger.root_sha256
        monkeypatch.setattr(budget,'_publish',fail_head)
        with pytest.raises(OSError): ledger.import_startup_floor(runtime)
        assert ledger._poisoned
        with pytest.raises(RuntimeError): ledger.reserve(contract['workloads'][0]['work_id'])
    monkeypatch.setattr(budget,'_publish',publish)
    with budget.Ledger.open(directory,expected_root_sha256=root) as reopened:
        assert reopened.summary()['predecessor_floor']['charged_total_ns'] == 4_000_000_000
        assert reopened.summary()['remaining_total_ns'] == 26_000_000_000
        assert reopened.recovered_unanchored_events == 1


def test_rehashed_event_cannot_retarget_runtime_even_with_rehashed_head(replacement):
    directory,contract,runtime = replacement
    with budget.Ledger.create(directory,contract) as ledger:
        ledger.import_startup_floor(runtime); root = ledger.root_sha256
    path = directory/'events/000000.json'; event = deepcopy(unpack(read_json(path)))
    value = unpack(event['row']['runtime_manifest']); value['predecessor'] = None
    event['row']['runtime_manifest'] = envelope(value)
    record = envelope(event); path.write_bytes(canonical(record))
    head = unpack(read_json(directory/'head.json')); head['last_event_sha256'] = record['sha256']
    (directory/'head.json').write_bytes(canonical(envelope(head)))
    with pytest.raises(ValueError): budget.Ledger.open(directory,expected_root_sha256=root)
