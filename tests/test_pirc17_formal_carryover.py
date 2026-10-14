"""Synthetic read-only budget fixtures, not formal launch or process evidence."""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_budget as budget, formal_carryover as module
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, read_json, unpack


@pytest.fixture
def closed(tmp_path):
    directory = tmp_path/'ledger'
    contract = dict(schema_version=budget.VERSION, protocol_sha256=digest('protocol-fixture'),
        execution_sha256=digest('execution-fixture'), matrix_sha256=digest('matrix-fixture'),
        runtime_manifest_sha256=digest('runtime-fixture'), approval_sha256=digest('NOT-HUMAN-APPROVAL'),
        ledger_directory=str(directory.resolve()), phase_caps_ns={'input':1000,'science':1000},
        total_cap_ns=2000, max_generated_forecasts=1, max_attempts_per_item=1,
        workloads=[dict(work_id=digest(0),phase='input',max_active_ns=700,generated_forecasts=0),
                   dict(work_id=digest(1),phase='science',max_active_ns=500,generated_forecasts=1)])
    terminal = digest('SOFTWARE-TERMINAL-PROOF-NOT-NATIVE-CLOSURE')
    with budget.Ledger.create(directory,contract) as ledger:
        ledger.open_control(digest('control'),phase='input',credit_ns=100)
        reservation = ledger.reserve(digest(0))
        ledger.settle(reservation['reservation_sha256'],status='failure',elapsed_ns=250,
            completion_evidence_sha256=digest('fixture-failure'),result_sha256=None,reason='software fixture')
        ledger.halt('supervision_error',digest('fixture-failure'))
        ledger.close_control(digest('control'),observed_ns=80,evidence_sha256=digest('fixture-phase'),reason='stopped')
        ledger.finish_control_tail(digest('control'),observed_ns=90,evidence_sha256=terminal)
        root = ledger.root_sha256
    return directory, dict(expected_root_sha256=root,
        expected_head_sha256=read_json(directory/'head.json')['sha256'],expected_terminal_proof_sha256=terminal)


def inventory(directory):
    return {p.relative_to(directory).as_posix():file_hash(p) for p in directory.rglob('*') if p.is_file()}


def test_actual_journal_replay_preserves_categories_and_never_opens_writer(closed,monkeypatch):
    directory,pins = closed
    before = inventory(directory)
    def forbidden(*args,**kwargs):
        raise AssertionError('read-only inspection opened a ledger writer')
    monkeypatch.setattr(budget.Ledger,'open',forbidden)
    value = unpack(module.inspect_closed_ledger(directory,**pins))
    assert value['charged_ns_by_phase'] == {'input':350,'science':0}
    assert value['measured_ns_by_phase'] == {'input':250,'science':0}
    assert value['conservatively_charged_ns_by_phase'] == {'input':100,'science':0}
    assert value['control_charged_ns_by_phase'] == {'input':100,'science':0}
    assert value['control_observed_ns_by_phase'] == {'input':90,'science':0}
    assert value['work_dispositions'] == {digest(0):'failure'}
    assert value['charged_total_ns'] == 350 and value['generated_forecasts_reserved'] == 0
    assert value['read_only'] is True and value['authorizes_execution'] is False
    assert inventory(directory) == before


@pytest.mark.parametrize('key',['expected_root_sha256','expected_head_sha256','expected_terminal_proof_sha256'])
def test_each_independently_pinned_identity_is_required(closed,key):
    directory,pins = closed
    pins = dict(pins,**{key:digest('different pinned identity')})
    with pytest.raises(ValueError):
        module.inspect_closed_ledger(directory,**pins)


@pytest.mark.parametrize('extra',['000006.json','.interrupted.pending','foreign.txt'])
def test_unanchored_or_partial_events_are_not_recovered_or_deleted(closed,extra):
    directory,pins = closed
    path = directory/'events'/extra
    path.write_bytes(b'fixture residue')
    before = inventory(directory)
    with pytest.raises(ValueError,match='missing/extra/unanchored/partial'):
        module.inspect_closed_ledger(directory,**pins)
    assert inventory(directory) == before


def test_deleted_suffix_cannot_masquerade_as_closed_state(closed):
    directory,pins = closed
    (directory/'events/000005.json').unlink()
    with pytest.raises(ValueError,match='missing/extra'):
        module.inspect_closed_ledger(directory,**pins)


def test_rehashed_event_from_another_root_is_rejected(closed):
    directory,pins = closed
    path = directory/'events/000000.json'
    payload = deepcopy(unpack(read_json(path)))
    payload['root_sha256'] = digest('wrong root')
    path.write_bytes(canonical(envelope(payload)))
    with pytest.raises(ValueError,match='event root'):
        module.inspect_closed_ledger(directory,**pins)


def test_bound_limits_reject_oversized_event_before_decode(closed,monkeypatch):
    directory,pins = closed
    path = directory/'events/000000.json'
    path.write_bytes(b'x'*5000)
    monkeypatch.setattr(module,'MAX_RECORD_BYTES',4096)
    with pytest.raises(ValueError,match='bounded regular'):
        module.inspect_closed_ledger(directory,**pins)


@pytest.mark.parametrize('retained_count',[1,2,4,5])
def test_pending_or_missing_terminal_tail_is_not_a_closed_snapshot(closed,retained_count):
    directory,pins = closed
    for index in range(retained_count,6):
        (directory/f'events/{index:06d}.json').unlink()
    head = unpack(read_json(directory/'head.json'))
    head['event_count'] = retained_count
    head['last_event_sha256'] = read_json(directory/f'events/{retained_count-1:06d}.json')['sha256']
    record = envelope(head)
    (directory/'head.json').write_bytes(canonical(record))
    pins['expected_head_sha256'] = record['sha256']
    with pytest.raises(ValueError,match='accounting is incomplete'):
        module.inspect_closed_ledger(directory,**pins)


def test_event_bound_is_checked_before_enumeration(closed,monkeypatch):
    directory,pins = closed
    monkeypatch.setattr(module,'MAX_EVENTS',5)
    with pytest.raises(ValueError,match='bounded nonempty'):
        module.inspect_closed_ledger(directory,**pins)


def test_byte_identity_is_bound_to_the_same_decode_and_changes_are_rejected(closed,monkeypatch):
    directory,pins = closed
    read = module._read_bound
    changed = False
    def replace_after_read(path,max_bytes):
        nonlocal changed
        record, checksum = read(path,max_bytes)
        if path.name == '000000.json' and not changed:
            # Same payload/digest but different file bytes must still be caught.
            path.write_bytes(canonical(record)+b'\n\n')
            changed = True
        return record,checksum
    monkeypatch.setattr(module,'_read_bound',replace_after_read)
    with pytest.raises(ValueError,match='bytes changed'):
        module.inspect_closed_ledger(directory,**pins)


def test_relative_predecessor_location_is_not_inferred(closed):
    _,pins = closed
    with pytest.raises(ValueError,match='absolute predecessor'):
        module.inspect_closed_ledger('ledger',**pins)
