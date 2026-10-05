"""An actual in-flight soft signal cannot add ready-response authority scans.

Use the existing separate600-step/6s controls. Original300-step/3s/6s
acceptance and native process observations are not changed or replaced.
"""
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import ResearchError, atomic_write, digest, encode
from tests.test_research_checkpoint_budget_phase import (
    test_ready_response_budget_and_save_use_two_actual_physical_passes as assert_ready_response_work,
)
from tests.test_research_checkpoint_phase_work import live_owner


def test_ready_response_during_actual_request_publication_keeps_two_physical_passes(
        tmp_path, monkeypatch):
    original, published = CheckpointExchange.request, []

    def request(channel):
        result = original(channel)
        published.append(result)
        # Real published request/worker response, while the sending call is
        # still in flight. No frame, clock or checkpoint result is substituted.
        time.sleep(0.1)
        return result

    monkeypatch.setattr(CheckpointExchange, 'request', request)
    assert_ready_response_work(tmp_path, monkeypatch)
    assert len(published) == 1


def test_actual_request_journal_corruption_never_reaches_ack(tmp_path, monkeypatch):
    store, value, runner = live_owner(tmp_path)
    original_append, original_ack = store._append, CheckpointExchange.acknowledge
    corrupted, acknowledgements = [], []

    def append(kind, *args, **kwargs):
        result = original_append(kind, *args, **kwargs)
        if kind == 'CHECKPOINT_REQUESTED' and not corrupted:
            corrupted.append(True)
            (store.path / 'events/0000000000000001.json').write_bytes(b'{}')
        return result

    def acknowledge(channel, artifact):
        acknowledgements.append(artifact)
        return original_ack(channel, artifact)

    monkeypatch.setattr(store, '_append', append)
    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert corrupted == [True] and not acknowledgements
    assert store._read_snapshot() is None


def test_inflight_control_publication_cannot_extend_actual_hard_fuse(tmp_path, monkeypatch):
    store, value, runner = live_owner(tmp_path)
    original_request, original_ack = CheckpointExchange.request, CheckpointExchange.acknowledge
    channels, acknowledgements = [], []

    def request(channel):
        result = original_request(channel)
        channels.append(channel)
        time.sleep(1.35)  # Publication completion crosses the unchanged6s fuse.
        return result

    def acknowledge(channel, artifact):
        acknowledgements.append(artifact)
        return original_ack(channel, artifact)

    monkeypatch.setattr(CheckpointExchange, 'request', request)
    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    result = runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert len(channels) == 1
    assert result['state'] == 'TIMEOUT' and result['exit_code'] != 0, result
    assert 'checkpoint' not in result and not acknowledgements
    assert not (channels[0].directory / 'checkpoint-ack.json').exists()
    balance = BudgetLedger(store).balance('affine')
    assert balance['closed'] and balance['committed_ms'] >= 6000
    assert store._read_snapshot() is None


@pytest.mark.parametrize('change', ['closed-arm', 'recovery-hold'])
def test_actual_inflight_signal_cannot_borrow_budget_before_current_closure(
        tmp_path, monkeypatch, change):
    store, value, runner = live_owner(tmp_path)
    original_request, original_ack = CheckpointExchange.request, CheckpointExchange.acknowledge
    changed, acknowledgements = [], []

    def request(channel):
        result = original_request(channel)
        # A distinct actual writer changes authority after the control frame;
        # the owner must not hold an authority lock while waiting on the signal.
        if change == 'closed-arm':
            store.append('ARM_CLOSED', {'arm_id': 'affine', 'reason': 'actual inflight close'})
        else:
            atomic_write(store.path / 'recovery-hold.json', encode({'reason': 'actual inflight hold'}))
        changed.append(channel)
        return result

    def acknowledge(channel, artifact):
        acknowledgements.append(artifact)
        return original_ack(channel, artifact)

    monkeypatch.setattr(CheckpointExchange, 'request', request)
    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    result = runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert len(changed) == 1
    assert result['state'] == 'BUDGET_EXHAUSTED' and 'checkpoint' not in result, result
    assert not acknowledgements and not (changed[0].directory / 'checkpoint-ack.json').exists()
    assert BudgetLedger(store).balance('affine')['closed']
    assert store._read_snapshot() is None
