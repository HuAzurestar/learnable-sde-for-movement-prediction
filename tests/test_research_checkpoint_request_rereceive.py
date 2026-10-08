"""Fresh IPC after request journalling, never a borrowed save approval.

Use the existing 600-step/6-second adversarial worker for ordering controls.
The original 300-step/3-second acceptance and native stops stay unchanged.
"""

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from infrastructure.research_control import CheckpointExchange, ControlError
from infrastructure.research_store import ResearchError, atomic_write, digest, encode
from tests.test_research_checkpoint_phase_work import live_owner


@pytest.mark.parametrize('change', ['ready', 'invalid', 'closed-arm', 'recovery-hold'])
def test_fresh_response_after_durable_request_precedes_another_budget_poll(
        tmp_path, monkeypatch, change):
    store, value, runner = live_owner(tmp_path)
    original_response = CheckpointExchange.response
    original_append, original_balance = store.append, BudgetLedger.balance
    trace, exchanges, injected = [], [], []

    def response(exchange):
        exchanges.append(exchange)
        if len(exchanges) == 1:
            trace.append('first-missing')
            return None  # Suppress this IPC poll only, not the actual worker.
        trace.append('response')
        if not injected:
            injected.append(True)
            if change == 'invalid':
                raise ControlError('invalid fresh response after request journal')
            if change == 'closed-arm':
                store.append('ARM_CLOSED', {'arm_id': 'affine', 'reason': 'actual rereceive closure'})
            elif change == 'recovery-hold':
                atomic_write(store.path / 'recovery-hold.json', encode({'reason': 'actual rereceive hold'}))
        return original_response(exchange)

    def append(kind, *args, **kwargs):
        result = original_append(kind, *args, **kwargs)
        if kind == 'CHECKPOINT_REQUESTED':
            trace.append('request-durable')
        return result

    def balance(ledger, *args, **kwargs):
        if ledger.store is store and 'request-durable' in trace:
            trace.append('budget')
        return original_balance(ledger, *args, **kwargs)

    monkeypatch.setattr(CheckpointExchange, 'response', response)
    monkeypatch.setattr(store, 'append', append)
    monkeypatch.setattr(BudgetLedger, 'balance', balance)
    if change == 'invalid':
        with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
            runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    else:
        result = runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
        expected = 'FAILED' if change == 'ready' else 'BUDGET_EXHAUSTED'
        assert result['state'] == expected, result
        if change == 'ready':
            assert store.attempts()[result['attempt_id']]['error_code'] == 'CHECKPOINT_SAVED'
            assert 'checkpoint' in result
        else:
            assert 'checkpoint' not in result

    # Check ordering only after native stop/settlement: a failing assertion must
    # not interrupt the actual worker or manufacture a successful native exit.
    assert trace[trace.index('request-durable') + 1] == 'response', trace
    assert injected == [True]
    events = store.events()
    assert len([event for event in events if event['event_kind'] == 'CHECKPOINT_REQUESTED']) == 1
    assert len([event for event in events if event['event_kind'] == 'WORKER_TREE_STOPPED']) == 1
    assert len([event for event in events if event['event_kind'] == 'SETTLE']) == 1
    assert BudgetLedger(store).balance('affine')['committed_ms'] > 0
    if change != 'ready':
        assert not any(exchange.accepted for exchange in exchanges)
        assert not any(event['event_kind'] == 'CHECKPOINT_SAVED' for event in events)
