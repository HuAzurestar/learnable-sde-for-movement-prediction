"""A run trace and its cumulative cost must use one authority snapshot."""

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError
from tests.test_research_comparison_query_cost import listing_store


def registered(tmp_path):
    store, value = listing_store(tmp_path)
    run = store.register_run(value['study_id'], value['cells'][0])
    attempt = store.new_attempt(run)
    return store, run, attempt


def test_valid_run_budget_and_sources_use_the_same_reservation(tmp_path):
    store, run, attempt = registered(tmp_path)
    reservation = BudgetLedger(store).reserve(attempt, BudgetSpec(1))
    result = ResearchQuery(store, 'ui').run(run)
    assert result['budget']['committed_ms'] == 1000
    assert len(result['budget']['sources']) == 1
    assert result['budget']['sources'][0]['reservation_id'] == reservation['reservation_id']
    assert result['budget']['sources'][0]['reserved_ms'] == 1000


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_run_budget_update_during_metadata_assembly_rejects_mixed_trace(tmp_path, monkeypatch, phase):
    store, run, attempt = registered(tmp_path)
    original = BudgetLedger.balance
    changed = []

    def update_balance(ledger, arm):
        if phase == 'before' and not changed:
            changed.append(True)
            ledger.reserve(attempt, BudgetSpec(1))
        result = original(ledger, arm)
        if phase == 'after' and not changed:
            changed.append(True)
            ledger.reserve(attempt, BudgetSpec(1))
        return result

    monkeypatch.setattr(BudgetLedger, 'balance', update_balance)
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        ResearchQuery(store, 'ui').run(run)
    assert changed
    assert BudgetLedger(store).balance('affine')['committed_ms'] == 1000


def test_run_recovery_hold_during_metadata_assembly_rejects_stale_closed_status(tmp_path, monkeypatch):
    store, run, _ = registered(tmp_path)
    original = BudgetLedger.balance
    changed = []

    def hold_after_balance(ledger, arm):
        result = original(ledger, arm)
        if not changed:
            changed.append(True)
            store.quarantine_tail('synthetic run-detail recovery hold')
        return result

    monkeypatch.setattr(BudgetLedger, 'balance', hold_after_balance)
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        ResearchQuery(store, 'ui').run(run)
    assert changed
    assert (store.path / 'recovery-hold.json').is_file()
    assert BudgetLedger(store).balance('affine')['closed'] is True
