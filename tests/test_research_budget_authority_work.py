"""Real ledger phases bound physical scans, not permissions or remaining funds."""
import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_queue import ResearchQueue
from infrastructure.research_store import ResearchError
from tests.test_research_budget import registered


@pytest.mark.parametrize("phase", ["reserve", "settle", "recover", "claim"])
def test_atomic_budget_phase_verifies_one_initial_and_one_final_chain(tmp_path, monkeypatch, phase):
    store, attempts = registered(tmp_path)
    ledger = BudgetLedger(store)
    if phase != "reserve":
        reservation = ledger.reserve(attempts[0], BudgetSpec(1))
    if phase == "claim":
        queue = ResearchQueue(store)
        queue.enqueue(reservation)
    original, reads = store._json, []
    def read(path):
        if path.parent == store.path / "events":
            reads.append(path)
        return original(path)
    monkeypatch.setattr(store, "_json", read)
    if phase == "reserve":
        result = ledger.reserve(attempts[0], BudgetSpec(1))
        assert result["reserved_ms"] == 1000 and not result["settled"]
    elif phase == "settle":
        result = ledger.settle(reservation["reservation_id"], 100, outcome="SUCCEEDED")
        assert result["charged_ms"] == 100 and result["settled"]
    elif phase == "recover":
        result = ledger.recover_unknown(reservation["reservation_id"])
        assert result["charged_ms"] == 1000 and result["settled"]
    else:
        assert queue.claim(reservation["reservation_id"])["slot"] == 0
    physical_reads = len(reads)
    final_events = store.events()
    assert physical_reads > 0, "no real authority event files were read"
    assert physical_reads <= 2 * len(final_events), (
        f"{phase} repeatedly reread one locked chain: {physical_reads} real event reads for {len(final_events)} events")


def test_reservation_cannot_return_after_authority_corruption_at_publication(tmp_path, monkeypatch):
    store, attempts = registered(tmp_path)
    original, corrupted = store._append, []
    def append(kind, *args, **kwargs):
        result = original(kind, *args, **kwargs)
        if kind == "RESERVE":
            corrupted.append(True)
            (store.path / "events/0000000000000001.json").write_bytes(b"{}")
        return result
    monkeypatch.setattr(store, "_append", append)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        BudgetLedger(store).reserve(attempts[0], BudgetSpec(1))
    assert corrupted == [True]
    assert store._read_snapshot() is None
