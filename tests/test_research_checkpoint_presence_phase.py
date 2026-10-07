"""No-frame scheduling cannot borrow budget or grant checkpoint authority."""

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import atomic_write, digest, encode
from tests.test_research_live_checkpoint import prepared


@pytest.mark.parametrize("change", [None, "closed-arm", "recovery-hold"])
def test_absent_response_skips_publication_but_keeps_actual_fuse_and_fresh_budget(tmp_path, monkeypatch, change):
    store, value, registry, adapters, _ = prepared(tmp_path / "no-response", "exact", "no-response")
    observed = []
    def missing(exchange):
        if not observed:
            observed.append(exchange)
            if change == "closed-arm":
                store.append("ARM_CLOSED", {"arm_id": "affine", "reason": "actual no-response close"})
            elif change == "recovery-hold":
                atomic_write(store.path / "recovery-hold.json", encode({"reason": "actual no-response hold"}))
        return False  # Deliberately suppress only the non-authoritative hint.
    def forbidden():
        pytest.fail("absent-response poll entered checkpoint publication")
    monkeypatch.setattr(CheckpointExchange, "response_present", missing)
    monkeypatch.setattr(store, "_checkpoint_publication", forbidden)
    result = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
        value["study_id"], digest(value["cells"][0]), budget=BudgetSpec(3))
    assert len(observed) == 1
    expected = "TIMEOUT" if change is None else "BUDGET_EXHAUSTED"
    assert result["state"] == expected and "checkpoint" not in result, result
    assert not observed[0].accepted and not (observed[0].directory / "checkpoint-ack.json").exists()
    events = store.events()
    assert len([event for event in events if event["event_kind"] == "CHECKPOINT_REQUESTED"]) == 1
    assert not any(event["event_kind"] == "CHECKPOINT_SAVED" for event in events)
    assert len([event for event in events if event["event_kind"] == "WORKER_TREE_STOPPED"]) == 1
    assert [event["payload"]["outcome"] for event in events if event["event_kind"] == "SETTLE"] == [expected]
    balance = BudgetLedger(store).balance("affine")
    assert balance["closed"] and balance["committed_ms"] > 0
    if change is None:
        assert balance["committed_ms"] >= 3000
