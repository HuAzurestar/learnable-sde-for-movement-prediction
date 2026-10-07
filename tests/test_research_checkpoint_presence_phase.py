"""No-frame scheduling cannot borrow budget or grant checkpoint authority."""

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_control import CheckpointExchange, ControlError
from infrastructure.research_store import ResearchError, atomic_write, digest, encode
from tests.test_research_checkpoint_phase_work import live_owner
from tests.test_research_live_checkpoint import prepared


@pytest.mark.parametrize("change", [None, "closed-arm", "recovery-hold"])
def test_no_ready_response_skips_publication_but_keeps_actual_fuse_and_fresh_budget(tmp_path, monkeypatch, change):
    store, value, registry, adapters, _ = prepared(tmp_path / "no-response", "exact", "no-response")
    observed = []
    def missing(exchange):
        if not observed:
            observed.append(exchange)
            if change == "closed-arm":
                store.append("ARM_CLOSED", {"arm_id": "affine", "reason": "actual no-response close"})
            elif change == "recovery-hold":
                atomic_write(store.path / "recovery-hold.json", encode({"reason": "actual no-response hold"}))
        return None  # Deliberately suppress only the incoming control message.
    def forbidden():
        pytest.fail("absent-response poll entered checkpoint publication")
    monkeypatch.setattr(CheckpointExchange, "response", missing)
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


def test_invalid_incoming_frame_retains_request_cost_and_native_stop_without_publication(tmp_path, monkeypatch):
    # Existing600-step/6second adversarial fixture; original300-step/3second
    # acceptance controls and all actual deadlines remain unchanged.
    store, value, runner = live_owner(tmp_path)
    observed = []
    def invalid(exchange):
        observed.append(exchange)
        raise ControlError("synthetic invalid incoming frame")
    def forbidden():
        pytest.fail("invalid incoming frame entered checkpoint publication")
    monkeypatch.setattr(CheckpointExchange, "response", invalid)
    monkeypatch.setattr(store, "_checkpoint_publication", forbidden)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))
    assert len(observed) == 1 and not observed[0].accepted
    events = store.events()
    assert len([event for event in events if event["event_kind"] == "CHECKPOINT_REQUESTED"]) == 1
    assert len([event for event in events if event["event_kind"] == "WORKER_TREE_STOPPED"]) == 1
    assert not any(event["event_kind"] == "CHECKPOINT_SAVED" for event in events)
    attempt, = store.attempts().values()
    assert attempt["state"] == "INTERRUPTED" and attempt["error_code"] == "SUPERVISOR_ERROR"
    assert BudgetLedger(store).balance("affine")["committed_ms"] > 0
