"""First durable request shares its owned save phase, not a budget decision.

Retain the existing independently registered600-step/6s ordering control with
its one explicitly suppressed first IPC poll. No clock/native result/authority
is replaced. Original300-step/3s/6s acceptance remains untouched.
"""
from application.research_budget import BudgetSpec
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import ResearchStore
from tests.test_research_checkpoint_request_rereceive import (
    test_fresh_response_after_durable_request_precedes_another_budget_poll as original_control,
)


def test_first_missing_response_request_uses_the_original_owned_save_phase(tmp_path, monkeypatch):
    original_append = ResearchStore._append
    original_ack = CheckpointExchange.acknowledge
    requests, saved, acknowledged = [], [], []

    def append(store, kind, *args, **kwargs):
        result = original_append(store, kind, *args, **kwargs)
        if kind == "CHECKPOINT_REQUESTED":
            requests.append((store, store._read_snapshot(),
                             getattr(store._read_scope, "checkpoint_head", None)))
        elif kind == "CHECKPOINT_SAVED":
            saved.append((store, store._read_snapshot(),
                          getattr(store._read_scope, "checkpoint_head", None)))
        return result

    def acknowledge(exchange, artifact_id):
        # Actual durable request and save must have completed their original
        # final physical pass before the real ACK, not inside a retained scope.
        assert requests and saved
        assert requests[0][0]._read_snapshot() is None
        acknowledged.append(artifact_id)
        return original_ack(exchange, artifact_id)

    monkeypatch.setattr(ResearchStore, "_append", append)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    original_control(tmp_path, monkeypatch, "ready")

    # Assert only AFTER the original native stop/settlement and its ordering,
    # costs/one-request checks; never abort the worker to manufacture success.
    assert len(requests) == len(saved) == len(acknowledged) == 1
    request, checkpoint = requests[0], saved[0]
    assert request[1] is not None and request[2] is not None, (
        "first durable REQUEST made a separate prefix read before the actual save phase")
    assert request[1] is checkpoint[1] and request[2] is checkpoint[2], (
        "first REQUEST and actual SAVE did not share one owned physical phase")
    store = request[0]
    events = store.events()
    assert [event["payload"]["artifact_id"] for event in events
            if event["event_kind"] == "CHECKPOINT_SAVED"] == acknowledged
    assert store._read_snapshot() is None
