"""REQUEST is a read scope; only actual ready IPC starts checkpoint publication.

The initial ce85 control overconstrained REQUEST to checkpoint_head presence.
This stricter separation retains shared physical-prefix identity and requires
that REQUEST be durable WITHOUT granting checkpoint publication. Original
missing/invalid/closed/hold/native300-step tests are untouched.
"""
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import ResearchStore
from tests.test_research_checkpoint_request_rereceive import (
    test_fresh_response_after_durable_request_precedes_another_budget_poll as original_control,
)


def test_first_missing_response_request_uses_the_original_owned_read_scope(tmp_path, monkeypatch):
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
        assert requests and saved
        assert requests[0][0]._read_snapshot() is None
        acknowledged.append(artifact_id)
        return original_ack(exchange, artifact_id)

    monkeypatch.setattr(ResearchStore, "_append", append)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    original_control(tmp_path, monkeypatch, "ready")

    # AFTER actual native stop, real funded settlement and original order/
    # refusal checks: do not manufacture any native result to satisfy this.
    assert len(requests) == len(saved) == len(acknowledged) == 1
    request, checkpoint = requests[0], saved[0]
    assert request[1] is not None, "first REQUEST lost its owned physical read prefix"
    assert request[1] is checkpoint[1], "REQUEST/SAVE repeated independent physical prefixes"
    assert request[2] is None, "missing-response REQUEST incorrectly started checkpoint publication"
    assert checkpoint[2] is not None, "actual ready SAVE lost its checkpoint publication scope"
    store = request[0]
    events = store.events()
    assert [event["payload"]["artifact_id"] for event in events
            if event["event_kind"] == "CHECKPOINT_SAVED"] == acknowledged
    assert store._read_snapshot() is None
