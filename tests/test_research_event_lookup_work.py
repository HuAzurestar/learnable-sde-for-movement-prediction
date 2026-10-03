"""Measure real logical prefix work separately from physical integrity reads."""

import pytest

import infrastructure.research_store as store_module
from infrastructure.research_store import ResearchError
from application.research_data import EvaluationExposureLedger
from tests.test_research_artifact_read_bounds import published
from tests.test_research_input_gates import final_eval, freeze_evidence, authorize


def fill(store, count):
    with store._read_transaction():
        for number in range(count):
            store.append("LOOKUP_FIXTURE", {"number": number}, "lookup-fixture-" + str(number))


@pytest.mark.parametrize("count", [16, 64])
def test_manifest_and_replay_lookup_do_not_reenumerate_owned_prefix(tmp_path, monkeypatch, count):
    store, artifact, grant = published(tmp_path)
    fill(store, count)
    original_events, original_json = store._events, store._json
    active, visits, manifests = [], [], []

    class ObservedPrefix(list):
        def __iter__(self):
            for event in super().__iter__():
                if active:
                    visits.append(event["sequence"])
                yield event  # Actual verified event, unchanged.

    def events():
        result = original_events()
        return ObservedPrefix(result) if store._read_snapshot() is None else result

    def read(path):
        if active and path.parent == store.path / "manifests":
            manifests.append(path)
        return original_json(path)

    monkeypatch.setattr(store, "_events", events)
    monkeypatch.setattr(store, "_json", read)
    with store._read_transaction():
        active.append(True)
        for _ in range(24):
            assert store.manifest("artifact-" + artifact["artifact_id"]) == artifact
            assert store.authorization(grant["authorization_id"]) == grant
            replay = store.append("LOOKUP_FIXTURE", {"number": 0}, "lookup-fixture-0")
            replay["payload"]["number"] = -1
        with pytest.raises(ResearchError, match="MISSING_INPUT"):
            store.manifest("not-published")
        active.clear()
    assert len(manifests) == 48, "logical index must not cache manifest bytes or approval"
    assert len(visits) <= count + 2, (
        f"24 keyed checks revisited {len(visits)} actual events in a {count + 2}-event prefix")
    assert store.events()[2]["payload"] == {"number": 0}


@pytest.mark.parametrize("reader", ["artifact", "provider"])
def test_actual_authorized_read_does_not_clone_the_complete_prefix(tmp_path, monkeypatch, reader):
    if reader == "artifact":
        store, artifact, grant = published(tmp_path)
        expected = b'{"value":2}'
        call = lambda: store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=grant)
    else:
        store, expected, protocol = final_eval(tmp_path, None)
        protocol, _, _ = freeze_evidence(store, protocol)
        ledger = EvaluationExposureLedger(store)
        ledger.register_protocol(protocol, store_module.digest(protocol))
        authorize(store, protocol)
        call = lambda: ledger.read("reserved", "test-block", purpose="evaluate",
                                  authorization_id="evaluate", data_root=tmp_path)
    fill(store, 64)
    initial = len(store.events())
    original_encode, original_json = store_module.encode, store._json
    copies, physical = [], []

    def encode(value):
        result = original_encode(value)
        if (store._read_snapshot() is not None and isinstance(value, list)
                and len(value) >= initial and value and isinstance(value[0], dict)
                and "event_kind" in value[0]):
            copies.append(len(result))
        return result  # Actual serialization, not a synthetic work counter.

    def read(path):
        if path.parent == store.path / "events":
            physical.append(path)
        return original_json(path)

    monkeypatch.setattr(store_module, "encode", encode)
    monkeypatch.setattr(store, "_json", read)
    assert call() == expected
    physical_count = len(physical)
    final_events = store.events()
    assert [event["event_kind"] for event in final_events][-3:] == [
        "EXPOSURE_ALLOWED", "READ_STARTED", "READ_COMPLETED"]
    assert physical_count == initial + len(final_events), "retain both complete physical prefix validations"
    assert not copies, f"one authorized {reader} read made {len(copies)} full-prefix copies ({sum(copies)} bytes)"
