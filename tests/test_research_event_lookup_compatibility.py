"""Keyed prefix projection preserves conservative exposure and isolation."""

import math

import pytest

from application.research_preregistration import same_source
from infrastructure.research_event_lookup import VerifiedEventLookup
from infrastructure.research_store import ResearchError, digest
from application.research_data import EvaluationExposureLedger
from tests.test_research_artifact_read_bounds import published
from tests.test_research_event_lookup_work import fill
from tests.test_research_input_gates import final_eval, freeze_evidence, authorize
from tests.research_file_observation import observe_file


@pytest.mark.parametrize("kind", ["READ_STARTED", "READ_COMPLETED", "READ_FAILED", "EXPOSURE_ALLOWED"])
@pytest.mark.parametrize("variant", ["same", "same-content", "same-source", "unknown", "invalid-hash",
                                    "other", "artifact", "legacy-block"])
def test_indexed_prior_candidates_preserve_actual_legacy_predicate(tmp_path, kind, variant):
    store, _, _ = published(tmp_path)
    identity = {"dataset_id": "dataset", "release_id": "v1", "source_block_id": "source",
                "sha256": digest("original content")}
    payload = dict(identity)
    if variant == "same-content":
        payload.update(dataset_id="renamed", source_block_id="renamed")
    elif variant == "same-source":
        payload.update(sha256=digest("changed content"), release_id="renamed")
    elif variant in {"unknown", "invalid-hash"}:
        payload.update(source_block_id="renamed", sha256=None if variant == "unknown" else "not-a-hash")
    elif variant == "other":
        payload.update(dataset_id="other", source_block_id="other", sha256=digest("other content"))
    elif variant == "artifact":
        payload = {"artifact_id": "legacy-artifact"}
    elif variant == "legacy-block":
        payload.pop("source_block_id")
        payload.update(block_id="source", sha256=digest("changed content"))
    with store._read_transaction():
        early = store.append(kind, payload)
        boundary = store.append("BOUNDARY", {})["sequence"]
        store.append(kind, payload)
        actual = store.events()
        expected = [event for event in actual if event["sequence"] < boundary
                    and event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "READ_FAILED"}
                    and (same_source(event["payload"], identity) or event["payload"].get("artifact_id"))]
        indexed = store._prior_read_events(identity, boundary)
        # Index may conservatively include a same-source candidate that the
        # original predicate rejects; it must never drop a real veto.
        filtered = [event for event in indexed
                    if same_source(event["payload"], identity) or event["payload"].get("artifact_id")]
        assert filtered == expected
        assert all(event["sequence"] < boundary for event in indexed)
        if indexed:
            indexed[0]["payload"].clear()
            assert store.events()[early["sequence"] - 1]["payload"] == payload


def test_first_freeze_latest_manifest_and_appended_prefix_remain_distinct(tmp_path):
    store, _, grant = published(tmp_path)
    key = "authorization-" + grant["authorization_id"]
    with store._read_transaction():
        first = store._manifest_event(key)
        repeated = store.append("MANIFEST", {"object_id": key, "sha256": digest(grant)})
        assert store._manifest_event(key)["sequence"] == first["sequence"]
        assert store._manifest_event(key, last=True)["sequence"] == repeated["sequence"]
        assert store.manifest(key) == grant
        rows = store._manifest_events("authorization-")
        assert rows == [first, repeated]
        rows[-1]["payload"].clear()
        assert store._manifest_event(key, last=True) == repeated


def test_index_update_failure_after_real_publication_forces_fresh_prefix(tmp_path, monkeypatch):
    store, _, grant = published(tmp_path)
    original = VerifiedEventLookup.append
    failed = []

    def fail_once(index, event):
        original(index, event)
        if not failed:
            failed.append(True)
            raise OSError("projection failed after durable event publication")

    with store._read_transaction():
        assert store.authorization(grant["authorization_id"]) == grant
        monkeypatch.setattr(VerifiedEventLookup, "append", fail_once)
        with pytest.raises(OSError, match="projection failed"):
            store.append("FIRST", {"value": 1}, "first")
        assert not hasattr(store._read_scope, "event_lookup")
        store.append("SECOND", {"value": 2}, "second")
    events = store.events()
    assert [event["event_id"] for event in events][-2:] == ["first", "second"]
    assert events[-1]["previous_hash"] == events[-2]["hash"]


@pytest.mark.parametrize("count", [16, 256])
def test_actual_sorted_key_access_is_logarithmic_and_scope_owned(tmp_path, count):
    store, _, grant = published(tmp_path)
    fill(store, count)
    with store._read_transaction():
        lookup = store._event_lookup()
        comparisons = []

        class ObservedKeys(list):
            def __getitem__(self, position):
                comparisons.append(position)
                return super().__getitem__(position)

        lookup._keys = ObservedKeys(lookup._keys)
        for event_id in ["lookup-fixture-0", "lookup-fixture-" + str(count - 1), "missing"]:
            comparisons.clear()
            lookup.get(("event", event_id))
            assert len(comparisons) <= math.ceil(math.log2(len(lookup._keys) + 1)) + 1
        assert store.authorization(grant["authorization_id"]) == grant
    assert not hasattr(store._read_scope, "event_lookup")
    assert store._read_snapshot() is None


def test_unrelated_legacy_payload_does_not_become_authority_or_break_lookup(tmp_path):
    store, _, grant = published(tmp_path)
    with store._read_transaction():
        store.append("LEGACY_NOTE", [])
        assert store.authorization(grant["authorization_id"]) == grant
        assert store._prior_read_events({"dataset_id": "dataset", "source_block_id": "source",
                                         "sha256": digest("content")}, 1000) == []


@pytest.mark.parametrize("kind,payload", [("READ_STARTED", []), ("READ_COMPLETED", []),
                                         ("READ_FAILED", []), ("MANIFEST", []),
                                         ("MANIFEST", {"object_id": 3})])
@pytest.mark.parametrize("after_freeze", [False, True])
def test_malformed_read_scope_cannot_disappear_from_blind_check(tmp_path, monkeypatch, kind, payload, after_freeze):
    store, _, protocol = final_eval(tmp_path, None)
    if not after_freeze:
        store.append(kind, payload)
    protocol, _, _ = freeze_evidence(store, protocol)
    if after_freeze:
        store.append(kind, payload)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    opened = []
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=lambda: opened.append(True))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA|CORRUPT_ARTIFACT"):
        ledger.read("reserved", "test-block", purpose="evaluate", authorization_id="evaluate", data_root=tmp_path)
    assert not opened, "unknown read scope was ignored before real reserved-block access"
    if kind != "MANIFEST":
        assert store.events()[-1]["event_kind"] == "EXPOSURE_DENIED"
    else:
        # A malformed authority prefix cannot even resolve the protocol. Stop
        # with the integrity error; do not invent a reliable data-scope journal.
        assert not any(event["event_kind"] == "READ_STARTED" for event in store.events())
