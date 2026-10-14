"""Actual settled fit; isolated model-only publication and original grants."""

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from application.pirc26_fitted_model import publish_fitted, model_source, read_model, VERSION
from application.research_budget import BudgetLedger
from application.research_reuse import verified_success_metadata
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_pirc26_dsde import contracts
from tests.test_pirc26_population_runtime import prepare, execute, receipt_for, one_thread


def test_actual_model_only_publication_proves_success_cost_and_original_consumer_authority(tmp_path, contracts, monkeypatch):
    fixture = prepare(tmp_path, contracts)
    store, value, plugin, _, _, grant, *_ = fixture
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        publish_fitted(store, "absent-attempt", authorization=grant)
    outcome = execute(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    attempt = store.attempts()[outcome["attempt_id"]]
    result = json.loads(store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant))
    attempts, balance = store.attempts(), BudgetLedger(store).balance("affine")
    reference = publish_fitted(store, attempt["attempt_id"], authorization=grant)
    assert publish_fitted(store, attempt["attempt_id"], authorization=grant) == reference
    assert store.attempts() == attempts and BudgetLedger(store).balance("affine") == balance
    disclosure = {**grant, "authorization_id": "model-only-disclosure", "consumer_study_ids": ["selection-consumer"]}
    store.authorize(disclosure)
    proof = store.manifest(reference["manifest_id"])
    assert proof["scientific_qualification"] == "not-established"
    assert proof["source_result_artifact_id"] == attempt["artifact_id"]
    assert proof["checkpoint_hash"] == result["fit"]["checkpoint"]["sha256"]
    assert proof["cost"]["settled"] and proof["cost"]["charged_ms"] > 0
    assert proof["training_population"] == result["fit"]["training_population"]
    before = store.events()
    with monkeypatch.context() as patch:
        patch.setattr(store, "_verified_artifact_content", lambda *a,**k: pytest.fail("metadata preflight opened payload"))
        metadata = verified_success_metadata(store, attempt, value, value["cells"][0], plugin)
        assert metadata["receipt"] == receipt_for(store, attempt["attempt_id"])
        assert model_source(store, reference, consumer_study_id="selection-consumer", authorization=disclosure) == proof
    assert store.events() == before  # No data exposure or misleading successful reuse.
    opened = []
    physical = store._verified_artifact_content
    def tracked(metadata):
        opened.append(metadata["artifact_id"])
        return physical(metadata)
    with monkeypatch.context() as patch:
        patch.setattr(store, "_verified_artifact_content", tracked)
        body = read_model(store, reference, consumer_study_id="selection-consumer", authorization=disclosure)
    assert opened == [reference["artifact_id"]]  # Never opens mixed training diagnostics.
    assert body == {"schema_version": VERSION, "checkpoint": result["fit"]["checkpoint"],
        "training_population": result["fit"]["training_population"],
        "producer_attempt_id": attempt["attempt_id"], "scientific_qualification": "not-established"}
    journal = store.events()[len(before):]
    read_start = next(e for e in journal if e["event_kind"] == "READ_STARTED")
    read_end = next(e for e in journal if e["event_kind"] == "READ_COMPLETED")
    assert read_start["sequence"] < read_end["sequence"]
    assert read_start["payload"]["artifact_id"] == reference["artifact_id"]
    with monkeypatch.context() as patch:
        patch.setattr(store, "_verified_artifact_content", lambda *a,**k: pytest.fail("denied read opened payload"))
        for fault in ("consumer", "unpublished-grant", "wrong-protocol", "model-receipt", "artifact-reference"):
            ref, authorization, consumer = deepcopy(reference), deepcopy(disclosure), "selection-consumer"
            if fault == "consumer":
                consumer = "not-authorized"
            elif fault == "unpublished-grant":
                authorization["consumer_study_ids"].append("extra-consumer")
            elif fault == "wrong-protocol":
                authorization["protocol_hash"] = "f" * 64
            elif fault == "model-receipt":
                ref["receipt_hash"] = "f" * 64
            else:
                ref["artifact_id"] = attempt["artifact_id"]
            with pytest.raises(ResearchError):
                read_model(store, ref, consumer_study_id=consumer, authorization=authorization)
    # Substitution after a genuine attachment read is rejected by body checks.
    for fault in ("weights", "membership", "qualification", "extra-coordinates"):
        changed = deepcopy(body)
        if fault == "weights":
            changed["checkpoint"]["sha256"] = "f" * 64
        elif fault == "membership":
            changed["training_population"]["transitions"] -= 1
        elif fault == "qualification":
            changed["scientific_qualification"] = "qualified"
        else:
            changed["forecast"] = result["forecast"]
        with monkeypatch.context() as patch:
            patch.setattr(store, "read_artifact", lambda *a,**k: encode(changed))
            with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
                read_model(store, reference, consumer_study_id="selection-consumer", authorization=disclosure)
    # Authoritative settlement must remain bound, even when model bytes exist.
    original_events = store._events
    def substituted_cost():
        events = original_events()
        event = next(e for e in events if e["event_kind"] == "SETTLE"
            and e["payload"].get("attempt_id") == attempt["attempt_id"])
        event["payload"]["charged_ms"] += 1
        return events
    with monkeypatch.context() as patch:
        patch.setattr(store, "_events", substituted_cost)
        with pytest.raises(ResearchError):
            model_source(store, reference, consumer_study_id="selection-consumer", authorization=disclosure)
    assert store.attempts() == attempts and BudgetLedger(store).balance("affine") == balance
    # Only permission clocks are injected AFTER the native worker has stopped.
    import application.pirc26_fitted_model as attachment
    import application.pirc26_preparation as preparation
    import application.pirc26_population_runtime as population
    import application.research_data as data
    import infrastructure.research_store as store_module
    expired = [False]
    class PermissionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100,1,1,tzinfo=timezone.utc) if expired[0] else datetime.now(tz)
    for module in (attachment, preparation, population, data, store_module):
        monkeypatch.setattr(module, "datetime", PermissionClock, raising=False)
    completion = store._read_completion
    def expire_at_final_guard(authority, expiry):
        def checked():
            authority()
            expired[0] = True
        completion(checked, expiry)
    monkeypatch.setattr(store, "_read_completion", expire_at_final_guard)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read_model(store, reference, consumer_study_id="selection-consumer", authorization=disclosure)
    assert expired[0] and store.attempts()[attempt["attempt_id"]]["state"] == "SUCCEEDED"


@pytest.mark.parametrize("reference", [None, {}, {"artifact_id":"f"*64}, []])
def test_model_reference_is_closed_before_store_access(reference):
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        model_source(None, reference, consumer_study_id="no-consumer", authorization={})
