"""Actual disposable synthetic fit and renewed model-only read bridge."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from application.pirc26_fitted_model import publish_fitted
from application.pirc26_model_continuation import continued_model_source, read_continued_model
from application.research_continuation import renew_grant, renewed_source_grant
from experiments.pirc25.affine import code_hash
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc26_dsde import contracts
from tests.test_pirc26_population_runtime import prepare, execute, receipt_for, one_thread
from tests.test_research_continuation import approved


def test_actual_historic_fit_bridge_checks_old_proofs_current_grants_and_model_only_body(tmp_path, contracts, monkeypatch):
    fixture = prepare(tmp_path, contracts)
    store, spec, _, _, _, grant, *_ = fixture
    result = execute(fixture)
    assert result["state"] == "SUCCEEDED", result
    reference = publish_fitted(store, result["attempt_id"], authorization=grant)
    admitted = receipt_for(store, result["attempt_id"])
    preparation = admitted["documents"]["package"]["payload"]["pirc26_population"]["preparation_ref"]
    request = store.manifest(preparation["request_manifest_id"])
    old = [store.authorization(s["selection"]["authorization_id"],
                               version=s["selection"]["authorization_version"])
           for s in request["sources"] if s["split_role"] == "train"] + [grant]
    unique = {digest(g): g for g in old}
    # Fixture grants intentionally live to2099; the renewal uses a later fixture
    # expiry, not any change to real research permissions or timestamps.
    approval = approved(store, historical_code_hash=spec["code_hash"],
                        authorization_ids=[g["authorization_id"] for g in unique.values()],
                        authorization_expires_at="2100-01-01T00:00:00+00:00")
    renewals = {h: renew_grant(store, g["authorization_id"], old_version=g.get("version"),
                              new_version="renewed-fixture-v1", approval_ref=approval)
                for h, g in unique.items()}
    bridge = {"schema_version": "pirc26-model-continuation-v1", "approval": approval,
              "historical_code_hash": spec["code_hash"], "current_code_hash": code_hash(),
              "renewals": renewals, "model_reference": reference}
    continuation = {"manifest_id": "model-continuation-" + digest(bridge), "sha256": digest(bridge)}
    store.publish(continuation["manifest_id"], bridge)
    authorization = renewed_source_grant(store, grant, renewals)
    consumer = spec["study_id"]
    before = store.events()
    with monkeypatch.context() as patch:
        patch.setattr(store, "_verified_artifact_content", lambda *a, **k: pytest.fail("metadata opened payload"))
        proof = continued_model_source(store, reference, continuation=continuation,
                                       consumer_study_id=consumer, authorization=authorization)
    assert store.events() == before
    opened = []
    physical = store._verified_artifact_content
    with monkeypatch.context() as patch:
        def tracked(metadata):
            opened.append(metadata["artifact_id"])
            return physical(metadata)
        patch.setattr(store, "_verified_artifact_content", tracked)
        body = read_continued_model(store, reference, continuation=continuation,
                                    consumer_study_id=consumer, authorization=authorization)
    assert opened == [reference["artifact_id"]]
    assert body["checkpoint"]["sha256"] == proof["checkpoint_hash"]
    assert body["scientific_qualification"] == "not-established"
    assert all(e["payload"].get("artifact_id") == reference["artifact_id"]
               for e in store.events()[len(before):] if e["event_kind"] == "READ_COMPLETED")
    with monkeypatch.context() as patch:
        patch.setattr(store, "_verified_artifact_content", lambda *a, **k: pytest.fail("denied read opened payload"))
        for fault in ("consumer", "reference", "bridge-hash", "code-epoch", "missing-renewal", "grant"):
            ref, ctx, current, who = deepcopy(reference), deepcopy(continuation), deepcopy(authorization), consumer
            if fault == "consumer":
                who = "unapproved-consumer"
            elif fault == "reference":
                ref["receipt_hash"] = "f" * 64
            elif fault == "bridge-hash":
                ctx["sha256"] = "f" * 64
            elif fault == "grant":
                current["purposes"].append("export")
            else:
                changed = deepcopy(bridge)
                if fault == "code-epoch":
                    changed["current_code_hash"] = "f" * 64
                else:
                    changed["renewals"].pop(next(iter(changed["renewals"])))
                ctx = {"manifest_id": "bad-bridge-" + digest(changed), "sha256": digest(changed)}
                store.publish(ctx["manifest_id"], changed)
            with pytest.raises(ResearchError):
                read_continued_model(store, ref, continuation=ctx,
                                     consumer_study_id=who, authorization=current)
