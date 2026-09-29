"""Crash visibility, immutable identities, concurrency and disclosure order."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from infrastructure.research_store import ResearchStore, ResearchError, digest
from infrastructure.research_index import ResearchIndex


def spec():
    return {"schema_version": "pirc25-contract-v1", "study_id": "synthetic",
            "experiment_id": "baseline", "comparison_family": "fixture",
            **{key: digest(key) for key in ("protocol_hash", "code_hash", "data_hash", "feature_hash", "selection_hash")},
            "arms": [{"arm_id": "affine", "model_family_id": "affine", "method_family_id": "exact",
                      "objective_id": "energy", "budget_seconds": 86400}],
            "cells": [{"arm_id": "affine", "seed": 1, "block_id": "fixture-1"}]}


@pytest.fixture
def store(tmp_path):
    return ResearchStore(tmp_path, "fixture", initialize=True)


def test_store_identity_is_explicit_and_outside_git(tmp_path):
    with pytest.raises(ResearchError, match="STORE_MISSING"):
        ResearchStore(tmp_path, "fixture")
    ResearchStore(tmp_path, "fixture", initialize=True)
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        ResearchStore(tmp_path, "another", initialize=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ResearchError, match="outside Git"):
        ResearchStore(repo / "runtime", "fixture", initialize=True)


def test_concurrent_registration_is_idempotent_and_immutable(store):
    value = spec()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda _: store.register(value, digest(value)), range(8)))
    assert all(outcome == outcomes[0] for outcome in outcomes)
    assert len(store.events()) == 1
    changed = {**value, "experiment_id": "changed"}
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        store.register(changed, digest(changed))


def test_event_replay_and_truncated_or_missing_tail_fail_closed(store):
    store.append("TEST", {"a": 1}, "same")
    store.append("TEST", {"a": 1}, "same")
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        store.append("TEST", {"a": 2}, "same")
    path = store.path / "events/0000000000000001.json"
    original = path.read_bytes()
    path.write_bytes(original[:15])
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        store.events()
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(ResearchError, match="tail lost"):
        store.events()


def test_unpublished_staging_is_not_visible_and_can_be_retried(store, monkeypatch):
    real_append = store._append
    monkeypatch.setattr(store, "_append", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        store.publish("object", {"value": 1})
    with pytest.raises(ResearchError, match="not published"):
        store.manifest("object")
    monkeypatch.setattr(store, "_append", real_append)
    store.publish("object", {"value": 1})
    assert store.manifest("object") == {"value": 1}


def test_index_staleness_corruption_and_rebuild(store):
    value = spec()
    store.register(value, digest(value))
    index = ResearchIndex(store)
    index.rebuild()
    page = index.list()
    assert page["items"][0]["manifest"]["spec"] == value
    store.append("TEST", {})
    with pytest.raises(ResearchError, match="INDEX_STALE"):
        index.list()
    index.rebuild()
    with pytest.raises(ResearchError, match="CURSOR_STALE"):
        index.list(watermark=page["watermark"])
    index.path.write_bytes(b"corrupt database")
    index.rebuild()
    assert len(index.list()["items"]) == 1
    assert list(store.path.glob("research.sqlite3.corrupt-*"))


def test_disclosure_records_before_read_and_keeps_failed_exposure(store):
    artifact = store.artifact(b'{"value":2}', role="aggregate", visibility="restricted", block_ids=["b"], study_id="synthetic")
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        store.read_artifact(artifact["artifact_id"], purpose="preview", authorization={})
    assert store.events()[-1]["event_kind"] == "EXPOSURE_DENIED"
    authorization = {"authorization_id": "grant1", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
                     "evidence_hash": digest("fixture permission"),
                     "purposes": ["preview"], "visibilities": ["restricted"], "block_ids": ["b"]}
    store.authorize(authorization)
    assert json.loads(store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=authorization)) == {"value": 2}
    assert [e["event_kind"] for e in store.events()][-3:] == ["EXPOSURE_ALLOWED", "READ_STARTED", "READ_COMPLETED"]
    (store.path / "artifacts" / artifact["artifact_id"]).write_bytes(b"bad")
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=authorization)
    assert store.events()[-1]["event_kind"] == "READ_FAILED"


def test_ledger_failure_prevents_read(store, monkeypatch):
    artifact = store.artifact(b"{}", role="aggregate", visibility="synthetic", block_ids=[], study_id="synthetic")
    monkeypatch.setattr(store, "_append", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        store.read_artifact(artifact["artifact_id"], purpose="preview", authorization={
            "purposes": ["preview"], "visibilities": ["synthetic"], "block_ids": []})
    with pytest.raises(ResearchError, match="invalid artifact ID"):
        store.read_artifact("../store.json", purpose="preview", authorization={})


def test_data_roles_are_checked_before_open_and_exposure_cannot_be_reset(store, tmp_path):
    import hashlib
    from application.research_data import EvaluationExposureLedger

    content = b"synthetic observations"
    (tmp_path / "block.json").write_bytes(content)
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "p1",
                "study_id": "synthetic", "blocks": [{"block_id": "b", "dataset_id": "fixture",
                "release_id": "v1", "split_role": "train", "fit_scope": True,
                "path": "block.json", "sha256": hashlib.sha256(content).hexdigest()}]}
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorization = {"authorization_id": "grant", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
                     "evidence_hash": digest("fixture permission"), "protocol_hash": digest(protocol),
                     "purposes": ["fit"], "visibilities": ["synthetic"], "block_ids": ["b"]}
    store.authorize(authorization)
    assert ledger.read("p1", "b", purpose="fit", authorization_id="grant", data_root=tmp_path) == content
    (tmp_path / "block.json").unlink()
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ledger.read("p1", "b", purpose="evaluate", authorization_id="grant", data_root=tmp_path)
    assert sum(e["event_kind"] == "READ_COMPLETED" for e in store.events()) == 1
    changed = {**protocol, "history_status": "unknown"}
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        ledger.register_protocol(changed, digest(changed))


def test_forged_grant_and_cross_study_disclosure_are_rejected(store):
    artifact = store.artifact(b"{}", role="aggregate", visibility="restricted", block_ids=["b"], study_id="studyA")
    forged = {"authorization_id": "fake", "study_id": "studyA", "expires_at": "2099-01-01T00:00:00+00:00",
              "evidence_hash": digest("fake"), "purposes": ["preview"], "visibilities": ["restricted"], "block_ids": ["b"]}
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=forged)
    store.authorize({**forged, "study_id": "studyB"})
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=forged)


def test_run_and_attempt_identity_terminal_success_and_retry(store):
    value = spec()
    store.register(value, digest(value))
    run_id = store.register_run("synthetic", value["cells"][0])
    assert run_id == store.register_run("synthetic", value["cells"][0])
    attempt = store.new_attempt(run_id)
    with pytest.raises(ResearchError, match="live attempt"):
        store.new_attempt(run_id)
    store.transition(attempt, "RUNNING")
    store.transition(attempt, "FAILED", error_code="TRANSIENT")
    with pytest.raises(ResearchError, match="terminal state"):
        store.transition(attempt, "RUNNING")
    with pytest.raises(ResearchError, match="retry requires"):
        store.new_attempt(run_id)
    retry = store.new_attempt(run_id, parent_attempt_id=attempt, reason="explicit retry")
    store.transition(retry, "RUNNING")
    with pytest.raises(ResearchError, match="success needs"):
        store.transition(retry, "SUCCEEDED")
    artifact = store.artifact(b'{"value":1}', role="result", visibility="synthetic", block_ids=[], study_id="synthetic")
    store.transition(retry, "SUCCEEDED", artifact_id=artifact["artifact_id"])
    with pytest.raises(ResearchError, match="successful cell"):
        store.new_attempt(run_id, parent_attempt_id=retry, reason="overwrite")
    assert len(store.attempts()) == 2
