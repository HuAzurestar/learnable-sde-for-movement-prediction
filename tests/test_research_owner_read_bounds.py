"""Owner-only artifact validation must keep the same frozen allocation bound."""

from pathlib import Path

import pytest

from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_store import spec
from tests.research_file_observation import observe_file


def prepared(tmp_path, content=b'{"synthetic":true}'):
    store = ResearchStore(tmp_path, "owner-bounds", initialize=True)
    value = spec()
    store.register(value, digest(value))
    artifact = store.artifact(content, role="result", visibility="synthetic",
        block_ids=["fixture-1"], study_id="synthetic")
    attempt = store.new_attempt(store.register_run("synthetic", value["cells"][0]))
    store.transition(attempt, "RUNNING")
    return store, artifact, attempt, content


def validate(store, artifact, attempt, content, operation):
    if operation == "success":
        return store.transition(attempt, "SUCCEEDED", artifact_id=artifact["artifact_id"])
    return store.artifact(content, role="result", visibility="synthetic",
        block_ids=["fixture-1"], study_id="synthetic")


def observe(monkeypatch, target, *, grow_after_open=False):
    original = Path.open
    def grow(*_):
        if grow_after_open:
            with original(target, "ab") as writer:
                writer.write(b"x" * 8192)
    reads, _ = observe_file(monkeypatch, target, before_read=grow)
    return reads


@pytest.mark.parametrize("operation", ["success", "duplicate"])
def test_actual_owner_validation_uses_explicit_frozen_read_bound(tmp_path, monkeypatch, operation):
    store, artifact, attempt, content = prepared(tmp_path)
    target = store.path / "artifacts" / artifact["artifact_id"]
    reads = observe(monkeypatch, target)
    result = validate(store, artifact, attempt, content, operation)
    assert result["state"] == "SUCCEEDED" if operation == "success" else result == artifact
    assert reads and all(0 <= row["size"] <= len(content) + 1 for row in reads), "owner validation allocates without frozen size"
    assert target.read_bytes() == content


@pytest.mark.parametrize("operation", ["success", "duplicate"])
def test_grown_owner_artifact_is_rejected_before_content_read(tmp_path, monkeypatch, operation):
    store, artifact, attempt, content = prepared(tmp_path)
    target = store.path / "artifacts" / artifact["artifact_id"]
    target.write_bytes(content + b"x" * 8192)
    before = store.events()
    reads = observe(monkeypatch, target)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        validate(store, artifact, attempt, content, operation)
    assert reads == [], "grown owner artifact was allocated before frozen-size rejection"
    assert store.events() == before and store.attempts()[attempt]["state"] == "RUNNING"


@pytest.mark.parametrize("operation", ["success", "duplicate"])
def test_growth_after_owner_handle_stat_stays_bounded_and_cannot_publish(tmp_path, monkeypatch, operation):
    store, artifact, attempt, content = prepared(tmp_path)
    target = store.path / "artifacts" / artifact["artifact_id"]
    before = store.events()
    reads = observe(monkeypatch, target, grow_after_open=True)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        validate(store, artifact, attempt, content, operation)
    assert reads and all(0 <= row["size"] <= len(content) + 1 for row in reads)
    assert store.events() == before and store.attempts()[attempt]["state"] == "RUNNING"


@pytest.mark.parametrize("operation", ["success", "duplicate"])
@pytest.mark.parametrize("content", [b"", b'{"synthetic":true}'])
def test_valid_owner_empty_and_exact_bytes_keep_success_and_duplicate_semantics(tmp_path, operation, content):
    store, artifact, attempt, content = prepared(tmp_path, content)
    result = validate(store, artifact, attempt, content, operation)
    if operation == "success":
        assert result["state"] == "SUCCEEDED" and result["artifact_manifest_hash"] == digest(artifact)
    else:
        assert result == artifact
    assert (store.path / "artifacts" / artifact["artifact_id"]).read_bytes() == content
