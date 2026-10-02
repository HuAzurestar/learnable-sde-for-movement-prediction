"""Allocation bounds on actual authorized and owner integrity artifact reads."""

from pathlib import Path

import pytest

from infrastructure.research_store import ResearchStore, ResearchError, digest


def published(tmp_path):
    store = ResearchStore(tmp_path, "read-bounds", initialize=True)
    artifact = store.artifact(b'{"value":2}', role="aggregate", visibility="synthetic",
        block_ids=["b"], study_id="synthetic")
    grant = {"authorization_id": "bounds", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
        "evidence_hash": digest("synthetic bounds permission"), "purposes": ["preview", "resume"],
        "visibilities": ["synthetic"], "block_ids": ["b"]}
    store.authorize(grant)
    return store, artifact, grant


@pytest.mark.parametrize("purpose", ["preview", "resume"])
def test_authorized_artifact_read_uses_frozen_size_bound(tmp_path, monkeypatch, purpose):
    store, artifact, grant = published(tmp_path)
    target = store.path / "artifacts" / artifact["artifact_id"]
    original_open = Path.open
    reads = []
    class BoundedStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def read(self, size=-1):
            assert 0 <= size <= artifact["size_bytes"] + 1, "artifact read allocates without its frozen size quota"
            reads.append(size)
            return self.stream.read(size)
    def open_guard(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return BoundedStream(stream) if path == target else stream
    monkeypatch.setattr(Path, "open", open_guard)
    assert store.read_artifact(artifact["artifact_id"], purpose=purpose, authorization=grant) == b'{"value":2}'
    assert reads and store.events()[-1]["event_kind"] == "READ_COMPLETED"


def test_grown_corrupt_artifact_rejects_before_content_allocation(tmp_path, monkeypatch):
    store, artifact, grant = published(tmp_path)
    target = store.path / "artifacts" / artifact["artifact_id"]
    target.write_bytes(b"x" * 8192)
    original_read = Path.read_bytes
    def deny_content(path):
        if path == target:
            pytest.fail("oversized corrupt artifact reached an unbounded content read")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", deny_content)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=grant)
    assert store.events()[-1]["event_kind"] == "READ_FAILED"


@pytest.mark.parametrize("size", [True, -1, 1.0, "10", None])
def test_invalid_frozen_size_rejects_before_content_read(tmp_path, monkeypatch, size):
    store, artifact, _ = published(tmp_path)
    target = store.path / "artifacts" / artifact["artifact_id"]
    original_read = Path.read_bytes
    def deny_content(path):
        if path == target:
            pytest.fail("invalid artifact size reached an unbounded content read")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", deny_content)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        store._verified_artifact_content({**artifact, "size_bytes": size})
