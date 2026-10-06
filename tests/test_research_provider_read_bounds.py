"""Actual legacy providers preserve large inputs but do not allocate corrupt ones."""

import hashlib
from pathlib import Path
import sys
import tracemalloc

import pytest

from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.research_file_observation import observe_file
from tests.test_research_read_authorization import provider_source


PURPOSES = ["fit", "select", "validate", "evaluate"]


def read(ledger, grant, root, purpose):
    return ledger.read("reserved", "test-block", purpose=purpose,
        authorization_id=grant["authorization_id"], data_root=root)


@pytest.mark.parametrize("purpose", PURPOSES)
def test_actual_provider_uses_admitted_size_not_unbounded_read(tmp_path, monkeypatch, purpose):
    store, ledger, grant, expected = provider_source(tmp_path, purpose)
    reads, handles = observe_file(monkeypatch, tmp_path / "block.bin")
    assert read(ledger, grant, tmp_path, purpose) == expected
    assert reads and all(0 <= row["size"] <= len(expected) + 1 for row in reads)
    assert handles == [True]
    assert store.events()[-1]["event_kind"] == "READ_COMPLETED"


@pytest.mark.parametrize("purpose", PURPOSES)
def test_actual_provider_growth_after_open_stays_inside_original_size(tmp_path, monkeypatch, purpose):
    store, ledger, grant, expected = provider_source(tmp_path, purpose)
    target, original = tmp_path / "block.bin", Path.open

    def grow(*_):
        with original(target, "ab") as writer:
            writer.write(b"x" * (1024 * 1024))

    reads, handles = observe_file(monkeypatch, target, before_read=grow)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        read(ledger, grant, tmp_path, purpose)
    assert reads and all(0 <= row["size"] <= len(expected) + 1 for row in reads)
    assert max(row["bytes"] for row in reads) <= len(expected) + 1
    assert handles == [True] and store.events()[-1]["event_kind"] == "READ_FAILED"


@pytest.mark.parametrize("purpose", PURPOSES)
def test_corrupt_pre_grown_provider_does_not_materialize_entire_input(tmp_path, monkeypatch, purpose):
    store, ledger, grant, expected = provider_source(tmp_path, purpose)
    target = tmp_path / "block.bin"
    target.write_bytes(expected + b"x" * (8 * 1024 * 1024))
    reads, handles = observe_file(monkeypatch, target)
    tracemalloc.start()
    try:
        with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
            read(ledger, grant, tmp_path, purpose)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert reads and all(0 <= row["size"] <= 1024 * 1024 for row in reads)
    assert peak < 4 * 1024 * 1024, "corrupt input was materialized before its frozen hash was verified"
    assert handles == [True] and store.events()[-1]["event_kind"] == "READ_FAILED"


@pytest.mark.parametrize("operation", ["read", "verify"])
def test_real_legacy_provider_accepts_exact_valid_input_larger_than_comparison_cap(tmp_path, monkeypatch, operation):
    content = b"synthetic large authorized block\n" * ((17 * 1024 * 1024) // 33 + 1)
    target = tmp_path / "large.bin"
    target.write_bytes(content)
    store = ResearchStore(tmp_path, "large-provider", initialize=True)
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "large", "study_id": "synthetic",
        "blocks": [{"block_id": "large-block", "dataset_id": "fixture", "release_id": "v1",
            "split_role": "train", "fit_scope": True, "path": "large.bin", "sha256": hashlib.sha256(content).hexdigest()}]}
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    grant = {"authorization_id": "large-grant", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
        "evidence_hash": digest("large synthetic permission"), "protocol_hash": digest(protocol),
        "purposes": ["fit"], "visibilities": ["synthetic"], "block_ids": ["large-block"]}
    store.authorize(grant)
    reads, handles = observe_file(monkeypatch, target)
    tracemalloc.start()
    try:
        actual = getattr(ledger, operation)("large", "large-block", purpose="fit", authorization_id="large-grant", data_root=tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(content) > 16 * 1024 * 1024
    if operation == "read":
        assert actual == content
    else:
        assert actual == {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
        assert peak < 4 * 1024 * 1024, "verification materialized a legitimate whole large input"
        assert all(0 <= row["size"] <= 1024 * 1024 for row in reads)
    assert reads and handles == [True]
    assert store.events()[-1]["event_kind"] == "READ_COMPLETED"


def test_streaming_verify_releases_consumed_chunk_before_next_real_read(tmp_path, monkeypatch):
    content = b"synthetic verified chunk\n" * ((2 * 1024 * 1024) // 25 + 1)
    target = tmp_path / "chunks.bin"
    target.write_bytes(content)
    store = ResearchStore(tmp_path, "chunk-lifetime", initialize=True)
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "chunks", "study_id": "synthetic",
        "blocks": [{"block_id": "block", "dataset_id": "fixture", "release_id": "v1",
            "split_role": "train", "fit_scope": True, "path": "chunks.bin", "sha256": hashlib.sha256(content).hexdigest()}]}
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    grant = {"authorization_id": "chunks-grant", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
        "evidence_hash": digest("synthetic chunk permission"), "protocol_hash": digest(protocol),
        "purposes": ["fit"], "visibilities": ["synthetic"], "block_ids": ["block"]}
    store.authorize(grant)
    retained = []

    def before_read(stream, size):
        if size <= 1:
            return
        frame = sys._getframe(1)
        try:
            while frame is not None and frame.f_code is not EvaluationExposureLedger._read.__code__:
                frame = frame.f_back
            assert frame is not None, "actual provider verification frame missing"
            retained.append(len(frame.f_locals.get("chunk", b"")))
        finally:
            del frame  # Do not retain the real frame or its input in the observer.

    reads, handles = observe_file(monkeypatch, target, before_read=before_read)
    assert ledger.verify("chunks", "block", purpose="fit", authorization_id="chunks-grant", data_root=tmp_path) == {
        "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    assert len(retained) >= 2 and all(value == 0 for value in retained), retained
    assert sum(row["bytes"] for row in reads) == len(content) and handles == [True]
    assert all(row["size"] <= 1024 * 1024 for row in reads)


@pytest.mark.parametrize("purpose", PURPOSES)
def test_streaming_verify_keeps_actual_role_grant_and_three_event_journal(tmp_path, monkeypatch, purpose):
    store, ledger, grant, content = provider_source(tmp_path, purpose)
    reads, handles = observe_file(monkeypatch, tmp_path / "block.bin")
    assert ledger.verify("reserved", "test-block", purpose=purpose,
        authorization_id=grant["authorization_id"], data_root=tmp_path) == {
            "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    assert reads and handles == [True]
    assert all(0 <= row["size"] <= len(content) for row in reads), "verification took materialization pass"
    assert [event["event_kind"] for event in store.events()][-3:] == ["EXPOSURE_ALLOWED", "READ_STARTED", "READ_COMPLETED"]


@pytest.mark.parametrize("formal", [False, True])
def test_actual_admission_streams_real_input_and_retains_consumer_receipt(tmp_path, monkeypatch, formal):
    from application.research_admission import AdmissionGate
    from tests.test_research_admission_chain import prepared

    store, value, registry, _ = prepared(tmp_path, formal=formal)
    store.register(value, digest(value))
    cell = value["cells"][0]
    attempt = store.new_attempt(store.register_run(value["study_id"], cell))
    binding = cell["execution"]
    plugin = registry.resolve(cell["plugin_id"], cell["capability"],
        version=binding["component_version"], entry_hash=binding["registry_entry_hash"])
    target = tmp_path / "plugin-input.bin"
    size = target.stat().st_size
    reads, handles = observe_file(monkeypatch, target)
    receipt = AdmissionGate(store).prepare(value, cell, plugin, attempt)
    assert reads and handles == [True]
    assert all(0 <= row["size"] <= size for row in reads), "actual admission retained unused input materialization"
    assert receipt["input_evidence"][-1]["payload"]["attempt_id"] == attempt
    assert receipt["input_evidence"][-1]["payload"]["entrypoint"] == "shared-admission"
    assert store.events()[-1]["event_kind"] == "ADMISSION"
