"""Actual legacy providers preserve large inputs but do not allocate corrupt ones."""

import hashlib
from pathlib import Path
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


def test_real_legacy_provider_accepts_exact_valid_input_larger_than_comparison_cap(tmp_path, monkeypatch):
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
    actual = ledger.read("large", "large-block", purpose="fit", authorization_id="large-grant", data_root=tmp_path)
    assert len(actual) > 16 * 1024 * 1024 and actual == content
    assert reads and handles == [True]
    assert store.events()[-1]["event_kind"] == "READ_COMPLETED"
