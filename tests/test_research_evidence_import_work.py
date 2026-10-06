"""Actual package import keeps whole-chain and original-root closing guards."""
import pytest

from application.research_evidence import accept_evidence_package
from infrastructure.research_store import ResearchError
from tests.test_research_comparison import source


def test_package_import_verifies_one_initial_and_one_final_chain(source, tmp_path, monkeypatch):
    store, _, _, _, package = source
    initial_events = len(store.events())
    original, reads = store._json, []

    def read(path):
        if path.parent == store.path / "events":
            reads.append(path)
        return original(path)

    monkeypatch.setattr(store, "_json", read)
    imported = accept_evidence_package(store, tmp_path / "source-evidence", package["aggregate_hash"])
    physical_reads = len(reads)
    final_events = store.events()
    assert imported == package and store._read_snapshot() is None
    assert physical_reads == initial_events + len(final_events), (
        f"package import read {physical_reads} real event files instead of "
        f"initial {initial_events} plus final {len(final_events)} complete prefixes")
    assert len(store.attempts()) == 8


def test_import_cannot_return_after_final_comparison_publication_corrupts_chain(source, tmp_path, monkeypatch):
    store, _, _, _, package = source
    original, corrupted = store.publish, []

    def publish(object_id, value):
        result = original(object_id, value)
        if object_id == "comparison-" + package["aggregate_hash"]:
            corrupted.append(True)
            (store.path / "events/0000000000000001.json").write_bytes(b"{}")
        return result

    monkeypatch.setattr(store, "publish", publish)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        accept_evidence_package(store, tmp_path / "source-evidence", package["aggregate_hash"])
    assert corrupted == [True] and store._read_snapshot() is None


def test_import_keeps_original_root_guard_after_final_physical_validation(source, tmp_path, monkeypatch):
    store, _, _, _, package = source
    root, moved = tmp_path / "source-evidence", tmp_path / "original-source-evidence"
    original_publish, original_events = store.publish, store._events
    published, replaced = [], []

    def publish(object_id, value):
        result = original_publish(object_id, value)
        if object_id == "comparison-" + package["aggregate_hash"]:
            published.append(True)
        return result

    def events():
        result = original_events()
        if published and not replaced and store._read_snapshot() is None:
            assert root.resolve().is_relative_to(tmp_path.resolve())
            assert moved.resolve().is_relative_to(tmp_path.resolve())
            assert not moved.exists()
            root.rename(moved)
            root.mkdir()  # A different real directory, not a mocked identity.
            replaced.append(True)
        return result

    monkeypatch.setattr(store, "publish", publish)
    monkeypatch.setattr(store, "_events", events)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        accept_evidence_package(store, root, package["aggregate_hash"])
    assert published == [True] and replaced == [True]
    assert list(root.iterdir()) == [] and (moved / "aggregate.json").is_file()
    assert store._read_snapshot() is None
