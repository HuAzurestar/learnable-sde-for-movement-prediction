"""Real regular handles retain three fresh boundaries without duplicate resolution."""
from infrastructure import research_files
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchStore
from tests.research_file_observation import observe_file


def test_regular_handle_resolves_the_complete_source_once_per_boundary(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "synthetic.bin"
    path.write_bytes(b"synthetic bytes")
    original, calls = research_files._source_in_root, []
    def resolve(root, path):
        calls.append(path)
        return original(root, path)
    reads, handles = observe_file(monkeypatch, path)
    monkeypatch.setattr(research_files, "_source_in_root", resolve)
    with opened_regular_file(root, path) as (stream, size, _):
        assert stream.read(size + 1) == b"synthetic bytes"
    assert reads and handles == [True]
    assert calls == [path, path, path], "each fresh boundary redundantly resolves the source AND its root"


def test_actual_metadata_load_keeps_the_same_three_source_boundaries(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path, "handle-work", initialize=True)
    value = {"synthetic": "metadata"}
    store.publish("synthetic", value)
    path = store.path / "manifests/synthetic.json"
    original, calls = research_files._source_in_root, []
    def resolve(root, path):
        calls.append(path)
        return original(root, path)
    reads, handles = observe_file(monkeypatch, path)
    monkeypatch.setattr(research_files, "_source_in_root", resolve)
    assert store._json(path) == value
    assert reads and handles == [True]
    assert calls == [path, path, path], "real metadata loading retained redundant root resolution"
