"""Index projection verifies a single locked chain, not one scan per object."""

import json

import pytest

from infrastructure.research_index import ResearchIndex
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_store import spec


def test_rebuild_does_not_rescan_authority_for_each_manifest(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path, "index-work", initialize=True)
    value = spec()
    store.register(value, digest(value))
    for number in range(20):
        store.publish("diagnostic-" + str(number), {"synthetic": number})
    original = store._events
    calls = []
    def observed():
        calls.append(1)
        return original()
    monkeypatch.setattr(store, "_events", observed)
    ResearchIndex(store).rebuild()
    assert len(calls) == 1, "quadratic event-chain rereads stall small UI matrices"


def test_locked_projection_still_rejects_changed_manifest(tmp_path):
    store = ResearchStore(tmp_path, "index-integrity", initialize=True)
    value = spec()
    store.register(value, digest(value))
    path = store.path / "manifests/study-synthetic.json"
    changed = json.loads(path.read_bytes())
    changed["spec"]["experiment_id"] = "damaged-synthetic-source"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        ResearchIndex(store).rebuild()
