"""Registration stays atomic/fresh without rescanning a journal per study."""
from copy import deepcopy
import json

import pytest

from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_research_store import spec


def test_physical_registration_passes_do_not_grow_with_historical_studies(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path, 'registration-scope', initialize=True)
    for number in range(12):
        value = {**spec(), 'study_id': 'fixture-'+str(number)}
        store.register(value, digest(value))
    physical = store._events
    scans = []
    def tracked():
        if store._read_snapshot() is None:
            scans.append(True)
        return physical()
    monkeypatch.setattr(store, '_events', tracked)
    value = {**spec(), 'study_id': 'after-many-studies'}
    store.register(value, digest(value))
    assert len(scans) == 2  # Entry and final uncached physical prefix.
    assert store.manifest('study-after-many-studies')['spec'] == value


def test_registry_still_refuses_new_arm_identity_for_an_existing_family(tmp_path):
    store = ResearchStore(tmp_path, 'registration-scope', initialize=True)
    value = spec()
    store.register(value, digest(value))
    forged = deepcopy(value)
    forged['study_id'] = 'budget-reset'
    forged['arms'][0]['arm_id'] = 'affine-new-budget'
    forged['cells'][0]['arm_id'] = 'affine-new-budget'
    before = store.events()
    with pytest.raises(ResearchError, match='relabeled'):
        store.register(forged, digest(forged))
    assert store.events() == before


def test_final_registration_guard_rehashes_original_manifest_after_publication(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path, 'registration-scope', initialize=True)
    old = spec()
    store.register(old, digest(old))
    append = store._append
    path = store.path / 'manifests/study-synthetic.json'
    def tampered(*args, **kwargs):
        result = append(*args, **kwargs)
        body = json.loads(path.read_text())
        body['spec']['arms'][0]['objective_id'] = 'tampered'
        path.write_text(json.dumps(body))
        return result
    monkeypatch.setattr(store, '_append', tampered)
    value = {**spec(), 'study_id': 'new-study'}
    with pytest.raises(ResearchError, match='hash mismatch'):
        store.register(value, digest(value))
