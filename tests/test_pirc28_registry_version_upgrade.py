"""Admission against retained immutable catalogs in disposable fixture stores.

Synthetic operator attestations here are engineering controls, not production
permissions. No supervisor, reservation, worker or scientific retry is used.
"""

from copy import deepcopy

import pytest

from application.research_admission import AdmissionGate
from experiments.pirc28.runtime import switching_plugin
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc28_runtime import prepare


def retain_previous_mapping(store, plugin, version):
    entry = deepcopy(plugin.registry_entry.manifest())
    entry['version'] = version
    entry['code_hash'] = digest('a previous immutable adapter implementation')
    entry_hash = store.publish('registry-entry-' + digest(entry), entry)
    reference = 'registry-version-' + digest({'id': plugin.plugin_id, 'version': version})
    mapping = {'component_id': plugin.plugin_id, 'component_version': version,
               'registry_entry_hash': entry_hash}
    store.publish(reference, mapping)
    return reference, mapping, entry


def assert_no_execution(store):
    assert not any(event['event_kind'] in {'RESERVE', 'QUEUED', 'QUEUE_CLAIMED', 'WORKER_STARTED'}
                   for event in store.events())


def test_new_adapter_version_admits_without_overwriting_retained_legacy_mapping(tmp_path):
    store, spec, _, _, _ = prepare(tmp_path / 'legacy-catalog')
    plugin = switching_plugin()
    old_ref, old_mapping, old_entry = retain_previous_mapping(store, plugin, '1.1.0')
    cell = spec['cells'][0]
    attempt = store.new_attempt(store.register_run(spec['study_id'], cell))
    prefix = store.events()

    receipt = AdmissionGate(store).prepare(spec, cell, plugin, attempt)

    assert plugin.registry_entry.version != '1.1.0'
    assert receipt['registry_entry'] == plugin.registry_entry.manifest()
    current_ref = 'registry-version-' + digest({'id': plugin.plugin_id,
                                               'version': plugin.registry_entry.version})
    assert current_ref != old_ref
    assert store.manifest(current_ref)['registry_entry_hash'] == digest(plugin.registry_entry.manifest())
    assert store.manifest(old_ref) == old_mapping
    assert store.manifest('registry-entry-' + old_mapping['registry_entry_hash']) == old_entry
    assert store.events()[:len(prefix)] == prefix
    assert store.attempts()[attempt]['state'] == 'REGISTERED'
    assert_no_execution(store)


def test_same_version_different_implementation_still_refuses_immutable_replacement(tmp_path):
    store, spec, _, _, _ = prepare(tmp_path / 'same-version-conflict')
    plugin = switching_plugin()
    old_ref, old_mapping, old_entry = retain_previous_mapping(store, plugin, plugin.registry_entry.version)
    cell = spec['cells'][0]
    attempt = store.new_attempt(store.register_run(spec['study_id'], cell))
    prefix = store.events()

    with pytest.raises(ResearchError, match='IDENTITY_CONFLICT.*immutable object already exists'):
        AdmissionGate(store).prepare(spec, cell, plugin, attempt)

    assert store.manifest(old_ref) == old_mapping
    assert store.manifest('registry-entry-' + old_mapping['registry_entry_hash']) == old_entry
    assert store.events()[:len(prefix)] == prefix
    assert_no_execution(store)
