"""Raw/provider/recovery reads cannot reuse permission after durable I/O."""

from datetime import datetime, timezone
from contextlib import nullcontext
from pathlib import Path

import pytest

import application.research_data as data_module
import infrastructure.research_store as store_module
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_research_input_gates import final_eval, freeze_evidence
from tests.test_research_recovery import setup as recovery_setup
from tests.research_file_observation import observe_file


@pytest.fixture
def read_clock(monkeypatch):
    expired = [False]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100 if expired[0] else 2030, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(store_module, 'datetime', Clock)
    monkeypatch.setattr(data_module, 'datetime', Clock)
    return expired


def raw_source(tmp_path):
    store = ResearchStore(tmp_path, 'raw-permission', initialize=True)
    content = b'{"synthetic_owner_fixture":true}'
    artifact = store.artifact(content, role='checkpoint', visibility='restricted',
        block_ids=['owner-block'], study_id='owner-study')
    grant = {'authorization_id': 'owner-grant', 'study_id': 'owner-study',
        'expires_at': '2099-01-01T00:00:00+00:00', 'evidence_hash': digest('owner synthetic permission'),
        'purposes': ['preview', 'export', 'evaluate', 'resume'],
        'visibilities': ['restricted'], 'block_ids': ['owner-block']}
    store.authorize(grant)
    return store, artifact, grant, content


def fault_after_journal(store, grant, phase, clock, monkeypatch):
    original = store._append
    touched = []

    def append(kind, *args, **kwargs):
        result = original(kind, *args, **kwargs)
        match = phase.split(':')[-1]
        if kind == match:
            touched.append(True)
            if phase.startswith('grant:'):
                path = store.path / 'manifests' / ('authorization-' + grant['authorization_id'] + '.json')
                path.write_bytes(encode({**grant, 'purposes': []}))
            else:
                clock[0] = True
        return result

    monkeypatch.setattr(store, '_append', append)
    return touched


@pytest.mark.parametrize('purpose', ['preview', 'export', 'evaluate', 'resume'])
@pytest.mark.parametrize('phase', ['EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED', 'physical'])
def test_raw_read_expiry_after_real_io_refuses_bytes(tmp_path, monkeypatch, read_clock, purpose, phase):
    store, artifact, grant, _ = raw_source(tmp_path)
    touched = fault_after_journal(store, grant, phase, read_clock, monkeypatch)
    original_read, original_events = store._verified_artifact_content, store._events
    reads = []

    def read(metadata):
        content = original_read(metadata)
        reads.append(True)
        return content

    def physical():
        result = original_events()
        if phase == 'physical' and reads and store._read_snapshot() is None:
            touched.append(True)
            read_clock[0] = True
        return result

    monkeypatch.setattr(store, '_verified_artifact_content', read)
    monkeypatch.setattr(store, '_events', physical)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        store.read_artifact(artifact['artifact_id'], purpose=purpose, authorization=grant)
    assert touched
    assert len(reads) == (0 if phase in {'EXPOSURE_ALLOWED', 'READ_STARTED'} else 1)
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'
    assert store.events()[-1]['payload']['purpose'] == purpose


@pytest.mark.parametrize('purpose', ['preview', 'export', 'evaluate', 'resume'])
@pytest.mark.parametrize('phase', ['EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED'])
def test_raw_read_rehashes_actual_grant_after_each_journal(tmp_path, monkeypatch, read_clock, purpose, phase):
    store, artifact, grant, _ = raw_source(tmp_path)
    touched = fault_after_journal(store, grant, 'grant:' + phase, read_clock, monkeypatch)
    original = store._verified_artifact_content
    reads = []

    def read(metadata):
        reads.append(True)
        return original(metadata)

    monkeypatch.setattr(store, '_verified_artifact_content', read)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        store.read_artifact(artifact['artifact_id'], purpose=purpose, authorization=grant)
    assert touched
    assert len(reads) == (1 if phase == 'READ_COMPLETED' else 0)
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'


@pytest.mark.parametrize('purpose', ['preview', 'export', 'evaluate', 'resume'])
def test_valid_raw_read_keeps_original_bytes_and_durable_three_event_journal(tmp_path, read_clock, purpose):
    store, artifact, grant, content = raw_source(tmp_path)
    assert store.read_artifact(artifact['artifact_id'], purpose=purpose, authorization=grant) == content
    assert [e['event_kind'] for e in store.events()][-3:] == ['EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED']
    assert store._read_snapshot() is None


def fault_after_final_chain(store, grant, fault, reads, clock, monkeypatch):
    original = store._events
    touched = []

    def physical():
        result = original()
        if reads and store._read_snapshot() is None and not touched:
            touched.append(True)
            if fault == 'grant':
                path = store.path / 'manifests' / ('authorization-' + grant['authorization_id'] + '.json')
                path.write_bytes(encode({**grant, 'purposes': []}))
            else:
                clock[0] = True
        return result

    monkeypatch.setattr(store, '_events', physical)
    return touched


@pytest.mark.parametrize('purpose', ['preview', 'export', 'evaluate', 'resume'])
@pytest.mark.parametrize('outer', [False, True])
def test_raw_read_rehashes_grant_after_real_final_physical_validation(tmp_path, monkeypatch, read_clock, purpose, outer):
    store, artifact, grant, _ = raw_source(tmp_path)
    reads = []
    original = store._verified_artifact_content

    def read(metadata):
        result = original(metadata)
        reads.append(True)
        return result

    monkeypatch.setattr(store, '_verified_artifact_content', read)
    touched = fault_after_final_chain(store, grant, 'grant', reads, read_clock, monkeypatch)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        with store._read_transaction() if outer else nullcontext():
            store.read_artifact(artifact['artifact_id'], purpose=purpose, authorization=grant)
    assert touched and len(reads) == 1
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'
    assert store._read_snapshot() is None
    assert not hasattr(store._read_scope, 'completions')


def provider_source(tmp_path, purpose):
    store, content, protocol = final_eval(tmp_path, None)
    block = protocol['blocks'][0]
    block['split_role'] = {'fit': 'train', 'select': 'selection', 'validate': 'validation', 'evaluate': 'final-eval'}[purpose]
    block['fit_scope'] = purpose == 'fit'
    if purpose == 'evaluate':
        protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    grant = {'authorization_id': 'provider-grant', 'study_id': 'synthetic',
        'expires_at': '2099-01-01T00:00:00+00:00', 'evidence_hash': digest('synthetic provider permission'),
        'protocol_hash': digest(protocol), 'purposes': [purpose], 'visibilities': ['restricted'],
        'block_ids': ['test-block'], 'test_authorization': True}
    store.authorize(grant)
    return store, ledger, grant, content


@pytest.mark.parametrize('purpose', ['fit', 'select', 'validate', 'evaluate'])
@pytest.mark.parametrize('phase', ['EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED', 'grant:READ_STARTED'])
@pytest.mark.parametrize('operation', ['read', 'verify'])
def test_actual_data_provider_cannot_use_post_journal_stale_grant(tmp_path, monkeypatch, read_clock, purpose, phase, operation):
    store, ledger, grant, _ = provider_source(tmp_path, purpose)
    touched = fault_after_journal(store, grant, phase, read_clock, monkeypatch)
    _, reads = observe_file(monkeypatch, tmp_path / 'block.bin')
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        getattr(ledger, operation)('reserved', 'test-block', purpose=purpose,
                    authorization_id=grant['authorization_id'], data_root=tmp_path)
    assert touched
    assert len(reads) == (1 if phase == 'READ_COMPLETED' else 0)
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'
    assert store.events()[-1]['payload']['purpose'] == purpose


@pytest.mark.parametrize('purpose', ['fit', 'select', 'validate', 'evaluate'])
def test_valid_provider_read_keeps_role_scope_bytes_and_frozen_evidence(tmp_path, read_clock, purpose):
    store, ledger, grant, content = provider_source(tmp_path, purpose)
    assert ledger.read('reserved', 'test-block', purpose=purpose,
        authorization_id=grant['authorization_id'], data_root=tmp_path) == content
    events = store.events()
    assert [e['event_kind'] for e in events][-3:] == ['EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED']
    if purpose == 'evaluate':
        assert events[-1]['payload']['preregistration_hash']
        assert events[-1]['payload']['frozen_sequence'] < events[-1]['sequence']


@pytest.mark.parametrize('purpose', ['fit', 'select', 'validate', 'evaluate'])
@pytest.mark.parametrize('fault', ['grant', 'expiry'])
@pytest.mark.parametrize('outer', [False, True])
@pytest.mark.parametrize('operation', ['read', 'verify'])
def test_provider_revalidates_after_real_final_physical_validation(tmp_path, monkeypatch, read_clock, purpose, fault, outer, operation):
    store, ledger, grant, _ = provider_source(tmp_path, purpose)
    _, reads = observe_file(monkeypatch, tmp_path / 'block.bin')
    touched = fault_after_final_chain(store, grant, fault, reads, read_clock, monkeypatch)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        with store._read_transaction() if outer else nullcontext():
            getattr(ledger, operation)('reserved', 'test-block', purpose=purpose,
                        authorization_id=grant['authorization_id'], data_root=tmp_path)
    assert touched and len(reads) == 1
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'
    assert store._read_snapshot() is None
    assert not hasattr(store._read_scope, 'completions')


def test_last_expiry_checks_follow_all_nested_read_guard_io(tmp_path, monkeypatch, read_clock):
    store, artifact, grant, content = raw_source(tmp_path)
    second = {**grant, 'authorization_id': 'second', 'expires_at': '2199-01-01T00:00:00+00:00'}
    store.authorize(second)
    original_manifest, original_events = store._manifest, store._events
    final_validation = []
    touched = []

    def physical():
        result = original_events()
        if store._read_snapshot() is None and len(final_validation) == 1:
            final_validation.append(True)
        return result

    def metadata(object_id):
        result = original_manifest(object_id)
        if len(final_validation) == 2 and object_id == 'authorization-second':
            touched.append(True)
            read_clock[0] = True
        return result

    monkeypatch.setattr(store, '_events', physical)
    monkeypatch.setattr(store, '_manifest', metadata)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        with store._read_transaction():
            assert store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant) == content
            assert store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=second) == content
            final_validation.append(True)
    assert touched
    assert store.events()[-1]['payload']['authorization_hash'] == digest(grant)
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'
    assert store._read_snapshot() is None
    assert not hasattr(store._read_scope, 'completions')


def test_valid_nested_reads_keep_two_physical_scans_and_release_guards(tmp_path, monkeypatch, read_clock):
    store, artifact, grant, content = raw_source(tmp_path)
    original = store._json
    physical_reads = []

    def observe(path):
        if path.parent == store.path / 'events':
            physical_reads.append(path)
        return original(path)

    monkeypatch.setattr(store, '_json', observe)
    with store._read_transaction():
        for purpose in ('preview', 'export'):
            assert store.read_artifact(artifact['artifact_id'], purpose=purpose, authorization=grant) == content
    count = len(physical_reads)
    events = store.events()
    assert count <= 2 * len(events)
    assert [e['event_kind'] for e in events][-6:] == ['EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED'] * 2
    assert store._read_snapshot() is None
    assert not hasattr(store._read_scope, 'completions')


def test_final_guard_audit_write_failure_refuses_return_and_clears_state(tmp_path, monkeypatch, read_clock):
    store, artifact, grant, _ = raw_source(tmp_path)
    original_read, original_append = store._verified_artifact_content, store._append
    reads, denials = [], []

    def read(metadata):
        result = original_read(metadata)
        reads.append(True)
        return result

    def append(kind, *args, **kwargs):
        if kind == 'EXPOSURE_DENIED':
            denials.append(True)
            raise OSError('final authority journal unavailable')
        return original_append(kind, *args, **kwargs)

    monkeypatch.setattr(store, '_verified_artifact_content', read)
    monkeypatch.setattr(store, '_append', append)
    touched = fault_after_final_chain(store, grant, 'grant', reads, read_clock, monkeypatch)
    with pytest.raises(OSError, match='final authority journal unavailable'):
        store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant)
    assert touched and denials and len(reads) == 1
    assert store._read_snapshot() is None
    assert not hasattr(store._read_scope, 'completions')
    store.events()  # The read journal is still an intact physical chain.


@pytest.mark.parametrize('level', ['exact', 'chunk'])
@pytest.mark.parametrize('phase', ['READ_STARTED', 'grant:READ_STARTED', 'source-metadata', 'grant:source-metadata'])
def test_actual_recovery_prepare_refuses_state_after_raw_read_permission_changes(tmp_path, monkeypatch, read_clock, level, phase):
    store, recovery, attempt, grant = recovery_setup(tmp_path, level)
    reservation = BudgetLedger(store).reserve(attempt, BudgetSpec(10))
    store.transition(attempt, 'RUNNING')
    state = {'step': 4, 'data_position': 12, 'method_state': {'weights': [1, 2]},
             'rng_state': {'numpy': [4, 5], 'brownian_id': 'fixture-stream'}, 'chunk_complete': True}
    checkpoint = recovery.checkpoint(attempt, state)
    BudgetLedger(store).settle(reservation['reservation_id'], 6000, outcome='FAILED')
    store.transition(attempt, 'FAILED', error_code='TRANSIENT')
    touched = fault_after_journal(store, grant, phase, read_clock, monkeypatch)
    original_manifest = store.manifest

    def metadata(object_id):
        result = original_manifest(object_id)
        if phase.endswith('source-metadata') and object_id == 'artifact-' + checkpoint:
            touched.append(True)
            if phase.startswith('grant:'):
                path = store.path / 'manifests' / ('authorization-' + grant['authorization_id'] + '.json')
                path.write_bytes(encode({**grant, 'purposes': []}))
            else:
                read_clock[0] = True
        return result

    monkeypatch.setattr(store, 'manifest', metadata)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        recovery.prepare(attempt, checkpoint, authorization=grant)
    assert touched
    assert len(store.attempts()) == 1
    assert BudgetLedger(store).balance('affine')['committed_ms'] == 6000
