"""Locked read snapshots must not become permission or integrity caches."""

import threading

import pytest

import infrastructure.research_store as store_module
from infrastructure.research_store import ResearchError, encode
from tests.test_research_artifact_read_bounds import published


def test_single_read_bounds_physical_chain_reads_and_preserves_journal(tmp_path, monkeypatch):
    store, artifact, grant = published(tmp_path)
    original = store._json
    reads = []

    def observe(path):
        if path.parent == store.path / 'events':
            reads.append(path)
        return original(path)

    monkeypatch.setattr(store, '_json', observe)
    assert store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant) == b'{"value":2}'
    physical_reads = len(reads)
    events = store.events()
    assert physical_reads <= 2 * len(events), 'authorized read rescans the entire chain for every metadata/journal operation'
    assert [event['event_kind'] for event in events][-3:] == ['EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED']


def test_nested_scope_detaches_events_and_releases_cache(tmp_path):
    store, artifact, grant = published(tmp_path)
    with store._read_transaction():
        with store._read_transaction():
            events = store.events()
            events[-1]['payload'].clear()
            assert store.manifest('authorization-' + grant['authorization_id']) == grant
            assert store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant) == b'{"value":2}'
    store.events()  # Physically revalidate the updated chain, not a leaked scope.


def test_exception_clears_snapshot_before_next_request(tmp_path):
    store, _, grant = published(tmp_path)
    with pytest.raises(LookupError, match='interrupted'):
        with store._read_transaction():
            store.manifest('authorization-' + grant['authorization_id'])
            raise LookupError('interrupted')
    (store.path / 'events/0000000000000001.json').write_bytes(b'{}')
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        store.events()


def test_corrupt_chain_during_byte_read_never_returns_content(tmp_path, monkeypatch):
    store, artifact, grant = published(tmp_path)
    original = store._verified_artifact_content

    def corrupt_after_read(metadata):
        content = original(metadata)
        (store.path / 'events/0000000000000001.json').write_bytes(b'{}')
        return content

    monkeypatch.setattr(store, '_verified_artifact_content', corrupt_after_read)
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant)
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        store.events()


def test_each_read_rehashes_grant_manifest_inside_one_scope(tmp_path):
    store, artifact, grant = published(tmp_path)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        with store._read_transaction():
            assert store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant) == b'{"value":2}'
            path = store.path / 'manifests' / ('authorization-' + grant['authorization_id'] + '.json')
            path.write_bytes(encode({**grant, 'purposes': []}))
            with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
                store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant)
        # Even if the consumer catches the second read's denial, the first
        # protected bytes cannot leave an outer response with invalid authority.
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'
    assert store._read_snapshot() is None
    assert not hasattr(store._read_scope, 'completions')


def test_event_flushed_before_failed_head_is_not_overwritten_in_same_scope(tmp_path, monkeypatch):
    store, _, _ = published(tmp_path)
    original = store_module.atomic_write
    failed = []

    def fail_head_once(path, content):
        if path == store.path / 'head.json' and not failed:
            failed.append(True)
            raise OSError('head publication failed')
        return original(path, content)

    with store._read_transaction():
        monkeypatch.setattr(store_module, 'atomic_write', fail_head_once)
        with pytest.raises(OSError, match='head publication failed'):
            store.append('FIRST', {'value': 1}, 'first')
        store.append('SECOND', {'value': 2}, 'second')
    assert [event['event_id'] for event in store.events()][-2:] == ['first', 'second']


def test_other_thread_cannot_borrow_the_locked_snapshot(tmp_path):
    store, _, grant = published(tmp_path)
    entered, release, other_done = threading.Event(), threading.Event(), threading.Event()
    failures = []

    def owner():
        try:
            with store._read_transaction():
                entered.set()
                assert release.wait(5)
        except BaseException as error:
            failures.append(error)

    def other():
        try:
            store.manifest('authorization-' + grant['authorization_id'])
            other_done.set()
        except BaseException as error:
            failures.append(error)

    first, second = threading.Thread(target=owner), threading.Thread(target=other)
    try:
        first.start()
        assert entered.wait(2), failures
        second.start()
        assert not other_done.wait(0.1), 'another thread bypassed the active OS lock'
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not failures, failures
    assert other_done.is_set()


def test_journal_failure_still_prevents_byte_read(tmp_path, monkeypatch):
    store, artifact, grant = published(tmp_path)
    reads = []

    def refuse(*args, **kwargs):
        raise OSError('journal unavailable')

    monkeypatch.setattr(store, '_append', refuse)
    monkeypatch.setattr(store, '_verified_artifact_content', lambda metadata: reads.append(metadata))
    with pytest.raises(OSError, match='journal unavailable'):
        store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant)
    assert not reads


def test_event_rename_then_error_forces_physical_reread_before_next_append(tmp_path, monkeypatch):
    store, _, _ = published(tmp_path)
    original = store_module.atomic_write
    failed = []

    def fail_after_event_write(path, content):
        original(path, content)
        if path.parent == store.path / 'events' and not failed:
            failed.append(True)
            raise OSError('directory sync failed after rename')

    with store._read_transaction():
        monkeypatch.setattr(store_module, 'atomic_write', fail_after_event_write)
        with pytest.raises(OSError, match='directory sync failed'):
            store.append('FIRST', {'value': 1}, 'first')
        store.append('SECOND', {'value': 2}, 'second')
    assert [event['event_id'] for event in store.events()][-2:] == ['first', 'second']


def test_foreign_pid_snapshot_cannot_bypass_physical_lock_or_chain(tmp_path, monkeypatch):
    store, _, grant = published(tmp_path)
    original = store_module.file_lock
    locks = []

    def observe_lock(path, *args, **kwargs):
        locks.append(path)
        return original(path, *args, **kwargs)

    # Model inherited thread-local state without forking the test runner.
    store._read_scope.snapshot = (-1, store.events())
    monkeypatch.setattr(store_module, 'file_lock', observe_lock)
    try:
        assert store.manifest('authorization-' + grant['authorization_id']) == grant
        assert locks == [store.path / '.writer.lock']
        (store.path / 'events/0000000000000001.json').write_bytes(b'{}')
        with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
            store.events()
    finally:
        del store._read_scope.snapshot


def test_appended_payload_and_returned_event_cannot_mutate_snapshot(tmp_path):
    store, _, _ = published(tmp_path)
    payload = {'nested': {'value': 1}}
    with store._read_transaction():
        event = store.append('FIRST', payload, 'first')
        payload['nested']['value'] = 2
        event['payload']['nested']['value'] = 3
        assert store.events()[-1]['payload'] == {'nested': {'value': 1}}
        store.append('SECOND', {}, 'second')
    assert store.events()[-2]['payload'] == {'nested': {'value': 1}}


def test_explicit_tail_recovery_invalidates_verified_read_snapshot(tmp_path):
    store, _, _ = published(tmp_path)
    with store._read_transaction():
        last = store.events()[-1]
        path = store.path / 'events' / f"{last['sequence']:016d}.json"
        path.write_bytes(b'{}')
        hold = store.quarantine_tail('synthetic corruption during locked read')
        assert store.events()[-1]['event_kind'] == 'RECOVERY_HOLD'
        assert store.events()[-1]['sequence'] == last['sequence']
    assert (store.path / hold['quarantine'] / path.name).read_bytes() == b'{}'
    assert (store.path / 'recovery-hold.json').is_file()
