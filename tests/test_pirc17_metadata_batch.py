"""Restoration-local metadata/software checks, not empirical qualification."""
from copy import deepcopy
import os
from threading import Thread
from types import SimpleNamespace

import pytest

from experiments.pirc17 import formal_metadata_batch as module
from experiments.pirc17.formal_pinned_metadata import PinnedMetadata, OwnedRecord, record_payload
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, publish


def pinned(root):
    path, record = publish(root, dict(rows=[dict(software_only=True, i=i) for i in range(100)]))
    ref = dict(path=str(path), content_sha256=record['sha256'], file_sha256=file_hash(path))
    return path, ref, PinnedMetadata(ref, root=root)


def test_thousands_of_consumptions_use_two_full_checks_not_thousands(tmp_path, monkeypatch):
    path, ref, reader = pinned(tmp_path)
    owner = SimpleNamespace()
    calls = []
    original = PinnedMetadata._bytes
    def counted(self, reference):
        calls.append((str(path), len(path.read_bytes())))
        return original(self, reference)
    monkeypatch.setattr(PinnedMetadata, '_bytes', counted)
    with module.metadata_batch(owner) as batch:
        records = [module.read_metadata(owner, reader, ref)[0] for _ in range(1000)]
        assert len(calls) == 1 and all(record is records[0] for record in records)
        with pytest.raises(TypeError): records[0]['payload']['rows'].append('external mutation')
    assert len(calls) == 2 and batch.status == 'closed'
    assert owner._metadata_batch is None and batch.readers == {} and batch.owners == []
    module.read_metadata(owner, reader, ref)
    module.read_metadata(owner, reader, ref)
    assert len(calls) == 4  # Strict ordinary reads resume immediately.
    with pytest.raises(ValueError, match='remain open'): batch.read(reader, ref)


def test_same_size_restored_mtime_corruption_fails_before_successful_batch_exit(tmp_path):
    path, ref, reader = pinned(tmp_path)
    owner = SimpleNamespace()
    passed = False
    before = path.stat()
    with pytest.raises(ValueError, match='bytes changed'):
        with module.metadata_batch(owner) as batch:
            original, _ = module.read_metadata(owner, reader, ref)
            raw = path.read_bytes()
            altered = raw.replace(b'"i":0', b'"i":9', 1)
            assert len(altered) == len(raw) and altered != raw
            path.write_bytes(altered)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert module.read_metadata(owner, reader, ref)[0] is original
        passed = True  # Represents the caller's publication/admission point.
    assert not passed and batch.status == 'failed' and owner._metadata_batch is None
    with pytest.raises(ValueError, match='bytes changed'):
        module.read_metadata(owner, reader, ref)


@pytest.mark.parametrize('field', ['path', 'content_sha256', 'file_sha256'])
def test_complete_reference_changes_rejected_even_on_cached_read(tmp_path, field):
    _, ref, reader = pinned(tmp_path)
    owner = SimpleNamespace()
    with module.metadata_batch(owner):
        module.read_metadata(owner, reader, ref)
        altered = dict(ref, **{field:str(tmp_path/'wrong') if field == 'path' else digest('wrong')})
        with pytest.raises(ValueError, match='reference changed'):
            module.read_metadata(owner, reader, altered)


def test_body_failure_cannot_close_or_leak_cache_into_next_consumer(tmp_path):
    _, ref, reader = pinned(tmp_path)
    parent, child = SimpleNamespace(), SimpleNamespace()
    with pytest.raises(RuntimeError, match='domain failed'):
        with module.metadata_batch(parent) as batch:
            module.share_batch(parent, child)
            module.read_metadata(child, reader, ref)
            raise RuntimeError('domain failed')
    assert batch.status == 'failed'
    assert parent._metadata_batch is child._metadata_batch is None
    assert batch.readers == {}
    with module.metadata_batch(parent) as later:
        module.read_metadata(parent, reader, ref)
    assert later.status == 'closed'


def test_overlapping_batch_does_not_detach_existing_owner(tmp_path):
    _, ref, reader = pinned(tmp_path)
    owner = SimpleNamespace()
    with module.metadata_batch(owner) as existing:
        with pytest.raises(ValueError, match='overlapping'):
            with module.metadata_batch(owner): pytest.fail('overlap admitted')
        assert owner._metadata_batch is existing
        module.read_metadata(owner, reader, ref)


def test_other_thread_cannot_read_attach_or_finish_batch(tmp_path):
    _, ref, reader = pinned(tmp_path)
    owner = SimpleNamespace()
    failures = []
    with module.metadata_batch(owner) as batch:
        def other_thread():
            for action in (lambda:batch.read(reader, ref), lambda:batch.attach(SimpleNamespace()), batch.finish):
                try: action()
                except ValueError: failures.append(True)
        thread = Thread(target=other_thread)
        thread.start(); thread.join(timeout=5)
        assert not thread.is_alive() and failures == [True, True, True]
        module.read_metadata(owner, reader, ref)


@pytest.mark.parametrize('limit', ['MAX_READERS', 'MAX_OWNERS'])
def test_finite_inventory_limits_fail_closed_and_detach(tmp_path, monkeypatch, limit):
    _, ref, one = pinned(tmp_path/'one')
    _, second_ref, two = pinned(tmp_path/'two')
    owner, child = SimpleNamespace(), SimpleNamespace()
    monkeypatch.setattr(module, limit, 1)
    with pytest.raises(ValueError, match='unbounded|bounded'):
        with module.metadata_batch(owner) as batch:
            module.read_metadata(owner, one, ref)
            if limit == 'MAX_READERS': module.read_metadata(owner, two, second_ref)
            else: module.share_batch(owner, child)
    assert owner._metadata_batch is None and batch.status == 'failed'


def test_each_independent_reader_is_checked_and_mutable_producer_not_admitted(tmp_path):
    _, ref, first = pinned(tmp_path)
    second = PinnedMetadata(ref, root=tmp_path)
    parent, nested = SimpleNamespace(), SimpleNamespace()
    with module.metadata_batch(parent) as batch:
        module.share_batch(parent, nested)
        one, _ = module.read_metadata(parent, first, ref)
        two, _ = module.read_metadata(nested, second, ref)
        assert one is not two and canonical(one) == canonical(two)
        assert len(batch.readers) == 2
        with pytest.raises(ValueError, match='private pinned'):
            batch.read(SimpleNamespace(reference=ref, producer_pass=True), ref)
    assert parent._metadata_batch is nested._metadata_batch is None


def test_owned_scope_meaning_never_trusts_a_mutable_claimed_header():
    caller = envelope(dict(forecast_contract=dict(forecast=dict(particles=512)),
                           source_catalog=[dict(software_only=True, i=i) for i in range(1000)]))
    owned = OwnedRecord(caller)
    another = deepcopy(caller)
    another['payload']['forecast_contract']['forecast']['particles'] = 100
    assert record_payload(owned)['forecast_contract']['forecast']['particles'] == 512
    with pytest.raises(ValueError, match='content identity'): record_payload(another)
    with pytest.raises(TypeError): record_payload(owned)['source_catalog'][0]['i'] = 42


def test_imported_inventory_batch_keeps_every_owned_file_and_original_completion_check(tmp_path, monkeypatch):
    from experiments.pirc17 import formal_partial_imports as imports
    source = dict(result_sha256=digest('result'), settlement_sha256=digest('settlement'),
                  observation_sha256=digest('observation'), controller_elapsed_ns=42)
    path, record = publish(tmp_path, dict(imported_entries={'work':dict(directory=str(tmp_path/'item'),
        manifest=dict(artifact_sha256=digest('software artifact')), artifacts={'item':dict(bytes=1)},
        source_completion=source)}))
    ref = dict(path=str(path), content_sha256=record['sha256'], file_sha256=file_hash(path))
    reader = imports.ImportedOutputs.__new__(imports.ImportedOutputs)
    reader.reference = ref
    reader._manifest_reader = PinnedMetadata(ref, root=tmp_path)
    reader.record, reader.manifest = reader._manifest_reader.read(ref)
    reader.closed = SimpleNamespace(_state=SimpleNamespace(partial_imports=dict(software_only=True)))
    calls = []
    original = dict(source, directory=tmp_path/'source')
    reader.original = SimpleNamespace(read=lambda work:(calls.append('original bytes') or original))
    monkeypatch.setattr(imports, '_files', lambda *a:calls.append('owned bytes'))
    byte_checks = []
    actual_bytes = PinnedMetadata._bytes
    def counted(self, reference):
        byte_checks.append(reference['path'])
        return actual_bytes(self, reference)
    monkeypatch.setattr(PinnedMetadata, '_bytes', counted)
    work = dict(work_id='work')  # Explicit software native/domain seam.
    with module.metadata_batch(reader) as batch:
        for _ in range(20): assert reader.read(work)['imported'] is True
        assert calls == ['owned bytes', 'original bytes']*20 and len(byte_checks) == 1
        assert reader.original._metadata_batch is batch
        original['settlement_sha256'] = digest('changed original completion')
        with pytest.raises(ValueError, match='completion differs'): reader.read(work)
    assert len(byte_checks) == 2 and reader.original._metadata_batch is None
    assert calls == ['owned bytes', 'original bytes']*21
    reader._manifest_reader.read(ref)
    assert len(byte_checks) == 3  # No scope-wide exemption leaks into cold reads.
