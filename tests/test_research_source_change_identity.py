"""Actual same-inode source writes/restored mtime, no identity/result stand-ins."""
import hashlib
import os

import pytest

from infrastructure.research_files import opened_regular_file, source_file_hash
from infrastructure.research_store import ResearchError, ResearchStore, encode
from tests.research_file_observation import observe_file


ORIGINAL = b'# original\n'
CHANGED = b'# modified\n'


def change_source(path, observed):
    before = path.stat()
    assert path.write_bytes(CHANGED) == len(ORIGINAL) == len(CHANGED)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    observed.update(before={key: getattr(before, key) for key in keys},
                    after={key: getattr(after, key) for key in keys})


@pytest.mark.parametrize('phase', ['before-open', 'after-first-read', 'explicit-verify', 'context-close'])
def test_regular_source_rejects_actual_equal_size_write_with_restored_mtime(tmp_path, monkeypatch, phase):
    path = tmp_path / 'synthetic-source.py'
    path.write_bytes(ORIGINAL)
    observed = {'platform': os.name, 'phase': phase, 'error': None}
    original_open = type(path).open
    touched = []
    def change():
        if not touched:
            touched.append(True)
            change_source(path, observed)
    reads, handles = observe_file(monkeypatch, path,
        before_open=change if phase == 'before-open' else None,
        after_read=(lambda stream, content: change()) if phase == 'after-first-read' else None)
    try:
        with opened_regular_file(tmp_path, path, expected_size=len(ORIGINAL)) as (stream, size, verify):
            observed['value'] = stream.read(size + 1).decode()
            if phase in {'explicit-verify', 'context-close'}:
                change()
            if phase == 'explicit-verify':
                verify()
    except ResearchError as exc:
        observed['error'] = exc.code
    observed.update(reads=list(reads), handles=list(handles))
    with original_open(path, 'rb') as stream:
        observed['changed_hash'] = hashlib.sha256(stream.read()).hexdigest()
    (tmp_path / 'source-identity-observed.json').write_bytes(encode(observed))
    assert touched == [True] and observed['changed_hash'] == hashlib.sha256(CHANGED).hexdigest()
    assert observed['error'] == 'CORRUPT_ARTIFACT', observed
    if phase == 'before-open':
        assert reads == [], 'changed source bytes consumed before the opened-handle guard'


@pytest.mark.parametrize('phase', ['before-open', 'after-first-read'])
def test_actual_source_hash_cannot_return_after_restored_mtime_write(tmp_path, monkeypatch, phase):
    path = tmp_path / 'synthetic-source.py'
    path.write_bytes(ORIGINAL)
    observed = {'platform': os.name, 'phase': phase, 'error': None}
    touched = []
    def change():
        if not touched:
            touched.append(True)
            change_source(path, observed)
    reads, _ = observe_file(monkeypatch, path,
        before_open=change if phase == 'before-open' else None,
        after_read=(lambda stream, content: change()) if phase == 'after-first-read' else None)
    try:
        observed['returned_hash'] = source_file_hash(tmp_path, path)
    except ResearchError as exc:
        observed['error'] = exc.code
    observed['reads'] = list(reads)
    (tmp_path / 'source-hash-observed.json').write_bytes(encode(observed))
    assert touched == [True]
    assert observed['error'] == 'CORRUPT_ARTIFACT', observed
    if phase == 'before-open':
        assert reads == []


def test_ordinary_source_reads_seek_and_hash_remain_valid(tmp_path):
    path = tmp_path / '源文件 with spaces.py'
    path.write_bytes(ORIGINAL)
    with opened_regular_file(tmp_path, path) as (stream, size, verify):
        assert size == len(ORIGINAL) and stream.read(size + 1) == ORIGINAL
        stream.seek(0)
        assert stream.read(size + 1) == ORIGINAL
        verify()
    assert source_file_hash(tmp_path, path) == hashlib.sha256(ORIGINAL).hexdigest()


def test_actual_metadata_loader_rejects_same_size_write_after_read_with_restored_mtime(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path, 'metadata-change', initialize=True)
    store.publish('synthetic', {'number': 1})
    path = store.path / 'manifests/synthetic.json'
    before, touched = path.stat(), []
    def change(*_):
        if not touched:
            touched.append(True)
            path.write_bytes(encode({'number': 2}))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    reads, handles = observe_file(monkeypatch, path, after_read=change)
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        store._json(path)
    assert path.stat().st_size == before.st_size and path.stat().st_ino == before.st_ino
    assert touched == [True] and reads and handles == [True]


@pytest.mark.parametrize('purpose', ['fit', 'select', 'validate', 'evaluate'])
@pytest.mark.parametrize('operation', ['read', 'verify'])
def test_actual_authorized_provider_never_completes_after_same_size_restored_mtime_write(
        tmp_path, monkeypatch, purpose, operation):
    from tests.test_research_read_authorization import provider_source
    store, ledger, grant, content = provider_source(tmp_path, purpose)
    path = tmp_path / 'block.bin'
    before, touched = path.stat(), []
    def change(*_):
        if not touched:
            touched.append(True)
            path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    reads, handles = observe_file(monkeypatch, path, after_read=change)
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        getattr(ledger, operation)('reserved', 'test-block', purpose=purpose,
            authorization_id=grant['authorization_id'], data_root=tmp_path)
    assert path.stat().st_size == before.st_size and path.stat().st_ino == before.st_ino
    assert touched == [True] and reads and handles == [True]
    assert store.events()[-1]['event_kind'] == 'READ_FAILED'
