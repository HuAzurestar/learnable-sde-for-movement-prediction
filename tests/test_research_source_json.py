"""Operation-local syntax reuse must not become a metadata authority cache."""

from contextvars import copy_context
from io import BytesIO
import threading

import pytest

from infrastructure import research_source_json as cached
from infrastructure.research_store import ResearchError, ResearchStore, encode


def content(note='a'):
    return encode({'note': note * (cached.JSON_BLOCK_BYTES + 10), 'values': [1, 2, 3]})


@pytest.fixture
def parses(monkeypatch):
    calls = []
    original = cached.read_json
    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(cached, 'read_json', observed)
    return calls


def read(data, verify=lambda: None):
    return cached.read_source_json(BytesIO(data), len(data), verify_identity=verify)


def test_reuse_still_reads_every_actual_byte_in_bounded_chunks_and_returns_detached_values(parses):
    data = content()
    class Observed(BytesIO):
        def __init__(self):
            super().__init__(data)
            self.calls = []
        def read(self, size=-1):
            self.calls.append(size)
            return super().read(size)
    verified = []
    with cached.source_json_scope():
        first = read(data)
        first['values'].append(99)
        stream = Observed()
        second = cached.read_source_json(stream, len(data), verify_identity=lambda: verified.append(True))
        assert second['values'] == [1, 2, 3]
        assert stream.tell() == len(data) and sum(stream.calls) == len(data) + 1
        assert all(0 <= size <= cached.JSON_BLOCK_BYTES for size in stream.calls)
        second['values'].clear()
        assert read(data)['values'] == [1, 2, 3]
    assert len(parses) == 1 and verified == [True]


def test_pool_cannot_escape_an_operation_or_reuse_changed_bytes(parses):
    with cached.source_json_scope():
        assert read(content('a'))['note'].startswith('a')
        assert read(content('b'))['note'].startswith('b')
    assert cached._active.get() is None
    with cached.source_json_scope():
        read(content('a'))
    assert len(parses) == 3


def test_physical_refusal_vetoes_a_warm_hit(parses):
    def changed():
        raise ResearchError('CORRUPT_ARTIFACT', 'physical identity changed')
    with cached.source_json_scope():
        read(content())
        with pytest.raises(ResearchError, match='physical identity'):
            read(content(), changed)
    assert len(parses) == 1


def test_failed_post_parse_physical_check_never_seeds_a_later_hit(parses):
    checks = []
    def late_change():
        checks.append(True)
        if len(checks) == 3:
            raise ResearchError('CORRUPT_ARTIFACT', 'changed during final decode')
    with cached.source_json_scope():
        with pytest.raises(ResearchError, match='final decode'):
            read(content(), late_change)
        assert not cached._active.get()['values']
        assert read(content())['values'] == [1, 2, 3]
    assert len(parses) == 2


@pytest.mark.parametrize('change', [-1, 1])
def test_growth_and_truncation_refuse_before_a_warm_value_can_escape(parses, change):
    data = content()
    with cached.source_json_scope():
        read(data)
        changed = data[:-1] if change < 0 else data + b' '
        with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
            cached.read_source_json(BytesIO(changed), len(data), verify_identity=lambda: None)
    assert len(parses) == 1


def test_entry_count_and_byte_quota_are_bounded(monkeypatch, parses):
    monkeypatch.setattr(cached, 'MAXIMUM_ENTRIES', 1)
    monkeypatch.setattr(cached, 'MAXIMUM_POOL_BYTES', len(content()))
    with cached.source_json_scope():
        read(content('a'))
        read(content('b'))
        assert len(cached._active.get()['values']) == 1
        assert cached._active.get()['bytes'] == len(content())
        read(content('a'))
    assert len(parses) == 3


def test_copied_context_in_another_thread_cannot_borrow_owner_pool(parses):
    results = []
    with cached.source_json_scope():
        read(content())
        context = copy_context()
        worker = threading.Thread(target=lambda: results.append(context.run(read, content())))
        worker.start()
        worker.join(10)
        assert not worker.is_alive()
    assert results[0]['values'] == [1, 2, 3]
    assert len(parses) == 2


@pytest.mark.parametrize('name,uses_pool', [
    ('pirc28-study-definition-example.json', True), ('pirc28-source-epoch-bridge-example.json', True),
    ('authorization-example.json', False), ('policy-example.json', False), ('head.json', False)])
def test_store_only_opts_in_definition_and_bridge_syntax(tmp_path, monkeypatch, name, uses_pool):
    directory = tmp_path / 'manifests'
    directory.mkdir()
    path = directory / name
    path.write_bytes(content())
    calls = []
    original = cached.read_source_json
    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(cached, 'read_source_json', observed)
    with cached.source_json_scope():
        assert ResearchStore._json(path)['values'] == [1, 2, 3]
        assert ResearchStore._json(path)['values'] == [1, 2, 3]
    assert len(calls) == (2 if uses_pool else 0)
