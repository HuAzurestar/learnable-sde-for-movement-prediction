"""Actual statistical worker input reads must enforce the admitted byte cap."""

from pathlib import Path
import sys

import pytest

from experiments.pirc25 import compare_worker
from infrastructure.research_store import digest, encode


def inputs(tmp_path, monkeypatch):
    bundle = {'cells': []}
    bundle['bundle_hash'] = digest(bundle)
    request = {'computation_ref': {'source_bundle_hash': bundle['bundle_hash']},
               'runtime_code_hash': 'fixture-code', 'paper_identity': 'fixture-paper',
               'resource_plan': {'maximum_operations': 1}}
    paths = {name: tmp_path / (name + '.json') for name in ('request', 'bundle')}
    paths['request'].write_bytes(encode(request))
    paths['bundle'].write_bytes(encode(bundle))
    monkeypatch.setattr(sys, 'argv', ['compare_worker', str(paths['request']), str(paths['bundle']),
                                    str(tmp_path), str(tmp_path / 'unused-output.json')])
    monkeypatch.setattr(compare_worker, 'code_hash', lambda: 'fixture-code')
    monkeypatch.setattr(compare_worker, 'paper_identity', lambda _: 'fixture-paper')

    def stop_before_computation(*args):
        raise LookupError('both input reads completed')

    monkeypatch.setattr(compare_worker, 'comparison_plan', stop_before_computation)
    return paths


@pytest.mark.parametrize('name', ['request', 'bundle'])
def test_actual_worker_reads_input_with_explicit_byte_bound(tmp_path, monkeypatch, name):
    paths = inputs(tmp_path, monkeypatch)
    target = paths[name]
    original = Path.open
    reads = []

    class BoundedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, key):
            return getattr(self.stream, key)

        def read(self, size=-1):
            assert 0 <= size <= compare_worker.MAX_INPUT_BYTES + 1, 'worker input read is unbounded after stat'
            reads.append(size)
            return self.stream.read(size)

    def guarded(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return BoundedStream(stream) if path == target else stream

    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises(LookupError, match='both input reads completed'):
        compare_worker.main()
    assert reads
    assert not (tmp_path / 'unused-output.json').exists()


@pytest.mark.parametrize('name', ['request', 'bundle'])
def test_growth_after_size_check_is_rejected_before_json_parse(tmp_path, monkeypatch, name):
    paths = inputs(tmp_path, monkeypatch)
    target = paths[name]
    original = Path.open
    monkeypatch.setattr(compare_worker, 'MAX_INPUT_BYTES', 512)
    reads = []

    class GrowingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, key):
            return getattr(self.stream, key)

        def read(self, size=-1):
            with original(target, 'ab') as output:
                output.write(b'x' * 513)
            reads.append(size)
            return self.stream.read(size)

    def growing(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return GrowingStream(stream) if path == target else stream

    monkeypatch.setattr(Path, 'open', growing)
    with pytest.raises(ValueError, match='RESOURCE_PLAN_REJECTED'):
        compare_worker.main()
    assert reads and all(0 <= size <= 513 for size in reads)
    assert not (tmp_path / 'unused-output.json').exists()


def test_exact_input_limit_is_accepted_without_truncation(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    limit = max(path.stat().st_size for path in paths.values())
    monkeypatch.setattr(compare_worker, 'MAX_INPUT_BYTES', limit)
    with pytest.raises(LookupError, match='both input reads completed'):
        compare_worker.main()
