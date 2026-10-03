"""Actual Windows final-path queries; three uncached source boundaries."""

import os
from pathlib import Path

import pytest

from infrastructure.research_files import opened_regular_file


@pytest.mark.skipif(os.name != 'nt', reason='Windows native final-path query')
@pytest.mark.parametrize('name', ['source.bin', 'source with spaces.bin', '源文件-π.bin', 'MiXeD.bin'])
def test_actual_windows_path_query_once_at_each_complete_boundary(tmp_path, monkeypatch, name):
    import nt
    import ntpath
    root = tmp_path / 'source'
    root.mkdir()
    path = root / name
    path.write_bytes(b'actual bounded source')
    original = nt._getfinalpathname
    queries = []

    def resolve(value):
        result = original(value)
        queries.append((value, result))
        return result

    # ntpath holds a pre-bound native function on Python3.10/3.12.
    monkeypatch.setattr(nt, '_getfinalpathname', resolve)
    monkeypatch.setattr(ntpath, '_getfinalpathname', resolve)
    with opened_regular_file(root, path) as (stream, size, _):
        assert size == len(b'actual bounded source')
        assert stream.read(size + 1) == b'actual bounded source'
    assert len(queries) == 3, 'initial/opened/completion need one actual full query each, not duplicate prefix roundtrips: ' + str(queries)
    assert all(Path(value) == path for value, _ in queries)

