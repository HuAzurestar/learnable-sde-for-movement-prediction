"""Actual Windows final-path queries; three uncached source boundaries."""

import os
from pathlib import Path
import subprocess

import pytest

from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError
from tests.research_file_observation import observe_file


@pytest.mark.skipif(os.name != 'nt', reason='Windows native final-path query')
@pytest.mark.parametrize('name', ['source.bin', 'source with spaces.bin', '源文件-π.bin', 'MiXeD.bin', 'case-alias.bin'])
def test_actual_windows_path_query_once_at_each_complete_boundary(tmp_path, monkeypatch, name):
    import nt
    import ntpath
    root = tmp_path / 'source'
    root.mkdir()
    path = root / name
    path.write_bytes(b'actual bounded source')
    if name == 'case-alias.bin':
        root, path = Path(str(root).swapcase()), Path(str(path).swapcase())
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


@pytest.mark.skipif(os.name != 'nt', reason='Windows native extended path spelling')
def test_explicit_extended_paths_keep_legacy_namespace_behavior(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    path = root / 'source.bin'
    path.write_bytes(b'actual bounded source')
    extended_root, extended_path = Path('\\\\?\\' + str(root)), Path('\\\\?\\' + str(path))
    assert extended_path.resolve().is_relative_to(extended_root)
    with opened_regular_file(extended_root, extended_path) as (stream, size, _):
        assert stream.read(size + 1) == b'actual bounded source'


@pytest.mark.skipif(os.name != 'nt', reason='Windows native root/ancestor junction')
@pytest.mark.parametrize('kind', ['root', 'ancestor'])
@pytest.mark.parametrize('phase', ['initial', 'before-open', 'after-read'])
def test_native_junction_to_moved_original_is_not_trusted(tmp_path, monkeypatch, kind, phase):
    root = tmp_path / 'holder' / 'source'
    root.mkdir(parents=True)
    path = root / 'source.bin'
    path.write_bytes(b'same original inode, no trusted lexical root')
    alias = root if kind == 'root' else root.parent
    relocated = tmp_path / 'relocated'
    installed = []

    def redirect():
        assert not installed
        try:
            alias.rename(relocated)
        except OSError as exc:
            pytest.skip('native directory rename while source is open unavailable: ' + str(exc))
        env = dict(os.environ, PIRC_TEST_JUNCTION_PATH=str(alias),
                   PIRC_TEST_JUNCTION_TARGET=str(relocated))
        try:
            subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                '$ErrorActionPreference = "Stop"; New-Item -ItemType Junction -Path '
                '$env:PIRC_TEST_JUNCTION_PATH -Target $env:PIRC_TEST_JUNCTION_TARGET | Out-Null'],
                env=env, check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.CalledProcessError) as exc:
            relocated.rename(alias)
            pytest.skip('native junction capability unavailable: ' + str(exc))
        installed.append(True)

    try:
        if phase == 'initial':
            redirect()
        reads, _ = observe_file(monkeypatch, path,
            before_open=redirect if phase == 'before-open' else None,
            after_read=(lambda stream, content: redirect()) if phase == 'after-read' else None)
        with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA|CORRUPT_ARTIFACT'):
            with opened_regular_file(root, path) as (stream, size, _):
                stream.read(size + 1)
        assert installed == [True]
        assert (len(reads) == 1) if phase == 'after-read' else reads == []
    finally:
        if installed:
            # Remove only the newly created junction entry, never its target.
            # Both exact paths were created inside this fresh synthetic fixture.
            assert alias.is_relative_to(tmp_path) and relocated.is_relative_to(tmp_path)
            alias.rmdir()
            relocated.rename(alias)
