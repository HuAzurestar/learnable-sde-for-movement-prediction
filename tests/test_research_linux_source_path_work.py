"""Actual Linux path-query work, payload observation and current-name aliases."""
import hashlib
import os
from pathlib import Path
import stat
import sys

import pytest

from infrastructure.research_files import source_file_hash
from infrastructure.research_store import ResearchError, encode
from tests.research_file_observation import observe_file


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='actual Linux source-path queries')


@pytest.mark.parametrize('depth', [2, 10])
def test_real_source_hash_does_not_rewalk_every_ancestor_at_each_boundary(tmp_path, monkeypatch, depth):
    root = tmp_path / 'source'
    parent = root
    for number in range(depth):
        parent /= 'level-' + str(number)
    parent.mkdir(parents=True)
    path = parent / '源文件 with spaces.py'
    content = '# π synthetic\r\nx=1\ry=2\n'.encode('utf-8')
    path.write_bytes(content)
    original_lstat, original_path_lstat, calls = os.lstat, Path.lstat, []

    def record(operation, args, information):
        calls.append({'operation': operation, 'path': os.fspath(args[0]),
                      'dev': information.st_dev, 'ino': information.st_ino})

    def lstat(*args, **kwargs):
        information = original_lstat(*args, **kwargs)
        record('lstat', args, information)
        return information

    def path_lstat(path, *args, **kwargs):
        # Python 3.10's pathlib accessor captures the original os.lstat;
        # Python 3.12 delegates to os.stat(..., follow_symlinks=False).
        # Forward the actual public boundary on both without replacing facts.
        information = original_path_lstat(path, *args, **kwargs)
        record('Path.lstat', (path,), information)
        return information

    with monkeypatch.context() as observed:
        observed.setattr(os, 'lstat', lstat)
        observed.setattr(Path, 'lstat', path_lstat)
        actual = source_file_hash(root, path)
    expected = hashlib.sha256(content.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n').encode()).hexdigest()
    assert actual == expected
    (tmp_path / 'source-path-work-observed.json').write_bytes(encode(
        {'depth': depth, 'actual_hash': actual, 'actual_python_nofollow_calls': calls,
         'limitation': 'Forwarded Path.lstat boundaries and os.lstat canonical walks, not hidden kernel queries or wall-clock speed.'}))
    assert len([call for call in calls if call['operation'] == 'Path.lstat']) == 6
    assert len(calls) == 6, 'three fresh root/file stamps must not repeatedly walk all path components'


def test_payload_observer_distinguishes_real_query_only_open(tmp_path, monkeypatch):
    path = tmp_path / 'query-only.py'
    path.write_bytes(b'# synthetic\n')
    payload_opens = []
    reads, handles = observe_file(monkeypatch, path, before_open=lambda: payload_opens.append(True))
    fd = os.open(path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        assert stat.S_ISREG(os.fstat(fd).st_mode)
    finally:
        os.close(fd)
    (tmp_path / 'query-observer-observed.json').write_bytes(encode(
        {'payload_opens_after_query_only': payload_opens[:], 'reads': reads[:], 'handles': handles[:]}))
    assert payload_opens == [], 'a real O_PATH query cannot open/read payload or trigger its mutation hook'
    assert source_file_hash(tmp_path, path) == hashlib.sha256(b'# synthetic\n').hexdigest()
    assert payload_opens == [True] and reads and handles == [True]


@pytest.mark.parametrize('scope', ['root', 'ancestor', 'member-parent'])
@pytest.mark.parametrize('phase', ['initial', 'after-first-read'])
def test_actual_same_inode_external_parent_redirect_is_not_an_old_fd_name(
        tmp_path, monkeypatch, scope, phase):
    anchor = tmp_path / 'anchor'
    root = anchor / 'source'
    path = root / 'nested' / 'source.py'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'# original synthetic source\n')
    redirect = {'root': root, 'ancestor': anchor, 'member-parent': path.parent}[scope]
    outside = tmp_path / 'outside'
    foreign = outside / path.relative_to(redirect)
    foreign.parent.mkdir(parents=True)
    os.link(path, foreign)
    before = path.stat()
    assert (before.st_dev, before.st_ino) == (foreign.stat().st_dev, foreign.stat().st_ino)
    # Move a member parent INSIDE root: the old FD's kernel name stays within
    # root although its current lexical name redirects to an external hardlink.
    parked = root / 'parked' if scope == 'member-parent' else tmp_path / 'parked'
    touched = []

    def change(*_):
        if not touched:
            redirect.rename(parked)
            redirect.symlink_to(outside, target_is_directory=True)
            touched.append(True)

    reads, handles = observe_file(monkeypatch, path,
        after_read=change if phase == 'after-first-read' else None)
    if phase == 'initial':
        change()
    try:
        with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
            source_file_hash(root, path)
        assert touched == [True]
        assert (path.stat().st_dev, path.stat().st_ino) == (before.st_dev, before.st_ino)
        if phase == 'initial':
            assert reads == [] and handles == [], 'external payload consumed at initial redirect'
        else:
            assert reads and handles == [True]
        (tmp_path / 'same-inode-parent-observed.json').write_bytes(encode(
            {'scope': scope, 'phase': phase, 'same_inode': True, 'reads': reads, 'handles': handles}))
    finally:
        if redirect.is_symlink():
            redirect.unlink()
