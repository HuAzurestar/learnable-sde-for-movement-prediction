"""Metadata contention tolerance does not remove exclusion or explicit limits."""
import os
from types import SimpleNamespace

import pytest

import infrastructure.research_store as module
from infrastructure.research_store import ResearchError, file_lock


def contention(monkeypatch):
    clock = SimpleNamespace(now=0.)
    def sleep(seconds):
        clock.now += seconds
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    if os.name == 'nt':
        import msvcrt
        def acquire(descriptor, mode, count):
            if mode != msvcrt.LK_UNLCK and clock.now < 20:
                raise OSError('simulated still-owned writer')
        monkeypatch.setattr(msvcrt, 'locking', acquire)
    else:
        import fcntl
        def acquire(descriptor, mode):
            if mode != fcntl.LOCK_UN and clock.now < 20:
                raise OSError('simulated still-owned writer')
        monkeypatch.setattr(fcntl, 'flock', acquire)
    return clock


def test_default_metadata_wait_tolerates_a_legitimate_writer_over_fifteen_seconds(tmp_path, monkeypatch):
    clock = contention(monkeypatch)
    with file_lock(tmp_path / '.writer.lock'):
        assert 20 <= clock.now < 21


def test_explicit_short_wait_still_refuses_the_owned_writer(tmp_path, monkeypatch):
    contention(monkeypatch)
    with pytest.raises(ResearchError, match='WRITER_BUSY'):
        with file_lock(tmp_path / '.writer.lock', timeout=1):
            pytest.fail('writer exclusion was bypassed')
