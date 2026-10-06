"""Fault actual event/head staging I/O; never mock grants/provider outcomes."""
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import stat
import sys

import infrastructure.research_store as store_module


@contextmanager
def audit_fault(store, kind, monkeypatch, *, stage="event-fsync", occurrence=1):
    original_write, original_fsync = store_module.atomic_write, os.fsync
    original_replace = store_module._publish_replace
    current, pending, fired = [None], [False], []
    seen = [0]
    def write(path, content, **options):
        location = None
        if Path(path).parent == store.path / "events" and json.loads(content).get("event_kind") == kind:
            seen[0] += 1
            if seen[0] == occurrence:
                location = "event"
                pending[0] = True
        elif Path(path) == store.path / "head.json" and pending[0]:
            location = "head"
        current[0] = location
        try:
            return original_write(path, content, **options)
        finally:
            current[0] = None
            if location == "head":
                pending[0] = False
    def fail(location, size=None):
        fired.append({"event_kind": kind, "stage": stage, "location": location, "size_bytes": size})
        raise OSError(errno.ENOSPC, "PRIVATE filesystem fault at " + str(store.path))
    def fsync(fd):
        info = os.fstat(fd)
        if not fired and current[0]:
            location = current[0]
            if stage == location + "-fsync" and stat.S_ISREG(info.st_mode):
                assert info.st_size > 0, "fault must reach an actual written staging file"
                fail(location, info.st_size)
            if stage == "event-directory-sync" and location == "event" and stat.S_ISDIR(info.st_mode):
                fail("event-directory")
        return original_fsync(fd)
    def replace(source, target):
        if not fired and current[0] == "event" and stage == "event-rename":
            info = os.fstat(source.stream.fileno())
            assert stat.S_ISREG(info.st_mode) and info.st_size > 0
            assert source.temporary.is_file()
            fail("event", info.st_size)
        return original_replace(source, target)
    with monkeypatch.context() as changes:
        changes.setattr(store_module, "atomic_write", write)
        changes.setattr(os, "fsync", fsync)
        changes.setattr(store_module, "_publish_replace", replace)
        yield fired


@contextmanager
def payload_opens(paths):
    targets = {os.path.normcase(os.path.abspath(path)) for path in paths}
    active, opened = [True], []
    def audit(event, params):
        if (active[0] and event == "open" and isinstance(params[0], (str, bytes))
                # O_PATH queries the name/identity but cannot read payload.
                and not (params[2] & getattr(os, "O_PATH", 0))
                and os.path.normcase(os.path.abspath(os.fsdecode(params[0]))) in targets):
            opened.append(os.fsdecode(params[0]))
    sys.addaudithook(audit)
    try:
        yield opened
    finally:
        active[0] = False
