"""Observe real path/descriptor reads across old and hardened reader wiring."""

import os
from pathlib import Path


def observe_file(monkeypatch, target, *, before_open=None, before_read=None, after_read=None):
    target = Path(target)
    original_path_open, original_os_open, original_fdopen = Path.open, os.open, os.fdopen
    descriptors = set()
    reads, handles = [], []

    class Stream:
        def __init__(self, actual):
            self.actual = actual
            self.observed = False

        def __enter__(self):
            self.actual.__enter__()
            return self

        def __exit__(self, *args):
            return self.actual.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def read(self, size=-1):
            if before_read is not None:
                before_read(self.actual, size)
            content = self.actual.read(size)
            if not self.observed:
                handles.append(True)
                self.observed = True
            reads.append({"size": size, "bytes": len(content)})
            if after_read is not None:
                after_read(self.actual, content)
            return content

    def path_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        selected = path == target and mode in {"rb", "r"}
        if selected and before_open is not None:
            before_open()
        actual = original_path_open(path, *args, **kwargs)
        return Stream(actual) if selected else actual

    def os_open(path, *args, **kwargs):
        selected = Path(path) == target
        if selected and before_open is not None:
            before_open()
        fd = original_os_open(path, *args, **kwargs)
        if selected:
            descriptors.add(fd)
        return fd

    def fdopen(fd, *args, **kwargs):
        selected = fd in descriptors
        descriptors.discard(fd)  # Do not wrap an unrelated later descriptor reuse.
        actual = original_fdopen(fd, *args, **kwargs)
        return Stream(actual) if selected else actual

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(os, "open", os_open)
    monkeypatch.setattr(os, "fdopen", fdopen)
    return reads, handles
