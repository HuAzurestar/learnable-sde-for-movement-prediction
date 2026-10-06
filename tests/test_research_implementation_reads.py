"""Actual source handles, mutations and legacy identity for plugin factories."""

import builtins
import hashlib
import inspect
import linecache
import os
from pathlib import Path
import sys
import tokenize
from types import ModuleType
import uuid

import pytest

from application import research_registry as core
from infrastructure.research_store import ResearchError, digest


SOURCE = b'def factory(config):\n    return 1\n'


def load_source(monkeypatch, path, payload=SOURCE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    name = 'pirc38_source_' + uuid.uuid4().hex
    module = ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    linecache.cache.pop(str(path), None)
    return module


def legacy_hash(builder):
    """The previous public identity on an unchanged, legitimate source file."""
    payload = Path(inspect.getsourcefile(builder)).read_bytes()
    codes = ({'classes': core._class_identity(builder)} if inspect.isclass(builder)
             else {'callable': core._function_identity(builder)})
    return digest({'source': inspect.getsource(builder), 'module': builder.__module__,
        'qualname': builder.__qualname__, 'defining_module_hash':
        hashlib.sha256(payload.decode('utf-8').replace('\r\n', '\n').encode()).hexdigest(),
        'codes': codes})


def observe_reads(monkeypatch, path, after_read=None):
    """Wrap real streams only; never invent metadata, bytes or file identities."""
    identity = path.stat()
    reads, opens = [], []
    original_path, original_builtin, original_fd = Path.open, builtins.open, os.fdopen
    original_token_open = tokenize._builtin_open
    changed = False

    class Stream:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def __enter__(self):
            self.actual.__enter__()
            return self

        def __exit__(self, *args):
            return self.actual.__exit__(*args)

        def read(self, size=-1):
            nonlocal changed
            value = self.actual.read(size)
            reads.append(('read', size, len(value)))
            if after_read is not None and not changed:
                changed = True
                after_read()
            return value

        def readline(self, size=-1):
            value = self.actual.readline(size)
            reads.append(('readline', size, len(value)))
            return value

        def read1(self, size=-1):
            value = self.actual.read1(size)
            reads.append(('read1', size, len(value)))
            return value

    def wrap(actual):
        stat = os.fstat(actual.fileno())
        if ((stat.st_dev, stat.st_ino) == (identity.st_dev, identity.st_ino)
                and actual.readable() and not actual.writable()):
            opens.append(actual.name)
            return Stream(actual)
        return actual

    monkeypatch.setattr(Path, 'open', lambda *a, **k: wrap(original_path(*a, **k)))
    monkeypatch.setattr(builtins, 'open', lambda *a, **k: wrap(original_builtin(*a, **k)))
    monkeypatch.setattr(tokenize, '_builtin_open', lambda *a, **k: wrap(original_token_open(*a, **k)))
    monkeypatch.setattr(os, 'fdopen', lambda *a, **k: wrap(original_fd(*a, **k)))
    return opens, reads


def test_inspection_uses_one_bounded_physical_source_read(monkeypatch, tmp_path):
    path = tmp_path / 'source' / 'factory.py'
    module = load_source(monkeypatch, path)
    expected = legacy_hash(module.factory)
    linecache.cache.pop(str(path), None)
    opens, reads = observe_reads(monkeypatch, path)
    assert core.implementation_hash(module.factory) == expected
    assert len(opens) == 1, opens
    assert reads and all(0 < size <= 4 * 1024 * 1024 + 1 for _, size, _ in reads), reads


@pytest.mark.parametrize('mutation', ['grow', 'same-size', 'restored-mtime'])
def test_actual_source_change_after_bounded_read_is_denied(monkeypatch, tmp_path, mutation):
    path = tmp_path / 'source' / 'factory.py'
    module = load_source(monkeypatch, path)

    def change():
        before = path.stat()
        payload = SOURCE.replace(b'return 1', b'return 2')
        if mutation == 'grow':
            payload += b'#' + b'x' * (4 * 1024 * 1024) + b'\n'
        path.write_bytes(payload)
        information = path.stat()
        times = ((before.st_atime_ns, before.st_mtime_ns) if mutation == 'restored-mtime'
                 else (information.st_atime_ns, information.st_mtime_ns + 10_000_000))
        os.utime(path, ns=times)

    _, reads = observe_reads(monkeypatch, path, change)
    with pytest.raises(ResearchError):
        core.implementation_hash(module.factory)
    assert reads and all(0 < size <= 4 * 1024 * 1024 + 1 for _, size, _ in reads), reads


@pytest.mark.parametrize('restore_mtime', [False, True])
def test_actual_source_change_during_code_identity_is_denied(monkeypatch, tmp_path, restore_mtime):
    path = tmp_path / 'source' / 'factory.py'
    module = load_source(monkeypatch, path)
    original = core._function_identity

    def capture(function, *args):
        result = original(function, *args)
        before = path.stat()
        path.write_bytes(SOURCE.replace(b'return 1', b'return 2'))
        information = path.stat()
        times = ((before.st_atime_ns, before.st_mtime_ns) if restore_mtime
                 else (information.st_atime_ns, information.st_mtime_ns + 10_000_000))
        os.utime(path, ns=times)
        return result

    monkeypatch.setattr(core, '_function_identity', capture)
    with pytest.raises(ResearchError):
        core.implementation_hash(module.factory)


@pytest.mark.parametrize('redirect', ['file', 'root', 'ancestor'])
def test_actual_source_alias_denied_before_source_bytes(monkeypatch, tmp_path, redirect):
    root = tmp_path / 'container' / 'source'
    path = root / 'factory.py'
    module = load_source(monkeypatch, path)
    target = {'file': path, 'root': root, 'ancestor': root.parent}[redirect]
    external = tmp_path / ('external-' + redirect)
    target.rename(external)
    try:
        target.symlink_to(external, target_is_directory=redirect != 'file')
    except OSError as exc:
        external.rename(target)
        pytest.skip('native symlink capability unavailable: ' + str(exc))
    opens, reads = observe_reads(monkeypatch, path)
    with pytest.raises(ResearchError):
        core.implementation_hash(module.factory)
    assert opens == [] and reads == []


@pytest.mark.parametrize('ending', [b'\n', b'\r\n', b'\r'])
def test_legal_exact_quota_and_newline_identity_retained(monkeypatch, tmp_path, ending):
    payload = SOURCE.replace(b'\n', ending)
    # Exact historical 4 MiB module quota; large comments are not factory code.
    payload += b'#' + b'x' * (4 * 1024 * 1024 - len(payload) - 1)
    path = tmp_path / 'source' / 'factory.py'
    module = load_source(monkeypatch, path, payload)
    assert core.implementation_hash(module.factory) == legacy_hash(module.factory)


@pytest.mark.parametrize('name', ['factory', 'wrapped', 'async_factory', 'AsyncFactory', 'Nested', 'Tagged', 'lambda_factory'])
def test_legal_factory_forms_keep_original_hash(monkeypatch, tmp_path, name):
    payload = b'''from functools import wraps
def factory(config, offset=2):
    return offset
@wraps(factory)
def wrapped(config):
    return factory(config)
async def async_factory(config):
    return config
class AsyncFactory:
    async def __call__(self, config):
        return config
def nested():
    class Nested:
        def __new__(cls, config):
            return config
    return Nested
Nested = nested()
def tag(value):
    return value
@tag
class Tagged:
    @property
    def constant(self):
        return 2
lambda_factory = lambda config: config
'''
    module = load_source(monkeypatch, tmp_path / 'source' / 'factory.py', payload)
    factory = getattr(module, name)
    assert core.implementation_hash(factory) == legacy_hash(factory)


def test_cross_module_wrapping_keeps_original_identity(monkeypatch, tmp_path):
    original = load_source(monkeypatch, tmp_path / 'original' / 'factory.py')
    payload = ('from functools import wraps\nfrom ' + original.__name__ +
        ' import factory as original\n@wraps(original)\ndef factory(config):\n'
        '    return original(config)\n').encode()
    module = load_source(monkeypatch, tmp_path / 'wrapper' / 'factory.py', payload)
    assert core.implementation_hash(module.factory) == legacy_hash(module.factory)


def test_stale_inspect_cache_is_not_source_authority(monkeypatch, tmp_path):
    path = tmp_path / 'source' / 'factory.py'
    module = load_source(monkeypatch, path)
    previous = legacy_hash(module.factory)  # Populate the real inspect cache.
    information = path.stat()
    path.write_bytes(SOURCE.replace(b'return 1', b'return 2'))
    os.utime(path, ns=(information.st_atime_ns, information.st_mtime_ns))
    cached_identity = legacy_hash(module.factory)
    linecache.cache.pop(str(path), None)
    current_identity = legacy_hash(module.factory)
    assert current_identity != cached_identity and current_identity != previous
    # Reinstall the genuine previous cached source, not a fake inspect result.
    linecache.cache[str(path)] = (len(SOURCE), path.stat().st_mtime,
        SOURCE.decode().splitlines(keepends=True), str(path))
    assert core.implementation_hash(module.factory) == current_identity
