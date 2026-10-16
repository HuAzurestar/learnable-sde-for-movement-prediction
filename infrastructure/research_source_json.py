"""Bounded decoded syntax reuse, never publication or execution authority.

Only PIRC-28 immutable definition/bridge files opt in. Every hit still reads all
actual bytes from the currently pinned descriptor and verifies its physical
identity. No permission, grant, journal, attempt, policy or balance is cached.
The pool exists only during one source-preflight call and cannot escape it.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import os
import sys
import threading

from .research_json import JSON_BLOCK_BYTES, read_json
from .research_store import ResearchError

MAXIMUM_POOL_BYTES = 8 * 2**20
MAXIMUM_ENTRY_BYTES = 2 * 2**20
MAXIMUM_ENTRIES = 32
_active = ContextVar('pirc28_immutable_json_decode_scope', default=None)


@contextmanager
def source_json_scope():
    """Fresh operation-local pool, also suitable as a function decorator."""
    owner = (os.getpid(), threading.get_ident())
    if _active.get() is not None and _active.get()['owner'] == owner:
        yield
        return
    token = _active.set({'values': OrderedDict(), 'bytes': 0, 'owner': owner})
    try:
        yield
    finally:
        _active.reset(token)


def read_source_json(stream, size, *, verify_identity):
    pool = _active.get()
    if (pool is None or pool['owner'] != (os.getpid(), threading.get_ident())
            or not JSON_BLOCK_BYTES < size <= MAXIMUM_ENTRY_BYTES):
        return read_json(stream, size, verify_identity=verify_identity)
    # Never retain an encoded file-sized buffer or trust mtime/size as content.
    hasher, remaining = hashlib.sha256(), size
    while remaining:
        chunk = stream.read(min(JSON_BLOCK_BYTES, remaining))
        if not chunk or len(chunk) > remaining:
            raise ResearchError('CORRUPT_ARTIFACT', 'immutable JSON was truncated during read')
        remaining -= len(chunk)
        hasher.update(chunk)
    if stream.read(1):
        raise ResearchError('CORRUPT_ARTIFACT', 'immutable JSON grew during read')
    verify_identity()
    # Preserve interpreter-dependent integer/depth refusal semantics too.
    key = (size, hasher.digest(), getattr(sys, 'get_int_max_str_digits', lambda: 0)(),
           sys.getrecursionlimit())
    entries = pool['values']
    if key in entries:
        entries.move_to_end(key)
        return deepcopy(entries[key])
    stream.seek(0)
    value = read_json(stream, size, verify_identity=verify_identity)
    # A failed final physical check must never seed a later cache hit with a
    # value parsed after the bytes used for its key were changed.
    verify_identity()
    while entries and (pool['bytes'] + size > MAXIMUM_POOL_BYTES or len(entries) >= MAXIMUM_ENTRIES):
        evicted, _ = entries.popitem(last=False)
        pool['bytes'] -= evicted[0]
    if size <= MAXIMUM_POOL_BYTES:
        entries[key] = deepcopy(value)
        pool['bytes'] += size
    return value
