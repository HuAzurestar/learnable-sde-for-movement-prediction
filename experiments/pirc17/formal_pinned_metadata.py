"""Parse a pinned metadata file once; recheck ALL its bytes on every read.

This is a reader-local immutable JSON snapshot, not a stat/mtime cache, source
admission, domain verdict or authorization. Models and arrays still go through
their independent consumers. No global cache and no file/writer mutation.
"""
import hashlib
from pathlib import Path

from . import formal_budget as budget
from .formal_carryover import MAX_ROOT_BYTES
from .formal_scope_cache import _Dict, _freeze, _immutable
from .protocol_core import canonical, decode, sha256, under, unpack


class OwnedRecord(_Dict):
    """Complete validated, private immutable envelope; no domain PASS flag.

    Its bytes and payload belong to this snapshot. Mutable external records
    must still be reserialized/unpacked; a claimed header hash cannot hit it.
    """
    __slots__ = ('_encoded', '_payload')
    __setattr__ = __delattr__ = _immutable

    def __init__(self, record):
        if hasattr(self, '_encoded'):
            _immutable()
        raw = canonical(record)
        copied = decode(raw)
        unpack(copied)
        dict.__init__(self, _freeze(copied))
        object.__setattr__(self, '_encoded', raw)
        object.__setattr__(self, '_payload', self['payload'])


def record_payload(record):
    return record._payload if type(record) is OwnedRecord else unpack(record)


def record_bytes(record):
    return record._encoded if type(record) is OwnedRecord else canonical(record)


def mutable_record(record):
    """Fresh deserializer input where legacy domain consumers mutate copies."""
    return decode(record._encoded) if type(record) is OwnedRecord else record


class PinnedMetadata:
    """Own verified meaning; never reuse mutable caller data or producer PASS.

    Later reads retain the original byte pin and bounded regular-file/path
    checks. Identical pinned bytes have the same already validated meaning,
    so reparsing/recanonicalizing a large inventory per item adds no proof.
    Normal mutations of returned nested containers are rejected. As with the
    owned matrix, malicious in-process base-class mutation is not a boundary.
    """
    __slots__ = ('reference', 'root', '_record', '_payload')

    def __setattr__(self, name, value):
        raise TypeError('pinned metadata snapshot is immutable')

    def __delattr__(self, name):
        raise TypeError('pinned metadata snapshot is immutable')

    def __init__(self, reference, *, root):
        if hasattr(self, '_record'):
            raise TypeError('pinned metadata snapshot is immutable')
        object.__setattr__(self, 'reference', _freeze(dict(reference)))
        object.__setattr__(self, 'root', Path(root))
        raw = self._bytes(reference)
        record = decode(raw)
        unpack(record, expected_sha256=sha256(reference['content_sha256']))
        object.__setattr__(self, '_record', OwnedRecord(record))
        object.__setattr__(self, '_payload', self._record['payload'])

    def _bytes(self, reference):
        budget._fields(reference, ('path', 'content_sha256', 'file_sha256'))
        # Canonical complete reference comparison, including bool/int types.
        if canonical(reference) != canonical(self.reference):
            raise ValueError('pinned metadata reference changed')
        path = Path(reference['path'])
        if (not path.is_absolute() or path.is_symlink()
                or str(path.resolve()) != str(path) or not path.is_relative_to(self.root)
                or under(self.root, path.relative_to(self.root).as_posix()) != path):
            raise ValueError('pinned metadata must remain in its canonical owned root')
        if not path.is_file() or path.stat().st_size > MAX_ROOT_BYTES:
            raise ValueError('bounded regular pinned metadata required')
        with path.open('rb') as stream:
            raw = stream.read(MAX_ROOT_BYTES + 1)
        if len(raw) > MAX_ROOT_BYTES:
            raise ValueError('pinned metadata grew beyond its bound')
        if hashlib.sha256(raw).hexdigest() != sha256(reference['file_sha256']):
            raise ValueError('pinned metadata bytes changed')
        return raw

    def read(self, reference):
        """Recheck current complete bytes, returning immutable parsed meaning."""
        self._bytes(reference)
        return self._record, self._payload
