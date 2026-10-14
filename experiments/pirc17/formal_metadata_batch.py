"""Finite restoration-local reuse, never a source/domain admission receipt.

The normal reader still checks complete current bytes on EVERY consumption.
Only an explicitly bracketed restoration batch may reuse a private immutable
metadata snapshot between a full entry check and a full exit check. The caller
MUST exit successfully before publishing/binding admission or forwarding the
collection. Per-item artifact, completion and scientific domain checks are not
cached here. No stat/mtime trust, global cache, arrays or producer PASS flags.
"""
from contextlib import contextmanager
from threading import get_ident

from .formal_pinned_metadata import PinnedMetadata
from .protocol_core import canonical

MAX_READERS = 256
MAX_OWNERS = 256
MAX_VALIDATIONS = 64


class _Batch:
    def __init__(self):
        self.thread = get_ident()
        self.status = 'open'
        self.readers = {}
        self.owners = []
        self.validations = {}

    def _open(self):
        if self.status != 'open' or self.thread != get_ident():
            raise ValueError('metadata batch must remain open on its owner thread')

    def attach(self, owner):
        self._open()
        previous = getattr(owner, '_metadata_batch', None)
        if previous is self:
            return
        if previous is not None or len(self.owners) >= MAX_OWNERS:
            raise ValueError('overlapping or unbounded metadata batch owners')
        owner._metadata_batch = self
        self.owners.append(owner)

    def read(self, reader, reference):
        self._open()
        if type(reader) is not PinnedMetadata:
            raise ValueError('metadata batch requires a private pinned reader')
        # Exact complete reference, never a caller's claimed content hash.
        if canonical(reference) != canonical(reader.reference):
            raise ValueError('pinned metadata reference changed in batch')
        key = id(reader)
        if key not in self.readers:
            if len(self.readers) >= MAX_READERS:
                raise ValueError('bounded metadata batch reader inventory required')
            record, value = reader.read(reference)  # Full current bytes at entry.
            self.readers[key] = reader, record, value
        _, record, value = self.readers[key]
        return record, value  # Private immutable meaning, not a file PASS.

    def finish(self):
        self._open()
        try:
            for reader, record, _ in self.readers.values():
                checked, _ = reader.read(reader.reference)  # ALL current bytes.
                if checked is not record:
                    raise ValueError('metadata batch lost its owned snapshot')
            # Reexecute original semantic/link validators, not producer flags.
            for _, closing_check in self.validations.values():
                closing_check()
        except BaseException:
            self.status = 'failed'
            raise
        self.status = 'closed'

    def detach(self):
        # No cache may survive on a fit-only predictor or cold reader.
        for owner in self.owners:
            if getattr(owner, '_metadata_batch', None) is self:
                owner._metadata_batch = None
        self.owners.clear()
        self.readers.clear()
        self.validations.clear()

    def validation(self, key, factory):
        self._open()
        if key not in self.validations:
            if len(self.validations) >= MAX_VALIDATIONS:
                raise ValueError('bounded metadata batch validation inventory required')
            self.validations[key] = factory()
        return self.validations[key][0]


@contextmanager
def metadata_batch(*owners):
    batch = _Batch()
    try:
        for owner in owners:
            batch.attach(owner)
        yield batch
        batch.finish()
    except BaseException:
        if batch.status == 'open':
            batch.status = 'failed'
        raise
    finally:
        batch.detach()


def read_metadata(owner, reader, reference):
    batch = getattr(owner, '_metadata_batch', None)
    return reader.read(reference) if batch is None else batch.read(reader, reference)


def share_batch(owner, child):
    batch = getattr(owner, '_metadata_batch', None)
    if batch is not None:
        batch.attach(child)
