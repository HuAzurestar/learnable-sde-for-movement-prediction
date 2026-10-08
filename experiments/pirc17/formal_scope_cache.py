"""Owned immutable matrix metadata; no artifacts, models or authorization.

Validate and copy the complete envelope once per independently checked saved
collection. Fast reads are available ONLY on this owned immutable copy, never
on a mutable caller object or its claimed hash. Ordinary mutable inputs keep
the full original content check. All artifact/domain readers remain separate.
"""
from types import MappingProxyType

from .protocol_core import canonical, decode, digest, unpack


def _immutable(*args, **kwargs):
    raise TypeError('owned matrix metadata is immutable')


class _Dict(dict):
    __slots__ = ()
    __init__ = _immutable
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self


class _List(list):
    __slots__ = ()
    __init__ = _immutable
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = _immutable
    __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo):
        return self


def _freeze(value):
    if isinstance(value, dict):
        result = dict.__new__(_Dict)
        dict.__init__(result, {key: _freeze(item) for key, item in value.items()})
        return result
    if isinstance(value, list):
        result = list.__new__(_List)
        list.__init__(result, (_freeze(item) for item in value))
        return result
    return value


class OwnedMatrix(_Dict):
    """JSON-compatible private snapshot, not a cache of mutable object IDs.

    Its canonical bytes and exact work index belong to this object only.
    Recursive normal mutation APIs fail. deepcopy is safe because the entire
    tree is immutable; no targets, fitted objects or producer flags are cached.
    This does not defend against arbitrary malicious Python code bypassing
    containers with base-class mutation calls inside the trusted process.
    """
    __slots__ = ('_encoded', '_work')
    __setattr__ = __delattr__ = _immutable

    def __init__(self, record):
        if hasattr(self, '_encoded'):
            _immutable()
        # Round-trip removes external aliases and rejects non-JSON metadata.
        encoded = canonical(record)
        copied = decode(encoded)
        payload = unpack(copied)
        rows = payload.get('workloads')
        if not isinstance(rows, list):
            raise ValueError('owned matrix needs its complete registered workloads')
        frozen = _freeze(copied)
        work = {}
        for row in frozen['payload']['workloads']:
            if (not isinstance(row, dict) or 'work_id' not in row or row['work_id'] in work
                    or row['work_id'] != digest({key:value for key,value in row.items() if key != 'work_id'})):
                raise ValueError('owned matrix needs exact unique registered work descriptors')
            work[row['work_id']] = row
        dict.__init__(self, frozen)
        object.__setattr__(self, '_encoded', encoded)
        object.__setattr__(self, '_work', MappingProxyType(work))


def own_matrix(record):
    return record if type(record) is OwnedMatrix else OwnedMatrix(record)


def matrix_payload(record):
    return record['payload'] if type(record) is OwnedMatrix else unpack(record)


def owned_work_index(record):
    """Read-only exact descriptors checked when this complete copy was owned.

    Never reuse a mutable caller's header/index. Only the private OwnedMatrix
    constructor has validated all unique descriptors and their complete IDs.
    Collections copy this mapping, not the rows, and keep independent state.
    """
    if type(record) is not OwnedMatrix:
        raise ValueError('owned complete matrix required before work-index reuse')
    return record._work


def matrix_bytes(record):
    return record._encoded if type(record) is OwnedMatrix else canonical(record)


def same_matrix(left, right):
    # Compare complete validated bytes, not just header hashes. A mutable
    # external consumer is always serialized again and cannot inherit trust.
    return matrix_bytes(left) == matrix_bytes(right)


def matches_work(matrix, work):
    if type(matrix) is OwnedMatrix:
        registered = matrix._work.get(work.get('work_id'))
        return registered is not None and canonical(registered) == canonical(work)
    rows = unpack(matrix)['workloads']
    return [row for row in rows if row['work_id'] == work['work_id']] == [work]
