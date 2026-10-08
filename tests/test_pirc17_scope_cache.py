"""Metadata-only scope tests; no final-eval, fitting or prediction."""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_scope_cache as cache
from experiments.pirc17.protocol_core import canonical, digest, envelope, unpack


def fixture_matrix():
    rows = [dict(kind='scientific_forecast', seed=index, scientific=True,
                 tags=['SYNTHETIC', index]) for index in range(3)]
    for row in rows:
        row['work_id'] = digest(row)
    return envelope(dict(workloads=rows, schema_version='SYNTHETIC metadata only'))


def test_owned_snapshot_preserves_complete_canonical_bytes_and_has_no_external_alias():
    original = fixture_matrix()
    owned = cache.own_matrix(original)
    assert canonical(owned) == canonical(original)
    assert cache.matrix_bytes(owned) == canonical(original)
    assert unpack(owned) == unpack(original)
    assert deepcopy(owned) is owned and cache.own_matrix(owned) is owned
    original['payload']['workloads'][0]['tags'].append('changed external alias')
    assert not cache.same_matrix(owned, original)
    assert cache.matrix_payload(owned)['workloads'][0]['tags'] == ['SYNTHETIC', 0]


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(sha256='changed'),
    lambda x: x.__setitem__('sha256', 'changed'),
    lambda x: x.pop('payload'),
    lambda x: x['payload'].clear(),
    lambda x: x['payload']['workloads'].append({}),
    lambda x: x['payload']['workloads'].__setitem__(0, {}),
    lambda x: x['payload']['workloads'][0].update(seed=99),
    lambda x: x['payload']['workloads'][0]['tags'].extend(['changed']),
    lambda x: x._work.__setitem__('changed', {}),
    lambda x: setattr(x, '_encoded', b'changed'),
    lambda x: x.__init__(fixture_matrix()),
    lambda x: x['payload'].__init__({'workloads': []}),
    lambda x: x['payload']['workloads'].__init__([]),
])
def test_all_normal_nested_mutation_paths_are_rejected(mutation):
    owned = cache.own_matrix(fixture_matrix())
    with pytest.raises((TypeError, AttributeError)):
        mutation(owned)


def test_owned_fast_reads_never_hash_or_serialize_the_complete_matrix_again(monkeypatch):
    original = fixture_matrix()
    owned = cache.own_matrix(original)
    other = cache.own_matrix(original)
    work = original['payload']['workloads'][0]
    canonical_fn = cache.canonical
    def only_small_records(value):
        if 'payload' in value and 'workloads' in value['payload']:
            pytest.fail('serialized full immutable matrix during a per-work read')
        return canonical_fn(value)
    monkeypatch.setattr(cache, 'canonical', only_small_records)
    monkeypatch.setattr(cache, 'unpack', lambda *a, **kw: pytest.fail('unpacked immutable matrix again'))
    for _ in range(5):
        assert cache.same_matrix(owned, other)
        assert cache.matches_work(owned, work)
        assert cache.matrix_payload(owned)['workloads'][0] == work


def test_foreign_or_changed_work_cannot_use_the_owned_index():
    original = fixture_matrix()
    owned = cache.own_matrix(original)
    work = deepcopy(original['payload']['workloads'][0])
    work['scientific'] = 1  # Python equality alone would confuse bool and int.
    assert not cache.matches_work(owned, work)
    work['work_id'] = digest('other owner')
    assert not cache.matches_work(owned, work)


def test_mutable_header_hash_does_not_confer_cached_trust():
    original = fixture_matrix()
    owned = cache.own_matrix(original)
    original['payload']['workloads'][0]['seed'] += 10
    assert not cache.same_matrix(owned, original)
    with pytest.raises(ValueError, match='content identity'):
        cache.matches_work(original, original['payload']['workloads'][1])
    with pytest.raises(ValueError, match='content identity'):
        cache.own_matrix(original)


def test_rehashed_duplicate_or_changed_owner_registration_is_rejected():
    payload = unpack(fixture_matrix())
    payload['workloads'].append(deepcopy(payload['workloads'][0]))
    with pytest.raises(ValueError, match='unique registered'):
        cache.own_matrix(envelope(payload))
    payload['workloads'].pop()
    payload['workloads'][0]['seed'] += 10
    with pytest.raises(ValueError, match='unique registered'):
        cache.own_matrix(envelope(payload))


def test_owned_work_index_is_complete_immutable_and_never_rehashes_descriptors(monkeypatch):
    original=fixture_matrix()
    owned=cache.own_matrix(original)
    expected={w['work_id']:w for w in original['payload']['workloads']}
    monkeypatch.setattr(cache,'digest',lambda *a:pytest.fail('rehashed immutable work descriptor'))
    monkeypatch.setattr(cache,'unpack',lambda *a,**kw:pytest.fail('reparsed immutable matrix'))
    index=cache.owned_work_index(owned)
    assert dict(index)==expected and len(index)==len(owned['payload']['workloads'])
    for row in owned['payload']['workloads']:assert index[row['work_id']] is row
    copied=dict(index)
    copied.clear()
    assert dict(index)==expected
    with pytest.raises(TypeError):index['foreign']={}
    with pytest.raises(TypeError):next(iter(index.values()))['seed']=123
    with pytest.raises(ValueError,match='owned complete'):
        cache.owned_work_index(original)
