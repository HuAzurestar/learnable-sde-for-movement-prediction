"""Independent arithmetic on synthetic full-clock saved paths, not authority."""
from copy import deepcopy

import pytest

from experiments.pirc17 import checkpoint_audit as module
from experiments.pirc17.checkpoint_scoring import finish_blocks, start_access, NotReady
from experiments.pirc17.formal_scoring import ScoringConsumers
from experiments.pirc17.protocol_core import envelope, read_json, unpack
from tests.test_pirc17_formal_forecasts import prepared
from tests.test_pirc17_formal_scoring import scored, score_work
from tests.test_pirc17_checkpoint_scoring import cached, available_work


def test_fresh_scorer_reads_original_arrays_not_producer_rows(scored, prepared, tmp_path, monkeypatch):
    producer = cached(scored, prepared, tmp_path)
    work = available_work(producer)
    expected = producer.row(work)
    corrupted = read_json(producer.row_path(work))
    corrupted['payload']['row']['score_m'] = -99999
    # The producer can rehash a numeric lie; independent arithmetic must not
    # treat cache integrity as proof of the number.
    from experiments.pirc17.checkpoint_state import atomic_json
    atomic_json(producer.row_path(work), envelope(corrupted['payload']))
    fresh = module.FreshScorer(saved=producer.saved, cache_sha256=producer.cache_identity['sha256'])
    assert fresh.row(work) == expected
    assert fresh.row(work)['score_m'] != producer.row(work)['score_m']
    assert len(fresh.fresh_rows) == 1
    monkeypatch.setattr(fresh.saved, 'read', lambda *a: pytest.fail('same audit recomputed row twice'))
    assert fresh.row(work) == expected
    changed = dict(work); changed['subject'] = 'changed'
    with pytest.raises(ValueError, match='registered'):
        fresh.row(changed)


def test_independent_block_replay_rejects_rehashed_status_lie(scored, prepared, tmp_path):
    producer = cached(scored, prepared, tmp_path)
    work = score_work(producer, rank=1)
    for dependency in producer.dependencies[work['work_id']]:
        producer.row(dependency)
    state = dict(cache_sha256=producer.cache_identity['sha256'], rows={}, blocks={})
    access = start_access(tmp_path, producer.saved.checkpoint_settings)
    finish_blocks(producer, state, access)
    binding = state['blocks'][work['work_id']]
    record = read_json(producer.cache_root/binding['path'])
    fresh = module.FreshScorer(saved=producer.saved, cache_sha256=producer.cache_identity['sha256'])
    proof = fresh.verify(record, work=work, access_started_sha256=access['sha256'])
    assert proof['scores_recomputed'] and not proof['producer_cache_used_for_numbers']
    assert len(fresh.fresh_rows) == 196 and fresh.verified_blocks == 1
    lie = deepcopy(unpack(record)); lie['rows'][0]['status'] = 'success'
    with pytest.raises(ValueError, match='fresh original-array'):
        fresh.verify(envelope(lie), work=work, access_started_sha256=access['sha256'])


def test_final_audit_cannot_accept_partial_prediction_or_score_scope(scored, prepared, tmp_path):
    producer = cached(scored, prepared, tmp_path)
    with pytest.raises(NotReady, match='unfinished'):
        module.require_complete(producer.saved, {'blocks':{}})
    assert len(producer.saved.work) == 11659


def test_final_audit_cannot_shrink_the_registered_matrix(scored, prepared, tmp_path):
    producer = cached(scored, prepared, tmp_path)
    producer.saved.work.pop(next(iter(producer.saved.work)))
    with pytest.raises(ValueError, match='full11659'):
        module.require_complete(producer.saved, {'blocks':{}})
