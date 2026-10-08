import json

import pytest

from experiments.pirc17.checkpoint_preview import read_cached_row, summarize
from experiments.pirc17.protocol_core import digest, envelope


def good_row(score=10.):
    return dict(status='success', score_m=score, scores=dict(
        time_weighted_energy_score_m=score, ade_grid_mean_m=12., fde_m=20.,
        by_time=[dict(energy_score_m=score, region=dict(levels=[dict(level=.9, covered=True)])) for _ in range(4)]))


def test_missing_and_failed_are_not_zero_filled():
    for rows in ([good_row()], [good_row(), dict(status='failed')], []):
        result = summarize(rows, 2)
        assert result['status'] == 'statistical_failure'
        assert result['weighted_es_m'] is None
        assert result['es_by_time_m'] is None


def test_true_zero_score_is_valid():
    result = summarize([good_row(0.), good_row(20.)], 2)
    assert result['status'] == 'computed'
    assert result['weighted_es_m'] == 10.
    assert result['coverage_90_by_time'] == [1.] * 4


def fixture(cache, source, work, score=10.):
    scope = dict(cache_sha256=cache.name, work_sha256=digest(work), source_sha256=source)
    row = dict(good_row(score), forecast_work_id=work['work_id'], configuration=work['subject'],
        matrix=work['matrix'], seed=work['seed'], origin_mode=work['origin_mode'],
        origin_rank=work['origin_rank'], partition='final_eval', scientific=True)
    value = dict(schema_version='pirc17-checkpoint-scoring-v1-row', scope=scope, row=row)
    path = cache/'rows'/work['work_id']/(digest(scope)+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope(value)), encoding='utf-8')


def test_selects_index_bound_source_without_writing_cache(tmp_path):
    cache = tmp_path/('a'*64)
    work = dict(work_id='b'*64, subject='base', matrix='terrain', seed=20260814,
                origin_mode='causal_prefix', origin_rank=0)
    fixture(cache, 'old', work, 1.)
    fixture(cache, 'current', work, 20.)
    before = {p:p.read_bytes() for p in cache.rglob('*.json')}
    row, _ = read_cached_row(cache, work, dict(source_sha256='current', status='success'), 'current')
    assert row['score_m'] == 20.
    assert before == {p:p.read_bytes() for p in cache.rglob('*.json')}
    with pytest.raises(ValueError, match='source/work/scope'):
        read_cached_row(cache, work, dict(source_sha256='current', status='success'), 'wrong')


def test_empty_success_is_rejected(tmp_path):
    cache = tmp_path/('a'*64)
    work = dict(work_id='b'*64, subject='base', matrix='terrain', seed=20260814,
                origin_mode='causal_prefix', origin_rank=0)
    fixture(cache, 'current', work)
    path = next((cache/'rows'/work['work_id']).glob('*.json'))
    value = json.loads(path.read_text())['payload']
    value['row']['scores'] = None
    path.write_text(json.dumps(envelope(value)), encoding='utf-8')
    with pytest.raises(ValueError, match='empty or invalid'):
        read_cached_row(cache, work, dict(source_sha256='current', status='success'), 'current')
