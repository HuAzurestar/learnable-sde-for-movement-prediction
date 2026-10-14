"""Original full-clock metrics on synthetic forecasts, not empirical approval."""
from copy import deepcopy
from types import MethodType

import pytest

from experiments.pirc17 import checkpoint_scoring as module
from experiments.pirc17 import formal_forecasts
from experiments.pirc17.checkpoint_saved import CheckpointSavedForecasts
from experiments.pirc17.checkpoint_state import atomic_json
from experiments.pirc17.protocol_core import digest, envelope, read_json, unpack
from tests.test_pirc17_formal_forecasts import prepared
from tests.test_pirc17_formal_scoring import scored, positions, score_work


def cached(scored, prepared, tmp_path):
    saved = deepcopy(scored[-1].saved)
    saved.positions, saved.failures = positions(prepared), {}
    saved.terminal = MethodType(lambda s,w: w['work_id'] in s.index or s.case(w) is None, saved)
    saved.checkpoint_settings = dict(settings_id=digest('SOFTWARE checkpoint'),
        execution_sha256=saved.execution['sha256'], matrix_sha256=saved.matrix['sha256'],
        context_sha256=digest('SOFTWARE saved context'), approval_sha256=digest('SOFTWARE NOT HUMAN AUTHORITY'))
    return module.CachedScorer(saved=saved, directory=tmp_path)


def available_work(scorer):
    return next(w for w in scorer.saved.forecasts.values()
                if w['work_id'] in scorer.saved.index and w['subject'] == 'arm-01/full')


def test_cached_row_equals_original_metrics_and_restarts_without_array_reads(scored, prepared, tmp_path, monkeypatch):
    scorer = cached(scored, prepared, tmp_path)
    work = available_work(scorer)
    expected = scored[-1].row(work)
    monkeypatch.setattr(formal_forecasts, 'forecast_method', lambda *a, **kw: pytest.fail('new forecast'))
    monkeypatch.setattr(formal_forecasts, 'rollout', lambda *a, **kw: pytest.fail('new terrain rollout'))
    assert scorer.row(work) == expected
    assert scorer.computed_rows == 1
    restored = module.CachedScorer(saved=scorer.saved, directory=tmp_path)
    monkeypatch.setattr(restored.saved, 'read', lambda *a: pytest.fail('cached row reopened old arrays'))
    assert restored.row(work) == expected
    assert restored.cached_rows == 1 and restored.computed_rows == 0


def test_unfinished_forecast_is_not_frozen_as_unavailable(scored, prepared, tmp_path):
    scorer = cached(scored, prepared, tmp_path)
    work = next(w for w in scorer.dependencies[score_work(scorer)['work_id']]
                if not scorer.terminal(w))
    with pytest.raises(module.NotReady, match='unfinished'):
        scorer.row(work)
    assert not (scorer.cache_root/'rows'/work['work_id']).exists()


def test_partial_dependencies_do_not_publish_complete_common_block(scored, prepared, tmp_path):
    scorer = cached(scored, prepared, tmp_path)
    work = score_work(scorer)
    for forecast in scorer.dependencies[work['work_id']]:
        if scorer.terminal(forecast):
            scorer.row(forecast)
    assert not scorer.ready(work)
    state = dict(cache_sha256=scorer.cache_identity['sha256'], rows={}, blocks={})
    access = module.start_access(tmp_path, scorer.saved.checkpoint_settings)
    module.finish_blocks(scorer, state, access)
    assert state['blocks'] == {}


def test_complete_capacity_disposition_keeps_all196_rows_and_honest_replay_flag(scored, prepared, tmp_path):
    scorer = cached(scored, prepared, tmp_path)
    work = score_work(scorer, rank=1)
    for forecast in scorer.dependencies[work['work_id']]:
        assert scorer.row(forecast)['status'] == 'NOT_ADMITTED'
    state = dict(cache_sha256=scorer.cache_identity['sha256'], rows={}, blocks={})
    access = module.start_access(tmp_path, scorer.saved.checkpoint_settings)
    module.finish_blocks(scorer, state, access)
    binding = state['blocks'][work['work_id']]
    record = read_json(scorer.cache_root/binding['path'])
    value = unpack(record)
    assert value['expected_rows'] == len(value['rows']) == 196
    assert value['counts_by_status'] == {'NOT_ADMITTED':196}
    assert sum(r['scientific'] for r in value['rows']) == 190
    assert value['new_forecasts'] == value['new_fits'] == value['new_scoring_draws'] == 0
    assert value['independent_raw_output_replay_completed'] is False
    checked = scorer.verify(record, work=work, access_started_sha256=access['sha256'])
    assert checked['cached_score_rows_verified'] and not checked['scores_recomputed']
    scores = module.CheckpointScores(scorer=scorer, root=scorer.cache_root, index=state['blocks'],
                                     access_journal=tmp_path/'metric-access')
    scores._access(access['sha256'])
    wrong = deepcopy(value); wrong['expected_rows'] = 195
    with pytest.raises(ValueError, match='exact cached rows'):
        scorer.verify(envelope(wrong), work=work, access_started_sha256=access['sha256'])


def test_source_or_scoring_code_change_cannot_reuse_old_row(scored, prepared, tmp_path, monkeypatch):
    scorer = cached(scored, prepared, tmp_path)
    work = available_work(scorer)
    scorer.row(work)
    original = scorer.row_path(work)
    scorer.saved.index[work['work_id']]['content_sha256'] = digest('changed source')
    assert scorer.row_path(work) != original
    with pytest.raises(ValueError, match='identity'):
        scorer.row(work)
    assert original.is_file()
    monkeypatch.setattr(module, 'file_hash', lambda *a: '1'*64)
    changed = module.CachedScorer(saved=scorer.saved, directory=tmp_path)
    assert changed.cache_root != scorer.cache_root


def test_corrupt_cache_is_not_silently_accepted(scored, prepared, tmp_path):
    scorer = cached(scored, prepared, tmp_path)
    work = available_work(scorer)
    scorer.row(work)
    path = scorer.row_path(work)
    record = read_json(path)
    record['payload']['row']['score_m'] = -99999
    atomic_json(path, record)
    with pytest.raises(ValueError, match='identity'):
        scorer.row(work)


def test_refresh_reads_only_progress_and_never_reloads_models_or_old_files(tmp_path):
    saved = CheckpointSavedForecasts.__new__(CheckpointSavedForecasts)
    saved.root, saved.checkpoint_settings = tmp_path, {'settings_id':'s'}
    wid = digest('new work')
    saved.forecasts, saved.index, saved._imported_ids = {wid:{}}, {}, frozenset()
    result = dict(artifact_path='never/open/old/arrays', artifact_sha256=digest('artifact'))
    value = dict(settings_id='s', completed={wid:result}, failures={})
    atomic_json(tmp_path/'progress.json', value)
    saved.refresh(); saved.refresh()
    assert saved.index[wid]['path'] == result['artifact_path']
    value['completed'][wid]['artifact_path'] = 'replaced/source'
    atomic_json(tmp_path/'progress.json', value)
    with pytest.raises(ValueError, match='replaced'):
        saved.refresh()
