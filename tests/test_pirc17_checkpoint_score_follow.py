"""Changed-binding scheduling with unchanged ORIGINAL cached score math."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.pirc17 import checkpoint_score_follow as module
from experiments.pirc17 import checkpoint_scoring as original
from experiments.pirc17.protocol_core import digest, read_json, unpack
from tests.test_pirc17_checkpoint_scoring import cached, available_work, prepared, scored, score_work


def fake(tmp_path):
    work=dict(work_id='w',kind='scientific_forecast')
    saved=SimpleNamespace(forecasts={'w':work},index={'w':{'content_sha256':'old'}},failures={})
    calls=[]
    def scope(w):
        calls.append('scope')
        return dict(source_sha256=digest(dict(binding=saved.index.get('w'),failure=saved.failures.get('w'))))
    def row(w): calls.append('row'); return dict(status='success')
    def path(w): calls.append('path'); return tmp_path/'cached.json'
    scorer=SimpleNamespace(saved=saved,cache_root=tmp_path,row_scope=scope,row_path=path,row=row,
        terminal=lambda w:'w' in saved.index or 'w' in saved.failures)
    return scorer,dict(rows={},blocks={}),{},calls


def test_unchanged_follow_polls_no_scopes_paths_or_arrays(tmp_path):
    scorer,state,observed,calls=fake(tmp_path)
    assert module.update_rows(scorer,state,observed)==1
    initial=calls[:]
    for _ in range(1000): assert module.update_rows(scorer,state,observed)==0
    assert calls==initial==['scope','row']


def test_existing_cache_checked_once_then_reused_without_row_read(tmp_path):
    scorer,state,observed,calls=fake(tmp_path)
    state['rows']['w']=dict(source_sha256=scorer.row_scope(scorer.saved.forecasts['w'])['source_sha256'],status='success')
    (tmp_path/'cached.json').touch()
    calls.clear()
    assert module.update_rows(scorer,state,observed)==0
    assert calls==['scope','path']
    calls.clear()
    assert module.update_rows(scorer,state,observed)==0 and not calls


def test_changed_binding_invalidates_blocks_and_scores_only_changed_item(tmp_path):
    scorer,state,observed,calls=fake(tmp_path)
    module.update_rows(scorer,state,observed)
    state['blocks']['b']={}
    scorer.saved.index['w']['content_sha256']='changed'
    calls.clear()
    assert module.update_rows(scorer,state,observed)==1
    assert calls==['scope','row'] and state['blocks']=={}
    assert observed['w'][0]['content_sha256']=='changed'


def test_unfinished_never_frozen_and_terminal_rollback_not_stale(tmp_path):
    scorer,state,observed,calls=fake(tmp_path)
    module.update_rows(scorer,state,observed)
    state['blocks']['b']={}; scorer.saved.index.clear(); calls.clear()
    assert module.update_rows(scorer,state,observed)==0
    assert not calls and state['rows']=={} and state['blocks']=={} and observed=={}
    scorer.saved.failures['w']='interrupted'
    assert module.update_rows(scorer,state,observed)==1


def test_maximum_does_not_invent_completion(tmp_path):
    scorer,state,observed,calls=fake(tmp_path)
    assert module.update_rows(scorer,state,observed,maximum=0)==0
    assert 'row' not in calls and state['rows']=={}


def test_incomplete_blocks_do_not_hash_cached_paths(tmp_path):
    scorer,state,observed,calls=fake(tmp_path)
    scorer.work={'b':dict(work_id='b')}; scorer.dependencies={'b':[dict(work_id='w')]}
    scorer.compute=lambda *a:pytest.fail('partial block computed')
    module.complete_blocks(scorer,state,dict(sha256='access'))
    assert not calls and state['blocks']=={}


def test_original_numeric_row_reuse_and_complete196_block(scored,prepared,tmp_path,monkeypatch):
    scorer=cached(scored,prepared,tmp_path)
    work=available_work(scorer)
    expected=scored[-1].row(work)
    original_index=deepcopy(scorer.saved.forecasts)
    scorer.saved.forecasts={work['work_id']:work}
    state=dict(cache_sha256=scorer.cache_identity['sha256'],rows={},blocks={})
    observed={}
    assert module.update_rows(scorer,state,observed)==1
    assert scorer.row(work)==expected
    scope_calls=[]
    original_scope=scorer.row_scope
    monkeypatch.setattr(scorer,'row_scope',lambda w:(scope_calls.append(w) or original_scope(w)))
    monkeypatch.setattr(scorer.saved,'read',lambda *a:pytest.fail('unchanged row reopened arrays'))
    assert module.update_rows(scorer,state,observed)==0 and not scope_calls
    monkeypatch.undo()
    scorer.saved.forecasts=original_index
    block=score_work(scorer,rank=1)  # All196 ORIGINAL NOT_ADMITTED fixture rows.
    scorer.saved.forecasts={w['work_id']:w for w in scorer.dependencies[block['work_id']]}
    assert module.update_rows(scorer,state,observed)==196
    access=original.start_access(tmp_path,scorer.saved.checkpoint_settings)
    module.complete_blocks(scorer,state,access)
    record=read_json(scorer.cache_root/state['blocks'][block['work_id']]['path'])
    payload=unpack(record)
    assert payload['expected_rows']==len(payload['rows'])==196
    assert payload['counts_by_status']=={'NOT_ADMITTED':196}
    assert scorer.verify(record,work=block,access_started_sha256=access['sha256'])['cached_score_rows_verified']
    scorer.saved.forecasts=original_index
    assert scorer.cache_identity['payload']['sources']==original.CachedScorer(saved=scorer.saved,directory=tmp_path).cache_identity['payload']['sources']
