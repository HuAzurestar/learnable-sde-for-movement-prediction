"""Local downstream orchestration software, never empirical qualification."""
from pathlib import Path

import pytest

from experiments.pirc17 import checkpoint_review as module
from experiments.pirc17.checkpoint_state import atomic_json
from experiments.pirc17.protocol_core import digest, envelope, publish
from tests.test_pirc17_evidence_cards import card_inputs, full_export, catalog_inputs, prepared, scored


def indexes(tmp_path, *, runtime_done=True, rows=11368, pending=None):
    settings = dict(settings_id=digest('settings'), workloads=[
        dict(work_id='science',kind='scientific_forecast'),
        dict(work_id='runtime',kind='runtime_warm'),
        *[dict(work_id=str(i),kind='common_scores') for i in range(58)]])
    completed = {'runtime':{}} if runtime_done else {}
    atomic_json(tmp_path/'progress.json',dict(settings_id=settings['settings_id'],
        completed=completed,failures={'science':'original interruption'},pending=pending))
    score = tmp_path/'scores.json'
    atomic_json(score,dict(rows={str(i):{} for i in range(rows)},blocks={str(i):{} for i in range(58)}))
    return settings,score


def test_readiness_retains_failure_but_requires_runtime(tmp_path):
    settings,score=indexes(tmp_path,runtime_done=False)
    state=module.readiness(tmp_path,score,settings,{})
    assert not state['ready'] and state['unfinished_forecasts']==1
    assert state['terminal_failure_records']==1 and state['score_rows']==11368
    assert state['new_forecasts']==state['new_fits']==0 and not state['human_accepted']
    settings,score=indexes(tmp_path)
    assert module.readiness(tmp_path,score,settings,{})['ready']


@pytest.mark.parametrize('change',['rows','pending','blocks','settings','missing_score'])
def test_readiness_does_not_admit_partial_or_foreign_indexes(tmp_path,change):
    settings,score=indexes(tmp_path,rows=11367 if change=='rows' else 11368,
        pending={'work_id':'runtime'} if change=='pending' else None)
    if change=='blocks':
        atomic_json(score,dict(rows={str(i):{} for i in range(11368)},blocks={str(i+1):{} for i in range(58)}))
    if change=='missing_score': score=tmp_path/'not-produced.json'
    if change=='settings':
        with pytest.raises(ValueError,match='settings changed'):
            module.readiness(tmp_path,score,dict(settings,settings_id=digest('other')), {})
    else:
        assert not module.readiness(tmp_path,score,settings,{})['ready']


def test_follow_loads_metadata_once_and_never_launches_predictions(tmp_path,monkeypatch):
    settings,score=indexes(tmp_path,runtime_done=False)
    loads=[]; sleeps=[]
    monkeypatch.setattr(module,'load',lambda root: (loads.append(root) or settings,{}))
    def advance(seconds):
        sleeps.append(seconds); indexes(tmp_path)
    monkeypatch.setattr(module.time,'sleep',advance)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('waiting launched subprocess'))
    assert module.wait_ready(tmp_path,score,follow=True,poll_seconds=60)['ready']
    assert loads==[tmp_path] and sleeps==[60]


def test_nonfollow_unfinished_is_immediate_and_untouched(tmp_path,monkeypatch):
    settings,score=indexes(tmp_path,runtime_done=False)
    monkeypatch.setattr(module,'load',lambda root:(settings,{}))
    before=(tmp_path/'progress.json').read_bytes()
    monkeypatch.setattr(module.time,'sleep',lambda *a:pytest.fail('nonfollow waited'))
    with pytest.raises(RuntimeError,match='unfinished'):
        module.wait_ready(tmp_path,score)
    assert (tmp_path/'progress.json').read_bytes()==before


def consumers(tmp_path,monkeypatch,*,existing_audit=False):
    from experiments.pirc17 import checkpoint_audit,checkpoint_export,evidence_cards
    calls=[]
    tsde=tmp_path/'tsde'; (tsde/'scripts').mkdir(parents=True)
    for name in ('aggregate_pirc17.py','plot_pirc17.py'): (tsde/'scripts'/name).touch()
    catalog,record=publish(tmp_path/'catalog',dict(schema_version='software-fixture'))
    progress=tmp_path/'scores'/'progress.json'
    progress.parent.mkdir()
    if existing_audit: atomic_json(progress.parent/'audit.json',dict(content_sha256=digest('audit')))
    def stage(label,value):
        def perform(*args,**kwargs): calls.append(label); return value
        return perform
    monkeypatch.setattr(checkpoint_audit,'analyze',stage('analysis',dict(content_sha256=digest('analysis'))))
    monkeypatch.setattr(checkpoint_audit,'audit',stage('audit',dict(content_sha256=digest('audit'))))
    monkeypatch.setattr(checkpoint_export,'export',stage('export',dict(path='public.json',content_sha256=digest('export'))))
    monkeypatch.setattr(evidence_cards,'generate',stage('cards',dict(path='cards.json',content_sha256=digest('cards'))))
    def render(command,**kwargs):
        calls.append(command[2]); assert kwargs['cwd']==tsde.resolve()
        assert 'PYTHONPATH' not in kwargs['env'] and kwargs['check']
        from types import SimpleNamespace
        return SimpleNamespace(stdout='{"software_test":true}',stderr='')
    monkeypatch.setattr(module.subprocess,'run',render)
    return (tmp_path,progress,catalog,record['sha256'],tmp_path/'output',tsde),calls


@pytest.mark.parametrize('existing',[False,True])
def test_original_stage_order_and_existing_audit_not_repeated(tmp_path,monkeypatch,existing):
    args,calls=consumers(tmp_path,monkeypatch,existing_audit=existing)
    result=module.consume(*args)
    assert calls==(['analysis']+([] if existing else ['audit'])+
        ['export','cards','scripts.aggregate_pirc17','scripts.plot_pirc17'])
    assert not any(result[k] for k in ('human_accepted','scientific_claim_authorized',
        'test02_qualification_asserted','manuscript_written','software_fixture_only','new_forecasts','new_fits'))


def test_stale_audit_export_error_propagates_without_retry_or_render(tmp_path,monkeypatch):
    from experiments.pirc17 import checkpoint_export
    args,calls=consumers(tmp_path,monkeypatch,existing_audit=True)
    def fail(*a): calls.append('export'); raise ValueError('stale audit')
    monkeypatch.setattr(checkpoint_export,'export',fail)
    with pytest.raises(ValueError,match='stale audit'): module.consume(*args)
    assert calls==['analysis','export']


def test_bad_catalog_pin_prevents_all_expensive_work(tmp_path,monkeypatch):
    args,calls=consumers(tmp_path,monkeypatch)
    changed=list(args); changed[3]=digest('wrong')
    with pytest.raises(ValueError): module.consume(*changed)
    assert calls==[]


def test_actual_synthetic_producers_through_review_command(full_export,card_inputs,tmp_path,monkeypatch):
    """Original numeric fixture/audit/export/cards and real TSDE CLI; NOT science."""
    from experiments.pirc17 import checkpoint_audit,checkpoint_export
    source=full_export
    settings=source['saved'].checkpoint_settings
    input_id=next(wid for wid,w in source['saved'].work.items() if w['kind']=='input_qualification_and_population')
    for target in (checkpoint_audit,checkpoint_export):
        monkeypatch.setattr(target,'load',lambda root:(settings,{input_id:{}}))
        monkeypatch.setattr(target,'sources',lambda root:(source['saved'],source['cache'],source['state']))
    atomic_json(source['directory']/'progress.json',source['progress'])
    catalog,record=publish(tmp_path/'catalog',card_inputs[1]['payload'])
    tsde=Path(__file__).resolve().parents[2]/'TSDE-SDE'
    result=module.consume(source['directory'],source['cache'].cache_root/'progress.json',
        catalog,record['sha256'],tmp_path/'review',tsde,software_fixture=True)
    assert result['audit']==__import__('json').loads((source['cache'].cache_root/'audit.json').read_text())
    assert len(result['artifacts']['tables']['table_row_counts'])==16
    assert result['artifacts']['figures']['figure_count']==16
    assert result['artifacts']['figures']['image_count']==32
    assert result['software_fixture_only'] and not result['human_accepted']
