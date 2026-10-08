"""Full fixed matrix on synthetic saved outputs, NOT empirical acceptance.

Most synthetic forecasts deliberately fail or lack admitted origins. This
tests complete denominators and honest unavailable conclusions, not success.
Only constructor/input transport is substituted; metrics, analysis, fresh
audit, runtime replay and public arithmetic execute their original kernels.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType

import pytest

from experiments.pirc17 import checkpoint_audit as audit_module
from experiments.pirc17 import checkpoint_export as module
from experiments.pirc17.checkpoint_scoring import CheckpointScores, start_access, NotReady
from experiments.pirc17.checkpoint_state import atomic_json
from experiments.pirc17.formal_analysis import AnalysisConsumers
from experiments.pirc17.formal_saved import SavedPrediction
from experiments.pirc17.formal_scoring import ScoringConsumers
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, publish, unpack
from tests.test_pirc17_formal_forecasts import prepared
from tests.test_pirc17_formal_scoring import scored
from tests.test_pirc17_checkpoint_scoring import cached


@pytest.fixture(scope='module')
def full_export(scored, prepared, tmp_path_factory):
    directory = tmp_path_factory.mktemp('checkpoint-public-export-software')
    cache = cached(scored, prepared, directory)
    saved = cache.saved
    original_saved = deepcopy(saved)
    saved.index = {wid:dict(binding,path=str(saved.root/binding['path'])) for wid,binding in saved.index.items()}
    saved.failures = {wid:'SOFTWARE deliberate interruption; not a scientific observation'
        for wid,w in saved.forecasts.items() if wid not in saved.index and saved.case(w) is not None}
    def read(s,work):
        if work['work_id'] in s.failures:
            return SavedPrediction('failed',s.failures[work['work_id']],None,None,None)
        return original_saved.read(work)
    saved.read = MethodType(read,saved)
    saved.terminal = MethodType(lambda s,w: w['work_id'] in s.index or w['work_id'] in s.failures or s.case(w) is None,saved)
    # Avoid thousands of producer cache files in software fixtures. Original
    # numeric row computation is retained, including all available arrays.
    cache.row = MethodType(ScoringConsumers.row,cache)
    access = start_access(directory,saved.checkpoint_settings)
    state = dict(cache_sha256=cache.cache_identity['sha256'],rows={},blocks={})
    for wid,work in cache.work.items():
        payload = cache.compute(work)
        payload.update(metrics_access_started_sha256=access['sha256'],
            checkpoint_cache_sha256=cache.cache_identity['sha256'],independent_raw_output_replay_completed=False)
        path,record = publish(cache.cache_root/'blocks',payload)
        state['blocks'][wid] = dict(path=path.relative_to(cache.cache_root).as_posix(),
            content_sha256=record['sha256'],file_sha256=file_hash(path),metrics_access_started_sha256=access['sha256'])
    scores = CheckpointScores(scorer=cache,root=cache.cache_root,index=state['blocks'],access_journal=directory/'metric-access')
    consumer = AnalysisConsumers(scores=scores)
    analysis = consumer.compute(next(iter(consumer.work.values())))
    analysis.update(checkpoint_cache_sha256=cache.cache_identity['sha256'],
        metrics_access_started_sha256=access['sha256'],independent_raw_output_replay_completed=False)
    path,analysis_record = publish(cache.cache_root/'analysis',analysis)
    atomic_json(cache.cache_root/'analysis.json',dict(path=str(path),content_sha256=analysis_record['sha256'],file_sha256=file_hash(path)))
    input_id = next(wid for wid,w in saved.work.items() if w['kind']=='input_qualification_and_population')
    # Explicit synthetic transport substitution only: no legacy human PASS
    # is constructed and the full real audit computation is not stubbed.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(audit_module,'load',lambda root:(saved.checkpoint_settings,{input_id:{}}))
        patch.setattr(audit_module,'sources',lambda root:(saved,cache,state))
        binding = audit_module.audit(directory)
    audit_record = module.bound_record(binding,cache.cache_root)
    progress = dict(charged_ns_by_phase={'method_forecasts':123},generated=13)
    public = module.project(saved,cache,state,analysis_record,audit_record,progress=progress,access_sha256=access['sha256'])
    return dict(directory=directory,saved=saved,cache=cache,state=state,scores=scores,
        analysis=analysis_record,audit=audit_record,progress=progress,access=access,public=public)


def test_full_public_projection_replays_fresh_audit_and_keeps_failures(full_export,monkeypatch):
    source = full_export
    monkeypatch.setattr(source['saved'],'read',lambda *a:pytest.fail('export reopened raw predictions'))
    public = module.project(source['saved'],source['cache'],source['state'],source['analysis'],source['audit'],
        progress=source['progress'],access_sha256=source['access']['sha256'])
    assert public == source['public']
    assert public['schema_version'] == module.VERSION
    assert len(public['inputs']['score_rows']) == 11368
    assert len(public['audit_summary']['work_inventory']) == 11659
    assert len(public['score_work_dispositions']) == 58 and len(public['method_ledger']) == 36
    assert sum(r['disposition']=='EXCLUDED' for r in public['method_ledger']) == 8
    assert any(r['status']=='failed' for r in public['inputs']['score_rows'])
    assert module.replay_public(public['inputs']) == public['recomputed']
    assert all(r['verdict']=='unavailable' for r in public['recomputed']['factor_conclusions'].values())
    assert public['new_forecasts'] == public['new_fits'] == 0
    for key in ('retired_bootstrap_gate_asserted','raw_trajectories_exported','original_identifiers_exported',
                'scientific_claim_authorized','numerically_qualified','human_accepted'):
        assert public[key] is False
    raw = canonical(public)
    for token in (b'center_m',b'positions_m',b'sample_id',b'target_sha256',b'loaded_threadpools',str(source['directory']).encode()):
        assert token not in raw
    for cases in source['saved'].cases.values():
        for case in cases:
            assert case.sample_id.encode() not in raw and case.independent_block_id.encode() not in raw
    assert not public['budget_snapshot']['auxiliary_scoring_analysis_audit_costs_included']


def test_public_query_accounting_comes_from_single_final_audit_without_reopening_outputs(full_export, monkeypatch):
    source = full_export
    monkeypatch.setattr(source['saved'], 'read', lambda *a: pytest.fail('query accounting reopened forecast arrays'))
    public = module.project(source['saved'], source['cache'], source['state'], source['analysis'], source['audit'],
        progress=source['progress'], access_sha256=source['access']['sha256'])
    assert public['audit_summary']['feature_query_accounting_scope'] == audit_module.QUERY_ACCOUNTING_SCOPE
    original = {r['work_id']: r for r in unpack(source['audit'])['full_work_inventory']}
    forecasts = [r for r in public['audit_summary']['work_inventory'] if r['kind'] in audit_module.KINDS]
    assert len(forecasts) == 11571
    for row in forecasts:
        assert row['feature_query_accounting'] == original[row['work_id']]['feature_query_accounting']
    assert any(r['feature_query_accounting']['feature_query_rows'] is None for r in forecasts)
    assert any(r['feature_query_accounting']['feature_query_rows'] is not None for r in forecasts)
    assert not public['scientific_claim_authorized'] and not public['human_accepted']


@pytest.mark.parametrize('change', ['negative', 'invalid_exceeds_total', 'scope', 'missing'])
def test_export_rejects_rehashed_query_counter_or_scope_lies(full_export, change):
    s = full_export
    audit = deepcopy(unpack(s['audit']))
    row = next(r for r in audit['full_work_inventory'] if r['kind'] in audit_module.KINDS)
    if change == 'scope':
        audit['feature_query_accounting_scope']['per_column_validity_rates_available'] = True
    elif change == 'missing':
        row.pop('feature_query_accounting')
    else:
        row['feature_query_accounting'] = dict(feature_query_rows=100, invalid_feature_rows=7,
            raw_map_query_rows_this_forecast=100)
        row['feature_query_accounting']['invalid_feature_rows'] = -1 if change == 'negative' else 101
    with pytest.raises(ValueError):
        module.verify_sources(s['saved'], s['state'], s['analysis'], envelope(audit),
            score_index_sha256=s['scores'].identity['sha256'])


@pytest.mark.parametrize('change',['cached_only','short_rows','changed_inventory','changed_analysis','duplicate_work',
                                  'forecast_axes','negative_cost','nonfinite_cost'])
def test_export_rejects_incomplete_or_changed_fresh_audit(full_export,change):
    s = full_export
    value = deepcopy(unpack(s['audit']))
    if change=='cached_only': value['producer_cache_used_for_numbers'] = True
    elif change=='short_rows': value['raw_score_rows_recomputed'] = 1
    elif change=='changed_inventory': value['checkpoint_inventory_sha256'] = digest('another run')
    elif change=='changed_analysis': value['recomputed_analysis_payload_sha256'] = digest('numeric lie')
    elif change=='duplicate_work': value['full_work_inventory'][0] = deepcopy(value['full_work_inventory'][1])
    else:
        row = next(r for r in value['full_work_inventory'] if r['kind']=='scientific_forecast')
        if change=='forecast_axes': row['forecast_axes']['origin_rank']+=1
        elif change=='negative_cost': row['kernel_prediction_seconds']=-1
        else: row['kernel_prediction_seconds']=float('inf')
    with pytest.raises(ValueError):
        module.verify_sources(s['saved'],s['state'],s['analysis'],envelope(value),score_index_sha256=s['scores'].identity['sha256'])


def test_export_requires_all_original_common_blocks(full_export):
    s = full_export
    state = dict(s['state'],blocks=dict(s['state']['blocks']))
    state['blocks'].pop(next(iter(state['blocks'])))
    with pytest.raises(NotReady):
        module.verify_sources(s['saved'],state,s['analysis'],s['audit'],score_index_sha256=s['scores'].identity['sha256'])


def test_bound_pointer_rejects_an_outside_file(full_export,tmp_path):
    source = full_export
    path,record = publish(tmp_path,{'SOFTWARE':'outside logical result root'})
    with pytest.raises(ValueError,match='escapes'):
        module.bound_record(dict(path=str(path),content_sha256=record['sha256'],file_sha256=file_hash(path)),source['cache'].cache_root)


def test_completed_export_reuses_snapshot_and_standalone_public_replay(full_export,monkeypatch,tmp_path):
    source = full_export
    directory = source['directory']
    monkeypatch.setattr(module,'load',lambda root:(source['saved'].checkpoint_settings,{}))
    monkeypatch.setattr(module,'sources',lambda root:(source['saved'],source['cache'],source['state']))
    monkeypatch.setattr(source['saved'],'read',lambda *a:pytest.fail('export restarted raw-output arithmetic'))
    atomic_json(directory/'progress.json',source['progress'])
    binding = module.export(directory)
    assert module.bound_record(binding,source['cache'].cache_root)['sha256'] == binding['content_sha256']
    later = deepcopy(source['progress']); later['charged_ns_by_phase']['method_forecasts'] += 100
    atomic_json(directory/'progress.json',later)
    assert module.export(directory) == binding
    from experiments.pirc17.formal_export import main
    assert main(['--input',binding['path'],'--sha256',binding['content_sha256'],
                 '--output-directory',str(tmp_path/'standalone')]) == 0
    earlier = deepcopy(source['progress']); earlier['generated'] -= 1
    atomic_json(directory/'progress.json',earlier)
    with pytest.raises(ValueError,match='generation history decreased'):
        module.export(directory)
    atomic_json(directory/'progress.json',source['progress'])
