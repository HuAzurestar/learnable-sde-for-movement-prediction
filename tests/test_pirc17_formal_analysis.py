"""Real saved-array/score/analysis consumers on explicitly synthetic inputs."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.pirc17 import formal_analysis as module
from experiments.pirc17 import formal_forecasts as forecasts
from experiments.pirc17.formal_scoring import ScoringConsumers
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, publish, read_json, unpack
from tests.test_pirc17_formal_forecasts import prepared
from tests.test_pirc17_formal_scoring import scored, positions, score_work, access_for
from tests.test_pirc17_formal_inputs import qualified_fixture


@pytest.fixture(scope='module')
def bound(scored, prepared):
    root = prepared['root']/'analysis-integration'
    scorer = ScoringConsumers(saved=scored[-1].saved, positions=positions(prepared))
    access = access_for(scorer)
    # Explicit SOFTWARE event; not a production approval. The public guard's
    # actual denial tests below use its separate qualified synthetic fixture.
    _, started = publish(root/'access', dict(schema_version='pirc17-final-access-event-v1', event='started',
        access_kind='final_eval_metrics', attempt_id='a'*32, at_utc='2026-09-30T00:00:00+00:00',
        **{k: getattr(access, k) for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')}))
    access = replace(access, access_started_sha256=started['sha256'])
    work = score_work(scorer)
    output = scorer._execute(access, work, root/'common')
    index = {work['work_id']: dict(path=Path(output['artifact_path']).relative_to(root).as_posix(),
        file_sha256=file_hash(output['artifact_path']), content_sha256=output['artifact_sha256'],
        metrics_access_started_sha256=access.access_started_sha256)}
    args = dict(scorer=scorer, root=root, index=index, access_journal=root/'access')
    sources = module.SavedScores(**args)
    analysis = module.AnalysisConsumers(scores=sources)
    return args, analysis, next(iter(analysis.work.values())), access


def test_real_saved_sources_recompute_all_family_and_ledger_dispositions_without_predictions(bound, monkeypatch):
    args, analysis, work, _ = bound
    monkeypatch.setattr(forecasts, 'forecast_method', lambda *a, **kw: pytest.fail('analysis regenerated a method prediction'))
    monkeypatch.setattr(forecasts, 'rollout', lambda *a, **kw: pytest.fail('analysis regenerated terrain'))
    result = analysis.compute(work)
    assert result['new_forecasts'] == result['new_fits'] == 0 and not result['scientific_claim_authorized']
    assert len(result['score_work_dispositions']) == 58
    assert sum(r['score_artifact_status'] == 'verified' for r in result['score_work_dispositions']) == 1
    assert sum(r['admitted'] for r in result['score_work_dispositions']) == 3
    assert set(result['modes']) == {'causal_prefix', 'known_velocity', 'point_only'}
    for mode, report in result['modes'].items():
        assert report['independent_block_count'] == 1 and report['scientific_score_rows'] == 190
        assert len(report['families']) == 7 and sum(len(r['results']) for r in report['families'].values()) == 30
        assert all(r['inference'] is None for r in report['families'].values())
        assert all(not r['hypothesis_tests_performed'] and not r['factor_verdict_authorized'] for r in report['terrain_scoring_slots'].values())
    assert set(result['factor_conclusions']) == {'road', 'river', 'worldcover', 'surface'}
    assert all(r['verdict'] == 'unavailable' for r in result['factor_conclusions'].values())
    assert len(result['method_ledger']) == 36 and sum(r['disposition'] == 'EXCLUDED' for r in result['method_ledger']) == 8
    assert result['history'] == dict(role='baseline-only', terrain_verdict=None)
    raw = unpack(result['raw_mechanisms'])
    assert len(raw['fitted_model_gates']) == 21 and raw['scoring_only_gaussian_draws_evaluated'] == 2048
    owners = unpack(result['terrain_ownership'])
    assert all(r['owner_closure_verified'] and r['independent_fit_verified'] for r in owners['configurations'].values())
    assert owners['configurations']['base']['definition']['conditioner_input_dimension'] == 4
    assert owners['configurations']['all-terrain']['definition']['conditioner_input_dimension'] == 56


def test_missing_common_artifact_is_not_replaced_by_scoring_available_predictions(bound, monkeypatch):
    args, _, _, _ = bound
    monkeypatch.setattr(args['scorer'], 'verify', lambda *a, **kw: pytest.fail('missing score artifact was implicitly rescored'))
    sources = module.SavedScores(**dict(args, index={}))
    rows, dispositions = sources.load()
    assert all(row['status'] == 'unavailable' and row['score_m'] is None for mode in rows.values() for row in mode)
    assert sum(len(mode) for mode in rows.values()) == 3*190
    assert all(row['score_artifact_status'] == 'unavailable' for row in dispositions)


@pytest.mark.parametrize('mutation', ['file', 'content', 'path', 'access', 'extra', 'duplicate'])
def test_saved_score_index_and_actual_access_receipt_are_verified(bound, mutation):
    args, _, _, _ = bound
    index = deepcopy(args['index']); key = next(iter(index)); binding = index[key]
    if mutation == 'file': binding['file_sha256'] = '0'*64
    elif mutation == 'content': binding['content_sha256'] = '0'*64
    elif mutation == 'path': binding['path'] = '../outside.json'
    elif mutation == 'access': binding['metrics_access_started_sha256'] = '0'*64
    elif mutation == 'extra': binding['passed'] = True
    else: index[score_work(args['scorer'], rank=1)['work_id']] = deepcopy(binding)
    with pytest.raises(ValueError): module.SavedScores(**dict(args, index=index)).load()


@pytest.mark.parametrize('field,value', [('access_kind', 'final_eval_positions'), ('at_utc', 0),
    ('at_utc', '2026-09-30T00:00:00'), ('attempt_id', 'not-a-real-attempt'), ('extra', 'unregistered')])
def test_wrong_actual_access_event_cannot_be_relabelled_as_metric_authority(bound, field, value):
    args, _, _, _ = bound
    index = deepcopy(args['index']); binding = next(iter(index.values()))
    event = deepcopy(unpack(read_json(args['access_journal']/(binding['metrics_access_started_sha256']+'.json'))))
    event[field] = value
    _, changed = publish(args['access_journal'], event)
    binding['metrics_access_started_sha256'] = changed['sha256']
    with pytest.raises(ValueError, match='matching metric access'):
        module.SavedScores(**dict(args, index=index)).load()


def test_rehashed_common_score_forgery_is_checked_against_actual_arrays(bound, tmp_path):
    args, _, _, _ = bound
    index = deepcopy(args['index']); binding = next(iter(index.values()))
    original = unpack(read_json(args['root']/binding['path']))
    changed = deepcopy(original)
    row = next(r for r in changed['rows'] if r['status'] == 'success')
    row['score_m'] = row['scores']['time_weighted_energy_score_m'] = -9999.
    path = args['root']/'forged-score.json'
    record = envelope(changed); path.write_bytes(canonical(record))
    binding.update(path=path.name, file_sha256=file_hash(path), content_sha256=record['sha256'])
    with pytest.raises(ValueError, match='actual prediction/target'):
        module.SavedScores(**dict(args, index=index)).load()


def test_actual_fitted_owner_dimension_is_not_a_caller_pass_flag(bound, monkeypatch):
    saved = bound[1].saved
    identity = saved.models['terrain-fit:all-terrain'].identity
    monkeypatch.setitem(identity['conditioner_checkpoint'], 'input_dim', 4)
    with pytest.raises(ValueError, match='owner/dimension'): module.owner_evidence(saved)


def test_absent_independent_fit_keeps_unavailable_owner_evidence(bound, monkeypatch):
    saved = bound[1].saved
    monkeypatch.delitem(saved.models, 'terrain-fit:loo-road')
    monkeypatch.delitem(saved.receipts, 'terrain-fit:loo-road')
    row = unpack(module.owner_evidence(saved))['configurations']['loo-road']
    assert row['status'] == 'unavailable' and row['owner_closure_verified']
    assert not row['independent_fit_verified'] and row['fit_receipt_sha256'] is None


def test_analysis_publication_and_independent_reanalysis_reject_rehashed_verdict(bound, tmp_path):
    _, analysis, work, access = bound
    output = analysis._execute(access, work, tmp_path)
    record = read_json(output['artifact_path'])
    details = analysis.verify(record, work=work, access_started_sha256=access.access_started_sha256)
    assert details['paired_statistics_recomputed'] and details['raw_mechanisms_recomputed']
    changed = deepcopy(unpack(record)); changed['factor_conclusions']['road']['verdict'] = 'retain'
    with pytest.raises(ValueError, match='paired-statistic recomputation'):
        analysis.verify(envelope(changed), work=work, access_started_sha256=access.access_started_sha256)
    with pytest.raises(ValueError, match='already attempted'): analysis._execute(access, work, tmp_path)


@pytest.mark.parametrize('failure', ['approval', 'population'])
def test_real_public_guard_denies_before_analysis(tmp_path, monkeypatch, failure):
    args = qualified_fixture(tmp_path, monkeypatch)
    keys = ('protocol', 'execution', 'approval_path', 'approval_sha256', 'test_path', 'review_path',
        'journal_directory', 'population_path', 'population_sha256', 'eligibility_path')
    call = {k: args[k] for k in keys}
    class Trap:
        def _execute(self, *a): pytest.fail('unauthorized analysis reached raw sources')
    if failure == 'approval': call['approval_sha256'] = '0'*64
    else: call['population_path'] = call['population_sha256'] = None
    with pytest.raises(ValueError):
        module.analyze_formal_work(analysis=Trap(), work={}, output_directory=tmp_path/'forbidden', **call)
    assert not (tmp_path/'forbidden').exists()
