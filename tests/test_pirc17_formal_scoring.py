"""Saved-source software integration, not empirical qualification/approval."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.pirc17 import formal_forecasts as forecasts
from experiments.pirc17 import formal_scoring as scoring
from experiments.pirc17.final_eval_guard import VerifiedAccess
from experiments.pirc17.formal_inputs import FinalPositionInputs, ScoringTargets
from experiments.pirc17.formal_mechanisms import MechanismConsumers, aggregate_gate
from experiments.pirc17.formal_saved import SavedForecasts, ScoringInputs
from experiments.pirc17.inference import SEEDS
from experiments.pirc17.method_mechanisms import EXACT_REFERENCE, mechanism_registry, score_diagnostic
from experiments.pirc17.metrics import EntropyGrid, score_path
from experiments.pirc17.protocol_core import digest, envelope, file_hash, read_json, unpack
from tests.test_pirc17_formal_forecasts import prepared, consumer, select
from tests.test_pirc17_formal_inputs import qualified_fixture


def reader(owner, root, outputs, **changes):
    index = {wid: dict(path=Path(result['artifact_path']).relative_to(root).as_posix(),
        file_sha256=file_hash(result['artifact_path']), content_sha256=result['artifact_sha256']) for wid, result in outputs.items()}
    args = dict(protocol=owner.protocol, execution=owner.execution, matrix=owner.matrix, input_identity=owner.input_identity,
        cases=owner.cases, population=owner.population if hasattr(owner, 'population') else None,
        fit_receipts=owner.receipts, map_catalog=owner.maps.catalog, root=root, index=index)
    args.update(changes)
    return SavedForecasts(**args)


def positions(prepared, **changes):
    targets = tuple(ScoringTargets(p.sample_id, p.window_sha256, p.score_seconds.copy(),
        p.terrain_origin.velocity_mps[None]*p.score_seconds[:, None]+np.array([20., -10.])) for p in prepared['prefixes'])
    args = dict(prefixes=prepared['prefixes'], targets=targets, population_sha256=prepared['population']['sha256'],
        access_started_sha256=digest('SOFTWARE POSITION ACCESS NOT HUMAN APPROVAL'))
    args.update(changes)
    return FinalPositionInputs(**args)


def score_work(scorer, *, rank=0, mode='causal_prefix'):
    return next(w for w in scorer.work.values() if w['origin_mode'] == mode and w['origin_rank'] == rank)


def access_for(scorer):
    scope = unpack(scorer.saved.input_identity)
    return VerifiedAccess(access_kind='final_eval_metrics', legacy_cohort_ack=True,
        **{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')},
        access_started_sha256=digest('SOFTWARE METRIC ACCESS NOT HUMAN APPROVAL'))


@pytest.fixture(scope='module')
def scored(prepared):
    owner = consumer(prepared)
    root = prepared['root']/'scoring-integration'
    outputs = {}
    pairs = [(s, 'scientific_forecast') for s in ('arm-01/full', 'arm-18/full', 'arm-19/em',
        'arm-19/euler', 'arm-10/d2_mc', 'arm-10/d2_closed', 'arm-20/full', 'arm-21/mc', 'arm-21/crn', 'base')]
    pairs += [(EXACT_REFERENCE, 'same_grid_reference'), ('all', 'inertial_path')]
    for subject, kind in pairs:
        w = select(owner, subject, kind=kind)
        outputs[w['work_id']] = owner.execute(w, output_directory=root/w['work_id'])
        assert outputs[w['work_id']]['status'] == 'success'
    # One honest numerical failure, still in the same frozen denominator.
    w = select(owner, 'arm-02/pointwise')
    with pytest.MonkeyPatch.context() as patch:
        def fail(*a, **kw): raise ValueError('SOFTWARE numerical failure; no retry')
        patch.setattr(forecasts, 'forecast_method', fail)
        outputs[w['work_id']] = owner.execute(w, output_directory=root/w['work_id'])
    saved = reader(owner, root, outputs, population=prepared['population'])
    scorer = scoring.ScoringConsumers(saved=saved, positions=positions(prepared))
    return owner, root, outputs, scorer


def test_actual_saved_scores_retain_all_scientific_reference_and_inertial_rows(scored, monkeypatch):
    _, _, _, scorer = scored
    monkeypatch.setattr(forecasts, 'forecast_method', lambda *a, **kw: pytest.fail('offline scoring regenerated a forecast'))
    monkeypatch.setattr(forecasts, 'rollout', lambda *a, **kw: pytest.fail('offline scoring regenerated terrain'))
    result = scorer.compute(score_work(scorer))
    assert result['expected_rows'] == len(result['rows']) == 196
    assert result['counts_by_status'] == {'success': 12, 'failed': 1, 'unavailable': 183}
    assert sum(r['scientific'] for r in result['rows']) == 190
    assert result['new_forecasts'] == result['new_fits'] == result['new_scoring_draws'] == 0
    full = next(r for r in result['rows'] if r['configuration'] == 'arm-01/full' and r['seed'] == SEEDS[0])
    saved = scorer.saved.read(scorer.saved.work[full['forecast_work_id']]).forecast
    target = scorer.inputs.targets[full['sample_id']]
    expected = score_path(saved.positions_m, target.positions_m, target.elapsed_seconds, time_weights=[.25]*4, entropy_grid=scorer.grid)
    assert full['score_m'] == expected['time_weighted_energy_score_m']
    for key, value in expected.items():
        if key != 'mode_entropy': assert full['scores'][key] == value
    assert full['scores']['mode_entropy']['status'] == 'computed'
    assert full['scores']['mode_entropy']['source'] == 'fitted labelled iid mode-redraw probabilities'
    terrain = next(r for r in result['rows'] if r['configuration'] == 'base' and r['seed'] == SEEDS[0])
    assert terrain['scores']['mode_entropy']['status'] == 'unavailable'
    inertial = [r for r in result['rows'] if r['matrix'] == 'inertial']
    assert len(inertial) == 1 and inertial[0]['seed'] is None and not inertial[0]['scientific']
    assert inertial[0]['scores']['particle_count'] == 1
    assert inertial[0]['score_m'] == pytest.approx(np.sqrt(500.))
    assert all(r['score_m'] is None and r['scores'] is None for r in result['rows'] if r['status'] != 'success')


@pytest.mark.parametrize('mode', ['causal_prefix', 'known_velocity', 'point_only'])
def test_unadmitted_capacity_is_not_counted_as_observed_failure(scored, mode):
    scorer = scored[-1]
    result = scorer.compute(score_work(scorer, rank=1, mode=mode))
    assert result['counts_by_status'] == {'NOT_ADMITTED': 196}
    assert all(r['sample_id'] is None and r['score_m'] is None for r in result['rows'])


def test_point_mass_metrics_are_analytic_and_preserve_overflow():
    path = np.array([[[20000., 0.], [1., 2.], [2., 3.], [4., 5.]]])
    target = path[0]+np.array([[3., 4.], [0., 0.], [-1., 0.], [0., -2.]])
    grid = EntropyGrid((-10000., 0., 10000.), (-10000., 0., 10000.))
    m = scoring.point_mass_scores(path, target, [60., 300., 900., 1800.], weights=[.25]*4, grid=grid)
    assert m['time_weighted_energy_score_m'] == 2.
    assert m['path_energy_score_m'] == pytest.approx(np.sqrt(30./4))
    assert m['by_time'][0]['marginal_crps_m'] == [3., 4.]
    assert m['by_time'][0]['position_entropy']['overflow_mass'] == 1.
    assert m['by_time'][0]['position_entropy']['entropy_nats'] == 0.
    assert all(level['covered'] for level in m['by_time'][1]['region']['levels'])
    assert not any(level['covered'] for level in m['by_time'][0]['region']['levels'])
    assert m['fde_m'] == 2. and m['particle_endpoint_error_quantiles_m'] == {'0.5': 2., '0.9': 2., '0.95': 2.}
    assert m['nll']['status'] == 'unavailable'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'sample', 'window', 'time', 'nonfinite', 'frame', 'access'])
def test_target_binding_rejects_subsets_substitution_and_fabricated_clock(scored, prepared, mutation):
    original = positions(prepared)
    t, p = original.targets[0], original.prefixes[0]
    if mutation == 'missing': original = replace(original, targets=())
    elif mutation == 'duplicate': original = replace(original, targets=(t, t))
    elif mutation == 'sample': original = replace(original, targets=(replace(t, sample_id='0'*64),))
    elif mutation == 'window': original = replace(original, targets=(replace(t, window_sha256='0'*64),))
    elif mutation == 'time': original = replace(original, targets=(replace(t, elapsed_seconds=t.elapsed_seconds+1.),))
    elif mutation == 'nonfinite': original = replace(original, targets=(replace(t, positions_m=np.full((4, 2), np.nan)),))
    elif mutation == 'frame': original = replace(original, prefixes=(replace(p, scoring_frame=replace(p.scoring_frame, longitude=0.)),))
    elif mutation == 'access': original = replace(original, access_started_sha256='not an access receipt')
    with pytest.raises(ValueError): ScoringInputs(original, scored[-1].saved)


def test_truth_copy_is_frozen_and_never_enters_semantic_random_stream(scored, prepared):
    loaded = positions(prepared)
    bound = ScoringInputs(loaded, scored[-1].saved)
    before = deepcopy(bound.identity)
    loaded.targets[0].positions_m[:] = 123456.
    assert bound.identity == before
    assert bound.targets[loaded.targets[0].sample_id].positions_m[0, 0] != 123456.
    assert not bound.targets[loaded.targets[0].sample_id].positions_m.flags.writeable
    assert not hasattr(scored[-1].saved, 'targets') and not hasattr(scored[-1].saved, 'execute')


@pytest.mark.parametrize('mutation', ['file', 'content', 'path', 'extra', 'duplicate'])
def test_bound_index_does_not_turn_corruption_into_unavailable(scored, prepared, mutation):
    owner, root, outputs, scorer = scored
    index = deepcopy(scorer.saved.index)
    work = select(owner)
    binding = index[work['work_id']]
    if mutation == 'file': binding['file_sha256'] = '0'*64
    elif mutation == 'content': binding['content_sha256'] = '0'*64
    elif mutation == 'path': binding['path'] = '../outside.json'
    elif mutation == 'extra': binding['passed'] = True
    else: index[select(owner, 'arm-03/single_gaussian')['work_id']] = deepcopy(binding)
    with pytest.raises(ValueError):
        saved = reader(owner, root, outputs, population=prepared['population'], index=index)
        saved.read(work)


def test_scoring_publication_and_raw_score_verification_reject_rehashed_lie(scored, tmp_path):
    scorer = scored[-1]
    work, access = score_work(scorer), access_for(scorer)
    result = scorer._execute(access, work, tmp_path)
    record = read_json(result['artifact_path'])
    checked = scorer.verify(record, work=work, access_started_sha256=access.access_started_sha256)
    assert checked['scores_recomputed'] and checked['new_forecasts'] == 0
    changed = deepcopy(unpack(record))
    full = next(r for r in changed['rows'] if r['status'] == 'success')
    full['score_m'] = full['scores']['time_weighted_energy_score_m'] = -99999.
    with pytest.raises(ValueError, match='actual prediction/target'):
        scorer.verify(envelope(changed), work=work, access_started_sha256=access.access_started_sha256)
    with pytest.raises(ValueError, match='already attempted'): scorer._execute(access, work, tmp_path)


@pytest.mark.parametrize('failure', ['approval', 'population'])
def test_actual_public_metric_guard_denies_before_scoring(tmp_path, monkeypatch, failure):
    args = qualified_fixture(tmp_path, monkeypatch)
    keys = ('protocol', 'execution', 'approval_path', 'approval_sha256', 'test_path', 'review_path',
        'journal_directory', 'population_path', 'population_sha256', 'eligibility_path')
    call = {k: args[k] for k in keys}
    class Trap:
        def _execute(self, *a): pytest.fail('unauthorized scoring reached implementation')
    if failure == 'approval': call['approval_sha256'] = '0'*64
    else: call['population_path'] = call['population_sha256'] = None
    with pytest.raises(ValueError):
        scoring.score_formal_work(scorer=Trap(), work={}, output_directory=tmp_path/'forbidden', **call)
    assert not (tmp_path/'forbidden').exists()


def test_actual_raw_mechanisms_keep_missing_seeds_exclusions_and_failed_gates(scored, monkeypatch):
    monkeypatch.setattr(forecasts, 'forecast_method', lambda *a, **kw: pytest.fail('mechanism verification regenerated paths'))
    owner = MechanismConsumers(scored[-1])
    result = owner.compute()
    p = unpack(result)
    assert len(p['fitted_model_gates']) == 21 and len(p['excluded_slots']) == 8
    assert all(r['gate']['status'] == 'computed' for r in p['fitted_model_gates'].values())
    assert all(r['scientific_rejection_implied'] is False for r in p['excluded_slots'].values())
    primary = p['modes']['causal_prefix']
    assert len(primary['gates']) == 28
    for slot, rows in primary['raw_diagnostics'].items():
        assert len(rows) == 5 and sum(r['gate'] is not None for r in rows) == 1
        assert rows[0]['gate']['status'] == 'computed'
        assert primary['gates'][slot]['status'] == 'unavailable'
        assert primary['gates'][slot]['passed'] is None
        assert primary['gates'][slot]['expected_count'] == 5
    assert primary['variance']['expected_rows'] == 15 and primary['variance']['available_rows'] == 3
    assert all(primary['gates'][slot]['status'] == 'unavailable' for slot in ('arm-21/mc', 'arm-21/crn'))
    assert p['scoring_only_gaussian_draws_evaluated'] == 2*4*256
    assert p['registered_scoring_only_draw_ceiling'] == 593920
    assert p['new_forecasts'] == p['new_fits'] == p['new_seeds'] == 0
    assert p['numerically_qualified'] is False
    assert owner.verify(result)['raw_mechanisms_recomputed']
    forged = deepcopy(p)
    forged['modes']['causal_prefix']['gates']['arm-19/euler']['passed'] = True
    with pytest.raises(ValueError, match='raw saved-source'): owner.verify(envelope(forged))


def test_aggregation_recomputes_raw_predicate_and_requires_every_instance():
    rng = np.random.default_rng(43)
    gate = score_diagnostic('arm-10/d2_mc', rng.normal(size=(512, 4, 2)), np.ones((4, 2)),
        seed=SEEDS[0], origin_id='SOFTWARE', input_identity_sha256=digest('SOFTWARE'))
    definition = mechanism_registry()['slots']['arm-10/d2_mc']
    result = aggregate_gate(definition, [dict(gate=gate)], expected_count=1)
    assert result['passed'] == (gate['value'] <= gate['threshold'])
    partial = aggregate_gate(definition, [dict(gate=gate), dict(gate=None)], expected_count=2)
    assert partial['status'] == 'unavailable' and partial['value'] is None
    with pytest.raises(ValueError, match='denominator'): aggregate_gate(definition, [], expected_count=1)
    gate['passed'] = not gate['passed']
    with pytest.raises(ValueError): aggregate_gate(definition, [dict(gate=gate)], expected_count=1)


def test_append_only_closed_outputs_do_not_invalidate_unrelated_scores_or_reload_models(scored, prepared, monkeypatch):
    owner, root, outputs, _ = scored
    saved = reader(owner, root, outputs, population=prepared['population'])
    scorer = scoring.ScoringConsumers(saved=saved, positions=positions(prepared))
    work = score_work(scorer)
    before = scorer.compute(work)
    old_context = deepcopy(saved.identity)
    late = select(owner, rank=1)
    result = owner.execute(late, output_directory=root/'late'/late['work_id'])
    binding = dict(path=Path(result['artifact_path']).relative_to(root).as_posix(),
        file_sha256=file_hash(result['artifact_path']), content_sha256=result['artifact_sha256'])
    from experiments.pirc17 import formal_saved
    monkeypatch.setattr(formal_saved, 'restore_registered_fit', lambda *a, **kw: pytest.fail('admission restored models again'))
    saved.admit(late, binding)
    assert saved.identity == old_context and scorer.compute(work) == before
    assert saved.read(late).status == 'NOT_ADMITTED'
    binding['file_sha256'] = '0'*64  # Admission owns its copied binding.
    assert saved.read(late).status == 'NOT_ADMITTED'
    with pytest.raises(ValueError, match='cannot replace'): saved.admit(late, binding)
    with pytest.raises(ValueError, match='several physical'): saved.admit(select(owner, rank=2), binding)
    with pytest.raises(ValueError, match='unique registered'): saved.index_identity([work['work_id']])


def test_new_required_evidence_invalidates_stale_missing_score_record(scored, prepared):
    owner, root, outputs, _ = scored
    saved = reader(owner, root, outputs, population=prepared['population'], index={})
    scorer = scoring.ScoringConsumers(saved=saved, positions=positions(prepared))
    work, access = score_work(scorer), access_for(scorer)
    old = scorer.compute(work)
    old['metrics_access_started_sha256'] = access.access_started_sha256
    forecast_work = select(owner)
    result = outputs[forecast_work['work_id']]
    saved.admit(forecast_work, dict(path=Path(result['artifact_path']).relative_to(root).as_posix(),
        file_sha256=file_hash(result['artifact_path']), content_sha256=result['artifact_sha256']))
    assert scorer.compute(work)['saved_forecast_index_sha256'] != old['saved_forecast_index_sha256']
    with pytest.raises(ValueError, match='actual prediction/target'):
        scorer.verify(envelope(old), work=work, access_started_sha256=access.access_started_sha256)


def test_complete_actual_five_seed_variance_triplets_reach_raw_statistic(scored, prepared):
    _, _, original_outputs, _ = scored
    owner = consumer(prepared)
    root, outputs = prepared['root'], dict(original_outputs)
    for work in owner.work.values():
        if (work['kind'] == 'scientific_forecast' and work['subject'] in {'arm-20/full', 'arm-21/mc', 'arm-21/crn'}
                and work['origin_mode'] == 'causal_prefix' and work['origin_rank'] == 0 and work['seed'] in SEEDS[1:]):
            outputs[work['work_id']] = owner.execute(work, output_directory=root/'full-variance'/work['work_id'])
            assert outputs[work['work_id']]['status'] == 'success'
    saved = reader(owner, root, outputs, population=prepared['population'])
    scorer = scoring.ScoringConsumers(saved=saved, positions=positions(prepared))
    report = unpack(MechanismConsumers(scorer).compute())['modes']['causal_prefix']
    assert report['variance']['expected_rows'] == report['variance']['available_rows'] == 15
    gates = report['variance']['gates']
    assert gates['arm-21/mc']['evidence'] == gates['arm-21/crn']['evidence']
    evidence = gates['arm-21/crn']['evidence']
    assert evidence['independent_block_count'] == 1 and evidence['seed_count'] == 5
    assert evidence['numerator_m2'] >= 0 and evidence['denominator_m2'] >= 0
    assert evidence['new_forecasts'] == 0  # The diagnostic, not its software fixture.
    assert all(s['status'] == 'success' for s in report['variance']['sources'])
    if evidence['denominator_m2'] == 0:
        assert gates['arm-21/crn']['status'] == 'unavailable' and gates['arm-21/crn']['value'] is None
    else:
        assert gates['arm-21/crn']['value'] == pytest.approx(evidence['numerator_m2']/evidence['denominator_m2'])
        assert report['gates']['arm-21/crn']['status'] == 'computed'
