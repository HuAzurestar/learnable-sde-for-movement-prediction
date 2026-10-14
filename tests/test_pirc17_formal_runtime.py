"""Actual saved CPU trials on synthetic inputs, not empirical qualification."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.pirc17 import formal_forecasts as forecasts
from experiments.pirc17 import formal_runtime as module
from experiments.pirc17 import formal_timing as timing
from experiments.pirc17.protocol_core import envelope, file_hash, publish, read_json, unpack
from tests.test_pirc17_formal_forecasts import prepared, consumer, select, fixture_maps, restore
from tests.test_pirc17_formal_scoring import reader


@pytest.fixture(scope='module')
def measured(prepared):
    owner = consumer(prepared)
    root, outputs, fresh = prepared['root']/'runtime-software', {}, []
    def recreate():
        maps = fixture_maps(unpack(owner.input_identity), prepared['row']); fresh.append(maps)
        return maps
    def run(work):
        output = owner.execute(work, output_directory=root/work['work_id'])
        outputs[work['work_id']] = output
        return output
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner.maps, 'fresh_provider', recreate)
        for subject in ('arm-01/full', 'base'):
            for kind in ('scientific_forecast', 'forecast_replay'):
                assert run(select(owner, subject, kind=kind))['status'] == 'success'
        # All eleven actual trials for one method; three actual terrain trials
        # exercise provider initialization, nested query timing and warm reuse.
        for subject, counts in [('arm-01/full', (5, 1, 5)), ('lio-road', (1, 1, 1))]:
            for kind, count in zip(('runtime_cold', 'runtime_warmup', 'runtime_warm'), counts):
                for i in range(count):
                    assert run(select(owner, subject, kind=kind, repeat=i))['status'] == 'success'
        # Honest, deliberately injected numerical failures, including a failed
        # warmup. They retain measured elapsed time and never become a success.
        def fail(*a, **kw): raise ValueError('SOFTWARE numerical failure; no retry')
        patch.setattr(forecasts, 'forecast_method', fail)
        for kind in ('scientific_forecast', 'forecast_replay'):
            assert run(select(owner, 'arm-02/pointwise', kind=kind))['status'] == 'failed'
        for kind, count in [('runtime_cold', 5), ('runtime_warmup', 1), ('runtime_warm', 5)]:
            for i in range(count):
                assert run(select(owner, 'arm-04/gmm_kernel', kind=kind, repeat=i))['status'] == 'failed'
    owner.close()
    saved = reader(owner, root, outputs, population=prepared['population'])
    return dict(owner=owner, root=root, outputs=outputs, saved=saved, fresh=fresh, prepared=prepared)


def subject(result, name):
    return next(r for r in result['runtime_subjects'] if r['subject'] == name)


def changed_reader(measured, changes):
    owner, saved = measured['owner'], measured['saved']
    index = deepcopy(saved.index)
    for work, payload, folder in changes:
        path, record = publish(folder, payload)
        index[work['work_id']] = dict(path=path.relative_to(measured['root']).as_posix(),
            file_sha256=file_hash(path), content_sha256=record['sha256'])
    return reader(owner, measured['root'], measured['outputs'], population=measured['prepared']['population'], index=index)


def test_real_cpu_trials_preserve_full_denominators_units_memory_and_replay(measured, monkeypatch):
    monkeypatch.setattr(forecasts, 'forecast_method', lambda *a, **kw: pytest.fail('saved audit regenerated methods'))
    monkeypatch.setattr(forecasts, 'rollout', lambda *a, **kw: pytest.fail('saved audit regenerated terrain'))
    result = unpack(module.RuntimeEvidence(measured['saved']).compute())
    assert result['registered_counts'] == dict(replays=38, cold=75, warmup=15, warm=75)
    assert len(result['runtime_subjects']) == 15 and sum(len(r['rows']) for r in result['runtime_subjects']) == 165
    assert len(result['replay_rows']) == 38 and result['replay_counts_by_status'] == {'reproduced': 2, 'reproduced_failure': 1, 'unavailable': 35}
    assert result['new_forecasts'] == result['new_fits'] == 0 and not result['scientific_claim_authorized']
    full = subject(result, 'arm-01/full')
    for kind in ('runtime_cold', 'runtime_warm'):
        summary = full['conditions'][kind]
        trials = [r['runtime']['trial'] for r in full['rows'] if r['kind'] == kind]
        assert summary['status'] == 'computed' and summary['summary']['trial_count'] == 5
        assert summary['summary']['failure_rate'] == 0
        assert summary['summary']['total_latency_p50_ms'] == pytest.approx(np.median([t['end_to_end_ms'] for t in trials]))
        assert summary['summary']['mean_ms_per_forecast_minute'] == pytest.approx(np.mean([t['end_to_end_ms'] for t in trials])/30.)
        assert all(t['sampled_peak_rss_bytes'] >= t['baseline_rss_bytes'] > 0 for t in trials)
        assert all(t['inclusive_stage_ms']['rollout'] <= t['end_to_end_ms'] for t in trials)
    assert full['warmup_excluded_from_latency_quantiles'] and len(result['hardware']) == 1
    warmup = next(r for r in full['rows'] if r['kind'] == 'runtime_warmup')
    assert all(r['runtime']['warmup_result_sha256'] == warmup['forecast_sha256'] and r['warm_condition_verified']
        for r in full['rows'] if r['kind'] == 'runtime_warm')
    terrain = subject(result, 'lio-road')
    observed = [r['runtime'] for r in terrain['rows'] if r['runtime'] is not None]
    assert len(measured['fresh']) == 2 and all(m.closed for m in measured['fresh'])
    assert [r['provider_created'] for r in observed] == [True, True, False]
    assert all(r['trial']['inclusive_stage_ms']['terrain_io_and_query'] > 0 for r in observed)
    assert all(r['trial']['inclusive_stage_ms']['checkpoint_and_input_io'] > 0 for r in observed[:2])
    assert all(r['status'] == 'unavailable' and r['summary'] is None for r in terrain['conditions'].values())
    failed = subject(result, 'arm-04/gmm_kernel')
    assert failed['conditions']['runtime_cold']['summary']['failure_rate'] == 1
    assert all(r['runtime']['trial']['end_to_end_ms'] > 0 and r['status'] == 'failed' for r in failed['rows'])
    assert failed['conditions']['runtime_warm']['summary'] is None  # Failed warmup is not a valid warm condition.


def test_same_subject_warmup_is_mandatory_and_attempt_cannot_retry(prepared, tmp_path):
    owner = consumer(prepared); work = select(owner, kind='runtime_warm', repeat=0)
    with pytest.raises(ValueError, match='warmup must precede'): owner.execute(work, output_directory=tmp_path)
    with pytest.raises(ValueError, match='already attempted'): owner.execute(work, output_directory=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mutation', ['clock', 'memory', 'stage', 'horizon', 'condition', 'hardware', 'missing'])
def test_rehashed_invalid_measurements_fail_actual_forecast_domain_validation(measured, mutation):
    owner = measured['owner']; work = select(owner, kind='runtime_cold', repeat=0)
    output = measured['outputs'][work['work_id']]
    payload = deepcopy(unpack(read_json(output['artifact_path'])))
    raw = payload['runtime']; trial = raw['trial']
    if mutation == 'clock': trial['end_to_end_ms'] += 1.
    elif mutation == 'memory': trial['sampled_peak_increment_bytes'] += 1
    elif mutation == 'stage': trial['inclusive_stage_ms']['rollout'] = trial['end_to_end_ms']+1
    elif mutation == 'horizon': raw['actual_horizon_seconds'] = 86400.
    elif mutation == 'condition': raw['condition'] = 'runtime_warm'
    elif mutation == 'hardware': raw['hardware'] = envelope({'device': 'cpu'})
    else: payload['runtime'] = None
    from experiments.pirc17.formal_forecast_records import restore_forecast
    with pytest.raises(ValueError):
        restore_forecast(envelope(payload), directory=Path(output['artifact_path']).parent, work=work,
            protocol=owner.protocol, execution=owner.execution, matrix=owner.matrix, input_identity=owner.input_identity,
            case=owner.cases['causal_prefix'][0], fit_receipt=owner.receipts[work['fit_identity']],
            model=owner.models[work['fit_identity']], map_catalog=owner.maps.catalog)


@pytest.mark.parametrize('mutation', ['warmup-source', 'provider', 'hardware', 'order'])
def test_actual_saved_trial_cross_bindings_are_not_caller_flags(measured, tmp_path, mutation):
    owner = measured['owner']; work = select(owner, kind='runtime_warm', repeat=1)
    output = measured['outputs'][work['work_id']]; payload = deepcopy(unpack(read_json(output['artifact_path'])))
    raw = payload['runtime']
    if mutation == 'warmup-source': raw['warmup_result_sha256'] = '0'*64
    elif mutation == 'provider': raw['provider_instance_id'] = '0'*64
    elif mutation == 'hardware':
        hardware = deepcopy(unpack(raw['hardware'])); hardware['torch_threads'] += 1; raw['hardware'] = envelope(hardware)
    else:
        duration = raw['trial']['ended_monotonic_ns']-raw['trial']['started_monotonic_ns']
        raw['trial'].update(started_monotonic_ns=1, ended_monotonic_ns=1+duration)
    # Same arrays, different content-addressed record in the same directory.
    changed = changed_reader(measured, [(work, payload, Path(output['artifact_path']).parent)])
    with pytest.raises(ValueError): module.RuntimeEvidence(changed).compute()


def test_missing_warmup_artifact_cannot_be_substituted_or_removed_from_denominator(measured):
    saved, owner = measured['saved'], measured['owner']
    index = deepcopy(saved.index); del index[select(owner, kind='runtime_warmup', repeat=0)['work_id']]
    changed = reader(owner, measured['root'], measured['outputs'], population=measured['prepared']['population'], index=index)
    result = subject(unpack(module.RuntimeEvidence(changed).compute()), 'arm-01/full')
    assert len(result['rows']) == 11 and result['conditions']['runtime_cold']['status'] == 'computed'
    assert result['conditions']['runtime_warm']['status'] == 'unavailable'
    assert result['conditions']['runtime_warm']['summary'] is None


def test_rehashed_changed_saved_terrain_array_is_a_replay_mismatch_not_a_new_tolerance(measured):
    owner = measured['owner']; work = select(owner, 'base', kind='forecast_replay')
    output = measured['outputs'][work['work_id']]; payload = deepcopy(unpack(read_json(output['artifact_path'])))
    folder = measured['root']/'altered-replay'; folder.mkdir()
    with np.load(Path(output['artifact_path']).parent/'forecast.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays['positions_m'][0, 0, 0] += 1.e-9
    path = folder/'forecast.npz'; np.savez(path, **arrays)
    payload['arrays'].update(sha256=file_hash(path), bytes=path.stat().st_size)
    changed = changed_reader(measured, [(work, payload, folder)])
    result = unpack(module.RuntimeEvidence(changed).compute())
    replay = next(r for r in result['replay_rows'] if r['subject'] == 'base')
    assert replay['status'] == 'mismatch' and replay['output_identity_equal'] is False


def test_independent_saved_evidence_recompute_rejects_rehashed_summary_lie(measured):
    audit = module.RuntimeEvidence(measured['saved']); record = audit.compute()
    assert audit.verify(record)['historical_latency_independently_remeasured'] is False
    payload = deepcopy(unpack(record)); subject(payload, 'arm-01/full')['conditions']['runtime_cold']['summary']['failure_rate'] = 1.
    with pytest.raises(ValueError, match='actual saved sources'): audit.verify(envelope(payload))


def test_sampler_does_not_swallow_fatal_integrity_errors_or_leak_its_thread():
    trial = timing.RuntimeTrial()
    with pytest.raises(RuntimeError, match='SOFTWARE integrity'):
        with trial:
            with trial.stages.span('rollout'):
                raise RuntimeError('SOFTWARE integrity')
    assert not trial.sampler.is_alive() and trial.measurement['end_to_end_ms'] >= 0


def test_missing_runtime_fit_does_not_fabricate_zero_cost_measured_trial(prepared, tmp_path):
    owner = consumer(prepared); work = select(owner, kind='runtime_cold', repeat=0)
    owner.models.pop(work['fit_identity']); owner.receipts.pop(work['fit_identity'])
    output = owner.execute(work, output_directory=tmp_path)
    assert output['status'] == 'DEPENDENCY_UNAVAILABLE'
    assert unpack(read_json(output['artifact_path']))['runtime'] is None
    assert restore(owner, work, output) == (None, None)
