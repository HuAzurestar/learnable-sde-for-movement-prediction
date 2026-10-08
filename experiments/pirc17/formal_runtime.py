"""Saved runtime and replay evidence for the registered finite audit workload.

No forecast/training entrypoint, extra measurement loop, access authorization
or successful-subset replacement. The independent audit/export owner calls
this inside its measured scope after closed output registration.
"""
from collections import Counter
from copy import deepcopy
import hashlib

import numpy as np

from .formal_saved import SavedForecasts
from .formal_timing import SCOPE
from .inference import SEEDS
from .protocol_core import digest, envelope, unpack
from .runtime import summarize_trials

VERSION = 'pirc17-formal-runtime-replay-evidence-v1'


def prediction_identity(prediction):
    """Canonical array bytes/meaning, excluding ZIP metadata and elapsed costs."""
    if prediction.forecast is None:
        return None
    p, forecast, native = unpack(prediction.record), prediction.forecast, prediction.native
    arrays = dict(elapsed_seconds=forecast.elapsed_seconds, positions_m=forecast.positions_m)
    if native is not None:
        arrays['method_positions_m'] = native.forecast.positions_m
        if native.conditional_means_m is not None:
            arrays.update(conditional_means_m=native.conditional_means_m, conditional_covariances_m2=native.conditional_covariances_m2)
    identity = {k: p[k] for k in ('origin_id', 'case_sha256', 'fit_identity', 'fit_receipt_sha256', 'parameter_identity',
                                 'forecast_version', 'invalid_feature_rows', 'feature_query_rows', 'diagnostics')}
    identity['arrays'] = {key: dict(shape=list(value.shape), float64_bytes_sha256=hashlib.sha256(
        np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()) for key, value in arrays.items()}
    identity['raw_map_query_rows'] = None if p['maps'] is None else p['maps']['query_rows_this_forecast']
    return digest(identity)


class RuntimeEvidence:
    def __init__(self, saved):
        if not isinstance(saved, SavedForecasts):
            raise ValueError('bound actual saved forecast collection required')
        self.saved = saved
        p, matrix = unpack(saved.protocol), unpack(saved.matrix)
        policy = p['components']['finite_delivery']['replay']['runtime']
        method_subjects = {('NEX326-methods', row['slot_id']) for row in matrix['method_ledger'] if row['disposition'] != 'EXCLUDED'}
        terrain_subjects = {('terrain', name) for name in matrix['terrain_ledger']}
        self.replays = { (w['matrix'], w['subject']): w for w in saved.forecasts.values() if w['kind'] == 'forecast_replay'}
        groups = terrain_subjects | {('NEX326-methods', name) for name in policy['methods']}
        self.runtime = {key: [w for w in saved.forecasts.values() if w['kind'].startswith('runtime_')
            and (w['matrix'], w['subject']) == key] for key in sorted(groups)}
        runtime_count = sum(w['kind'].startswith('runtime_') for w in saved.forecasts.values())
        if (set(self.replays) != method_subjects | terrain_subjects or len(self.replays) != 38
                or sum(w['kind'] == 'forecast_replay' for w in saved.forecasts.values()) != 38
                or len(self.runtime) != 15 or runtime_count != 165 or sum(map(len, self.runtime.values())) != runtime_count):
            raise ValueError('complete registered 38-replay/15-subject runtime inventory required')
        expected_axes = {(kind, i) for kind, n in [('runtime_cold', 5), ('runtime_warmup', 1), ('runtime_warm', 5)] for i in range(n)}
        if (policy['cold_trials'] != 5 or policy['warmup_trials'] != 1 or policy['warm_trials'] != 5
                or policy['seed'] != SEEDS[0] or any(len(rows) != 11 or {(w['kind'], w['repetition']) for w in rows} != expected_axes
                    for rows in self.runtime.values())):
            raise ValueError('registered cold/warmup/warm denominator changed')
        works = [*self.replays.values(), *[w for rows in self.runtime.values() for w in rows]]
        if any(w['origin_mode'] != 'causal_prefix' or w['origin_rank'] != 0 or w['seed'] != SEEDS[0]
               or w['scientific'] or w['generated_forecasts'] != 1 for w in works):
            raise ValueError('runtime/replay must retain first primary origin/seed, never scientific replication')
        self.dependencies = {w['work_id'] for w in works}
        self.baselines = {}
        for key in self.replays:
            self.baselines[key] = saved.find(key[1], 'causal_prefix', 0, SEEDS[0], matrix=key[0])
            self.dependencies.add(self.baselines[key]['work_id'])
        self.settings = deepcopy(p['forecast_contract']['forecast'])

    def compute(self):
        saved, replays = self.saved, []
        for key, work in self.replays.items():
            original, replay = saved.read(self.baselines[key]), saved.read(work)
            a, b = prediction_identity(original), prediction_identity(replay)
            if original.status == replay.status == 'success':
                status, same = ('reproduced' if a == b else 'mismatch'), a == b
            elif original.status == replay.status == 'failed':
                same = original.reason == replay.reason
                status = 'reproduced_failure' if same else 'mismatch'
            elif original.status in {'success', 'failed'} and replay.status in {'success', 'failed'}:
                status, same = 'mismatch', False
            else:
                status, same = ('NOT_ADMITTED' if saved.case(work) is None else 'unavailable'), None
            replays.append(dict(matrix=key[0], subject=key[1], original_work_id=self.baselines[key]['work_id'], replay_work_id=work['work_id'],
                original_sha256=None if original.record is None else original.record['sha256'],
                replay_sha256=None if replay.record is None else replay.record['sha256'],
                original_status=original.status, replay_status=replay.status, original_reason=original.reason, replay_reason=replay.reason,
                original_prediction_identity=a, replay_prediction_identity=b, status=status, output_identity_equal=same,
                successful_prediction_reproduced=status == 'reproduced'))
        hardware, providers, runtime = {}, set(), []
        for key, works in self.runtime.items():
            rows, previous_end = [], None
            for work in works:
                prediction = saved.read(work)
                payload = None if prediction.record is None else unpack(prediction.record)
                timing = None if payload is None else payload['runtime']
                if timing is not None:
                    if previous_end is not None and timing['trial']['started_monotonic_ns'] < previous_end:
                        raise ValueError('registered runtime trials overlap or changed execution order')
                    previous_end = timing['trial']['ended_monotonic_ns']
                    hardware[timing['hardware']['sha256']] = timing['hardware']
                    if work['kind'] != 'runtime_warm':
                        if timing['provider_instance_id'] in providers:
                            raise ValueError('cold/warmup trials reused a different trial provider identity')
                        providers.add(timing['provider_instance_id'])
                rows.append(dict(work_id=work['work_id'], kind=work['kind'], repetition=work['repetition'], status=prediction.status,
                    reason=prediction.reason, forecast_sha256=None if prediction.record is None else prediction.record['sha256'],
                    runtime=timing, warm_condition_verified=None))
            warmup = next(r for r in rows if r['kind'] == 'runtime_warmup')
            for row in rows:
                timing = row['runtime']
                if row['kind'] != 'runtime_warm' or timing is None: continue
                prior = warmup['runtime']
                if prior is None:
                    row['warm_condition_verified'] = False
                else:
                    if (timing['warmup_result_sha256'] != warmup['forecast_sha256'] or timing['warmup_status'] != warmup['status']
                            or timing['provider_instance_id'] != prior['provider_instance_id']
                            or timing['hardware'] != prior['hardware']
                            or timing['trial']['started_monotonic_ns'] < prior['trial']['ended_monotonic_ns']):
                        raise ValueError('warm trial differs from actual same-subject provider/warmup source')
                    row['warm_condition_verified'] = warmup['status'] == 'success'
            summaries = {}
            for kind in ('runtime_cold', 'runtime_warm'):
                trials = [r for r in rows if r['kind'] == kind]
                complete = all(r['runtime'] is not None and r['status'] in {'success', 'failed'}
                    and (kind != 'runtime_warm' or r['warm_condition_verified']) for r in trials)
                values = [dict(status='success' if r['status'] == 'success' else 'failure',
                    end_to_end_ms=r['runtime']['trial']['end_to_end_ms']) for r in trials] if complete else []
                horizons = {r['runtime']['actual_horizon_seconds'] for r in trials if r['runtime'] is not None}
                if len(horizons) > 1:
                    raise ValueError('runtime trials changed the fixed original scoring horizon')
                summary = summarize_trials(values, horizon_seconds=next(iter(horizons))) if complete else None
                raw = [r['runtime']['trial'] for r in trials if r['runtime'] is not None]
                lifetime = [r['process_lifetime_peak_rss_bytes'] for r in raw if r['process_lifetime_peak_rss_bytes'] is not None]
                successes = [r['runtime']['trial'] for r in trials if r['status'] == 'success' and r['runtime'] is not None]
                summaries[kind] = dict(required_trials=5, status='computed' if complete else 'unavailable',
                    counts_by_status=dict(Counter(r['status'] for r in trials)), summary=summary,
                    reason=None if complete else 'incomplete actual trials or unsuccessful/unverified same-subject warmup; no substitute run',
                    observed_sampled_peak_rss_bytes=max((r['sampled_peak_rss_bytes'] for r in raw), default=None),
                    observed_process_lifetime_peak_rss_bytes=max(lifetime, default=None),
                    successful_inclusive_stage_p50_ms=None if not complete or not successes else {
                        name: float(np.quantile([r['inclusive_stage_ms'][name] for r in successes], .5)) for name in successes[0]['inclusive_stage_ms']})
            runtime.append(dict(matrix=key[0], subject=key[1], rows=rows, conditions=summaries,
                warmup_excluded_from_latency_quantiles=True, memory_scope=SCOPE['memory']))
        if len(hardware) > 1:
            raise ValueError('runtime trials changed their actual hardware/thread environment')
        return envelope(dict(schema_version=VERSION, protocol_sha256=saved.protocol['sha256'], execution_sha256=saved.execution['sha256'],
            matrix_sha256=saved.matrix['sha256'], population_sha256=saved.population['sha256'],
            saved_forecast_index_sha256=saved.index_identity(self.dependencies)['sha256'],
            settings=self.settings, hardware=hardware, timing_scope=deepcopy(SCOPE), replay_rows=replays,
            replay_counts_by_status=dict(Counter(r['status'] for r in replays)), runtime_subjects=runtime,
            registered_counts=dict(replays=38, cold=75, warmup=15, warm=75),
            replay_scope='Exact saved prediction identity on first primary origin/seed; retained failures are not successful forecasts or independent empirical replication.',
            verification_scope='Saved bytes/domain, timing arithmetic and actual warmup linkage; not independent remeasurement of historical latency. Controller closure and charged work costs remain separate.',
            latency_scope='Five trials per condition give descriptive quantiles only; unavailable denominators do not trigger new trials.',
            new_forecasts=0, new_fits=0, scientific_claim_authorized=False))

    def verify(self, record):
        unpack(record)
        if record != self.compute():
            raise ValueError('runtime/replay evidence differs from actual saved sources')
        return dict(saved_runtime_replay_recomputed=True, new_forecasts=0, new_fits=0,
            historical_latency_independently_remeasured=False)
