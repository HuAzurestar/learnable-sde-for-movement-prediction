"""Final raw-output replay for checkpoint results, never a restart gate.

Fresh metrics are computed once from original saved paths/targets and reused
only within this independent audit process. No producer row cache supplies
numbers. Original mechanism/bootstrap/Holm/terrain/runtime kernels are reused.
This does not fabricate the retired legacy ledger/guard/bootstrap PASS.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from .checkpoint_resume import load
from .checkpoint_saved import CheckpointSavedForecasts
from .checkpoint_scoring import CachedScorer, CheckpointScores, NotReady, start_access
from .checkpoint_state import atomic_json, single_writer
from .formal_analysis import AnalysisConsumers
from .formal_forecast_records import KINDS
from .formal_runtime import RuntimeEvidence
from .formal_scoring import ScoringConsumers
from .protocol_core import canonical, digest, file_hash, publish, read_json, unpack

VERSION = 'pirc17-checkpoint-independent-audit-v2'
QUERY_ACCOUNTING_SCOPE = dict(
    feature_rows='Particle-integration feature-evaluation rows; terrain counters include selected history and map channels. Not unique locations or independent samples.',
    invalid_rows='Rows with at least one invalid selected numeric feature; a union over selected channels, not per-column or map-only missingness.',
    raw_map_rows='Saved query_rows_this_forecast only; cumulative provider attempted/completed counters and asset lists are deliberately not exported or summed.',
    absence='A missing source counter remains null; zero is a recorded count, not imputed absence. Nonterrain records may have no map counter.',
    ordinary_method_scope='Ordinary-method invalid_feature_rows is contractually zero, not a measured map-validity rate; no cross-matrix mask comparison is implied.',
    failed_scope='Failed or unavailable forecasts remain in the full inventory; retained query counts do not repair missing predictions or qualify comparisons.',
    per_column_validity_rates_available=False, history_and_map_missingness_separated=False,
    missing_input_robustness_established=False, independent_samples_implied=False)
COUNTS = dict(input_qualification_and_population=1, method_fit=16, terrain_fit=10,
    scientific_forecast=11020, same_grid_reference=290, inertial_path=58, common_scores=58,
    forecast_replay=38, runtime_cold=75, runtime_warmup=15, runtime_warm=75,
    mechanisms_and_inference=1, independent_reanalysis=1, aggregate_export=1)


def validate_query_accounting(value):
    fields = {'feature_query_rows', 'invalid_feature_rows', 'raw_map_query_rows_this_forecast'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('exact nullable query accounting counters required')
    if any(n is not None and (type(n) is not int or n < 0) for n in value.values()):
        raise ValueError('nonnegative integer query counter or explicit absence required')
    total, invalid = value['feature_query_rows'], value['invalid_feature_rows']
    if invalid is not None and (total is None or invalid > total):
        raise ValueError('invalid selected-feature rows require their full row denominator')
    return value


def forecast_query_accounting(payload):
    """Copy three existing scalar counters without arrays/maps or new queries."""
    payload = {} if payload is None else payload
    maps = payload.get('maps')
    result = dict(feature_query_rows=payload.get('feature_query_rows'),
        invalid_feature_rows=payload.get('invalid_feature_rows'),
        raw_map_query_rows_this_forecast=None if maps is None else maps.get('query_rows_this_forecast'))
    return validate_query_accounting(result)


class FreshScorer(ScoringConsumers):
    """Audit-owned fresh arithmetic, NOT on-disk producer cache verification."""
    def __init__(self, *, saved, cache_sha256, on_block=lambda: None):
        super().__init__(saved=saved, positions=saved.positions)
        self.cache_sha256, self.on_block = cache_sha256, on_block
        self.fresh_rows, self.verified_blocks = {}, 0

    def row(self, work, saved=None):
        wid = work['work_id']
        if self.saved.forecasts.get(wid) != work:
            raise ValueError('exact registered fresh-audit forecast required')
        if wid not in self.fresh_rows:
            self.fresh_rows[wid] = super().row(work, saved)
        return deepcopy(self.fresh_rows[wid])

    def verify(self, record, *, work, access_started_sha256):
        expected = self.compute(work)
        expected.update(metrics_access_started_sha256=access_started_sha256,
            checkpoint_cache_sha256=self.cache_sha256, independent_raw_output_replay_completed=False)
        if canonical(unpack(record)) != canonical(expected):
            raise ValueError('common scores differ from fresh original-array metric replay')
        self.verified_blocks += 1
        self.on_block()
        return dict(scores_recomputed=True, producer_cache_used_for_numbers=False,
                    expected_rows=196, new_forecasts=0, new_fits=0)


def sources(directory):
    saved = CheckpointSavedForecasts(directory)
    cache = CachedScorer(saved=saved, directory=directory)
    state = read_json(cache.cache_root/'progress.json')
    if state['cache_sha256'] != cache.cache_identity['sha256'] or set(state['blocks']) != set(cache.work):
        raise NotReady('all58 original common-score blocks required; audit is not a startup gate')
    return saved, cache, state


def analyze(directory):
    """Original inference with an honest access receipt and immutable pointer."""
    # Published pointers must not depend on the caller's working directory.
    directory = Path(directory).resolve()
    settings, _ = load(directory)
    access = start_access(directory, settings)
    saved, cache, state = sources(directory)
    pointer = cache.cache_root/'analysis.json'
    with single_writer(cache.cache_root/'analysis-owner'):
        if pointer.is_file():
            binding = read_json(pointer)
            payload = unpack(read_json(binding['path'], expected_file_sha256=binding['file_sha256']),
                             expected_sha256=binding['content_sha256'])
            if payload['saved_score_index_sha256'] != CheckpointScores(scorer=cache, root=cache.cache_root,
                    index=state['blocks'], access_journal=Path(directory)/'metric-access').identity['sha256']:
                raise ValueError('saved analysis uses another score index')
            return binding
        scores = CheckpointScores(scorer=cache, root=cache.cache_root, index=state['blocks'],
                                 access_journal=Path(directory)/'metric-access')
        analysis = AnalysisConsumers(scores=scores)
        work = next(iter(analysis.work.values()))
        value = analysis.compute(work)
        value.update(checkpoint_cache_sha256=cache.cache_identity['sha256'],
            metrics_access_started_sha256=access['sha256'], independent_raw_output_replay_completed=False)
        path, record = publish(cache.cache_root/'analysis', value)
        binding = dict(path=str(path), content_sha256=record['sha256'], file_sha256=file_hash(path))
        atomic_json(pointer, binding)
        return binding


def require_complete(saved, state):
    counts = Counter(w['kind'] for w in saved.work.values())
    if counts != COUNTS or unpack(saved.matrix)['counts_by_kind'] != COUNTS:
        raise ValueError('full11659-work registered denominator required')
    missing = [w['work_id'] for w in saved.forecasts.values() if not saved.terminal(w)]
    score_ids = {wid for wid,w in saved.work.items() if w['kind'] == 'common_scores'}
    if missing or set(state['blocks']) != score_ids:
        raise NotReady(f'final audit needs all original terminal forecasts/58scores; {len(missing)} forecasts unfinished')


def audit(directory, *, report_seconds=600):
    directory = Path(directory).resolve()
    settings, imported = load(directory)
    access = start_access(directory, settings)
    saved, cache, state = sources(directory)
    require_complete(saved, state)
    analysis_binding = read_json(cache.cache_root/'analysis.json')
    original = read_json(analysis_binding['path'], expected_file_sha256=analysis_binding['file_sha256'])
    original_value = unpack(original, expected_sha256=analysis_binding['content_sha256'])
    root = cache.cache_root/'independent-audit'
    with single_writer(root):
        started, reported = time.monotonic(), time.monotonic()
        def report(force=False):
            nonlocal reported
            now = time.monotonic()
            if not force and now-reported < report_seconds:
                return
            print(json.dumps(dict(event='independent_audit_progress', elapsed_seconds=now-started,
                fresh_rows=len(fresh.fresh_rows), expected_rows=11368, verified_blocks=fresh.verified_blocks,
                expected_blocks=58, new_forecasts=0, new_fits=0)), flush=True)
            reported = now
        fresh = FreshScorer(saved=saved, cache_sha256=cache.cache_identity['sha256'], on_block=report)
        scores = CheckpointScores(scorer=fresh, root=cache.cache_root, index=state['blocks'],
                                 access_journal=directory/'metric-access')
        scores._access(original_value['metrics_access_started_sha256'])
        analysis = AnalysisConsumers(scores=scores)
        work = next(iter(analysis.work.values()))
        recomputed = analysis.compute(work)
        expected = dict(recomputed, checkpoint_cache_sha256=cache.cache_identity['sha256'],
            metrics_access_started_sha256=original_value['metrics_access_started_sha256'],
            independent_raw_output_replay_completed=False)
        if canonical(original_value) != canonical(expected):
            raise ValueError('analysis differs from fresh raw-score/mechanism/statistical replay')
        runtime = RuntimeEvidence(saved).compute()
        inventory = []
        for wid,w in saved.work.items():
            kind = w['kind']
            if kind in KINDS:
                binding = saved.index.get(wid)
                value = unpack(read_json(binding['path']), expected_sha256=binding['content_sha256']) if binding else None
                status = value['status'] if value else 'failed' if wid in saved.failures else 'NOT_ADMITTED'
                evidence = None if binding is None else binding['content_sha256']
                role = 'forecast'
            elif kind in {'method_fit','terrain_fit'}:
                status, evidence, role = 'restored_original_fit', saved.receipts[w['fit_identity']]['sha256'], 'fit'
            elif kind == 'common_scores':
                status, evidence, role = 'fresh_scores_verified', state['blocks'][wid]['content_sha256'], 'scores'
            elif kind == 'input_qualification_and_population':
                status, evidence, role = 'retained_guarded_context', settings['context_sha256'], 'input'
                if wid not in imported:
                    raise ValueError('original retained input owner missing')
            elif kind == 'mechanisms_and_inference':
                status, evidence, role = 'fresh_analysis_verified', original['sha256'], 'analysis'
            else:
                status, evidence, role = 'current_audit' if kind == 'independent_reanalysis' else 'future_export', None, kind
            row = dict(work_id=wid, kind=kind, phase=w['phase'], status=status, role=role, evidence_sha256=evidence)
            if kind in KINDS:
                row['forecast_axes'] = {k:w[k] for k in ('matrix','subject','origin_mode','origin_rank','seed','repetition')}
                row['kernel_prediction_seconds'] = None if value is None else value.get('prediction_seconds')
                row['feature_query_accounting'] = forecast_query_accounting(value)
            inventory.append(row)
        report(force=True)
        value = dict(schema_version=VERSION, protocol_sha256=saved.protocol['sha256'],
            execution_sha256=saved.execution['sha256'], matrix_sha256=saved.matrix['sha256'],
            input_context_sha256=settings['context_sha256'], population_sha256=saved.population['sha256'],
            saved_score_index_sha256=scores.identity['sha256'], analysis_sha256=original['sha256'],
            recomputed_analysis_payload_sha256=digest(recomputed), factor_conclusions=recomputed['factor_conclusions'],
            metric_access_started_sha256=access['sha256'], full_work_inventory=inventory,
            checkpoint_inventory_sha256=digest(dict(index=saved.index, failures=saved.failures, settings_id=settings['settings_id'])),
            counts_by_kind=dict(Counter(r['kind'] for r in inventory)),
            counts_by_status=dict(Counter(r['status'] for r in inventory)), runtime_replay=runtime,
            feature_query_accounting_scope=deepcopy(QUERY_ACCOUNTING_SCOPE),
            raw_score_rows_recomputed=len(fresh.fresh_rows), common_blocks_recomputed=fresh.verified_blocks,
            producer_cache_used_for_numbers=False, independent_saved_output_reanalysis=True,
            new_forecasts=0, new_fits=0, raw_sources_independently_reloaded=False,
            retired_bootstrap_gate_asserted=False, scientific_claim_authorized=False, human_accepted=False,
            scope='Full fixed inventory/fresh saved-path scoring and statistics; failures retained,not new empirical replication.')
        path, record = publish(root/'results', value)
        binding = dict(path=str(path), content_sha256=record['sha256'], file_sha256=file_hash(path))
        atomic_json(cache.cache_root/'audit.json', binding)
        return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analysis','audit'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--report-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if args.report_seconds <= 0:
        parser.error('positive reporting interval required')
    print(json.dumps(analyze(args.directory) if args.command == 'analysis'
                     else audit(args.directory, report_seconds=args.report_seconds)), flush=True)


if __name__ == '__main__':
    main()
