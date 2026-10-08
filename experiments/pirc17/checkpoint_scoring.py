"""Incremental private scoring, not another forecast-startup gate.

Uses original metrics/targets and actual saved arrays. Completed row caches
survive interruption; unfinished forecasts are NOT frozen as unavailable.
Full common-score blocks retain all 196 rows. Cached arithmetic is explicitly
not an independent raw-output replay; that remains a final delivery step.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from .checkpoint_resume import load
from .checkpoint_saved import CheckpointSavedForecasts
from .checkpoint_state import atomic_json, single_writer
from .formal_analysis import AnalysisConsumers, SavedScores
from .formal_scoring import ScoringConsumers
from .metrics import METRICS_VERSION
from .protocol_core import digest, envelope, file_hash, publish, read_json, unpack

VERSION = 'pirc17-checkpoint-scoring-v1'
SCORE_KINDS = {'scientific_forecast', 'same_grid_reference', 'inertial_path'}


class NotReady(RuntimeError):
    pass


def access_scope(settings):
    return {k:settings[k] for k in ('settings_id', 'execution_sha256', 'matrix_sha256',
                                 'context_sha256', 'approval_sha256')}


def start_access(directory, settings):
    """Honest new orchestration receipt; never fabricate legacy VerifiedAccess."""
    return publish(Path(directory)/'metric-access', dict(schema_version=VERSION+'-access',
        event='started', access_kind='final_eval_metrics', scope=access_scope(settings),
        attempt_id=uuid.uuid4().hex, at_utc=datetime.now(timezone.utc).isoformat(),
        authority_source='existing user-authorized checkpoint continuation; original approval retained',
        legacy_admission_asserted=False, scientific_claim_authorized=False))[1]


class CachedScorer(ScoringConsumers):
    def __init__(self, *, saved, directory):
        super().__init__(saved=saved, positions=saved.positions)
        sources = {name:file_hash(Path(__file__).with_name(name)) for name in
                   ('checkpoint_scoring.py','checkpoint_saved.py','formal_scoring.py','metrics.py')}
        self.cache_identity = envelope(dict(schema_version=VERSION+'-cache', metrics_version=METRICS_VERSION,
            context_sha256=saved.identity['sha256'], scoring_inputs_sha256=self.inputs.identity['sha256'],
            sources=sources, weights=self.weights.tolist(), entropy_grid_sha256=self.grid.identity))
        self.cache_root = Path(directory)/'offline-scores'/self.cache_identity['sha256']
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.cached_rows, self.computed_rows = 0, 0

    def terminal(self, work):
        return self.saved.terminal(work)

    def row_scope(self, work):
        if self.saved.forecasts.get(work.get('work_id')) != work:
            raise ValueError('exact registered scoring dependency required')
        if not self.terminal(work):
            raise NotReady('forecast is unfinished; do not cache it as unavailable')
        case = self.saved.case(work)
        return dict(cache_sha256=self.cache_identity['sha256'], work_sha256=digest(work),
            source_sha256=digest(dict(binding=self.saved.index.get(work['work_id']),
                                     failure=self.saved.failures.get(work['work_id']))),
            target_sha256=None if case is None else self.inputs.target_sha256[case.sample_id],
            scoring_context_sha256=None if case is None else self.inputs.context_sha256(case))

    def row_path(self, work):
        scope = self.row_scope(work)
        return self.cache_root/'rows'/work['work_id']/(digest(scope)+'.json')

    def row(self, work, saved=None):
        scope, path = self.row_scope(work), self.row_path(work)
        if path.is_file():
            value = unpack(read_json(path))
            if (value.get('schema_version') != VERSION+'-row' or value.get('scope') != scope
                    or value['row']['forecast_work_id'] != work['work_id']):
                raise ValueError('score cache source/context/work changed')
            self.cached_rows += 1
            return deepcopy(value['row'])
        row = super().row(work, saved)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, envelope(dict(schema_version=VERSION+'-row', scope=scope, row=row)))
        self.computed_rows += 1
        return row

    def ready(self, work):
        return all(self.terminal(w) and self.row_path(w).is_file() for w in self.dependencies[work['work_id']])

    def verify(self, record, *, work, access_started_sha256):
        expected = self.compute(work)
        expected['metrics_access_started_sha256'] = access_started_sha256
        expected['checkpoint_cache_sha256'] = self.cache_identity['sha256']
        expected['independent_raw_output_replay_completed'] = False
        if unpack(record) != expected:
            raise ValueError('common score artifact differs from its exact cached rows')
        return dict(expected_rows=196, cached_score_rows_verified=True,
                    scores_recomputed=False, new_forecasts=0, new_fits=0)


class CheckpointScores(SavedScores):
    def _access(self, key):
        event = unpack(read_json(self.access_journal/(key+'.json')), expected_sha256=key)
        if (event.get('schema_version') != VERSION+'-access' or event.get('event') != 'started'
                or event.get('access_kind') != 'final_eval_metrics'
                or event.get('scope') != access_scope(self.scorer.saved.checkpoint_settings)
                or event.get('legacy_admission_asserted') is not False):
            raise ValueError('common scores lack matching checkpoint metric access receipt')


def finish_blocks(scorer, state, access):
    for wid, work in scorer.work.items():
        if wid in state['blocks'] or not scorer.ready(work):
            continue
        payload = scorer.compute(work)
        payload.update(metrics_access_started_sha256=access['sha256'],
            checkpoint_cache_sha256=scorer.cache_identity['sha256'],
            independent_raw_output_replay_completed=False)
        path, record = publish(scorer.cache_root/'common'/wid, payload)
        state['blocks'][wid] = dict(path=path.relative_to(scorer.cache_root).as_posix(),
            file_sha256=file_hash(path), content_sha256=record['sha256'],
            metrics_access_started_sha256=access['sha256'])
        atomic_json(scorer.cache_root/'progress.json', state)


def score(directory, *, maximum_rows=None, follow=False, report_seconds=600):
    directory = Path(directory).resolve()
    settings, _ = load(directory)
    # Record standing metric access before loading the saved target context.
    access = start_access(directory, settings)
    saved = CheckpointSavedForecasts(directory)
    scorer = CachedScorer(saved=saved, directory=directory)
    started, reported = time.monotonic(), time.monotonic()
    path = scorer.cache_root/'progress.json'
    with single_writer(scorer.cache_root):
        state = read_json(path) if path.is_file() else dict(cache_sha256=scorer.cache_identity['sha256'], rows={}, blocks={})
        if state['cache_sha256'] != scorer.cache_identity['sha256']:
            raise ValueError('score cache identity changed')
        works = [w for w in saved.forecasts.values() if w['kind'] in SCORE_KINDS]
        maximum = len(works) if maximum_rows is None else maximum_rows
        processed = 0
        def report():
            result = dict(event='scoring_progress', elapsed_seconds=time.monotonic()-started,
                cache_directory=str(scorer.cache_root), cached_rows=len(state['rows']), expected_rows=len(works),
                complete_blocks=len(state['blocks']), expected_blocks=len(scorer.work),
                rows_computed_this_run=scorer.computed_rows, rows_reused_this_run=scorer.cached_rows,
                counts_by_status=dict(Counter(v['status'] for v in state['rows'].values())),
                new_forecasts=0, new_fits=0, independent_raw_output_replay_completed=False)
            atomic_json(scorer.cache_root/'latest-report.json', result)
            print(json.dumps(result), flush=True)
            return result
        report()
        while True:
            saved.refresh()
            for work in works:
                wid = work['work_id']
                if not scorer.terminal(work):
                    continue
                scope = scorer.row_scope(work)
                previous = state['rows'].get(wid)
                if (previous is not None and previous['source_sha256'] == scope['source_sha256']
                        and scorer.row_path(work).is_file()):
                    continue
                if previous is not None:
                    # An explicitly changed terminal source cannot leave a
                    # stale complete block accepted. Old artifacts stay saved.
                    state['blocks'].clear()
                if processed >= maximum:
                    break
                row = scorer.row(work)
                state['rows'][wid] = dict(status=row['status'], source_sha256=scope['source_sha256'])
                atomic_json(path, state)
                processed += 1
                if time.monotonic()-reported >= report_seconds:
                    report(); reported = time.monotonic()
            finish_blocks(scorer, state, access)
            if len(state['rows']) == len(works) or processed >= maximum or not follow:
                return report()
            # Wait only for new forecast completions, not locks or restoration.
            time.sleep(2)


def analyze(directory):
    saved = CheckpointSavedForecasts(directory)
    scorer = CachedScorer(saved=saved, directory=directory)
    state = read_json(scorer.cache_root/'progress.json')
    if set(state['blocks']) != set(scorer.work):
        raise NotReady('all original common-score blocks required before inference')
    scores = CheckpointScores(scorer=scorer, root=scorer.cache_root, index=state['blocks'],
        access_journal=Path(directory)/'metric-access')
    analysis = AnalysisConsumers(scores=scores)
    work = next(iter(analysis.work.values()))
    result = analysis.compute(work)
    result.update(checkpoint_cache_sha256=scorer.cache_identity['sha256'],
        independent_raw_output_replay_completed=False)
    path, record = publish(scorer.cache_root/'analysis', result)
    return dict(path=str(path), sha256=record['sha256'], independent_raw_output_replay_completed=False,
                remaining='Independent raw-output replay/runtime/export/cards/full manuscript/final human acceptance')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('score', 'analyze'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--maximum-rows', type=int)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--report-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if args.maximum_rows is not None and args.maximum_rows < 0 or args.report_seconds <= 0:
        parser.error('nonnegative maximum rows and positive reporting interval required')
    result = score(args.directory, maximum_rows=args.maximum_rows, follow=args.follow,
                   report_seconds=args.report_seconds) if args.command == 'score' else analyze(args.directory)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
