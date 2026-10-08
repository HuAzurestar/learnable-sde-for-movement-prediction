"""Low-overhead follower using the ORIGINAL scoring cache and arithmetic.

Numeric/source-binding files remain unchanged. Validate cached row references
once at pickup, then compare in-memory source bindings; unchanged rows need
no repeated context hashing/path lookup. Final fresh audit remains separate.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from .checkpoint_resume import load
from .checkpoint_saved import CheckpointSavedForecasts
from .checkpoint_scoring import CachedScorer, SCORE_KINDS, start_access
from .checkpoint_state import atomic_json, single_writer
from .protocol_core import file_hash, publish, read_json


def update_rows(scorer, state, observed, *, maximum=None):
    """Process changed/new terminal bindings only; immutable metadata is shared."""
    computed = 0
    for work in scorer.saved.forecasts.values():
        if work['kind'] not in SCORE_KINDS:
            continue
        wid = work['work_id']
        if not scorer.terminal(work):
            if wid in state['rows']:
                del state['rows'][wid]
                state['blocks'].clear()
                observed.pop(wid,None)
                atomic_json(scorer.cache_root/'progress.json',state)
            continue  # Never cache a future unfinished forecast as unavailable.
        binding = (scorer.saved.index.get(wid),scorer.saved.failures.get(wid))
        if wid in observed and observed[wid]==binding and wid in state['rows']:
            continue  # NO repeated row_scope/row_path/stat/target-array read.
        scope = scorer.row_scope(work)
        previous = state['rows'].get(wid)
        if (previous is not None and previous['source_sha256']==scope['source_sha256']
                and scorer.row_path(work).is_file()):
            observed[wid] = deepcopy(binding)
            continue
        if maximum is not None and computed>=maximum:
            break
        if previous is not None:
            state['blocks'].clear()
        row = scorer.row(work)  # Original scoped cache or ORIGINAL metric math.
        state['rows'][wid] = dict(status=row['status'],source_sha256=scope['source_sha256'])
        atomic_json(scorer.cache_root/'progress.json',state)
        observed[wid] = deepcopy(binding)
        computed += 1
    return computed


def complete_blocks(scorer, state, access):
    """Read/hash original row artifacts only when all196 dependencies exist."""
    for wid,work in scorer.work.items():
        if wid in state['blocks']:
            continue
        dependencies = scorer.dependencies[wid]
        if not all(w['work_id'] in state['rows'] and scorer.terminal(w) for w in dependencies):
            continue
        payload = scorer.compute(work)  # Original cached row validation/numbers.
        payload.update(metrics_access_started_sha256=access['sha256'],
            checkpoint_cache_sha256=scorer.cache_identity['sha256'],
            independent_raw_output_replay_completed=False)
        path,record = publish(scorer.cache_root/'common'/wid,payload)
        state['blocks'][wid] = dict(path=path.relative_to(scorer.cache_root).as_posix(),
            file_sha256=file_hash(path),content_sha256=record['sha256'],
            metrics_access_started_sha256=access['sha256'])
        atomic_json(scorer.cache_root/'progress.json',state)


def score(directory, *, follow=False, maximum_rows=None, poll_seconds=10, report_seconds=600):
    if poll_seconds<=0 or report_seconds<=0 or maximum_rows is not None and maximum_rows<0:
        raise ValueError('positive intervals/nonnegative maximum required')
    directory = Path(directory).resolve()
    settings,_ = load(directory)
    access = start_access(directory,settings)
    saved = CheckpointSavedForecasts(directory)
    scorer = CachedScorer(saved=saved,directory=directory)
    started = reported = time.monotonic()
    path = scorer.cache_root/'progress.json'
    with single_writer(scorer.cache_root):  # Same immediate duplicate rejection.
        state = read_json(path) if path.is_file() else dict(cache_sha256=scorer.cache_identity['sha256'],rows={},blocks={})
        if state['cache_sha256']!=scorer.cache_identity['sha256']:
            raise ValueError('original score cache identity changed')
        observed,processed = {},0
        expected = sum(w['kind'] in SCORE_KINDS for w in saved.forecasts.values())
        def report():
            result = dict(event='scoring_progress',scheduler='changed-binding-follower-v1',
                elapsed_seconds=time.monotonic()-started,cache_directory=str(scorer.cache_root),
                cached_rows=len(state['rows']),expected_rows=expected,complete_blocks=len(state['blocks']),
                expected_blocks=len(scorer.work),rows_computed_this_run=scorer.computed_rows,
                rows_reused_this_run=scorer.cached_rows,
                counts_by_status=dict(Counter(v['status'] for v in state['rows'].values())),
                new_forecasts=0,new_fits=0,independent_raw_output_replay_completed=False)
            atomic_json(scorer.cache_root/'latest-report.json',result)
            print(json.dumps(result),flush=True)
            return result
        report()
        while True:
            saved.refresh()
            processed += update_rows(scorer,state,observed,
                maximum=None if maximum_rows is None else maximum_rows-processed)
            complete_blocks(scorer,state,access)
            now = time.monotonic()
            if now-reported>=report_seconds:
                report(); reported=now
            if (len(state['rows'])==expected and len(state['blocks'])==len(scorer.work)
                    or maximum_rows is not None and processed>=maximum_rows or not follow):
                return report()
            time.sleep(poll_seconds)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--follow',action='store_true')
    parser.add_argument('--maximum-rows',type=int)
    parser.add_argument('--poll-seconds',type=float,default=10)
    parser.add_argument('--report-seconds',type=float,default=600)
    args=parser.parse_args(argv)
    print(json.dumps(score(args.directory,follow=args.follow,maximum_rows=args.maximum_rows,
        poll_seconds=args.poll_seconds,report_seconds=args.report_seconds)),flush=True)


if __name__=='__main__': main()
