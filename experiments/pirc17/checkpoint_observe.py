"""Read-only forecast observation with durable 30-minute summaries.

Writes only its own observation journal. Never reserves, stops or launches work.
The eight-hour entry requests a concurrency review; it does not change lanes.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import psutil

from .checkpoint_state import atomic_json
from .protocol_core import read_json


def utcnow():
    return datetime.now(timezone.utc)


def alive(identity):
    try:
        process = psutil.Process(identity['pid'])
        return abs(process.create_time()-identity['created']) < .1 and process.is_running()
    except psutil.Error:
        return False


def snapshot(session, scores):
    run = read_json(session/'run.json')
    report = read_json(session/'progress-report.json')
    assignment = read_json(session/'assignment.json')
    rows = read_json(scores)
    ready = read_json(session/'ready.json')
    resources = dict(available_memory_bytes=psutil.virtual_memory().available,
        checkpoint_disk_free_bytes=psutil.disk_usage(str(session.anchor)).free)
    worker_alive = False
    # The controller's PID/creation identity is authoritative; a recycled worker
    # PID after controller death must not be reported as this experiment.
    if alive(run['owner']):
        try:
            worker = psutil.Process(ready['pid'])
            worker_alive = worker.create_time() >= datetime.fromisoformat(run['started_utc']).timestamp()-1
            if worker_alive:
                memory, cpu = worker.memory_info(), worker.cpu_times()
                resources.update(worker_rss_bytes=memory.rss,
                    worker_private_bytes=getattr(memory, 'private', memory.vms),
                    worker_cpu_seconds=cpu.user+cpu.system)
        except psutil.Error:
            pass
    return dict(observed_utc=utcnow().isoformat(), session=session.name,
        controller_alive=alive(run['owner']), worker_alive=worker_alive,
        report_utc=datetime.fromtimestamp((session/'progress-report.json').stat().st_mtime, timezone.utc).isoformat(),
        science_success=report['completed_scientific'], science_total=11020,
        science_failed=report['failed_scientific'], science_runnable=report['runnable_scientific'],
        lanes={lane:dict(success=report['lane_science_completed'][lane], assigned=len(ids))
               for lane, ids in assignment['lanes'].items()},
        score_rows=len(rows['rows']), score_total=11368,
        score_blocks=len(rows['blocks']), score_blocks_total=58,
        cumulative_science_per_hour=report.get('scientific_successes_per_hour'),
        conditional_forecast_eta_hours=(report['rough_remaining_seconds']/3600
            if report.get('rough_remaining_seconds') is not None else None), resources=resources)


def summary(current, previous, *, event, review_due=False):
    value = dict(current, event=event, eight_hour_review_due=review_due,
        eta_scope='remaining scientific forecast queue only; excludes auxiliary work, analysis, audit and paper')
    if previous and previous['session'] == current['session']:
        # Use the actual producer report interval: its 10-minute report may be
        # older than our observation, so wall-clock polling time is not its span.
        span = (datetime.fromisoformat(current['report_utc'])-
                datetime.fromisoformat(previous['report_utc'])).total_seconds()
        delta = current['science_success']-previous['science_success']
        value.update(interval_seconds=span, science_added=delta,
            interval_science_per_hour=delta*3600/span if span > 0 else None,
            lane_added={lane:row['success']-previous['lanes'][lane]['success']
                        for lane, row in current['lanes'].items()})
    if review_due:
        if not current['science_runnable']:
            decision = 'Science queue terminal; continue downstream work with no extra forecast lanes.'
        elif current['conditional_forecast_eta_hours'] is not None and current['conditional_forecast_eta_hours'] <= 1:
            decision = 'Keep two lanes: remaining forecast queue is under one measured hour.'
        else:
            decision = 'Evaluate four lanes from current throughput, per-lane balance and resource headroom; implementation and a benefit estimate are required before a change.'
        value['concurrency_review'] = decision
    return value


def observe(session, scores, output, *, interval_seconds=1800, review_seconds=28800, once=False):
    output.mkdir(parents=True, exist_ok=True)
    state_path = output/'schedule.json'
    now = utcnow()
    if state_path.is_file():
        state = read_json(state_path)
        if state['session'] != session.name:
            raise ValueError('Use a new observation directory for a different forecast session.')
    else:
        state = dict(session=session.name, started_utc=now.isoformat(),
            next_summary_utc=(now+timedelta(seconds=interval_seconds)).isoformat(),
            review_due_utc=(now+timedelta(seconds=review_seconds)).isoformat(),
            review_reported=False, previous=None)
    state['observer'] = dict(pid=__import__('os').getpid(), created=psutil.Process().create_time())
    while True:
        current = snapshot(session, scores)
        now = utcnow()
        review_due = now >= datetime.fromisoformat(state['review_due_utc']) and not state['review_reported']
        due = now >= datetime.fromisoformat(state['next_summary_utc'])
        if state['previous'] is None or due or review_due or once:
            value = summary(current, state['previous'], event='eight_hour_review' if review_due else 'forecast_summary', review_due=review_due)
            atomic_json(output/'latest.json', value)
            with (output/'summaries.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, sort_keys=True)+'\n')
            print(json.dumps(value, sort_keys=True), flush=True)
            state['previous'] = current
            if due:
                # Skip missed notification windows after a computer shutdown;
                # never fabricate observations for those intervals.
                while datetime.fromisoformat(state['next_summary_utc']) <= now:
                    state['next_summary_utc'] = (datetime.fromisoformat(state['next_summary_utc'])+
                        timedelta(seconds=interval_seconds)).isoformat()
            state['review_reported'] |= review_due
        atomic_json(state_path, state)
        if once:
            return
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--score-progress', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    observe(args.session.resolve(), args.score_progress.resolve(), args.output_directory.resolve(), once=args.once)


if __name__ == '__main__':
    main()
