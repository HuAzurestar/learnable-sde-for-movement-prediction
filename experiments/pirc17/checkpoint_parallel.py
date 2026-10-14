"""Two authorized terrain lanes, one snapshot writer and one model load.

Only original missing scientific terrain WorkIDs run concurrently. Original
auxiliary/replay/runtime work stays on lane zero AFTER both science lanes.
No historical array scan, refit, new seed/grid, application lock or retry.
"""
import argparse
from collections import deque
from copy import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .checkpoint_resume import INPUT, KINDS, load, resource_observation, status, wait_reply
from .checkpoint_state import NS, Progress, atomic_json, single_writer
from .protocol_core import digest, file_hash, read_json

MODULE = 'experiments.pirc17.checkpoint_parallel'
SOURCES = ('checkpoint_parallel.py', 'checkpoint_state.py', 'checkpoint_inputs.py')


def assignment(progress):
    """Freeze two disjoint lists in original matrix order, not score order."""
    missing = progress.missing(kinds=KINDS)
    science = [w for w in missing if w['kind'] == 'scientific_forecast']
    if any(w['matrix'] != 'terrain' for w in science):
        raise ValueError('this authorization is only for remaining terrain science')
    cut = (len(science)+1)//2
    order = {phase:i for i,phase in enumerate(progress.settings['phase_order'])}
    auxiliary = sorted((w for w in missing if w['kind'] != 'scientific_forecast'),
                       key=lambda w: order[w['phase']])
    value = dict(schema_version='pirc17-two-terrain-lanes-v1',
        settings_id=progress.settings['settings_id'], baseline=status(progress),
        lanes={'0':[w['work_id'] for w in science[:cut]], '1':[w['work_id'] for w in science[cut:]]},
        serial_after_science=[w['work_id'] for w in auxiliary],
        scope='original missing WorkIDs only; old successes/failures excluded; runtime serial after both lanes')
    value['assignment_id'] = digest(value)
    return value


def clone_consumer(shared):
    """No constructor/refit: immutable fitted state shared; mutable state owned."""
    result = copy(shared)
    result.maps = shared.maps.fresh_provider()
    result.attempted = set(shared.attempted)
    result.driver_origin, result.drivers = None, {}
    result.runtime_hardware = result.runtime_maps = result.runtime_state = None
    result.runtime_session = uuid.uuid4().hex
    return result


def lane_loop(consumer, session, lane, stop):
    """A provider remains on one dedicated OS thread for its whole life."""
    sequence = 1
    try:
        atomic_json(session/f'lane-{lane}-ready.json', dict(lane=lane, thread_id=threading.get_native_id()))
        while not stop.is_set():
            request_path = session/'requests'/lane/f'{sequence:06d}.json'
            if request_path.is_file():
                request = read_json(request_path)
                wid = request['work_id']
                output = Path(request['output_directory'])
                try:
                    result = consumer.execute(consumer.work[wid], output_directory=output)
                    message = dict(work_id=wid, result=result)
                except Exception as exc:
                    output.mkdir(parents=True, exist_ok=True)
                    message = dict(work_id=wid, error=f'{type(exc).__name__}: {exc}')
                atomic_json(output/'reply.json', message)
                sequence += 1
            stop.wait(.05)
    finally:
        consumer.close_all()


def worker(directory, session):
    from .checkpoint_inputs import CheckpointForecasts
    settings, imported = load(directory)
    shared = CheckpointForecasts(settings, imported)  # Exactly ONE load of 26 saved models.
    session = Path(session)
    stop = threading.Event()
    errors = []
    def lane(lane_id):
        consumer = None
        try:
            consumer = shared if lane_id == '0' else clone_consumer(shared)
            lane_loop(consumer, session, lane_id, stop)
        except BaseException as exc:
            errors.append(f'lane {lane_id}: {type(exc).__name__}: {exc}')
            stop.set()
        finally:
            if consumer is not None:
                consumer.close_all()
    threads = [threading.Thread(target=lane, args=(str(i),), name=f'pirc17-terrain-{i}') for i in range(2)]
    try:
        for thread in threads:
            thread.start()
        while not all((session/f'lane-{i}-ready.json').is_file() for i in range(2)):
            if errors:
                raise RuntimeError('; '.join(errors))
            time.sleep(.05)
        atomic_json(session/'ready.json', dict(models=len(shared.models), lanes=2,
            pid=__import__('os').getpid(), model_loads=1))
        while not stop.wait(.1):
            if (session/'worker-stop.json').is_file():
                stop.set()
        if errors:
            raise RuntimeError('; '.join(errors))
    finally:
        stop.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        shared.close_all()


def run(directory, *, report_seconds=600, _worker_command=None):
    """Single external controller; lane callbacks never access Progress."""
    from .seed_resume_session import OwnedJob
    directory = Path(directory).resolve()
    with single_writer(directory) as owner:
        started = time.monotonic_ns()
        settings, imported = load(directory)
        progress = Progress(directory, settings=settings, imported_ids=imported)
        progress.recover_pending()  # At most TWO receipts, never old arrays.
        plan = assignment(progress)
        queues = {i:deque(plan['lanes'][i]) for i in ('0','1')}
        auxiliary = deque(plan['serial_after_science'])
        if not any(queues.values()) and not auxiliary:
            return status(progress)
        session = directory/'sessions'/uuid.uuid4().hex
        for lane in queues:
            (session/'requests'/lane).mkdir(parents=True)
        atomic_json(session/'assignment.json', plan)
        sources = {name:file_hash(Path(__file__).with_name(name)) for name in SOURCES}
        atomic_json(session/'run.json', dict(owner=owner, started_ns=started,
            started_utc=datetime.now(timezone.utc).isoformat(), execution_sha256=settings['execution_sha256'],
            ignore_time_budgets=True, report_seconds=report_seconds, lanes=2,
            baseline_scientific=plan['baseline']['completed_scientific'], assignment_id=plan['assignment_id'],
            orchestration_sources=sources, cost_scope='sum of lane elapsed seconds; concurrent wall time reported separately'))
        initial = status(progress)['completed_scientific']
        dispatch_time, sequences = {}, {'0':0, '1':0}
        lane_done = {'0':0, '1':0}
        next_report = started+int(report_seconds*NS)
        ready_time = None
        def report(force=False):
            nonlocal next_report
            now = time.monotonic_ns()
            if not force and now < next_report:
                return
            current = status(progress)
            successes = current['completed_scientific']-initial
            elapsed = (now-ready_time)/NS if ready_time else 0
            rate = successes/elapsed if successes and elapsed > 0 else None
            event = dict(event='progress_report', **current, lanes=2,
                elapsed_seconds=(now-started)/NS, new_scientific_this_run=successes,
                ignore_time_budgets=True, scientific_successes_per_hour=rate*3600 if rate else None,
                rough_remaining_seconds=current['runnable_scientific']/rate if rate else None,
                estimate_scope='since readiness measured wall throughput, not CPU time or full-project ETA',
                pending_by_lane=progress.value.get('parallel_pending', {}),
                lane_science_completed=lane_done, assignment_id=plan['assignment_id'])
            atomic_json(session/'progress-report.json', event)
            print(json.dumps(event), flush=True)
            resource_observation(directory)
            next_report = now+int(report_seconds*NS)
        stop_path = directory/'stop.json'
        stop_requested = lambda: stop_path.is_file() and read_json(stop_path)['token'] == owner['token']
        job, child, assigned = OwnedJob(), None, False
        progress.reserve(phase=INPUT, maximum_ns=120*NS, ignore_time_budgets=True)
        try:
            with (session/'worker.log').open('xb') as output:
                bootstrap = ('import sys; sys.stdin.buffer.read(1)==b"R" or sys.exit(125); '
                    'import subprocess; sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL, '
                    'creationflags=subprocess.CREATE_NO_WINDOW))')
                command = (_worker_command(directory, session) if _worker_command else
                    [sys.executable,'-u','-m',MODULE,'worker','--directory',str(directory),'--session',str(session)])
                child = subprocess.Popen([sys._base_executable,'-I','-S','-u','-c',bootstrap,*command],
                    stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                job.assign(child)
                assigned = True
                child.stdin.write(b'R'); child.stdin.flush(); child.stdin.close()
                ready = wait_reply(session/'ready.json', child=child, deadline_ns=None,
                    stop_requested=stop_requested, report=report)
                ready_time = time.monotonic_ns()
                progress.settle(elapsed_ns=ready_time-started)
                print(json.dumps(dict(event='parallel_checkpoint_ready', session=str(session),
                    startup_seconds=(ready_time-started)/NS, models=ready['models'], lanes=2,
                    worker_pid=ready.get('pid'), assignments={i:len(q) for i,q in queues.items()},
                    serial_after_science=len(auxiliary), baseline_scientific=initial,
                    assignment_id=plan['assignment_id'])), flush=True)
                report(True)
                while any(queues.values()) or auxiliary or progress.value.get('parallel_pending'):
                    if stop_requested():
                        raise KeyboardInterrupt('checkpoint stop requested')
                    for lane, pending in list(progress.value.get('parallel_pending', {}).items()):
                        if Path(pending['reply']).is_file():
                            message = read_json(pending['reply'])
                            if message['work_id'] != pending['work_id']:
                                raise ValueError('worker reply work differs')
                            result = message.get('result')
                            progress.settle_lane(lane, elapsed_ns=time.monotonic_ns()-dispatch_time.pop(lane),
                                result=result, failure=message.get('error'))
                            if result and result.get('status') == 'success' and progress.work[pending['work_id']]['kind'] == 'scientific_forecast':
                                lane_done[lane] += 1
                            if message.get('error'):
                                raise RuntimeError(message['error'])
                    science_active = any(queues.values()) or any(
                        progress.work[p['work_id']]['kind'] == 'scientific_forecast'
                        for p in progress.value.get('parallel_pending', {}).values())
                    for lane in queues:
                        if lane in progress.value.get('parallel_pending', {}):
                            continue
                        queue = queues[lane] if science_active else auxiliary if lane == '0' else deque()
                        if not queue:
                            continue
                        wid = queue.popleft()
                        work = progress.work[wid]
                        output_directory = directory/'outputs'/wid
                        reply = output_directory/'reply.json'
                        began = time.monotonic_ns()
                        progress.reserve_lane(lane, phase=work['phase'], maximum_ns=int(work['max_active_seconds']*NS),
                            work_id=wid, reply=str(reply), ignore_time_budgets=True)
                        dispatch_time[lane] = began
                        sequences[lane] += 1
                        atomic_json(session/'requests'/lane/f'{sequences[lane]:06d}.json',
                            dict(work_id=wid, output_directory=str(output_directory)))
                    report()
                    if child.poll() is not None:
                        raise RuntimeError(f'checkpoint worker exited {child.returncode}; see worker.log')
                    time.sleep(.05)
        finally:
            try:
                if assigned:
                    job.terminate()
                elif child is not None and child.poll() is None:
                    child.kill()
                if child is not None:
                    child.wait(timeout=20)
                end = time.monotonic_ns()+20*NS
                while job.accounting()['active_processes']:
                    if time.monotonic_ns() >= end:
                        raise RuntimeError('owned process tree did not terminate')
                    time.sleep(.05)
                for lane, pending in list(progress.value.get('parallel_pending', {}).items()):
                    progress.settle_lane(lane, elapsed_ns=time.monotonic_ns()-dispatch_time[lane],
                        result=progress._pending_result(pending), failure='interrupted; no silent retry')
                if progress.value['pending'] is not None:
                    progress.settle(elapsed_ns=time.monotonic_ns()-started,
                        result=progress.pending_result(), failure='startup interrupted')
                atomic_json(session/'closed.json', dict(accounting=job.accounting(),
                    process_tree_closed=True, orchestration_sources=sources,
                    controller_wall_seconds=(time.monotonic_ns()-started)/NS))
            finally:
                job.close()
        report(True)
        return status(progress)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run','worker'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--report-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if args.report_seconds <= 0 or (args.command == 'worker' and args.session is None):
        parser.error('positive report interval and worker session required')
    if args.command == 'worker':
        worker(args.directory, args.session)
    else:
        print(json.dumps(run(args.directory, report_seconds=args.report_seconds)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
