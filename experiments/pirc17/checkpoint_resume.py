"""Simple PIRC-17 saved-state continuation, replacing restart-wide admission.

init reads the existing index ONCE; run loads causal context and 26 models,
then executes only missing forecasts with the unchanged scientific kernels.
No import copies, historical ledger replay, full snapshot scan, parent domain
revalidation, profiling hooks, monitor threads or reentrant meter locks.

The old ledgers remain untouched. Each new forecast has an atomic checkpoint
and measured cumulative budget. A retained Windows Job is only a process-tree
kill switch (including supervisor death), not an experiment admission lock.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import psutil

from .checkpoint_state import NS, Progress, atomic_json, single_writer
from .protocol_core import digest, file_hash, read_json, unpack

KINDS = {'scientific_forecast', 'same_grid_reference', 'inertial_path', 'forecast_replay',
         'runtime_cold', 'runtime_warmup', 'runtime_warm'}
INPUT = 'input_qualification_and_binding'
MODULE = 'experiments.pirc17.checkpoint_resume'


def initialize(directory, *, bundle_path, import_manifest, stopped_startup_ns, closure_note):
    """Reuse the completed saved bootstrap, without asserting its old gate PASS."""
    if type(stopped_startup_ns) is not int or stopped_startup_ns <= 0 or not closure_note:
        raise ValueError('explicit retained stopped-run charge and closure note required')
    with single_writer(directory):
        directory = Path(directory).resolve()
        path = directory/'settings.json'
        if path.exists():
            raise FileExistsError('checkpoint already initialized; use run, not init again')
        bundle = unpack(read_json(bundle_path))
        runtime = unpack(bundle['runtime'])
        policy = unpack(unpack(runtime['predecessor'])['resource_policy'])
        manifest_record = read_json(import_manifest)
        manifest = unpack(manifest_record)
        scope = unpack(read_json(manifest['import_scope']['path']))
        matrix = unpack(bundle['matrix'])
        if (manifest['execution_sha256'] != bundle['execution']['sha256']
                or manifest['matrix_sha256'] != bundle['matrix']['sha256']
                or len(manifest['imported_entries']) != policy['retained_success_count']):
            raise ValueError('saved index does not match the retained experiment')
        charged = dict(policy['retained_charged_ns_by_phase'])
        charged[INPUT] += stopped_startup_ns
        settings = dict(schema_version='pirc17-simple-checkpoint-v1', bundle=str(Path(bundle_path).resolve()),
            bundle_sha256=digest(bundle), execution_sha256=bundle['execution']['sha256'],
            matrix_sha256=bundle['matrix']['sha256'], import_manifest=str(Path(import_manifest).resolve()),
            import_manifest_sha256=manifest_record['sha256'], context=scope['successor_context']['path'],
            context_sha256=scope['successor_context']['content_sha256'], approval_sha256=manifest['approval_sha256'],
            input_paths=runtime['input_paths'], workloads=matrix['workloads'], phase_order=runtime['phase_order'],
            charged_ns_by_phase=charged, phase_caps_ns=policy['phase_caps_ns'], total_cap_ns=policy['total_cap_ns'],
            generated=policy['retained_generation_reservations'], generation_limit=policy['effective_generation_limit'],
            stopped_startup_ns=stopped_startup_ns, closure_note=closure_note,
            mode='user-authorized simple checkpoint; old bootstrap gate NOT asserted passed')
        settings['settings_id'] = digest(settings)
        atomic_json(path, settings)
        progress = Progress(directory, settings=settings, imported_ids=manifest['imported_entries'])
        return status(progress)


def load(directory):
    directory = Path(directory).resolve()
    settings = read_json(directory/'settings.json')
    if digest({k:v for k,v in settings.items() if k != 'settings_id'}) != settings['settings_id']:
        raise ValueError('checkpoint settings changed')
    # This import is metadata-only; don't import the scientific libraries in
    # the supervisor or rehash model/prediction arrays here.
    manifest = unpack(read_json(settings['import_manifest']), expected_sha256=settings['import_manifest_sha256'])
    if manifest['execution_sha256'] != settings['execution_sha256']:
        raise ValueError('saved index belongs to another execution')
    return settings, manifest['imported_entries']


def status(progress):
    completed = progress.imported_ids | progress.value['completed'].keys()
    scientific = [w for w in progress.work.values() if w['kind'] == 'scientific_forecast']
    failed = progress.value['failures'].keys()
    return dict(completed_scientific=sum(w['work_id'] in completed for w in scientific),
        remaining_scientific=sum(w['work_id'] not in completed for w in scientific),
        failed_scientific=sum(w['work_id'] in failed for w in scientific),
        runnable_scientific=sum(w['work_id'] not in completed and w['work_id'] not in failed
                                for w in scientific),
        new_successes=len(progress.value['completed']), failures=len(progress.value['failures']),
        pending=progress.value['pending'], generated=progress.value['generated'],
        charged_seconds={k:v/NS for k,v in progress.value['charged_ns_by_phase'].items()})


def resource_observation(directory):
    """Advisory only: a transient machine-wide reading must not kill a rollout.

    Actual allocation/write errors still surface normally. Observe at startup
    and scheduled reports, not every 50ms or before every forecast.
    """
    warnings = []
    try:
        available = psutil.virtual_memory().available
        if available < 2*1024**3:
            warnings.append(dict(resource='ram', available_bytes=available))
    except (OSError, psutil.Error) as exc:
        warnings.append(dict(resource='ram_observation', error=str(exc)))
    for root in {Path(directory).anchor, Path.cwd().anchor}:
        try:
            free = shutil.disk_usage(root).free
            if free < 10*1024**3:
                warnings.append(dict(resource='disk', root=root, available_bytes=free))
        except OSError as exc:
            warnings.append(dict(resource='disk_observation', root=root, error=str(exc)))
    if warnings:
        print(json.dumps(dict(event='resource_warning', advisory_only=True, warnings=warnings)), flush=True)


def wait_reply(path, *, child, deadline_ns, stop_requested, report=lambda: None):
    """External bounded poll, not a callback inside scientific code/locks."""
    while True:
        if stop_requested():
            raise KeyboardInterrupt('checkpoint stop requested')
        report()  # Supervisor loop only; never a profiler or worker callback.
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError('checkpoint worker deadline')
        if Path(path).is_file():
            return read_json(path)
        if child.poll() is not None:
            raise RuntimeError(f'checkpoint worker exited {child.returncode}; see worker.log')
        time.sleep(.05)


def run(directory, *, maximum_items=None, startup_seconds=120, ignore_time_budgets=False,
        report_seconds=None, _worker_command=None):
    directory = Path(directory).resolve()
    with single_writer(directory) as owner:
        started = time.monotonic_ns()
        settings, imported = load(directory)
        progress = Progress(directory, settings=settings, imported_ids=imported)
        progress.recover_pending()
        remaining = progress.missing(kinds=KINDS)
        order = {phase:i for i,phase in enumerate(settings['phase_order'])}
        remaining.sort(key=lambda w: order[w['phase']])
        if not ignore_time_budgets:
            remaining = [w for w in remaining if progress.remaining_ns(w['phase']) >= int(w['max_active_seconds']*NS)]
        if not remaining or maximum_items == 0:
            return status(progress)
        resource_observation(directory)
        session = directory/'sessions'/uuid.uuid4().hex
        session.mkdir(parents=True)
        (session/'requests').mkdir()
        atomic_json(session/'run.json', dict(owner=owner, started_ns=started,
            started_utc=datetime.now(timezone.utc).isoformat(), execution_sha256=settings['execution_sha256'],
            ignore_time_budgets=ignore_time_budgets, report_seconds=report_seconds, maximum_items=maximum_items,
            baseline_scientific=status(progress)['completed_scientific'], supported_kinds=sorted(KINDS),
            orchestration_sources={name:file_hash(Path(__file__).with_name(name)) for name in
                ('checkpoint_resume.py', 'checkpoint_state.py', 'checkpoint_inputs.py')}))
        ready = session/'ready.json'
        limit = int(startup_seconds*NS)
        if not ignore_time_budgets:
            limit = min(limit, progress.remaining_ns(INPUT))
        # Charge setup/file IO/native launch too, not just worker readiness.
        progress.reserve(phase=INPUT, maximum_ns=limit, ignore_time_budgets=ignore_time_budgets)
        from .seed_resume_session import OwnedJob
        job = OwnedJob()
        child, assigned = None, False
        accounted_until, last_phase = started, INPUT
        stop_path = directory/'stop.json'
        stop_requested = lambda: stop_path.is_file() and read_json(stop_path)['token'] == owner['token']
        next_report = started+int(report_seconds*NS) if report_seconds else None
        initial_science = status(progress)['completed_scientific']
        def report(*, force=False):
            nonlocal next_report
            now = time.monotonic_ns()
            if not force and (next_report is None or now < next_report):
                return
            current = status(progress)
            done = progress.imported_ids | progress.value['completed'].keys()
            failed = progress.value['failures'].keys()
            by_matrix = {}
            for matrix in ('NEX326-methods', 'terrain'):
                works = [w for w in progress.work.values() if w['kind'] == 'scientific_forecast' and w['matrix'] == matrix]
                left = sum(w['work_id'] not in done for w in works)
                failures = sum(w['work_id'] in failed for w in works)
                runnable = left-failures
                times = [r['elapsed_ns']/NS for r in progress.value.get('recent_work', [])
                    if r['matrix'] == matrix and r['kind'] == 'scientific_forecast' and r['success']]
                by_matrix[matrix] = dict(completed=len(works)-left, remaining=left,
                    failed=failures, runnable=runnable,
                    recent_samples=len(times), recent_mean_seconds=sum(times)/len(times) if times else None,
                    rough_remaining_seconds=0 if not runnable else runnable*sum(times)/len(times) if times else None)
            event = dict(event='progress_report', elapsed_seconds=(now-started)/NS,
                new_scientific_this_run=current['completed_scientific']-initial_science,
                ignore_time_budgets=ignore_time_budgets, by_matrix=by_matrix, **current)
            atomic_json(session/'progress-report.json', event)
            print(json.dumps(event), flush=True)
            resource_observation(directory)
            if report_seconds:
                next_report = now+int(report_seconds*NS)
        try:
            with (session/'worker.log').open('xb') as output:
                # Stdlib-only bootstrap waits before creating ANY descendant;
                # venv launcher and actual worker inherit the kill-on-close Job.
                bootstrap = ('import sys; sys.stdin.buffer.read(1)==b"R" or sys.exit(125); '
                    'import subprocess; sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL, '
                    'creationflags=subprocess.CREATE_NO_WINDOW))')
                command = (_worker_command(directory, session) if _worker_command else
                    [sys.executable, '-u', '-m', MODULE, 'worker', '--directory', str(directory), '--session', str(session)])
                child = subprocess.Popen([sys._base_executable, '-I', '-S', '-u', '-c', bootstrap, *command],
                    stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                job.assign(child)
                assigned = True
                child.stdin.write(b'R'); child.stdin.flush(); child.stdin.close()
                wait_reply(ready, child=child, deadline_ns=None if ignore_time_budgets else started+limit,
                           stop_requested=stop_requested, report=report)
                accounted_until = time.monotonic_ns()
                progress.settle(elapsed_ns=accounted_until-started)
                print(json.dumps(dict(event='checkpoint_ready', startup_seconds=(time.monotonic_ns()-started)/NS,
                    imported=len(imported), models=26)), flush=True)
                attempted, exhausted = 0, set()
                for work in remaining:
                    if maximum_items is not None and attempted >= maximum_items:
                        break
                    phase, wid = work['phase'], work['work_id']
                    if phase in exhausted:
                        continue
                    maximum = int(work['max_active_seconds']*NS)
                    if not ignore_time_budgets and progress.remaining_ns(phase) < maximum:
                        print(json.dumps(dict(event='phase_budget_exhausted', phase=phase)), flush=True)
                        exhausted.add(phase)
                        continue  # Other phases use their OWN budget, not this phase's.
                    began = accounted_until  # Include preceding checkpoint/print/controller IO.
                    last_phase = phase
                    output_directory = directory/'outputs'/wid
                    reply = output_directory/'reply.json'
                    progress.reserve(phase=phase, maximum_ns=maximum, work_id=wid, reply=str(reply),
                                     ignore_time_budgets=ignore_time_budgets)
                    attempted += 1
                    # Immutable one-way handoff: on Windows an open reader can
                    # deny replacement of a reused request.json. Never rewrite
                    # a request the worker may still have open.
                    atomic_json(session/'requests'/f'{attempted:06d}.json',
                                dict(work_id=wid, output_directory=str(output_directory)))
                    message = wait_reply(reply, child=child, deadline_ns=None if ignore_time_budgets else began+maximum,
                                         stop_requested=stop_requested, report=report)
                    if message['work_id'] != wid:
                        raise ValueError('worker reply work differs')
                    accounted_until = time.monotonic_ns()
                    elapsed = accounted_until-began
                    progress.settle(elapsed_ns=elapsed, result=message.get('result'), failure=message.get('error'))
                    if report_seconds is None:
                        print(json.dumps(dict(event='forecast_checkpoint', work_id=wid, elapsed_seconds=elapsed/NS,
                            status=message.get('result', {}).get('status', 'error'), **status(progress))), flush=True)
                    report()
                    if message.get('error'):
                        break
        finally:
            # No thread can prevent termination: this is the owning OS Job,
            # not the old watchdog/meter/credit lock chain.
            try:
                if assigned:
                    job.terminate()
                elif child is not None and child.poll() is None:
                    child.kill()  # Only our blocked, unassigned bootstrap.
                if child is not None:
                    child.wait(timeout=20)
                end = time.monotonic_ns()+20*NS
                while job.accounting()['active_processes']:
                    if time.monotonic_ns() >= end:
                        raise RuntimeError('owned process tree did not terminate')
                    time.sleep(.05)
                if progress.value['pending'] is not None:
                    # Actual stop cost, including timeout overrun, never reset.
                    began = locals().get('began', started)
                    result = progress.pending_result()
                    progress.settle(elapsed_ns=time.monotonic_ns()-began, result=result,
                                    failure='interrupted/deadline/error; output retained')
                else:
                    progress.charge(last_phase, time.monotonic_ns()-accounted_until)
                atomic_json(session/'closed.json', dict(accounting=job.accounting(), process_tree_closed=True,
                    orchestration_sources={name:file_hash(Path(__file__).with_name(name)) for name in
                        ('checkpoint_resume.py', 'checkpoint_state.py', 'checkpoint_inputs.py')}))
            finally:
                job.close()
        report(force=True)
        return status(progress)


def worker(directory, session):
    from .checkpoint_inputs import CheckpointForecasts
    settings, imported = load(directory)
    consumer = CheckpointForecasts(settings, imported)
    try:
        atomic_json(Path(session)/'ready.json', dict(models=len(consumer.models)))
        sequence = 1
        while True:
            request_path = Path(session)/'requests'/f'{sequence:06d}.json'
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
            time.sleep(.05)
    finally:
        consumer.close_all()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--bundle', type=Path, required=True)
    init.add_argument('--import-manifest', type=Path, required=True)
    init.add_argument('--stopped-startup-seconds', type=float, required=True)
    init.add_argument('--closure-note', required=True)
    for name in ('run', 'status', 'stop', 'worker'):
        sub.add_parser(name)
    for command in sub.choices.values():
        command.add_argument('--directory', type=Path, required=True)
    sub.choices['run'].add_argument('--maximum-items', type=int)
    sub.choices['run'].add_argument('--startup-seconds', type=float, default=120)
    sub.choices['run'].add_argument('--ignore-time-budgets', action='store_true',
        help='explicit user override: no phase/total/per-item time cutoff; retain measured costs and generation scope')
    sub.choices['run'].add_argument('--report-seconds', type=float,
        help='emit periodic progress and rolling phase estimates instead of one stdout line per item')
    sub.choices['worker'].add_argument('--session', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'init':
        result = initialize(args.directory, bundle_path=args.bundle, import_manifest=args.import_manifest,
            stopped_startup_ns=int(args.stopped_startup_seconds*NS), closure_note=args.closure_note)
    elif args.command == 'run':
        if (args.maximum_items is not None and args.maximum_items < 0 or args.startup_seconds <= 0
                or args.report_seconds is not None and args.report_seconds <= 0):
            parser.error('positive startup limit and nonnegative maximum-items required')
        result = run(args.directory, maximum_items=args.maximum_items, startup_seconds=args.startup_seconds,
                     ignore_time_budgets=args.ignore_time_budgets, report_seconds=args.report_seconds)
    elif args.command == 'worker':
        worker(args.directory, args.session)
        return 0
    elif args.command == 'stop':
        owner = read_json(args.directory/'writer.json')
        atomic_json(args.directory/'stop.json', dict(token=owner['token']))
        result = dict(stop_requested=True)
    else:
        settings, imported = load(args.directory)
        result = status(Progress(args.directory, settings=settings, imported_ids=imported))
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
