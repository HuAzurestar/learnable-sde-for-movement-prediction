"""One bounded reconstruction using the surviving PIRC-21 producer."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import traceback

from .seed_resume_session import require_owned_job, run_owned

DATA = Path(os.environ.get('PIRC17_DATA_REPOSITORY', '../DSDE-SDE'))
PRODUCER = Path(os.environ.get('PIRC17_FEATURE_PRODUCER', 'private/pirc21-producer'))
ROOT = Path(os.environ.get('PIRC17_RECOVERY_ROOT', 'private/pirc17-recovery/pirc17-20261001-original-v1'))
FIRST_ROOT = ROOT
INDEXED_ROOT = ROOT.with_name('pirc17-20261001-file-index-v2')
RESUME_ROOT = ROOT.with_name('pirc17-20261001-partial-resume-v3')
INDEXED = False
RESUME = False
GLOBAL_DEADLINE = None
SNAPSHOT_ID = 'pirc21-production-features-pirc18-20260921-v1'
RELEASE = DATA / 'private/pirc20/releases/pirc20-r1t-nex326-midpoint-20260912-v1'
SIDECAR = DATA / 'private/pirc21/sidecars/pirc21-production-pirc18-composed-20260920-v1/manifest.json'
SPEC = PRODUCER / 'registry/pirc21_feature_spec.contract.json'
TIME_LIMIT = 1800
EXPECTED = {
    'producer': '7cc7649f4ee495ad193729a2fde4d9fa5e81505b5e1ab2d102641fc79c11b751',
    'sidecar': '77ae28da9bf5714ab2e48f810d40f3ee81b8f9f9dc0db5ac0493ab9ccef14d31',
    'inventory': '31a41177fb2521a434c64c60dc670652c07442fb576623d76bf3d1f30376bb0f',
    'manifest': '927f1d33df2e803d6f3aae403ea5e7046c888d792d1eedf2d2be23a015a08956',
    'coverage': '32f65eea653f730ca7a5664088657199ee1c1c76d70d85947517b9b63d930d60',
}


class FileIndexConnection:
    """Change only a temporary access index, never the queries or returned rows."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query, *args, **kwargs):
        if (isinstance(query, str) and query.startswith('CREATE INDEX sidecar_')
                and query.endswith('(file_id, point_index)')):
            query = query.removesuffix('(file_id, point_index)') + '(file_id)'
        return self.connection.execute(query, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.connection, name)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish(name, value):
    with (ROOT / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_recipe():
    for path, expected in ((PRODUCER / 'trajectory/feature_materialization.py', EXPECTED['producer']),
                           (SIDECAR, EXPECTED['sidecar'])):
        if sha(path) != expected:
            raise ValueError(f'recovery recipe changed: {path.name}')
    return {name: sha(PRODUCER / 'trajectory' / name)
            for name in ('feature_materialization.py', 'feature_contract.py', 'feature_geometry.py')}


def closed_cache_directories(producer):
    directories = []
    sources = check_recipe()
    spec_sha = sha(SPEC)
    for root in (FIRST_ROOT, INDEXED_ROOT):
        start = json.loads((root / 'start.json').read_text(encoding='utf-8'))
        result = json.loads((root / 'process-result.json').read_text(encoding='utf-8'))
        if (not result['process_tree_closed'] or result['accounting']['active_processes'] != 0
                or start['sources'] != sources or start['expected'] != EXPECTED
                or start['feature_spec_source_sha256'] != spec_sha):
            raise ValueError('partial recovery cache is live or belongs to a different input/producer recipe')
        stages = list((root / 'publication/snapshots').glob('.staging-' + SNAPSHOT_ID + '-*'))
        if len(stages) != 1 or not stages[0].is_dir() or stages[0].is_symlink():
            raise ValueError('closed recovery staging locator is missing or ambiguous')
        if (json.loads((stages[0] / 'feature_spec.json').read_text(encoding='utf-8'))
                != producer.load_feature_spec(SPEC)):
            raise ValueError('partial recovery feature specification differs')
        directories.append(stages[0])
    return directories


def recovery_job_name():
    return ('Local\\PIRC17-snapshot-recovery-20261001-partial-resume-v3' if RESUME
            else 'Local\\PIRC17-snapshot-recovery-20261001-file-index-v2' if INDEXED
            else 'Local\\PIRC17-snapshot-recovery-20261001-v1')


def worker():
    require_owned_job(recovery_job_name())
    check_recipe()
    sys.path.insert(0, str(PRODUCER))
    sys.argv = ['feature_materialization', '--pirc20-release', str(RELEASE),
                '--condition-root', str(DATA / 'cond_slices'), '--feature-spec', str(SPEC),
                '--output-root', str(ROOT / 'publication'), '--snapshot-id', SNAPSHOT_ID,
                '--history-window-points', '3', '--sidecar-manifest', str(SIDECAR)]
    with (ROOT / 'worker.log').open('x', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                if INDEXED:
                    import duckdb
                    producer = importlib.import_module('trajectory.feature_materialization')
                    original_connect = duckdb.connect
                    duckdb.connect = lambda *args, **kwargs: FileIndexConnection(original_connect(*args, **kwargs))
                    try:
                        if RESUME:
                            from .snapshot_partial_reuse import PartialReuse
                            reuse = PartialReuse(producer, closed_cache_directories(producer), sha)
                            try:
                                with reuse.installed():
                                    producer.main()
                            finally:
                                publish('reuse-result.json', reuse.stats)
                        else:
                            producer.main()
                    finally:
                        duckdb.connect = original_connect
                else:
                    runpy.run_module('trajectory.feature_materialization', run_name='__main__')
            except BaseException:
                traceback.print_exc()
                raise


def verify():
    import pyarrow.parquet as pq
    begin = time.perf_counter()
    directory = ROOT / 'publication/snapshots' / SNAPSHOT_ID
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    inventory = hashlib.sha256()
    rows = size = 0
    seen = set()
    for record in sorted(manifest['files'], key=lambda r: r['path']):
        if time.perf_counter() - begin >= 300:
            raise TimeoutError('verification reached its separate 300-second limit')
        path = (directory / record['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or path in seen:
            raise ValueError('unsafe or duplicate feature path')
        seen.add(path)
        actual = sha(path)
        count = pq.ParquetFile(path).metadata.num_rows
        if actual != record['sha256'] or count != record['row_count']:
            raise ValueError('feature bytes or row count differ from manifest')
        rows += count
        size += path.stat().st_size
        inventory.update(f"{record['path']}\0{actual}\0{count}\n".encode('utf-8'))
    identities = {'inventory': inventory.hexdigest(), 'manifest': sha(manifest_path),
                  'coverage': sha(directory / 'coverage_report.json')}
    matches = {key: value == EXPECTED[key] for key, value in identities.items()}
    report = {'snapshot_directory': str(directory), 'files': len(seen), 'rows': rows,
              'feature_bytes': size, 'identities': identities, 'matches': matches,
              'manifest_inventory_matches_files': inventory.hexdigest() == manifest['content_inventory_sha256'],
              'elapsed_seconds': time.perf_counter() - begin}
    report['exact_restoration'] = (all(matches.values()) and len(seen) == 7618 and rows == 14738300
                                   and report['manifest_inventory_matches_files'] and manifest['status'] == 'valid')
    publish('verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report['exact_restoration']


def supervise():
    sources = check_recipe()
    if INDEXED:
        first_result = json.loads((FIRST_ROOT / 'process-result.json').read_text(encoding='utf-8'))
        if not first_result['process_tree_closed'] or first_result['worker_returncode'] != 99:
            raise ValueError('indexed continuation requires the deliberately stopped original owned tree')
    if shutil.disk_usage(ROOT.parent.parent).free < 40 * 1024**3:
        raise OSError('recovery requires at least 40 GiB free on its output volume')
    ROOT.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', '-m', 'experiments.pirc17.snapshot_recovery',
               'worker-resume' if RESUME else 'worker-indexed' if INDEXED else 'worker']
    publish('start.json', {'time_limit_seconds': TIME_LIMIT, 'sources': sources,
                          'feature_spec_source_sha256': sha(SPEC), 'command': command,
                          'expected': EXPECTED, 'scientific_fit_forecast_score_calls': 0,
                          'purpose': 'user-authorized mechanical feature reconstruction across original splits',
                          'wrapper_sha256': sha(__file__), 'temporary_file_only_index': INDEXED,
                          'first_attempt_root': str(FIRST_ROOT), 'shared_deadline_epoch': GLOBAL_DEADLINE,
                          'reuse_closed_partial_files': RESUME,
                          'reuse_module_sha256': sha(Path(__file__).with_name('snapshot_partial_reuse.py')) if RESUME else None,
                          'target_feature_files': 7618, 'target_feature_rows': 14738300,
                          'first_process_receipt_sha256': sha(FIRST_ROOT / 'process-result.json') if INDEXED else None})
    next_update = 0.0

    def monitor():
        nonlocal next_update
        now = time.monotonic()
        if now >= next_update:
            outputs = list((ROOT / 'publication/snapshots').glob('*/features/*/*.parquet'))
            sizes = []
            for path in outputs:
                try:
                    sizes.append(path.stat().st_size)
                except FileNotFoundError:
                    pass  # The producer may promote or clean its own staging tree.
            print(json.dumps({'recovery_files_written': len(outputs),
                              'feature_bytes_written': sum(sizes)}), flush=True)
            next_update = now + 30
        return shutil.disk_usage(ROOT).free < 10 * 1024**3

    result = run_owned(command, timeout=TIME_LIMIT, stop_requested=monitor, job_name=recovery_job_name())
    publish('process-result.json', result)
    print(json.dumps(result), flush=True)
    if result['worker_returncode'] != 0 or not result['process_tree_closed']:
        return 1
    return 0 if verify() else 2


def main():
    global ROOT, TIME_LIMIT, INDEXED, RESUME, GLOBAL_DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'worker', 'verify', 'run-indexed', 'worker-indexed', 'verify-indexed',
                                         'run-resume', 'worker-resume', 'verify-resume'))
    action = parser.parse_args().action
    if action.endswith('-resume'):
        INDEXED = RESUME = True
        ROOT = RESUME_ROOT
        TIME_LIMIT = 7200  # One fixed remaining-work phase, not a repeated cost-window attempt.
        action = action.removesuffix('-resume')
    elif action.endswith('-indexed'):
        INDEXED = True
        ROOT = INDEXED_ROOT
        GLOBAL_DEADLINE = (FIRST_ROOT / 'start.json').stat().st_ctime + 1800
        TIME_LIMIT = int(GLOBAL_DEADLINE - time.time())
        if action != 'verify-indexed' and TIME_LIMIT <= 0:
            raise TimeoutError('the original shared recovery deadline has expired; no budget extension')
        action = action.removesuffix('-indexed')
    if action == 'worker':
        worker()
        return 0
    return supervise() if action == 'run' else (0 if verify() else 2)


if __name__ == '__main__':
    raise SystemExit(main())
