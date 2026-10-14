"""Legacy shard runner/verifier retained for prior PIRC-28 CI receipts.

Current CI uses ci_pytest_partition.py with module-stable partitions and an
independently recollected full roster. Do not use this legacy verifier as the
current required gate. Its historical receipt checks remain regression tested.

Sharding changes placement only, not test bodies, marks, fixtures or deadlines.
The required version check verifies all collected IDs were started exactly once
and every shard succeeded. A missing/cancelled shard cannot produce a green gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SCHEMA = 'complete-pytest-shards-v1'


def partition(nodeids, index, count):
    if (type(index) is not int or type(count) is not int
            or not 0 <= index < count <= 32):
        raise ValueError('invalid shard index/count')
    if (type(nodeids) is not list or not all(type(x) is str and x for x in nodeids)
            or len(set(nodeids)) != len(nodeids)):
        raise ValueError('unique nonempty collected IDs required')
    chosen = set(sorted(nodeids)[index::count])
    return [nodeid for nodeid in nodeids if nodeid in chosen]


def run(root, index, count, report_path, *, revision):
    import pytest

    partition([], index, count)
    if os.environ.get('PYTEST_ADDOPTS', '').strip():
        raise ValueError('complete CI inventory does not accept PYTEST_ADDOPTS filters')
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=SCHEMA, shard_index=index, shard_count=count,
                  python_version=f'{sys.version_info.major}.{sys.version_info.minor}',
                  revision=revision, collected=[], selected=[], started=[], exit_code=None)

    class Selection:
        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self, session, config, items):
            result['collected'] = [item.nodeid for item in items]
            result['selected'] = partition(result['collected'], index, count)
            chosen = set(result['selected'])
            omitted = [item for item in items if item.nodeid not in chosen]
            items[:] = [item for item in items if item.nodeid in chosen]
            config.hook.pytest_deselected(items=omitted)

        def pytest_collection_finish(self, session):
            # Pytest may reorder parameterized fixtures. Placement may change,
            # but a second selector may neither remove nor duplicate an item.
            if Counter(item.nodeid for item in session.items) != Counter(result['selected']):
                raise pytest.UsageError('another selector changed the complete shard inventory')

        def pytest_runtest_logreport(self, report):
            if report.when == 'setup':
                result['started'].append(report.nodeid)
            if report.failed:
                # Preserve actionable failures even if a hosted job later stops.
                print('\nSHARD FAILURE: ' + report.nodeid + '\n' + str(report.longrepr), flush=True)

        def pytest_sessionfinish(self, session, exitstatus):
            result['exit_code'] = int(exitstatus)
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    original = Path.cwd()
    try:
        os.chdir(root)
        return int(pytest.main(['-q', '--tb=short', '--durations=20',
            '--junitxml=' + str(report_path.with_suffix('.xml'))], plugins=[Selection()]))
    finally:
        os.chdir(original)


def verify(reports, *, count, revision, python_version):
    if type(count) is not int or not 1 <= count <= 32 or len(reports) != count:
        raise ValueError('every requested shard report is required')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('literal checked-out revision required')
    indexed = {}
    for report in reports:
        if type(report) is not dict:
            raise ValueError('structured shard report required')
        index = report.get('shard_index')
        if (type(index) is not int or index in indexed or not 0 <= index < count
                or report.get('schema_version') != SCHEMA
                or type(report.get('shard_count')) is not int or report['shard_count'] != count
                or report.get('revision') != revision or report.get('python_version') != python_version
                or type(report.get('exit_code')) is not int or report['exit_code'] != 0):
            raise ValueError('missing, failed, duplicate or mismatched shard report')
        indexed[index] = report
    inventory = indexed[0]['collected']
    if not inventory:
        raise ValueError('nonempty complete collection required')
    started = []
    for index in range(count):
        report = indexed[index]
        expected = partition(inventory, index, count)
        if (report.get('collected') != inventory or report.get('selected') != expected
                or type(report.get('started')) is not list
                or Counter(report['started']) != Counter(expected)):
            raise ValueError('collection, selected or executed IDs differ from the complete partition')
        started.extend(report['started'])
    if Counter(started) != Counter(inventory):
        raise ValueError('complete test inventory was not executed exactly once')
    return {'collected': len(inventory), 'started': len(started), 'shards': count,
            'revision': revision, 'python_version': python_version}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--index', type=int, required=True)
    execute.add_argument('--count', type=int, required=True)
    execute.add_argument('--report', type=Path, required=True)
    audit = commands.add_parser('verify')
    audit.add_argument('--count', type=int, required=True)
    audit.add_argument('--revision', required=True)
    audit.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(root))
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                      capture_output=True, check=True).stdout.strip()
            return run(root, args.index, args.count, args.report, revision=revision)
        reports = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(args.reports.glob('*.json'))]
        print(json.dumps(verify(reports, count=args.count, revision=args.revision,
            python_version=f'{sys.version_info.major}.{sys.version_info.minor}')))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('Shard audit refused: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
