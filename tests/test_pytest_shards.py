"""Complete inventory and real pytest failure/skip/selection audit controls."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.pytest_shards import SCHEMA, partition, verify


REVISION = 'a' * 40
VERSION = f'{sys.version_info.major}.{sys.version_info.minor}'


def reports():
    inventory = [f'tests/test_sample.py::test_case[{index}]' for index in range(19)]
    return [dict(schema_version=SCHEMA, shard_index=index, shard_count=4,
        python_version=VERSION, revision=REVISION, collected=inventory[:],
        selected=partition(inventory, index, 4), started=partition(inventory, index, 4),
        exit_code=0) for index in range(4)]


@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_complete_partition_is_stable_disjoint_and_preserves_original_order(count):
    inventory = ['z::test_last', 'a::test_first'] + [f'b::test_case[{i}]' for i in range(39)]
    parts = [partition(inventory, index, count) for index in range(count)]
    assert sorted(value for part in parts for value in part) == sorted(inventory)
    assert sum(map(len, parts)) == len(set(inventory))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1
    for index, part in enumerate(parts):
        assert [x for x in inventory if x in part] == part
        assert set(partition(list(reversed(inventory)), index, count)) == set(part)


@pytest.mark.parametrize('index,count,ids', [(-1, 4, []), (4, 4, []), (0, 0, []),
    (True, 4, []), (0, 33, []), (0, 1, ['duplicate', 'duplicate']), (0, 1, [''])])
def test_invalid_or_duplicate_collection_is_not_shardable(index, count, ids):
    with pytest.raises(ValueError):
        partition(ids, index, count)


def test_version_gate_requires_complete_exactly_once_execution():
    values = reports()
    values[0]['started'].reverse()  # Legal fixture reordering is not omission.
    result = verify(values, count=4, revision=REVISION, python_version=VERSION)
    assert result['collected'] == result['started'] == 19


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'revision', 'version', 'collection',
    'selection', 'unstarted', 'failed', 'cancelled', 'boolean_status', 'repeated_start', 'malformed'])
def test_incomplete_or_failed_shards_cannot_pass_version_gate(fault):
    values = deepcopy(reports())
    if fault == 'missing':
        values.pop()
    elif fault == 'duplicate':
        values[-1] = values[0]
    elif fault == 'revision':
        values[0]['revision'] = 'b' * 40
    elif fault == 'version':
        values[0]['python_version'] = 'other'
    elif fault == 'collection':
        values[0]['collected'].pop()
    elif fault == 'selection':
        values[0]['selected'].pop()
    elif fault == 'unstarted':
        values[0]['started'].pop()
    elif fault == 'failed':
        values[0]['exit_code'] = 1
    elif fault == 'cancelled':
        values[0]['exit_code'] = 2
    elif fault == 'boolean_status':
        values[0]['exit_code'] = False
    elif fault == 'repeated_start':
        values[0]['started'].append(values[0]['started'][0])
    elif fault == 'malformed':
        values[0] = []
    with pytest.raises(ValueError):
        verify(values, count=4, revision=REVISION, python_version=VERSION)


def real_shard(root, index, *, environment=None):
    repository = Path(__file__).resolve().parents[1]
    report = root / f'shard-{index}.json'
    command = ('import sys; from pathlib import Path; '
        f'sys.path.insert(0, {str(repository)!r}); '
        'from scripts.pytest_shards import run; '
        f'sys.exit(run(Path({str(root)!r}), {index}, 2, Path({str(report)!r}), revision={REVISION!r}))')
    completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', command],
        cwd=root, capture_output=True, text=True, timeout=60, env=environment)
    return completed, json.loads(report.read_text(encoding='utf-8')) if report.exists() else None


@pytest.mark.parametrize('failure', [False, True])
def test_real_pytest_runs_all_selected_items_and_preserves_failure_and_skip(tmp_path, failure):
    (tmp_path / 'test_toy.py').write_text(
        'import pytest\ndef test_ok(): pass\ndef test_bad():\n'
        + ('    assert False, "actual shard failure"\n' if failure else '    pass\n')
        + 'def test_skip(): pytest.skip("unavailable toy capability")\n', encoding='utf-8')
    values = []
    for index in range(2):
        completed, report = real_shard(tmp_path, index)
        assert report['exit_code'] == completed.returncode
        assert report['started'] == report['selected']
        assert (tmp_path / f'shard-{index}.xml').exists()
        values.append(report)
        if completed.returncode:
            assert failure and 'SHARD FAILURE:' in completed.stdout and 'actual shard failure' in completed.stdout
    if failure:
        with pytest.raises(ValueError, match='failed'):
            verify(values, count=2, revision=REVISION, python_version=VERSION)
    else:
        assert verify(values, count=2, revision=REVISION, python_version=VERSION)['started'] == 3


def test_real_external_selector_cannot_silently_remove_tests(tmp_path):
    (tmp_path / 'test_toy.py').write_text('def test_one(): pass\ndef test_two(): pass\n', encoding='utf-8')
    (tmp_path / 'conftest.py').write_text(
        'import pytest\n@pytest.hookimpl(trylast=True)\n'
        'def pytest_collection_modifyitems(items): items.clear()\n', encoding='utf-8')
    completed, report = real_shard(tmp_path, 0)
    assert completed.returncode != 0
    assert report is None or report['exit_code'] != 0
    assert 'another selector changed' in completed.stdout + completed.stderr
