"""Actual immutable comparison publication and existing-target read boundaries."""
import os
import json

import pytest

import application.research_comparison as comparison
from infrastructure.research_store import ResearchError, _matches_file_content, encode
from tests.research_file_observation import observe_file


@pytest.mark.parametrize('existing', ['identical', 'different', 'oversized'])
def test_managed_package_existing_member_does_not_take_unbounded_read(tmp_path, monkeypatch, existing):
    directory = tmp_path / 'package'
    directory.mkdir()
    path, expected = directory / 'metrics.csv', b'frozen managed csv\n'
    old = expected if existing == 'identical' else (
        b'x' + expected[1:] if existing == 'different' else b'x' * (8 * 1024 * 1024))
    path.write_bytes(old)
    reads, handles = observe_file(monkeypatch, path)
    observed = {'existing': existing, 'error': None}
    try:
        comparison.ComparisonRunner._write_files(directory, {'metrics.csv': expected})
    except ResearchError as error:
        observed['error'] = error.code
    observed.update(reads=list(reads), handles=list(handles), remaining_size=path.stat().st_size)
    (tmp_path / 'managed-existing-observed.json').write_bytes(encode(observed))
    assert observed['error'] == (None if existing == 'identical' else 'IDENTITY_CONFLICT'), observed
    if existing == 'oversized':
        assert reads == [], observed
    else:
        assert reads, observed
    assert all(0 <= row['size'] <= 64 * 1024 for row in reads), observed
    assert path.read_bytes() == old
    assert not list(directory.glob('.*.staging'))


def test_managed_package_does_not_overwrite_real_competitor_after_absence_check(tmp_path, monkeypatch):
    directory, expected = tmp_path / 'package', b'frozen managed csv\n'
    path, competitor = directory / 'metrics.csv', b'actual competing publisher bytes\n'
    original, touched = comparison.atomic_write, []
    def publish(target, content, *args, **kwargs):
        assert target == path and not target.exists()
        target.write_bytes(competitor)
        touched.append(True)
        return original(target, content, *args, **kwargs)
    monkeypatch.setattr(comparison, 'atomic_write', publish)
    observed = {'error': None}
    try:
        comparison.ComparisonRunner._write_files(directory, {'metrics.csv': expected})
    except ResearchError as error:
        observed['error'] = error.code
    observed.update(touched=list(touched), competitor_preserved=path.read_bytes() == competitor)
    (tmp_path / 'managed-race-observed.json').write_bytes(encode(observed))
    assert touched == [True] and observed['error'] == 'IDENTITY_CONFLICT', observed
    assert observed['competitor_preserved'], observed
    assert not list(directory.glob('.*.staging'))


def test_real_existing_target_write_with_restored_mtime_cannot_match_after_read(tmp_path, monkeypatch):
    path, expected = tmp_path / 'existing.bin', b'original expected bytes\n'
    changed = b'X' + expected[1:]
    path.write_bytes(expected)
    before, touched = path.stat(), []
    def change(*_):
        if not touched:
            touched.append(True)
            path.write_bytes(changed)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    reads, handles = observe_file(monkeypatch, path, after_read=change)
    matched = _matches_file_content(path, expected)
    observed = {'matched': matched, 'touched': list(touched), 'reads': list(reads), 'handles': list(handles)}
    (tmp_path / 'existing-target-identity-observed.json').write_bytes(encode(observed))
    assert touched == [True] and path.stat().st_ino == before.st_ino
    assert path.stat().st_size == len(expected) and path.stat().st_mtime_ns == before.st_mtime_ns
    assert not matched, observed


def test_legal_existing_target_comparison_streams_without_whole_source_allocation(tmp_path, monkeypatch):
    path = tmp_path / 'existing.bin'
    expected = b'legitimate expected bytes\n' * 9000
    path.write_bytes(expected)
    reads, handles = observe_file(monkeypatch, path)
    assert _matches_file_content(path, expected)
    assert sum(row['bytes'] for row in reads) == len(expected) and handles == [True]
    assert all(0 < row['size'] <= 64 * 1024 for row in reads), reads


def test_new_managed_package_and_identical_retry_remain_valid(tmp_path):
    directory = tmp_path / 'package'
    files = {'metrics.csv': b'frozen managed csv\n', 'manifest.json': encode({'synthetic': True})}
    comparison.ComparisonRunner._write_files(directory, files)
    comparison.ComparisonRunner._write_files(directory, files)
    assert {name: (directory / name).read_bytes() for name in files} == files
    assert not list(directory.glob('.*.staging'))


@pytest.mark.parametrize('case', ['competing-publication', 'oversized-existing'])
def test_actual_managed_cli_refuses_conflict_without_extra_job_or_unbounded_read(
        tmp_path, monkeypatch, capsys, case):
    from application.research_budget import BudgetLedger
    from experiments.pirc25.__main__ import main
    from tests.test_research_comparison import comparison_source, compute
    source = comparison_source(tmp_path / 'fixture')
    store, _, _, paper, package = source
    computed = compute(source)
    assert computed['state'] == 'SUCCEEDED' and computed['exit_code'] == 0
    before, attempts = BudgetLedger(store).balance('affine'), store.attempts()
    directory = tmp_path / 'output'
    directory.mkdir()
    target, old = directory / 'metrics.csv', b'actual competing publisher bytes\n'
    if case == 'oversized-existing':
        old = b'x' * (8 * 1024 * 1024)
        target.write_bytes(old)
    original, touched = comparison.atomic_write, []
    def publish(path, content, *args, **kwargs):
        if path == target and case == 'competing-publication':
            assert not target.exists()
            target.write_bytes(old)
            touched.append(True)
        return original(path, content, *args, **kwargs)
    monkeypatch.setattr(comparison, 'atomic_write', publish)
    reads, handles = observe_file(monkeypatch, target)
    arguments = ['--root', str(store.path.parent), '--store-id', store.store_id,
        'compare', 'synthetic', package['aggregate_hash'], '--paper-root', str(paper),
        '--authorization-id', 'viewer', '--seconds', '10', '--output', str(directory)]
    exit_code = main(arguments)
    response = json.loads(capsys.readouterr().out)
    observed = {'arguments': arguments, 'exit_code': exit_code, 'response': response,
        'reads': list(reads), 'handles': list(handles), 'touched': list(touched)}
    (tmp_path / 'managed-cli-conflict-observed.json').write_bytes(encode(observed))
    assert exit_code == 1 and response['error']['code'] == 'IDENTITY_CONFLICT', observed
    assert reads == [], 'different frozen size must reject before existing payload bytes'
    assert target.read_bytes() == old and not list(directory.glob('.*.staging'))
    assert touched == ([True] if case == 'competing-publication' else [])
    assert BudgetLedger(store).balance('affine') == before and store.attempts() == attempts
