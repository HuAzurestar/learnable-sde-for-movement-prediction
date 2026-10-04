"""Actual publication root redirects and staging-name ownership, not outcomes."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess

import pytest

import application.research_comparison as comparison
from infrastructure.research_store import ResearchError, atomic_write, encode


@contextmanager
def redirected_directory(tmp_path, directory=None):
    directory = directory or tmp_path / 'original-package'
    directory.mkdir(parents=True, exist_ok=True)
    moved, outside = tmp_path / 'moved-original', tmp_path / 'outside'
    outside.mkdir()
    installed = []

    def redirect():
        assert not installed
        # Exact synthetic targets only; never move/delete a workspace root.
        assert Path(os.path.abspath(directory)).is_relative_to(tmp_path.resolve())
        assert moved.resolve().is_relative_to(tmp_path.resolve())
        assert outside.resolve().is_relative_to(tmp_path.resolve())
        directory.rename(moved)
        try:
            if os.name == 'nt':
                env = dict(os.environ, PIRC_TEST_JUNCTION_PATH=str(directory),
                           PIRC_TEST_JUNCTION_TARGET=str(outside))
                subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                    '$ErrorActionPreference = "Stop"; New-Item -ItemType Junction -Path '
                    '$env:PIRC_TEST_JUNCTION_PATH -Target $env:PIRC_TEST_JUNCTION_TARGET | Out-Null'],
                    env=env, check=True, capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                directory.symlink_to(outside, target_is_directory=True)
        except BaseException:
            moved.rename(directory)
            raise
        installed.append(True)

    try:
        yield directory, outside, moved, redirect, installed
    finally:
        if installed:
            assert Path(os.path.abspath(directory)).is_relative_to(tmp_path.resolve())
            assert moved.resolve().is_relative_to(tmp_path.resolve())
            if os.name == 'nt':
                directory.rmdir()  # Only the new junction entry, not its target.
            else:
                directory.unlink()  # Only the new symlink entry.
            moved.rename(directory)


@pytest.mark.parametrize('phase', ['initial', 'before-staging-open', 'after-fsync'])
def test_managed_publication_rejects_actual_directory_redirect(tmp_path, monkeypatch, phase):
    expected, observed = b'explicit synthetic managed bytes\n', {'error': None, 'phase': phase}
    with redirected_directory(tmp_path) as (directory, outside, moved, redirect, installed):
        original_open = Path.open
        if phase == 'initial':
            redirect()

        def opened(path, mode='r', *args, **kwargs):
            if (phase == 'before-staging-open' and path.parent == directory
                    and mode.startswith('x') and not installed):
                redirect()
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, 'open', opened)
        try:
            comparison.ComparisonRunner._write_files(directory, {'metrics.csv': expected},
                before_replace=redirect if phase == 'after-fsync' else None)
        except (ResearchError, OSError) as error:
            observed['error'] = error.code if isinstance(error, ResearchError) else type(error).__name__
        observed.update(installed=list(installed), outside_files=[path.name for path in outside.iterdir()],
            outside_target_exists=(outside / 'metrics.csv').exists(),
            moved_staging=[path.name for path in moved.glob('.*.staging')])
        (tmp_path / 'publication-root-observed.json').write_bytes(encode(observed))
        assert installed == [True], observed
        assert observed['error'] == 'UNAUTHORIZED_DATA', observed
        assert not observed['outside_files'] and not observed['moved_staging'], observed


@pytest.mark.parametrize('immutable', [False, True])
def test_actual_root_redirect_cannot_publish_or_delete_foreign_staging(tmp_path, immutable):
    expected, foreign = b'owned staged bytes\n', b'new writer owns reused staging name\n'
    observed = {'error': None, 'immutable': immutable}
    with redirected_directory(tmp_path) as (directory, outside, moved, redirect, installed):
        target = directory / 'export.bin'
        stages = []

        def guard():
            stage, = directory.glob('.*.staging')
            assert stage.read_bytes() == expected
            stages.append(stage.name)
            redirect()
            (outside / stage.name).write_bytes(foreign)

        try:
            atomic_write(target, expected, immutable=immutable, before_replace=guard)
        except (ResearchError, OSError) as error:
            observed['error'] = error.code if isinstance(error, ResearchError) else type(error).__name__
        foreign_path = outside / stages[0]
        observed.update(installed=list(installed), foreign_preserved=foreign_path.exists(),
            outside_target_exists=(outside / target.name).exists(),
            moved_staging=[path.name for path in moved.glob('.*.staging')])
        (tmp_path / 'staging-root-ownership-observed.json').write_bytes(encode(observed))
        assert observed['error'] == 'UNAUTHORIZED_DATA', observed
        assert foreign_path.read_bytes() == foreign and not observed['outside_target_exists'], observed
        assert not observed['moved_staging'], observed


@pytest.mark.parametrize('immutable', [False, True])
def test_actual_staging_replacement_never_becomes_the_declared_payload(tmp_path, immutable):
    target, expected = tmp_path / 'export.bin', b'original owned staging bytes\n'
    foreign = b'foreign writer replacement bytes\n'
    moved, stages, observed = tmp_path / 'moved-owned.bin', [], {'error': None, 'immutable': immutable}

    def guard():
        stage, = tmp_path.glob('.*.staging')
        assert stage.read_bytes() == expected
        stages.append(stage)
        stage.rename(moved)
        stage.write_bytes(foreign)

    try:
        atomic_write(target, expected, immutable=immutable, before_replace=guard)
    except (ResearchError, OSError) as error:
        observed['error'] = error.code if isinstance(error, ResearchError) else type(error).__name__
    stage = stages[0]
    observed.update(foreign_preserved=stage.exists(), target_exists=target.exists(),
                    target_matches=target.exists() and target.read_bytes() == expected)
    (tmp_path / 'staging-file-ownership-observed.json').write_bytes(encode(observed))
    assert stage.exists() and stage.read_bytes() == foreign, observed
    if observed['error'] is None:
        assert observed['target_matches'], observed
    else:
        assert observed['error'] in {'UNAUTHORIZED_DATA', 'CORRUPT_ARTIFACT', 'IDENTITY_CONFLICT'}, observed
        assert not target.exists(), observed


def test_normal_unicode_space_directory_keeps_publication_and_identical_retry(tmp_path):
    directory = tmp_path / '合法目录 with spaces'
    files = {'metrics.csv': b'explicit synthetic normal output\n'}
    comparison.ComparisonRunner._write_files(directory, files)
    comparison.ComparisonRunner._write_files(directory, files)
    assert (directory / 'metrics.csv').read_bytes() == files['metrics.csv']
    assert not list(directory.glob('.*.staging'))


def test_actual_managed_cli_artifact_root_redirect_cannot_write_external_package(tmp_path, monkeypatch, capsys):
    from application.research_budget import BudgetLedger
    from experiments.pirc25.__main__ import main
    from tests.test_research_comparison import comparison_source, compute
    source = comparison_source(tmp_path / 'fixture')
    store, _, _, paper, package = source
    computed = compute(source)
    (tmp_path / 'managed-root-initial-computation.json').write_bytes(encode(computed))
    assert computed['state'] == 'SUCCEEDED' and computed['exit_code'] == 0, computed
    before, attempts = BudgetLedger(store).balance('affine'), store.attempts()
    original = comparison.ComparisonRunner._write_files
    with redirected_directory(tmp_path, store.path / 'artifacts') as (directory, outside, moved, redirect, installed):
        def publish(path, *args, **kwargs):
            assert path.parent == directory
            if not installed:
                redirect()
            return original(path, *args, **kwargs)

        monkeypatch.setattr(comparison.ComparisonRunner, '_write_files', staticmethod(publish))
        arguments = ['--root', str(store.path.parent), '--store-id', store.store_id,
            'compare', 'synthetic', package['aggregate_hash'], '--paper-root', str(paper),
            '--authorization-id', 'viewer', '--seconds', '10']
        exit_code = main(arguments)
        response = json.loads(capsys.readouterr().out)
        observed = {'arguments': arguments, 'exit_code': exit_code, 'response': response,
            'installed': list(installed), 'outside_files': [path.relative_to(outside).as_posix()
                for path in outside.rglob('*') if path.is_file()],
            'budget_unchanged': BudgetLedger(store).balance('affine') == before,
            'attempts_unchanged': store.attempts() == attempts}
        (tmp_path / 'managed-cli-artifact-root-observed.json').write_bytes(encode(observed))
        assert installed == [True] and exit_code == 1 and response['error']['code'] == 'UNAUTHORIZED_DATA', observed
        assert observed['outside_files'] == [], observed
        assert observed['budget_unchanged'] and observed['attempts_unchanged'], observed
