"""Real incoming-file change identity and root admission through public import."""
import json
import os
from pathlib import Path
import shutil

import pytest

import application.research_evidence as evidence
from application.research_budget import BudgetSpec
from experiments.pirc25.__main__ import main
from infrastructure.research_store import ResearchError, encode
from tests.research_file_observation import is_query_only_open
from tests.test_research_comparison import comparison_source, compute
from tests.test_research_package_read_bounds import FILES, observe_target
from tests.test_research_publication_paths import redirected_directory


@pytest.fixture(scope='module')
def package_source(tmp_path_factory):
    root = tmp_path_factory.mktemp('source-identity-managed-package')
    source = comparison_source(root)
    # Keep the existing package-read fixture's original20s supervised budget.
    result = compute(source, budget=BudgetSpec(20), output=root / 'managed')
    (root / 'initial-computation-observed.json').write_bytes(encode(result))
    assert result['state'] == 'SUCCEEDED' and result['exit_code'] == 0, result
    return source[0], root / 'managed', result['comparison']['aggregate_hash']


def copy_package(package_source, tmp_path, name):
    store, original, aggregate_hash = package_source
    root = tmp_path / 'incoming'
    shutil.copytree(original, root)
    filename = next(root.glob('*.svg')).name if name == 'figure' else name
    return store, root, aggregate_hash, root / filename


@pytest.mark.parametrize('name,limit', FILES)
@pytest.mark.parametrize('phase', ['before-open', 'after-read'])
def test_real_restored_mtime_package_source_is_rejected_before_return(tmp_path, monkeypatch, name, limit, phase):
    target = tmp_path / ('figure.svg' if name == 'figure' else name)
    expected = b'explicit synthetic source bytes\n'
    target.write_bytes(expected)
    before, touched = target.stat(), []

    def change():
        assert not touched
        target.write_bytes(expected if phase == 'before-open' else b'!' + expected[1:])
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        touched.append(True)

    reads = observe_target(monkeypatch, target, after_read=change if phase == 'after-read' else None)
    if phase == 'before-open':
        original = os.open

        def opened(path, *args, **kwargs):
            if Path(path) == target and not is_query_only_open(args, kwargs):
                change()
            return original(path, *args, **kwargs)

        monkeypatch.setattr(os, 'open', opened)
    observed = {'phase': phase, 'name': name, 'error': None}
    try:
        observed['returned_original'] = evidence._package_file_bytes(tmp_path, target.name, limit) == expected
    except ResearchError as error:
        observed['error'] = error.code
    current = target.stat()
    observed.update(touched=list(touched), reads=list(reads), same_inode=current.st_ino == before.st_ino,
                    same_size=current.st_size == before.st_size, same_mtime=current.st_mtime_ns == before.st_mtime_ns)
    (tmp_path / 'source-change-observed.json').write_bytes(encode(observed))
    assert touched == [True] and observed['same_inode'] and observed['same_size'] and observed['same_mtime'], observed
    assert observed['error'] == 'CORRUPT_ARTIFACT', observed
    assert reads == ([] if phase == 'before-open' else [len(expected) + 1]), observed


@pytest.mark.parametrize('name,limit', FILES)
def test_actual_import_rejects_source_modified_after_read_with_restored_mtime(
        package_source, tmp_path, monkeypatch, name, limit):
    store, root, aggregate_hash, target = copy_package(package_source, tmp_path, name)
    original, before, touched = target.read_bytes(), target.stat(), []
    original_open = Path.open

    def change():
        assert not touched
        target.write_bytes(b'!' + original[1:])
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        touched.append(True)

    reads = observe_target(monkeypatch, target, after_read=change)
    before_events, observed = store.events(), {'name': name, 'error': None}
    try:
        observed['imported'] = evidence.accept_evidence_package(store, root, aggregate_hash)
    except ResearchError as error:
        observed['error'] = error.code
    # Verification must not re-enter the production-reader observer and fire
    # the mutation twice. Read the ACTUAL source through the original open.
    with original_open(target, 'rb') as actual:
        source_changed = actual.read() != original
    observed.update(touched=list(touched), reads=list(reads), events_unchanged=store.events() == before_events,
                    actual_source_changed=source_changed)
    (tmp_path / 'public-import-source-observed.json').write_bytes(encode(observed))
    assert touched == [True] and observed['actual_source_changed'] and reads, observed
    assert observed['error'] == 'CORRUPT_ARTIFACT' and observed['events_unchanged'], observed


def test_actual_import_cli_cannot_promote_initial_redirected_package_root(package_source, tmp_path, capsys):
    store, original, aggregate_hash = package_source
    with redirected_directory(tmp_path) as (root, outside, moved, redirect, installed):
        shutil.copytree(original, outside, dirs_exist_ok=True)
        redirect()
        before = store.events()
        arguments = ['--root', str(store.path.parent), '--store-id', store.store_id,
                     'import-evidence', str(root), '--expected-hash', aggregate_hash]
        code = main(arguments)
        response = json.loads(capsys.readouterr().out)
        observed = {'arguments': arguments, 'exit_code': code, 'response': response,
                    'installed': list(installed), 'events_unchanged': store.events() == before}
        (tmp_path / 'import-cli-root-observed.json').write_bytes(encode(observed))
        assert installed == [True] and code == 1 and response['error']['code'] == 'UNAUTHORIZED_DATA', observed
        assert observed['events_unchanged'], observed


def test_actual_import_cannot_adopt_another_real_directory_between_members(package_source, tmp_path, monkeypatch):
    store, root, aggregate_hash, _ = copy_package(package_source, tmp_path, 'manifest.json')
    moved, original_read, replaced = tmp_path / 'original-incoming', evidence._package_file_bytes, []

    def read(directory, name, limit, **options):
        content = original_read(directory, name, limit, **options)
        if not replaced:
            assert name == 'manifest.json'
            assert root.resolve().is_relative_to(tmp_path.resolve()) and moved.resolve().is_relative_to(tmp_path.resolve())
            root.rename(moved)
            shutil.copytree(moved, root)  # Actual distinct directory; all member bytes are identical.
            assert root.stat().st_ino != moved.stat().st_ino
            replaced.append(True)
        return content

    monkeypatch.setattr(evidence, '_package_file_bytes', read)
    before, observed = store.events(), {'error': None}
    try:
        observed['imported'] = evidence.accept_evidence_package(store, root, aggregate_hash)
    except ResearchError as error:
        observed['error'] = error.code
    observed.update(replaced=list(replaced), events_unchanged=store.events() == before)
    (tmp_path / 'import-root-lifetime-observed.json').write_bytes(encode(observed))
    assert replaced == [True] and observed['error'] == 'UNAUTHORIZED_DATA' and observed['events_unchanged'], observed


def test_unchanged_public_import_keeps_original_package_and_attempts(package_source, tmp_path):
    store, root, aggregate_hash, _ = copy_package(package_source, tmp_path, 'manifest.json')
    attempts, original = store.attempts(), {path.name: path.read_bytes() for path in root.iterdir()}
    package = evidence.accept_evidence_package(store, root, aggregate_hash)
    assert evidence.accept_evidence_package(store, root, aggregate_hash) == package
    assert store.attempts() == attempts
    assert {path.name: path.read_bytes() for path in root.iterdir()} == original
