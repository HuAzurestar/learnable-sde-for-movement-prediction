"""Actual caller admission and original root across package members."""
import json
import os
from pathlib import Path

import application.research_comparison as comparison
from infrastructure.research_store import ResearchError, encode
from experiments.pirc25.__main__ import main
from tests.test_research_export_authorization import source
from tests.test_research_publication_paths import redirected_directory


def test_actual_export_cli_cannot_promote_initial_redirected_output_root(source, tmp_path, capsys):
    store, _, _, _ = source
    with redirected_directory(tmp_path) as (directory, outside, moved, redirect, installed):
        redirect()
        arguments = ['--root', str(store.path.parent), '--store-id', store.store_id,
            'export', 'synthetic', '--authorization-id', 'export', '--output', str(directory / 'bundle.json')]
        code = main(arguments)
        response = json.loads(capsys.readouterr().out)
        observed = {'arguments': arguments, 'exit_code': code, 'response': response,
            'installed': list(installed), 'outside_files': [path.name for path in outside.iterdir()]}
        (tmp_path / 'export-caller-root-observed.json').write_bytes(encode(observed))
        assert installed == [True] and code == 1 and response['error']['code'] == 'UNAUTHORIZED_DATA', observed
        assert observed['outside_files'] == [] and not list(moved.iterdir()), observed


def test_package_cannot_adopt_another_real_directory_between_members(tmp_path, monkeypatch):
    directory, moved = tmp_path / 'package', tmp_path / 'original-package'
    files = {'metrics.csv': b'original declared first member\n',
             'manifest.json': b'original declared second member\n'}
    original, replaced, observed = comparison.atomic_write, [], {'error': None}

    def publish(path, content, **options):
        result = original(path, content, **options)
        if not replaced:
            assert path == directory / 'metrics.csv' and path.read_bytes() == files['metrics.csv']
            assert Path(os.path.abspath(directory)).is_relative_to(tmp_path.resolve())
            assert moved.resolve().is_relative_to(tmp_path.resolve())
            directory.rename(moved)  # Actual move AFTER the first original atomic write has returned.
            directory.mkdir()  # Another real directory, not just a symlink.
            replaced.append(True)
        return result

    monkeypatch.setattr(comparison, 'atomic_write', publish)
    try:
        comparison.ComparisonRunner._write_files(directory, files)
    except (ResearchError, OSError) as error:
        observed['error'] = error.code if isinstance(error, ResearchError) else type(error).__name__
    observed.update(replaced=list(replaced), original_first_preserved=(moved / 'metrics.csv').read_bytes() == files['metrics.csv'],
                    replacement_files=[path.name for path in directory.iterdir()])
    (tmp_path / 'package-root-lifetime-observed.json').write_bytes(encode(observed))
    assert replaced == [True] and observed['error'] == 'UNAUTHORIZED_DATA', observed
    assert observed['original_first_preserved'] and observed['replacement_files'] == [], observed
