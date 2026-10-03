"""Actual CLI publication must not overwrite a racing immutable export."""

import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence
from experiments.pirc25.__main__ import main
import infrastructure.research_store as store_module
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_research_export_authorization import source, foreign_source, permission_clock


def arguments(store, output):
    return ['--root', str(store.path.parent), '--store-id', store.store_id,
            'export', 'synthetic', '--authorization-id', 'export', '--output', str(output)]


@pytest.mark.parametrize('existing', ['identical', 'different', 'oversized'])
def test_actual_cli_existing_target_comparison_uses_bounded_open_handle_reads(
        source, tmp_path, monkeypatch, capsys, existing):
    store, value, grant, _ = source
    expected = encode(export_evidence(store, value['study_id'], grant))
    old = expected if existing == 'identical' else (b'x' + expected[1:] if existing == 'different'
                                                  else expected + b'extra bytes')
    output = tmp_path / 'existing.json'
    output.write_bytes(old)
    original_open = Path.open
    original_fdopen = store_module.os.fdopen
    reads = []

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            reads.append(size)
            return self.stream.read(size)

    def opened(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = kwargs.get('mode', args[0] if args else 'r')
        return Reader(stream) if path == output and mode == 'rb' else stream

    monkeypatch.setattr(Path, 'open', opened)
    def fdopened(fd, *args, **kwargs):
        # Production's opened-handle comparator must actually exercise the
        # bound; retain the original Path.open hook for the failing baseline.
        stream = original_fdopen(fd, *args, **kwargs)
        return Reader(stream)
    monkeypatch.setattr(store_module.os, 'fdopen', fdopened)
    status = main(arguments(store, output))
    response = json.loads(capsys.readouterr().out)
    assert status == (0 if existing == 'identical' else 1)
    if existing != 'identical':
        assert response['error']['code'] == 'IDENTITY_CONFLICT'
    with original_open(output, 'rb') as stream:
        assert stream.read(len(old) + 1) == old
    assert all(0 <= size <= len(expected) + 1 for size in reads), reads
    if existing != 'oversized':
        assert reads, 'target equality must compare actual bytes, not only size'
    assert not list(output.parent.glob('.' + output.name + '.*.staging'))


@pytest.mark.parametrize('existing', [False, True])
def test_actual_cli_conflict_created_after_target_fsync_is_not_overwritten(
        source, tmp_path, monkeypatch, capsys, existing):
    store, value, grant, _ = source
    expected = encode(export_evidence(store, value['study_id'], grant))
    output = tmp_path / 'racing.json'
    if existing:
        output.write_bytes(expected)
    conflicting = b'unrelated immutable target created during export'
    original_write, original_fsync = store_module.atomic_write, store_module.os.fsync
    touched = []

    def write(path, *args, **kwargs):
        if path != output:
            return original_write(path, *args, **kwargs)
        with monkeypatch.context() as context:
            def fsync(fd):
                original_fsync(fd)
                if not touched:
                    output.write_bytes(conflicting)
                    touched.append(True)
            context.setattr(store_module.os, 'fsync', fsync)
            return original_write(path, *args, **kwargs)

    monkeypatch.setattr(store_module, 'atomic_write', write)
    status = main(arguments(store, output))
    response = json.loads(capsys.readouterr().out)
    assert touched
    assert status == 1, response
    assert response['error']['code'] == 'IDENTITY_CONFLICT'
    assert output.read_bytes() == conflicting
    assert not list(output.parent.glob('.' + output.name + '.*.staging'))


def test_two_actual_cli_publishers_keep_the_completed_export_and_refuse_stale_writer(
        source, tmp_path, monkeypatch, capsys):
    store, value, grant, artifact = source
    original_bundle = export_evidence(store, value['study_id'], grant)
    original_bytes = encode(original_bundle)
    output = tmp_path / 'shared.json'
    original_write, original_fsync = store_module.atomic_write, store_module.os.fsync
    armed = set()
    ready, completed = threading.Event(), threading.Event()
    outcomes, errors = [], []

    def fsync(fd):
        original_fsync(fd)
        if threading.get_ident() in armed and not ready.is_set():
            ready.set()
            assert completed.wait(10), 'second actual publisher did not finish'

    def write(path, content, **kwargs):
        thread = threading.get_ident()
        if path == output and content == original_bytes:
            armed.add(thread)
            try:
                return original_write(path, content, **kwargs)
            finally:
                armed.remove(thread)
        return original_write(path, content, **kwargs)

    def first():
        try:
            outcomes.append(main(arguments(store, output)))
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(store_module, 'atomic_write', write)
    monkeypatch.setattr(store_module.os, 'fsync', fsync)
    worker = threading.Thread(target=first)
    worker.start()
    try:
        assert ready.wait(10)
        # The second actual publisher sees a new successful cell. The first
        # retains a valid authorized historical snapshot, not the same bytes.
        cell = value['cells'][1]
        run = store.register_run(value['study_id'], cell)
        attempt = store.new_attempt(run)
        reservation = BudgetLedger(store).reserve(attempt, BudgetSpec(1))
        store.transition(attempt, 'RUNNING')
        result = json.loads(store._verified_artifact_content(artifact))
        result.update(cell_hash=digest(cell), metrics={'error': 2})
        second_artifact = store.artifact(encode(result), role='result', visibility='synthetic',
                                        block_ids=[cell['block_id']], study_id=value['study_id'])
        BudgetLedger(store).settle(reservation['reservation_id'], 200, outcome='SUCCEEDED')
        store.transition(attempt, 'SUCCEEDED', artifact_id=second_artifact['artifact_id'])
        assert main(arguments(store, output)) == 0
        published = json.loads(output.read_bytes())
        assert [row['status'] for row in published['cells']] == ['SUCCEEDED', 'SUCCEEDED']
        assert published['bundle_hash'] != original_bundle['bundle_hash']
    finally:
        completed.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
    responses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert not errors
    assert outcomes == [1], responses
    assert any(response.get('error', {}).get('code') == 'IDENTITY_CONFLICT' for response in responses)
    assert json.loads(output.read_bytes()) == published
    assert not list(output.parent.glob('.' + output.name + '.*.staging'))


def test_actual_cli_new_and_repeated_export_remain_identical(source, tmp_path, capsys):
    store, value, grant, _ = source
    output = tmp_path / 'valid.json'
    expected = encode(export_evidence(store, value['study_id'], grant))
    assert main(arguments(store, output)) == 0
    assert output.read_bytes() == expected
    assert main(arguments(store, output)) == 0
    assert output.read_bytes() == expected
    assert all(json.loads(line)['bundle_hash'] == json.loads(expected)['bundle_hash']
               for line in capsys.readouterr().out.splitlines())
    assert not list(output.parent.glob('.' + output.name + '.*.staging'))


@pytest.mark.parametrize('foreign', [False, True])
def test_actual_cli_identical_target_read_cannot_outlive_export_permission(
        request, tmp_path, monkeypatch, capsys, permission_clock, foreign):
    fixture = request.getfixturevalue('foreign_source' if foreign else 'source')
    store, value, grant = fixture[:3]
    expected = encode(export_evidence(store, value['study_id'], grant))
    output = tmp_path / 'identical-expiry.json'
    output.write_bytes(expected)
    original = store_module._matches_file_content
    touched = []

    def compare(path, content):
        result = original(path, content)
        assert path == output and result
        touched.append(True)
        permission_clock[0] = True
        return result

    monkeypatch.setattr(store_module, '_matches_file_content', compare)
    args = arguments(store, output)
    args[args.index('--authorization-id') + 1] = grant['authorization_id']
    assert main(args) == 1
    response = json.loads(capsys.readouterr().out)
    assert response['error']['code'] == 'UNAUTHORIZED_DATA'
    assert touched and output.read_bytes() == expected
    denial = store.events()[-1]
    assert denial['event_kind'] == 'DISCLOSURE_DENIED'
    assert denial['payload']['study_id'] == ('model-study' if foreign else value['study_id'])
    assert denial['payload']['purpose'] == 'export'
    assert not list(output.parent.glob('.' + output.name + '.*.staging'))


@pytest.mark.parametrize('failure', ['fsync', 'publish', 'directory-sync'])
def test_actual_cli_publication_io_failure_preserves_complete_target_and_retry(
        source, tmp_path, monkeypatch, capsys, failure):
    store, value, grant, _ = source
    expected = encode(export_evidence(store, value['study_id'], grant))
    output = tmp_path / 'failed.json'
    original_write = store_module.atomic_write
    operation = {'fsync': 'fsync', 'publish': '_publish_no_replace',
                 'directory-sync': '_sync_directory'}[failure]
    touched = []

    def write(path, *args, **kwargs):
        if path != output:
            return original_write(path, *args, **kwargs)
        with monkeypatch.context() as context:
            owner = store_module.os if failure == 'fsync' else store_module
            original = getattr(owner, operation)

            def failed(*op_args, **op_kwargs):
                if failure == 'directory-sync' and op_args[0] != output.parent:
                    return original(*op_args, **op_kwargs)
                # Exercise the real flush or real directory sync first. A
                # publish fault must happen before any target exists.
                if failure != 'publish':
                    original(*op_args, **op_kwargs)
                touched.append(True)
                raise OSError('owned synthetic export I/O fault')

            context.setattr(owner, operation, failed)
            return original_write(path, *args, **kwargs)

    monkeypatch.setattr(store_module, 'atomic_write', write)
    assert main(arguments(store, output)) == 1
    assert json.loads(capsys.readouterr().out)['error']['code'] == 'RUNTIME_ERROR'
    assert touched
    if failure == 'directory-sync':
        assert output.read_bytes() == expected
    else:
        assert not output.exists()
    assert not list(output.parent.glob('.' + output.name + '.*.staging'))
    monkeypatch.setattr(store_module, 'atomic_write', original_write)
    assert main(arguments(store, output)) == 0
    assert output.read_bytes() == expected


def test_identical_export_retry_does_not_replace_existing_inode(source, tmp_path, capsys):
    store, value, grant, _ = source
    output = tmp_path / 'identity.json'
    expected = encode(export_evidence(store, value['study_id'], grant))
    output.write_bytes(expected)
    original = output.stat()
    assert main(arguments(store, output)) == 0
    assert output.read_bytes() == expected
    final = output.stat()
    assert (final.st_dev, final.st_ino, final.st_mtime_ns) == (
        original.st_dev, original.st_ino, original.st_mtime_ns)
    assert json.loads(capsys.readouterr().out)['bundle_hash'] == json.loads(expected)['bundle_hash']


@pytest.mark.parametrize('immutable', [False, True])
def test_colliding_staging_name_is_never_deleted(tmp_path, monkeypatch, immutable):
    output = tmp_path / 'collision.json'
    output.write_bytes(b'existing target')
    foreign = output.with_name('.' + output.name + '.controlled-collision.staging')
    foreign.write_bytes(b'owned by another interrupted writer')
    monkeypatch.setattr(store_module.uuid, 'uuid4', lambda: SimpleNamespace(hex='controlled-collision'))
    with pytest.raises(FileExistsError):
        store_module.atomic_write(output, b'new writer', immutable=immutable)
    assert output.read_bytes() == b'existing target'
    assert foreign.read_bytes() == b'owned by another interrupted writer'


def test_native_publication_probe_remains_executable():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'check_evidence_publication.py'
    completed = subprocess.run([sys.executable, '-B', str(script)], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'Production SHA256:' in completed.stdout
    assert 'Ran 6 tests' in completed.stderr
