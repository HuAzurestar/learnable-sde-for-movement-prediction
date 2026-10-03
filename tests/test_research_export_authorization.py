"""Complete evidence export cannot reuse entry permission after actual I/O."""

from datetime import datetime, timezone
import json
import subprocess
import sys
import threading

import pytest

import application.research_evidence as evidence_module
import infrastructure.research_store as store_module
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_admission_fixtures import attach_foreign_model
from tests.test_research_admission_chain import prepared
from tests.test_research_evidence import setup


@pytest.fixture
def permission_clock(monkeypatch):
    expired = [False]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100 if expired[0] else 2030, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(evidence_module, 'datetime', Clock)
    monkeypatch.setattr(store_module, 'datetime', Clock)
    return expired


@pytest.fixture
def source(tmp_path):
    """Owner-published legacy synthetic result and real ledger, not science."""
    store, value, grant = setup(tmp_path)
    cell = value['cells'][0]
    run = store.register_run(value['study_id'], cell)
    attempt = store.new_attempt(run)
    reservation = BudgetLedger(store).reserve(attempt, BudgetSpec(1))
    store.transition(attempt, 'RUNNING')
    result = {'spec_hash': digest(value), 'cell_hash': digest(cell), 'protocol_hash': value['protocol_hash'],
              'metrics': {'error': 1}, 'metric_units': {'error': 'm'}, 'qualification': 'fixture',
              'state_order': ['x', 'y', 'vx', 'vy'], 'units': ['m', 'm', 'm/s', 'm/s']}
    artifact = store.artifact(encode(result), role='result', visibility='synthetic',
                             block_ids=['fixture-1'], study_id=value['study_id'])
    BudgetLedger(store).settle(reservation['reservation_id'], 100, outcome='SUCCEEDED')
    store.transition(attempt, 'SUCCEEDED', artifact_id=artifact['artifact_id'])
    return store, value, grant, artifact


def bundles(store):
    return [event for event in store.events() if event['event_kind'] == 'MANIFEST'
            and event['payload']['object_id'].startswith('bundle-')]


def test_valid_export_preserves_matrix_metrics_cost_and_idempotent_source(source, permission_clock):
    store, value, grant, artifact = source
    result = export_evidence(store, value['study_id'], grant)
    assert len(result['cells']) == len(result['expected_cells']) == 2
    assert [cell['status'] for cell in result['cells']] == ['SUCCEEDED', 'MISSING']
    assert result['cells'][0]['artifact_id'] == artifact['artifact_id']
    assert result['cells'][0]['metrics'] == {'error': 1}
    assert result['cells'][0]['cost']['charged_ms'] == 100
    assert result['cells'][0]['cost']['sources'][0]['payload']['monotonic_elapsed_ms'] == 100
    assert result == export_evidence(store, value['study_id'], grant)
    assert len(bundles(store)) == 1
    assert store._read_snapshot() is None


@pytest.mark.parametrize('phase', ['read', 'before-publication', 'publication', 'physical'])
def test_export_expiry_across_actual_reads_and_publication_denies_return(source, permission_clock,
                                                                      monkeypatch, phase):
    store, value, grant, _ = source
    original_read, original_publish = store.read_artifact, store.publish
    original_visibility, original_events = evidence_module.require_export_visibility, store._events
    read_done, expired_checks = [], []

    def expire():
        permission_clock[0] = True
        expired_checks.append(True)

    def read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        read_done.append(True)
        if phase == 'read':
            expire()
        return result

    def visibility(*args, **kwargs):
        result = original_visibility(*args, **kwargs)
        if read_done and phase == 'before-publication':
            expire()
        return result

    def publish(object_id, payload):
        result = original_publish(object_id, payload)
        if object_id.startswith('bundle-') and phase == 'publication':
            expire()
        return result

    def physical_check():
        result = original_events()
        if read_done and phase == 'physical' and store._read_snapshot() is None:
            expire()
        return result

    monkeypatch.setattr(store, 'read_artifact', read)
    monkeypatch.setattr(store, 'publish', publish)
    monkeypatch.setattr(evidence_module, 'require_export_visibility', visibility)
    monkeypatch.setattr(store, '_events', physical_check)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        export_evidence(store, value['study_id'], grant)
    assert read_done and expired_checks
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'
    if phase in {'read', 'before-publication'}:
        assert not bundles(store), 'already expired permission published a new evidence bundle'
    if phase == 'publication':
        # Keep the authorized, durably written private candidate; do not delete
        # immutable evidence to conceal an expiry that prevented disclosure.
        assert len(bundles(store)) == 1


def test_export_rehashes_main_grant_after_actual_result_read(source, permission_clock, monkeypatch):
    store, value, grant, _ = source
    original = store.read_artifact

    def corrupt_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        (store.path / 'manifests' / ('authorization-' + grant['authorization_id'] + '.json')).write_bytes(
            encode({**grant, 'purposes': []}))
        return result

    monkeypatch.setattr(store, 'read_artifact', corrupt_after_read)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        export_evidence(store, value['study_id'], grant)
    assert not bundles(store)
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'


def test_failed_final_export_authorization_journal_prevents_publication(source, permission_clock, monkeypatch):
    store, value, grant, _ = source
    original = store.append
    allowed = []

    def fail_final_journal(kind, *args, **kwargs):
        if kind == 'DISCLOSURE_ALLOWED':
            allowed.append(True)
            if len(allowed) == 2:
                raise OSError('final export journal unavailable')
        return original(kind, *args, **kwargs)

    monkeypatch.setattr(store, 'append', fail_final_journal)
    with pytest.raises(OSError, match='final export journal unavailable'):
        export_evidence(store, value['study_id'], grant)
    assert len(allowed) == 2
    assert not bundles(store)
    assert store._read_snapshot() is None


@pytest.mark.parametrize('change', ['reservation', 'recovery-hold'])
def test_export_rejects_cost_or_recovery_changes_during_source_assembly(source, permission_clock,
                                                                      monkeypatch, change):
    store, value, grant, _ = source
    original = store.read_artifact
    changed = []

    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        if not changed:
            changed.append(True)
            if change == 'reservation':
                run = store.register_run(value['study_id'], value['cells'][1])
                attempt = store.new_attempt(run)
                BudgetLedger(store).reserve(attempt, BudgetSpec(1))
            else:
                store.quarantine_tail('synthetic export snapshot recovery hold')
        return result

    monkeypatch.setattr(store, 'read_artifact', change_after_read)
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        export_evidence(store, value['study_id'], grant)
    assert changed and not bundles(store)
    balance = BudgetLedger(store).balance('affine')
    assert balance['committed_ms'] == (1100 if change == 'reservation' else 100)
    assert balance['closed'] is (change == 'recovery-hold')


def test_corrupt_chain_during_bundle_publication_never_returns_evidence(source, permission_clock, monkeypatch):
    store, value, grant, _ = source
    original = store.publish
    corrupted = store.path / 'events/0000000000000001.json'
    changed = []

    def corrupt_after_publish(object_id, payload):
        result = original(object_id, payload)
        if object_id.startswith('bundle-'):
            changed.append(True)
            corrupted.write_bytes(b'{}')
        return result

    monkeypatch.setattr(store, 'publish', corrupt_after_publish)
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        export_evidence(store, value['study_id'], grant)
    assert changed and corrupted.read_bytes() == b'{}'
    assert store._read_snapshot() is None


def test_export_bounds_physical_chain_work_without_dropping_actual_read_journals(source, permission_clock,
                                                                              monkeypatch):
    store, value, grant, artifact = source
    original = store._json
    reads = []

    def observe(path):
        if path.parent == store.path / 'events':
            reads.append(path)
        return original(path)

    monkeypatch.setattr(store, '_json', observe)
    before = len(store.events())
    reads.clear()
    result = export_evidence(store, value['study_id'], grant)
    physical_reads = len(reads)
    events = store.events()
    assert physical_reads <= 2 * len(events)
    assert result['cells'][0]['artifact_id'] == artifact['artifact_id']
    assert [event['event_kind'] for event in events[before:] if event['event_kind'].startswith('READ_')] == [
        'READ_STARTED', 'READ_COMPLETED']
    assert sum(event['event_kind'] == 'EXPOSURE_ALLOWED' for event in events[before:]) == 1


def test_parallel_authorized_disclosure_does_not_invalidate_an_unchanged_export(source, monkeypatch):
    store, value, grant, _ = source
    original = store.read_artifact
    started, done = threading.Event(), threading.Event()
    workers, errors = [], []

    def disclose():
        started.set()
        try:
            evidence_module.authorize_study(store, value['study_id'], grant, 'export')
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    def interleave(*args, **kwargs):
        result = original(*args, **kwargs)
        worker = threading.Thread(target=disclose)
        workers.append(worker)
        worker.start()
        assert started.wait(1)
        # The new complete read scope should retain its own OS lock, not
        # borrow another thread's authority or wait for a blocked journal.
        done.wait(0.2)
        return result

    monkeypatch.setattr(store, 'read_artifact', interleave)
    try:
        result = export_evidence(store, value['study_id'], grant)
        assert [cell['status'] for cell in result['cells']] == ['SUCCEEDED', 'MISSING']
        assert result['cells'][0]['cost']['charged_ms'] == 100
    finally:
        for worker in workers:
            worker.join(timeout=5)
        assert all(not worker.is_alive() for worker in workers)
    assert done.is_set() and not errors


CLI_EXPIRE = r'''
from datetime import datetime, timezone
from importlib import import_module
import sys
from unittest.mock import patch
import infrastructure.research_store as store_module
ResearchStore = store_module.ResearchStore
cli = import_module('experiments.pirc25.__main__')
expired = [False]
armed = [False]
original = ResearchStore.read_artifact
original_write, original_fsync = cli.atomic_write, store_module.os.fsync
class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2100 if expired[0] else 2030, 1, 1, tzinfo=timezone.utc)
def read(store, *args, **kwargs):
    result = original(store, *args, **kwargs)
    if sys.argv[4] == 'read':
        expired[0] = True
    return result
def write(*args, **kwargs):
    armed[0] = True
    try:
        return original_write(*args, **kwargs)
    finally:
        armed[0] = False
def fsync(fd):
    original_fsync(fd)
    if armed[0] and sys.argv[4] == 'target-fsync':
        expired[0] = True
with patch('application.research_evidence.datetime', Clock), patch('infrastructure.research_store.datetime', Clock), patch.object(ResearchStore, 'read_artifact', read), patch.object(cli, 'atomic_write', write), patch.object(store_module.os, 'fsync', fsync):
    raise SystemExit(cli.main(['--root', sys.argv[1], '--store-id', sys.argv[2], 'export', 'synthetic', '--authorization-id', sys.argv[5], '--output', sys.argv[3]]))
'''


@pytest.mark.parametrize('phase', ['read', 'target-fsync'])
def test_actual_cli_expiry_returns_error_without_writing_bundle_file(source, tmp_path, phase):
    store, _, _, _ = source
    output = tmp_path / 'denied-bundle.json'
    completed = subprocess.run([sys.executable, '-B', '-c', CLI_EXPIRE, str(store.path.parent), store.store_id,
                                str(output), phase, 'export'], capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert json.loads(completed.stdout)['error']['code'] == 'UNAUTHORIZED_DATA'
    assert not output.exists()
    assert not list(output.parent.glob('.' + output.name + '.*.staging'))
    if phase == 'read':
        assert not bundles(store)


def test_actual_valid_cli_exports_complete_original_bundle(source, tmp_path):
    store, _, _, artifact = source
    output = tmp_path / 'allowed-bundle.json'
    completed = subprocess.run([sys.executable, '-B', '-m', 'experiments.pirc25', '--root', str(store.path.parent),
        '--store-id', store.store_id, 'export', 'synthetic', '--authorization-id', 'export', '--output', str(output)],
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_bytes())
    assert output.read_bytes() == encode(result)
    assert result['cells'][0]['artifact_id'] == artifact['artifact_id']
    assert [cell['status'] for cell in result['cells']] == ['SUCCEEDED', 'MISSING']
    assert result['cells'][0]['cost']['charged_ms'] == 100


@pytest.fixture
def foreign_source(tmp_path):
    # Use actual AdmissionGate/SharedRunner/native worker evidence, not a
    # manually invented foreign-model receipt or scientific qualification.
    store, value, registry, grant = prepared(tmp_path, formal=True, two_arms=True)
    attach_foreign_model(store, value)
    store.register(value, digest(value))
    outcome = SharedRunner(store, registry).run_cell(value['study_id'], digest(value['cells'][0]), budget=BudgetSpec(10))
    assert outcome['state'] == 'SUCCEEDED', outcome
    main = {**grant, 'authorization_id': 'long-lived-consumer', 'expires_at': '2199-01-01T00:00:00+00:00'}
    store.authorize(main)
    return store, value, main


def test_valid_foreign_model_export_keeps_actual_admission_and_partial_matrix(foreign_source, permission_clock):
    store, value, grant = foreign_source
    result = export_evidence(store, value['study_id'], grant)
    assert [cell['status'] for cell in result['cells']] == ['SUCCEEDED', 'MISSING']
    documents = result['cells'][0]['admission']['documents']
    assert documents['model_authorization']['authorization_id'] == 'model-consumer'
    assert documents['model_qualification_evidence']


@pytest.mark.parametrize('phase', ['read-expiry', 'read-grant-change', 'publication-expiry'])
def test_main_permission_cannot_mask_expired_or_changed_foreign_model_export(foreign_source,
                                                                          permission_clock,
                                                                          monkeypatch, phase):
    store, value, main = foreign_source
    original_read, original_publish = store.read_artifact, store.publish
    touched = []

    def read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        if kwargs['authorization']['authorization_id'] == 'model-consumer':
            touched.append(True)
            if phase == 'read-expiry':
                permission_clock[0] = True
            elif phase == 'read-grant-change':
                foreign = store.manifest('authorization-model-consumer')
                (store.path / 'manifests/authorization-model-consumer.json').write_bytes(
                    encode({**foreign, 'purposes': ['evaluate']}))
        return result

    def publish(object_id, payload):
        result = original_publish(object_id, payload)
        if object_id.startswith('bundle-') and phase == 'publication-expiry':
            permission_clock[0] = True
        return result

    monkeypatch.setattr(store, 'read_artifact', read)
    monkeypatch.setattr(store, 'publish', publish)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        export_evidence(store, value['study_id'], main)
    assert touched
    denial = store.events()[-1]
    assert denial['event_kind'] == 'DISCLOSURE_DENIED'
    assert denial['payload']['study_id'] == 'model-study'
    assert denial['payload']['purpose'] == 'export'
    if phase != 'publication-expiry':
        assert not bundles(store)


def test_actual_cli_target_fsync_cannot_publish_expired_foreign_model_evidence(foreign_source, tmp_path):
    store, _, main = foreign_source
    output = tmp_path / 'denied-model-bundle.json'
    completed = subprocess.run([sys.executable, '-B', '-c', CLI_EXPIRE, str(store.path.parent), store.store_id,
        str(output), 'target-fsync', main['authorization_id']], capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert json.loads(completed.stdout)['error']['code'] == 'UNAUTHORIZED_DATA'
    assert not output.exists()
    assert not list(output.parent.glob('.' + output.name + '.*.staging'))
    denial = store.events()[-1]
    assert denial['event_kind'] == 'DISCLOSURE_DENIED'
    assert denial['payload']['study_id'] == 'model-study'
