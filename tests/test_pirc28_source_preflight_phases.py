"""Phased immutable metadata/real-authority controls, not scientific fitting.

All stores and grants are disposable SIMULATED engineering fixtures. The full
matrix test keeps the actual frozen 64x16x2 plan, but produces no numeric result,
worker, production authorization or evidence of scientific qualification.
"""

import copy
from datetime import timedelta
import hashlib
import os
import threading
import time

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc28 import source_preflight
from experiments.pirc28.plan import REFERENCE_PLAN, load_plan, prepare_definition
from experiments.pirc28.runtime import switching_plugin
from experiments.pirc28.study import SyntheticStudyDefinition
from infrastructure.research_store import ResearchError, encode
from tests import test_pirc28_study as study_tests
from tests.test_pirc28_approximation import v2_master
from tests.test_pirc28_source_preflight import admitted_sources


def _admitted(root, monkeypatch, definition_factory):
    monkeypatch.setattr(study_tests, 'definition_value', definition_factory)
    store, master, _, _, studies = study_tests.prepared(root)
    study = store.manifest('study-'+studies[0]['study_id'])['spec']
    cell = next(value for value in study['cells'] if value['fit_K'] == 1)
    run = store.register_run(study['study_id'], cell)
    attempt_id = store.new_attempt(run)
    BudgetLedger(store).reserve(attempt_id, BudgetSpec(60, category='pilot'))
    AdmissionGate(store).prepare(study, cell, switching_plugin(), attempt_id)
    attempt = store.transition(attempt_id, 'RUNNING')
    return store, master, study, cell, attempt


@pytest.fixture
def isolated_source(tmp_path, monkeypatch):
    return _admitted(tmp_path/'store', monkeypatch, v2_master)


def _observe_pure_phases(store, monkeypatch):
    observed = []
    original_from = SyntheticStudyDefinition.from_dict
    original_binding = source_preflight.study_definition_binding
    original_fields = SyntheticStudyDefinition.preregistration_fields

    def from_dict(cls, value):
        assert store._read_snapshot() is None, 'whole master reconstruction owns writer lock'
        observed.append('master')
        return original_from(value)

    def binding(*args, **kwargs):
        assert store._read_snapshot() is None, 'complete source binding owns writer lock'
        observed.append('binding')
        return original_binding(*args, **kwargs)

    def fields(self, protocols):
        assert store._read_snapshot() is None, 'complete preregistration derivation owns writer lock'
        observed.append('preregistration')
        return original_fields(self, protocols)

    monkeypatch.setattr(SyntheticStudyDefinition, 'from_dict', classmethod(from_dict))
    monkeypatch.setattr(source_preflight, 'study_definition_binding', binding)
    monkeypatch.setattr(SyntheticStudyDefinition, 'preregistration_fields', fields)
    return observed


def test_all_pure_matrix_phases_and_polls_release_writer_lock(admitted_sources, monkeypatch):
    store, master, _, sources = admitted_sources
    observed = _observe_pure_phases(store, monkeypatch)
    polls = []

    def poll():
        assert store._read_snapshot() is None
        polls.append(time.monotonic())

    result = source_preflight.validate_scientific_source(store, *sources[0], check_budget=poll)
    assert result['study_definition_hash'] == master.definition_hash
    assert {'master', 'binding', 'preregistration'} <= set(observed)
    assert len(polls) > 2*len(master.study_ids)


def test_inherited_outer_writer_lock_is_refused_before_poll(admitted_sources):
    store, _, _, sources = admitted_sources
    polls = []
    with store._read_transaction():
        with pytest.raises(ResearchError) as error:
            source_preflight.validate_scientific_source(store, *sources[0], check_budget=lambda: polls.append(True))
        assert error.value.code == 'CONTRACT_MISMATCH'
    assert not polls


def test_malformed_first_physical_read_has_typed_refusal(isolated_source):
    store, _, study, cell, attempt = isolated_source
    path = store.path/'manifests'/('study-'+study['study_id']+'.json')
    with path.open('r+b') as stream:
        stream.write(b'!')
        stream.flush()
        os.fsync(stream.fileno())
    with pytest.raises(ResearchError) as error:
        source_preflight.validate_scientific_source(store, study, cell, attempt, check_budget=lambda: None)
    assert error.value.code == 'CORRUPT_ARTIFACT'


def test_grant_expiring_during_pure_phase_is_durably_denied(isolated_source, monkeypatch):
    from application import research_evidence
    store, _, study, cell, attempt = isolated_source
    real_clock = research_evidence.datetime
    settings = study['admission']
    expiry = real_clock.fromisoformat(store.authorization(settings['authorization_id'],
        version=settings.get('authorization_version'))['expires_at'])
    assert expiry > real_clock.now(expiry.tzinfo)
    original_binding = source_preflight.study_definition_binding

    class FutureClock(real_clock):
        @classmethod
        def now(cls, timezone=None):
            return (expiry+timedelta(seconds=1)).astimezone(timezone)

    def binding(*args, **kwargs):
        value = original_binding(*args, **kwargs)
        monkeypatch.setattr(research_evidence, 'datetime', FutureClock)
        return value

    monkeypatch.setattr(source_preflight, 'study_definition_binding', binding)
    with pytest.raises(ResearchError) as error:
        source_preflight.validate_scientific_source(store, study, cell, attempt, check_budget=lambda: None)
    assert error.value.code == 'UNAUTHORIZED_DATA'
    assert any(event['event_kind'] == 'DISCLOSURE_DENIED'
        and event['payload']['study_id'] == study['study_id'] for event in store.events())


@pytest.mark.parametrize('damage,code', [
    ('attempt', 'IDENTITY_CONFLICT'), ('settlement', 'UNQUALIFIED'),
    ('closure', 'BUDGET_EXHAUSTED'), ('hold', 'RECOVERY_REQUIRED'),
    ('grant', 'UNAUTHORIZED_DATA'), ('source', 'CORRUPT_ARTIFACT'),
    ('restored-source', 'CORRUPT_ARTIFACT'), ('events', 'CORRUPT_ARTIFACT'),
    ('missing-protocol', 'CORRUPT_ARTIFACT'), ('missing-source', 'CORRUPT_ARTIFACT')])
def test_between_phase_mutations_veto_handoff(isolated_source, monkeypatch, damage, code):
    store, _, study, cell, attempt = isolated_source
    original_binding = source_preflight.study_definition_binding
    changed = []

    def binding(*args, **kwargs):
        assert store._read_snapshot() is None
        changed.append(damage)
        if damage == 'attempt':
            store.transition(attempt['attempt_id'], 'INTERRUPTED', error_code='SIMULATED_CONTROL')
        elif damage == 'settlement':
            reservation = next(event['payload'] for event in store.events()
                if event['event_kind'] == 'RESERVE' and event['payload']['attempt_id'] == attempt['attempt_id'])
            BudgetLedger(store).settle(reservation['reservation_id'], 1, outcome='FAILED')
        elif damage == 'closure':
            store.append('ARM_CLOSED', {'arm_id': cell['arm_id'], 'reason': 'SIMULATED_CONTROL'})
        elif damage == 'hold':
            (store.path/'recovery-hold.json').write_bytes(encode({'reason': 'SIMULATED_CONTROL'}))
        elif damage in ('missing-protocol', 'missing-source'):
            name = ('protocol-'+study['admission']['protocol_id'] if damage == 'missing-protocol'
                else 'study-'+study['study_id'])
            (store.path/'manifests'/(name+'.json')).unlink()
        elif damage == 'grant':
            original = store.authorization

            def authorization(*args, **kwargs):
                value = copy.deepcopy(original(*args, **kwargs))
                value['purposes'] = ['export']
                return value

            # A changed current-authority projection, not a forged native grant
            # or a permission granted to the production runtime.
            monkeypatch.setattr(store, 'authorization', authorization)
        else:
            path = (store.path/'events'/'0000000000000001.json' if damage == 'events'
                else store.path/'manifests'/('study-'+study['study_id']+'.json'))
            content, stamp = path.read_bytes(), path.stat()
            replacement = bytearray(content)
            replacement[-2] = ord(' ') if replacement[-2] != ord(' ') else ord('\n')
            # Same inode, same size. Restoring bytes AND mtime must not erase
            # the native change identity seen by the original pinned descriptor.
            with path.open('r+b') as stream:
                stream.write(replacement)
                stream.flush()
                os.fsync(stream.fileno())
                if damage == 'restored-source':
                    stream.seek(0)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            if damage == 'restored-source':
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return original_binding(*args, **kwargs)

    monkeypatch.setattr(source_preflight, 'study_definition_binding', binding)
    with pytest.raises(ResearchError) as error:
        source_preflight.validate_scientific_source(store, study, cell, attempt, check_budget=lambda: None)
    assert error.value.code == code
    assert changed == [damage]
    if damage != 'events':
        assert not any(event['event_kind'] == 'WORKER_STARTED' for event in store.events())


def test_physical_completion_follows_final_authority_io(isolated_source, monkeypatch):
    store, _, study, cell, attempt = isolated_source
    original = store._events
    changed = []
    baseline = sum(event['event_kind'] == 'DISCLOSURE_ALLOWED'
        and event['payload']['study_id'] == study['study_id'] for event in store.events())

    def events():
        values = original()
        allowed = [event for event in values if event['event_kind'] == 'DISCLOSURE_ALLOWED'
            and event['payload']['study_id'] == study['study_id']]
        completions = getattr(store._read_scope, 'completions', ())
        if (store._read_snapshot() is None and len(completions) == 4
                and len(allowed) == baseline+2 and not changed):
            path = store.path/'manifests'/('study-'+study['study_id']+'.json')
            content = path.read_bytes()
            # The original prefix has just been physically checked. Corrupt an
            # earlier consumed source before the final completion callbacks.
            with path.open('r+b') as stream:
                stream.seek(len(content)-2)
                stream.write(b' ')
                stream.flush()
                os.fsync(stream.fileno())
            changed.append((len(completions), len(allowed)-baseline))
        return values

    monkeypatch.setattr(store, '_events', events)
    with pytest.raises(ResearchError) as error:
        source_preflight.validate_scientific_source(store, study, cell, attempt, check_budget=lambda: None)
    assert error.value.code == 'CORRUPT_ARTIFACT'
    assert changed == [(4, 2)]


def test_frozen_full_64_matrix_allows_original_timeout_owner_heartbeats(tmp_path, monkeypatch):
    assert hashlib.sha256(REFERENCE_PLAN.read_bytes()).hexdigest() == (
        'a1877da2e835fd5c1053fc88e8e8d19103741430eefe5e7663399901b21a7a4b')
    store, master, study, cell, attempt = _admitted(tmp_path/'store', monkeypatch,
        lambda: prepare_definition(load_plan(REFERENCE_PLAN)).to_dict())
    assert len(master.study_ids) == 64
    assert len(master.to_dict()['identifiability_spec']['replicate_ids']) == 16
    assert sum(len(store.manifest('study-'+name)['spec']['cells']) for name in master.study_ids) == 2048
    observed = _observe_pure_phases(store, monkeypatch)
    stopped = threading.Event()
    receipts, errors = [], []

    def heartbeat():
        while not stopped.is_set():
            started = time.monotonic()
            try:
                store.append('HEARTBEAT', {'attempt_id': attempt['attempt_id'],
                    'monotonic_elapsed_ms': 0})
                receipts.append(time.monotonic()-started)
            except BaseException as error:
                errors.append(error)
                return
            stopped.wait(0.2)

    writer = threading.Thread(target=heartbeat)
    writer.start()

    def poll():
        assert store._read_snapshot() is None

    try:
        result = source_preflight.validate_scientific_source(store, study, cell, attempt, check_budget=poll)
    finally:
        stopped.set()
        writer.join(25)
        assert not writer.is_alive(), 'full-matrix observer must be terminal, never detached'
    assert not errors, errors
    assert len(receipts) >= 2
    assert {'master', 'binding', 'preregistration'} <= set(observed)
    assert result['study_definition_hash'] == master.definition_hash
    assert result['qualified_identifiability'] is False and result['real_data_authorized'] is False
    assert not any(event['event_kind'] == 'WORKER_STARTED' for event in store.events())
