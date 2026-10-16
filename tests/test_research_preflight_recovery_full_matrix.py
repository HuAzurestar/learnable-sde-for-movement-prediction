"""Full frozen-matrix recovery metadata liveness, not fitting or science."""

import hashlib
import threading
import time

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, ResourceBudgetSpec
from experiments.pirc25.affine import code_hash
from experiments.pirc28 import source_preflight, study as definitions
from experiments.pirc28.plan import REFERENCE_PLAN
from experiments.pirc28.runtime import switching_plugin
from tests.test_research_preflight_recovery_native import native_case


@pytest.mark.parametrize('native_case', ['full64'], indirect=True)
def test_full_64_one_time_recovery_keeps_original_15s_writer_live(native_case, monkeypatch):
    store, policy, epoch, _, receipt, *_ = native_case
    assert hashlib.sha256(REFERENCE_PLAN.read_bytes()).hexdigest() == (
        'a1877da2e835fd5c1053fc88e8e8d19103741430eefe5e7663399901b21a7a4b')
    master = definitions.SyntheticStudyDefinition.from_dict(
        store.manifest('pirc28-study-definition-' + epoch['study_definition_hash']))
    assert len(master.study_ids) == 64
    assert len(master.to_dict()['identifiability_spec']['replicate_ids']) == 16
    assert sum(len(store.manifest('study-' + name)['spec']['cells']) for name in master.study_ids) == 2048
    run = store.manifest('run-' + receipt['current_run_id'])
    spec = store.manifest('study-' + run['study_id'])['spec']
    BudgetLedger(store).reserve(receipt['retry_attempt_id'], ResourceBudgetSpec(policy, code_hash()))
    AdmissionGate(store).prepare(spec, run['cell'], switching_plugin(), receipt['retry_attempt_id'])
    attempt = store.transition(receipt['retry_attempt_id'], 'RUNNING')
    store.lock_wait_seconds = 15  # Do not inherit upstream's enlarged 300s wait.
    definitions._validated_definition.cache_clear()
    original_validate = definitions._validate
    original_binding = source_preflight.study_definition_binding
    original_fields = definitions.SyntheticStudyDefinition.preregistration_fields
    observed = []

    def cold(value):
        assert store._read_snapshot() is None, 'cold epoch definition validation owns writer'
        observed.append('cold-definition')
        return original_validate(value)

    def binding(*args, **kwargs):
        assert store._read_snapshot() is None, 'complete source binding owns writer'
        observed.append('binding')
        return original_binding(*args, **kwargs)

    def fields(self, protocols):
        assert store._read_snapshot() is None, 'complete preregistration derivation owns writer'
        observed.append('preregistration')
        return original_fields(self, protocols)

    monkeypatch.setattr(definitions, '_validate', cold)
    monkeypatch.setattr(source_preflight, 'study_definition_binding', binding)
    monkeypatch.setattr(definitions.SyntheticStudyDefinition, 'preregistration_fields', fields)
    stopped = threading.Event()
    receipts, errors = [], []

    def heartbeat():
        while not stopped.is_set():
            started = time.monotonic()
            try:
                store.append('HEARTBEAT', {'attempt_id': attempt['attempt_id'], 'monotonic_elapsed_ms': 0})
                receipts.append(time.monotonic() - started)
            except BaseException as error:
                errors.append(error)
                return
            stopped.wait(0.2)

    writer = threading.Thread(target=heartbeat)
    writer.start()

    def poll():
        assert store._read_snapshot() is None

    try:
        result = source_preflight.validate_scientific_source(store, spec, run['cell'], attempt,
                                                            check_budget=poll)
    finally:
        stopped.set()
        writer.join(25)
        assert not writer.is_alive(), 'one-time full-matrix observer must be terminal'
    assert not errors, errors
    assert len(receipts) >= 2
    assert observed.count('cold-definition') == 3
    assert {'binding', 'preregistration'} <= set(observed)
    assert result['study_definition_hash'] == master.definition_hash
    assert result['qualified_identifiability'] is False and result['real_data_authorized'] is False
    assert not any(e['event_kind'] == 'WORKER_STARTED' for e in store.events()
                   if e['payload'].get('attempt_id') == attempt['attempt_id'])
