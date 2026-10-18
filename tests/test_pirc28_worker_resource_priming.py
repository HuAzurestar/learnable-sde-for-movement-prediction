"""Simulate a fresh worker's empty pure cache before its first budget check."""

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, ResourceBudgetSpec
from application.research_restart_adjudication import adjudicate_restart
from experiments.pirc25.affine import code_hash
from experiments.pirc28 import source_preflight, study as definitions
from experiments.pirc28.runtime import switching_plugin
from infrastructure.research_store import digest
from tests import test_pirc28_study as study_tests
from tests.test_pirc28_approximation import v2_master
from tests.test_research_restart_adjudication import epoch_fixture, interrupted_parent


def test_worker_primes_before_the_first_closed_arm_authority_transaction(tmp_path, monkeypatch):
    monkeypatch.setattr(study_tests, 'definition_value', v2_master)
    store, historical, current, policy, bridge = epoch_fixture(tmp_path)
    monkeypatch.setattr('infrastructure.process_tree.process_may_be_alive', lambda pid: False)
    parent, _ = interrupted_parent(store, historical)
    restart = adjudicate_restart(store, parent, source_bridge=bridge,
                                  execution_policy=policy, source_code_hash=code_hash())
    receipt = store.manifest(restart['manifest_id'])
    run = store.manifest('run-' + receipt['current_run_id'])
    study = store.manifest('study-' + run['study_id'])['spec']
    cell = next(c for c in study['cells'] if digest(c) == run['cell_hash'])
    attempt_id = receipt['retry_attempt_id']
    BudgetLedger(store).reserve(attempt_id, ResourceBudgetSpec(policy, code_hash()))
    AdmissionGate(store).prepare(study, cell, switching_plugin(), attempt_id)
    attempt = store.transition(attempt_id, 'RUNNING')
    definitions._validated_definition.cache_clear()
    validate = definitions._validate
    scopes = []

    def cold(document):
        scopes.append(store._read_snapshot())
        assert store._read_snapshot() is None, 'fresh worker cold validation owns the shared writer lock'
        return validate(document)

    monkeypatch.setattr(definitions, '_validate', cold)
    result = source_preflight.validate_scientific_source(store, study, cell, attempt,
                                                        check_budget=lambda: None)
    assert result['study_definition_hash'] == current.definition_hash
    assert result['qualified_identifiability'] is False
    assert scopes == [None, None]
