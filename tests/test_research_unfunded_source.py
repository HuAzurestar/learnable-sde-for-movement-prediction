"""Unfunded source correction preserves paid retry ordinals and all history."""

from copy import deepcopy

import pytest

from application.research_budget import BudgetLedger, ResourceBudgetSpec
from application.research_restart_adjudication import adjudicate_restart, resolve_restart_adjudication
from application.research_unfunded_source import supersede_unfunded_restart, resolve_unfunded_supersession
from experiments.pirc25.affine import code_hash
from experiments.pirc28.source_epoch import register_source_epoch
from experiments.pirc28.study import SyntheticStudyDefinition
from infrastructure.research_store import ResearchError, digest
from tests.test_research_execution_policy import policy
from tests.test_research_prelaunch_epoch import cancelled_epoch


def proposal(tmp_path, monkeypatch):
    store, parent, _, _, _, current_policy, source, cancellation = cancelled_epoch(tmp_path, monkeypatch)
    ref = adjudicate_restart(store, parent, source_bridge=source['source_bridge'],
        execution_policy=current_policy, source_code_hash=code_hash(), prelaunch_cancellations=[cancellation])
    return store, parent, current_policy, source, cancellation, ref


def test_unfunded_replacement_preserves_ordinal_and_never_creates_a_cost(tmp_path, monkeypatch):
    store, parent, current_policy, source, cancellation, ref = proposal(tmp_path, monkeypatch)
    old = store.manifest(ref['manifest_id'])
    before = store.events()
    options = dict(source_code_hash='d' * 64, task_id='test-only', session_id='source-correction',
                   reason='worker must prime definitions before its first authority scope')
    supersession = supersede_unfunded_restart(store, ref, **options)
    retired = resolve_unfunded_supersession(store, supersession, source_code_hash='d' * 64)
    assert store.attempts()[old['retry_attempt_id']]['state'] == 'CANCELLED'
    assert not any(e['event_kind'] in {'RESERVE', 'SETTLE', 'WORKER_STARTED'}
                   for e in store.events()[len(before):])
    after = store.events()
    assert supersede_unfunded_restart(store, ref, **options) == supersession
    assert store.events() == after
    assert resolve_restart_adjudication(store, ref, execution_policy=current_policy,
                                        source_code_hash=code_hash())['retry_ordinal'] == 2

    bridge = store.manifest(source['source_bridge']['manifest_id'])
    value = store.manifest('pirc28-study-definition-' + source['study_definition_hash'])
    value['runtime_code_hash'] = 'd' * 64
    future = SyntheticStudyDefinition.from_dict(value)
    decision = policy(store.store_id)
    decision['source_code_hash'] = 'd' * 64
    future_policy = {'manifest_id': 'policy-' + digest(decision), 'sha256': digest(decision)}
    store.publish(future_policy['manifest_id'], decision)
    with monkeypatch.context() as patch:
        patch.setattr('experiments.pirc28.source_epoch.code_hash', lambda: 'd' * 64)
        patch.setattr('experiments.pirc28.study.code_hash', lambda: 'd' * 64)
        updated = register_source_epoch(store, bridge['historical_definition_hash'], future,
                                        execution_policy=future_policy)
    arguments = dict(source_bridge=updated['source_bridge'], execution_policy=future_policy,
                     source_code_hash='d' * 64, prelaunch_cancellations=[cancellation])
    with pytest.raises(ResearchError):
        adjudicate_restart(store, parent, **arguments)
    new = adjudicate_restart(store, parent, **arguments, unfunded_supersession=supersession)
    receipt = resolve_restart_adjudication(store, new,
        execution_policy=future_policy, source_code_hash='d' * 64)
    assert receipt['schema_version'] == 'pirc28-restart-adjudication-v3'
    assert receipt['retry_ordinal'] == 2 and receipt['retained_cost_ms'] == 47892
    assert receipt['unfunded_supersession'] == supersession
    assert retired['attempt_id'] != receipt['retry_attempt_id']
    BudgetLedger(store).reserve(receipt['retry_attempt_id'], ResourceBudgetSpec(future_policy, 'd' * 64))


def test_any_funding_makes_source_supersession_unavailable(tmp_path, monkeypatch):
    store, _, current_policy, _, _, ref = proposal(tmp_path, monkeypatch)
    receipt = store.manifest(ref['manifest_id'])
    BudgetLedger(store).reserve(receipt['retry_attempt_id'], ResourceBudgetSpec(current_policy, code_hash()))
    before = deepcopy(store.events())
    with pytest.raises(ResearchError, match='funded or admitted'):
        supersede_unfunded_restart(store, ref, source_code_hash='d' * 64,
                                  task_id='test-only', session_id='refused', reason='source correction')
    assert store.events() == before
