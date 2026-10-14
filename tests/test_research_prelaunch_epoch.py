"""A cancelled source epoch remains visible and consumes its retry ordinal."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from application.research_budget import BudgetLedger, ResourceBudgetSpec
from application.research_queue import ResearchQueue
from application.research_queue_recovery import cancel_unclaimed
from application.research_restart_adjudication import adjudicate_restart, resolve_restart_adjudication
from experiments.pirc25.affine import code_hash
from experiments.pirc28.source_epoch import register_source_epoch
from experiments.pirc28.study import SyntheticStudyDefinition
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc28_source_epoch import historical_source
from tests.test_research_execution_policy import policy
from tests.test_research_restart_adjudication import interrupted_parent


def cancelled_epoch(tmp_path, monkeypatch):
    store, historical, current, current_policy = historical_source(tmp_path)
    old_value = current.to_dict()
    old_value['runtime_code_hash'] = 'c' * 64
    middle = SyntheticStudyDefinition.from_dict(old_value)
    decision = policy(store.store_id)
    decision['source_code_hash'] = 'c' * 64
    middle_policy = {'manifest_id': 'policy-' + digest(decision), 'sha256': digest(decision)}
    store.publish(middle_policy['manifest_id'], decision)
    with monkeypatch.context() as patch:
        patch.setattr('experiments.pirc28.source_epoch.code_hash', lambda: 'c' * 64)
        patch.setattr('experiments.pirc28.study.code_hash', lambda: 'c' * 64)
        middle_source = register_source_epoch(store, historical.definition_hash, middle,
                                             execution_policy=middle_policy)
    monkeypatch.setattr('infrastructure.process_tree.process_may_be_alive', lambda pid: False)
    parent, _ = interrupted_parent(store, historical)
    prior = adjudicate_restart(store, parent, source_bridge=middle_source['source_bridge'],
                               execution_policy=middle_policy, source_code_hash='c' * 64)
    prior_receipt = resolve_restart_adjudication(store, prior,
        execution_policy=middle_policy, source_code_hash='c' * 64)
    reservation = BudgetLedger(store).reserve(prior_receipt['retry_attempt_id'],
                                              ResourceBudgetSpec(middle_policy, 'c' * 64))
    ResearchQueue(store).enqueue(reservation)
    stop = {'schema_version': 'pirc28-prelaunch-owner-stop-v1', 'task_id': 'test-only',
        'session_id': 'epoch-recovery', 'pid': 123,
        'created_at': (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        'stop_confirmed_at': datetime.now(timezone.utc).isoformat(), 'native_child_count': 0,
        'reservation_id': reservation['reservation_id'], 'attempt_id': prior_receipt['retry_attempt_id'],
        'native_identity_check': 'exact-command-parent-and-creation-time'}
    stop_ref = {'manifest_id': 'stop-' + digest(stop), 'sha256': digest(stop)}
    store.publish(stop_ref['manifest_id'], stop)
    cancellation = cancel_unclaimed(store, reservation['reservation_id'], stop_evidence=stop_ref)
    current_source = register_source_epoch(store, historical.definition_hash, current,
                                           execution_policy=current_policy)
    return store, parent, prior, prior_receipt, middle_policy, current_policy, current_source, cancellation


def test_renewed_epoch_retains_cancelled_child_and_original_cost(tmp_path, monkeypatch):
    store, parent, prior, original, old_policy, policy_ref, source, cancellation = cancelled_epoch(
        tmp_path, monkeypatch)
    before = store.events()
    options = dict(source_bridge=source['source_bridge'], execution_policy=policy_ref,
                   source_code_hash=code_hash(), prelaunch_cancellations=[cancellation])
    reference = adjudicate_restart(store, parent, **options)
    current = resolve_restart_adjudication(store, reference,
        execution_policy=policy_ref, source_code_hash=code_hash())
    assert current['schema_version'] == 'pirc28-restart-adjudication-v2'
    assert current['retry_ordinal'] == 2
    assert current['retained_cost_ms'] == 47892
    assert store.events()[:len(before)] == before
    assert store.attempts()[original['retry_attempt_id']]['state'] == 'PREFLIGHT_FAILED'
    assert resolve_restart_adjudication(store, prior,
        execution_policy=old_policy, source_code_hash='c' * 64) == original
    assert adjudicate_restart(store, parent, **options) == reference
    reservation = BudgetLedger(store).reserve(current['retry_attempt_id'],
                                              ResourceBudgetSpec(policy_ref, code_hash()))
    assert reservation['restart_adjudication'] == reference
    assert len([e for e in store.events() if e['event_kind'] == 'WORKER_STARTED']) == 1


def test_renewal_cannot_omit_or_duplicate_prelaunch_history(tmp_path, monkeypatch):
    store, parent, _, _, _, policy_ref, source, cancellation = cancelled_epoch(tmp_path, monkeypatch)
    attempts_before = deepcopy(store.attempts())
    options = dict(source_bridge=source['source_bridge'], execution_policy=policy_ref,
                   source_code_hash=code_hash())
    for history in (None, [cancellation, cancellation]):
        with pytest.raises(ResearchError) as error:
            adjudicate_restart(store, parent, **options, prelaunch_cancellations=history)
        assert error.value.code == 'RETRY_EXHAUSTED'
    assert store.attempts() == attempts_before
