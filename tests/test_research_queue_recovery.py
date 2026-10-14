"""Pre-claim recovery never guesses that a claimed worker incurred zero cost."""

from datetime import datetime, timezone
import subprocess
import sys

import pytest

from application.research_budget import BudgetLedger
from application.research_queue import ResearchQueue
from application.research_queue_recovery import cancel_unclaimed, resolve_prelaunch_cancellation
from infrastructure.research_store import ResearchError, digest
from tests.test_research_resource_budget import registered


def pending(tmp_path):
    store, attempts, budget = registered(tmp_path, 1)
    reservation = BudgetLedger(store).reserve(attempts[0], budget)
    queue = ResearchQueue(store)
    queue.enqueue(reservation)
    created = datetime.now(timezone.utc).isoformat()
    with subprocess.Popen([sys.executable, '-c', 'pass']) as owner:
        owner.wait(timeout=10)
        pid = owner.pid
    evidence = {'schema_version': 'pirc28-prelaunch-owner-stop-v1',
        'task_id': 'disposable-recovery-test', 'session_id': 'native-owner-only',
        'pid': pid, 'created_at': created,
        'stop_confirmed_at': datetime.now(timezone.utc).isoformat(),
        'native_child_count': 0, 'native_identity_check': 'exact-command-parent-and-creation-time'}
    reference = {'manifest_id': 'stop-' + digest(evidence), 'sha256': digest(evidence)}
    store.publish(reference['manifest_id'], evidence)
    return store, attempts[0], reservation, reference, queue


def test_native_preclaim_stop_settles_only_own_zero_worker_cost_and_is_idempotent(tmp_path):
    store, attempt_id, reservation, stop, _ = pending(tmp_path)
    before = store.events()
    ref = cancel_unclaimed(store, reservation['reservation_id'], stop_evidence=stop)
    receipt = resolve_prelaunch_cancellation(store, ref)
    assert receipt['worker_charged_ms'] == 0
    assert store.attempts()[attempt_id]['state'] == 'PREFLIGHT_FAILED'
    assert store.attempts()[attempt_id]['started_at'] is None
    assert store.events()[:len(before)] == before
    assert BudgetLedger(store)._state()[0][reservation['reservation_id']]['charged_ms'] == 0
    after = store.events()
    assert cancel_unclaimed(store, reservation['reservation_id'], stop_evidence=stop) == ref
    assert store.events() == after
    assert not any(e['event_kind'] in {'WORKER_STARTED', 'ARM_CLOSED'} for e in after)


@pytest.mark.parametrize('fault', ['claimed', 'running', 'launch', 'prior-cost', 'live-owner', 'hold'])
def test_unsafe_prelaunch_recovery_is_refused_before_any_write(tmp_path, monkeypatch, fault):
    store, attempt_id, reservation, stop, queue = pending(tmp_path)
    if fault == 'claimed':
        queue.claim(reservation['reservation_id'])
    elif fault == 'running':
        store.transition(attempt_id, 'RUNNING')
    elif fault == 'launch':
        store.append('WORKER_STARTED', {'attempt_id': attempt_id, 'pid': 123})
    elif fault == 'prior-cost':
        BudgetLedger(store).settle(reservation['reservation_id'], 17, outcome='FAILED')
    elif fault == 'live-owner':
        monkeypatch.setattr('infrastructure.process_tree.process_may_be_alive', lambda pid: True)
    else:
        (store.path / 'recovery-hold.json').write_text('{}')
    before = store.events()
    with pytest.raises(ResearchError):
        cancel_unclaimed(store, reservation['reservation_id'], stop_evidence=stop)
    assert store.events() == before


def test_cancel_replay_can_finish_after_terminal_publication_fault(tmp_path, monkeypatch):
    store, attempt_id, reservation, stop, _ = pending(tmp_path)
    original = store._transition
    monkeypatch.setattr(store, '_transition', lambda *a, **k: (_ for _ in ()).throw(OSError('power-loss')))
    with pytest.raises(OSError, match='power-loss'):
        cancel_unclaimed(store, reservation['reservation_id'], stop_evidence=stop)
    assert store.attempts()[attempt_id]['state'] == 'REGISTERED'
    monkeypatch.setattr(store, '_transition', original)
    ref = cancel_unclaimed(store, reservation['reservation_id'], stop_evidence=stop)
    assert resolve_prelaunch_cancellation(store, ref)['attempt_id'] == attempt_id
    assert len([e for e in store.events() if e['event_kind'] == 'SETTLE']) == 1
