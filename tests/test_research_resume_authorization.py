"""Actual recovery command handoff must not reuse prepare's stale grant.

The owner-created checkpoint is a valid input to the actual registered native
worker. This permission test does not substitute for native checkpoint-save and
continuous-versus-resumed RNG qualification in test_research_live_checkpoint.
"""

from datetime import datetime, timezone
import json
import math
import random

import pytest

import application.research_data as data_module
import infrastructure.research_store as store_module
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_recovery import SharedRecovery
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_research_live_checkpoint import prepared


@pytest.fixture
def resume_clock(monkeypatch):
    expired = [False]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            # Execution's distinct2099 grant stays valid; the restore2080 grant
            # expires. Input authorization cannot replace resume authorization.
            return datetime(2090 if expired[0] else 2030, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(store_module, 'datetime', Clock)
    monkeypatch.setattr(data_module, 'datetime', Clock)
    return expired


def resume_source(tmp_path, level):
    store, value, registry, adapters, execution_grant = prepared(tmp_path / 'resume-source', level, 'resume-source')
    grant = {**execution_grant, 'authorization_id': 'restore-only',
             'expires_at': '2080-01-01T00:00:00+00:00', 'purposes': ['resume']}
    store.authorize(grant)
    run = store.register_run(value['study_id'], value['cells'][0])
    parent = store.new_attempt(run)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(parent, BudgetSpec(10))
    store.transition(parent, 'RUNNING')
    rng = random.Random(value['cells'][0]['seed'])
    accumulator = [0., 0., 0., 0.]
    for _ in range(200):
        for index in range(4):
            accumulator[index] += math.sqrt(0.01) * rng.gauss(0., 1.)
    state = {'step': 200, 'data_position': 200, 'method_state': {'accumulator': accumulator},
             'rng_state': {'python': json.loads(encode(rng.getstate())),
                           'stream_id': str(value['cells'][0]['seed'])}, 'chunk_complete': True}
    recovery = SharedRecovery(store, registry, adapters)
    checkpoint = recovery.checkpoint(parent, state)
    ledger.settle(reservation['reservation_id'], 6000, outcome='FAILED')
    store.transition(parent, 'FAILED', error_code='TRANSIENT')
    return store, recovery, parent, checkpoint, grant


@pytest.mark.parametrize('level', ['exact', 'chunk'])
def test_valid_resume_only_grant_runs_actual_registered_worker_and_preserves_cost(tmp_path, resume_clock, level):
    store, recovery, parent, checkpoint, grant = resume_source(tmp_path, level)
    result = recovery.resume(parent, checkpoint, authorization=grant, budget=BudgetSpec(6))
    assert result['state'] == 'SUCCEEDED', result
    output = json.loads((store.path / 'artifacts' / result['artifact_id']).read_bytes())
    assert output['forecast']['steps'] == 300
    assert store.attempts()[result['attempt_id']]['parent_attempt_id'] == parent
    assert BudgetLedger(store).balance('affine')['committed_ms'] > 6000


@pytest.mark.parametrize('level', ['exact', 'chunk'])
@pytest.mark.parametrize('phase', ['RESUME', 'ADMISSION'])
@pytest.mark.parametrize('fault', ['expiry', 'grant'])
def test_resume_rechecks_restore_grant_before_actual_command_handoff(tmp_path, monkeypatch, resume_clock,
                                                                   level, phase, fault):
    store, recovery, parent, checkpoint, grant = resume_source(tmp_path, level)
    original = store._append
    touched = []

    def append(kind, *args, **kwargs):
        result = original(kind, *args, **kwargs)
        if kind == phase:
            touched.append(True)
            if fault == 'expiry':
                resume_clock[0] = True
            else:
                path = store.path / 'manifests/authorization-restore-only.json'
                path.write_bytes(encode({**grant, 'purposes': []}))
        return result

    monkeypatch.setattr(store, '_append', append)
    try:
        result = recovery.resume(parent, checkpoint, authorization=grant, budget=BudgetSpec(6))
    except ResearchError as error:
        assert error.code == 'UNAUTHORIZED_DATA'
    else:
        pytest.fail('stale restore grant reached actual native recovery: ' + result['state'])
    assert touched
    attempts = store.attempts()
    child = next(attempt for attempt in attempts.values() if attempt.get('parent_attempt_id') == parent)
    assert child['state'] == 'PREFLIGHT_FAILED'
    assert child['error_code'] == 'UNAUTHORIZED_DATA'
    events = store.events()
    assert not any(event['event_kind'] == 'WORKER_STARTED' and
                   event['payload']['attempt_id'] == child['attempt_id'] for event in events)
    assert not (store.path / 'artifacts' / ('.attempt-' + child['attempt_id']) / 'resume-state.json').exists()
    assert BudgetLedger(store).balance('affine')['committed_ms'] == 6000
    assert not BudgetLedger(store).balance('affine')['closed']
    assert any(event['event_kind'] == 'EXPOSURE_DENIED' and
               event['payload'].get('authorization_hash') == digest(grant) for event in events)
