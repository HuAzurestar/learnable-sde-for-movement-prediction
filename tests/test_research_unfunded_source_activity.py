"""A registered attempt is not unused if any execution/exposure work exists."""

import pytest

from application.research_unfunded_source import supersede_unfunded_restart
from infrastructure.research_store import ResearchError
from tests.test_research_unfunded_source import proposal


@pytest.mark.parametrize('kind', [
    'READ_STARTED', 'READ_FAILED', 'CHECKPOINT_REQUESTED', 'CHECKPOINT_SAVED',
    'HEARTBEAT', 'UNKNOWN_ATTEMPT_ACTIVITY',
])
def test_activity_without_reservation_still_refuses_source_supersession(tmp_path, monkeypatch, kind):
    store, _, _, _, _, ref = proposal(tmp_path, monkeypatch)
    attempt_id = store.manifest(ref['manifest_id'])['retry_attempt_id']
    store.append(kind, {'attempt_id': attempt_id})
    before = store.events()
    with pytest.raises(ResearchError, match='funded or admitted|activity'):
        supersede_unfunded_restart(store, ref, source_code_hash='d' * 64,
            task_id='test-only', session_id='activity-refusal', reason='unused source correction')
    assert store.events() == before
    assert store.attempts()[attempt_id]['state'] == 'REGISTERED'


def test_unrelated_activity_is_not_borrowed_but_duplicate_own_terminal_is_refused(tmp_path, monkeypatch):
    store, _, _, _, _, ref = proposal(tmp_path, monkeypatch)
    attempt_id = store.manifest(ref['manifest_id'])['retry_attempt_id']
    store.append('READ_FAILED', {'attempt_id': 'another-attempt'})
    options = dict(source_code_hash='d' * 64, task_id='test-only', session_id='terminal-proof',
                   reason='unused source correction')
    supersession = supersede_unfunded_restart(store, ref, **options)
    assert supersede_unfunded_restart(store, ref, **options) == supersession
    store.append('ATTEMPT', store.attempts()[attempt_id])
    before = store.events()
    with pytest.raises(ResearchError, match='activity'):
        supersede_unfunded_restart(store, ref, **options)
    assert store.events() == before
