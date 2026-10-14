"""An occupied slot must not run source adjudication under the writer lock."""

import subprocess
import sys

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_queue import ResearchQueue
from infrastructure.research_store import digest
from tests.test_research_resource_budget import registered
from tests.test_research_store import spec


def test_occupied_slot_defers_resource_adjudication_and_allows_peer_settlement(tmp_path, monkeypatch):
    store, attempts, budget = registered(tmp_path, 1)
    peer = spec()
    peer['study_id'] = 'peer'
    peer['arms'][0]['method_family_id'] = 'peer-exact'
    store.register(peer, digest(peer))
    ledger = BudgetLedger(store)
    peer_attempt = store.new_attempt(store.register_run('peer', peer['cells'][0]))
    peer_reservation = ledger.reserve(peer_attempt, BudgetSpec(5))
    queue = ResearchQueue(store)
    queue.enqueue(peer_reservation)
    assert queue.claim(peer_reservation['reservation_id'])['slot'] == 0

    own = ledger.reserve(attempts[0], budget)
    queue.enqueue(own)
    store.append('ARM_CLOSED', {'arm_id': 'pirc28-k2', 'reason': 'INTERRUPTED'})

    def full_source_check(*args, **kwargs):
        raise AssertionError('complete source adjudication while every slot is occupied')

    monkeypatch.setattr('application.research_restart_adjudication.resource_arm_continuation',
                        full_source_check)
    before = store.events()
    assert queue.claim(own['reservation_id']) is None
    assert store.events() == before
    # A separate Python process must still be able to acquire the actual OS
    # writer lock and settle its own claim, using the original lock timeout.
    writer = subprocess.run([sys.executable, '-c',
        'from pathlib import Path; import sys; '
        'from infrastructure.research_store import ResearchStore; '
        'from application.research_budget import BudgetLedger; '
        's=ResearchStore(Path(sys.argv[1]),sys.argv[2]); '
        'BudgetLedger(s).settle(sys.argv[3],17,outcome="SUCCEEDED")',
        str(tmp_path), store.store_id, peer_reservation['reservation_id']],
        capture_output=True, text=True, timeout=25)
    assert writer.returncode == 0, writer.stderr
    assert ledger._state()[0][peer_reservation['reservation_id']]['charged_ms'] == 17
    assert not ledger._state()[0][own['reservation_id']]['settled']
    # Capacity becoming available restores the complete eligibility check;
    # waiting must not become a reusable authorization to dispatch.
    import pytest
    with pytest.raises(AssertionError, match='complete source adjudication'):
        queue.claim(own['reservation_id'])
