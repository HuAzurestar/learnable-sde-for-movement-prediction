from concurrent.futures import ThreadPoolExecutor

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_queue import ResearchQueue
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_store import spec


def fixture(tmp_path):
    store = ResearchStore(tmp_path, "queue", initialize=True)
    value = spec()
    value["arms"].append({**value["arms"][0], "arm_id": "other", "method_family_id": "other"})
    value["cells"] = [{"arm_id": arm, "seed": i, "block_id": "fixture"}
                      for i, arm in enumerate(["affine", "affine", "other", "other"])]
    store.register(value, digest(value))
    ledger = BudgetLedger(store)
    reservations = [ledger.reserve(store.new_attempt(store.register_run("synthetic", cell)), BudgetSpec(5))
                    for cell in value["cells"]]
    queue = ResearchQueue(store)
    for reservation in reservations:
        queue.enqueue(reservation)
    return store, ledger, queue, reservations


def test_fixed_slots_fifo_and_arm_rotation_survive_restart(tmp_path):
    store, ledger, queue, reservations = fixture(tmp_path)
    ids = [r["reservation_id"] for r in reservations]
    assert queue.claim(ids[1]) is None
    assert queue.claim(ids[0])["slot"] == 0
    assert queue.claim(ids[2]) is None
    with pytest.raises(ResearchError, match="already claimed"):
        ResearchQueue(ResearchStore(tmp_path, "queue")).claim(ids[0])
    ledger.settle(ids[0], 20, outcome="SUCCEEDED")
    assert queue.claim(ids[1]) is None  # another arm receives its turn
    assert queue.claim(ids[2])["slot"] == 0
    ledger.settle(ids[2], 20, outcome="SUCCEEDED")
    assert queue.claim(ids[1])["slot"] == 0
    with pytest.raises(ResearchError, match="immutable"):
        ResearchQueue(store, slots={"cpu": 2, "gpu": 0})
    with pytest.raises(ResearchError, match="no configured slot"):
        queue.enqueue(reservations[3], "gpu")


def test_concurrent_duplicate_claim_cannot_launch_twice(tmp_path):
    store, ledger, queue, reservations = fixture(tmp_path)
    def claim(_):
        try:
            return queue.claim(reservations[0]["reservation_id"])
        except ResearchError as exc:
            assert exc.code == "IDENTITY_CONFLICT"
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(value is not None for value in pool.map(claim, range(4))) == 1
    # Unknown worker recovery retains the full charge and releases its slot,
    # but never schedules the closed arm's remaining cells.
    ledger.recover_unknown(reservations[0]["reservation_id"])
    assert queue.claim(reservations[2]["reservation_id"])["slot"] == 0
    with pytest.raises(ResearchError, match="closed"):
        queue.claim(reservations[1]["reservation_id"])
