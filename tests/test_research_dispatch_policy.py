"""Isolated concurrency permission, write integrity and budget preservation."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_dispatch_policy import SCHEMA, activate_policy
from application.research_queue import ResearchQueue
from application.research_supervisor import ResearchSupervisor
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_research_queue import fixture


def policy(store, **changes):
    return {"schema_version": SCHEMA, "store_id": store.store_id, "decision": "approved",
            "human_issuer": "synthetic engineering fixture owner",
            "quoted_authorization": "Explicit fixture permission for independent compute claims; preserve all writes and budgets.",
            "resources": ["cpu"], "arm_prefixes": ["affine", "other"],
            "write_policy": "atomic-single-writer", "budget_policy_changed": False,
            "data_scope_changed": False, **changes}


def publish(store, value):
    name = "dispatch-policy-" + digest(value)
    store.publish(name, value)
    return {"manifest_id": name, "sha256": digest(value)}


def enable(store, **changes):
    reference = publish(store, policy(store, **changes))
    activate_policy(store, reference)
    return reference


def test_permission_does_not_overwrite_legacy_resources_or_reset_cost(tmp_path):
    store, ledger, queue, reservations = fixture(tmp_path)
    before = store.manifest("runtime-resources")
    queue.claim(reservations[0]["reservation_id"])
    balance = ledger.balance("affine")
    ref = enable(store)
    # Ignore both single-slot capacity and abandoned older FIFO requests.
    later = queue.claim(reservations[3]["reservation_id"])
    assert later["slot"] == 1 and later["dispatch_policy"] == ref
    assert queue.claim(reservations[1]["reservation_id"])["slot"] == 2
    assert store.manifest("runtime-resources") == before == {"slots": {"cpu": 1, "gpu": 0}}
    assert ledger.balance("affine") == balance
    assert not ledger._state()[0][reservations[0]["reservation_id"]]["settled"]


def test_unactivated_manifest_is_not_execution_permission(tmp_path):
    store, ledger, queue, reservations = fixture(tmp_path)
    publish(store, policy(store))
    queue.claim(reservations[0]["reservation_id"])
    assert queue.claim(reservations[2]["reservation_id"]) is None


def test_retroactive_publication_cannot_validate_a_forged_activation(tmp_path):
    store, _, queue, reservations = fixture(tmp_path)
    value = policy(store)
    reference = {"manifest_id": "dispatch-policy-" + digest(value), "sha256": digest(value)}
    store.append("DISPATCH_POLICY_ACTIVATED", {"reference": reference, "store_id": store.store_id}, "bad-order")
    store.publish(reference["manifest_id"], value)
    with pytest.raises(ResearchError, match="before activation"):
        queue.claim(reservations[0]["reservation_id"])


def test_forged_queued_arm_cannot_borrow_another_arms_permission(tmp_path):
    store, _, queue, reservations = fixture(tmp_path)
    enable(store, arm_prefixes=["affine"])
    before = store.events()
    forged = {**reservations[2], "arm_id": "affine"}
    with pytest.raises(ResearchError, match="identity"):
        queue.enqueue(forged)
    assert store.events() == before


def test_gpu_permission_still_requires_registered_resource_class(tmp_path):
    store, _, queue, reservations = fixture(tmp_path)
    enable(store, resources=["cpu", "gpu"])
    with pytest.raises(ResearchError, match="registered run"):
        queue.enqueue(reservations[0], "gpu")


def test_policy_activation_is_idempotent_and_claims_survive_restart(tmp_path):
    store, _, _, reservations = fixture(tmp_path)
    ref = enable(store)
    before = store.events()
    activate_policy(store, ref)
    assert store.events() == before
    queue = ResearchQueue(ResearchStore(tmp_path, "queue"))
    queue.claim(reservations[0]["reservation_id"])
    with pytest.raises(ResearchError, match="already claimed"):
        ResearchQueue(ResearchStore(tmp_path, "queue")).claim(reservations[0]["reservation_id"])
    assert queue.claim(reservations[2]["reservation_id"])["slot"] == 1


def test_parallel_distinct_claims_have_unique_tickets_and_one_winner_per_attempt(tmp_path):
    store, ledger, queue, reservations = fixture(tmp_path)
    enable(store)
    ids = [r["reservation_id"] for r in reservations]
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(queue.claim, ids))
    assert len({c["slot"] for c in claims}) == 4
    def duplicate(_):
        try:
            return queue.claim(ids[0])
        except ResearchError as exc:
            assert exc.code == "IDENTITY_CONFLICT"
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(c is None for c in pool.map(duplicate, range(4)))
    assert len([e for e in store.events() if e["event_kind"] == "QUEUE_CLAIMED"]) == 4
    assert ledger.balance("affine")["committed_ms"] == 10000
    assert ledger.balance("other")["committed_ms"] == 10000


def test_scope_and_resource_authority_do_not_leak(tmp_path):
    store, _, queue, reservations = fixture(tmp_path)
    enable(store, arm_prefixes=["affine"])
    queue.claim(reservations[0]["reservation_id"])
    assert queue.claim(reservations[1]["reservation_id"])["slot"] == 1
    assert queue.claim(reservations[2]["reservation_id"]) is None
    with pytest.raises(ResearchError, match="no configured slot"):
        queue.enqueue(reservations[3], "gpu")


@pytest.mark.parametrize("changes", [
    {"store_id": "foreign"}, {"decision": "pending"}, {"human_issuer": ""},
    {"quoted_authorization": ""}, {"resources": []}, {"resources": ["cpu", "cpu"]},
    {"resources": [{}]}, {"resources": ["disk"]}, {"arm_prefixes": [""]},
    {"arm_prefixes": []}, {"arm_prefixes": [True]}, {"write_policy": "no-lock"},
    {"budget_policy_changed": True}, {"budget_policy_changed": 0},
    {"data_scope_changed": True}, {"extra": "undeclared"},
])
def test_malformed_or_expanded_policy_cannot_activate(tmp_path, changes):
    store, _, _, _ = fixture(tmp_path)
    ref = publish(store, policy(store, **changes))
    before = store.events()
    with pytest.raises(ResearchError) as error:
        activate_policy(store, ref)
    assert error.value.code == "CONTRACT_MISMATCH"
    assert store.events() == before


def test_missing_hash_mismatch_or_corrupt_policy_never_opens_capacity(tmp_path):
    store, _, queue, reservations = fixture(tmp_path)
    with pytest.raises(ResearchError) as error:
        activate_policy(store, {"manifest_id": "unpublished", "sha256": digest("missing")})
    assert error.value.code == "MISSING_INPUT"
    ref = publish(store, policy(store))
    with pytest.raises(ResearchError) as error:
        activate_policy(store, {**ref, "sha256": "0" * 64})
    assert error.value.code == "CORRUPT_ARTIFACT"
    activate_policy(store, ref)
    (store.path / "manifests" / (ref["manifest_id"] + ".json")).write_text('{}', encoding="utf8")
    with pytest.raises(ResearchError) as error:
        queue.claim(reservations[0]["reservation_id"])
    assert error.value.code == "CORRUPT_ARTIFACT"


def test_recovery_hold_and_closed_arm_remain_vetoes(tmp_path):
    store, ledger, queue, reservations = fixture(tmp_path)
    enable(store)
    ledger.recover_unknown(reservations[0]["reservation_id"])
    with pytest.raises(ResearchError, match="closed"):
        queue.claim(reservations[1]["reservation_id"])
    hold = store.path / "recovery-hold.json"
    hold.write_text('{}', encoding="utf8")
    with pytest.raises(ResearchError, match="held"):
        queue.claim(reservations[2]["reservation_id"])
    with pytest.raises(ResearchError, match="hold"):
        activate_policy(store, publish(store, policy(store, arm_prefixes=["other"])))


def test_budget_cap_and_funded_arm_cannot_be_bypassed(tmp_path):
    store, ledger, queue, reservations = fixture(tmp_path)
    enable(store)
    forged = {**reservations[2], "arm_id": "affine"}
    before = store.events()
    with pytest.raises(ResearchError, match="identity"):
        queue.enqueue(forged)
    assert store.events() == before
    ledger.settle(reservations[0]["reservation_id"], 86400000, outcome="BUDGET_EXHAUSTED")
    with pytest.raises(ResearchError, match="closed"):
        queue.claim(reservations[1]["reservation_id"])
    assert queue.claim(reservations[2]["reservation_id"]) is not None


def test_final_fresh_scope_rechecks_closure_before_return(tmp_path, monkeypatch):
    store, _, queue, reservations = fixture(tmp_path)
    enable(store)
    original = store._append
    def append(kind, value, event_id=None):
        result = original(kind, value, event_id)
        if kind == "QUEUE_CLAIMED":
            original("ARM_CLOSED", {"arm_id": "affine", "reason": "fixture-race"}, "fixture-race")
        return result
    monkeypatch.setattr(store, "_append", append)
    with pytest.raises(ResearchError, match="held or closed"):
        queue.claim(reservations[0]["reservation_id"])


def test_native_two_supervised_workers_overlap_without_conflicting_result_writes(tmp_path):
    from tests.test_research_store import spec
    store = ResearchStore(tmp_path, "native-concurrent", initialize=True)
    value = spec()
    value["cells"] = [{"arm_id": "affine", "seed": i, "block_id": "fixture-" + str(i)} for i in range(2)]
    store.register(value, digest(value))
    queue = ResearchQueue(store)
    enable(store, arm_prefixes=["affine"])
    attempts = [store.new_attempt(store.register_run("synthetic", c)) for c in value["cells"]]
    script = (
        "import json,sys,time;from pathlib import Path;"
        "output,root,index=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3];"
        "(root/('ready-'+index)).write_text('ready',encoding='utf8');"
        "deadline=time.monotonic()+25;\n"
        "while not all((root/('ready-'+str(i))).exists() for i in range(2)) and time.monotonic()<deadline:time.sleep(.02)\n"
        "ok=all((root/('ready-'+str(i))).exists() for i in range(2));"
        "time.sleep(.2);output.write_text(json.dumps({'overlap':ok,'worker':index}),encoding='utf8')"
    )
    def execute(index):
        def command(output):
            return [sys.executable, "-B", "-c", script, str(output), str(tmp_path), str(index)]
        return ResearchSupervisor(store).run(attempts[index], command, BudgetSpec(120))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, range(2)))
    assert all(r["state"] == "SUCCEEDED" for r in results)
    for i, result in enumerate(results):
        # Isolated synthetic fixture integrity, not a production disclosure.
        metadata = store.manifest("artifact-" + result["artifact_id"])
        document = json.loads(store._verified_artifact_content(metadata))
        assert document == {"overlap": True, "worker": str(i)}
    claims = [e["payload"] for e in store.events() if e["event_kind"] == "QUEUE_CLAIMED"]
    assert {c["slot"] for c in claims} == {0, 1}
    settled, _ = BudgetLedger(store)._state()
    assert all(r["settled"] for r in settled.values())
    assert sum(r["charged_ms"] for r in settled.values()) > 0
