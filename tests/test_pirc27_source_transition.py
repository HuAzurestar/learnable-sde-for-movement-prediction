"""Disposable synthetic metadata controls, never the shared research store."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from application.pirc27_source_transition import numerical_identity
from application.research_budget import BudgetLedger, BudgetSpec
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_continuation import approved
from tests.test_research_store import spec


def batch_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pirc27_first_batch.py"
    loader = importlib.util.spec_from_file_location("pirc27_batch_transition_test", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def configuration():
    batch = batch_module()
    arms = {m: {"arm_id": "pirc27-affine-" + m + "-mean-x", "model_family_id": batch.FAMILY,
            "method_family_id": m, "objective_id": "endpoint-x", "budget_seconds": 86400}
            for m in ("exact", "gaussian")}
    return arms, batch.frozen_cells(arms)[0]


@pytest.fixture
def bridge(tmp_path, configuration):
    arms, frozen = configuration
    target_cell = deepcopy(frozen)
    source_cell = deepcopy(target_cell)
    source_cell["affine_qualification_policy"]["code_hash"] = digest("old")
    source_cell["execution"]["config"]["qualification_policy_hash"] = digest(source_cell["affine_qualification_policy"])
    store = ResearchStore(tmp_path, "pirc27-transition-test", initialize=True)
    runs = []
    for study, cell, source_hash in (("old-pilot", source_cell, digest("old")),
                                     ("new-pilot", target_cell, target_cell["affine_qualification_policy"]["code_hash"])):
        value = {**spec(), "study_id": study, "experiment_id": study, "arms": [arms["exact"]],
                 "cells": [cell], "code_hash": source_hash, "admission": {"mode": "pilot"}}
        store.register(value, digest(value))
        runs.append(store.register_run(study, cell))
    parent = store.new_attempt(runs[0])
    store.transition(parent, "PREFLIGHT_FAILED", error_code="QUEUE_REJECTED")
    approval_ref = approved(store, arm_ids=[arms["exact"]["arm_id"]],
                            pirc27_total_slot_seconds=14400, infrastructure_retries=True)
    value = {"schema_version": "pirc27-source-transition-v1", "source_run_id": runs[0],
             "target_run_id": runs[1], "source_parent_attempt_id": parent,
             "source_code_hash": digest("old"), "target_code_hash": target_cell["affine_qualification_policy"]["code_hash"],
             "approval": approval_ref}
    reference = {"manifest_id": "pirc27-transition-" + digest(value), "sha256": digest(value)}
    store.publish(reference["manifest_id"], value)
    return store, runs, parent, reference


def test_explicit_cross_source_parent_keeps_history_and_count_across_restart(bridge):
    store, runs, parent, reference = bridge
    before = store.events()
    child = store.new_attempt(runs[1], parent_attempt_id=parent, reason="authorized infrastructure fix",
                              source_transition_ref=reference)
    assert store.attempts()[child]["parent_attempt_id"] == parent
    assert store.attempts()[child]["source_transition_ref"] == reference
    assert store.events()[:len(before)] == before
    store.transition(child, "PREFLIGHT_FAILED", error_code="WRITER_BUSY")
    reopened = ResearchStore(store.path.parent, store.store_id)
    # Even omitting the ref cannot reset the cross-source counter.
    final = reopened.new_attempt(runs[1], parent_attempt_id=child, reason="explicit second retry")
    reopened.transition(final, "PREFLIGHT_FAILED", error_code="WRITER_BUSY")
    with pytest.raises(ResearchError, match="RETRY_EXHAUSTED"):
        reopened.new_attempt(runs[1], parent_attempt_id=final, reason="not permitted")
    assert len(reopened.attempts()) == 3


@pytest.mark.parametrize("fault", ["ref", "reason", "parent", "live", "scientific", "scope", "expiry", "cost", "stop", "seed", "threshold"])
def test_refuses_unsafe_source_restart_without_creating_attempt(bridge, fault):
    store, runs, parent, reference = bridge
    reference = deepcopy(reference)
    reason = "explicit fix"
    chosen_parent = parent
    if fault == "ref":
        reference["sha256"] = digest("wrong")
    if fault == "reason":
        reason = None
    if fault == "parent":
        chosen_parent = "absent"
    if fault in {"live", "scientific", "cost", "stop"}:
        # A new explicit same-run attempt, preserving the prior failed one.
        chosen_parent = store.new_attempt(runs[0], parent_attempt_id=parent, reason="fixture")
        if fault in {"cost", "stop"}:
            reserved = BudgetLedger(store).reserve(chosen_parent, BudgetSpec(1))
            if fault == "stop":
                BudgetLedger(store).settle(reserved["reservation_id"], 10, outcome="INTERRUPTED")
                store.append("WORKER_STARTED", {"attempt_id": chosen_parent, "reservation_id": reserved["reservation_id"], "pid": 123})
        if fault != "live":
            store.transition(chosen_parent, "PREFLIGHT_FAILED", error_code="NONFINITE" if fault == "scientific" else "WRITER_BUSY")
    if fault in {"scope", "expiry"}:
        value = store.manifest(reference["manifest_id"])
        value["approval"] = approved(store, arm_ids=["other"] if fault == "scope" else ["pirc27-affine-exact-mean-x"],
            pirc27_total_slot_seconds=14400, infrastructure_retries=fault != "scope",
            **({"work_started_at": "2020-01-01T00:00:00+00:00", "work_deadline": "2020-01-01T01:00:00+00:00",
                "authorization_expires_at": "2020-01-01T02:00:00+00:00"} if fault == "expiry" else {}))
        reference = {"manifest_id": "changed-transition-" + digest(value), "sha256": digest(value)}
        store.publish(reference["manifest_id"], value)
    if fault in {"seed", "threshold"}:
        target = store.manifest("study-new-pilot")["spec"]
        target["study_id"] = "changed-pilot"
        target["cells"] = deepcopy(target["cells"])
        cell = target["cells"][0]
        if fault == "seed":
            cell["seed"] += 1
        else:
            cell["affine_qualification_policy"]["maximum_time_bias"] = 10
            cell["execution"]["config"]["qualification_policy_hash"] = digest(cell["affine_qualification_policy"])
        store.register(target, digest(target))
        runs[1] = store.register_run(target["study_id"], cell)
        value = store.manifest(reference["manifest_id"])
        value["target_run_id"] = runs[1]
        reference = {"manifest_id": "changed-transition-" + digest(value), "sha256": digest(value)}
        store.publish(reference["manifest_id"], value)
    count = len(store.attempts())
    with pytest.raises(ResearchError):
        store.new_attempt(runs[1], parent_attempt_id=chosen_parent, reason=reason,
                          source_transition_ref=reference)
    assert len(store.attempts()) == count


def test_cross_run_parent_without_explicit_transition_is_still_refused(bridge):
    store, runs, parent, _ = bridge
    with pytest.raises(ResearchError, match="retry parent not found"):
        store.new_attempt(runs[1], parent_attempt_id=parent, reason="no evidence")


def test_identity_preserves_request_and_threshold_and_ignores_only_source(bridge):
    store, runs, _, _ = bridge
    old, new = [store.manifest("run-" + r)["cell"] for r in runs]
    assert numerical_identity(old) == numerical_identity(new)
    new["propagation_request"]["steps"] *= 2
    assert numerical_identity(old) != numerical_identity(new)


def test_continuation_freezes_seven_cells_and_restores_only_original_costs(tmp_path, configuration, monkeypatch):
    import sys
    batch = batch_module()
    monkeypatch.syspath_prepend(str(Path(batch.__file__).parent))
    import pirc27_affine_continuation as continuation
    arms, _ = configuration
    cells = batch.frozen_cells(arms)
    store = ResearchStore(tmp_path, "pirc27-continuation-control", initialize=True)
    with store._read_transaction():
        store.publish(batch.ORIGINAL, {"arms": list(arms.values())})
        for study, matrix in ((continuation.original.STUDY, cells),
                              (continuation.supplement.STUDY, [cells[5], cells[7]])):
            value = {**spec(), "study_id": study, "experiment_id": study, "arms": list(arms.values()),
                     "cells": matrix, "code_hash": cells[0]["affine_qualification_policy"]["code_hash"],
                     "admission": {"mode": "pilot"}}
            store.register(value, digest(value))
        ledger = BudgetLedger(store)
        exact = store.new_attempt(store.register_run(continuation.original.STUDY, cells[0]))
        reserved = ledger.reserve(exact, BudgetSpec(1))
        ledger.settle(reserved["reservation_id"], 10, outcome="INTERRUPTED")
        store.transition(exact, "INTERRUPTED", error_code="SUPERVISOR_FINALIZATION_UNPROVEN")
        stable = store.new_attempt(store.register_run(continuation.original.STUDY, cells[1]))
        reserved = ledger.reserve(stable, BudgetSpec(1))
        store.transition(stable, "RUNNING")
        # Metadata-only fixture result, not numerical or admission qualification.
        artifact = store.artifact(b'{"fixture":"metadata-only"}', role="result", visibility="synthetic",
                                  block_ids=[batch.FAMILY], study_id=continuation.original.STUDY)
        ledger.settle(reserved["reservation_id"], 20, outcome="SUCCEEDED")
        store.transition(stable, "SUCCEEDED", artifact_id=artifact["artifact_id"])
        store.append("ARM_CLOSED", {"arm_id": arms["gaussian"]["arm_id"], "reason": "fixture closure"})
    result = continuation.prepare(store)
    assert result["cells"] == 7 and result["charged_ms"] == 30 and result["reserved_ms"] == 0
    frozen = store.manifest("study-" + continuation.STUDY)["spec"]
    assert frozen["arms"] == list(arms.values())
    assert frozen["runtime_binding"] == {"root": str(store.path.parent), "store_id": store.store_id}
    assert frozen["scientific_independent_n"] == 0
    assert {numerical_identity(c)["comparison_dimensions"]["model_configuration"] for c in frozen["cells"]} == {
        "stable-v1", "near-critical-v1", "anisotropic-v1", "uncertain-initial-v1"}
    assert not any(c["execution"]["config"]["method"] == "gaussian" and
                   c["comparison_dimensions"]["model_configuration"] == "stable-v1" for c in frozen["cells"])
    assert sum(ledger.balance(a["arm_id"])["committed_ms"] for a in arms.values()) == 30
    assert all(not ledger.balance(a["arm_id"])["closed"] for a in arms.values())
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())
    # Actual supervised synthetic control of the complete new operator path.
    # This isolated fixture has no shared-store connection or research data.
    outcome = continuation.run_next(store)
    assert outcome["state"] == "SUCCEEDED"
    attempts = store.attempts()
    child = next(a for a in attempts.values() if a["parent_attempt_id"] == exact)
    assert child["state"] == "SUCCEEDED" and child["source_transition_ref"]
    assert continuation.accounting(store)["charged_ms"] > 30
