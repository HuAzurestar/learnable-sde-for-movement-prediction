"""One fresh checkpoint save context, without cached admission or permission."""

from copy import deepcopy

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_recovery import SharedRecovery
from infrastructure.research_store import ResearchError, digest
from tests.test_research_live_checkpoint import prepared


def owner(tmp_path, level):
    store, spec, registry, adapters, _ = prepared(tmp_path / "owner", level, "owner")
    cell = spec["cells"][0]
    plugin = registry.resolve(cell["plugin_id"], cell["capability"], version="1.0.0")
    attempt = store.new_attempt(store.register_run("owner", cell))
    BudgetLedger(store).reserve(attempt, BudgetSpec(3))
    receipt = AdmissionGate(store).prepare(spec, cell, plugin, attempt)
    store.transition(attempt, "RUNNING")
    recovery = SharedRecovery(store, registry, adapters)
    state = {"step": 1, "data_position": 1, "method_state": {},
             "rng_state": {"fixture": [1]}, "chunk_complete": True}
    progress = {"completed_steps": 1, "total_steps": 300,
                "throughput_per_second": 100, "eta_seconds": 2.99}
    return store, recovery, attempt, receipt, state, progress, plugin


@pytest.mark.parametrize("level", ["exact", "chunk"])
def test_actual_handler_builds_one_fresh_save_context(tmp_path, monkeypatch, level):
    store, recovery, attempt, receipt, state, progress, _ = owner(tmp_path, level)
    original, calls = recovery._context, []

    def context(identity):
        calls.append(identity)
        return original(identity)

    monkeypatch.setattr(recovery, "_context", context)
    reference = recovery.checkpoint_handler(attempt)(state, progress, receipt, None)
    assert calls == [attempt], "one save repeats complete version/resource/context validation"
    metadata = store.manifest("artifact-" + reference["artifact_id"])
    assert metadata["role"] == "checkpoint"
    assert reference["resume_level"] == level
    assert store.events()[-1]["event_kind"] == "CHECKPOINT"
    assert store._read_snapshot() is None


@pytest.mark.parametrize("change", ["attempt", "cell", "steps", "state-step", "stopped"])
def test_single_save_still_refuses_incompatible_actual_receipt(tmp_path, change):
    store, recovery, attempt, receipt, state, progress, _ = owner(tmp_path, "exact")
    receipt, state, progress = deepcopy(receipt), deepcopy(state), deepcopy(progress)
    if change == "attempt":
        receipt["attempt_id"] = "another-attempt"
    elif change == "cell":
        receipt["cell_hash"] = digest("another-cell")
    elif change == "steps":
        progress["total_steps"] = receipt["resource_plan"]["counts"]["steps"] + 1
    elif change == "state-step":
        state["step"] += 1
    else:
        store.transition(attempt, "FAILED", error_code="TRANSIENT")
    before = len(store.events())
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        recovery.checkpoint_handler(attempt)(state, progress, receipt, None)
    assert len(store.events()) == before, "refusal must precede artifact publication"
    assert store._read_snapshot() is None


@pytest.mark.parametrize("component", ["execution", "recovery"])
def test_each_save_rechecks_actual_registered_implementation(tmp_path, monkeypatch, component):
    store, recovery, attempt, receipt, state, progress, plugin = owner(tmp_path, "exact")
    save = recovery.checkpoint_handler(attempt)
    # Warm a real save, then mutate actual callable content without a new grant,
    # registration or ledger event. A phase-local refactor must not cache approval.
    assert save(state, progress, receipt, None)
    builder = (plugin.command_builder if component == "execution" else
               recovery.recovery.plugins[(plugin.plugin_id, "1.0.0")][0].command_builder)
    monkeypatch.setattr(builder, "__defaults__", ("changed-implementation",))
    before = len(store.events())
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT|CONTRACT_MISMATCH"):
        save(state, progress, receipt, None)
    assert len(store.events()) == before
    assert store._read_snapshot() is None
