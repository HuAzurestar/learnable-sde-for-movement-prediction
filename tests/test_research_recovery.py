"""Recovery bindings, actual RNG payload and non-resetting budget contracts."""

from dataclasses import replace
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry, ExecutionPlugin
from application.research_recovery import RecoveryPlugin, RecoveryRegistry, SharedRecovery
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_research_store import spec
from tests.research_admission_fixtures import synthetic_plugin, bind_fixture_execution


def setup(tmp_path, level="exact"):
    store = ResearchStore(tmp_path, "recovery", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="fixture", capability="generic-rollout", visibility="synthetic")
    execution = CapabilityRegistry()
    plugin = synthetic_plugin("fixture", frozenset({"generic-rollout"}), ("x", "y", "vx", "vy"),
                             ("m", "m", "m/s", "m/s"), level, lambda *args: [])
    execution.register(plugin)
    bind_fixture_execution(value, plugin)
    store.register(value, digest(value))
    run = store.register_run("synthetic", value["cells"][0])
    attempt = store.new_attempt(run)
    adapters = RecoveryRegistry()
    if level != "restart-only":
        adapters.register(RecoveryPlugin("fixture", level, lambda *args: [], "1.0.0"))
    recovery = SharedRecovery(store, execution, adapters)
    authorization = {"authorization_id": "recovery", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
                     "evidence_hash": digest("fixture permission"), "purposes": ["resume"],
                     "visibilities": ["synthetic"], "block_ids": ["fixture-1"]}
    store.authorize(authorization)
    return store, recovery, attempt, authorization


@pytest.mark.parametrize("level", ["exact", "chunk"])
def test_declared_recovery_roundtrip_preserves_state_and_spent_budget(tmp_path, level):
    store, recovery, attempt, grant = setup(tmp_path, level)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempt, BudgetSpec(10))
    store.transition(attempt, "RUNNING")
    state = {"step": 4, "data_position": 12, "method_state": {"weights": [1, 2]},
             "rng_state": {"numpy": [4, 5], "brownian_id": "fixture-stream"}, "chunk_complete": True}
    checkpoint = recovery.checkpoint(attempt, state)
    ledger.settle(reservation["reservation_id"], 6000, outcome="FAILED")
    store.transition(attempt, "FAILED", error_code="TRANSIENT")
    reopened = ResearchStore(tmp_path, "recovery")
    recovery.store = reopened
    prepared = recovery.prepare(attempt, checkpoint, authorization=grant)
    assert prepared["state"] == state
    assert BudgetLedger(reopened).balance("affine")["committed_ms"] == 6000


def test_restart_only_refuses_fictitious_exact_checkpoint(tmp_path):
    store, recovery, attempt, grant = setup(tmp_path, "restart-only")
    with pytest.raises(ResearchError, match="does not implement"):
        recovery.checkpoint(attempt, {})


@pytest.mark.parametrize("field", ["code_hash", "data_hash", "model_hash", "objective_hash", "protocol_hash", "cell_hash"])
def test_incompatible_checkpoint_rejected_before_retry(tmp_path, field):
    store, recovery, attempt, grant = setup(tmp_path)
    state = {"step": 1, "data_position": 1, "method_state": {}, "rng_state": {"state": [1]}}
    checkpoint = recovery.checkpoint(attempt, state)
    value = json.loads((store.path / "artifacts" / checkpoint).read_bytes())
    value["bindings"][field] = "0" * 64
    changed = store.artifact(encode(value), role="checkpoint", visibility="synthetic", block_ids=["fixture-1"], study_id="synthetic")
    store.transition(attempt, "FAILED", error_code="TRANSIENT")
    with pytest.raises(ResearchError, match="hashes differ"):
        recovery.prepare(attempt, changed["artifact_id"], authorization=grant)
    assert len(store.attempts()) == 1


def test_chunk_boundary_and_budget_rollback_are_rejected(tmp_path):
    store, recovery, attempt, grant = setup(tmp_path, "chunk")
    state = {"step": 1, "data_position": 1, "method_state": {}, "rng_state": {}}
    with pytest.raises(ResearchError, match="completed boundary"):
        recovery.checkpoint(attempt, state)
    with pytest.raises(ResearchError, match="restore budget"):
        recovery.checkpoint(attempt, {**state, "chunk_complete": True, "budget": 86400})
