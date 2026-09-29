"""A new fixture method reuses the supervisor, result contract and recovery."""

import sys

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry, ExecutionPlugin
from application.research_recovery import RecoveryPlugin, RecoveryRegistry, SharedRecovery
from infrastructure.research_store import ResearchStore, digest, encode
from tests.test_research_store import spec


def test_fixture_method_exact_state_recovery_uses_existing_supervisor(tmp_path):
    store = ResearchStore(tmp_path, "onboarding", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="counter-fixture", capability="generic-rollout", visibility="synthetic")
    cell = value["cells"][0]

    def command(output, spec_value, cell_value, state):
        payload = {"metrics": {"counter": state["method_state"]["counter"]}, "forecast": {},
                   "fit": {"fixture_only": True}, "source_schema": "counter-fixture-v1"}
        result = {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED", "spec_hash": digest(spec_value),
                  "cell_hash": digest(cell_value), "protocol_hash": spec_value["protocol_hash"],
                  "input_hash": spec_value["data_hash"], "output_hash": digest(payload),
                  "state_order": ["x", "y", "vx", "vy"], "units": ["m", "m", "m/s", "m/s"],
                  "resume_level": "exact", "qualification": "fixture", "metric_units": {"counter": "count"}, **payload}
        return [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
                str(output), encode(result).decode()]

    execution = CapabilityRegistry()
    execution.register(ExecutionPlugin("counter-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "exact", command))
    from tests.research_admission_fixtures import admit_fixture
    admit_fixture(store, value, execution.resolve("counter-fixture", "generic-rollout"), tmp_path)
    store.register(value, digest(value))
    recovery_registry = RecoveryRegistry()
    recovery_registry.register(RecoveryPlugin("counter-fixture", "exact", command))
    recovery = SharedRecovery(store, execution, recovery_registry)
    run = store.register_run("synthetic", cell)
    parent = store.new_attempt(run)
    reservation = BudgetLedger(store).reserve(parent, BudgetSpec(1))
    store.transition(parent, "RUNNING")
    checkpoint = recovery.checkpoint(parent, {"step": 3, "data_position": 3,
        "method_state": {"counter": 7}, "rng_state": {"deterministic_fixture": 19}})
    BudgetLedger(store).settle(reservation["reservation_id"], 400, outcome="FAILED")
    store.transition(parent, "FAILED", error_code="TRANSIENT")
    grant = {"authorization_id": "resume", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
             "evidence_hash": digest("fixture"), "purposes": ["resume"], "visibilities": ["synthetic"], "block_ids": ["fixture-1"]}
    store.authorize(grant)
    result = recovery.resume(parent, checkpoint, authorization=grant, budget=BudgetSpec(10))
    assert result["state"] == "SUCCEEDED"
    child = store.attempts()[result["attempt_id"]]
    assert child["parent_attempt_id"] == parent
    assert BudgetLedger(store).balance("affine")["committed_ms"] >= 400 + result["elapsed_ms"]
