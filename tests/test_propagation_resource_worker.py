"""Real isolated workers, never the shared research store or dataset grants."""
import json

import pytest

from application.propagation_resources import validate_packet
from application.research_admission import AdmissionGate
from application.research_execution import resolve_execution
from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import digest, encode, ResearchError
from tests.test_propagation_shared_adapter import prepare


@pytest.mark.parametrize("method", ["euler", "exact"])
def test_real_owner_worker_records_its_own_bound_observation_without_changing_cost(tmp_path, method):
    store, spec, registry = prepare(tmp_path, method, resource_observation=True)
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60))
    assert outcome["state"] == "SUCCEEDED", outcome
    grant = store.authorization(spec["admission"]["authorization_id"])
    result = json.loads(store.read_artifact(outcome["artifact_id"], purpose="evaluate", authorization=grant))
    packet = result["resource_observation"]
    validate_packet(packet, spec, cell, outcome["attempt_id"], execution_state="SUCCEEDED")
    assert packet["memory"]["status"] == "MEASURED"
    assert packet["resumed"] is False and packet["duration"]["wall_ns"] > 0
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] == outcome["elapsed_ms"] > 0
    before = BudgetLedger(store).balance(cell["arm_id"])
    reused = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell))
    assert reused["reused"] and reused["artifact_id"] == outcome["artifact_id"]
    assert BudgetLedger(store).balance(cell["arm_id"]) == before
    result["resource_observation"]["attempt_id"] = "b"*32
    receipt = store.manifest("admission-"+result["admission_hash"])
    with pytest.raises(ResearchError, match="binding differs"):
        AdmissionGate(store).result_validator(receipt, spec, cell, resolve_execution(registry, cell))(result)


def test_invalid_resource_policy_refuses_before_reserving_or_starting_worker(tmp_path):
    store, spec, registry = prepare(tmp_path, resource_observation=True)
    cell = spec["cells"][0]
    cell["resource_observation_policy"]["required"] = 1
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60))
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED"} for e in store.events())
    assert next(iter(store.attempts().values()))["state"] == "PREFLIGHT_FAILED"


def test_owner_refuses_missing_packet_before_result_publication(tmp_path):
    from application.propagation_execution import execute_propagation
    store, spec, registry = prepare(tmp_path, resource_observation=True)
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(spec["study_id"], cell))
    plugin = resolve_execution(registry, cell)
    receipt = AdmissionGate(store).prepare(spec, cell, plugin, attempt)
    # Match the real worker's canonical JSON boundary, not Python tuple values.
    result = json.loads(encode(execute_propagation(spec, cell)))  # Isolated kernel control only.
    with pytest.raises(ResearchError, match="packet fields differ"):
        AdmissionGate(store).result_validator(receipt, spec, cell, plugin)(result)
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED"} for e in store.events())
