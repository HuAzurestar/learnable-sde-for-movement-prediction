"""Actual shared admission must reject incomplete/versionless resource bindings.

These tests deliberately exercise a real registered synthetic package and its
worker, not a stand-in registry. Core-only checks cannot satisfy this boundary.
"""

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, digest
from tests.test_research_admission_chain import prepared


@pytest.mark.parametrize("fault", ["missing-binding", "unknown-version", "over-global-quota"])
def test_actual_runner_requires_versioned_resource_preflight_before_input_or_worker(tmp_path, fault):
    store, value, registry, _ = prepared(tmp_path)
    cell = value["cells"][0]
    if fault != "missing-binding":
        cell["execution"] = {"schema_version": "pirc25-execution-binding-v1",
            "component_id": cell["plugin_id"], "component_version": "unknown-version" if fault == "unknown-version" else "1.0.0",
            "registry_entry_hash": digest("unregistered-entry"),
            "config": {"paths": 2_000_000 if fault == "over-global-quota" else 1, "steps": 1},
            "inputs": {"observations": 1}, "resource_plan_hash": digest("unverified-plan")}
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH|RESOURCE_PLAN_REJECTED"):
        observed = SharedRunner(store, registry).run_cell(value["study_id"], digest(cell), budget=BudgetSpec(10))
        kinds = [event["event_kind"] for event in store.events()]
        pytest.fail("invalid execution binding was not denied: state=" + str(observed.get("state")) +
            ", input_reads=" + str(kinds.count("READ_STARTED")) + ", workers=" + str(kinds.count("WORKER_STARTED")))
    assert not any(event["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for event in store.events()), \
        "version/resource denial must precede protected-input reads and worker startup"
    assert not any(attempt["state"] == "SUCCEEDED" for attempt in store.attempts().values())
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] == 0
