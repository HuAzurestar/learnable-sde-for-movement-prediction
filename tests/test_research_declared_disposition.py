"""Declared non-execution is retained shared metadata, never a free worker."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_disposition import declared_execution_disposition
from application.research_execution import execution_plan, resolve_execution
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.plugin import propagation_plugin
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_store import spec


def declaration(status="NOT_IMPLEMENTED"):
    return {"schema_version": "pirc25-execution-disposition-v1", "status": status,
            "reason": "no frozen implementation for this planned method"}


def prepared(tmp_path, *, marker=None):
    store = ResearchStore(tmp_path, "declared-unit", initialize=True)
    value = spec()
    cell = value["cells"][0]
    cell.update(plugin_id="unavailable", visibility="synthetic", execution_disposition=declaration() if marker is None else marker)
    store.register(value, digest(value))
    return store, value, cell


@pytest.mark.parametrize("status", ["NOT_IMPLEMENTED", "INELIGIBLE", "MISSING_INPUT", "UNQUALIFIED"])
def test_refusal_is_terminal_idempotent_and_does_not_create_a_worker_or_budget(tmp_path, status):
    store, value, cell = prepared(tmp_path, marker=declaration(status))
    runner = SharedRunner(store, CapabilityRegistry())
    first = runner.run_cell(value["study_id"], digest(cell))
    assert first["state"] == "PREFLIGHT_FAILED" and first["error_code"] == status
    assert not first["reused_preflight"] and "artifact_id" not in first
    assert store.attempts()[first["attempt_id"]]["started_at"] is None
    events = store.events()
    assert not {event["event_kind"] for event in events} & {"RESERVE", "QUEUED", "WORKER_STARTED", "SETTLE"}
    reopened = ResearchStore(tmp_path, "declared-unit")
    second = SharedRunner(reopened, CapabilityRegistry()).run_cell(value["study_id"], digest(cell))
    assert second == {**first, "reused_preflight": True}
    assert reopened.events() == events
    assert len(reopened.attempts()) == 1
    assert BudgetLedger(reopened).balance(cell["arm_id"])["committed_ms"] == 0
    with pytest.raises(ResearchError, match="cannot be retried"):
        runner.run_cell(value["study_id"], digest(cell), parent_attempt_id=first["attempt_id"], reason="new attempt")
    assert store.events() == events


def test_concurrent_metadata_refusal_has_exactly_one_attempt(tmp_path):
    store, value, cell = prepared(tmp_path)
    runner = SharedRunner(store, CapabilityRegistry())
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda _: runner.run_cell(value["study_id"], digest(cell)), range(8)))
    assert len({outcome["attempt_id"] for outcome in outcomes}) == len(store.attempts()) == 1
    assert sum(not outcome["reused_preflight"] for outcome in outcomes) == 1


@pytest.mark.parametrize("marker", [True, [], "missing", {},
    {**declaration(), "status": "SUCCEEDED"}, {**declaration(), "status": []},
    {**declaration(), "reason": ""}, {**declaration(), "reason": "unsafe\nlog"},
    {**declaration(), "reason": "x"*513}, {**declaration(), "budget_seconds": 999999},
    {**declaration(), "schema_version": "unknown"},
])
def test_malformed_declarations_refuse_before_any_attempt_or_launch(tmp_path, marker):
    store, value, cell = prepared(tmp_path, marker=marker)
    events = store.events()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        SharedRunner(store, CapabilityRegistry()).run_cell(value["study_id"], digest(cell))
    assert store.events() == events and not store.attempts()


def test_execution_resolution_and_direct_owner_plan_cannot_bypass_declared_refusal(tmp_path):
    store, value, cell = prepared(tmp_path, marker=declaration("INELIGIBLE"))
    registry = CapabilityRegistry()
    plugin = propagation_plugin()
    registry.register(plugin)
    with pytest.raises(ResearchError, match="INELIGIBLE"):
        resolve_execution(registry, cell)
    with pytest.raises(ResearchError, match="INELIGIBLE"):
        execution_plan(value, cell, plugin)
    # Attaching a binding is not permission to override the immutable refusal.
    altered = {**cell, "execution": {"component_id": plugin.plugin_id}}
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        resolve_execution(registry, altered)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        execution_plan(value, altered, plugin)
    assert not store.attempts()


def test_refusal_cannot_replace_an_existing_incompatible_attempt_history(tmp_path):
    store, value, cell = prepared(tmp_path)
    attempt = store.new_attempt(store.register_run(value["study_id"], cell))
    store.transition(attempt, "FAILED", error_code="OTHER_FAILURE")
    before = deepcopy(store.attempts())
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        SharedRunner(store, CapabilityRegistry()).run_cell(value["study_id"], digest(cell))
    assert store.attempts() == before


def test_declaration_accessor_returns_detached_data_and_ordinary_cells_remain_unchanged():
    marker = declaration()
    cell = {"execution_disposition": marker}
    detached = declared_execution_disposition(cell)
    detached["status"] = "changed"
    assert marker["status"] == "NOT_IMPLEMENTED"
    assert declared_execution_disposition({"plugin_id": "ordinary"}) is None


@pytest.mark.parametrize("status", ["NOT_IMPLEMENTED", "INELIGIBLE", "MISSING_INPUT", "UNQUALIFIED"])
def test_direct_budget_reservation_cannot_fund_a_declared_non_executable_cell(tmp_path, status):
    store, value, cell = prepared(tmp_path, marker=declaration(status))
    attempt = store.new_attempt(store.register_run(value["study_id"], cell))
    with pytest.raises(ResearchError, match=status):
        BudgetLedger(store).reserve(attempt, BudgetSpec(1))
    assert store.attempts()[attempt]["state"] == "PREFLIGHT_FAILED"
    assert store.attempts()[attempt]["error_code"] == status
    assert not any(event["event_kind"] == "RESERVE" for event in store.events())
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] == 0
