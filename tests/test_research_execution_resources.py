"""Actual shared admission must reject incomplete/versionless resource bindings.

These tests deliberately exercise a real registered synthetic package and its
worker, not a stand-in registry. Core-only checks cannot satisfy this boundary.
"""

import pytest
from copy import deepcopy
from dataclasses import replace
import json

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, digest
from tests.test_research_admission_chain import prepared
from tests.test_research_admission_chain import fixture_command
from tests.research_admission_fixtures import admit_fixture, synthetic_plugin
from tests.test_research_store import spec
from application.research_contracts import CapabilityRegistry
from infrastructure.research_store import ResearchStore
from infrastructure.research_store import encode


def resource_fixture(tmp_path, *, result_bytes=4096):
    store = ResearchStore(tmp_path, "resource-fixture", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="resource-fixture", capability="generic-rollout", visibility="synthetic")
    plugin = synthetic_plugin("resource-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", fixture_command)
    entry = plugin.registry_entry
    entry.config_schema.update(properties={"paths": {"type": "integer", "minimum": 1, "maximum": 1000000},
        "steps": {"type": "integer", "minimum": 1, "maximum": 1000000},
        "dtype": {"type": "string", "enum": ["float64"]}, "device": {"type": "string", "enum": ["cpu"]},
        "noise_dim": {"type": "integer", "enum": [0]}}, required=["paths", "steps", "dtype", "device", "noise_dim"])
    entry.input_schema.update(properties={"observations": {"type": "integer", "minimum": 1},
        "state_dim": {"type": "integer", "enum": [4]}}, required=["observations", "state_dim"])
    entry.resource_contract["counts"].update(paths={"config": ["paths"]}, steps={"config": ["steps"]},
        observations={"input": ["observations"]})
    entry.resource_contract["tensors"] = [{"name": "declared_sample_upper_bound", "axes": ["paths", "steps", "state_dim"], "item_bytes": 8}]
    if result_bytes == "exact-worker-output":
        # Digests always occupy 64 bytes; adding the eventual execution binding
        # changes their content, not this deterministic fixture output length.
        result_bytes = len(encode(json.loads(fixture_command("unused", {**value,
            "admission": {"mode": "fixture"}}, value["cells"][0])[-1])))
    entry.resource_contract["limits"].update(paths=4, steps=4, observations=4, matrix_cells=2,
        tensor_elements=16, tensor_bytes=128, result_bytes=result_bytes)
    registry = CapabilityRegistry()
    registry.register(plugin)
    grant = admit_fixture(store, value, plugin, tmp_path,
        execution_config={"paths": 1, "steps": 1, "dtype": "float64", "device": "cpu", "noise_dim": 0},
        execution_inputs={"observations": 1, "state_dim": 4})
    return store, value, registry, grant


@pytest.mark.parametrize("fault", ["missing-binding", "unknown-version", "over-global-quota"])
def test_actual_runner_requires_versioned_resource_preflight_before_input_or_worker(tmp_path, fault):
    store, value, registry, _ = prepared(tmp_path)
    cell = value["cells"][0]
    if fault == "missing-binding":
        cell.pop("execution")
    else:
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


def test_valid_versioned_execution_receipt_binds_plan_and_reuse_without_new_charge(tmp_path):
    store, value, registry, grant = resource_fixture(tmp_path)
    store.register(value, digest(value))
    cell = value["cells"][0]
    runner = SharedRunner(store, registry)
    result = runner.run_cell("synthetic", digest(cell), budget=BudgetSpec(10))
    assert result["state"] == "SUCCEEDED"
    content = json.loads(store.read_artifact(result["artifact_id"], purpose="preview", authorization=grant))
    receipt = store.manifest("admission-" + content["admission_hash"])
    assert receipt["resource_plan"]["resource_plan_hash"] == cell["execution"]["resource_plan_hash"]
    assert receipt["registry_entry"]["version"] == cell["execution"]["component_version"]
    assert receipt["resource_plan"]["tensor_bytes"] == 32
    assert store.manifest("registry-entry-" + cell["execution"]["registry_entry_hash"]) == receipt["registry_entry"]
    balance = BudgetLedger(store).balance(cell["arm_id"])
    assert runner.run_cell("synthetic", digest(cell))["reused"]
    assert BudgetLedger(store).balance(cell["arm_id"]) == balance


@pytest.mark.parametrize("fault", ["path", "step", "observation", "tensor", "matrix", "dtype", "device", "state-shape", "noise-support", "changed-plan"])
def test_valid_identity_cannot_bypass_actual_resource_or_compatibility_gate(tmp_path, fault):
    store, value, registry, _ = resource_fixture(tmp_path)
    cell = value["cells"][0]
    binding = cell["execution"]
    if fault in {"path", "step"}:
        binding["config"]["paths" if fault == "path" else "steps"] = 5
    elif fault == "observation":
        binding["inputs"]["observations"] = 5
    elif fault == "tensor":
        binding["config"].update(paths=4, steps=4)
    elif fault == "matrix":
        value["cells"].extend([{**deepcopy(cell), "seed": seed} for seed in (2, 3)])
    elif fault in {"dtype", "device", "noise-support"}:
        key = "noise_dim" if fault == "noise-support" else fault
        binding["config"][key] = {"dtype": "float32", "device": "cuda", "noise_dim": 2}[key]
    elif fault == "state-shape":
        binding["inputs"]["state_dim"] = 2
    else:
        binding["resource_plan_hash"] = "0" * 64
    store.register(value, digest(value))
    code = "RESOURCE_PLAN_REJECTED" if fault in {"path", "step", "observation", "tensor", "matrix"} else "CONTRACT_MISMATCH"
    with pytest.raises(ResearchError, match=code):
        SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))
    assert not any(event["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for event in store.events())
    assert list(store.attempts().values())[0]["state"] == "PREFLIGHT_FAILED"
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] == 0


def test_admitted_output_byte_quota_is_enforced_before_parse_or_success(tmp_path):
    store, value, registry, _ = resource_fixture(tmp_path, result_bytes=64)
    store.register(value, digest(value))
    cell = value["cells"][0]
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))
    assert any(event["event_kind"] == "WORKER_STARTED" for event in store.events())
    assert not any(attempt["state"] == "SUCCEEDED" for attempt in store.attempts().values())
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] > 0


def test_direct_admission_cannot_bypass_runner_version_or_resource_checks(tmp_path):
    from application.research_admission import AdmissionGate
    store, value, registry, _ = resource_fixture(tmp_path)
    cell = value["cells"][0]
    cell["execution"]["config"]["paths"] = 5
    store.register(value, digest(value))
    attempt = store.new_attempt(store.register_run("synthetic", cell))
    plugin = registry.resolve(cell["plugin_id"], cell["capability"], version="1.0.0")
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        AdmissionGate(store).prepare(value, cell, plugin, attempt)
    assert not any(event["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for event in store.events())


def test_supervisor_added_provenance_cannot_exceed_final_publication_quota(tmp_path):
    store, value, registry, _ = resource_fixture(tmp_path, result_bytes="exact-worker-output")
    cell = value["cells"][0]
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        observed = SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))
        metadata = store.manifest("artifact-" + observed["artifact_id"])
        pytest.fail("final publication exceeds admitted result quota: published=" + str(metadata["size_bytes"]) +
            ", limit=" + str(registry.resolve(cell["plugin_id"], cell["capability"], version="1.0.0").registry_entry.resource_contract["limits"]["result_bytes"]))
    assert not any(attempt["state"] == "SUCCEEDED" for attempt in store.attempts().values())
