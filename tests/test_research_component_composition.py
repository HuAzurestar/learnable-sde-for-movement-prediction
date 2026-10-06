"""Both real affine workers must consume the owner-frozen internal components.

The fixtures are synthetic engineering controls, not scientific qualification.
"""

import json
from copy import deepcopy

import pytest

from application.registry import ComponentRegistry
from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25 import affine
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest


@pytest.mark.parametrize("dimensions", [1, 4])
def test_actual_managed_affine_cannot_run_without_internal_component_bindings(tmp_path, dimensions):
    store = ResearchStore(tmp_path, "composition-missing", initialize=True)
    value = affine.fixture_spec(dimensions=dimensions)
    cell = value["cells"][0]
    cell["execution"].pop("components", None)
    cell["execution"].pop("component_plan_hash", None)
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        observed = SharedRunner(store).run_cell(value["study_id"], digest(cell), budget=BudgetSpec(60))
        pytest.fail("actual affine worker succeeded without internal model/trainer/predictor bindings: " + observed["state"])
    assert not any(event["event_kind"] in {"WORKER_STARTED", "READ_STARTED"} for event in store.events())
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] == 0


@pytest.mark.parametrize("dimensions", [1, 4])
def test_worker_revalidates_component_binding_before_any_numerical_chain(monkeypatch, dimensions):
    value = affine.fixture_spec(dimensions=dimensions)
    cell = value["cells"][0]
    cell["execution"].pop("components", None)
    cell["execution"].pop("component_plan_hash", None)
    monkeypatch.setattr(affine, "single_axis", lambda *a, **k: pytest.fail("unbound single-axis chain executed"))
    monkeypatch.setattr(affine, "four_state", lambda *a, **k: pytest.fail("unbound four-state chain executed"))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        affine.execute(value, cell)


@pytest.mark.parametrize("dimensions", [1, 4])
def test_managed_real_components_have_persisted_version_plan_and_uncharged_reuse(tmp_path, dimensions):
    store = ResearchStore(tmp_path, "composition-positive", initialize=True)
    value = affine.fixture_spec(dimensions=dimensions)
    cell = value["cells"][0]
    store.register(value, digest(value))
    observed = SharedRunner(store).run_cell(value["study_id"], digest(cell), budget=BudgetSpec(60))
    assert observed["state"] == "SUCCEEDED"
    result = json.loads((store.path / "artifacts" / observed["artifact_id"]).read_bytes())
    receipt = store.manifest("admission-" + result["admission_hash"])
    components = cell["execution"].get("components")
    assert type(components) is dict and set(components) == {"model", "trainer", "predictor"}
    composition = receipt["resource_plan"]["composition"]
    assert result["component_plan_hash"] == cell["execution"]["component_plan_hash"] == composition["component_plan_hash"]
    for role, entry in composition["entries"].items():
        assert entry["component_kind"] == role
        assert digest(entry) == components[role]["registry_entry_hash"]
        assert store.manifest("registry-entry-" + digest(entry)) == entry
        reference = store.manifest("registry-version-" + digest({"id": entry["component_id"], "version": entry["version"]}))
        assert reference["registry_entry_hash"] == digest(entry)
    balance = BudgetLedger(store).balance(cell["arm_id"])
    assert SharedRunner(store).run_cell(value["study_id"], digest(cell))["reused"]
    assert BudgetLedger(store).balance(cell["arm_id"]) == balance


@pytest.mark.parametrize("dimensions", [1, 4])
def test_existing_chain_constructs_all_three_registered_roles(monkeypatch, dimensions):
    value = affine.fixture_spec(dimensions=dimensions)
    seen = []
    original = ComponentRegistry.create_bound
    def trace(registry, binding, **kwargs):
        seen.append(binding["registry_entry_hash"])
        return original(registry, binding, **kwargs)
    monkeypatch.setattr(ComponentRegistry, "create_bound", trace)
    affine.execute(value, value["cells"][0])
    components = value["cells"][0]["execution"].get("components", {})
    assert len(seen) == 3, "existing numerical pipeline bypassed registered internal factories"
    assert sorted(seen) == sorted(binding["registry_entry_hash"] for binding in components.values())


@pytest.mark.parametrize("dimensions", [1, 4])
@pytest.mark.parametrize("fault", ["version", "entry", "plan", "missing-role", "paths", "noise", "seed"])
def test_component_denial_precedes_actual_worker_or_input(tmp_path, dimensions, fault):
    store = ResearchStore(tmp_path, "composition-denial", initialize=True)
    value = affine.fixture_spec(dimensions=dimensions)
    cell = value["cells"][0]
    components = cell["execution"].get("components")
    assert isinstance(components, dict), "required internal components are not frozen"
    components = deepcopy(components)
    cell["execution"]["components"] = components
    if fault == "version":
        components["model"]["component_version"] = "unavailable"
    elif fault == "entry":
        components["trainer"]["registry_entry_hash"] = "0" * 64
    elif fault == "plan":
        cell["execution"]["component_plan_hash"] = "0" * 64
    elif fault == "missing-role":
        components.pop("predictor")
    elif fault == "paths":
        components["predictor"]["inputs"]["paths"] = 1000001
    elif fault == "noise":
        components["model"]["inputs"]["noise_dim"] = 3
    else:
        components["model"]["config"]["seed"] += 1
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH|RESOURCE_PLAN_REJECTED"):
        SharedRunner(store).run_cell(value["study_id"], digest(cell), budget=BudgetSpec(60))
    assert not any(event["event_kind"] in {"WORKER_STARTED", "READ_STARTED"} for event in store.events())
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] == 0
