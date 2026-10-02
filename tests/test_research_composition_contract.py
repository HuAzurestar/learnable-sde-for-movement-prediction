"""Allocation-free generic composition controls, beyond fixed affine recipes.

Every binding has a valid registry identity and its own valid resource plan.
No fixture factory may execute during these admission checks.
"""

from copy import deepcopy
from dataclasses import replace

import pytest

from application.registry import ComponentRegistry
from application.research_composition import combined_plan, component_plan
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from application.research_registry import plan_resources
from infrastructure.research_store import ResearchError, digest
from tests.research_admission_fixtures import synthetic_plugin


def factory(config, inputs, context):
    pytest.fail("composition planning invoked a numerical factory")


def fixture(*, shared=True):
    plugin = synthetic_plugin("generic-composition", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", factory)
    rules = {role: {"required_capabilities": ["generic-rollout"],
        "required_model_capabilities": [], "seed_path": None} for role in ("model", "trainer", "predictor")}
    entry = replace(plugin.registry_entry, composition={"schema_version": "pirc25-composition-contract-v1",
        "shared_configuration": shared, "roles": rules})
    entry.input_schema.update(additionalProperties=True)
    entry.config_schema.update(additionalProperties=True)
    profile = {"state_dim": 4, "noise_dim": 2, "diffusion_support": ["vx", "vy"],
        "dtype": "float64", "device": "cpu"}
    registries, bindings = {}, {}
    for role in rules:
        child = replace(deepcopy(entry), component_id="generic-" + role, component_kind=role, composition=None)
        registry = ComponentRegistry()
        registry.register_version(child, factory)
        registries[role] = registry
        bindings[role] = execution_binding(child, {}, profile, matrix_cells=1)
    return replace(plugin, registry_entry=entry, component_registries=registries), bindings


def rebind(plugin, bindings, role, *, profile=None, config=None, change=None):
    registry = plugin.component_registries[role]
    entry = registry.resolve_version("generic-" + role, "1.0.0").entry
    entry = replace(entry, version="2.0.0")
    if change is not None:
        entry = change(entry)
    registry.register_version(entry, factory)
    previous = bindings[role]
    bindings[role] = execution_binding(entry,
        previous["config"] if config is None else config,
        previous["inputs"] if profile is None else profile, matrix_cells=1)


def plan(plugin, bindings):
    return component_plan(plugin.registry_entry, plugin.component_registries, bindings, matrix_cells=1)


def test_generic_valid_plan_is_deterministic_detached_and_allocation_free():
    plugin, bindings = fixture()
    observed = plan(plugin, bindings)
    assert observed == plan(plugin, deepcopy(bindings))
    assert observed["tensor_elements"] == 12 and observed["tensor_bytes"] == 96
    assert observed["binding_hash"] == digest(bindings)
    bindings["model"]["inputs"]["noise_dim"] = 0
    assert observed["profile"]["noise_dim"] == 2


@pytest.mark.parametrize("value", [[], {}])
def test_valid_child_schema_cannot_leak_typeerror_for_malformed_dtype(value):
    plugin, bindings = fixture()
    profile = {**bindings["model"]["inputs"], "dtype": value}
    rebind(plugin, bindings, "model", profile=profile)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        plan(plugin, bindings)


def test_null_configuration_is_not_an_unset_shared_configuration():
    plugin, bindings = fixture()
    child = plugin.component_registries["model"].resolve_version("generic-model", "1.0.0").entry
    child = replace(child, version="null-config", config_schema={"type": "null"})
    plugin.component_registries["model"].register_version(child, factory)
    bindings["model"] = execution_binding(child, None, bindings["model"]["inputs"], matrix_cells=1)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        plan(plugin, bindings)


def test_resolved_registry_mapping_cannot_mutate_the_registered_adapter():
    plugin, bindings = fixture()
    registry = CapabilityRegistry()
    registry.register(plugin)
    first = registry.resolve(plugin.plugin_id, "generic-rollout", version=plugin.registry_entry.version)
    first.component_registries.pop("model")
    second = registry.resolve(plugin.plugin_id, "generic-rollout", version=plugin.registry_entry.version)
    assert set(second.component_registries) == {"model", "trainer", "predictor"}
    assert plan(second, bindings)["binding_hash"] == digest(bindings)


def test_invalid_registry_value_rejects_as_contract_error_without_factory():
    plugin, bindings = fixture()
    plugin.component_registries["model"] = None
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        plan(plugin, bindings)


@pytest.mark.parametrize("fault", ["state-order", "units", "resource-class", "role", "capability",
    "model-requirement", "explicit-requires", "noise", "dtype", "support", "shared-config"])
def test_individually_valid_bindings_cannot_bypass_combination_compatibility(fault):
    plugin, bindings = fixture()
    if fault in {"noise", "dtype", "support"}:
        profile = deepcopy(bindings["predictor"]["inputs"])
        profile[{"noise": "noise_dim", "dtype": "dtype", "support": "diffusion_support"}[fault]] = {
            "noise": 1, "dtype": "float32", "support": ["vx"]}[fault]
        rebind(plugin, bindings, "predictor", profile=profile)
    elif fault == "shared-config":
        rebind(plugin, bindings, "trainer", config={"different": True})
    elif fault == "model-requirement":
        plugin.registry_entry.composition["roles"]["predictor"]["required_model_capabilities"] = ["exact-transition"]
    else:
        changes = {"state-order": {"state_order": ("y", "x", "vx", "vy")},
            "units": {"units": ("km", "km", "m/s", "m/s")},
            "resource-class": {"resource_class": "gpu"}, "role": {"component_kind": "model"},
            "capability": {"capabilities": frozenset({"exact-transition"})},
            "explicit-requires": {"capabilities": frozenset({"generic-rollout", "requires-exact-transition"})}}
        rebind(plugin, bindings, "predictor", change=lambda entry: replace(entry, **changes[fault]))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        plan(plugin, bindings)


def test_shared_configuration_can_be_explicitly_disabled():
    plugin, bindings = fixture(shared=False)
    rebind(plugin, bindings, "trainer", config={"different": True})
    assert plan(plugin, bindings)["binding_hash"] == digest(bindings)


@pytest.mark.parametrize("field", ["tensor_elements", "tensor_bytes"])
def test_joint_allocation_quota_includes_adapter_and_all_children(field):
    plugin, bindings = fixture()
    entry = plugin.registry_entry
    exact = {"tensor_elements": 16, "tensor_bytes": 128}[field]
    entry.resource_contract["limits"][field] = exact
    base = plan_resources(entry, {}, {}, matrix_cells=1)
    assert combined_plan(entry, base, plugin.component_registries, bindings)[field] == exact
    entry.resource_contract["limits"][field] = exact - 1
    base = plan_resources(entry, {}, {}, matrix_cells=1)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        combined_plan(entry, base, plugin.component_registries, bindings)


def test_child_counts_cannot_exceed_adapter_bound_with_valid_hashes():
    plugin, bindings = fixture()
    def change(entry):
        entry.resource_contract["counts"]["paths"] = {"input": ["paths"]}
        return entry
    rebind(plugin, bindings, "predictor", profile={**bindings["predictor"]["inputs"], "paths": 2}, change=change)
    base = plan_resources(plugin.registry_entry, {}, {}, matrix_cells=1)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        combined_plan(plugin.registry_entry, base, plugin.component_registries, bindings)


@pytest.mark.parametrize("field,value", [("noise_dim", 1), ("dtype", "float32"), ("diffusion_support", ["vx"])])
def test_adapter_profile_and_all_child_profiles_must_agree(field, value):
    plugin, bindings = fixture()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        execution_binding(plugin.registry_entry, {field: value}, {}, matrix_cells=1,
            components=bindings, component_registries=plugin.component_registries)


def test_policy_is_snapshotted_before_child_planning(monkeypatch):
    plugin, bindings = fixture()
    plugin.registry_entry.composition["roles"]["predictor"]["required_model_capabilities"] = ["exact-transition"]
    model = plugin.component_registries["model"]
    original = model.plan_bound
    def mutate(binding, **kwargs):
        observed = original(binding, **kwargs)
        plugin.registry_entry.composition["roles"]["predictor"]["required_model_capabilities"].clear()
        return observed
    monkeypatch.setattr(model, "plan_bound", mutate)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        plan(plugin, bindings)
