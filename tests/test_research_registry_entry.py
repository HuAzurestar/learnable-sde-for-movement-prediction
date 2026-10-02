"""Shared version/resource core; runner/admission wiring is a separate required check."""

from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util

import pytest

from infrastructure.research_store import ResearchError, digest


def api():
    assert importlib.util.find_spec("application.research_registry") is not None, "required version/resource registry core is absent"
    return importlib.import_module("application.research_registry")


def builder(config):
    return {"value": config["paths"]}


def other_builder(config):
    return {"value": config["paths"] + 1}


def default_builder(config, offset=1):
    return {"value": config["paths"] + offset}


def captured_builder(offset):
    def captured(config):
        return {"value": config["paths"] + offset}
    return captured


def entry(command=builder, version="1.0.0"):
    core = api()
    count_schema = {"type": "integer", "minimum": 1, "maximum": 1_000_000}
    return core.RegistryEntry(component_id="synthetic-predictor", component_kind="predictor", version=version,
        code_hash=core.implementation_hash(command),
        config_schema={"type": "object", "properties": {"paths": count_schema, "steps": count_schema},
                       "required": ["paths", "steps"], "additionalProperties": False},
        input_schema={"type": "object", "properties": {"observations": count_schema},
                      "required": ["observations"], "additionalProperties": False},
        output_schema={"type": "object", "properties": {"value": {"type": "number"}},
                       "required": ["value"], "additionalProperties": False},
        state_order=("x", "y", "vx", "vy"), units=("m", "m", "m/s", "m/s"),
        capabilities=frozenset({"generic-rollout"}), resource_class="cpu", resume_level="exact",
        resource_contract={"schema_version": "pirc25-resource-contract-v1",
            "counts": {"paths": {"config": ["paths"]}, "steps": {"config": ["steps"]},
                "mixtures": {"constant": 1}, "components": {"constant": 1},
                "observations": {"input": ["observations"]}, "state_dim": {"constant": 4}},
            "tensors": [{"name": "samples", "axes": ["paths", "steps", "state_dim"], "item_bytes": 8}],
            "limits": {"matrix_cells": 1000, "paths": 100, "steps": 100, "mixtures": 4,
                "components": 4, "observations": 1000, "tensor_elements": 10000,
                "tensor_bytes": 80000, "result_bytes": 64000}})


def test_entry_has_required_canonical_metadata_and_exact_version_resolution():
    core, value = api(), entry()
    registry = core.VersionedRegistry()
    reference = registry.register(value, builder)
    assert reference == digest(value.manifest())
    assert registry.register(value, builder) == reference
    registration = registry.resolve(value.component_id, value.version,
        required_capabilities={"generic-rollout"}, entry_hash=reference)
    assert registration.entry.manifest() == value.manifest() and registration.builder is builder
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        registry.resolve(value.component_id, "unregistered-version")
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        registry.resolve(value.component_id, value.version, required_capabilities={"exact-transition"})
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        registry.resolve(value.component_id, value.version, entry_hash="0" * 64)


def test_same_version_different_implementation_rejected_and_new_version_kept():
    core = api()
    registry = core.VersionedRegistry()
    registry.register(entry(), builder)
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        registry.register(entry(other_builder), other_builder)
    registry.register(entry(other_builder, "2.0.0"), other_builder)
    assert registry.resolve("synthetic-predictor", "1.0.0").builder is builder
    assert registry.resolve("synthetic-predictor", "2.0.0").builder is other_builder
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        core.VersionedRegistry().register(entry(other_builder), builder)


def test_registration_seals_mutable_metadata_and_rechecks_live_implementation(monkeypatch):
    core, value = api(), entry()
    registry = core.VersionedRegistry()
    reference = registry.register(value, builder)
    snapshot = deepcopy(value.manifest())
    value.config_schema["properties"]["paths"]["maximum"] = 2
    registration = registry.resolve(value.component_id, value.version, entry_hash=reference)
    registration.entry.resource_contract["limits"]["paths"] = 0
    assert registry.resolve(value.component_id, value.version).entry.manifest() == snapshot
    monkeypatch.setattr(builder, "__code__", other_builder.__code__)
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        registry.resolve(value.component_id, value.version)


@pytest.mark.parametrize("field,bad", [("version", ""), ("code_hash", "missing"),
    ("state_order", ("x", "x")), ("units", ("m",)), ("capabilities", frozenset()),
    ("resource_class", "unregistered-device"), ("resume_level", "weights-only"),
    ("config_schema", {"$ref": "https://external.invalid/schema"}),
    ("input_schema", {}), ("output_schema", {"type": "object", "unsupported": True})])
def test_missing_or_unsupported_metadata_is_rejected(field, bad):
    core = api()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        core.VersionedRegistry().register(replace(entry(), **{field: bad}), builder)


def test_pure_resource_plan_is_derived_from_frozen_schema_and_declared_shapes():
    core, value = api(), entry()
    config, inputs = {"paths": 4, "steps": 3}, {"observations": 10}
    plan = core.plan_resources(value, config, inputs, matrix_cells=8)
    assert plan["registry_entry_hash"] == digest(value.manifest())
    assert plan["config_hash"] == digest(config) and plan["input_hash"] == digest(inputs)
    assert plan["matrix_cells"] == 8 and plan["counts"]["state_dim"] == 4
    assert plan["tensor_elements"] == 48 and plan["tensor_bytes"] == 384
    assert plan["resource_plan_hash"] == digest({key: item for key, item in plan.items() if key != "resource_plan_hash"})


@pytest.mark.parametrize("config,inputs,matrix", [({"paths": 101, "steps": 3}, {"observations": 10}, 1),
    ({"paths": 4, "steps": 101}, {"observations": 10}, 1),
    ({"paths": 4, "steps": 3}, {"observations": 1001}, 1),
    ({"paths": 4, "steps": 3}, {"observations": 10}, 1001),
    ({"paths": 100, "steps": 100}, {"observations": 10}, 1)])
def test_over_quota_denied_without_invoking_factory(config, inputs, matrix):
    core = api()
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        core.plan_resources(entry(), config, inputs, matrix_cells=matrix)


@pytest.mark.parametrize("config", [{"paths": True, "steps": 3}, {"paths": 2.5, "steps": 3},
    {"paths": 4}, {"paths": 4, "steps": 3, "extra": "not-declared"}])
def test_bad_configuration_does_not_coerce_or_silently_drop_fields(config):
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        api().plan_resources(entry(), config, {"observations": 10}, matrix_cells=1)


@pytest.mark.parametrize("mutation", ["unknown-axis", "invalid-count", "state-disagreement", "over-global-quota"])
def test_bad_resource_declarations_are_rejected_before_planning(mutation):
    core, value = api(), entry()
    if mutation == "unknown-axis":
        value.resource_contract["tensors"][0]["axes"].append("not-a-declared-count")
    elif mutation == "invalid-count":
        value.resource_contract["counts"]["paths"] = {"constant": True}
    elif mutation == "state-disagreement":
        value.resource_contract["counts"]["state_dim"] = {"constant": 2}
    else:
        value.resource_contract["limits"]["tensor_bytes"] = 2 ** 63
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH|RESOURCE_PLAN_REJECTED"):
        core.plan_resources(value, {"paths": 4, "steps": 3}, {"observations": 10}, matrix_cells=1)


def test_default_and_capture_changes_cannot_reuse_the_same_registered_identity(monkeypatch):
    core = api()
    registry = core.VersionedRegistry()
    value = entry(default_builder)
    registry.register(value, default_builder)
    monkeypatch.setattr(default_builder, "__defaults__", (2,))
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        registry.resolve(value.component_id, value.version)
    first, second = captured_builder(1), captured_builder(2)
    registry = core.VersionedRegistry()
    registry.register(entry(first), first)
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        registry.register(entry(second), second)


def test_plan_does_not_share_mutable_resource_policy_with_the_caller():
    core, value = api(), entry()
    plan = core.plan_resources(value, {"paths": 4, "steps": 3}, {"observations": 10}, matrix_cells=1)
    before = deepcopy(plan)
    value.resource_contract["limits"]["paths"] = 1
    assert plan == before
    assert plan["resource_plan_hash"] == digest({key: item for key, item in plan.items() if key != "resource_plan_hash"})


@pytest.mark.parametrize("schema", [{"type": []}, {"type": "object", "properties": []},
    {"type": "array", "items": {"type": {}}}, {"type": "object", "required": [False]},
    {"type": "number", "minimum": True}])
def test_malformed_schema_types_have_typed_rejections(schema):
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        api().validate_schema(schema)


@pytest.mark.parametrize("field", ["component_kind", "resource_class", "resume_level"])
def test_malformed_metadata_types_have_typed_rejections(field):
    core = api()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        core.VersionedRegistry().register(replace(entry(), **{field: []}), builder)


def test_integer_schema_supports_real_128_bit_rng_state_but_rejects_unbounded_input():
    core = api()
    core.validate_value({"type": "integer", "minimum": 0}, 2 ** 127 + 19)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        core.validate_value({"type": "integer", "minimum": 0}, 2 ** 4096)
