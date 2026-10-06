"""Versioned plugin/result validation without implementing child algorithms."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from infrastructure.research_store import ResearchError, digest
from .research_registry import RegistryEntry, VersionedRegistry, validate_entry

CAPABILITIES = {"exact-transition", "generic-rollout", "switching-transition", "coupled-level", "rare-event"}
RESUME_LEVELS = {"exact", "numerical-tolerance", "chunk", "restart-only"}
STATE = ["x", "y", "vx", "vy"]
UNITS = ["m", "m", "m/s", "m/s"]


def validate_package(package: dict, *, kind: str, formal=False, frozen_model=None):
    if package.get("schema_version") != "pirc25-package-v1" or package.get("kind") != kind:
        raise ResearchError("CONTRACT_MISMATCH", "package schema/kind mismatch")
    if package.get("state_order") != STATE or package.get("units") != UNITS:
        raise ResearchError("CONTRACT_MISMATCH", "package state order or units mismatch")
    capabilities = package.get("capabilities", [])
    if not capabilities or not set(capabilities) <= CAPABILITIES or package.get("resume_level") not in RESUME_LEVELS:
        raise ResearchError("CONTRACT_MISMATCH", "package capabilities or recovery declaration missing")
    for key in ("code_hash", "data_hash", "input_hash", "output_hash", "protocol_hash"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(package.get(key, ""))):
            raise ResearchError("CONTRACT_MISMATCH", f"missing package {key}")
    if package.get("qualification") not in {"fixture", "qualified", "failed", "unavailable"}:
        raise ResearchError("CONTRACT_MISMATCH", "package qualification missing")
    if formal and (package["qualification"] != "qualified"
                   or not re.fullmatch(r"[0-9a-f]{64}", str(package.get("qualification_hash", "")))
                   or not re.fullmatch(r"[0-9a-f]{64}", str(package.get("preregistration_hash", "")))):
        raise ResearchError("UNQUALIFIED", "formal comparison requires qualification and preregistration")
    if package.get("requires_frozen_model"):
        if frozen_model is None:
            raise ResearchError("MISSING_INPUT", "nonlinear comparison needs a qualified frozen model")
        validate_package(frozen_model, kind="FrozenDynamicsPackage", formal=True)
        if package.get("model_hash") != digest(frozen_model):
            raise ResearchError("CONTRACT_MISMATCH", "frozen model identity differs")
    return digest(package)


def accept_model(package, **kwargs):
    return validate_package(package, kind="FrozenDynamicsPackage", **kwargs)


def accept_propagation(package, **kwargs):
    return validate_package(package, kind="PropagationResult", **kwargs)


def accept_switching(package, **kwargs):
    return validate_package(package, kind="SwitchingResult", **kwargs)


@dataclass(frozen=True)
class ExecutionPlugin:
    plugin_id: str
    capabilities: frozenset[str]
    state_order: tuple[str, ...]
    units: tuple[str, ...]
    resume_level: str
    command_builder: object
    registry_entry: RegistryEntry | None = None
    component_registries: dict | None = None
    pre_read_validator: object = None
    checkpoint_validator: object = None


class CapabilityRegistry:
    def __init__(self):
        self._plugins = {}
        self._registry = VersionedRegistry()

    def register(self, plugin: ExecutionPlugin):
        validate_entry(plugin.registry_entry)
        entry = plugin.registry_entry
        if entry.composition is not None:
            from .registry import ComponentRegistry
            if (type(plugin.component_registries) is not dict or
                    set(plugin.component_registries) != {"model", "trainer", "predictor"} or
                    any(not isinstance(registry, ComponentRegistry) for registry in plugin.component_registries.values())):
                raise ResearchError("CONTRACT_MISMATCH", "declared composition needs real component registries")
        elif plugin.component_registries is not None:
            raise ResearchError("CONTRACT_MISMATCH", "component registries need a declared composition")
        if (not plugin.capabilities
                or not plugin.capabilities <= CAPABILITIES or plugin.resume_level not in RESUME_LEVELS
                or len(plugin.units) != len(plugin.state_order) or not callable(plugin.command_builder)
                or plugin.pre_read_validator is not None and not callable(plugin.pre_read_validator)
                or plugin.checkpoint_validator is not None and not callable(plugin.checkpoint_validator)
                or entry.component_id != plugin.plugin_id or entry.component_kind != "execution-adapter"
                or entry.capabilities != plugin.capabilities or entry.resume_level != plugin.resume_level
                or entry.state_order != plugin.state_order or entry.units != plugin.units):
            raise ResearchError("CONTRACT_MISMATCH", "invalid or duplicate execution plugin")
        reference = self._registry.register(entry, plugin.command_builder)
        self._plugins[(plugin.plugin_id, entry.version)] = replace(plugin,
            registry_entry=self._registry.resolve(plugin.plugin_id, entry.version).entry,
            component_registries=dict(plugin.component_registries) if plugin.component_registries is not None else None)
        return reference

    def resolve(self, plugin_id: str, required_capability: str, *, version=None, entry_hash=None) -> ExecutionPlugin:
        registration = self._registry.resolve(plugin_id, version,
            required_capabilities=(required_capability,), entry_hash=entry_hash)
        plugin = self._plugins.get((plugin_id, version))
        if plugin is None or required_capability not in plugin.capabilities:
            raise ResearchError("CONTRACT_MISMATCH", "requested capability is unavailable")
        return replace(plugin, registry_entry=registration.entry,
            component_registries=dict(plugin.component_registries) if plugin.component_registries is not None else None)


def validate_result(result: dict, *, spec: dict, cell: dict, plugin: ExecutionPlugin):
    if (result.get("schema_version") != "pirc25-result-v1" or result.get("status") != "SUCCEEDED"
            or result.get("spec_hash") != digest(spec) or result.get("cell_hash") != digest(cell)
            or result.get("state_order") != list(plugin.state_order) or result.get("units") != list(plugin.units)
            or result.get("resume_level") != plugin.resume_level
            or result.get("protocol_hash") != spec["protocol_hash"]):
        raise ResearchError("CONTRACT_MISMATCH", "worker result differs from registered execution contract")
    metrics = result.get("metrics", {})
    if not metrics or any(not isinstance(v, (int, float)) for v in metrics.values()):
        raise ResearchError("CONTRACT_MISMATCH", "result metrics missing")
    if set(result.get("metric_units", {})) != set(metrics) or result.get("input_hash") != spec["data_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "result units or input identity missing")
    output = {key: result[key] for key in ("metrics", "forecast", "fit", "source_schema")}
    if result.get("output_hash") != digest(output):
        raise ResearchError("CORRUPT_ARTIFACT", "result output hash mismatch")
    from infrastructure.research_store import encode
    encode(result)  # Reject NaN and infinities before publishing success.
