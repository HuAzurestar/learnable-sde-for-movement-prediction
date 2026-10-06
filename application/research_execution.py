"""Owner-side version and allocation preflight, shared by run/reuse/recovery.

Plans are derived from explicit immutable declarations, never from worker
claims. This is not an OS memory sandbox or scientific qualification.
"""

from .research_registry import implementation_hash, plan_resources, validate_entry
from infrastructure.research_store import ResearchError, digest
from infrastructure.research_store import encode
import json


BINDING_FIELDS = frozenset({"schema_version", "component_id", "component_version", "registry_entry_hash",
    "config", "inputs", "resource_plan_hash", "resource_class"})
COMPOSITION_FIELDS = frozenset({"components", "component_plan_hash"})


def execution_binding(entry, config, inputs, *, matrix_cells, components=None, component_registries=None):
    from .research_composition import combined_plan
    plan = plan_resources(entry, config, inputs, matrix_cells=matrix_cells)
    plan = combined_plan(entry, plan, component_registries, components, config=config, inputs=inputs)
    result = {"schema_version": "pirc25-execution-binding-v1", "component_id": entry.component_id,
        "component_version": entry.version, "registry_entry_hash": plan["registry_entry_hash"],
        "config": json.loads(encode(config)), "inputs": json.loads(encode(inputs)), "resource_plan_hash": plan["resource_plan_hash"],
        "resource_class": entry.resource_class}
    if entry.composition is not None:
        result.update(components=json.loads(encode(components)), component_plan_hash=plan["composition"]["component_plan_hash"])
    return result


def resolve_execution(registry, cell):
    binding = cell.get("execution")
    if type(binding) is not dict or set(binding) not in (BINDING_FIELDS, BINDING_FIELDS | COMPOSITION_FIELDS) or binding.get("schema_version") != "pirc25-execution-binding-v1":
        raise ResearchError("CONTRACT_MISMATCH", "explicit versioned execution/resource binding required")
    if binding.get("component_id") != cell.get("plugin_id"):
        raise ResearchError("CONTRACT_MISMATCH", "execution component differs from registered cell")
    plugin = registry.resolve(cell["plugin_id"], cell["capability"],
        version=binding["component_version"], entry_hash=binding["registry_entry_hash"])
    expected = BINDING_FIELDS | COMPOSITION_FIELDS if plugin.registry_entry.composition is not None else BINDING_FIELDS
    if set(binding) != expected:
        raise ResearchError("CONTRACT_MISMATCH", "declared internal component binding is missing or unexpected")
    return plugin


def execution_plan(spec, cell, plugin):
    """Revalidate even if AdmissionGate is called without SharedRunner."""
    entry = plugin.registry_entry
    validate_entry(entry)
    binding = cell.get("execution")
    expected = BINDING_FIELDS | COMPOSITION_FIELDS if entry.composition is not None else BINDING_FIELDS
    if (type(binding) is not dict or set(binding) != expected or
            binding.get("schema_version") != "pirc25-execution-binding-v1" or
            binding.get("component_id") != plugin.plugin_id or entry.component_id != plugin.plugin_id or
            entry.component_kind != "execution-adapter" or binding.get("component_version") != entry.version or
            binding.get("registry_entry_hash") != digest(entry.manifest()) or
            binding.get("resource_class") != entry.resource_class or cell.get("resource_class") != entry.resource_class or
            entry.state_order != plugin.state_order or entry.units != plugin.units or
            entry.capabilities != plugin.capabilities or entry.resume_level != plugin.resume_level or
            cell.get("capability") not in entry.capabilities or implementation_hash(plugin.command_builder) != entry.code_hash):
        raise ResearchError("CONTRACT_MISMATCH", "execution version, identity or compatibility binding differs")
    if type(spec.get("cells")) is not list or cell not in spec["cells"]:
        raise ResearchError("CONTRACT_MISMATCH", "resource plan needs the original registered matrix")
    plan = plan_resources(entry, binding["config"], binding["inputs"], matrix_cells=len(spec["cells"]))
    from .research_composition import combined_plan
    plan = combined_plan(entry, plan, plugin.component_registries, binding.get("components"), seed=cell.get("seed"),
        config=binding["config"], inputs=binding["inputs"])
    if entry.composition is not None and binding["component_plan_hash"] != plan["composition"]["component_plan_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "internal component plan differs from registered composition")
    if binding.get("resource_plan_hash") != plan["resource_plan_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "resource plan differs from registered configuration/input/matrix")
    return plan
