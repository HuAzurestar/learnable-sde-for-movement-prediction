"""Allocation-free composition checks; actual factories run in managed workers.

Factory declarations are not a Python sandbox or scientific qualification.
The command/source, data grants and method qualification still apply.
"""

from .research_registry import GLOBAL_LIMITS, _bounded_json, validate_composition
from infrastructure.research_store import ResearchError, digest, encode
import json


ROLES = frozenset({"model", "trainer", "predictor"})
PROFILE_FIELDS = ("state_dim", "noise_dim", "diffusion_support", "dtype", "device")


def component_plan(entry, registries, components, *, matrix_cells, seed=None):
    validate_composition(entry.composition)
    _bounded_json(components, nodes=65536, depth=32, byte_limit=4 * 1024 * 1024)
    if type(registries) is not dict or set(registries) != ROLES or type(components) is not dict or set(components) != ROLES:
        raise ResearchError("CONTRACT_MISMATCH", "all declared component registries and bindings required")
    # Detach once before inspecting the whole combination; no factory executes.
    components = json.loads(encode(components))
    entries, plans, profile, shared_config = {}, {}, None, None
    for role in sorted(ROLES):
        binding, rule = components[role], entry.composition["roles"][role]
        plan, registration = registries[role].plan_bound(binding, matrix_cells=matrix_cells,
            required_capabilities=rule["required_capabilities"], state_order=entry.state_order,
            units=entry.units, resource_class=entry.resource_class)
        child = registration.entry
        if child.component_kind != role:
            raise ResearchError("CONTRACT_MISMATCH", "component registry contains the wrong role")
        inputs = binding["inputs"]
        if type(inputs) is not dict or not set(PROFILE_FIELDS) <= set(inputs):
            raise ResearchError("CONTRACT_MISMATCH", "component input profile is incomplete")
        selected = {name: inputs[name] for name in PROFILE_FIELDS}
        if (type(selected["state_dim"]) is not int or selected["state_dim"] != len(entry.state_order)
                or type(selected["noise_dim"]) is not int or not 0 <= selected["noise_dim"] <= GLOBAL_LIMITS["components"]
                or type(selected["diffusion_support"]) is not list
                or any(type(name) is not str or name not in entry.state_order for name in selected["diffusion_support"])
                or len(set(selected["diffusion_support"])) != len(selected["diffusion_support"])
                or selected["dtype"] not in {"float32", "float64"}
                or selected["device"] != ("cpu" if entry.resource_class == "cpu" else "cuda")
                or profile is not None and profile != selected):
            raise ResearchError("CONTRACT_MISMATCH", "component state/noise/dtype/device profiles disagree")
        profile = selected
        config = binding["config"]
        if entry.composition["shared_configuration"] and shared_config is not None and shared_config != config:
            raise ResearchError("CONTRACT_MISMATCH", "component shared configurations disagree")
        shared_config = config
        if rule["seed_path"] is not None:
            selected_seed = config
            for key in rule["seed_path"]:
                if type(selected_seed) is not dict or key not in selected_seed:
                    raise ResearchError("CONTRACT_MISMATCH", "component random identity is missing")
                selected_seed = selected_seed[key]
            if type(selected_seed) is not int or seed is not None and selected_seed != seed:
                raise ResearchError("CONTRACT_MISMATCH", "component random identity differs from registered cell")
        entries[role], plans[role] = child.manifest(), plan
    for role, rule in entry.composition["roles"].items():
        if not set(rule["required_model_capabilities"]) <= set(entries["model"]["capabilities"]):
            raise ResearchError("CONTRACT_MISMATCH", "component requires an unavailable model capability")
        # Explicit requires-* capabilities are also used by the existing root.
        required = {capability.removeprefix("requires-") for capability in entries[role]["capabilities"]
            if capability.startswith("requires-")}
        if not required <= set(entries["model"]["capabilities"]):
            raise ResearchError("CONTRACT_MISMATCH", "registered component model requirement is unavailable")
    result = {"schema_version": "pirc25-component-plan-v1", "binding_hash": digest(components),
        "entries": entries, "plans": plans, "profile": profile,
        **{key: sum(plan[key] for plan in plans.values()) for key in ("tensor_elements", "tensor_bytes")}}
    result["component_plan_hash"] = digest(result)
    return result


def combined_plan(entry, plan, registries, components, *, seed=None):
    if entry.composition is None:
        if registries is not None or components is not None:
            raise ResearchError("CONTRACT_MISMATCH", "undeclared component composition")
        return plan
    composition = component_plan(entry, registries, components, matrix_cells=plan["matrix_cells"], seed=seed)
    for key in ("tensor_elements", "tensor_bytes"):
        if plan[key] + composition[key] > plan["limits"][key]:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "combined adapter/components exceed allocation quota")
    for child in composition["plans"].values():
        if any(child["counts"][name] > plan["counts"][name] for name in child["counts"]):
            raise ResearchError("RESOURCE_PLAN_REJECTED", "component counts exceed adapter allocation bound")
    result = {**plan, "adapter_resource_plan_hash": plan["resource_plan_hash"], "composition": composition,
        **{key: plan[key] + composition[key] for key in ("tensor_elements", "tensor_bytes")}}
    result.pop("resource_plan_hash")
    result["resource_plan_hash"] = digest(result)
    return result
