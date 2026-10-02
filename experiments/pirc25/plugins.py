"""Explicit bounded recipes for the existing synthetic affine adapters."""

import sys

from application.research_contracts import ExecutionPlugin
from application.research_registry import GLOBAL_LIMITS, RegistryEntry, implementation_hash
from infrastructure.research_store import digest


def affine_command(output, spec, cell, context):
    return [sys.executable, "-m", "experiments.pirc25.worker", context["root"],
        context["store_id"], spec["study_id"], digest(cell), str(output)]


def affine_configuration(dimensions):
    # Steps is a conservative bound for the eight-point synthetic segments,
    # including origin; single-axis forecasts have two registered horizons.
    return {"paths": 8, "steps": 8, "mixtures": 2 if dimensions == 1 else 1,
        "components": 4, "dtype": "float64", "device": "cpu", "noise_dim": 1 if dimensions == 1 else 2}, \
        {"observations": 40 if dimensions == 1 else 32, "state_dim": 2 if dimensions == 1 else 4}


def affine_plugin(dimensions):
    if dimensions not in (1, 4):
        raise ValueError("unsupported affine fixture dimension")
    config, inputs = affine_configuration(dimensions)
    states = ("x", "vx") if dimensions == 1 else ("x", "y", "vx", "vy")
    units = ("m", "m/s") if dimensions == 1 else ("m", "m", "m/s", "m/s")
    capabilities = frozenset({"exact-transition" if dimensions == 1 else "generic-rollout"})
    def fixed_schema(values):
        return {"type": "object", "properties": {key: {"type": "string" if type(value) is str else "integer", "enum": [value]}
            for key, value in values.items()}, "required": sorted(values), "additionalProperties": False}
    entry = RegistryEntry(component_id="affine-" + str(dimensions), component_kind="execution-adapter", version="1.0.0",
        code_hash=implementation_hash(affine_command), config_schema=fixed_schema(config), input_schema=fixed_schema(inputs),
        output_schema={"type": "object", "required": ["schema_version", "status"], "properties": {
            "schema_version": {"type": "string", "enum": ["pirc25-result-v1"]},
            "status": {"type": "string", "enum": ["SUCCEEDED"]}}, "additionalProperties": True},
        state_order=states, units=units, capabilities=capabilities, resource_class="cpu", resume_level="restart-only",
        resource_contract={"schema_version": "pirc25-resource-contract-v1", "counts": {
            **{key: {"config": [key]} for key in ("paths", "steps", "mixtures", "components")},
            "observations": {"input": ["observations"]}, "state_dim": {"constant": len(states)}},
            "tensors": [{"name": "rollout", "axes": ["paths", "steps", "state_dim"], "item_bytes": 8},
                {"name": "observations", "axes": ["observations", "state_dim"], "item_bytes": 8},
                {"name": "model_covariances", "axes": ["mixtures", "state_dim", "state_dim"], "item_bytes": 8},
                {"name": "fit_workspace", "axes": ["components", "observations", "state_dim", "state_dim"], "item_bytes": 8}],
            "limits": {**dict(GLOBAL_LIMITS), "matrix_cells": 10000, "result_bytes": 4 * 1024 * 1024}})
    return ExecutionPlugin(entry.component_id, capabilities, states, units, "restart-only", affine_command, entry)
