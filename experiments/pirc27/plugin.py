"""Explicit affine method adapter for the existing shared supervisor.

No store initialization, authorization generation or automatic research run.
Operators supply the frozen spec and existing admission records.
"""

from __future__ import annotations

import sys

from application.research_contracts import ExecutionPlugin
from application.research_registry import GLOBAL_LIMITS, RegistryEntry, implementation_hash
from infrastructure.research_store import ResearchError, digest


def propagation_command(output, spec, cell):
    binding = spec.get("runtime_binding", {})
    from pathlib import Path
    if (set(binding) != {"root", "store_id"} or type(binding["root"]) is not str
            or not Path(binding["root"]).is_absolute() or type(binding["store_id"]) is not str):
        raise ResearchError("CONTRACT_MISMATCH", "an explicit existing shared root/store binding is required")
    return [sys.executable, "-m", "experiments.pirc27.worker", binding["root"], binding["store_id"],
            spec["study_id"], digest(cell), str(output)]


def propagation_plugin():
    integer = lambda maximum, minimum=1: {"type": "integer", "minimum": minimum, "maximum": maximum}
    properties = {"method": {"type": "string", "enum": ["exact", "euler", "heun", "gaussian", "mlmc", "importance"]},
        "samples": integer(1_000_000, 2), "base_steps": integer(8192), "steps": integer(8192), "chunk_size": integer(256),
        "level_samples": {"type": "array", "items": integer(1_000_000, 2), "minItems": 0, "maxItems": 9},
        "proposal": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}
    config = {"type": "object", "properties": properties, "required": sorted(properties), "additionalProperties": False}
    inputs = {"state_dim": {"type": "integer", "enum": [4]}, "horizon": {"type": "number", "minimum": 0},
              "model_package_hash": {"type": "string", "minLength": 64, "maxLength": 64},
              "request_hash": {"type": "string", "minLength": 64, "maxLength": 64}}
    states, units = ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s")
    capabilities = frozenset({"exact-transition", "generic-rollout", "coupled-level", "rare-event"})
    entry = RegistryEntry("affine-propagation", "execution-adapter", "1.0.0", implementation_hash(propagation_command),
        config, {"type": "object", "properties": inputs, "required": sorted(inputs), "additionalProperties": False},
        {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["pirc25-result-v1"]},
            "status": {"type": "string", "enum": ["SUCCEEDED"]}}, "required": ["schema_version", "status"], "additionalProperties": True},
        states, units, capabilities, "cpu", "restart-only",
        {"schema_version": "pirc25-resource-contract-v1", "counts": {
            "paths": {"config": ["samples"]}, "steps": {"config": ["steps"]}, "mixtures": {"constant": 1},
            "components": {"config_length": ["level_samples"]}, "observations": {"config": ["chunk_size"]},
            "state_dim": {"constant": 4}},
         "tensors": [{"name": name, "axes": ["observations", "state_dim"], "item_bytes": 8}
                     for name in ("fine", "coarse", "noise", "rng-and-statistics", "update-workspace")]
                    + [{"name": "affine-workspace", "axes": ["state_dim", "state_dim", "state_dim"], "item_bytes": 8}],
         "limits": {**dict(GLOBAL_LIMITS), "matrix_cells": 10000, "paths": 1_000_000, "steps": 8192,
                    "observations": 256, "result_bytes": 512*1024}})
    return ExecutionPlugin(entry.component_id, capabilities, states, units, "restart-only", propagation_command, entry)


def execution_config(request, method, *, level_samples=(), proposal=(0.0, 0.0)):
    request.validate()
    if method not in {"exact", "euler", "heun", "gaussian", "mlmc", "importance"}:
        raise ResearchError("CONTRACT_MISMATCH", "method is not in the frozen adapter")
    if method == "mlmc":
        if (type(level_samples) is not tuple or not 1 <= len(level_samples) <= 9
                or any(type(n) is not int or n < 2 for n in level_samples)
                or sum(level_samples) != request.samples):
            raise ResearchError("CONTRACT_MISMATCH", "MLMC allocation must equal the registered total sample count")
    elif level_samples:
        raise ResearchError("CONTRACT_MISMATCH", "unexpected MLMC allocation for another method")
    if method != "importance" and proposal != (0.0, 0.0):
        raise ResearchError("CONTRACT_MISMATCH", "proposal drift is permitted only for the importance arm")
    steps = request.steps*2**(len(level_samples)-1) if method == "mlmc" else request.steps
    return {"method": method, "samples": request.samples, "base_steps": request.steps, "steps": steps,
            "chunk_size": request.chunk_size, "level_samples": list(level_samples), "proposal": list(proposal)}


def execution_inputs(request):
    return {"state_dim": 4, "horizon": request.horizons[0], "model_package_hash": request.model_package_hash,
            "request_hash": request.request_hash}
