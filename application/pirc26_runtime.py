"""Owner-side actual phase-space dispatch over the shared runtime.

Importing/registration reads no data and initializes no store. A command can
only be built for an existing attempt with the owner's immutable admission.
The worker receives a bounded owner-read transport, never raw-source paths.
"""

from pathlib import Path
import sys

from application.pirc26_components import (VERSION, STATE, UNITS, config_schema, profile_schema,
    component_registries, composition_contract, basis_family, resume_level, preflight_model)
from application.pirc26_data import read_block
from application.research_contracts import ExecutionPlugin
from application.research_execution import execution_plan
from application.research_recovery import RecoveryPlugin
from application.research_registry import GLOBAL_LIMITS, RegistryEntry, implementation_hash, validate_value
from infrastructure.research_control import canonical, read_frame
from infrastructure.research_store import ResearchError, ResearchStore, atomic_write, digest


HANDOFF_LIMIT = 16 * 1024 * 1024


def runtime_profile_schema():
    schema = profile_schema()
    schema["properties"].update(runtime_root={"type": "string", "minLength": 1, "maxLength": 4096},
                                store_id={"type": "string", "minLength": 1, "maxLength": 128})
    schema["required"] = sorted(schema["properties"])
    return schema


def execution_plugin(objective, family):
    registries = component_registries(objective, family)
    identity = "pirc26-" + family.lower() + "-" + objective.lower()
    entry = RegistryEntry(component_id=identity, component_kind="execution-adapter", version=VERSION,
        code_hash=implementation_hash(command), config_schema=config_schema(objective, family),
        input_schema=runtime_profile_schema(), output_schema={"type": "object", "additionalProperties": True},
        state_order=STATE, units=UNITS, capabilities=frozenset({"generic-rollout"}), resource_class="cpu",
        resume_level=resume_level(family), composition=composition_contract(objective, family),
        resource_contract={"schema_version": "pirc25-resource-contract-v1", "counts": {
            **{key: {"input": [key]} for key in ("paths", "steps", "components", "observations")},
            "state_dim": {"constant": 4}, "mixtures": {"constant": 1}},
            "tensors": [{"name": "owner_transport_indices", "axes": ["observations"], "item_bytes": 16},
                        *[{"name": "bounded_origin_output_" + str(i), "axes": ["paths", "steps", "state_dim"], "item_bytes": 16}
                          for i in range(64)]],
            "limits": {**dict(GLOBAL_LIMITS), "matrix_cells": 10000, "result_bytes": 4 * 1024 * 1024}})
    return ExecutionPlugin(identity, entry.capabilities, STATE, UNITS, entry.resume_level, command, entry, registries,
                           pre_read_validator=pre_read_validate, checkpoint_validator=validate_checkpoint_progress)


def recovery_plugin(objective, family):
    if basis_family(family):
        raise ResearchError("OBJECTIVE_INCOMPATIBLE", "basis QR declares restart-only, not checkpoint continuation")
    return RecoveryPlugin(execution_plugin(objective, family).plugin_id, "exact", resume_command, VERSION)


def pre_read_validate(receipt):
    """Consumer-specific validation before even shared admission verifies data."""
    validate_job(receipt["documents"]["package"]["payload"].get("pirc26_job"), receipt)


def validate_checkpoint_progress(receipt, state, progress):
    from application.pirc26_forecast_control import validate_saved_job, is_forecast_state
    job = receipt["documents"]["package"]["payload"]["pirc26_job"]
    if is_forecast_state(state):
        validate_saved_job(state, job, receipt)
        total = sum(r["sample_count"] * (len(r["time_grid"]) - 1) for r in job["origins"])
        if progress["total_steps"] != total:
            raise ResearchError("CONTRACT_MISMATCH", "forecast progress differs from registered per-path grid work")
    elif (job["operation"] != "fit-and-forecast" or progress["total_steps"] != receipt["cell"]["execution"]["config"]["plan"]["max_steps"]):
        raise ResearchError("CONTRACT_MISMATCH", "training progress differs from frozen plan")


def validate_job(job, receipt):
    """Preflight the immutable job before an authorized physical read."""
    config, inputs = receipt["cell"]["execution"]["config"], receipt["cell"]["execution"]["inputs"]
    recipe = {"segment_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "origin_index": {"type": "integer", "minimum": 0, "maximum": 4097},
        "time_grid": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": inputs["steps"]},
        "sample_count": {"type": "integer", "minimum": 2, "maximum": inputs["paths"]},
        "brownian_root_id": {"type": "string", "minLength": 64, "maxLength": 64},
        "chunk_size": {"type": "integer", "minimum": 1, "maximum": 256}}
    properties = {"schema_version": {"type": "string", "enum": ["pirc26-worker-job-v1"]},
        "operation": {"type": "string", "enum": ["fit-and-forecast", "forecast"]},
        "initial_checkpoint": {"type": "object", "additionalProperties": True},
        "batch_size": {"type": "integer", "minimum": 1, "maximum": 4096},
        "origins": {"type": "array", "minItems": 1, "maxItems": inputs["origins"],
            "items": {"type": "object", "properties": recipe, "required": sorted(recipe), "additionalProperties": False}}}
    # The shared registry schema intentionally has no union types; nullable
    # lineage is checked explicitly instead of adding another schema dialect.
    o1 = job.get("o1_result") if type(job) is dict else None
    properties["o1_result"] = {"type": "object", "additionalProperties": True} if o1 is not None else {"type": "null"}
    validate_value({"type": "object", "properties": properties, "required": sorted(properties),
                    "additionalProperties": False}, job)
    if sum(part["sample_count"] * len(part["time_grid"]) * 4 for part in job["origins"]) > 65536:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "combined origin publication workspace exceeds the bounded worker route")
    if job["initial_checkpoint"].get("sha256") != config["initial_model_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "job initial model differs from the registered component")
    preflight_model(job["initial_checkpoint"], config, inputs)
    if config["objective"] == "O1" and o1 is not None or config["objective"] == "O2" and digest(o1) != config["o1_lineage_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "job O1 lineage differs from the registered objective")
    protocol = receipt["documents"]["protocol"]
    block = next(b for b in protocol["blocks"] if b["block_id"] == receipt["cell"]["block_id"])
    if job["operation"] == "fit-and-forecast" and (block["split_role"] != "train" or receipt["spec"]["admission"]["purpose"] != "fit"):
        raise ResearchError("UNAUTHORIZED_DATA", "fitting is restricted to admitted train/fit blocks")
    if basis_family(config["family"]):
        validate_value(config_schema("O1", config["family"]), config)
        if job["batch_size"] > config["plan"]["max_batch_rows"]:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "job batch size exceeds its registered QR plan")
    if receipt["mode"] == "formal" or block["split_role"] in ("test", "final-eval"):
        model = receipt["documents"].get("frozen_model")
        if (job["operation"] != "forecast" or not model or model.get("qualification") != "qualified"
                or model.get("payload", {}).get("checkpoint") != job["initial_checkpoint"]):
            raise ResearchError("UNQUALIFIED", "test forecasts require the owner's independently qualified frozen checkpoint")
    from application.pirc26_metrics import metric_binding
    metric_binding(job, receipt)
    if not basis_family(config["family"]):
        from application.pirc26_forecast_control import preflight_job_capacity
        preflight_job_capacity(job, config, inputs)
    return job


def admitted_context(output, spec, cell, *, running=False, recovery=False):
    """Resolve only the store and attempt already owned by this output path."""
    binding = cell["execution"]
    config, inputs = binding["config"], binding["inputs"]
    if recovery and basis_family(config["family"]):
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "restart-only basis jobs cannot resume old factorization state")
    plugin = execution_plugin(config["objective"], config["family"])
    plan = execution_plan(spec, cell, plugin)
    child_inputs = {k: v for k, v in inputs.items() if k not in ("runtime_root", "store_id")}
    if any(part["config"] != config or part["inputs"] != child_inputs for part in binding["components"].values()):
        raise ResearchError("CONTRACT_MISMATCH", "adapter and actual component recipes differ")
    store = ResearchStore(Path(inputs["runtime_root"]), inputs["store_id"])
    output = Path(output).absolute()
    if (output.name != "result.json" or output.parent.parent != store.path / "artifacts"
            or not output.parent.name.startswith(".attempt-")):
        raise ResearchError("CONTRACT_MISMATCH", "declared store differs from the owner's attempt output")
    attempt_id = output.parent.name.removeprefix(".attempt-")
    attempt = store.attempts().get(attempt_id)
    if not attempt or running and attempt["state"] != "RUNNING":
        raise ResearchError("CONTRACT_MISMATCH", "actual live attempt is absent")
    events = [event["payload"] for event in store.events() if event["event_kind"] == "ADMISSION"
              and event["payload"].get("attempt_id") == attempt_id]
    if len(events) != 1:
        raise ResearchError("UNAUTHORIZED_DATA", "owner admission is required before dispatch")
    receipt = store.manifest("admission-" + events[0]["admission_hash"])
    expected_command = implementation_hash(resume_command if recovery else command)
    if (receipt["admission_hash"] != digest({k: v for k, v in receipt.items() if k != "admission_hash"})
            or receipt["spec_hash"] != digest(spec) or receipt["cell_hash"] != digest(cell)
            or receipt["spec"] != spec or receipt["cell"] != cell or receipt["attempt_id"] != attempt_id
            or receipt["run_id"] != attempt["run_id"] or receipt["resource_plan"] != plan
            or receipt["command_hash"] != expected_command
            or receipt["execution_kind"] != ("resume" if recovery else "run")
            or receipt["input_kind"] != "registered-protocol"):
        raise ResearchError("CONTRACT_MISMATCH", "actual admission/source/component identity differs")
    job = validate_job(receipt["documents"]["package"]["payload"].get("pirc26_job"), receipt)
    return store, receipt, plugin, job


def _dispatch(output, spec, cell, state, recovery):
    store, receipt, _, _ = admitted_context(output, spec, cell, recovery=recovery)
    settings = spec["admission"]
    transport = read_block(store, settings["protocol_id"], cell["block_id"],
        authorization_id=settings["authorization_id"], authorization_version=settings.get("authorization_version"),
        purpose=settings["purpose"], consumer={"attempt_id": receipt["attempt_id"], "run_id": receipt["run_id"],
                                             "entrypoint": "pirc26-owner-dispatch"})
    content = transport.pop("content_utf8")
    # Bounded canonical frames limit each string; preserve exact UTF-8 text in
    # chunks without another data format or a raw-source path in the worker.
    transport["content_chunks"] = [content[i:i+16384] for i in range(0, len(content), 16384)]
    handoff = {"schema_version": "pirc26-worker-handoff-v1", "spec": spec, "cell": cell,
        "admission_hash": receipt["admission_hash"], "transport": transport,
        "recovery": recovery, "restored_hash": digest(state) if recovery else None}
    directory = Path(output).parent
    atomic_write(directory / "pirc26-handoff.json", canonical(handoff, HANDOFF_LIMIT))
    if recovery:
        atomic_write(directory / "pirc26-restored.json", canonical(state, HANDOFF_LIMIT))
    return [sys.executable, "-m", "infrastructure.pirc26_worker", str(output), digest(handoff)]


def command(output, spec, cell):
    return _dispatch(output, spec, cell, None, False)


def resume_command(output, spec, cell, state):
    return _dispatch(output, spec, cell, state, True)


def load_handoff(output, expected_hash, control):
    """Worker-only read of the owner's regular bounded transport frame."""
    output = Path(output).absolute()
    from infrastructure.pirc26_worker_control import require_owned_worker
    ownership = require_owned_worker(output)
    if control is not None and (output.parent != control.directory or output.parent.name != ".attempt-" + control.descriptor["attempt_id"]
            or control.descriptor["deadline"] != ownership.deadline):
        raise ResearchError("UNAUTHORIZED_DATA", "worker requires the matching owner control directory")
    handoff = read_frame(output.parent / "pirc26-handoff.json", HANDOFF_LIMIT)
    if (type(handoff) is not dict or set(handoff) != {"schema_version", "spec", "cell", "admission_hash", "transport", "recovery", "restored_hash"}
            or handoff["schema_version"] != "pirc26-worker-handoff-v1" or digest(handoff) != expected_hash
            or type(handoff["recovery"]) is not bool):
        raise ResearchError("CONTRACT_MISMATCH", "owner handoff identity differs")
    store, receipt, plugin, job = admitted_context(output, handoff["spec"], handoff["cell"],
                                                   running=True, recovery=handoff["recovery"])
    if control is None and plugin.resume_level != "restart-only":
        raise ResearchError("UNAUTHORIZED_DATA", "optimizer workers require the actual owner checkpoint channel")
    if handoff["admission_hash"] != receipt["admission_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "worker handoff refers to another admission")
    from experiments.pirc25.affine import code_hash
    if code_hash() != receipt["spec"]["code_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "worker numerical source changed after owner admission")
    transport = handoff["transport"]
    if type(transport) is not dict or type(transport.get("content_chunks")) is not list:
        raise ResearchError("CONTRACT_MISMATCH", "bounded owner input chunks required")
    transport["content_utf8"] = "".join(transport.pop("content_chunks"))
    protocol = receipt["documents"]["protocol"]
    block = next(b for b in protocol["blocks"] if b["block_id"] == receipt["cell"]["block_id"])
    from application.research_preregistration import source_identity
    if (transport.get("protocol_hash") != receipt["spec"]["protocol_hash"]
            or transport.get("source_identity") != source_identity(block) or transport.get("block_id") != block["block_id"]
            or transport.get("split_role") != block["split_role"]
            or transport.get("purpose") != receipt["spec"]["admission"]["purpose"]):
        raise ResearchError("CONTRACT_MISMATCH", "worker transport differs from owner-admitted input")
    state = read_frame(output.parent / "pirc26-restored.json", HANDOFF_LIMIT) if handoff["recovery"] else None
    if (digest(state) if handoff["recovery"] else None) != handoff["restored_hash"]:
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "owner restored state identity differs")
    if state is not None:
        from application.pirc26_forecast_control import is_forecast_state, validate_saved_job
        if is_forecast_state(state):
            validate_saved_job(state, job, receipt)
    return receipt, plugin, job, transport, state, ownership
