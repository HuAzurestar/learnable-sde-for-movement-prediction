"""Complete-population O1 adapter on the original admission and supervisor.

No grants/stores/budget arms are created here. Pool forecasts are explicitly
training diagnostics, not held-out evidence. A fitted checkpoint is not
scientific qualification or permission to consume protected data.
"""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from application.pirc26_population import population_source, require
from application.research_recovery import RecoveryPlugin
from application.research_registry import implementation_hash
from infrastructure.research_store import ResearchStore, digest

VERSION = "1.0.0"
PREFIX = "pirc26-population-"
BINDING_SCHEMA = "pirc26-owned-population-binding-v1"


def command(output, spec, cell):
    from application.pirc26_runtime import _dispatch
    return _dispatch(output, spec, cell, None, False)


def resume_command(output, spec, cell, state):
    from application.pirc26_runtime import _dispatch
    return _dispatch(output, spec, cell, state, True)


def execution_plugin(family):
    from application.pirc26_runtime import execution_plugin as base_plugin
    base = base_plugin("O1", family)
    identity = PREFIX + family.lower() + "-o1"
    entry = replace(base.registry_entry, component_id=identity, version=VERSION,
                    code_hash=implementation_hash(command))
    return replace(base, plugin_id=identity, registry_entry=entry, command_builder=command,
                   pre_read_validator=pre_read_validate, checkpoint_validator=checkpoint_validate,
                   result_validator=result_validate, validator_store_context=True)


def recovery_plugin(family):
    plugin = execution_plugin(family)
    require(plugin.resume_level != "restart-only", "basis population training is restart-only", "CHECKPOINT_INCOMPATIBLE")
    return RecoveryPlugin(plugin.plugin_id, "exact", resume_command, VERSION)


def _binding(receipt):
    value = receipt["documents"]["package"]["payload"].get("pirc26_population")
    require(type(value) is dict and set(value) == {"schema_version", "preparation_ref", "population_hash"}
        and value["schema_version"] == BINDING_SCHEMA, "closed complete-population binding required")
    return value


def source(receipt, *, consumer_study_id=None, store=None):
    """Fresh metadata only, before any prospective consumer physical read."""
    from application.pirc26_preparation import _sources, _completion, _expiry
    from application.pirc26_runtime import validate_job
    spec, cell = receipt["spec"], receipt["cell"]
    config, inputs = cell["execution"]["config"], cell["execution"]["inputs"]
    job = receipt["documents"]["package"]["payload"]["pirc26_job"]
    validate_job(job, receipt)
    require(job["operation"] == "fit-and-forecast" and config["objective"] == "O1"
        and receipt["mode"] in {"fixture", "pilot"}
        and not spec.get("comparison_plan", {}).get("adjudication_spec"),
        "population fit is O1 train diagnostics, not held-out adjudication", "UNAUTHORIZED_DATA")
    binding = _binding(receipt)
    if store is None:
        store = ResearchStore(Path(inputs["runtime_root"]), inputs["store_id"])
    require(store.store_id == inputs["store_id"]
        and store.path == Path(inputs["runtime_root"]).resolve() / "pirc25",
        "owner validator store differs from admitted runtime", "CONTRACT_MISMATCH")
    population = population_source(store, binding["preparation_ref"])
    metadata, normalizer = population["provenance"], population["normalizer"]
    require(binding["population_hash"] == digest(metadata), "population membership hash differs")
    blocks = [b for b in receipt["documents"]["protocol"]["blocks"] if b["block_id"] == cell["block_id"]]
    require(blocks == [population["protocol_entry"]], "consumer block differs from actual complete population")
    model_spec = job["initial_checkpoint"]["model_card"]["spec"]
    require(all(model_spec[k] == metadata[k] for k in (
        "coordinate_frame", "train_binding_hash", "normalizer_hash", "context_hash"))
        and model_spec["means"] == normalizer["means"] and model_spec["scales"] == normalizer["scales"]
        and model_spec["context_dim"] == len(normalizer["means"]) - 4,
        "model does not use the actual common train-fitted normalizer")
    batches = sum((s["transitions"] + job["batch_size"] - 1) // job["batch_size"] for s in metadata["segments"])
    require(inputs["observations"] >= metadata["observations"] and inputs["batches"] >= batches
        and inputs["components"] >= 4 + model_spec["context_dim"],
        "complete population exceeds declared observation/batch/context workspace", "RESOURCE_PLAN_REJECTED")
    if "max_steps" in config["plan"]:
        require(config["plan"]["max_steps"] >= batches and config["plan"]["patience"] >= batches,
            "training plan may stop before visiting every registered batch", "RESOURCE_PLAN_REJECTED")
    consumer = spec["study_id"] if consumer_study_id is None else consumer_study_id
    with store._read_transaction():
        request = store._manifest(binding["preparation_ref"]["request_manifest_id"])
        original = store._manifest("study-" + request["original_study_id"])["spec"]
        arm = next(a for a in spec["arms"] if a["arm_id"] == cell["arm_id"])
        require(arm in original["arms"], "population consumer created another budget arm", "CONTRACT_MISMATCH")
        raw_sources = [s for s in request["sources"] if s["split_role"] == "train"]
        def authority():
            require(_sources(store, original, arm, [s["selection"] for s in raw_sources]) == raw_sources,
                "population source authority moved", "UNAUTHORIZED_DATA")
            for item in raw_sources:
                selected = item["selection"]
                grant = store.authorization(
                    selected["authorization_id"], version=selected["authorization_version"])
                require(consumer == original["study_id"] or consumer in grant.get("consumer_study_ids", []),
                    "raw source grant does not permit this population consumer", "UNAUTHORIZED_DATA")
        authority()
        grant = store.authorization(spec["admission"]["authorization_id"], version=spec["admission"].get("authorization_version"))
        def execution_authority():
            current = store.authorization(spec["admission"]["authorization_id"], version=spec["admission"].get("authorization_version"))
            require(current == grant and current["study_id"] == spec["study_id"]
                and current["protocol_hash"] == spec["protocol_hash"] and cell["block_id"] in current["block_ids"]
                and {"execute", "fit"} <= set(current["purposes"])
                and cell["visibility"] in current["visibilities"] and current.get("data_root") == population["data_root"]
                and datetime.fromisoformat(current["expires_at"]) > datetime.now(timezone.utc),
                "complete-population execution grant moved or expired", "UNAUTHORIZED_DATA")
        execution_authority()
        raw_grants = _completion(store, original, arm, raw_sources)
        store._read_completion(lambda: (authority(), execution_authority()), lambda: _expiry([grant]))
    _expiry([*raw_grants, grant])
    return population, {"schema_version": BINDING_SCHEMA, "preparation_ref": binding["preparation_ref"],
        "population_hash": binding["population_hash"], "observations": metadata["observations"],
        "transitions": metadata["transitions"], "batch_count": batches,
        "independent_block_ids": metadata["independent_block_ids"], "use": "training-diagnostics-not-held-out"}


def pre_read_validate(receipt, *, store=None):
    source(receipt, store=store)


def checkpoint_validate(receipt, state, progress, *, store=None):
    from application.pirc26_runtime import validate_checkpoint_progress
    from application.pirc26_forecast_control import is_forecast_state, materialize_fit
    _, expected = source(receipt, store=store)
    validate_checkpoint_progress(receipt, state, progress)
    if is_forecast_state(state):
        require(materialize_fit(state["method_state"]["fit"])["training_population"] == expected,
            "forecast recovery substituted fitted population", "CHECKPOINT_INCOMPATIBLE")


def result_validate(receipt, result, *, store=None):
    from application.pirc26_components import preflight_model
    _, expected = source(receipt, store=store)
    fit = result["fit"]
    require(fit.get("training_population") == expected, "fitted population evidence differs", "CORRUPT_ARTIFACT")
    config, inputs = receipt["cell"]["execution"]["config"], receipt["cell"]["execution"]["inputs"]
    checkpoint = fit["checkpoint"]
    preflight_model(checkpoint, {**config, "initial_model_hash": checkpoint["sha256"]}, inputs)
    if "max_steps" in config["plan"]:
        require(type(fit.get("steps")) is int and fit["steps"] >= expected["batch_count"]
            and fit["steps"] <= config["plan"]["max_steps"], "fitted model omitted registered training batches")


def read_fitted(store, attempt_id, *, consumer_study_id, authorization):
    """Actual supervised fit, measured cost and fresh model disclosure grant."""
    from application.research_admission import AdmissionGate
    from application.research_reuse import verified_reuse
    from application.pirc26_preparation import _expiry
    with store._read_transaction():
        attempt = store._attempts().get(attempt_id)
        require(attempt and attempt["state"] == "SUCCEEDED", "successful population fit absent", "UNQUALIFIED")
        run = store._manifest("run-" + attempt["run_id"])
        spec = store._manifest("study-" + run["study_id"])["spec"]
        cell = run["cell"]
        admissions = [e["payload"] for e in store._events() if e["event_kind"] == "ADMISSION"
                      and e["payload"].get("attempt_id") == attempt_id]
        require(len(admissions) == 1, "unique population fit admission absent", "UNQUALIFIED")
        receipt = store._manifest("admission-" + admissions[0]["admission_hash"])
        plugin = execution_plugin(cell["execution"]["config"]["family"])
        source(receipt, consumer_study_id=consumer_study_id, store=store)
        def authority():
            require(authorization["study_id"] == spec["study_id"] and authorization["protocol_hash"] == spec["protocol_hash"]
                and (consumer_study_id == spec["study_id"] or consumer_study_id in authorization.get("consumer_study_ids", [])),
                "fitted model grant does not authorize the consumer", "UNAUTHORIZED_DATA")
            store.verify_artifact_read(attempt["artifact_id"], purpose="evaluate", authorization=authorization)
        authority()
        verified_reuse(store, attempt, spec, cell, plugin)
        result = json.loads(store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=authorization))
        AdmissionGate(store).result_validator(receipt, spec, cell, plugin)(result)
        source(receipt, consumer_study_id=consumer_study_id, store=store)
        store._read_completion(authority, lambda: _expiry([authorization]))
    _expiry([authorization])
    return {"checkpoint": result["fit"]["checkpoint"], "training_population": result["fit"]["training_population"],
            "producer_attempt_id": attempt_id, "scientific_qualification": "not-established"}
