"""Separate owned selection forecast with actual fitted-model authority.

Not a fitting, normalizer, scientific qualification or final/test route.
All numeric work remains on the original shared supervisor and budget arm.
"""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from application.pirc26_fitted_model import model_source, read_model
from application.pirc26_population import require
from application.pirc26_preparation import _expiry
from application.pirc26_selection import selection_source
from application.research_recovery import RecoveryPlugin
from application.research_registry import implementation_hash
from infrastructure.research_store import ResearchStore, digest
from infrastructure.research_visibility import combine_visibility, study_visibility

VERSION = "1.0.0"
PREFIX = "pirc26-selection-"
BINDING_SCHEMA = "pirc26-owned-selection-binding-v1"


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
    require(plugin.resume_level != "restart-only", "basis selection is restart-only", "CHECKPOINT_INCOMPATIBLE")
    return RecoveryPlugin(plugin.plugin_id, "exact", resume_command, VERSION)


def source(receipt, *, store=None):
    """Metadata-only preflight, also repeated for result/reuse/recovery."""
    from application.pirc26_runtime import validate_job
    spec, cell = receipt["spec"], receipt["cell"]
    inputs = cell["execution"]["inputs"]
    job = receipt["documents"]["package"]["payload"]["pirc26_job"]
    validate_job(job, receipt)
    require(job["operation"] == "forecast" and receipt["mode"] in {"fixture", "pilot"}
        and spec["admission"]["purpose"] == "select" and cell["execution"]["config"]["objective"] == "O1"
        and not spec.get("comparison_plan", {}).get("adjudication_spec"),
        "unqualified attachment supports selection-only frozen inference", "UNQUALIFIED")
    binding = receipt["documents"]["package"]["payload"].get("pirc26_selection")
    require(type(binding) is dict and set(binding) == {"schema_version", "preparation_ref", "independent_block_id",
        "population_hash", "model_ref", "model_authorization"} and binding["schema_version"] == BINDING_SCHEMA,
        "closed selection/model binding required")
    model_authorization = binding["model_authorization"]
    require(type(model_authorization) is dict and set(model_authorization) == {
        "authorization_id", "authorization_version"}, "explicit model authorization reference required")
    if store is None:
        store = ResearchStore(Path(inputs["runtime_root"]), inputs["store_id"])
    require(store.store_id == inputs["store_id"]
        and store.path == Path(inputs["runtime_root"]).resolve() / "pirc25",
        "selection owner store differs from admitted runtime", "CONTRACT_MISMATCH")
    with store._read_transaction():
        arm = next(a for a in spec["arms"] if a["arm_id"] == cell["arm_id"])
        selected = selection_source(store, binding["preparation_ref"], binding["independent_block_id"],
            consumer_study_id=spec["study_id"], arm=arm)
        metadata, normalizer = selected["provenance"], selected["normalizer"]
        require(binding["population_hash"] == digest(metadata)
            and [b for b in receipt["documents"]["protocol"]["blocks"] if b["block_id"] == cell["block_id"]]
                == [selected["protocol_entry"]], "selection consumer changed complete original unit")
        disclosure = store.authorization(model_authorization["authorization_id"],
            version=model_authorization["authorization_version"])
        model = model_source(store, binding["model_ref"], consumer_study_id=spec["study_id"], authorization=disclosure)
        checkpoint = job["initial_checkpoint"]
        dynamics = checkpoint["model_card"]["spec"]
        require(checkpoint["sha256"] == model["checkpoint_hash"] and checkpoint["model_card"] == model["model_card"]
            and model["training_population"]["preparation_ref"] == binding["preparation_ref"]
            and model["cost"]["arm_id"] == cell["arm_id"]
            and all(dynamics[k] == metadata[k] for k in (
                "coordinate_frame", "train_binding_hash", "normalizer_hash", "context_hash"))
            and dynamics["means"] == normalizer["means"] and dynamics["scales"] == normalizer["scales"]
            and dynamics["context_dim"] == len(normalizer["means"]) - 4,
            "selection did not retain the actual fitted model/train transform")
        require(inputs["observations"] >= metadata["observations"] and inputs["components"] >= len(normalizer["means"])
            and {r["segment_id"] for r in job["origins"]} == {s["pooled_segment_id"] for s in metadata["segments"]},
            "selection job omits a declared segment or exceeds capacity", "RESOURCE_PLAN_REJECTED")
        artifact = store._manifest("artifact-" + selected["protocol_entry"]["path"])
        original = store._manifest("study-" + metadata["source_study_id"])["spec"]
        visibility = combine_visibility([artifact["visibility"], model["visibility"],
            study_visibility(store._manifest, original), study_visibility(store._manifest, spec)])
        require(cell["visibility"] == receipt["documents"]["package"]["visibility"] == visibility,
            "selection consumer would widen source/model visibility", "UNAUTHORIZED_DATA")
        settings = spec["admission"]
        grant = store.authorization(settings["authorization_id"], version=settings.get("authorization_version"))
        def authority():
            current = store.authorization(settings["authorization_id"], version=settings.get("authorization_version"))
            require(current == grant and grant["study_id"] == spec["study_id"]
                and grant["protocol_hash"] == spec["protocol_hash"] and cell["block_id"] in grant["block_ids"]
                and {"execute", "select"} <= set(grant["purposes"]) and visibility in grant["visibilities"]
                and grant.get("data_root") == selected["data_root"]
                and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc),
                "separate selection execution authority moved or expired", "UNAUTHORIZED_DATA")
        authority()
        store._read_completion(authority, lambda: _expiry([grant, disclosure]))
    _expiry([grant, disclosure])
    evidence = {"schema_version": BINDING_SCHEMA, "preparation_ref": binding["preparation_ref"],
        "population_hash": binding["population_hash"], "independent_block_id": binding["independent_block_id"],
        "model_ref": binding["model_ref"], "checkpoint_hash": model["checkpoint_hash"],
        "observations": metadata["observations"], "segment_ids": [s["pooled_segment_id"] for s in metadata["segments"]],
        "use": "selection-only-not-final-evidence", "scientific_qualification": "not-established"}
    return selected, evidence


def pre_read_validate(receipt, *, store=None):
    source(receipt, store=store)


def read_attached_model(store, receipt):
    source(receipt, store=store)
    binding = receipt["documents"]["package"]["payload"]["pirc26_selection"]
    authorization = binding["model_authorization"]
    body = read_model(store, binding["model_ref"], consumer_study_id=receipt["spec"]["study_id"],
        authorization=store.authorization(authorization["authorization_id"], version=authorization["authorization_version"]))
    require(body["checkpoint"] == receipt["documents"]["package"]["payload"]["pirc26_job"]["initial_checkpoint"],
        "actual authorized attachment differs from immutable forecast checkpoint", "CORRUPT_ARTIFACT")
    source(receipt, store=store)


def _frozen(receipt, fit, expected):
    require(fit.get("selection_input") == expected and fit.get("status") == "FROZEN"
        and fit.get("checkpoint") == receipt["documents"]["package"]["payload"]["pirc26_job"]["initial_checkpoint"]
        and not {"history", "steps", "objective", "training_population"} & set(fit),
        "selection substituted/refitted the frozen model or unit", "CORRUPT_ARTIFACT")


def checkpoint_validate(receipt, state, progress, *, store=None):
    from application.pirc26_runtime import validate_checkpoint_progress
    from application.pirc26_forecast_control import is_forecast_state, materialize_fit
    _, expected = source(receipt, store=store)
    require(is_forecast_state(state), "selection recovery cannot load training state", "CHECKPOINT_INCOMPATIBLE")
    validate_checkpoint_progress(receipt, state, progress)
    _frozen(receipt, materialize_fit(state["method_state"]["fit"]), expected)


def result_validate(receipt, result, *, store=None):
    _, expected = source(receipt, store=store)
    _frozen(receipt, result["fit"], expected)
