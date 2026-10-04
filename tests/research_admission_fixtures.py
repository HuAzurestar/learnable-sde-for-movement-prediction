"""Synthetic operator-side preparation; never imports real research data."""

import hashlib

from application.research_contracts import ExecutionPlugin
from application.research_execution import execution_binding
from application.research_registry import GLOBAL_LIMITS, RegistryEntry, implementation_hash

from application.research_admission import data_binding, package_binding, plugin_binding, command_binding
from application.research_data import EvaluationExposureLedger
from application.research_preregistration import PreregistrationGate, protocol_binding, source_identity
from application.research_upstream import register_acceptance_catalog, study_binding
from experiments.pirc25.affine import ROOT, code_hash
from experiments.pirc25.upstream import audit_inputs
from experiments.pirc25.snapshot import UpstreamSnapshot
from infrastructure.research_store import digest, encode


def reseal_substituted_upstream_spec(receipt):
    """Forge only a disposable transport claim to reach older refusal gates.

    Never publishes to the actual store or alters read/qualification/preregistration
    evidence. This is adversarial test preparation, not an admission repair.
    """
    evidence = receipt["documents"]["upstream_snapshot"]
    evidence["validation"]["consumer"]["spec_hash"] = digest(receipt["spec"])
    evidence["validation_hash"] = digest(evidence["validation"])
    publication = evidence["publication_events"]["validation"]
    publication["payload"] = {"object_id": "upstream-validation-" + evidence["validation_hash"],
                              "sha256": evidence["validation_hash"]}
    publication["hash"] = digest({key: value for key, value in publication.items() if key != "hash"})
    event = evidence["validation_event"]
    event["payload"]["validation_hash"] = evidence["validation_hash"]
    event["previous_hash"] = publication["hash"]
    event["hash"] = digest({key: value for key, value in event.items() if key != "hash"})


def synthetic_plugin(plugin_id, capabilities, state_order, units, resume_level, builder, *, version="1.0.0"):
    """Explicit tiny scalar/empty-forecast test recipe; not a research default."""
    entry = RegistryEntry(component_id=plugin_id, component_kind="execution-adapter", version=version,
        code_hash=implementation_hash(builder), config_schema={"type": "object", "additionalProperties": False},
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": True},
        state_order=state_order, units=units, capabilities=capabilities, resource_class="cpu", resume_level=resume_level,
        resource_contract={"schema_version": "pirc25-resource-contract-v1", "counts": {
            **{key: {"constant": 1} for key in ("paths", "steps", "mixtures", "components", "observations")},
            "state_dim": {"constant": len(state_order)}},
            "tensors": [{"name": "fixture_scalars", "axes": ["state_dim"], "item_bytes": 8}],
            "limits": {**dict(GLOBAL_LIMITS), "result_bytes": 64 * 1024}})
    return ExecutionPlugin(plugin_id, capabilities, state_order, units, resume_level, builder, entry)


def bind_fixture_execution(value, plugin, config=None, inputs=None):
    for cell in value["cells"]:
        if cell.get("plugin_id") == plugin.plugin_id:
            cell["resource_class"] = plugin.registry_entry.resource_class
            cell["execution"] = execution_binding(plugin.registry_entry, config or {}, inputs or {}, matrix_cells=len(value["cells"]))


def admit_fixture(store, value, plugin, root, *, formal=False, package_visibility="synthetic",
                  execution_config=None, execution_inputs=None, recovery_command_builder=None,
                  upstream_ids=(), upstream_inputs=None, accepted_versions=None,
                  upstream_dependencies=None, pirc22_cutover=None, upstream_visibility="synthetic", legacy_upstream=True):
    bind_fixture_execution(value, plugin, execution_config, execution_inputs)
    # Explicit engineering-only frozen metadata. These synthetic attestations
    # never claim acceptance of real PIRC-19--22 inputs or grant data access.
    if upstream_inputs is None:
        upstream_inputs = []
        for name in upstream_ids:
            metadata = {"schema_version": "synthetic-upstream-metadata-v1", "status": "synthetic-control"}
            metadata_path = root / ("synthetic-upstream-" + name + ".json")
            metadata_path.write_bytes(encode(metadata))
            upstream_inputs.append({"object_id": name, "issue": "synthetic-control",
                "acceptance_commit": hashlib.sha1(b"synthetic operator acceptance control, not Git proof").hexdigest(),
                "code_sha": hashlib.sha1(b"synthetic metadata code control").hexdigest(),
                "schema_version": metadata["schema_version"], "artifact_id": "synthetic-" + name,
                "artifact_hash": hashlib.sha256(encode(metadata)).hexdigest(), "artifact_size_bytes": len(encode(metadata)),
                "path": metadata_path.name, "format": "json-metadata", "role": "metadata-only", "kind": "metadata",
                "license": {"id": "synthetic-control", "scope": "metadata-only"}, "metadata_checks": metadata,
                **{key: {"not_applicable": "synthetic metadata control, no research input"}
                   for key in ("data_hash", "split_hash", "fold_hash", "feature_hash", "selection_hash")}})
    if accepted_versions is None:
        accepted_versions = [{"status": "accepted", "input": record} for record in upstream_inputs]
    source_evidence = {"entries": accepted_versions, "kind": "explicit synthetic operator attestation"}
    catalog = {"schema_version": "pirc25-upstream-acceptance-v1", "source": "synthetic test control only",
               "visibility": upstream_visibility,
               "entries": accepted_versions, "source_evidence": source_evidence,
               "source_evidence_hash": digest(source_evidence)}
    catalog_hash = register_acceptance_catalog(store, catalog, digest(catalog))
    study = {"study_id": value["study_id"], "pirc22_cutover": pirc22_cutover or {"mode": "not_applicable"}}
    snapshot = UpstreamSnapshot({"schema_version": "pirc25-upstream-snapshot-v1", "inputs": upstream_inputs,
        "visibility": upstream_visibility,
        "studies": [study], "cells": [{"cell_id": digest(cell), "study_id": value["study_id"],
            "role": cell.get("study_role", "primary"),
            **({"conditioner_binding": cell["conditioner_binding"]} if "conditioner_binding" in cell else {}),
            "upstream_ids": (list(upstream_dependencies[cell["arm_id"]]) if upstream_dependencies is not None
                             else [record["object_id"] for record in upstream_inputs])} for cell in value["cells"]]})
    snapshot_hash = store.publish("upstream-snapshot-" + snapshot.snapshot_hash, snapshot.definition)
    content = b"explicit synthetic plugin input"
    (root / "plugin-input.bin").write_bytes(content)
    blocks = [{"block_id": block, "dataset_id": "synthetic", "release_id": "v1",
               "source_block_id": block, "sha256": hashlib.sha256(content).hexdigest(),
               "path": "plugin-input.bin", "split_role": "final-eval" if formal else "train",
               "fit_scope": not formal} for block in sorted({cell["block_id"] for cell in value["cells"]})]
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "inputs",
                "study_id": value["study_id"], "blocks": blocks}
    if formal:
        gate = PreregistrationGate(store)
        history = {"records": [{**source_identity(block), "status": "unexposed"} for block in blocks]}
        report = {"schema_version": "pirc25-exposure-history-v1", "source": "synthetic fixture history",
                  "source_evidence": history, "source_evidence_hash": digest(history)}
        protocol["history_hash"] = gate.import_history(report, digest(report))
        plan = {"schema_version": "pirc25-preregistration-v1", "test_mode": "blind",
                "study_ids": [value["study_id"]], "protocol_bindings": [protocol_binding(protocol)],
                "primary_metrics": ["error"], "selection_rule": "fixed synthetic adapters",
                "stopping_rule": "fixed matrix", "qualification_checks": ["contract"],
                "upstream_bindings": {value["study_id"]: study_binding(snapshot_hash, catalog_hash, study)},
                "comparisons": [{"reference": value["arms"][0]["arm_id"],
                                 "candidates": [a["arm_id"] for a in value["arms"][1:]]}]}
        protocol["preregistration_hash"] = gate.register_preregistration(plan, digest(plan))
        value["comparison_plan"] = {"reference_arm_id": value["arms"][0]["arm_id"],
            "candidate_arm_ids": [a["arm_id"] for a in value["arms"][1:]],
            "preregistration_hash": digest(plan), "failure_policy": "retain-and-exclude-incomplete-blocks"}
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    value.update(code_hash=code_hash(), protocol_hash=digest(protocol), data_hash=data_binding(protocol))
    grant = {"authorization_id": "execution-fixture", "study_id": value["study_id"],
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("explicit synthetic operator grant"),
        "protocol_hash": digest(protocol), "data_root": str(root), "test_authorization": formal,
        "purposes": ["fit", "execute", "evaluate", "preview", "export", "resume"],
        "visibilities": sorted({"synthetic", package_visibility}), "block_ids": [b["block_id"] for b in blocks]}
    store.authorize(grant)
    upstream = audit_inputs(ROOT, tuple(upstream_ids))
    payload = {"synthetic_adapter": plugin.plugin_id}
    package = {"schema_version": "pirc25-package-v1", "kind": "PropagationResult",
        "study_id": value["study_id"], "visibility": package_visibility,
        "state_order": list(plugin.state_order), "units": list(plugin.units),
        "resume_level": plugin.resume_level, "capabilities": sorted(plugin.capabilities),
        "code_hash": value["code_hash"], "data_hash": value["data_hash"], "input_hash": value["data_hash"],
        "protocol_hash": value["protocol_hash"], "output_hash": digest(payload), "payload": payload,
        "plugin_hash": plugin_binding(plugin), "command_hash": command_binding(plugin.command_builder),
        "recovery_command_hash": command_binding(recovery_command_builder or plugin.command_builder), "upstream_hash": upstream["manifest_hash"],
        "upstream_snapshot_hash": snapshot_hash, "upstream_acceptance_hash": catalog_hash,
        "qualification": "qualified" if formal else "fixture"}
    if not legacy_upstream:
        package.pop("upstream_hash")
    if formal:
        package["preregistration_hash"] = protocol["preregistration_hash"]
        check = {"check_id": "contract", "outcome": "passed", "package_binding": package_binding(package),
                 "code_hash": value["code_hash"], "preregistration_hash": protocol["preregistration_hash"]}
        artifact = store.artifact(encode(check), role="qualification", visibility=package_visibility,
                                   block_ids=grant["block_ids"], study_id=value["study_id"])
        qualification = {"schema_version": "pirc25-qualification-v1", "status": "passed",
            "package_binding": package_binding(package), "code_hash": value["code_hash"],
            "preregistration_hash": protocol["preregistration_hash"],
            "checks": [{"check_id": "contract", "artifact_id": artifact["artifact_id"]}]}
        package["qualification_hash"] = store.publish("qualification-" + digest(qualification), qualification)
    reference = store.publish("package-" + digest(package), package)
    value["admission"] = {"mode": "formal" if formal else "fixture", "protocol_id": "inputs",
        "authorization_id": grant["authorization_id"], "package_hash": reference,
        "upstream_snapshot_hash": snapshot_hash, "upstream_acceptance_hash": catalog_hash, "upstream_root": str(root),
        "upstream_ids": list(upstream_ids), "upstream_hash": upstream["manifest_hash"], "purpose": "evaluate" if formal else "fit"}
    if not legacy_upstream:
        value["admission"].pop("upstream_ids")
        value["admission"].pop("upstream_hash")
    return grant


def attach_foreign_model(store, value, *, consumer=True, export=True, visibility="synthetic"):
    """Synthetic prior-study qualification imported for a propagation consumer."""
    package = store.manifest("package-" + value["admission"]["package_hash"])
    prereg = store.manifest("preregistration-" + package["preregistration_hash"])
    source_protocol = {**store.manifest("protocol-inputs"), "study_id": "model-study", "protocol_id": "model-inputs"}
    source_plan = {**prereg, "study_ids": ["model-study"], "protocol_bindings": [protocol_binding(source_protocol)]}
    plan_hash = PreregistrationGate(store).register_preregistration(source_plan, digest(source_plan))
    source_protocol["preregistration_hash"] = plan_hash
    EvaluationExposureLedger(store).register_protocol(source_protocol, digest(source_protocol))
    model = {**package, "kind": "FrozenDynamicsPackage", "study_id": "model-study", "visibility": visibility,
             "preregistration_hash": plan_hash, "protocol_hash": digest(source_protocol),
             "data_hash": data_binding(source_protocol), "input_hash": data_binding(source_protocol)}
    model.pop("qualification_hash")

    def qualify(candidate, plan_reference):
        check = {"check_id": "contract", "outcome": "passed", "package_binding": package_binding(candidate),
                 "code_hash": candidate["code_hash"], "preregistration_hash": plan_reference}
        artifact = store.artifact(encode(check), role="qualification", visibility=candidate["visibility"],
                                  block_ids=["fixture-1"], study_id=candidate["study_id"])
        report = {"schema_version": "pirc25-qualification-v1", "status": "passed",
                  "package_binding": package_binding(candidate), "code_hash": candidate["code_hash"],
                  "preregistration_hash": plan_reference,
                  "checks": [{"check_id": "contract", "artifact_id": artifact["artifact_id"]}]}
        candidate["qualification_hash"] = store.publish("qualification-" + digest(report), report)
        return store.publish("package-" + digest(candidate), candidate)

    model_hash = qualify(model, plan_hash)
    package.pop("qualification_hash")
    package.update(requires_frozen_model=True, model_hash=model_hash)
    value["admission"]["package_hash"] = qualify(package, package["preregistration_hash"])
    grant = {"authorization_id": "model-consumer", "study_id": "model-study",
        "protocol_hash": digest(source_protocol),
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("synthetic model-owner permission"),
        "consumer_study_ids": [value["study_id"]] if consumer else [],
        "purposes": ["evaluate", "export"] if export else ["evaluate"],
        "visibilities": sorted({"synthetic", visibility}), "block_ids": ["fixture-1"]}
    store.authorize(grant)
    value["admission"]["model_authorization_id"] = grant["authorization_id"]
    value["admission"]["model_protocol_id"] = "model-inputs"
