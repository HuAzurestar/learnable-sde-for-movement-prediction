"""Synthetic operator-side preparation; never imports real research data."""

import hashlib

from application.research_admission import data_binding, package_binding, plugin_binding, command_binding
from application.research_data import EvaluationExposureLedger
from application.research_preregistration import PreregistrationGate, protocol_binding, source_identity
from experiments.pirc25.affine import ROOT, code_hash
from experiments.pirc25.upstream import audit_inputs
from infrastructure.research_store import digest, encode


def admit_fixture(store, value, plugin, root, *, formal=False):
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
        "visibilities": ["synthetic"], "block_ids": [b["block_id"] for b in blocks]}
    store.authorize(grant)
    upstream = audit_inputs(ROOT, ())
    payload = {"synthetic_adapter": plugin.plugin_id}
    package = {"schema_version": "pirc25-package-v1", "kind": "PropagationResult",
        "study_id": value["study_id"],
        "state_order": list(plugin.state_order), "units": list(plugin.units),
        "resume_level": plugin.resume_level, "capabilities": sorted(plugin.capabilities),
        "code_hash": value["code_hash"], "data_hash": value["data_hash"], "input_hash": value["data_hash"],
        "protocol_hash": value["protocol_hash"], "output_hash": digest(payload), "payload": payload,
        "plugin_hash": plugin_binding(plugin), "command_hash": command_binding(plugin.command_builder),
        "recovery_command_hash": command_binding(plugin.command_builder), "upstream_hash": upstream["manifest_hash"],
        "qualification": "qualified" if formal else "fixture"}
    if formal:
        package["preregistration_hash"] = protocol["preregistration_hash"]
        check = {"check_id": "contract", "outcome": "passed", "package_binding": package_binding(package),
                 "code_hash": value["code_hash"], "preregistration_hash": protocol["preregistration_hash"]}
        artifact = store.artifact(encode(check), role="qualification", visibility="synthetic",
                                   block_ids=grant["block_ids"], study_id=value["study_id"])
        qualification = {"schema_version": "pirc25-qualification-v1", "status": "passed",
            "package_binding": package_binding(package), "code_hash": value["code_hash"],
            "preregistration_hash": protocol["preregistration_hash"],
            "checks": [{"check_id": "contract", "artifact_id": artifact["artifact_id"]}]}
        package["qualification_hash"] = store.publish("qualification-" + digest(qualification), qualification)
    reference = store.publish("package-" + digest(package), package)
    value["admission"] = {"mode": "formal" if formal else "fixture", "protocol_id": "inputs",
        "authorization_id": grant["authorization_id"], "package_hash": reference,
        "upstream_ids": [], "upstream_hash": upstream["manifest_hash"], "purpose": "evaluate" if formal else "fit"}
    return grant


def attach_foreign_model(store, value, *, consumer=True, export=True):
    """Synthetic prior-study qualification imported for a propagation consumer."""
    package = store.manifest("package-" + value["admission"]["package_hash"])
    prereg = store.manifest("preregistration-" + package["preregistration_hash"])
    source_protocol = {**store.manifest("protocol-inputs"), "study_id": "model-study", "protocol_id": "model-inputs"}
    source_plan = {**prereg, "study_ids": ["model-study"], "protocol_bindings": [protocol_binding(source_protocol)]}
    plan_hash = PreregistrationGate(store).register_preregistration(source_plan, digest(source_plan))
    source_protocol["preregistration_hash"] = plan_hash
    EvaluationExposureLedger(store).register_protocol(source_protocol, digest(source_protocol))
    model = {**package, "kind": "FrozenDynamicsPackage", "study_id": "model-study",
             "preregistration_hash": plan_hash, "protocol_hash": digest(source_protocol),
             "data_hash": data_binding(source_protocol), "input_hash": data_binding(source_protocol)}
    model.pop("qualification_hash")

    def qualify(candidate, plan_reference):
        check = {"check_id": "contract", "outcome": "passed", "package_binding": package_binding(candidate),
                 "code_hash": candidate["code_hash"], "preregistration_hash": plan_reference}
        artifact = store.artifact(encode(check), role="qualification", visibility="synthetic",
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
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("synthetic model-owner permission"),
        "consumer_study_ids": [value["study_id"]] if consumer else [],
        "purposes": ["evaluate", "export"] if export else ["evaluate"],
        "visibilities": ["synthetic"], "block_ids": ["fixture-1"]}
    store.authorize(grant)
    value["admission"]["model_authorization_id"] = grant["authorization_id"]
    value["admission"]["model_protocol_id"] = "model-inputs"
