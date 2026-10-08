"""Execution admission owned by the supervisor, never by a worker result.

Registered grants and imported qualification reports are operator-controlled
inputs. Hashes prove bindings, not the honesty of off-platform attestations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .research_contracts import validate_package, validate_result
from .research_data import EvaluationExposureLedger
from .research_preregistration import PreregistrationGate, hash_reference, source_identity, validate_preregistration, protocol_binding
from infrastructure.research_store import ResearchError, digest, encode, utc_now
from infrastructure.research_admission_selection import select_admission_package, verify_admission_selection


def data_binding(protocol):
    return digest([{**source_identity(block), "block_id": block["block_id"],
                    "split_role": block["split_role"]} for block in protocol["blocks"]])


def package_binding(package):
    return digest({key: value for key, value in package.items() if key != "qualification_hash"})


def command_binding(builder):
    from .research_registry import implementation_hash
    return implementation_hash(builder)


def plugin_binding(plugin):
    return digest({"plugin_id": plugin.plugin_id, "capabilities": sorted(plugin.capabilities),
                   "state_order": list(plugin.state_order), "units": list(plugin.units),
                   "resume_level": plugin.resume_level, "command_hash": command_binding(plugin.command_builder),
                   "registry_entry_hash": digest(plugin.registry_entry.manifest())})


class AdmissionGate:
    def __init__(self, store):
        self.store = store

    def _document(self, kind, reference):
        if not hash_reference(reference):
            raise ResearchError("MISSING_INPUT", "registered " + kind + " hash required")
        value = self.store.manifest(kind + "-" + reference)
        if digest(value) != reference:
            raise ResearchError("CONTRACT_MISMATCH", kind + " content binding differs")
        return value

    def _qualification(self, package, prereg, grant):
        validate_preregistration(prereg)
        if package.get("study_id") not in prereg["study_ids"] or grant["study_id"] != package.get("study_id"):
            raise ResearchError("UNQUALIFIED", "qualification source study differs")
        report = self._document("qualification", package.get("qualification_hash"))
        required = prereg.get("qualification_checks")
        if (not isinstance(required, list) or not required or len(set(required)) != len(required)
                or any(not isinstance(name, str) or not name for name in required)):
            raise ResearchError("UNQUALIFIED", "preregistered qualification checks required")
        if (report.get("schema_version") != "pirc25-qualification-v1" or report.get("status") != "passed"
                or report.get("package_binding") != package_binding(package)
                or report.get("code_hash") != package["code_hash"]
                or report.get("preregistration_hash") != digest(prereg)):
            raise ResearchError("UNQUALIFIED", "qualification report bindings differ")
        checks = report.get("checks", [])
        if (len(checks) != len(required) or {c.get("check_id") for c in checks} != set(required)):
            raise ResearchError("UNQUALIFIED", "qualification report omits preregistered checks")
        evidence = []
        for check in checks:
            metadata = self.store.manifest("artifact-" + check["artifact_id"])
            if metadata["role"] != "qualification" or metadata["size_bytes"] > 2 * 1024 * 1024:
                raise ResearchError("UNQUALIFIED", "qualification artifact role/size invalid")
            content = self.store.read_artifact(check["artifact_id"], purpose="evaluate", authorization=grant)
            value = json.loads(content)
            if content != encode(value):
                raise ResearchError("CONTRACT_MISMATCH", "qualification evidence must use canonical JSON encoding")
            expected = {"check_id": check["check_id"], "outcome": "passed",
                        "package_binding": package_binding(package), "code_hash": package["code_hash"],
                        "preregistration_hash": digest(prereg)}
            if any(value.get(key) != item for key, item in expected.items()):
                raise ResearchError("UNQUALIFIED", "qualification check did not pass on the bound inputs")
            evidence.append({"artifact": metadata, "content": value})
        return report, evidence

    def _model_documents(self, model, settings, spec, grant):
        """Model consumption permission is mandatory independently of run mode."""
        model_grant = grant
        if model.get("study_id") != spec["study_id"]:
            model_grant = self.store.authorization(settings["model_authorization_id"],
                version=settings.get("model_authorization_version"))
            if spec["study_id"] not in model_grant.get("consumer_study_ids", []):
                raise ResearchError("UNAUTHORIZED_DATA", "frozen model grant does not authorize this consumer study")
        if (model_grant.get("study_id") != model.get("study_id")
                or model_grant.get("protocol_hash") != model.get("protocol_hash")
                or "evaluate" not in model_grant.get("purposes", [])
                or datetime.fromisoformat(model_grant["expires_at"]) <= datetime.now(timezone.utc)
                or model.get("visibility", "restricted") not in model_grant["visibilities"]):
            raise ResearchError("UNAUTHORIZED_DATA", "model source grant scope, purpose or expiry differs")
        if not isinstance(model.get("payload"), dict) or digest(model["payload"]) != model["output_hash"]:
            raise ResearchError("UNQUALIFIED", "frozen model payload binding differs")
        model_prereg = self._document("preregistration", model.get("preregistration_hash"))
        model_protocol = self.store.manifest("protocol-" + settings["model_protocol_id"])
        if (model_protocol.get("schema_version") != "pirc25-data-protocol-v1"
                or model_protocol["study_id"] != model["study_id"]
                or digest(model_protocol) != model["protocol_hash"]
                or data_binding(model_protocol) != model["data_hash"]
                or model_protocol.get("preregistration_hash") != model["preregistration_hash"]
                or protocol_binding(model_protocol) not in model_prereg["protocol_bindings"]):
            raise ResearchError("UNQUALIFIED", "frozen model source protocol binding differs")
        model_report, model_evidence = self._qualification(model, model_prereg, model_grant)
        return {"frozen_model": model, "model_preregistration": model_prereg,
                "model_qualification": model_report, "model_qualification_evidence": model_evidence,
                "model_authorization": model_grant, "model_protocol": model_protocol}

    def prepare(self, spec, cell, plugin, attempt_id, *, builtin_fixture=False, recovery_builder=None):
        from experiments.pirc25.affine import ROOT, code_hash, fixture_spec
        from experiments.pirc25.upstream import audit_inputs
        from experiments.pirc27.calibration_bindings import require_consumer_support

        calibration = require_consumer_support(spec, cell)
        if calibration is not None and builtin_fixture:
            raise ResearchError("UNQUALIFIED", "calibrated geometry requires a settled consumer, not the builtin fixture")

        if spec["code_hash"] != code_hash():
            raise ResearchError("CONTRACT_MISMATCH", "execution code differs from registered code hash")
        from .research_execution import execution_plan
        plan = execution_plan(spec, cell, plugin)
        attempt = self.store.attempts()[attempt_id]
        run = self.store.manifest("run-" + attempt["run_id"])
        if run["spec_hash"] != digest(spec) or run["cell_hash"] != digest(cell):
            raise ResearchError("CONTRACT_MISMATCH", "admission attempt differs from registered input")
        receipt = {"schema_version": "pirc25-admission-v1", "attempt_id": attempt_id,
                   "run_id": attempt["run_id"], "spec_hash": digest(spec), "cell_hash": digest(cell),
                   "spec": spec, "cell": cell, "plugin_hash": plugin_binding(plugin),
                   "execution_kind": "resume" if recovery_builder is not None else "run",
                   "command_hash": command_binding(recovery_builder or plugin.command_builder),
                   "registry_entry": plugin.registry_entry.manifest(), "resource_plan": plan}
        entry_hash = plan["registry_entry_hash"]
        self.store.publish("registry-entry-" + entry_hash, receipt["registry_entry"])
        self.store.publish("registry-version-" + digest({"id": plugin.plugin_id, "version": plugin.registry_entry.version}),
            {"component_id": plugin.plugin_id, "component_version": plugin.registry_entry.version, "registry_entry_hash": entry_hash})
        for component in plan.get("composition", {}).get("entries", {}).values():
            component_hash = digest(component)
            self.store.publish("registry-entry-" + component_hash, component)
            self.store.publish("registry-version-" + digest({"id": component["component_id"], "version": component["version"]}),
                {"component_id": component["component_id"], "component_version": component["version"], "registry_entry_hash": component_hash})
        if builtin_fixture:
            expected = fixture_spec(spec["study_id"], cell["dimensions"], tuple(c["seed"] for c in spec["cells"]))
            if spec != expected or cell.get("visibility") != "synthetic":
                raise ResearchError("CONTRACT_MISMATCH", "built-in fixture input is not the frozen synthetic recipe")
            required = ("affine-4d",) if cell["dimensions"] == 4 else ("upstream-completion",)
            receipt.update(mode="fixture", qualification="fixture", input_kind="builtin-affine-generator",
                           upstream=audit_inputs(ROOT, required))
        else:
            settings, selection = select_admission_package(spec, cell)
            if selection is not None:
                receipt["admission_selection"] = selection
            mode = settings.get("mode")
            if mode not in {"fixture", "pilot", "formal"}:
                raise ResearchError("MISSING_INPUT", "explicit execution admission mode required")
            protocol = self.store.manifest("protocol-" + settings["protocol_id"])
            if digest(protocol) != spec["protocol_hash"] or protocol["study_id"] != spec["study_id"]:
                raise ResearchError("CONTRACT_MISMATCH", "execution data protocol binding differs")
            if data_binding(protocol) != spec["data_hash"]:
                raise ResearchError("CONTRACT_MISMATCH", "execution data identities differ")
            selected = [b for b in protocol["blocks"] if b["block_id"] == cell["block_id"]]
            if len(selected) != 1:
                raise ResearchError("MISSING_INPUT", "cell block absent from frozen protocol")
            block = selected[0]
            from experiments.pirc27.input_cases import validate_cell_input_binding
            validate_cell_input_binding(cell, protocol_block=block, selection_hash=spec.get("selection_hash"))
            if plugin.plugin_id in {"affine-cubature", "affine-cubature-qualification"}:
                # A numerical qualifier never becomes a formal target. Ordinary
                # affine targets need the dedicated owner proof below, pre-read.
                if (plugin.plugin_id == "affine-cubature-qualification" and
                        (mode == "formal" or block["split_role"] in {"test", "final-eval"})) or (
                        plugin.plugin_id == "affine-cubature" and mode != "formal"
                        and block["split_role"] in {"test", "final-eval"}):
                    raise ResearchError("UNQUALIFIED", "affine cubature requires dedicated held-out owner qualification")
                if plugin.plugin_id == "affine-cubature-qualification" and (
                        mode != "pilot" or block["split_role"] not in {"train", "validation"}):
                    raise ResearchError("UNAUTHORIZED_DATA", "cubature qualification requires train/validation pilot inputs")
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            if plugin.plugin_id in {"affine-mixture-chunk", "synthetic-mixture-chunk"}:
                # A generic operator report is not a method-specific qualifier.
                # Refuse BEFORE grant/package lookup and protected input reads.
                if mode == "formal" or block["split_role"] in {"test", "final-eval"}:
                    raise ResearchError("UNQUALIFIED", "bounded mixture has no dedicated held-out qualification")
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            if plugin.plugin_id == "affine-propagation-qualification":
                # Numerical qualification is real pilot computation, never
                # free preflight work and never exposure to held-out test data.
                if mode != "pilot" or block["split_role"] not in {"train", "validation"}:
                    raise ResearchError("UNAUTHORIZED_DATA", "analytic qualification cannot consume test/final-eval inputs")
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            if plugin.plugin_id == "affine-halfspace-calibration":
                # Refuse before any grant lookup, qualification attachment or
                # protected input. Geometry calibration is charged pilot work.
                if mode != "pilot" or block["split_role"] not in {"train", "validation"}:
                    raise ResearchError("UNAUTHORIZED_DATA", "probability calibration cannot consume test/final-eval inputs")
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            if plugin.plugin_id == "affine-mixture-qualification":
                if mode != "pilot" or block["split_role"] not in {"train", "validation"}:
                    raise ResearchError("UNAUTHORIZED_DATA", "mixture qualification cannot consume test/final-eval inputs")
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            if plugin.plugin_id == "affine-path-qualification":
                if mode != "pilot" or block["split_role"] not in {"train", "validation"}:
                    raise ResearchError("UNAUTHORIZED_DATA", "path qualification cannot consume test/final-eval inputs")
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            if plugin.plugin_id == "affine-mixture-production-chunk":
                # Dedicated producer does not make generic/fixture mixtures
                # eligible; validate its frozen formal registration pre-read.
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            if plugin.plugin_id == "affine-path-production-chunk":
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            if (plugin.plugin_id in {"affine-propagation-chunk", "synthetic-propagation-chunk", "affine-mlmc-qualification-chunk"}
                    and cell.get("execution", {}).get("config", {}).get("method") == "mlmc-pilot"):
                # BEFORE qualification artifacts or input exposure, including
                # a caller that bypasses the propagation command builder.
                if mode != "pilot" or block["split_role"] not in {"train", "validation"}:
                    raise ResearchError("UNAUTHORIZED_DATA", "MLMC pilot cannot consume test/final-eval inputs")
                from .propagation_execution import validate_propagation_cell
                validate_propagation_cell(spec, cell)
            grant = self.store.authorization(settings["authorization_id"], version=settings.get("authorization_version"))
            if (grant["study_id"] != spec["study_id"] or grant.get("protocol_hash") != spec["protocol_hash"]
                    or "execute" not in grant["purposes"] or cell["block_id"] not in grant["block_ids"]
                    or cell.get("visibility", "restricted") not in grant["visibilities"]
                    or datetime.fromisoformat(grant["expires_at"]) <= datetime.now(timezone.utc)):
                raise ResearchError("UNAUTHORIZED_DATA", "execution grant scope/expiry differs")
            package = self._document("package", settings.get("package_hash"))
            model = self._document("package", package["model_hash"]) if package.get("requires_frozen_model") else None
            if package.get("visibility", "restricted") not in grant["visibilities"]:
                raise ResearchError("UNAUTHORIZED_DATA", "execution grant cannot disclose this package")
            validate_package(package, kind=package["kind"], formal=mode == "formal", frozen_model=model)
            if (package["kind"] not in {"FrozenDynamicsPackage", "PropagationResult", "SwitchingResult"}
                    or package.get("qualification") not in {"fixture", "qualified"}
                    or any(package[key] != spec[key] for key in ("code_hash", "data_hash", "protocol_hash"))
                    or package.get("study_id") != spec["study_id"]
                    or package["input_hash"] != spec["data_hash"]
                    or not isinstance(package.get("payload"), dict)
                    or package["output_hash"] != digest(package["payload"])
                    or package.get("plugin_hash") != receipt["plugin_hash"]
                    or package.get("recovery_command_hash" if recovery_builder is not None else "command_hash") != receipt["command_hash"]
                    or cell["capability"] not in package["capabilities"]
                    or package["state_order"] != list(plugin.state_order) or package["units"] != list(plugin.units)
                    or package["resume_level"] != plugin.resume_level):
                raise ResearchError("CONTRACT_MISMATCH", "package differs from selected execution plugin/input")
            from .research_upstream import prepare_upstream
            needs_prereg = mode == "formal" or block["split_role"] in {"test", "final-eval"}
            upstream_prereg = (self._document("preregistration", protocol.get("preregistration_hash"))
                               if needs_prereg else None)
            snapshot_evidence = prepare_upstream(self.store, spec, cell, package, upstream_prereg,
                                                attempt_id=attempt_id, run_id=run["run_id"])
            documents = {"protocol": protocol, "authorization": grant, "package": package,
                         "upstream_snapshot": snapshot_evidence}
            # Preserve explicitly requested legacy public recipe bindings as
            # additional evidence. They never replace the mandatory snapshot.
            if "upstream_hash" in settings or "upstream_hash" in package:
                required = settings.get("upstream_ids")
                if not isinstance(required, list):
                    raise ResearchError("MISSING_INPUT", "explicit legacy upstream dependency list required")
                upstream = audit_inputs(ROOT, tuple(required))
                if (settings.get("upstream_hash") != upstream["manifest_hash"]
                        or package.get("upstream_hash") != upstream["manifest_hash"]):
                    raise ResearchError("CONTRACT_MISMATCH", "frozen upstream binding changed")
                documents["upstream"] = upstream
            if model is not None:
                documents.update(self._model_documents(model, settings, spec, grant))
            gate_evidence = {}
            if mode == "formal" or block["split_role"] in {"test", "final-eval"}:
                with self.store.lock():
                    gate_evidence = PreregistrationGate(self.store)._validate(protocol, block)
                prereg = self._document("preregistration", protocol.get("preregistration_hash"))
                history = self._document("exposure-history", protocol.get("history_hash"))
                documents.update(preregistration=prereg, history=history)
                documents["preregistration_event"] = next(event for event in self.store.events()
                    if event["event_kind"] == "MANIFEST" and
                    event["payload"]["object_id"] == "preregistration-" + protocol["preregistration_hash"])
            from .probability_calibration_consumption import prepare_calibrated_consumer
            if calibration is not None:
                documents["probability_calibration"] = prepare_calibrated_consumer(
                    self.store, spec, cell, preregistration=documents.get("preregistration"))
            if mode == "formal":
                if (block["split_role"] not in {"test", "final-eval"} or gate_evidence["test_mode"] != "blind"
                        or package.get("preregistration_hash") != protocol["preregistration_hash"]
                        or spec.get("comparison_plan", {}).get("preregistration_hash") != protocol["preregistration_hash"]):
                    raise ResearchError("UNQUALIFIED", "formal execution requires the bound blind preregistration")
                plan = spec.get("comparison_plan", {})
                if "adjudication_spec" in plan or "adjudication_hash" in plan:
                    policy = plan.get("adjudication_spec")
                    if (not isinstance(policy, dict) or policy.get("schema_version") != "pirc25-adjudication-spec-v1"
                            or plan.get("adjudication_hash") != digest(policy)
                            or prereg.get("adjudication_spec") != policy):
                        raise ResearchError("UNQUALIFIED", "adjudication policy differs from pre-read frozen preregistration")
                # Missing method proof must fail before even the generic
                # qualification attachment is read, not just before target data.
                if plugin.plugin_id == "affine-cubature":
                    from .cubature_qualification_admission import prepare_managed_cubature
                    documents["propagation_qualification"] = prepare_managed_cubature(
                        self.store, spec, cell, package, prereg)
                report, evidence = self._qualification(package, prereg, grant)
                documents.update(qualification=report, qualification_evidence=evidence)
                if plugin.plugin_id in {"affine-propagation", "affine-propagation-chunk",
                        "synthetic-propagation", "synthetic-propagation-chunk"}:
                    from .propagation_qualification_admission import prepare_managed_qualification
                    documents["propagation_qualification"] = prepare_managed_qualification(
                        self.store, spec, cell, package, prereg)
                elif plugin.plugin_id == "affine-mlmc-production-chunk":
                    from .mlmc_qualification_admission import prepare_managed_mlmc
                    documents["propagation_qualification"] = prepare_managed_mlmc(
                        self.store, spec, cell, package, prereg)
                elif plugin.plugin_id == "affine-mixture-production-chunk":
                    from .mixture_qualification_admission import prepare_managed_mixture
                    documents["propagation_qualification"] = prepare_managed_mixture(
                        self.store, spec, cell, package, prereg)
                elif plugin.plugin_id == "affine-path-production-chunk":
                    from .path_qualification_admission import prepare_managed_paths
                    documents["propagation_qualification"] = prepare_managed_paths(
                        self.store, spec, cell, package, prereg)
            purpose = settings.get("purpose")
            root = grant.get("data_root")
            if not isinstance(root, str) or not Path(root).is_absolute():
                raise ResearchError("UNAUTHORIZED_DATA", "execution needs an explicit authorized data root")
            # The same read gate streams real content before invoking a plugin;
            # retain its exposure receipt, not an unused whole input allocation.
            EvaluationExposureLedger(self.store).verify(protocol["protocol_id"], cell["block_id"], purpose=purpose,
                authorization_id=grant["authorization_id"], authorization_version=grant.get("version"), data_root=Path(root),
                consumer={"attempt_id": attempt_id, "run_id": attempt["run_id"], "entrypoint": "shared-admission"})
            reads = [event for event in self.store.events() if event["event_kind"] == "READ_COMPLETED"
                     and event["payload"].get("attempt_id") == attempt_id]
            if not reads:
                raise ResearchError("UNAUTHORIZED_DATA", "input admission has no completed exposure receipt")
            receipt.update(mode=mode, qualification="qualified" if mode == "formal" else "fixture",
                           input_kind="registered-protocol", documents=documents,
                           input_evidence=reads, gate_evidence=gate_evidence)
        receipt["admitted_at"] = utc_now()
        if not builtin_fixture:
            for permission in (documents["authorization"], documents.get("model_authorization", documents["authorization"])):
                if datetime.fromisoformat(permission["expires_at"]) <= datetime.fromisoformat(receipt["admitted_at"]):
                    raise ResearchError("UNAUTHORIZED_DATA", "execution permission expired during input admission")
            if "propagation_qualification" in documents:
                qualified = documents["propagation_qualification"]
                self.store.verify_artifact_read(qualified["source_artifact"]["artifact_id"],
                    purpose="evaluate", authorization=qualified["authorization"])
            if "probability_calibration" in documents:
                calibrated = documents["probability_calibration"]
                self.store.verify_artifact_read(calibrated["source_artifact"]["artifact_id"],
                    purpose="evaluate", authorization=calibrated["authorization"])
        receipt["admission_hash"] = digest(receipt)
        self.store.publish("admission-" + receipt["admission_hash"], receipt)
        self.store.append("ADMISSION", {"attempt_id": attempt_id, "run_id": run["run_id"],
                                       "admission_hash": receipt["admission_hash"]})
        return receipt

    def result_validator(self, receipt, spec, cell, plugin):
        def validate(result):
            if "cell_packages" in (spec.get("admission") or {}) or receipt.get("input_kind") == "registered-protocol":
                verify_admission_selection(spec, cell, receipt)
            from .research_execution import execution_plan
            from .research_registry import validate_value
            plan = execution_plan(spec, cell, plugin)
            if receipt.get("resource_plan") != plan or receipt.get("registry_entry") != plugin.registry_entry.manifest():
                raise ResearchError("CONTRACT_MISMATCH", "result resource/registry admission changed")
            validate_value(plugin.registry_entry.output_schema, result)
            validate_result(result, spec=spec, cell=cell, plugin=plugin)
            if "composition" in plan and result.get("component_plan_hash") != plan["composition"]["component_plan_hash"]:
                raise ResearchError("CONTRACT_MISMATCH", "worker result internal component provenance differs")
            if (receipt["mode"] == "formal" and
                    set(result["metrics"]) != set(receipt["documents"]["preregistration"]["primary_metrics"])):
                raise ResearchError("UNQUALIFIED", "formal result primary metrics differ from frozen plan")
            policy = spec.get("comparison_plan", {}).get("adjudication_spec")
            metric = policy.get("primary_metric") if isinstance(policy, dict) else None
            if isinstance(metric, dict) and all(metric.get(k) is not None for k in ("name", "definition", "unit")):
                if (result.get("metric_definitions", {}).get(metric["name"]) != metric["definition"]
                        or result["metric_units"].get(metric["name"]) != metric["unit"]):
                    raise ResearchError("UNQUALIFIED", "result metric definition/unit differs from frozen adjudication policy")
            if result.get("qualification") != receipt["qualification"]:
                raise ResearchError("UNQUALIFIED", "worker cannot change admitted qualification")
            from .probability_calibration_consumption import validate_calibrated_result
            validate_calibrated_result(self.store, receipt, spec, cell, result)
            if "propagation_qualification" in receipt.get("documents", {}):
                if plugin.plugin_id == "affine-cubature":
                    from .cubature_qualification_admission import validate_formal_cubature_result
                    validate_formal_cubature_result(receipt, spec, cell, result)
                elif plugin.plugin_id == "affine-path-production-chunk":
                    from .path_qualification_admission import validate_formal_path_result
                    validate_formal_path_result(receipt, spec, cell, result)
                elif plugin.plugin_id == "affine-mixture-production-chunk":
                    from .mixture_qualification_admission import validate_formal_mixture_result
                    validate_formal_mixture_result(receipt, spec, cell, result)
                elif plugin.plugin_id == "affine-mlmc-production-chunk":
                    from .mlmc_qualification_admission import validate_formal_mlmc_result
                    validate_formal_mlmc_result(receipt, spec, cell, result)
                else:
                    from .propagation_qualification_admission import validate_formal_analytic_result
                    validate_formal_analytic_result(receipt, spec, cell, result)
            if result.get("admission_hash", receipt["admission_hash"]) != receipt["admission_hash"]:
                raise ResearchError("CONTRACT_MISMATCH", "worker substituted another admission")
            result["admission_hash"] = receipt["admission_hash"]
        return validate

    def run(self, attempt_id, spec, cell, plugin, command_builder, budget, *, builtin_fixture=False, recovery_builder=None, checkpoint_handler=None):
        if plugin.plugin_id == "affine-path-production-chunk":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.path_production_plugin import production_policies
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                _, policy = production_policies(spec, cell, package, request, config)
                if budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "path target exceeds the frozen independent production job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id == "affine-mixture-production-chunk":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.mixture_production_plugin import production_policies
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                _, policy = production_policies(spec, cell, package, request, config)
                if budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "mixture target exceeds the frozen qualification job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id == "affine-path-qualification":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.path_qualification_plugin import path_qualification_policy
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                policy = path_qualification_policy(spec, cell, package, request, config)
                if budget.category != "pilot" or budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "path qualification requires the frozen pilot job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id == "affine-mixture-qualification":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.mixture_qualification_plugin import mixture_qualification_policies
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                _, policy = mixture_qualification_policies(spec, cell, package, request, config)
                if budget.category != "pilot" or budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "mixture qualification requires the frozen pilot job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id in {"affine-mixture-chunk", "synthetic-mixture-chunk"}:
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.mixture_plugin import mixture_policy
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                policy = mixture_policy(spec, cell, package, request, config)
                if budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "mixture exceeds frozen job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id == "affine-mlmc-production-chunk":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.mlmc_production_plugin import production_policy
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                policy = production_policy(spec, cell, package, request, config)
                if budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "MLMC production exceeds frozen job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id == "affine-propagation-qualification":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.qualification_plugin import qualification_policy
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                policy = qualification_policy(spec, cell, package, request, config)
                if budget.category != "pilot" or budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "analytic qualification requires the frozen pilot job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id == "affine-halfspace-calibration":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.calibration_plugin import calibration_policy
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                policy = calibration_policy(spec, cell, package, request, config)
                if budget.category != "pilot" or budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "probability calibration requires the frozen pilot job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id == "affine-cubature-qualification":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.cubature_plugin import cubature_policy
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                policy = cubature_policy(spec, cell, package, request, config)
                if budget.category != "pilot" or budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "cubature qualification requires the frozen pilot job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if plugin.plugin_id == "affine-mlmc-qualification-chunk":
            from .propagation_execution import validate_propagation_cell
            from experiments.pirc27.mlmc_qualification_plugin import mlmc_reference_policy
            try:
                package, request, config, _ = validate_propagation_cell(spec, cell)
                policy = mlmc_reference_policy(spec, cell, package, request, config)
                if budget.category != "pilot" or budget.job_seconds > policy.maximum_job_seconds:
                    raise ResearchError("CONTRACT_MISMATCH", "MLMC reference work requires the frozen pilot job budget")
            except ResearchError as exc:
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=exc.code)
                raise
        if (plugin.plugin_id in {"affine-propagation-chunk", "synthetic-propagation-chunk", "affine-mlmc-qualification-chunk"}
                and cell.get("execution", {}).get("config", {}).get("method") == "mlmc-pilot"
                and budget.category != "pilot"):
            self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code="CONTRACT_MISMATCH")
            raise ResearchError("CONTRACT_MISMATCH", "MLMC pilot must use the existing pilot budget category")
        from .research_supervisor import ResearchSupervisor
        admitted = {}
        resource_plan = {}
        def command(output):
            try:
                mode = "fixture" if builtin_fixture else spec.get("admission", {}).get("mode")
                cap = {"fixture": 900, "pilot": 1800, "formal": 7200}.get(mode, 7200)
                if budget.job_seconds > cap:
                    raise ResearchError("CONTRACT_MISMATCH", "execution stage budget exceeds its hard cap")
                admitted.update(self.prepare(spec, cell, plugin, attempt_id, builtin_fixture=builtin_fixture,
                                             recovery_builder=recovery_builder))
                resource_plan.update(admitted["resource_plan"])
            except (KeyError, TypeError, ValueError) as exc:
                if isinstance(exc, ResearchError):
                    raise
                raise ResearchError("CONTRACT_MISMATCH", "incomplete execution admission evidence") from exc
            command = command_builder(output)
            if "propagation_qualification" in admitted.get("documents", {}):
                command = [*command, "--admission-hash", admitted["admission_hash"]]
            return command
        def validate(result):
            return self.result_validator(admitted, spec, cell, plugin)(result)
        def checkpoint(state, progress, deadline):
            return checkpoint_handler(state, progress, admitted, deadline)
        return ResearchSupervisor(self.store).run(attempt_id, command, budget, result_validator=validate,
            resource_plan=resource_plan, checkpoint_handler=checkpoint if checkpoint_handler is not None else None)
