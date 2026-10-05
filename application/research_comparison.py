"""Managed statistical comparison without changing the scientific matrix."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from application.research_budget import BudgetSpec
from application.research_computation import (MAX_INPUT_BYTES, MAX_OUTPUT_BYTES, MAX_OPERATIONS,
    comparison_plan, paper_identity, verified_computation)
from application.research_evidence import (accept_evidence_package, authorize_evidence_publication,
    authorize_study, evidence_visibility,
    export_evidence, expected_metrics_csv, expected_paper_index)
from application.research_supervisor import ResearchSupervisor
from experiments.pirc25.affine import ROOT, code_hash
from infrastructure.research_store import ResearchError, atomic_write, digest, encode, identifier


class ComparisonRunner:
    def __init__(self, store, paper_root):
        self.store = store
        self.paper_root = Path(paper_root).resolve()

    def _validate_result(self, value, request, bundle):
        try:
            aggregate = value["aggregate"]
            decision = aggregate["adjudication"]
            table = value["metrics_csv"].encode()
            from application.research_figures import validate_figure_package
            validate_figure_package(aggregate, value["figure_index"], value["figures"])
            if (value["schema_version"] != "pirc25-computation-result-v1" or
                    value["request_hash"] != digest(request) or value["computation_ref"] != request["computation_ref"] or
                    aggregate["computation_ref"] != request["computation_ref"] or
                    aggregate["source_bundle_hash"] != bundle["bundle_hash"] or
                    aggregate["aggregate_hash"] != digest({k: v for k, v in aggregate.items() if k != "aggregate_hash"}) or
                    aggregate["spec_hash"] != bundle["spec_hash"] or aggregate["protocol_hash"] != bundle["protocol_hash"] or
                    aggregate["study_id"] != bundle["study_id"] or aggregate["cell_dispositions"] != bundle["cells"] or
                    aggregate["expected_cell_count"] != len(bundle["expected_cells"]) or
                    aggregate["qualification"] != ("formal" if request["formal"] else "engineering-fixture") or
                    decision["compare_hash"] != digest({k: v for k, v in decision.items() if k != "compare_hash"}) or
                    decision["source_bundle_hash"] != bundle["bundle_hash"] or
                    decision["adjudication_spec"] != (bundle.get("comparison_plan") or {}).get("adjudication_spec") or
                    table != expected_metrics_csv(aggregate) or value["paper_index"] != expected_paper_index(aggregate, table, value["figure_index"]) or
                    len(encode(value)) > MAX_OUTPUT_BYTES or
                    code_hash() != request["runtime_code_hash"] or paper_identity(self.paper_root) != request["paper_identity"]):
                raise ResearchError("CONTRACT_MISMATCH", "managed comparison output binding changed")
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchError("CONTRACT_MISMATCH", "incomplete statistical worker output") from exc

    def _prepare(self, *args, **kwargs):
        # One actual owner phase, not a grant or cross-poll prefix cache. Every
        # original manifest/artifact and authorization check still runs. The
        # outer scope rehashes the complete final chain and disclosure grants
        # before its result can reach supervision. Never hold this lock across
        # a worker, settlement, staging fsync or per-member publication guard.
        with self.store._read_transaction():
            return self._prepare_locked(*args, **kwargs)

    def _prepare_locked(self, study_id, input_aggregate_hash, *, authorization_id,
            max_operations, formal, parent_attempt_id, reason, authorization_version):
        grant = self.store.authorization(authorization_id, version=authorization_version)
        authorize_study(self.store, study_id, grant, "export")
        source = self.store.manifest("comparison-" + input_aggregate_hash)
        if source["study_id"] != study_id:
            raise ResearchError("UNAUTHORIZED_DATA", "comparison input is outside authorized study")
        metadata = self.store.manifest("artifact-" + source["aggregate_id"])
        if metadata["size_bytes"] > MAX_INPUT_BYTES:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "frozen aggregate input exceeds byte quota")
        aggregate = json.loads(self.store.read_artifact(source["aggregate_id"], purpose="export", authorization=grant))
        if aggregate["aggregate_hash"] != input_aggregate_hash:
            raise ResearchError("CORRUPT_ARTIFACT", "input aggregate identity changed")
        bundle = self.store.manifest("bundle-" + aggregate["source_bundle_hash"])
        if len(encode(bundle)) > MAX_INPUT_BYTES:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "frozen source bundle exceeds byte quota")
        original = self.store.manifest("study-" + study_id)["spec"]
        if (bundle["spec_hash"] != digest(original) or bundle["cells"] != aggregate["cell_dispositions"] or
                digest({k: v for k, v in bundle.items() if k != "bundle_hash"}) != bundle["bundle_hash"]):
            raise ResearchError("CORRUPT_ARTIFACT", "frozen comparison source differs from authority")
        # Revalidate every actual source/qualification disclosure under current
        # grants. Never replace the chosen frozen snapshot with this new export.
        export_evidence(self.store, study_id, grant)
        plan = comparison_plan(bundle, max_operations)
        if formal is None:
            formal = aggregate["qualification"] == "formal"
        if type(formal) is not bool:
            raise ResearchError("CONTRACT_MISMATCH", "formal mode must be explicit boolean")
        paper = paper_identity(self.paper_root)
        runtime_hash = code_hash()
        identity = digest(["shared-compare-v1", digest(original), input_aggregate_hash, runtime_hash, paper, formal, plan])
        # The reference arm is frozen, not a caller-controlled new budget name.
        reference_arm = (original.get("comparison_plan") or {}).get("reference_arm_id", min(a["arm_id"] for a in original["arms"]))
        arm = next((a for a in original["arms"] if a["arm_id"] == reference_arm), None)
        if arm is None:
            raise ResearchError("CONTRACT_MISMATCH", "reference arm is not registered")
        visibility = evidence_visibility(self.store, original, bundle["cells"])
        derived = {"schema_version": original["schema_version"], "study_id": "compare-" + identity,
            "experiment_id": "shared-comparison-v1", "comparison_family": original["comparison_family"],
            "protocol_hash": original["protocol_hash"], "data_hash": bundle["bundle_hash"], "code_hash": runtime_hash,
            "feature_hash": original["feature_hash"], "selection_hash": original["selection_hash"], "arms": [arm],
            "cells": [{"arm_id": arm["arm_id"], "block_id": "computation-input", "seed": 0,
                "visibility": visibility, "resource_class": "cpu", "plugin_id": "shared-comparison-v1"}],
            "computation_origin": {"study_id": study_id, "spec_hash": digest(original),
                "input_aggregate_hash": input_aggregate_hash, "source_bundle_hash": bundle["bundle_hash"],
                "paper_identity": paper, "resource_plan": plan, "formal": formal}}
        self.store.register(derived, digest(derived))
        run_id = self.store.register_run(derived["study_id"], derived["cells"][0])
        related = [a for a in self.store.attempts().values() if a["run_id"] == run_id]
        success = next((a for a in related if a["state"] == "SUCCEEDED"), None)
        if success:
            return grant, bundle, arm, success, None, None, None
        attempt_id = self.store.new_attempt(run_id, parent_attempt_id=parent_attempt_id, reason=reason)
        reference = {"manifest_id": "computation-" + attempt_id, "attempt_id": attempt_id, "run_id": run_id,
            "computation_spec_hash": digest(derived), "request_manifest_id": "computation-request-" + attempt_id,
            "reservation_id": digest([self.store.store_id, attempt_id]), "input_aggregate_hash": input_aggregate_hash,
            "source_bundle_hash": bundle["bundle_hash"], "allocation": plan["allocation"]}
        request = {"schema_version": "pirc25-computation-request-v1", "computation_ref": reference,
            "computation_study_id": derived["study_id"], "runtime_code_hash": runtime_hash,
            "paper_identity": paper, "resource_plan": plan, "formal": formal}
        return grant, bundle, arm, None, attempt_id, reference, request

    def run(self, study_id, input_aggregate_hash, *, authorization_id, budget=BudgetSpec(),
            max_operations=MAX_OPERATIONS, formal=None, parent_attempt_id=None, reason=None, output=None,
            authorization_version=None):
        study_id, input_aggregate_hash = identifier(study_id), identifier(input_aggregate_hash)
        budget.validate()
        grant, bundle, arm, success, attempt_id, reference, request = self._prepare(
            study_id, input_aggregate_hash, authorization_id=authorization_id,
            max_operations=max_operations, formal=formal, parent_attempt_id=parent_attempt_id,
            reason=reason, authorization_version=authorization_version)
        if success:
            result = {"attempt_id": success["attempt_id"], "state": "SUCCEEDED",
                      "artifact_id": success["artifact_id"], "exit_code": 0, "reused": True}
        else:
            def command(result_path):
                self.store.publish(reference["request_manifest_id"], request)
                atomic_write(result_path.parent / "request.json", encode(request))
                atomic_write(result_path.parent / "bundle.json", encode(bundle))
                return [sys.executable, "-B", str(ROOT / "experiments/pirc25/compare_worker.py"),
                    str(result_path.parent / "request.json"), str(result_path.parent / "bundle.json"),
                    str(self.paper_root), str(result_path)]

            result = ResearchSupervisor(self.store).run(attempt_id, command, budget,
                result_validator=lambda value: self._validate_result(value, request, bundle))
            result["reused"] = False
            if result["state"] != "SUCCEEDED":
                return result
        # Publishing is retryable without another job or budget charge. A crash
        # after settlement can complete this exact receipt on the next call.
        attempt = self.store.attempts()[result["attempt_id"]]
        with self.store.lock():
            metadata = self.store._manifest("artifact-" + attempt["artifact_id"])
            value = json.loads(self.store._verified_artifact_content(metadata))
        reference = value["computation_ref"]
        request = self.store.manifest(reference["request_manifest_id"])
        self._validate_result(value, request, bundle)
        settles = [e for e in self.store.events() if e["event_kind"] == "SETTLE" and
                   e["payload"].get("reservation_id") == reference["reservation_id"]]
        if len(settles) != 1:
            raise ResearchError("CORRUPT_ARTIFACT", "statistical job settlement is missing")
        settlement = settles[0]
        receipt = {"schema_version": "pirc25-computation-receipt-v1", "computation_ref": reference,
            "attempt_id": attempt["attempt_id"], "request_hash": digest(request),
            "result_artifact_id": attempt["artifact_id"], "aggregate_hash": value["aggregate"]["aggregate_hash"],
            "settlement_event_hash": settlement["hash"],
            "cost": {"arm_id": arm["arm_id"], "charged_ms": settlement["payload"]["charged_ms"],
                     "unit": "slot-ms", "scope": "whole-shared-computation-job", "basis": "measured-monotonic"}}
        self.store.publish(reference["manifest_id"], receipt)
        verified_computation(self.store, reference, receipt=receipt, aggregate=value["aggregate"])
        authorize_study(self.store, study_id, grant, "export")  # Expiry during the job must still deny disclosure.
        files = {"aggregate.json": encode(value["aggregate"]), "metrics.csv": value["metrics_csv"].encode(),
                 "PaperEvidenceIndex.json": encode(value["paper_index"]), "ComputationReceipt.json": encode(receipt),
                 "FigureIndex.json": encode(value["figure_index"]),
                 **{name: content.encode("utf-8") for name, content in value["figures"].items()}}
        files["manifest.json"] = encode({"schema_version": "pirc25-evidence-package-v1",
            "aggregate_hash": value["aggregate"]["aggregate_hash"],
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}})
        directory = self.store.path / "artifacts" / (".comparison-" + attempt["attempt_id"])
        def publication_guard():
            # Reauthenticate the chosen immutable source, including foreign
            # model grants. Earlier read/job permission cannot cover later
            # staging fsync, identical retries or final package disclosure.
            authorize_evidence_publication(self.store, bundle, grant)

        self._write_files(directory, files, before_replace=publication_guard)
        package = accept_evidence_package(self.store, directory, value["aggregate"]["aggregate_hash"])
        if output is not None:
            self._write_files(Path(output), files, before_replace=publication_guard)
        publication_guard()
        return {**result, "comparison": package}

    @staticmethod
    def _write_files(directory, files, *, before_replace=None):
        from infrastructure.research_publication import prepare_directory, opened_directory
        directory = Path(os.path.abspath(directory))
        if any((parent / ".git").exists() for parent in (directory, *directory.parents)):
            raise ResearchError("UNAUTHORIZED_DATA", "generated comparison must stay outside Git")
        directory = prepare_directory(directory)
        with opened_directory(directory) as binding:
            for name, content in files.items():
                binding[2]()
                path = directory / name
                if path.is_symlink() or not path.resolve().is_relative_to(directory):
                    raise ResearchError("UNAUTHORIZED_DATA", "comparison package path escapes root")
                # One original parent stays bound across ALL members; a new
                # real directory at the same lexical name is not our root.
                atomic_write(path, content, immutable=True, before_replace=before_replace, _directory=binding)
                binding[2]()
