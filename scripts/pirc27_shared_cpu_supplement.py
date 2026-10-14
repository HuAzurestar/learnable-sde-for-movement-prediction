"""Run the two unattempted PIRC-27 Gaussian cells with authorized CPU sharing.

This is a new immutable supplement to the original eight-cell pilot. It keeps
the original Gaussian arm and cumulative budget, does not retry failed cells,
does not reopen the exact arm, and never edits the foreign reservation that
occupies configured CPU slot zero.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from application.research_admission import command_binding, data_binding, plugin_binding
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_data import EvaluationExposureLedger
from application.research_execution import execution_binding
from application.research_preregistration import PreregistrationGate, protocol_binding
from application.research_queue import SHARED_CPU_POLICY
from application.research_upstream import register_acceptance_catalog, study_binding
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc25.snapshot import UpstreamSnapshot
from experiments.pirc27.oracles import oracle_suite
from experiments.pirc27.qualification_plugin import qualification_plugin
from infrastructure.research_store import ResearchError, digest, encode

import pirc27_first_batch as original


STUDY = "pirc27-affine-shared-cpu-supplement-20261013-v1"
AUTHORIZATION = "pirc27-shared-cpu-supplement-20261013-v1"
ORIGINAL_STUDY = original.STUDY
SELECTED_CASES = ("anisotropic-v1", "uncertain-initial-v1")
EXPECTED_CELLS = 2
DECISION = {
    "source": "actual user message in the PIRC-27 task conversation",
    "task_id": "974e4d01-8a50-40c2-9598-5133a545dc3a",
    "decision": "请无视 CPU 互斥锁, 此为我的授权. CPU 互斥锁毫无意义, 建议无视. 放开共享资源限制",
    "safety_constraint": "内容不允许写入冲突,只允许提交 PR 合并到 master",
    "execution_date_local": "2026-10-13",
    "resource_policy": SHARED_CPU_POLICY,
    "maximum_new_research_slot_seconds": original.TOTAL_APPROVED_SECONDS,
    "scope": "only the two never-attempted Gaussian cells; no retry, exact-arm reopening, final-eval or budget extension",
}


def require(ok, message):
    if not ok:
        raise ResearchError("CONTRACT_MISMATCH", message)


def settled_pirc27_ms(store):
    studies = {ORIGINAL_STUDY, STUDY}
    return sum(event["payload"]["charged_ms"] for event in store.events()
               if event["event_kind"] == "SETTLE"
               and event["payload"].get("study_id") in studies)


def supplement_cells(arms):
    plugin = qualification_plugin()
    selected = []
    for cell in original.frozen_cells(arms):
        dimensions = cell["comparison_dimensions"]
        if (cell["execution"]["config"]["method"] != "gaussian"
                or dimensions["model_configuration"] not in SELECTED_CASES):
            continue
        cell = json.loads(encode(cell))
        cell["resource_sharing"] = SHARED_CPU_POLICY
        cell["execution"] = execution_binding(
            plugin.registry_entry,
            cell["execution"]["config"],
            cell["execution"]["inputs"],
            matrix_cells=EXPECTED_CELLS,
        )
        selected.append(cell)
    require(len(selected) == EXPECTED_CELLS, "shared-CPU supplement selection changed")
    require(tuple(c["comparison_dimensions"]["model_configuration"] for c in selected) == SELECTED_CASES,
            "shared-CPU supplement order changed")
    return selected


def prepare(store):
    with store._read_transaction():
        source = store.manifest(original.ORIGINAL)
        arms = {arm["method_family_id"]: arm for arm in source["arms"]
                if arm["method_family_id"] in {"exact", "gaussian"}}
        require(set(arms) == {"exact", "gaussian"}, "original method arms missing")
        require(BudgetLedger(store).balance(arms["exact"]["arm_id"])["closed"],
                "supplement must not replace or reopen the original exact arm")
        require(not BudgetLedger(store).balance(arms["gaussian"]["arm_id"])["closed"],
                "original Gaussian arm is closed")
        settled_ms = settled_pirc27_ms(store)
        require(settled_ms + EXPECTED_CELLS * original.JOB_SECONDS * 1000
                <= original.TOTAL_APPROVED_SECONDS * 1000,
                "supplement worst case exceeds the approved aggregate 4H budget")
    cells = supplement_cells(arms)
    selected_oracles = [case for case in oracle_suite() if case.case_id in SELECTED_CASES]
    recipe = {"schema_version": "pirc27-shared-cpu-supplement-input-v1",
              "source_kind": "fixed-generator", "source_family": original.FAMILY,
              "models": [{"case_id": case.case_id, "model": case.package.manifest(),
                          "initial_mean": case.initial_mean, "initial_covariance": case.initial_covariance}
                         for case in selected_oracles],
              "independence": "same generator-family source as the original pilot; no new independent block"}
    input_artifact = store.artifact(encode(recipe), role="pilot-input", visibility="synthetic",
                                    block_ids=[original.FAMILY], study_id=STUDY)
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": STUDY + "-inputs",
                "study_id": STUDY, "blocks": [{"block_id": original.FAMILY,
                "dataset_id": original.FAMILY, "release_id": "affine-oracle-suite-v1",
                "source_block_id": original.FAMILY, "sha256": input_artifact["artifact_id"],
                "path": input_artifact["artifact_id"], "split_role": "validation", "fit_scope": False}]}
    evidence = {"entries": [], "reason": "fixed public oracle generator, no terrain dependency"}
    catalog = {"schema_version": "pirc25-upstream-acceptance-v1", "visibility": "synthetic",
               "source": "PIRC-27 shared-CPU supplement", "entries": [],
               "source_evidence": evidence, "source_evidence_hash": digest(evidence)}
    catalog_hash = register_acceptance_catalog(store, catalog, digest(catalog))
    cutover = {"study_id": STUDY, "pirc22_cutover": {"mode": "not_applicable"}}
    snapshot = UpstreamSnapshot({"schema_version": "pirc25-upstream-snapshot-v1",
        "visibility": "synthetic", "inputs": [], "studies": [cutover], "cells": [
            {"cell_id": digest(cell), "study_id": STUDY, "role": cell["study_role"], "upstream_ids": []}
            for cell in cells]})
    snapshot_hash = store.publish("upstream-snapshot-" + snapshot.snapshot_hash, snapshot.definition)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan = {"schema_version": "pirc25-preregistration-v1", "test_mode": "exploratory",
            "study_ids": [STUDY], "protocol_bindings": [protocol_binding(protocol)],
            "primary_metrics": ["functional_roundoff_upper", "absolute_time_bias_upper", "charged-slot-ms"],
            "selection_rule": "the two original never-attempted Gaussian cases only",
            "stopping_rule": "two frozen jobs; no retry, exact reopening or result-driven refinement",
            "comparisons": [{"reference": arms["exact"]["arm_id"],
                             "candidates": [arms["gaussian"]["arm_id"]]}],
            "upstream_bindings": {STUDY: study_binding(snapshot_hash, catalog_hash, cutover)},
            "research_scope": "supplemental pilot numerical case enclosures; no population inference",
            "orchestrator_source_sha256": source_sha,
            "qualification_policies": [cell["affine_qualification_policy"] for cell in cells],
            "user_execution_authority": DECISION}
    protocol["preregistration_hash"] = PreregistrationGate(store).register_preregistration(plan, digest(plan))
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    authority_hash = store.publish("pirc27-shared-cpu-authority-" + digest(DECISION), DECISION)
    grant = {"authorization_id": AUTHORIZATION, "study_id": STUDY,
             "expires_at": "2026-11-13T00:00:00+08:00", "evidence_hash": authority_hash,
             "protocol_hash": digest(protocol), "data_root": str(store.path / "artifacts"),
             "test_authorization": False, "purposes": ["execute", "validate", "evaluate", "preview", "export"],
             "visibilities": ["synthetic"], "block_ids": [original.FAMILY]}
    store.authorize(grant)
    plugin = qualification_plugin()
    payload = {"scope": "user-authorized shared-CPU supplement, not formal qualification",
               "source_artifact_id": input_artifact["artifact_id"], "plan_hash": digest(plan)}
    package = {"schema_version": "pirc25-package-v1", "kind": "PropagationResult",
        "study_id": STUDY, "visibility": "synthetic", "state_order": list(plugin.state_order),
        "units": list(plugin.units), "resume_level": plugin.resume_level,
        "capabilities": sorted(plugin.capabilities), "code_hash": code_hash(),
        "data_hash": data_binding(protocol), "input_hash": data_binding(protocol),
        "protocol_hash": digest(protocol), "output_hash": digest(payload), "payload": payload,
        "plugin_hash": plugin_binding(plugin), "command_hash": command_binding(plugin.command_builder),
        "recovery_command_hash": command_binding(plugin.command_builder),
        "upstream_snapshot_hash": snapshot_hash, "upstream_acceptance_hash": catalog_hash,
        "qualification": "fixture"}
    package_hash = store.publish("package-" + digest(package), package)
    gaussian = arms["gaussian"]
    spec = {"schema_version": "pirc25-contract-v1", "study_id": STUDY, "experiment_id": STUDY,
        "comparison_family": "pirc27-user-authorized-shared-cpu-supplement", "code_hash": code_hash(),
        "protocol_hash": digest(protocol), "data_hash": data_binding(protocol),
        "feature_hash": digest({"terrain": "none"}), "selection_hash": digest(plan),
        "arms": [gaussian], "cells": cells,
        "runtime_binding": {"root": str(store.path.parent), "store_id": store.store_id},
        "resource_sharing_policy": SHARED_CPU_POLICY,
        "user_authority_hash": authority_hash,
        "maximum_new_research_slot_seconds": original.TOTAL_APPROVED_SECONDS,
        "prospective_maximum_unit_slot_seconds": EXPECTED_CELLS * original.JOB_SECONDS,
        "orchestrator_source_sha256": source_sha, "scientific_independent_n": 0,
        "admission": {"mode": "pilot", "purpose": "validate", "protocol_id": protocol["protocol_id"],
            "authorization_id": AUTHORIZATION, "package_hash": package_hash,
            "upstream_snapshot_hash": snapshot_hash, "upstream_acceptance_hash": catalog_hash,
            "upstream_root": str(store.path / "artifacts")}}
    store.register(spec, digest(spec))
    return {"study_id": STUDY, "spec_hash": digest(spec), "cells": len(cells),
            "settled_pirc27_ms_before": settled_ms,
            "prospective_slot_seconds": EXPECTED_CELLS * original.JOB_SECONDS,
            "resource_sharing_policy": SHARED_CPU_POLICY, "state": "REGISTERED_NOT_STARTED"}


def inspect(store):
    with store._read_transaction():
        spec = store.manifest("study-" + STUDY)["spec"]
        runs = {event["payload"]["object_id"]: store.manifest(event["payload"]["object_id"])
                for event in store.events() if event["event_kind"] == "MANIFEST"
                and event["payload"]["object_id"].startswith("run-")}
        own = [attempt for attempt in store.attempts().values()
               if runs.get("run-" + attempt["run_id"], {}).get("study_id") == STUDY]
        tried = {runs["run-" + attempt["run_id"]]["cell_hash"] for attempt in own}
        pending = [cell for cell in spec["cells"] if digest(cell) not in tried]
        return {"study_id": STUDY, "attempts": own, "pending_cells": len(pending),
                "settled_pirc27_ms": settled_pirc27_ms(store),
                "gaussian_balance": BudgetLedger(store).balance(spec["arms"][0]["arm_id"])}


def run_next(store):
    spec = store.manifest("study-" + STUDY)["spec"]
    require(spec["code_hash"] == code_hash(), "source changed after supplement registration")
    require(spec["user_authority_hash"] == digest(DECISION), "shared-CPU authority differs")
    require(spec["resource_sharing_policy"] == SHARED_CPU_POLICY, "shared-CPU policy differs")
    require(spec["orchestrator_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "orchestrator changed after supplement registration")
    status = inspect(store)
    require(all(attempt["state"] in store.TERMINAL for attempt in status["attempts"]),
            "existing supplement attempt is still active")
    tried = {attempt["run_id"] for attempt in status["attempts"]}
    with store._read_transaction():
        tried_cells = {store.manifest("run-" + run_id)["cell_hash"] for run_id in tried}
    pending = [cell for cell in spec["cells"] if digest(cell) not in tried_cells]
    if not pending:
        return {**status, "state": "ALL_REGISTERED_CELLS_ATTEMPTED"}
    remaining = len(pending)
    require(status["settled_pirc27_ms"] + remaining * original.JOB_SECONDS * 1000
            <= original.TOTAL_APPROVED_SECONDS * 1000,
            "remaining supplement worst case exceeds the approved aggregate 4H budget")
    cell = pending[0]
    require(cell.get("resource_sharing") == SHARED_CPU_POLICY, "cell lacks frozen shared-CPU authority")
    require(not BudgetLedger(store).balance(cell["arm_id"])["closed"], "original Gaussian arm is closed")
    registry = CapabilityRegistry()
    registry.register(qualification_plugin())
    result = SharedRunner(store, registry).run_cell(
        STUDY, digest(cell), budget=BudgetSpec(original.JOB_SECONDS, category="pilot"))
    outcome = {"study_id": STUDY, "cell_hash": digest(cell),
               "model": cell["comparison_dimensions"]["model_configuration"],
               "method": "gaussian", "resource_sharing_policy": SHARED_CPU_POLICY, **result}
    if result.get("artifact_id"):
        grant = store.authorization(AUTHORIZATION)
        saved = json.loads(store.read_artifact(result["artifact_id"], purpose="evaluate", authorization=grant))
        analysis = saved["forecast"]["qualification_analysis"]
        outcome["numerical_analysis"] = {key: analysis[key] for key in
            ("status", "checks", "functional_roundoff_upper", "absolute_time_bias_upper",
             "reference_width_upper", "scientific_qualification")}
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "inspect", "run-next"))
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        store = original.connect(args.runtime_config)
        value = prepare(store) if args.command == "prepare" else inspect(store) if args.command == "inspect" else run_next(store)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if value.get("state") not in {"FAILED", "TIMEOUT", "INTERRUPTED", "PREFLIGHT_FAILED"} else 1
    except ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
