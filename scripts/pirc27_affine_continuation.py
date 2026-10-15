"""Explicit same-budget source continuation; one supervised attempt per call."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pirc27_first_batch as original
import pirc27_shared_cpu_supplement as supplement
from application.pirc27_source_transition import numerical_identity
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
from experiments.pirc27.qualification_plugin import qualification_plugin
from infrastructure.research_store import ResearchError, digest, encode

STUDY = "pirc27-affine-infrastructure-continuation-20261015-v1"
AUTHORIZATION = STUDY + "-grant"
APPROVAL = STUDY + "-approval"
STUDIES = {original.STUDY, supplement.STUDY, STUDY}
require = original.require
DECISION = {
    "source": "actual user reply to same-budget closed-arm and infrastructure-retry request",
    "task_id": "974e4d01-8a50-40c2-9598-5133a545dc3a",
    "quoted_reply": "请继续完成 PIRC-27，授权回复，GOAL 模式。",
    "approved_request": "restore original exact/Gaussian arms; explicit infrastructure retries; preserve all historical costs within original 4H",
    "scope": "original affine synthetic pilot only; no final-eval, training or real-data grant",
}


def accounting(store):
    reservations = {}
    for e in store.events():
        if (e["event_kind"] in {"RESERVE", "SETTLE"}
                and e["payload"].get("study_id", "").startswith("pirc27-")):
            reservations[e["payload"]["reservation_id"]] = e["payload"]
    return {"charged_ms": sum(r["charged_ms"] for r in reservations.values() if r["settled"]),
            "reserved_ms": sum(r["reserved_ms"] for r in reservations.values() if not r["settled"])}


def sources(store):
    """Retain the original eight origins and latest actual infrastructure parent."""
    old = store.manifest("study-" + original.STUDY)["spec"]
    extra = store.manifest("study-" + supplement.STUDY)["spec"]
    attempts = store.attempts()
    runs = {a["run_id"]: store.manifest("run-" + a["run_id"]) for a in attempts.values()}
    rows = []
    for cell in old["cells"]:
        identity = numerical_identity(cell)
        history = [a for a in attempts.values() if runs[a["run_id"]]["study_id"] in STUDIES
                   and numerical_identity(runs[a["run_id"]]["cell"]) == identity]
        origin = next((c for c in extra["cells"] if numerical_identity(c) == identity), cell)
        origin_study = supplement.STUDY if origin in extra["cells"] else original.STUDY
        parent = history[-1] if history else None
        rows.append({"cell": cell, "source_cell": origin, "source_study": origin_study,
                     "parent": parent, "history": history,
                     "succeeded": any(a["state"] == "SUCCEEDED" for a in history)})
    return old, rows


def prepare(store):
    # Metadata only: a single verified writer phase, never enclosing dispatch.
    with store._read_transaction():
        old, rows = sources(store)
        require(all(a["state"] in store.TERMINAL for r in rows for a in r["history"]),
                "old pilot still active")
        costs = accounting(store)
        require(costs["reserved_ms"] == 0, "old pilot reservation unsettled")
        arms = {a["method_family_id"]: a for a in old["arms"]}
        plugin = qualification_plugin()
        selected = []
        lineage = []
        for cell in original.frozen_cells(arms):
            row = next(r for r in rows if numerical_identity(r["cell"]) == numerical_identity(cell))
            if row["succeeded"]:
                continue  # Preserve old success, never pretend it is new-source evidence.
            cell["resource_sharing"] = SHARED_CPU_POLICY
            selected.append(cell)
            source_run = store.register_run(row["source_study"], row["source_cell"])
            if row["parent"]:
                source_run = row["parent"]["run_id"]
            lineage.append({"target_cell_hash": digest(cell), "source_run_id": source_run,
                            "source_parent_attempt_id": row["parent"]["attempt_id"] if row["parent"] else None})
        require(len(selected) == 7, "expected seven unfinished original cases; do not redefine frozen continuation")
        cells = []
        for c in selected:
            c["execution"] = execution_binding(plugin.registry_entry, c["execution"]["config"],
                                                 c["execution"]["inputs"], matrix_cells=len(selected))
            cells.append(c)
        # Cell IDs above must bind the final matrix resource plan.
        for link, cell in zip(lineage, cells):
            link["target_cell_hash"] = digest(cell)
        require(costs["charged_ms"] + len(cells) * original.JOB_SECONDS * 1000 <= 14400000,
                "aggregate 4H cap exceeded")
        now = datetime.now(timezone.utc)
        try:
            decision = store.manifest(APPROVAL)
        except ResearchError as exc:
            if exc.code != "MISSING_INPUT":
                raise
            decision = {"schema_version": "pirc25-continuation-approval-v1", "decision": "approved",
                "arm_seconds": 86400, "budget_mode": "cumulative-only-hard",
                "arm_ids": [a["arm_id"] for a in old["arms"]], "test_authorization": False,
                "public_export": False, "pirc27_total_slot_seconds": 14400,
                "infrastructure_retries": True, "human_decision": DECISION,
                "work_started_at": now.isoformat(), "work_deadline": (now + timedelta(hours=4)).isoformat(),
                "authorization_expires_at": (now + timedelta(hours=4)).isoformat()}
            store.publish(APPROVAL, decision)
        approval_ref = {"manifest_id": APPROVAL, "sha256": digest(decision)}
        for arm in old["arms"]:
            BudgetLedger(store).continue_arm(arm["arm_id"], approval_ref=approval_ref)
        recipe = {"schema_version": "pirc27-affine-continuation-input-v1",
                  "models": [c["frozen_dynamics"] for c in cells],
                  "source_family": original.FAMILY, "independent_n": 0}
        artifact = store.artifact(encode(recipe), role="pilot-input", visibility="synthetic",
                                  block_ids=[original.FAMILY], study_id=STUDY)
        protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": STUDY + "-inputs",
            "study_id": STUDY, "blocks": [{"block_id": original.FAMILY, "dataset_id": original.FAMILY,
            "release_id": "affine-oracle-suite-v1", "source_block_id": original.FAMILY,
            "sha256": artifact["artifact_id"], "path": artifact["artifact_id"],
            "split_role": "validation", "fit_scope": False}]}
        evidence = {"entries": [], "reason": "fixed public oracle, no terrain dependency"}
        catalog = {"schema_version": "pirc25-upstream-acceptance-v1", "visibility": "synthetic",
            "source": "PIRC-27 source continuation", "entries": [], "source_evidence": evidence,
            "source_evidence_hash": digest(evidence)}
        catalog_hash = register_acceptance_catalog(store, catalog, digest(catalog))
        cutover = {"study_id": STUDY, "pirc22_cutover": {"mode": "not_applicable"}}
        snapshot = UpstreamSnapshot({"schema_version": "pirc25-upstream-snapshot-v1", "visibility": "synthetic",
            "inputs": [], "studies": [cutover], "cells": [{"cell_id": digest(c), "study_id": STUDY,
            "role": c["study_role"], "upstream_ids": []} for c in cells]})
        snapshot_hash = store.publish("upstream-snapshot-" + snapshot.snapshot_hash, snapshot.definition)
        source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        plan = {"schema_version": "pirc25-preregistration-v1", "test_mode": "exploratory", "study_ids": [STUDY],
            "protocol_bindings": [protocol_binding(protocol)],
            "primary_metrics": ["functional_roundoff_upper", "absolute_time_bias_upper", "charged-slot-ms"],
            "selection_rule": "all seven unfinished original affine cells; old successful cell retained separately",
            "stopping_rule": "one explicit attempt per cell; any failure stops operator batch; no automatic retry",
            "comparisons": [{"reference": arms["exact"]["arm_id"], "candidates": [arms["gaussian"]["arm_id"]]}],
            "upstream_bindings": {STUDY: study_binding(snapshot_hash, catalog_hash, cutover)},
            "source_lineage": lineage, "continuation_approval": approval_ref,
            "orchestrator_source_sha256": source_sha, "user_execution_authority": DECISION}
        protocol["preregistration_hash"] = PreregistrationGate(store).register_preregistration(plan, digest(plan))
        EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
        store.authorize({"authorization_id": AUTHORIZATION, "study_id": STUDY,
            "expires_at": decision["authorization_expires_at"], "evidence_hash": digest(decision),
            "protocol_hash": digest(protocol), "data_root": str(store.path / "artifacts"),
            "test_authorization": False, "purposes": ["execute", "validate", "evaluate", "preview", "export"],
            "visibilities": ["synthetic"], "block_ids": [original.FAMILY]})
        payload = {"scope": "source-bound infrastructure continuation, not formal qualification", "plan_hash": digest(plan)}
        package = {"schema_version": "pirc25-package-v1", "kind": "PropagationResult", "study_id": STUDY,
            "visibility": "synthetic", "state_order": list(plugin.state_order), "units": list(plugin.units),
            "resume_level": plugin.resume_level, "capabilities": sorted(plugin.capabilities), "code_hash": code_hash(),
            "data_hash": data_binding(protocol), "input_hash": data_binding(protocol), "protocol_hash": digest(protocol),
            "output_hash": digest(payload), "payload": payload, "plugin_hash": plugin_binding(plugin),
            "command_hash": command_binding(plugin.command_builder), "recovery_command_hash": command_binding(plugin.command_builder),
            "upstream_snapshot_hash": snapshot_hash, "upstream_acceptance_hash": catalog_hash, "qualification": "fixture"}
        package_hash = store.publish("package-" + digest(package), package)
        spec = {"schema_version": "pirc25-contract-v1", "study_id": STUDY, "experiment_id": STUDY,
            "comparison_family": "pirc27-original-affine-source-continuation", "code_hash": code_hash(),
            "protocol_hash": digest(protocol), "data_hash": data_binding(protocol), "feature_hash": old["feature_hash"],
            "selection_hash": digest(plan), "arms": old["arms"], "cells": cells, "source_lineage": lineage,
            "runtime_binding": {"root": str(store.path.parent), "store_id": store.store_id},
            "continuation_approval": approval_ref, "orchestrator_source_sha256": source_sha, "scientific_independent_n": 0,
            "admission": {"mode": "pilot", "purpose": "validate", "protocol_id": protocol["protocol_id"],
            "authorization_id": AUTHORIZATION, "package_hash": package_hash, "upstream_snapshot_hash": snapshot_hash,
            "upstream_acceptance_hash": catalog_hash, "upstream_root": str(store.path / "artifacts")}}
        store.register(spec, digest(spec))
        return {"study_id": STUDY, "spec_hash": digest(spec), "cells": len(cells), **costs,
                "state": "REGISTERED_NOT_STARTED"}


def run_next(store):
    with store._read_transaction():
        spec = store.manifest("study-" + STUDY)["spec"]
        require(spec["code_hash"] == code_hash(), "source changed after continuation freeze")
        require(spec["orchestrator_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "orchestrator changed after freeze")
        attempts = store.attempts()
        runs = {a["run_id"]: store.manifest("run-" + a["run_id"]) for a in attempts.values()}
        own = [a for a in attempts.values() if runs[a["run_id"]]["study_id"] == STUDY]
        require(all(a["state"] == "SUCCEEDED" for a in own), "previous continuation failure/live attempt; stop batch")
        done = {runs[a["run_id"]]["cell_hash"] for a in own}
        pending = [c for c in spec["cells"] if digest(c) not in done]
        costs = accounting(store)
        require(costs["reserved_ms"] == 0, "PIRC-27 reservation still active")
        require(costs["charged_ms"] + len(pending) * original.JOB_SECONDS * 1000 <= 14400000,
                "aggregate 4H cap exceeded")
        if not pending:
            return {"state": "ALL_CONTINUATION_CELLS_SUCCEEDED", **costs}
        cell = pending[0]
        link = next(l for l in spec["source_lineage"] if l["target_cell_hash"] == digest(cell))
        reference = None
        if link["source_parent_attempt_id"]:
            run_id = store.register_run(STUDY, cell)
            source = store.manifest("run-" + link["source_run_id"])
            old_spec = store.manifest("study-" + source["study_id"])["spec"]
            transition = {"schema_version": "pirc27-source-transition-v1", **link,
                "target_run_id": run_id, "source_code_hash": old_spec["code_hash"],
                "target_code_hash": spec["code_hash"], "approval": spec["continuation_approval"]}
            reference = {"manifest_id": "pirc27-source-transition-" + digest(transition), "sha256": digest(transition)}
            store.publish(reference["manifest_id"], transition)
    registry = CapabilityRegistry()
    registry.register(qualification_plugin())
    result = SharedRunner(store, registry).run_cell(STUDY, digest(cell),
        budget=BudgetSpec(original.JOB_SECONDS, category="pilot"),
        parent_attempt_id=link["source_parent_attempt_id"],
        reason="explicit authorized restart after infrastructure failure, fixed worker writer wait" if reference else None,
        source_transition_ref=reference)
    return {"study_id": STUDY, "cell_hash": digest(cell),
            "model": cell["comparison_dimensions"]["model_configuration"],
            "method": cell["execution"]["config"]["method"], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run-next", "inspect"])
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    try:
        store = original.connect(args.runtime_config)
        if args.command == "inspect":
            with store._read_transaction():
                old, rows = sources(store)
                result = {**accounting(store), "origins": [{"case": r["cell"]["comparison_dimensions"]["model_configuration"],
                    "method": r["cell"]["execution"]["config"]["method"], "parent": r["parent"],
                    "succeeded": r["succeeded"]} for r in rows],
                    "balances": [BudgetLedger(store).balance(a["arm_id"]) for a in old["arms"]]}
        else:
            result = prepare(store) if args.command == "prepare" else run_next(store)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if result.get("state") not in {"FAILED", "TIMEOUT", "INTERRUPTED", "PREFLIGHT_FAILED"} else 1
    except ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
