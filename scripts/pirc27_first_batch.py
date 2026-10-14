"""User-authorized first research pilot, never an implicit full-study grant.

Only existing shared identities/APIs are used. No test helpers, private data,
new store, budget reset, retry or fabricated qualification is permitted.
The first frozen unit is eight analytic qualification jobs (4 cases x 2 methods).
Its worst-case 2400 slot-seconds is inside the approved aggregate 14400 cap.
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
from application.research_upstream import register_acceptance_catalog, study_binding
from domain.affine_qualification import AffineQualificationPolicy
from domain.propagation import PropagationRequest
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc25.snapshot import UpstreamSnapshot
from experiments.pirc27.oracles import oracle_suite
from experiments.pirc27.plugin import execution_inputs
from experiments.pirc27.qualification_plugin import qualification_config, qualification_plugin
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode


STUDY = "pirc27-first-affine-qualification-20261013-v1"
FAMILY = "pirc27-affine-oracle-v1"
ORIGINAL = "pirc27-preparation-9ab62b8389fd6733aca449fe423b5cbbe66611cbe78662fbffcba4700312af9a"
AUTHORIZATION = "pirc27-first-pilot-20261013-v1"
JOB_SECONDS = 300
TOTAL_APPROVED_SECONDS = 14400
EXPECTED_CELLS = 8
CONFIG_HASH = "878920e0bb29a674660f742300f354893b1ba1dc9d66f235932eea563a9df854"
DECISION = {
    "source": "actual user messages in this PIRC-27 task conversation",
    "task_id": "974e4d01-8a50-40c2-9598-5133a545dc3a",
    "budget_decision": "同意时间上限，不过我认为可以增加到 4H，目前不急，可以先跑数据没事儿。",
    "execution_decision": "同意开始实验，请执行，GOAL 模式完成。",
    "execution_date_local": "2026-10-13",
    "maximum_new_research_slot_seconds": TOTAL_APPROVED_SECONDS,
    "scope": "independent affine/synthetic qualification and pilot; not final-eval or private export",
}


def require(ok, message):
    if not ok:
        raise ResearchError("CONTRACT_MISMATCH", message)


def connect(config_path):
    require(config_path.is_file() and not config_path.is_symlink(), "regular existing config required")
    raw = config_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_HASH, "shared connection file changed")
    config = json.loads(raw)
    require(config["store_id"] == "pirc25-shared-research-v1"
            and "PIRC-27" in config["participants"], "original store/participant required")
    # Explicitly no initialize=True, no disposable-store fallback.
    return ResearchStore(Path(config["runtime_root"]), config["store_id"])


def frozen_cells(arms):
    plugin = qualification_plugin()
    result = []
    for case in oracle_suite():
        for method in ("exact", "gaussian"):
            request = PropagationRequest(
                request_id="first-qual-" + case.case_id + "-" + method,
                model_package_hash=case.package.package_hash,
                initial_mean=case.initial_mean, initial_covariance=case.initial_covariance,
                origin=0., history_cutoff=0., horizons=(1.,), functional="endpoint-x",
                seed=1147, coupling_id="first-affine-" + case.case_id,
                arm_id=arms[method]["arm_id"], steps=32, samples=2, chunk_size=2,
                tolerance=.05)
            policy = AffineQualificationPolicy(
                request.request_hash, case.package.package_hash, code_hash(), method,
                maximum_reference_width=1e-12, maximum_functional_roundoff=1e-8,
                maximum_time_bias=.05, maximum_scaled_transition_norm=1e6,
                state_scales=(1., 1., 1., 1.), maximum_operations=400001,
                maximum_job_seconds=JOB_SECONDS)
            policy.validate(case.package, request, method, code_hash())
            config = qualification_config(request, method, policy)
            result.append({"arm_id": request.arm_id, "block_id": FAMILY, "seed": request.seed,
                "horizon": 1., "plugin_id": plugin.plugin_id,
                "capability": "exact-transition" if method == "exact" else "generic-rollout",
                "resource_class": "cpu", "visibility": "synthetic", "dimensions": 4,
                "study_role": "secondary", "execution_role": "qualification",
                "comparison_dimensions": {"model_configuration": case.case_id,
                    "functional_kind": request.functional, "horizon": 1., "steps": 32},
                "propagation_request": json.loads(encode(asdict(request))),
                "frozen_dynamics": case.package.manifest(),
                "affine_qualification_policy": policy.manifest(),
                "execution": execution_binding(plugin.registry_entry, config, execution_inputs(request),
                                               matrix_cells=EXPECTED_CELLS)})
    require(len(result) == EXPECTED_CELLS, "frozen qualification count changed")
    require(len(result) * JOB_SECONDS <= TOTAL_APPROVED_SECONDS, "prospective batch exceeds 4H")
    return result


def prepare(store):
    """Publish truthful bindings through owner APIs, without starting a worker."""
    with store._read_transaction():
        original = store.manifest(ORIGINAL)
        arms = {a["method_family_id"]: a for a in original["arms"]
                if a["method_family_id"] in {"exact", "gaussian"}}
        require(set(arms) == {"exact", "gaussian"}, "original method arms missing")
        for arm in arms.values():
            require(arm["model_family_id"] == FAMILY and arm["objective_id"] == "endpoint-x",
                    "cannot relabel original family")
            balance = BudgetLedger(store).balance(arm["arm_id"])
            require(not balance["closed"], "original arm is closed; no automatic reopening")
    cells = frozen_cells(arms)
    recipe = {"schema_version": "pirc27-first-pilot-input-v1", "source_kind": "fixed-generator",
              "source_family": FAMILY, "models": [
                  {"case_id": c.case_id, "model": c.package.manifest(),
                   "initial_mean": c.initial_mean, "initial_covariance": c.initial_covariance}
                  for c in oracle_suite()],
              "independence": "one generator-family source; cases/seeds are not independent data blocks"}
    input_artifact = store.artifact(encode(recipe), role="pilot-input", visibility="synthetic",
                                   block_ids=[FAMILY], study_id=STUDY)
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": STUDY + "-inputs",
                "study_id": STUDY, "blocks": [{"block_id": FAMILY, "dataset_id": FAMILY,
                "release_id": "affine-oracle-suite-v1", "source_block_id": FAMILY,
                "sha256": input_artifact["artifact_id"], "path": input_artifact["artifact_id"],
                "split_role": "validation", "fit_scope": False}]}
    # Empty external dependencies are explicit and honest for this no-terrain pilot.
    evidence = {"entries": [], "reason": "fixed public oracle generator, no P19-22 or terrain dependencies"}
    catalog = {"schema_version": "pirc25-upstream-acceptance-v1", "visibility": "synthetic",
               "source": "PIRC-27 independent no-terrain pilot", "entries": [],
               "source_evidence": evidence, "source_evidence_hash": digest(evidence)}
    catalog_hash = register_acceptance_catalog(store, catalog, digest(catalog))
    cutover = {"study_id": STUDY, "pirc22_cutover": {"mode": "not_applicable"}}
    snapshot = UpstreamSnapshot({"schema_version": "pirc25-upstream-snapshot-v1",
        "visibility": "synthetic", "inputs": [], "studies": [cutover], "cells": [
            {"cell_id": digest(c), "study_id": STUDY, "role": c["study_role"], "upstream_ids": []}
            for c in cells]})
    snapshot_hash = store.publish("upstream-snapshot-" + snapshot.snapshot_hash, snapshot.definition)
    plan = {"schema_version": "pirc25-preregistration-v1", "test_mode": "exploratory",
            "study_ids": [STUDY], "protocol_bindings": [protocol_binding(protocol)],
            "primary_metrics": ["functional_roundoff_upper", "absolute_time_bias_upper", "charged-slot-ms"],
            "selection_rule": "all four frozen cases, both methods; no result-driven subset or thresholds",
            "stopping_rule": "eight frozen jobs; no retry or automatic grid/threshold refinement",
            "comparisons": [{"reference": arms["exact"]["arm_id"],
                             "candidates": [arms["gaussian"]["arm_id"]]}],
            "upstream_bindings": {STUDY: study_binding(snapshot_hash, catalog_hash, cutover)},
            "research_scope": "pilot numerical case enclosures; no population-level inference",
            "orchestrator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "qualification_policies": [c["affine_qualification_policy"] for c in cells],
            "user_execution_authority": DECISION}
    protocol["preregistration_hash"] = PreregistrationGate(store).register_preregistration(plan, digest(plan))
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    authority_hash = store.publish("pirc27-first-batch-authority-" + digest(DECISION), DECISION)
    grant = {"authorization_id": AUTHORIZATION, "study_id": STUDY,
             "expires_at": "2026-11-13T00:00:00+08:00", "evidence_hash": authority_hash,
             "protocol_hash": digest(protocol), "data_root": str(store.path / "artifacts"),
             "test_authorization": False, "purposes": ["execute", "validate", "evaluate", "preview", "export"],
             "visibilities": ["synthetic"], "block_ids": [FAMILY]}
    store.authorize(grant)
    plugin = qualification_plugin()
    payload = {"scope": "user-authorized numerical pilot, not formal qualification",
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
        "qualification": "fixture"}  # Shared schema label for non-formal pilot; never promote it.
    package_hash = store.publish("package-" + digest(package), package)
    spec = {"schema_version": "pirc25-contract-v1", "study_id": STUDY, "experiment_id": STUDY,
        "comparison_family": "pirc27-user-authorized-first-pilot", "code_hash": code_hash(),
        "protocol_hash": digest(protocol), "data_hash": data_binding(protocol),
        "feature_hash": digest({"terrain": "none"}), "selection_hash": digest(plan),
        "arms": list(arms.values()), "cells": cells,
        "runtime_binding": {"root": str(store.path.parent), "store_id": store.store_id},
        "first_batch_authority_hash": authority_hash, "maximum_new_research_slot_seconds": TOTAL_APPROVED_SECONDS,
        "prospective_maximum_unit_slot_seconds": EXPECTED_CELLS * JOB_SECONDS,
        "orchestrator_source_sha256": plan["orchestrator_source_sha256"],
        "scientific_independent_n": 0, "admission": {"mode": "pilot", "purpose": "validate",
            "protocol_id": protocol["protocol_id"], "authorization_id": AUTHORIZATION,
            "package_hash": package_hash, "upstream_snapshot_hash": snapshot_hash,
            "upstream_acceptance_hash": catalog_hash, "upstream_root": str(store.path / "artifacts")}}
    store.register(spec, digest(spec))
    return {"study_id": STUDY, "spec_hash": digest(spec), "cells": len(cells),
            "prospective_slot_seconds": EXPECTED_CELLS * JOB_SECONDS, "state": "REGISTERED_NOT_STARTED"}


def run_next(store):
    spec = store.manifest("study-" + STUDY)["spec"]
    require(spec["code_hash"] == code_hash(), "source changed after registration")
    require(spec["first_batch_authority_hash"] == digest(DECISION), "first-batch authority differs")
    require(spec["orchestrator_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "orchestrator changed after registration; do not silently alter batch")
    registry = CapabilityRegistry()
    registry.register(qualification_plugin())
    with store._read_transaction():
        attempts = store.attempts()
        runs = {e["payload"]["object_id"]: store.manifest(e["payload"]["object_id"])
                for e in store.events() if e["event_kind"] == "MANIFEST"
                and e["payload"]["object_id"].startswith("run-")}
        tried = {runs["run-" + a["run_id"]]["cell_hash"] for a in attempts.values()
                 if runs["run-" + a["run_id"]]["study_id"] == STUDY}
        own = [a for a in attempts.values() if runs["run-" + a["run_id"]]["study_id"] == STUDY]
        require(all(a["state"] in store.TERMINAL for a in own), "existing pilot attempt still active; do not duplicate")
    pending = [c for c in spec["cells"] if digest(c) not in tried]
    if not pending:
        return {"study_id": STUDY, "state": "ALL_REGISTERED_CELLS_ATTEMPTED", "attempts": len(own)}
    cell = pending[0]
    require(not BudgetLedger(store).balance(cell["arm_id"])["closed"], "selected original arm closed")
    result = SharedRunner(store, registry).run_cell(STUDY, digest(cell),
                                                  budget=BudgetSpec(JOB_SECONDS, category="pilot"))
    outcome = {"study_id": STUDY, "cell_hash": digest(cell),
               "model": cell["comparison_dimensions"]["model_configuration"],
               "method": cell["execution"]["config"]["method"], **result}
    if result.get("artifact_id"):
        grant = store.authorization(AUTHORIZATION)
        saved = json.loads(store.read_artifact(result["artifact_id"], purpose="evaluate", authorization=grant))
        analysis = saved["forecast"]["qualification_analysis"]
        outcome["numerical_analysis"] = {k: analysis[k] for k in
            ("status", "checks", "functional_roundoff_upper", "absolute_time_bias_upper",
             "reference_width_upper", "scientific_qualification")}
    return outcome


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run-next"))
    p.add_argument("--runtime-config", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        s = connect(args.runtime_config)
        value = prepare(s) if args.command == "prepare" else run_next(s)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if value.get("state") not in {"FAILED", "TIMEOUT", "INTERRUPTED", "PREFLIGHT_FAILED"} else 1
    except ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
