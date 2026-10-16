"""Sixteen new fixed path cells, never retries of the sealed Euler cohort.

Exploratory public-generator qualification only. Original semantic arms and
aggregate4H are retained; other remaining research is not completed here.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pirc27_first_batch as original
from pirc27_affine_continuation import accounting
from pirc27_oracle_horizon import budget_guard
from pirc27_pilot_registration import register_pilot
from application.propagation_resource_evidence import ResourceEventIndex
from application.propagation_resources import resource_policy
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from application.research_queue import SHARED_CPU_POLICY
from domain.path_qualification import PathQualificationPolicy
from domain.propagation import PropagationRequest
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.oracles import oracle_suite
from experiments.pirc27.path_qualification_plugin import path_qualification_plugin, path_qualification_config
from experiments.pirc27.plugin import execution_inputs
from infrastructure.research_store import ResearchError, digest, encode

STUDY = "pirc27-remaining-path-axes-20261016-v1"
AUTHORIZATION = STUDY + "-grant"
METHODS = ("heun", "reversible-heun", "euler")
EXPECTED_CELLS = 16
JOB_SECONDS = 300
DECISION = {**original.DECISION, "execution_date_local": "2026-10-16",
    "execution_continuation": "请继续完成 PIRC-27，授权回复，GOAL 模式。",
    "scope": "remaining exploratory affine path-method and step/sample axes; original4H and semantic arms; no real data, training, final-eval, failed-cell retries or production",
    "selection_basis": "pre-existing methods, canonical stable anchor and original step/sample axes, not winner-driven refinement; no formal claim after prior Euler exposure"}
require = original.require


def source_binding():
    names = (Path(__file__).name, "pirc27_first_batch.py", "pirc27_affine_continuation.py",
             "pirc27_oracle_horizon.py", "pirc27_pilot_registration.py")
    return digest({n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})


def frozen(arms):
    require(set(arms) == set(METHODS) and all(a["model_family_id"] == original.FAMILY
        and a["method_family_id"] == m and a["objective_id"] == "endpoint-x"
        for m, a in arms.items()), "original path family/method/objective differs")
    cases = oracle_suite()
    require(len(cases) == 4 and {c.case_id for c in cases} == {
        "stable-v1", "near-critical-v1", "anisotropic-v1", "uncertain-initial-v1"}, "canonical four cases required")
    stable = next(c for c in cases if c.case_id == "stable-v1")
    # Never repeat stableH1/32/512 method pilots, nor any failed Euler cell.
    settings = [(m, stable, h, 32, 512) for m in METHODS[:2] for h in (10., 100., 1000.)]
    settings += [(m, c, 1., 32, 512) for m in METHODS[:2] for c in cases if c.case_id != "stable-v1"]
    settings += [("euler", stable, 1., n, k) for n in (16, 64) for k in (256, 1024)]
    plugin, source, cells = path_qualification_plugin(), code_hash(), []
    for method, case, horizon, steps, samples in settings:
        key = digest([method, case.case_id, horizon, steps, samples])
        request = PropagationRequest("remaining-path-"+key, case.package.package_hash,
            case.initial_mean, case.initial_covariance, 0., 0., (horizon,), "endpoint-x", 1147,
            "remaining-path-paired-"+digest([case.case_id, horizon, steps, samples]), arms[method]["arm_id"],
            steps=steps, samples=samples, chunk_size=128, tolerance=.05)
        policy = PathQualificationPolicy(request.request_hash, case.package.package_hash, source,
            method, (0., 0.), (1.,)*4, 1e-12, .025, .025, .05, .025, 2., 1e6, 400001, JOB_SECONDS)
        policy.validate(case.package, request, source)
        config = path_qualification_config(request, policy)
        cells.append({"arm_id": request.arm_id, "block_id": original.FAMILY, "seed": request.seed,
            "horizon": horizon, "plugin_id": plugin.plugin_id, "capability": "generic-rollout",
            "resource_class": "cpu", "resource_sharing": SHARED_CPU_POLICY, "visibility": "synthetic",
            "dimensions": 4, "study_role": "secondary", "execution_role": "qualification",
            "comparison_dimensions": {"model_configuration": case.case_id, "functional_kind": "endpoint-x",
                "method": method, "horizon": horizon, "steps": steps, "samples": samples},
            "propagation_request": json.loads(encode(asdict(request))), "frozen_dynamics": case.package.manifest(),
            "path_qualification_policy": policy.manifest(), "resource_observation_policy": resource_policy(),
            "execution": execution_binding(plugin.registry_entry, config, execution_inputs(request), matrix_cells=EXPECTED_CELLS)})
    require(len(cells) == EXPECTED_CELLS and len({digest(c) for c in cells}) == EXPECTED_CELLS, "complete new path matrix required")
    return cases, cells, {plugin.plugin_id: plugin}


def prepare(store):
    with store._read_transaction():
        values = [a for a in store.manifest(original.ORIGINAL)["arms"] if a["method_family_id"] in METHODS]
        require(len(values) == len(METHODS), "original method arms absent/ambiguous; never replace budget identities")
        arms = {a["method_family_id"]: a for a in values}
        cases, cells, plugins = frozen(arms)
        require(all(not BudgetLedger(store).balance(a["arm_id"])["closed"] for a in values), "selected original path arm closed")
        costs = accounting(store)
        budget_guard(costs, EXPECTED_CELLS)
        recipe = {"schema_version": "pirc27-remaining-path-input-v1", "source_family": original.FAMILY,
            "models": [{"case_id": c.case_id, "model": c.package.manifest(), "initial_mean": c.initial_mean,
                "initial_covariance": c.initial_covariance} for c in cases], "independent_n": 0,
            "independence": "one public generator; cases and seeds are not independent data blocks"}
        spec = register_pilot(store, study=STUDY, authorization=AUTHORIZATION, arms=values, cells=cells,
            plugins=plugins, recipe=recipe, decision=DECISION, source_hash=source_binding(),
            selection_rule="fixed12new Heun/reversibleHeun model/horizon cells plus4canonicalstable Euler step/sample cells; exploratory; no ranking or failed-cell rerun",
            stopping_rule="each cell once; preserve every failure; confirmed native stop/settlement before next cell; a closed arm stays closed while other eligible frozen arms may proceed; no retries/tuning/production",
            policies=[{k: v for k, v in c.items() if k.endswith("_policy")} for c in cells])
        return {"state": "REGISTERED_NOT_STARTED", "study_id": STUDY, "spec_hash": digest(spec),
            "cells": len(cells), "prospective_slot_seconds": EXPECTED_CELLS*JOB_SECONDS, **costs}


def next_cell(spec, attempts, runs, balances):
    own = [a for a in attempts.values() if runs[a["run_id"]]["study_id"] == STUDY]
    require(all(a["state"] == "SUCCEEDED" or a["state"] == "FAILED" and a["error_code"] == "WORKER_FAILED"
        or a["state"] in {"TIMEOUT", "BUDGET_EXHAUSTED"} for a in own), "previous attempt live or admission/control failure; never retry")
    tried = {runs[a["run_id"]]["cell_hash"] for a in own}
    pending = [c for c in spec["cells"] if digest(c) not in tried]
    eligible = [c for c in pending if not balances[c["arm_id"]]["closed"]]
    blocked = [digest(c) for c in pending if balances[c["arm_id"]]["closed"]]
    return own, eligible, blocked


def run_next(store):
    with store._read_transaction():
        spec = store.manifest("study-"+STUDY)["spec"]
        require(spec["code_hash"] == code_hash() and spec["orchestrator_source_sha256"] == source_binding()
            and spec["user_authority_hash"] == digest(DECISION), "frozen source or authority differs")
        arms = {a["method_family_id"]: a for a in spec["arms"]}
        require(len(spec["arms"]) == len(METHODS) and encode(spec["cells"]) == encode(frozen(arms)[1]), "registered matrix differs")
        attempts = store.attempts()
        runs = {a["run_id"]: store.manifest("run-"+a["run_id"]) for a in attempts.values()}
        balances = {a["arm_id"]: BudgetLedger(store).balance(a["arm_id"]) for a in spec["arms"]}
        own, eligible, blocked = next_cell(spec, attempts, runs, balances)
        index = ResourceEventIndex(store.events())
        require(all(index.proof(a["attempt_id"])[2] is None for a in own), "previous native stop/settlement unconfirmed")
        costs = accounting(store)
        budget_guard(costs, len(eligible))
        if not eligible:
            return {"state": "ALL_ELIGIBLE_CELLS_ATTEMPTED", "closed_unattempted_cell_hashes": blocked, **costs}
        cell = eligible[0]
    registry = CapabilityRegistry()
    registry.register(path_qualification_plugin())
    result = SharedRunner(store, registry).run_cell(STUDY, digest(cell), budget=BudgetSpec(JOB_SECONDS, category="pilot"))
    return {"study_id": STUDY, "cell_hash": digest(cell), "dimensions": cell["comparison_dimensions"],
        "closed_unattempted_cell_hashes": blocked, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-next"))
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    try:
        store = original.connect(args.runtime_config)
        result = prepare(store) if args.command == "prepare" else run_next(store)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if result.get("state") not in {"TIMEOUT", "BUDGET_EXHAUSTED", "FAILED", "PREFLIGHT_FAILED", "INTERRUPTED"} else 1
    except ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
