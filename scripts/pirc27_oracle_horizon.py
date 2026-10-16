"""Fixed measured Euler horizon cohort; existing arm, store and aggregate cap."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pirc27_first_batch as original
from pirc27_affine_continuation import accounting
from pirc27_pilot_registration import register_pilot
from application.propagation_resources import resource_policy
from application.propagation_resource_evidence import ResourceEventIndex
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


STUDY = "pirc27-measured-euler-horizons-20261015-v1"
AUTHORIZATION = STUDY + "-grant"
HORIZONS = (1., 10., 100., 1000.)
EXPECTED_CELLS = 16
JOB_SECONDS = 300
DECISION = {**original.DECISION, "execution_date_local": "2026-10-15",
    "execution_continuation": "请继续完成 PIRC-27，授权回复，GOAL 模式。",
    "scope": "current-source measured affine Euler qualification/horizon cohort; original aggregate4H and semantic arm; no real data, training, final-eval, retries or automatic production",
    "selection_basis": "all four existing oracle cases and pre-existing four horizons; no performance-selected case or parameter refinement"}
require = original.require


def source_binding():
    files = (Path(__file__), Path(__file__).with_name("pirc27_pilot_registration.py"),
             Path(original.__file__), Path(__file__).with_name("pirc27_affine_continuation.py"))
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def frozen(arm):
    require(arm["model_family_id"] == original.FAMILY and arm["method_family_id"] == "euler"
            and arm["objective_id"] == "endpoint-x", "original Euler model/method/objective identity differs")
    cases = oracle_suite()
    require(len(cases) == 4 and len({c.case_id for c in cases}) == 4, "original four oracle cases required")
    plugin, source = path_qualification_plugin(), code_hash()
    cells = []
    for case in cases:
        for horizon in HORIZONS:
            request = PropagationRequest("measured-euler-"+case.case_id+"-"+str(int(horizon)),
                case.package.package_hash, case.initial_mean, case.initial_covariance, 0., 0., (horizon,),
                "endpoint-x", 1147, "pirc27-oracle-paired-"+case.case_id+"-"+str(int(horizon)),
                arm["arm_id"], samples=512, steps=32, chunk_size=128, tolerance=.05)
            # Retain the prior fixed path qualification limits. A long-horizon
            # failure is a result, not permission to loosen or rerun the policy.
            policy = PathQualificationPolicy(request.request_hash, case.package.package_hash, source,
                "euler", (0., 0.), (1.,)*4, 1e-12, .025, .025, .05, .025, 2., 1e6, 400001, JOB_SECONDS)
            policy.validate(case.package, request, source)
            config = path_qualification_config(request, policy)
            cells.append({"arm_id": arm["arm_id"], "block_id": original.FAMILY, "seed": request.seed,
                "horizon": horizon, "plugin_id": plugin.plugin_id, "capability": "generic-rollout",
                "resource_class": "cpu", "resource_sharing": SHARED_CPU_POLICY, "visibility": "synthetic",
                "dimensions": 4, "study_role": "secondary", "execution_role": "qualification",
                "comparison_dimensions": {"model_configuration": case.case_id, "functional_kind": "endpoint-x",
                    "method": "euler", "horizon": horizon, "steps": 32, "samples": 512},
                "propagation_request": json.loads(encode(asdict(request))), "frozen_dynamics": case.package.manifest(),
                "path_qualification_policy": policy.manifest(), "resource_observation_policy": resource_policy(),
                "execution": execution_binding(plugin.registry_entry, config, execution_inputs(request),
                                                 matrix_cells=EXPECTED_CELLS)})
    require(len(cells) == EXPECTED_CELLS and len({digest(c) for c in cells}) == EXPECTED_CELLS,
            "complete unique frozen horizon matrix required")
    return cases, cells, {plugin.plugin_id: plugin}


def budget_guard(costs, pending):
    require(type(pending) is int and 0 <= pending <= EXPECTED_CELLS
            and type(costs) is dict and set(costs) == {"charged_ms", "reserved_ms"}
            and all(type(v) is int and v >= 0 for v in costs.values()), "invalid original aggregate accounting")
    require(costs["reserved_ms"] == 0 and costs["charged_ms"] + pending*JOB_SECONDS*1000 <= 14400000,
            "original aggregate4H budget/settlement guard failed")


def prepare(store):
    with store._read_transaction():
        arms = [a for a in store.manifest(original.ORIGINAL)["arms"] if a["method_family_id"] == "euler"]
        require(len(arms) == 1, "original Euler arm absent/ambiguous; do not create another budget identity")
        cases, cells, plugins = frozen(arms[0])
        require(not BudgetLedger(store).balance(arms[0]["arm_id"])["closed"], "original Euler arm closed")
        costs = accounting(store)
        budget_guard(costs, EXPECTED_CELLS)
        recipe = {"schema_version": "pirc27-measured-oracle-horizon-input-v1", "source_family": original.FAMILY,
            "models": [{"case_id": c.case_id, "model": c.package.manifest(), "initial_mean": c.initial_mean,
                        "initial_covariance": c.initial_covariance} for c in cases],
            "independent_n": 0, "independence": "one public generator family; cases/seeds are not independent data blocks"}
        spec = register_pilot(store, study=STUDY, authorization=AUTHORIZATION, arms=arms, cells=cells,
            plugins=plugins, recipe=recipe, decision=DECISION, source_hash=source_binding(),
            selection_rule="all original4oracle cases x4pre-existing horizons; fixed Euler512paths32steps; no winner-driven refinement",
            stopping_rule="one attempt per frozen cell; no retries/tuning; preserve failed worker outcomes and advance only to other frozen cells after confirmed native stop/settlement; stop live/unsettled/closed or admission/control failures; no production",
            policies=[{k:v for k,v in c.items() if k.endswith("_policy")} for c in cells])
        return {"study_id": STUDY, "spec_hash": digest(spec), "cells": len(cells),
                "prospective_slot_seconds": EXPECTED_CELLS*JOB_SECONDS, **costs, "state": "REGISTERED_NOT_STARTED"}


def run_next(store):
    with store._read_transaction():
        spec = store.manifest("study-"+STUDY)["spec"]
        require(spec["code_hash"] == code_hash() and spec["orchestrator_source_sha256"] == source_binding()
                and spec["user_authority_hash"] == digest(DECISION), "frozen source or execution decision differs")
        require(len(spec["arms"]) == 1 and encode(spec["cells"]) == encode(frozen(spec["arms"][0])[1]),
                "registered cells differ from the complete frozen horizon recipe")
        attempts = store._attempts()
        own = [a for a in attempts.values() if store.manifest("run-"+a["run_id"])["study_id"] == STUDY]
        require(all(a["state"] == "SUCCEEDED" or (a["state"] == "FAILED" and a["error_code"] == "WORKER_FAILED")
                    for a in own), "previous attempt live or admission/control failure; stop, never retry automatically")
        index = ResourceEventIndex(store.events())
        require(all(index.proof(a["attempt_id"])[2] is None for a in own),
                "previous worker stop/cost not confirmed; no new dispatch")
        tried = {store.manifest("run-"+a["run_id"])["cell_hash"] for a in own}
        pending = [c for c in spec["cells"] if digest(c) not in tried]
        costs = accounting(store)
        budget_guard(costs, len(pending))
        if not pending:
            return {"state": "ALL_FROZEN_CELLS_ATTEMPTED", **costs}
        cell = pending[0]
        require(not BudgetLedger(store).balance(cell["arm_id"])["closed"], "original Euler arm closed")
    # Never enclose a native worker in the metadata writer phase.
    registry = CapabilityRegistry()
    registry.register(path_qualification_plugin())
    result = SharedRunner(store, registry).run_cell(STUDY, digest(cell), budget=BudgetSpec(JOB_SECONDS, category="pilot"))
    return {"study_id": STUDY, "cell_hash": digest(cell), "case": cell["comparison_dimensions"]["model_configuration"],
            "horizon": cell["horizon"], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run-next"])
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    try:
        store = original.connect(args.runtime_config)
        result = prepare(store) if args.command == "prepare" else run_next(store)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result.get("state") not in {"FAILED", "TIMEOUT", "INTERRUPTED", "PREFLIGHT_FAILED"} else 1
    except ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
