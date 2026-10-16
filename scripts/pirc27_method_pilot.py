"""Six predeclared stable affine method pilots within the original 4H total."""
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
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from application.research_queue import SHARED_CPU_POLICY
from application.research_recovery import RecoveryRegistry
from domain.affine_mlmc_qualification import AffineMLMCReferencePolicy
from domain.cubature_qualification import CubatureQualificationPolicy
from domain.mixture import MixtureSettings
from domain.mixture_qualification import MixtureQualificationPolicy
from domain.mlmc_pilot import MLMCPilotPolicy
from domain.path_qualification import PathQualificationPolicy
from domain.propagation import PropagationRequest
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.cubature_plugin import cubature_plugin, cubature_config
from experiments.pirc27.mixture_qualification_plugin import mixture_qualification_plugin, mixture_qualification_config
from experiments.pirc27.mlmc_qualification_plugin import (mlmc_qualification_plugin,
    mlmc_qualification_config, mlmc_qualification_recovery_plugin)
from experiments.pirc27.oracles import oracle_suite
from experiments.pirc27.path_qualification_plugin import path_qualification_plugin, path_qualification_config
from experiments.pirc27.plugin import execution_inputs, propagation_resume_command
from infrastructure.research_store import ResearchError, digest, encode

STUDY = "pirc27-stable-method-qualification-20261015-v1"
AUTHORIZATION = STUDY + "-grant"
METHODS = ("euler", "heun", "reversible-heun", "cubature", "mixture", "mlmc")
JOB_SECONDS = 300
require = original.require
DECISION = {**original.DECISION,
    "execution_date_local": "2026-10-15",
    "execution_continuation": "请继续完成 PIRC-27，授权回复，GOAL 模式。",
    "scope": "six stable affine synthetic method qualification pilots under the original aggregate 4H; no real data, final-eval, training or automatic production",
    "selection_basis": "required pre-existing method matrix, not selection by observed numerical performance"}


def source_binding():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__), Path(__file__).with_name("pirc27_pilot_registration.py"),
                    Path(original.__file__), Path(__file__).with_name("pirc27_affine_continuation.py"))})


def frozen(arms):
    case = next(c for c in oracle_suite() if c.case_id == "stable-v1")
    cells, plugins = [], {}
    source = code_hash()
    for method in METHODS:
        request = PropagationRequest("method-pilot-stable-" + method, case.package.package_hash,
            case.initial_mean, case.initial_covariance, 0., 0., (1.,), "endpoint-x", 1147,
            "pirc27-method-pilot-stable-paired-v1", arms[method]["arm_id"],
            samples=512, steps=8 if method == "mlmc" else 32, chunk_size=128, tolerance=.05)
        policies = {}
        role = "qualification"
        if method in {"euler", "heun", "reversible-heun"}:
            plugin = path_qualification_plugin()
            policy = PathQualificationPolicy(request.request_hash, case.package.package_hash, source,
                method, (0., 0.), (1.,)*4, 1e-12, .025, .025, .05, .025, 2., 1e6, 400001, JOB_SECONDS)
            policy.validate(case.package, request, source)
            policies["path_qualification_policy"] = policy.manifest()
            config = path_qualification_config(request, policy)
        elif method == "cubature":
            plugin = cubature_plugin(qualification=True)
            policy = CubatureQualificationPolicy(request.request_hash, case.package.package_hash,
                source, 1e-12, 1e-8, .025, 1e6, (1.,)*4, 400001, JOB_SECONDS)
            policy.validate(case.package, request, source)
            policies["cubature_qualification_policy"] = policy.manifest()
            config = cubature_config(request, policy=policy)
        elif method == "mixture":
            plugin = mixture_qualification_plugin()
            mixture = MixtureSettings(2, .25, 1e-5, (1.,)*4, .01, 1_000_000, JOB_SECONDS).bind(case.package, request, source)
            policy = MixtureQualificationPolicy(request.request_hash, case.package.package_hash,
                source, mixture.policy_hash, 1e-12, 1e-8, .025, .05, 1e6, 400001, JOB_SECONDS)
            policy.validate(case.package, request, mixture, source)
            policies.update(mixture_policy=mixture.manifest(), mixture_qualification_policy=policy.manifest())
            config = mixture_qualification_config(request, mixture, policy)
        else:
            plugin = mlmc_qualification_plugin()
            counts = (256, 128, 128)
            pilot = MLMCPilotPolicy(request.request_hash, 2053, "pirc27-method-pilot-independent-production-v1",
                .025, .025, .8, 3.5, 64, 2048, 600000)
            reference = AffineMLMCReferencePolicy(request.request_hash, case.package.package_hash,
                source, pilot.policy_hash, counts, 1e-12, 1e6, (1.,)*4, 400001, JOB_SECONDS)
            reference.validate(case.package, request, pilot, source)
            policies.update(mlmc_pilot_policy=asdict(pilot), affine_mlmc_reference_policy=reference.manifest())
            config = mlmc_qualification_config(request, reference)
            role = "pilot"
        plugins[plugin.plugin_id] = plugin
        cells.append({"arm_id": request.arm_id, "block_id": original.FAMILY, "seed": request.seed,
            "horizon": 1., "plugin_id": plugin.plugin_id,
            "capability": "coupled-level" if method == "mlmc" else "generic-rollout",
            "resource_class": "cpu", "resource_sharing": SHARED_CPU_POLICY, "visibility": "synthetic",
            "dimensions": 4, "study_role": "secondary", "execution_role": role,
            "comparison_dimensions": {"model_configuration": "stable-v1", "functional_kind": "endpoint-x",
                "method": method, "horizon": 1., "steps": request.steps, "samples": request.samples},
            "propagation_request": json.loads(encode(asdict(request))), "frozen_dynamics": case.package.manifest(),
            **policies, "execution": execution_binding(plugin.registry_entry, config, execution_inputs(request),
                                                       matrix_cells=len(METHODS))})
    return case, cells, plugins


def prepare(store):
    with store._read_transaction():
        source = store.manifest(original.ORIGINAL)
        arms = {a["method_family_id"]: a for a in source["arms"] if a["method_family_id"] in METHODS}
        require(set(arms) == set(METHODS), "original method arms absent; never create replacement budget arms")
        for a in arms.values():
            require(a["model_family_id"] == original.FAMILY and a["objective_id"] == "endpoint-x",
                    "original model/objective identity differs")
            require(not BudgetLedger(store).balance(a["arm_id"])["closed"], "original method arm closed")
        costs = accounting(store)
        require(costs["reserved_ms"] == 0 and costs["charged_ms"] + len(METHODS)*JOB_SECONDS*1000 <= 14400000,
                "aggregate 4H budget/settlement guard failed")
        case, cells, plugins = frozen(arms)
        recipe = {"schema_version": "pirc27-stable-method-pilot-input-v1", "source_family": original.FAMILY,
            "model": case.package.manifest(), "initial_mean": case.initial_mean,
            "initial_covariance": case.initial_covariance, "independent_n": 0}
        spec = register_pilot(store, study=STUDY, authorization=AUTHORIZATION, arms=list(arms.values()),
            cells=cells, plugins=plugins, recipe=recipe, decision=DECISION, source_hash=source_binding(),
            selection_rule="all six declared remaining mean-x methods on stable-v1 only; not a phase diagram",
            stopping_rule="each frozen cell once, no retries/refinement; retain failed qualification; stop closed arms; no production launch",
            policies=[{k:v for k,v in c.items() if k.endswith("_policy")} for c in cells],
            recovery_commands={mlmc_qualification_plugin().plugin_id: propagation_resume_command})
        return {"state": "REGISTERED_NOT_STARTED", "study_id": STUDY, "spec_hash": digest(spec),
                "cells": len(cells), "prospective_slot_seconds": len(cells)*JOB_SECONDS, **costs}


def run_next(store):
    with store._read_transaction():
        spec = store.manifest("study-" + STUDY)["spec"]
        require(spec["code_hash"] == code_hash() and spec["orchestrator_source_sha256"] == source_binding(),
                "frozen method-pilot source differs")
        require(spec["user_authority_hash"] == digest(DECISION), "method pilot authority differs")
        attempts = store.attempts()
        runs = {a["run_id"]:store.manifest("run-" + a["run_id"]) for a in attempts.values()}
        own = [a for a in attempts.values() if runs[a["run_id"]]["study_id"] == STUDY]
        require(all(a["state"] in store.TERMINAL for a in own), "existing pilot still active")
        tried = {runs[a["run_id"]]["cell_hash"] for a in own}
        pending = [c for c in spec["cells"] if digest(c) not in tried]
        costs = accounting(store)
        require(costs["reserved_ms"] == 0 and costs["charged_ms"] + len(pending)*JOB_SECONDS*1000 <= 14400000,
                "aggregate 4H budget/settlement guard failed")
        if not pending:
            return {"state": "ALL_FROZEN_CELLS_ATTEMPTED", **costs}
        cell = pending[0]
        require(not BudgetLedger(store).balance(cell["arm_id"])["closed"], "selected original arm closed")
        _, _, plugins = frozen({a["method_family_id"]:a for a in spec["arms"]})
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    for plugin in plugins.values():
        registry.register(plugin)
    recovery.register(mlmc_qualification_recovery_plugin())
    result = SharedRunner(store, registry, recovery_registry=recovery).run_cell(STUDY, digest(cell),
        budget=BudgetSpec(JOB_SECONDS, category="pilot"))
    return {"study_id": STUDY, "method": cell["comparison_dimensions"]["method"], "cell_hash": digest(cell), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run-next"])
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    try:
        store = original.connect(args.runtime_config)
        result = prepare(store) if args.command == "prepare" else run_next(store)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if result.get("state") not in {"TIMEOUT", "INTERRUPTED", "PREFLIGHT_FAILED", "FAILED"} else 1
    except ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
