"""Continue independent frozen cells after one explicitly diagnosed closure.

Never edit/reseal the original recipe, retry the interruption, or reopen its
arm. This separate entry point accepts one exact reviewed stopped receipt;
every other interruption remains a hard stop. No scientific gate is waived.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pirc27_remaining_path_axes as original
from application.propagation_resource_evidence import ResourceEventIndex, resource_history
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_evidence import authorize_study, require_unexpired_disclosure
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError, digest, encode

require = original.require
SPEC_HASH = "83c3f3aec76b0449cf6c6bc91e79b6ba832af1d1360dcb62ab829906ece3b08d"
SOURCE_BINDING = "e6e73dc7b049ab7fc809710ea2e0344ebb2d87aaf306c06552324938c4a865b2"
KNOWN_LOG = "60b45dc8e2b2e92775b30bbee6b690b05cca0c2fc0020a3eabfe0590ab370fec"
REVIEWED = {
    "attempt_id": "f6b05e77497e48a6bad2c5f04455c0f7",
    "cell_hash": "98dc0413b42fab65683e26f249e93a1053d2425fee8918610997e187e761ef8f",
    "arm_id": "pirc27-affine-reversible-heun-mean-x",
    "native_stop_hash": "136c52b01b4221e01fc0cbf0a706883f219e22cbce654467d8e0733d32ad7d56",
    "settlement_hash": "8eebe93c7d7b1c573f40137ee2f6d908caaa176d4dac49cce5c43e5a303a33e1",
    "charged_ms": 139783,
    "publication_hash": "7074ea7796520626a6d09b5558c36f8e568d89b792f3859d2bbdd7f9a0858507",
    "packet_hash": "7dcbb041f30f7e123706801f52ab21a3dcca9f196cc79a24665b750947d27f66",
}


def select_cells(spec, attempts, runs, balances, reviewed_ids):
    """Pure planning: the caller must verify every reviewed ID's actual proof."""
    own = [a for a in attempts.values() if runs[a["run_id"]]["study_id"] == original.STUDY]
    require(reviewed_ids == {REVIEWED["attempt_id"]}
        and reviewed_ids <= {a["attempt_id"] for a in own}, "exact diagnosed interruption required")
    for a in own:
        run = runs[a["run_id"]]
        allowed = (a["state"] == "SUCCEEDED"
            or a["state"] == "FAILED" and a["error_code"] == "WORKER_FAILED"
            or a["state"] in {"TIMEOUT", "BUDGET_EXHAUSTED"} and balances[run["arm_id"]]["closed"]
            or a["state"] == "INTERRUPTED" and a["error_code"] == "HEARTBEAT_LOST"
                and a["attempt_id"] in reviewed_ids and balances[run["arm_id"]]["closed"])
        require(allowed, "live, unknown or unreviewed control failure; no dispatch")
    tried = {runs[a["run_id"]]["cell_hash"] for a in own}
    require(len(tried) == len(own), "duplicate attempts; never retry or select best")
    pending = [c for c in spec["cells"] if digest(c) not in tried]
    eligible = [c for c in pending if not balances[c["arm_id"]]["closed"]]
    closed = [digest(c) for c in pending if balances[c["arm_id"]]["closed"]]
    return own, eligible, closed


def verify_reviewed(attempt, run, balance, stop, settled, resource):
    require(attempt["attempt_id"] == REVIEWED["attempt_id"]
        and attempt["state"] == "INTERRUPTED" and attempt["error_code"] == "HEARTBEAT_LOST"
        and attempt["artifact_id"] is None and run["cell_hash"] == REVIEWED["cell_hash"]
        and run["arm_id"] == REVIEWED["arm_id"] and balance["closed"] is True
        and stop["hash"] == REVIEWED["native_stop_hash"]
        and settled["hash"] == REVIEWED["settlement_hash"]
        and settled["payload"]["charged_ms"] == REVIEWED["charged_ms"]
        and resource["publication_event"]["hash"] == REVIEWED["publication_hash"]
        and resource["artifact"]["sha256"] == REVIEWED["packet_hash"],
        "diagnosed closed-arm receipt differs; no override or retry")


def verify_log(store, attempt_id):
    path = store.path / "artifacts" / (".attempt-"+attempt_id) / "worker.log"
    with opened_regular_file(store.path/"artifacts", path, maximum_bytes=262144) as (stream, size, verify):
        raw = stream.read(size+1)
        verify()
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == KNOWN_LOG
            and raw.decode("utf-8").splitlines()[-1]
                == "domain.errors.DataValidationError: discrete covariance is not symmetric",
            "owned stopped failure log is not the diagnosed numeric rejection")


def preflight(store):
    with store._read_transaction():
        spec = store.manifest("study-"+original.STUDY)["spec"]
        require(digest(spec) == SPEC_HASH and spec["code_hash"] == original.code_hash()
            and spec["orchestrator_source_sha256"] == original.source_binding() == SOURCE_BINDING
            and spec["user_authority_hash"] == digest(original.DECISION), "original frozen source/authority differs")
        arms = {a["method_family_id"]: a for a in spec["arms"]}
        originals = {a["method_family_id"]: a for a in store.manifest(original.original.ORIGINAL)["arms"]
                     if a["method_family_id"] in original.METHODS}
        require(arms == originals and len(spec["arms"]) == len(original.METHODS)
            and encode(spec["cells"]) == encode(original.frozen(arms)[1]), "original arms or matrix differ")
        attempts = store.attempts()
        runs = {a["run_id"]: store.manifest("run-"+a["run_id"]) for a in attempts.values()}
        own = [a for a in attempts.values() if runs[a["run_id"]]["study_id"] == original.STUDY]
        balances = {a["arm_id"]: BudgetLedger(store).balance(a["arm_id"]) for a in spec["arms"]}
        grant = store.authorization(original.AUTHORIZATION)
        authorize_study(store, original.STUDY, grant, "preview")
        index, reviewed = ResourceEventIndex(store.events()), set()
        for a in own:
            run = runs[a["run_id"]]
            cells = [c for c in spec["cells"] if digest(c) == run["cell_hash"]]
            require(len(cells) == 1 and run["spec_hash"] == SPEC_HASH
                and run["arm_id"] == cells[0]["arm_id"], "foreign or unbound attempt")
            stop, settled, why = index.proof(a["attempt_id"])
            require(why is None and settled["payload"]["outcome"] == a["state"]
                and settled["payload"]["charged_ms"] == stop["payload"]["observed_elapsed_ms"],
                "native stop/terminal settlement missing or inconsistent")
            history = resource_history(store, spec, cells[0], [a], grant, index=index)["attempts"][0]
            require(history["status"] == "RECORDED", "actual resource evidence required")
            if a["state"] == "INTERRUPTED":
                verify_reviewed(a, run, balances[run["arm_id"]], stop, settled, history)
                verify_log(store, a["attempt_id"])
                reviewed.add(a["attempt_id"])
            elif a["state"] == "FAILED" and a["error_code"] == "WORKER_FAILED":
                verify_log(store, a["attempt_id"])
        own, eligible, closed = select_cells(spec, attempts, runs, balances, reviewed)
        costs = original.accounting(store)
        original.budget_guard(costs, len(eligible))
        report = {"state": "VERIFIED_CLOSED_ARM_ISOLATION", "study_id": original.STUDY,
            "spec_hash": SPEC_HASH, "attempted_cells": len(own), "eligible_cells": len(eligible),
            "closed_unattempted_cell_hashes": closed, "next_cell": eligible[0] if eligible else None,
            "reviewed_interruption": dict(REVIEWED), "balances": balances, **costs}
        authorize_study(store, original.STUDY, grant, "preview")
    require_unexpired_disclosure(store, original.STUDY, grant, "preview")
    return report


def run_next(store):
    report = preflight(store)
    cell = report.pop("next_cell")
    if cell is None:
        return {**report, "state": "ALL_ELIGIBLE_CELLS_ATTEMPTED"}
    registry = CapabilityRegistry()
    registry.register(original.path_qualification_plugin())
    result = SharedRunner(store, registry).run_cell(original.STUDY, digest(cell),
        budget=BudgetSpec(original.JOB_SECONDS, category="pilot"))
    return {"study_id": original.STUDY, "cell_hash": digest(cell),
        "dimensions": cell["comparison_dimensions"], "closed_unattempted_cell_hashes":
            report["closed_unattempted_cell_hashes"], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run-next"))
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    try:
        store = original.original.connect(args.runtime_config)
        result = preflight(store) if args.command == "preflight" else run_next(store)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result.get("state") in {"VERIFIED_CLOSED_ARM_ISOLATION",
            "ALL_ELIGIBLE_CELLS_ATTEMPTED", "SUCCEEDED"} else 1
    except ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
