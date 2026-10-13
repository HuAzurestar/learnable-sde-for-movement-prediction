"""Dispatch the exact registered pilot without loading foreign run manifests.

The original planner and frozen study are unchanged. This entry point checks
both identities, uses the same SharedRunner, never retries an existing attempt,
and never holds the metadata read lock while waiting for a scientific worker.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pirc27_first_batch as batch

SPEC_HASH = "7640186a032e61100a2b83857edddb0ba1ea2e119795a9cfb0474934643724c1"


def run_identity(record, cell):
    spec = record["spec"]
    body = {"study_id": spec["study_id"], "experiment_id": spec["experiment_id"],
            "spec_hash": record["spec_hash"], "arm_id": cell["arm_id"],
            "cell_hash": batch.digest(cell), "seed": cell.get("seed"), "cell": cell}
    # Literal register_run content identity, not a new run or budget identity.
    return batch.digest(body)


def inspect_owned(store):
    with store._read_transaction():
        record = store.manifest("study-" + batch.STUDY)
        spec = record["spec"]
        batch.require(record["spec_hash"] == SPEC_HASH == batch.digest(spec), "frozen study differs")
        batch.require(spec["code_hash"] == batch.code_hash(), "worker source changed")
        batch.require(spec["first_batch_authority_hash"] == batch.digest(batch.DECISION), "authority changed")
        batch.require(spec["orchestrator_source_sha256"] ==
                      hashlib.sha256(Path(batch.__file__).read_bytes()).hexdigest(), "original planner changed")
        batch.require(len(spec["cells"]) == batch.EXPECTED_CELLS and
                      batch.EXPECTED_CELLS * batch.JOB_SECONDS ==
                      spec["prospective_maximum_unit_slot_seconds"] <= batch.TOTAL_APPROVED_SECONDS,
                      "original batch count or resource ceiling changed")
        owned_runs = {run_identity(record, c): c for c in spec["cells"]}
        attempts = [a for a in store.attempts().values() if a["run_id"] in owned_runs]
        pending = [c for c in spec["cells"] if not any(
            a["run_id"] == run_identity(record, c) for a in attempts)]
        balances = [batch.BudgetLedger(store).balance(a["arm_id"]) for a in spec["arms"]]
        return spec, attempts, pending, balances


def next_cell(store):
    spec, attempts, pending, balances = inspect_owned(store)
    batch.require(all(a["state"] in store.TERMINAL for a in attempts),
                  "original attempt still active/unknown; inspect without redispatch")
    if not pending:
        return {"study_id": batch.STUDY, "state": "ALL_REGISTERED_CELLS_ATTEMPTED",
                "attempts": attempts, "balances": balances}
    cell = pending[0]
    batch.require(not next(b for b in balances if b["arm_id"] == cell["arm_id"])["closed"],
                  "original arm closed; no reopening or identity replacement")
    provenance = {"schema_version": "pirc27-first-pilot-dispatch-v1", "study_id": batch.STUDY,
                  "spec_hash": SPEC_HASH, "planner_source_sha256": spec["orchestrator_source_sha256"],
                  "dispatcher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "original_job_seconds": batch.JOB_SECONDS, "automatic_retries": 0}
    store.publish("pirc27-pilot-dispatch-" + batch.digest(provenance), provenance)
    registry = batch.CapabilityRegistry()
    registry.register(batch.qualification_plugin())
    result = batch.SharedRunner(store, registry).run_cell(batch.STUDY, batch.digest(cell),
        budget=batch.BudgetSpec(batch.JOB_SECONDS, category="pilot"))
    # A result-disclosure failure does not trigger another numerical attempt.
    return {"study_id": batch.STUDY, "cell_hash": batch.digest(cell),
            "model": cell["comparison_dimensions"]["model_configuration"],
            "method": cell["execution"]["config"]["method"], **result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("inspect", "run-next"))
    p.add_argument("--runtime-config", type=Path, required=True)
    args = p.parse_args()
    try:
        store = batch.connect(args.runtime_config)
        if args.command == "inspect":
            _, attempts, pending, balances = inspect_owned(store)
            value = {"study_id": batch.STUDY, "owned_attempts": attempts,
                     "pending_cells": len(pending), "balances": balances}
        else:
            value = next_cell(store)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if value.get("state") not in {"FAILED", "TIMEOUT", "INTERRUPTED", "PREFLIGHT_FAILED"} else 1
    except batch.ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
