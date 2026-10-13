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
from application.research_admission import AdmissionGate
from infrastructure.process_tree import process_may_be_alive

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
    closed = {b["arm_id"] for b in balances if b["closed"]}
    blocked = [c for c in pending if c["arm_id"] in closed]
    eligible = [c for c in pending if c["arm_id"] not in closed]
    if not eligible:
        return {"study_id": batch.STUDY, "state": "REMAINING_CELLS_BLOCKED",
                "attempts": attempts, "balances": balances,
                "blocked_cell_hashes": [batch.digest(c) for c in blocked]}
    cell = eligible[0]
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
            "method": cell["execution"]["config"]["method"],
            "blocked_cell_hashes": [batch.digest(c) for c in blocked], **result}


def reconcile_published(store):
    """Finish one owner-validated result whose settlement write was interrupted.

    This never executes a worker.  It accepts only the exact frozen study, a
    still-running owned attempt, native whole-tree stop evidence, a dead
    recorded worker, and a durable successful settlement made by the original
    supervisor. Publication alone cannot establish its final deadline verdict.
    """
    spec, attempts, _, _ = inspect_owned(store)
    active = [a for a in attempts if a["state"] not in store.TERMINAL]
    batch.require(len(active) == 1 and active[0]["state"] == "RUNNING",
                  "exactly one owned RUNNING attempt is required")
    attempt = active[0]
    with store._read_transaction():
        current = store.attempts()[attempt["attempt_id"]]
        batch.require(current == attempt, "attempt changed during reconciliation")
        run = store.manifest("run-" + attempt["run_id"])
        cell = run["cell"]
        events = store.events()
        reservations = [e for e in events if e["event_kind"] == "RESERVE"
                        and e["payload"].get("attempt_id") == attempt["attempt_id"]]
        batch.require(len(reservations) == 1 and not reservations[0]["payload"]["settled"],
                      "one unsettled reservation is required")
        reservation = reservations[0]["payload"]
        settlements = [e for e in events if e["event_kind"] == "SETTLE"
                       and e["payload"].get("reservation_id") == reservation["reservation_id"]]
        batch.require(len(settlements) == 1 and
                      settlements[0]["payload"].get("outcome") == "SUCCEEDED",
                      "durable supervisor success settlement required; publication alone cannot promote success")
        stops = [e for e in events if e["event_kind"] == "WORKER_TREE_STOPPED"
                 and e["payload"].get("reservation_id") == reservation["reservation_id"]]
        workers = [e for e in events if e["event_kind"] == "WORKER_STARTED"
                   and e["payload"].get("reservation_id") == reservation["reservation_id"]]
        batch.require(len(stops) == len(workers) == 1, "unique worker/stop evidence is required")
        stop, worker = stops[0], workers[0]
        elapsed = stop["payload"].get("observed_elapsed_ms")
        batch.require(stop["payload"].get("confirmation") ==
                      "native-job-or-process-group-no-running-descendants",
                      "native whole-tree stop confirmation is required")
        batch.require(type(elapsed) is int and 0 <= elapsed <= reservation["reserved_ms"],
                      "measured stopped cost is outside the reservation")
        settled = settlements[0]["payload"]
        batch.require(settled.get("settled") is True and
                      settled.get("monotonic_elapsed_ms") == settled.get("charged_ms") == elapsed,
                      "successful settlement differs from native stopped cost")
        batch.require(not process_may_be_alive(worker["payload"].get("pid")),
                      "recorded worker may still be alive")
        admissions = [e for e in events if e["event_kind"] == "ADMISSION"
                      and e["payload"].get("attempt_id") == attempt["attempt_id"]]
        batch.require(len(admissions) == 1, "unique admission evidence is required")
        admission_hash = admissions[0]["payload"]["admission_hash"]
        receipt = store.manifest("admission-" + admission_hash)
        candidates = []
        for event in events:
            if event["sequence"] <= stop["sequence"]:
                continue
            object_id = event["payload"].get("object_id") if event["event_kind"] == "MANIFEST" else None
            if not isinstance(object_id, str) or not object_id.startswith("artifact-"):
                continue
            metadata = store.manifest(object_id)
            if (metadata.get("study_id") != batch.STUDY or metadata.get("role") != "result"
                    or event["sequence"] <= stop["sequence"]):
                continue
            content = store._verified_artifact_content(metadata)
            value = json.loads(content)
            if (value.get("spec_hash") == SPEC_HASH and value.get("cell_hash") == run["cell_hash"]
                    and value.get("admission_hash") == admission_hash):
                candidates.append((metadata, content, value))
        batch.require(len(candidates) == 1, "unique standard-published result is required")
        metadata, content, value = candidates[0]
        plugin = batch.qualification_plugin()
        AdmissionGate(store).result_validator(receipt, spec, cell, plugin)(value)
        batch.require(batch.encode(value) == content and value.get("status") == "SUCCEEDED",
                      "published result does not match owner validation")
    # Never reconstruct a lost supervisor verdict from its artifact alone.
    terminal = store.transition(attempt["attempt_id"], "SUCCEEDED",
                                artifact_id=metadata["artifact_id"])
    return {"study_id": batch.STUDY, "attempt_id": attempt["attempt_id"],
            "state": terminal["state"], "artifact_id": metadata["artifact_id"],
            "elapsed_ms": settled["charged_ms"], "reconciled": True}


def recover_stopped(store, *, attempt_id, stop_hash):
    """Settle a confirmed stopped orphan at recorded cost as INTERRUPTED.

    Uses the existing ledger's interrupted-arm policy. The published numerical
    result remains immutable partial evidence. A success settlement must instead
    use reconcile-published; no outcome, reservation or arm is rewritten here.
    """
    _, attempts, _, _ = inspect_owned(store)
    owned = [a for a in attempts if a["attempt_id"] == attempt_id]
    batch.require(len(owned) == 1, "attempt does not belong to frozen first batch")
    with store._read_transaction():
        current = store.attempts()[attempt_id]
        batch.require(current["state"] in {"RUNNING", "INTERRUPTED"},
                      "stopped recovery requires original running or interrupted attempt")
        events = store.events()
        reservations = [e for e in events if e["event_kind"] == "RESERVE"
                        and e["payload"].get("attempt_id") == attempt_id]
        batch.require(len(reservations) == 1, "unique original reservation required")
        reservation = reservations[0]["payload"]
        stops = [e for e in events if e["event_kind"] == "WORKER_TREE_STOPPED"
                 and e["payload"].get("reservation_id") == reservation["reservation_id"]]
        workers = [e for e in events if e["event_kind"] == "WORKER_STARTED"
                   and e["payload"].get("reservation_id") == reservation["reservation_id"]]
        batch.require(len(stops) == len(workers) == 1, "unique worker and native stop required")
        stop, worker = stops[0], workers[0]
        elapsed = stop["payload"].get("observed_elapsed_ms")
        batch.require(stop["hash"] == stop_hash and
                      stop["payload"].get("attempt_id") == attempt_id and
                      worker["payload"].get("attempt_id") == attempt_id and
                      reservations[0]["sequence"] < worker["sequence"] < stop["sequence"] and
                      stop["payload"].get("confirmation") ==
                      "native-job-or-process-group-no-running-descendants",
                      "supplied evidence is not the original native whole-tree stop")
        batch.require(type(elapsed) is int and elapsed > 0,
                      "native stopped measurement missing or invalid")
        batch.require(not process_may_be_alive(worker["payload"].get("pid")),
                      "recorded worker may still be alive")
        settlements = [e for e in events if e["event_kind"] == "SETTLE"
                       and e["payload"].get("reservation_id") == reservation["reservation_id"]]
        batch.require(not settlements or (len(settlements) == 1 and
                      settlements[0]["payload"].get("outcome") == "INTERRUPTED" and
                      settlements[0]["payload"].get("charged_ms") == elapsed and
                      settlements[0]["payload"].get("monotonic_elapsed_ms") == elapsed),
                      "existing settlement cannot be replaced")
        if current["state"] == "INTERRUPTED":
            batch.require(current["error_code"] == "SUPERVISOR_FINALIZATION_UNPROVEN" and
                          current.get("artifact_id") is None and len(settlements) == 1,
                          "terminal recovery identity differs")
        proof = {"schema_version": "pirc27-stopped-reconciliation-v1",
                 "study_id": batch.STUDY, "spec_hash": SPEC_HASH,
                 "attempt_id": attempt_id, "reservation_id": reservation["reservation_id"],
                 "worker_event_hash": worker["hash"], "stop_event_hash": stop_hash,
                 "measured_elapsed_ms": elapsed, "outcome": "INTERRUPTED"}
        store.append("STOPPED_RECONCILIATION", proof, "pirc27-recovery-" + batch.digest(proof))
        settled = batch.BudgetLedger(store).settle(reservation["reservation_id"], elapsed,
                                                  outcome="INTERRUPTED")
        terminal = store.transition(attempt_id, "INTERRUPTED",
                                   error_code="SUPERVISOR_FINALIZATION_UNPROVEN")
        return {"study_id": batch.STUDY, "attempt_id": attempt_id, "state": terminal["state"],
                "elapsed_ms": settled["charged_ms"], "reconciled": True,
                "arm_closed": batch.BudgetLedger(store).balance(reservation["arm_id"])["closed"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("inspect", "reconcile-published", "recover-stopped", "run-next"))
    p.add_argument("--runtime-config", type=Path, required=True)
    p.add_argument("--attempt-id")
    p.add_argument("--stop-evidence-hash")
    args = p.parse_args()
    if args.command == "recover-stopped" and not (args.attempt_id and args.stop_evidence_hash):
        p.error("recover-stopped requires --attempt-id and --stop-evidence-hash")
    try:
        store = batch.connect(args.runtime_config)
        if args.command == "inspect":
            _, attempts, pending, balances = inspect_owned(store)
            value = {"study_id": batch.STUDY, "owned_attempts": attempts,
                     "pending_cells": len(pending), "balances": balances}
        elif args.command == "reconcile-published":
            value = reconcile_published(store)
        elif args.command == "recover-stopped":
            value = recover_stopped(store, attempt_id=args.attempt_id, stop_hash=args.stop_evidence_hash)
        else:
            value = next_cell(store)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if value.get("state") not in {"FAILED", "TIMEOUT", "INTERRUPTED", "PREFLIGHT_FAILED",
                                              "REMAINING_CELLS_BLOCKED"} else 1
    except batch.ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
