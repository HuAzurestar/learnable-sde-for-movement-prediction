"""Read-only eligibility evidence for a stopped infrastructure interruption.

Returns a review candidate, NEVER a reopen/retry/funding permission. The old
events, closure and fees are untouched. Native liveness and version-lineage
reconciliation remain necessary before a separately reviewed atomic action.
"""

from copy import deepcopy

from infrastructure.research_store import ResearchError, digest, identifier


def _require(condition, message):
    if not condition:
        raise ResearchError("RECOVERY_REQUIRED", message)


def inspect_restart_review(store, attempt_id):
    """Bind exact terminal/settlement/native-stop/closure events, no writes."""
    identifier(attempt_id)
    result = {}
    def check():
        _require(not (store.path / "recovery-hold.json").exists(), "store recovery hold remains")
        events = store._events()
        attempts = store._attempts()
        parent = attempts.get(attempt_id)
        _require(parent is not None and parent["state"] == "INTERRUPTED"
                 and parent.get("error_code") == "SUPERVISOR_ERROR"
                 and parent.get("artifact_id") is None
                 and parent.get("artifact_manifest_hash") is None,
                 "parent is not a result-free supervisor interruption")
        run = store._manifest("run-" + parent["run_id"])
        _require(run["arm_id"] == "pirc28-k1", "review is scoped to original PIRC-28 K1")
        related = [value for value in attempts.values() if value["run_id"] == parent["run_id"]]
        _require(all(value["state"] in store.TERMINAL for value in related)
                 and not any(value["state"] == "SUCCEEDED" for value in related)
                 and len(related) < 3, "cell active, succeeded or finite retry allowance exhausted")
        reservations = [event for event in events if event["event_kind"] == "RESERVE"
                        and event["payload"].get("attempt_id") == attempt_id]
        _require(len(reservations) == 1, "one original reservation required")
        reserve = reservations[0]
        reservation = reserve["payload"]
        reservation_id = reservation["reservation_id"]
        settlements = [event for event in events if event["event_kind"] == "SETTLE"
                       and event["payload"].get("reservation_id") == reservation_id]
        _require(len(settlements) == 1, "one immutable original settlement required")
        settlement = settlements[0]
        charged = settlement["payload"]
        _require(charged.get("settled") is True and charged.get("outcome") == "INTERRUPTED"
                 and type(charged.get("monotonic_elapsed_ms")) is int
                 and 0 < charged["monotonic_elapsed_ms"] == charged.get("charged_ms")
                 and charged["charged_ms"] <= reservation["reserved_ms"]
                 and all(charged.get(field) == reservation.get(field) for field in
                         ("arm_id", "attempt_id", "run_id", "study_id", "reserved_ms", "worker_slot"))
                 and charged["arm_id"] == run["arm_id"], "cost or settlement source is unknown/inconsistent")
        stops = [event for event in events if event["event_kind"] == "WORKER_TREE_STOPPED"
                 and event["payload"].get("reservation_id") == reservation_id
                 and event["payload"].get("attempt_id") == attempt_id]
        _require(len(stops) == 1, "one original whole-tree stop required")
        stopped = stops[0]
        launches = [event for event in events if event["event_kind"] == "WORKER_STARTED"
                    and event["payload"].get("reservation_id") == reservation_id
                    and event["payload"].get("attempt_id") == attempt_id]
        _require(len(launches) == 1 and type(launches[0]["payload"].get("pid")) is int
                 and launches[0]["payload"]["pid"] > 1
                 and reserve["sequence"] < launches[0]["sequence"] < stopped["sequence"],
                 "one original native launch identity required")
        _require(stopped["payload"].get("confirmation") == "native-job-or-process-group-no-running-descendants"
                 and type(stopped["payload"].get("observed_elapsed_ms")) is int
                 and stopped["payload"].get("observed_elapsed_ms") == charged["charged_ms"]
                 and reserve["sequence"] < stopped["sequence"] < settlement["sequence"],
                 "stop evidence source/cost/order differs")
        terminals = [event for event in events if event["event_kind"] == "ATTEMPT"
                     and event["payload"].get("attempt_id") == attempt_id
                     and event["payload"].get("state") in store.TERMINAL]
        _require(len(terminals) == 1 and terminals[0]["payload"] == parent,
                 "one immutable terminal attempt required")
        closures = [event for event in events if event["event_kind"] == "ARM_CLOSED"
                    and event["payload"].get("arm_id") == run["arm_id"]]
        _require(len(closures) == 1 and closures[0]["payload"].get("reason") == "INTERRUPTED"
                 and settlement["sequence"] < closures[0]["sequence"] < terminals[0]["sequence"],
                 "other closure reason/history requires separate adjudication")
        _require(not any(event["event_kind"] == "CHECKPOINT"
                         and event["payload"].get("attempt_id") == attempt_id for event in events),
                 "checkpoint parent requires a distinct verified recovery path")
        result.clear()
        result.update({"schema_version": "pirc25-restart-review-candidate-v1",
            "state": "CANDIDATE_FOR_REVIEW", "execution_permitted": False,
            "store_id": store.store_id, "arm_id": run["arm_id"], "parent_attempt_id": attempt_id,
            "run_id": parent["run_id"], "study_id": run["study_id"], "cell_hash": run["cell_hash"],
            "reservation_id": reservation_id, "retained_cost_ms": charged["charged_ms"],
            "remaining_retry_count": 3-len(related),
            "event_hashes": {"reserve": reserve["hash"], "launch": launches[0]["hash"], "stop": stopped["hash"],
                "settlement": settlement["hash"], "closure": closures[0]["hash"],
                "terminal": terminals[0]["hash"]},
            "source_run_hash": digest(run),
            "requirements": ["fresh-native-worker-identity-check", "reviewed-atomic-adjudication",
                             "immutable-code-and-grant-lineage", "cross-version-cell-retry-accounting"],
            "recovery_level": "restart-only-no-checkpoint", "history_mutated": False})
    with store._read_transaction():
        check()
        store._read_completion(check, lambda: None)
    return deepcopy(result)
