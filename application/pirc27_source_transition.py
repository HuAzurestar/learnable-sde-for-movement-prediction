"""Explicit infrastructure-only source restarts of the original affine pilot.

Old studies, costs and attempts remain immutable. This narrowly scoped bridge
does not resume numerical state or transfer an old qualification to new code.
"""
from copy import deepcopy

from application.research_continuation import approval, require
from infrastructure.research_store import digest


FAMILY = "pirc27-affine-oracle-v1"
INFRASTRUCTURE_FAILURES = {"SUPERVISOR_FINALIZATION_UNPROVEN", "QUEUE_REJECTED",
                           "HEARTBEAT_LOST", "WRITER_BUSY", "BUDGET_BUSY"}


def numerical_identity(cell):
    """Drop only the enumerated infrastructure/source-derived fields."""
    value = deepcopy(cell)
    require(value.get("block_id") == FAMILY and value.get("dimensions") == 4
            and value.get("plugin_id") == "affine-propagation-qualification"
            and value.get("execution_role") == "qualification"
            and value.get("visibility") == "synthetic", "not the original affine pilot")
    policy = value["affine_qualification_policy"]
    require(value["execution"]["config"]["qualification_policy_hash"] == digest(policy),
            "qualification policy binding differs")
    policy.pop("code_hash")
    value["execution"]["config"].pop("qualification_policy_hash")
    value["execution"].pop("resource_plan_hash")
    value.pop("resource_sharing", None)
    return value


def validate_restart(store, run, parent_id, reference):
    require(type(reference) is dict and set(reference) == {"manifest_id", "sha256"},
            "exact source transition reference required")
    document = store._manifest(reference["manifest_id"])
    require(digest(document) == reference["sha256"]
            and document.get("schema_version") == "pirc27-source-transition-v1"
            and document.get("target_run_id") == run["run_id"], "source transition differs")
    decision = approval(store, document["approval"], live=True)
    require(decision.get("pirc27_total_slot_seconds") == 14400
            and decision.get("infrastructure_retries") is True
            and run["arm_id"] in decision["arm_ids"], "source restart outside approval")
    original = store._manifest("run-" + document["source_run_id"])
    source = store._manifest("study-" + original["study_id"])["spec"]
    target = store._manifest("study-" + run["study_id"])["spec"]
    require(source["admission"]["mode"] == target["admission"]["mode"] == "pilot"
            and source["code_hash"] == document["source_code_hash"]
            and target["code_hash"] == document["target_code_hash"]
            and original["cell"]["affine_qualification_policy"]["code_hash"] == source["code_hash"]
            and run["cell"]["affine_qualification_policy"]["code_hash"] == target["code_hash"]
            and source["arms"] and original["arm_id"] == run["arm_id"]
            and numerical_identity(original["cell"]) == numerical_identity(run["cell"]),
            "source restart changes numerical configuration or original arm")
    states = store._attempts()
    related = []
    for attempt in states.values():
        saved_run = store._manifest("run-" + attempt["run_id"])
        if (saved_run["arm_id"] == run["arm_id"]
                and saved_run["cell"].get("block_id") == FAMILY
                and numerical_identity(saved_run["cell"]) == numerical_identity(run["cell"])):
            related.append(attempt)
    require(related and related[-1]["attempt_id"] == parent_id
            and all(a["state"] in store.TERMINAL - {"SUCCEEDED"} for a in related)
            and related[-1]["error_code"] in INFRASTRUCTURE_FAILURES,
            "latest stopped infrastructure-failure parent required")
    require(states[document["source_parent_attempt_id"]]["run_id"] == original["run_id"]
            and any(a["attempt_id"] == document["source_parent_attempt_id"] for a in related),
            "original source parent differs")
    require(len(related) < 3, "at most two explicit cross-source retries", "RETRY_EXHAUSTED")
    reservations = {}
    events = store._events()
    ids = {a["attempt_id"] for a in related}
    for event in events:
        if event["event_kind"] in {"RESERVE", "SETTLE"} and event["payload"].get("attempt_id") in ids:
            reservations[event["payload"]["reservation_id"]] = event["payload"]
    require(all(r["settled"] for r in reservations.values()), "source cost unsettled", "RECOVERY_REQUIRED")
    stopped = {e["payload"].get("reservation_id") for e in events
               if e["event_kind"] in {"WORKER_TREE_STOPPED", "WORKER_STOP_CONFIRMED"}}
    require(all(e["payload"]["reservation_id"] in stopped for e in events
                if e["event_kind"] == "WORKER_STARTED" and e["payload"].get("attempt_id") in ids),
            "source worker stop proof missing", "RECOVERY_REQUIRED")
    return deepcopy(reference)
