"""Durable local FIFO queues with arm rotation and immutable resource slots.

Claims survive a supervisor crash. Only authoritative budget settlement releases
them; startup never guesses that an old worker has stopped.
"""
import uuid

from infrastructure.research_store import ResearchError, digest
from .research_budget import BudgetLedger


SHARED_CPU_POLICY = "explicit-user-authorized-shared-cpu-v1"


class ResearchQueue:
    def __init__(self, store, *, slots=None):
        self.store = store
        with store.lock():
            try:
                config = store._manifest("runtime-resources")
            except ResearchError as exc:
                if exc.code != "MISSING_INPUT":
                    raise
                config = {"slots": slots if slots is not None else {"cpu": 1, "gpu": 0}}
                values = config["slots"]
                if (set(values) != {"cpu", "gpu"} or any(type(v) is not int or v < 0 for v in values.values())
                        or not 1 <= sum(values.values()) <= 64):
                    raise ResearchError("CONTRACT_MISMATCH", "invalid fixed resource slots")
                sha = store._publish("runtime-resources", config)
                store._append("MANIFEST", {"object_id": "runtime-resources", "sha256": sha},
                              "manifest-" + digest(["runtime-resources", sha]))
            if slots is not None and config["slots"] != slots:
                raise ResearchError("IDENTITY_CONFLICT", "resource configuration is immutable")
            self.slots = config["slots"]

    def enqueue(self, reservation, resource="cpu"):
        if not self.slots.get(resource):
            raise ResearchError("CONTRACT_MISMATCH", "resource class has no configured slot")
        self.store.append("QUEUED", {"reservation_id": reservation["reservation_id"],
            "attempt_id": reservation["attempt_id"], "arm_id": reservation["arm_id"],
            "resource": resource, "dispatch_id": uuid.uuid4().hex}, "queue-" + reservation["reservation_id"])

    def claim(self, reservation_id, *, resource_sharing=None):
        # Each claim gets a new physically verified scope, not a worker-lifetime
        # lock or a cached budget. Duplicate/FIFO/rotation decisions stay atomic.
        with self.store._read_transaction():
            if (self.store.path / "recovery-hold.json").exists():
                raise ResearchError("RECOVERY_REQUIRED", "queue held for authoritative reconciliation")
            reservations, closed = BudgetLedger(self.store)._state()
            queued, claims, last_arm = {}, {}, {}
            for event in self.store._events():
                data = event["payload"]
                if event["event_kind"] == "QUEUED":
                    queued[data["reservation_id"]] = data
                elif event["event_kind"] == "QUEUE_CLAIMED":
                    claims[data["reservation_id"]] = data
                    last_arm[data["resource"]] = data["arm_id"]
            if reservation_id in claims:
                raise ResearchError("IDENTITY_CONFLICT", "attempt already claimed; reconcile its worker before recovery")
            if reservation_id not in queued or reservations[reservation_id]["settled"]:
                raise ResearchError("CONTRACT_MISMATCH", "claim requires an unsettled queued reservation")
            def eligible(item):
                if item["arm_id"] not in closed:
                    return True
                reservation = reservations[item["reservation_id"]]
                policy = reservation.get("execution_policy")
                source = reservation.get("source_code_hash")
                if (reservation.get("budget_mode") != "resource-guarded-no-time-cap"
                        or type(policy) is not dict or type(source) is not str):
                    return False
                from .research_restart_adjudication import (resolve_retry_adjudication,
                    resource_arm_continuation)
                restart = reservation.get("restart_adjudication")
                if restart is not None:
                    receipt = resolve_retry_adjudication(self.store, restart,
                        execution_policy=policy, source_code_hash=source)
                    return receipt["retry_attempt_id"] == reservation["attempt_id"]
                return resource_arm_continuation(self.store, item["arm_id"],
                    execution_policy=policy, source_code_hash=source) is not None
            target = queued[reservation_id]
            if resource_sharing is not None and resource_sharing != SHARED_CPU_POLICY:
                raise ResearchError("CONTRACT_MISMATCH", "unknown resource-sharing policy")
            if resource_sharing is not None and target["resource"] != "cpu":
                raise ResearchError("CONTRACT_MISMATCH", "only CPU work may use the shared-resource policy")
            resource = target["resource"]
            occupied = {item["slot"] for key, item in claims.items()
                        if item["resource"] == resource and not reservations[key]["settled"]}
            available = [slot for slot in range(self.slots[resource]) if slot not in occupied]
            if not available and resource_sharing != SHARED_CPU_POLICY:
                # Waiting grants no execution authority. Do not reconstruct a
                # complete source epoch under the writer lock when another
                # owner must acquire that same lock to settle its occupied slot.
                # Preserve cheap legacy closed-arm rejection; opt-in eligibility
                # is fully rechecked in the fresh claim scope once a slot opens.
                if (target["arm_id"] in closed and reservations[reservation_id].get("budget_mode")
                        != "resource-guarded-no-time-cap"):
                    raise ResearchError("BUDGET_EXHAUSTED", "queued arm closed")
                return None
            if not eligible(target):
                raise ResearchError("BUDGET_EXHAUSTED", "queued arm closed")
            pending = [item for key, item in queued.items() if item["resource"] == resource
                       and key not in claims and not reservations[key]["settled"]
                       and (key == reservation_id or eligible(item))]
            # FIFO inside each arm; yield to the oldest other arm between claims.
            candidates = [item for item in pending if item["arm_id"] != last_arm.get(resource)] or pending
            if not candidates or candidates[0]["reservation_id"] != reservation_id:
                return None
            if available:
                slot = available[0]
            elif resource_sharing == SHARED_CPU_POLICY:
                # CPU workers remain independently supervised and budgeted. The
                # store's OS lock still serializes every authority mutation; this
                # policy only removes compute-capacity exclusion for an explicitly
                # frozen cell. Keep a bounded unique logical slot for audit/UI.
                extra = [slot for slot in range(64) if slot not in occupied]
                if not extra:
                    return None
                slot = extra[0]
            else:
                return None
            claim = {**target, "slot": slot}
            if resource_sharing is not None:
                claim.update(resource_sharing=resource_sharing,
                             configured_slots=self.slots[resource])
            self.store._append("QUEUE_CLAIMED", claim, "claim-" + reservation_id)
            return claim
