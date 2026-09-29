"""Durable local FIFO queues with arm rotation and immutable resource slots.

Claims survive a supervisor crash. Only authoritative budget settlement releases
them; startup never guesses that an old worker has stopped.
"""
import uuid

from infrastructure.research_store import ResearchError, digest
from .research_budget import BudgetLedger


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

    def claim(self, reservation_id):
        with self.store.lock():
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
            target = queued[reservation_id]
            if target["arm_id"] in closed:
                raise ResearchError("BUDGET_EXHAUSTED", "queued arm closed")
            resource = target["resource"]
            occupied = {item["slot"] for key, item in claims.items()
                        if item["resource"] == resource and not reservations[key]["settled"]}
            available = [slot for slot in range(self.slots[resource]) if slot not in occupied]
            pending = [item for key, item in queued.items() if item["resource"] == resource
                       and key not in claims and not reservations[key]["settled"] and item["arm_id"] not in closed]
            # FIFO inside each arm; yield to the oldest other arm between claims.
            candidates = [item for item in pending if item["arm_id"] != last_arm.get(resource)] or pending
            if not available or not candidates or candidates[0]["reservation_id"] != reservation_id:
                return None
            claim = {**target, "slot": available[0]}
            self.store._append("QUEUE_CLAIMED", claim, "claim-" + reservation_id)
            return claim
