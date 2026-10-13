"""Durable compute claims with atomic writers and explicit concurrency permission.

Claims survive a supervisor crash. Only authoritative budget settlement releases
them; startup never guesses that an old worker has stopped.
"""
import uuid

from infrastructure.research_store import ResearchError, digest
from .research_budget import BudgetLedger
from .research_dispatch_policy import concurrent_permission


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
        with self.store._read_transaction():
            reservations, _ = BudgetLedger(self.store)._state()
            if resource not in ("cpu", "gpu") or (not self.slots.get(resource)
                    and concurrent_permission(self.store, reservation["arm_id"], resource) is None):
                raise ResearchError("CONTRACT_MISMATCH", "resource class has no configured slot")
            self._validate_identity(reservation, resource, reservations)
            self.store._append("QUEUED", {"reservation_id": reservation["reservation_id"],
                "attempt_id": reservation["attempt_id"], "arm_id": reservation["arm_id"],
                "resource": resource, "dispatch_id": uuid.uuid4().hex}, "queue-" + reservation["reservation_id"])

    def claim(self, reservation_id):
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
            target = queued[reservation_id]
            self._validate_identity(target, target["resource"], reservations)
            if target["arm_id"] in closed:
                raise ResearchError("BUDGET_EXHAUSTED", "queued arm closed")
            resource = target["resource"]
            occupied = {item["slot"] for key, item in claims.items()
                        if item["resource"] == resource and not reservations[key]["settled"]}
            permission = concurrent_permission(self.store, target["arm_id"], resource)
            if permission is not None:
                # The ticket identifies a unique claim, not an exclusive CPU.
                # An orphaned older claim or an abandoned QUEUED entry cannot
                # globally stop other funded cells. Unsettled reservations stay
                # fully charged against their own arms; claims remain one-shot.
                slot = next(i for i in range(len(occupied) + 1) if i not in occupied)
                claim = {**target, "slot": slot, "dispatch_policy": permission}
                self.store._append("QUEUE_CLAIMED", claim, "claim-" + reservation_id)
                self.store._read_completion(
                    lambda: self._verify_permission(target, permission), lambda: None)
                return claim
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

    def _verify_permission(self, target, expected):
        reservations, closed = BudgetLedger(self.store)._state()
        self._validate_identity(target, target["resource"], reservations)
        if (self.store.path / "recovery-hold.json").exists() or target["arm_id"] in closed:
            raise ResearchError("RECOVERY_REQUIRED", "dispatch became held or closed during claim")
        if concurrent_permission(self.store, target["arm_id"], target["resource"]) != expected:
            raise ResearchError("UNAUTHORIZED_DATA", "dispatch authority changed during claim")

    def _validate_identity(self, target, resource, reservations):
        actual = reservations.get(target.get("reservation_id"))
        if actual is None or any(target.get(key) != actual[key]
                                 for key in ("reservation_id", "attempt_id", "arm_id")):
            raise ResearchError("IDENTITY_CONFLICT", "queue identity differs from funded reservation")
        run = self.store._manifest("run-" + actual["run_id"])
        if resource != run["cell"].get("resource_class", "cpu"):
            raise ResearchError("CONTRACT_MISMATCH", "resource class differs from registered run")
