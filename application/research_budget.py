"""Atomic reservations derived only from the shared authoritative event chain."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from infrastructure.research_store import ResearchError, ResearchStore, digest


@dataclass(frozen=True)
class BudgetSpec:
    job_seconds: float = 7200
    arm_seconds: int = 86400
    category: str = "job"
    allocation_only: bool = False

    def validate(self):
        cap = {"smoke": 900, "pilot": 1800, "job": 7200}.get(self.category)
        if (cap is None or type(self.allocation_only) is not bool or not math.isfinite(self.job_seconds)
                or not 0 < self.job_seconds <= (86400 if self.allocation_only else cap)
                or self.arm_seconds != 86400):
            raise ResearchError("CONTRACT_MISMATCH", "job/category or cumulative arm budget invalid")


class BudgetLedger:
    def __init__(self, store: ResearchStore):
        self.store = store

    def _state(self):
        reservations, closed, soft_arms, closures = {}, set(), set(), {}
        events = self.store._events()
        for event in events:
            data = event["payload"]
            if event["event_kind"] in {"RESERVE", "SETTLE"}:
                reservations[data["reservation_id"]] = data
                if event["event_kind"] == "SETTLE" and (
                        data["monotonic_elapsed_ms"] is None
                        or data["outcome"] == "BUDGET_EXHAUSTED"
                        or (data["arm_id"] not in soft_arms
                            and data["outcome"] in {"TIMEOUT", "INTERRUPTED"})):
                    closed.add(data["arm_id"])
            elif event["event_kind"] == "ARM_CLOSED":
                closed.add(data["arm_id"])
                closures.setdefault(data["arm_id"], []).append(event["hash"])
            elif event["event_kind"] == "ARM_CONTINUATION":
                from .research_continuation import approval
                receipt = self.store._manifest(data["manifest_id"])
                decision = approval(self.store, receipt["approval"])
                arm = receipt["arm_id"]
                costs = [r for r in reservations.values() if r["arm_id"] == arm]
                used = sum(r["charged_ms"] if r["settled"] else r["reserved_ms"] for r in costs)
                from datetime import datetime
                approved_at = datetime.fromisoformat(event["created_at"])
                stops = [e["hash"] for e in events[:event["sequence"] - 1]
                         if e["event_kind"] in {"WORKER_TREE_STOPPED", "WORKER_STOP_CONFIRMED"}
                         and e["payload"].get("reservation_id") in {r["reservation_id"] for r in costs}]
                if (receipt.get("schema_version") != "pirc25-arm-continuation-v1"
                        or digest(receipt) != data["sha256"] or arm not in decision["arm_ids"]
                        or receipt["closed_event_hashes"] != closures.get(arm, [])
                        or receipt["retained_cost_ms"] != used or used >= 86400000
                        or receipt.get("stop_event_hashes") != stops
                        or not datetime.fromisoformat(decision["work_started_at"]) <= approved_at
                               < datetime.fromisoformat(decision["work_deadline"])
                        or not all(r["settled"] for r in costs)):
                    raise ResearchError("CONTRACT_MISMATCH", "continuation cost/closure/approval differs")
                closed.discard(arm)
                soft_arms.add(arm)
        return reservations, closed

    def _balance(self, arm_id):
        reservations, closed = self._state()
        used = sum(r["charged_ms"] if r["settled"] else r["reserved_ms"]
                   for r in reservations.values() if r["arm_id"] == arm_id)
        return {"arm_id": arm_id, "remaining_ms": max(0, 86400000 - used),
                "committed_ms": used, "closed": arm_id in closed or (self.store.path / "recovery-hold.json").exists()}

    def balance(self, arm_id):
        # Continuation proof has several manifest references. Reuse one verified
        # prefix ONLY in this short physical scope, then recompute on the final
        # uncached prefix before returning. No worker-lifetime authority cache.
        result = {}
        with self.store._read_transaction():
            result.update(self._balance(arm_id))
            self.store._read_completion(lambda: result.update(self._balance(arm_id)), lambda: None)
        return result

    def _continuation_policy(self, arm_id):
        from .research_continuation import approval
        reference = None
        for event in self.store._events():
            if event["event_kind"] == "ARM_CONTINUATION":
                receipt = self.store._manifest(event["payload"]["manifest_id"])
                if receipt["arm_id"] == arm_id:
                    reference = receipt["approval"]
        return None if reference is None else approval(self.store, reference, live=True)

    def continue_arm(self, arm_id, *, approval_ref):
        """Append bounded original-arm continuation after stopped/settled checks."""
        from .research_continuation import approval
        from infrastructure.process_tree import process_may_be_alive
        with self.store._read_transaction():
            decision = approval(self.store, approval_ref, live=True)
            if arm_id not in decision["arm_ids"]:
                raise ResearchError("UNAUTHORIZED_DATA", "arm continuation outside approval")
            if (self.store.path / "recovery-hold.json").exists():
                raise ResearchError("RECOVERY_REQUIRED", "recovery hold prohibits continuation")
            event_id = "continue-" + digest([arm_id, approval_ref])
            existing = [e for e in self.store._events() if e["event_id"] == event_id]
            if existing:
                self._state()
                return existing[0]["payload"]
            reservations, _ = self._state()
            costs = [r for r in reservations.values() if r["arm_id"] == arm_id]
            if not costs or any(not r["settled"] for r in costs):
                raise ResearchError("RECOVERY_REQUIRED", "all original reservations must be settled")
            used = sum(r["charged_ms"] for r in costs)
            if used >= 86400000:
                raise ResearchError("BUDGET_EXHAUSTED", "cumulative hard cap cannot be reopened")
            events = self.store._events()
            attempts = self.store._attempts()
            stops = {e["payload"].get("reservation_id") for e in events
                     if e["event_kind"] in {"WORKER_TREE_STOPPED", "WORKER_STOP_CONFIRMED"}}
            relevant = {r["reservation_id"] for r in costs}
            for worker in (e["payload"] for e in events if e["event_kind"] == "WORKER_STARTED"
                           and e["payload"].get("reservation_id") in relevant):
                if (worker["reservation_id"] not in stops
                        or attempts[worker["attempt_id"]]["state"] not in self.store.TERMINAL):
                    raise ResearchError("RECOVERY_REQUIRED", "whole-tree stop/terminal proof absent")
                if process_may_be_alive(worker.get("pid")):
                    raise ResearchError("WORKER_ACTIVE", "recorded worker may still be alive")
            receipt = {"schema_version": "pirc25-arm-continuation-v1", "arm_id": arm_id,
                       "approval": approval_ref, "retained_cost_ms": used,
                       "closed_event_hashes": [e["hash"] for e in events if e["event_kind"] == "ARM_CLOSED"
                                              and e["payload"]["arm_id"] == arm_id],
                       "stop_event_hashes": [e["hash"] for e in events
                           if e["event_kind"] in {"WORKER_TREE_STOPPED", "WORKER_STOP_CONFIRMED"}
                           and e["payload"].get("reservation_id") in relevant],
                       "verified_prefix_hash": events[-1]["hash"]}
            reference = {"manifest_id": "arm-continuation-" + digest(receipt), "sha256": digest(receipt)}
            self.store.publish(reference["manifest_id"], receipt)
            self.store._append("ARM_CONTINUATION", reference, event_id)
            return reference

    def reserve(self, attempt_id: str, budget: BudgetSpec, *, worker_slot=0):
        budget.validate()
        # Reuse only a verified prefix under the same short writer lock. The
        # final uncached physical pass precedes returning a funded reservation.
        with self.store._read_transaction():
            if (self.store.path / "recovery-hold.json").exists():
                raise ResearchError("RECOVERY_REQUIRED", "authoritative recovery holds all budgets for reconciliation")
            attempts = self.store._attempts()
            if attempt_id not in attempts:
                raise ResearchError("MISSING_INPUT", "attempt not registered")
            attempt = attempts[attempt_id]
            run = self.store._manifest("run-" + attempt["run_id"])
            reservation_id = digest([self.store.store_id, attempt_id])
            requested = math.ceil(budget.job_seconds * 1000)
            reservations, _ = self._state()
            if reservation_id in reservations:
                existing = reservations[reservation_id]
                if existing["reserved_ms"] != requested or existing["worker_slot"] != worker_slot:
                    raise ResearchError("IDENTITY_CONFLICT", "reservation replay changed budget")
                return existing
            if attempt["state"] != "REGISTERED":
                raise ResearchError("CONTRACT_MISMATCH", "reservation requires registered attempt")
            balance = self._balance(run["arm_id"])
            continuation_policy = self._continuation_policy(run["arm_id"])
            if budget.allocation_only and continuation_policy is None:
                raise ResearchError("UNAUTHORIZED_DATA", "soft allocation requires an approved continuation")
            if balance["closed"] or requested > balance["remaining_ms"]:
                # Classify and persist rejection while still holding the same
                # authority lock as the capacity check. No reservation exists
                # for this attempt, so never settle another caller's work.
                charged = sum(r["charged_ms"] for r in reservations.values()
                              if r["arm_id"] == run["arm_id"] and r["settled"])
                final = balance["closed"] or requested > 86400000 - charged
                code = "BUDGET_EXHAUSTED" if final else "BUDGET_BUSY"
                self.store._transition(attempt_id,
                    "BUDGET_EXHAUSTED" if final else "PREFLIGHT_FAILED", error_code=code)
                raise ResearchError(code, "arm budget cannot fund this job" if final else
                                    "capacity is reserved by other attempts; retry explicitly after settlement")
            data = {"reservation_id": reservation_id, "arm_id": run["arm_id"],
                    "attempt_id": attempt_id, "run_id": run["run_id"], "study_id": run["study_id"],
                    "reserved_ms": requested, "charged_ms": 0, "monotonic_elapsed_ms": None,
                    "worker_slot": worker_slot, "settled": False}
            if continuation_policy is not None:
                data["work_deadline"] = continuation_policy["work_deadline"]
            self.store._append("RESERVE", data, "reserve-" + reservation_id)
            return data

    def settle(self, reservation_id: str, elapsed_ms: int | None, *, outcome: str):
        with self.store._read_transaction():
            return self._settle(reservation_id, elapsed_ms, outcome=outcome)

    def _settle(self, reservation_id: str, elapsed_ms: int | None, *, outcome: str):
        if elapsed_ms is not None and (not isinstance(elapsed_ms, int) or elapsed_ms < 0):
            raise ResearchError("CONTRACT_MISMATCH", "invalid elapsed cost")
        reservations, _ = self._state()
        if reservation_id not in reservations:
            raise ResearchError("MISSING_INPUT", "reservation missing")
        reservation = reservations[reservation_id]
        if reservation["settled"]:
            if reservation["monotonic_elapsed_ms"] != elapsed_ms or reservation["outcome"] != outcome:
                raise ResearchError("IDENTITY_CONFLICT", "settlement replay changed cost/outcome")
            return reservation
        charged = reservation["reserved_ms"] if elapsed_ms is None else elapsed_ms
        data = {**reservation, "settled": True, "charged_ms": charged,
                "monotonic_elapsed_ms": elapsed_ms, "outcome": outcome}
        self.store._append("SETTLE", data, "settle-" + reservation_id)
        continued = any(e["event_kind"] == "ARM_CONTINUATION"
                        and self.store._manifest(e["payload"]["manifest_id"])["arm_id"] == reservation["arm_id"]
                        for e in self.store._events())
        if (elapsed_ms is None or outcome == "BUDGET_EXHAUSTED"
                or (not continued and outcome in {"TIMEOUT", "INTERRUPTED"})
                or self._balance(reservation["arm_id"])["remaining_ms"] == 0):
            self.store._append("ARM_CLOSED", {"arm_id": reservation["arm_id"], "reason": outcome},
                               "close-" + reservation_id)
        return data

    def recover_unknown(self, reservation_id: str, *, stop_evidence_hash=None):
        """Reconcile stopped work at full cost, never silently release a live slot.

        An operator must attest whole-tree stop for any possibly launched attempt.
        The OS probe is an additional veto, not a replacement for that evidence.
        """
        with self.store._read_transaction():
            reservations, _ = self._state()
            if reservation_id not in reservations:
                raise ResearchError("MISSING_INPUT", "reservation missing")
            result = reservations[reservation_id]
            if not result["settled"]:
                attempt = self.store._attempts()[result["attempt_id"]]
                workers = [e["payload"] for e in self.store._events()
                           if e["event_kind"] == "WORKER_STARTED"
                           and e["payload"].get("reservation_id") == reservation_id]
                if workers or attempt["state"] != "REGISTERED":
                    from infrastructure.process_tree import process_may_be_alive
                    if any(process_may_be_alive(worker.get("pid")) for worker in workers):
                        raise ResearchError("WORKER_ACTIVE", "recorded worker/tree may still be alive; reservation retained")
                    if not re.fullmatch(r"[0-9a-f]{64}", str(stop_evidence_hash or "")):
                        raise ResearchError("RECOVERY_REQUIRED", "whole-tree stop evidence is required before releasing a claimed slot")
                    self.store._append("WORKER_STOP_CONFIRMED", {
                        "reservation_id": reservation_id, "attempt_id": result["attempt_id"],
                        "stop_evidence_hash": stop_evidence_hash,
                        "kind": "operator-whole-tree-stop-attestation",
                        "observed_pids": [worker["pid"] for worker in workers]},
                        "stop-confirmed-" + reservation_id)
                result = self._settle(reservation_id, None, outcome="INTERRUPTED")
            attempt = self.store._attempts()[result["attempt_id"]]
            if attempt["state"] not in self.store.TERMINAL:
                self.store._append("ARM_CLOSED", {"arm_id": result["arm_id"], "reason": "recovered interrupted attempt"},
                                   "recovery-close-" + reservation_id)
                self.store._transition(result["attempt_id"], "INTERRUPTED", error_code="UNKNOWN_WORKER_COST")
            return result
