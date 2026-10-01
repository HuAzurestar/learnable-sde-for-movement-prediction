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

    def validate(self):
        cap = {"smoke": 900, "pilot": 1800, "job": 7200}.get(self.category)
        if (cap is None or not math.isfinite(self.job_seconds)
                or not 0 < self.job_seconds <= cap or self.arm_seconds != 86400):
            raise ResearchError("CONTRACT_MISMATCH", "job/category or cumulative arm budget invalid")


class BudgetLedger:
    def __init__(self, store: ResearchStore):
        self.store = store

    def _state(self):
        reservations, closed = {}, set()
        for event in self.store._events():
            data = event["payload"]
            if event["event_kind"] in {"RESERVE", "SETTLE"}:
                reservations[data["reservation_id"]] = data
                if event["event_kind"] == "SETTLE" and (
                        data["monotonic_elapsed_ms"] is None
                        or data["outcome"] in {"TIMEOUT", "BUDGET_EXHAUSTED", "INTERRUPTED"}):
                    closed.add(data["arm_id"])
            elif event["event_kind"] == "ARM_CLOSED":
                closed.add(data["arm_id"])
        return reservations, closed

    def _balance(self, arm_id):
        reservations, closed = self._state()
        used = sum(r["charged_ms"] if r["settled"] else r["reserved_ms"]
                   for r in reservations.values() if r["arm_id"] == arm_id)
        return {"arm_id": arm_id, "remaining_ms": max(0, 86400000 - used),
                "committed_ms": used, "closed": arm_id in closed or (self.store.path / "recovery-hold.json").exists()}

    def balance(self, arm_id):
        with self.store.lock():
            return self._balance(arm_id)

    def reserve(self, attempt_id: str, budget: BudgetSpec, *, worker_slot=0):
        budget.validate()
        with self.store.lock():
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
            self.store._append("RESERVE", data, "reserve-" + reservation_id)
            return data

    def settle(self, reservation_id: str, elapsed_ms: int | None, *, outcome: str):
        with self.store.lock():
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
        if (elapsed_ms is None or outcome in {"TIMEOUT", "BUDGET_EXHAUSTED", "INTERRUPTED"}
                or self._balance(reservation["arm_id"])["remaining_ms"] == 0):
            self.store._append("ARM_CLOSED", {"arm_id": reservation["arm_id"], "reason": outcome},
                               "close-" + reservation_id)
        return data

    def recover_unknown(self, reservation_id: str, *, stop_evidence_hash=None):
        """Reconcile stopped work at full cost, never silently release a live slot.

        An operator must attest whole-tree stop for any possibly launched attempt.
        The OS probe is an additional veto, not a replacement for that evidence.
        """
        with self.store.lock():
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
