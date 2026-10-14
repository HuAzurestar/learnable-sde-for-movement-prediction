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


@dataclass(frozen=True)
class ResourceBudgetSpec:
    """Explicit, published PIRC-28 policy; never inferred from a null cap."""
    execution_policy: dict
    source_code_hash: str
    job_seconds: None = None
    arm_seconds: None = None
    category: str = "resource-guarded"

    def validate(self):
        if self.job_seconds is not None or self.arm_seconds is not None or self.category != "resource-guarded":
            raise ResearchError("CONTRACT_MISMATCH", "resource budget requires explicit null time caps")


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

    def _balance(self, arm_id, *, resource_guarded=False, execution_policy=None,
                 source_code_hash=None):
        reservations, closed = self._state()
        selected = [r for r in reservations.values() if r["arm_id"] == arm_id]
        costs = [r["charged_ms"] if r["settled"] else r["reserved_ms"] for r in selected]
        unknown = any(cost is None for cost in costs)
        used = None if unknown else sum(costs)
        unbounded = any(r["reserved_ms"] is None for r in selected)
        is_closed = arm_id in closed or (not resource_guarded and unbounded)
        if resource_guarded and is_closed and execution_policy is not None:
            from .research_restart_adjudication import resource_arm_continuation
            is_closed = resource_arm_continuation(self.store, arm_id,
                execution_policy=execution_policy, source_code_hash=source_code_hash) is None
        return {"arm_id": arm_id,
                "remaining_ms": None if resource_guarded else max(0, 86400000 - (used or 0)),
                "committed_ms": used,
                "closed": is_closed or (self.store.path / "recovery-hold.json").exists()}

    def balance(self, arm_id, *, execution_policy=None, source_code_hash=None,
                attempt_id=None):
        with (self.store._read_transaction() if execution_policy is not None else self.store.lock()):
            if execution_policy is not None:
                from .research_execution_policy import resolve_policy
                resolve_policy(self.store, execution_policy, arm_id=arm_id, source_code_hash=source_code_hash)
            result = self._balance(arm_id, resource_guarded=execution_policy is not None,
                                   execution_policy=execution_policy, source_code_hash=source_code_hash)
            if (result["closed"] and execution_policy is not None and attempt_id is not None
                    and not (self.store.path / "recovery-hold.json").exists()):
                from .research_restart_adjudication import adjudication_for_attempt
                adjudication_for_attempt(self.store, attempt_id, execution_policy=execution_policy,
                                         source_code_hash=source_code_hash)
                result["closed"] = False
            return result

    def reserve(self, attempt_id: str, budget: BudgetSpec, *, worker_slot=0):
        budget.validate()
        if type(budget) is ResourceBudgetSpec:
            return self._reserve_resource_guarded(attempt_id, budget, worker_slot=worker_slot)
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
            from .research_disposition import declared_execution_disposition
            declaration = declared_execution_disposition(run["cell"])
            if declaration is not None:
                if attempt["state"] == "REGISTERED":
                    self.store._transition(attempt_id, "PREFLIGHT_FAILED", error_code=declaration["status"])
                raise ResearchError(declaration["status"], "registered cell is declared non-executable")
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
                if final and not balance["closed"]:
                    # The durable fuse must precede terminal persistence: a
                    # rejected job has no SETTLE event to preserve closure if
                    # recording its terminal attempt is interrupted.
                    self.store._append("ARM_CLOSED",
                        {"arm_id": run["arm_id"], "reason": code},
                        "refusal-close-" + reservation_id)
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

    def _reserve_resource_guarded(self, attempt_id, budget, *, worker_slot):
        from .research_execution_policy import resolve_policy
        if type(worker_slot) is not int or worker_slot != 0:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "resource policy permits only worker slot zero")
        with self.store._read_transaction():
            attempt = self.store._attempts().get(attempt_id)
            if attempt is None:
                raise ResearchError("MISSING_INPUT", "attempt not registered")
            run = self.store._manifest("run-" + attempt["run_id"])
            resolve_policy(self.store, budget.execution_policy, arm_id=run["arm_id"],
                           source_code_hash=budget.source_code_hash)
            spec = self.store._manifest("study-" + run["study_id"])["spec"]
            if spec["code_hash"] != budget.source_code_hash or run["cell"].get("visibility") != "synthetic":
                raise ResearchError("UNAUTHORIZED_DATA", "time policy requires the exact synthetic source epoch")
            reservation_id = digest([self.store.store_id, attempt_id])
            reservations, _ = self._state()
            binding = {"execution_policy": budget.execution_policy,
                       "source_code_hash": budget.source_code_hash, "budget_mode": "resource-guarded-no-time-cap"}
            if reservation_id in reservations:
                existing = reservations[reservation_id]
                if (existing["reserved_ms"] is not None or existing["worker_slot"] != worker_slot
                        or any(existing.get(key) != value for key, value in binding.items())):
                    raise ResearchError("IDENTITY_CONFLICT", "reservation replay changed resource policy")
                return existing
            if attempt["state"] != "REGISTERED":
                raise ResearchError("CONTRACT_MISMATCH", "reservation requires registered attempt")
            adjudication = None
            if self._balance(run["arm_id"], resource_guarded=True,
                    execution_policy=budget.execution_policy,
                    source_code_hash=budget.source_code_hash)["closed"]:
                from .research_restart_adjudication import adjudication_for_attempt
                adjudication = adjudication_for_attempt(self.store, attempt_id,
                    execution_policy=budget.execution_policy, source_code_hash=budget.source_code_hash)
            # One shared PIRC-28 worker, including other epochs and finite jobs.
            # Check and append under the same authority lock, before queueing.
            if any(not r["settled"] and r["arm_id"] in {"pirc28-k1", "pirc28-k2"}
                   for r in reservations.values()):
                raise ResearchError("BUDGET_BUSY", "PIRC-28 worker reservation is already occupied")
            data = {"reservation_id": reservation_id, "arm_id": run["arm_id"],
                "attempt_id": attempt_id, "run_id": run["run_id"], "study_id": run["study_id"],
                "reserved_ms": None, "charged_ms": 0, "monotonic_elapsed_ms": None,
                "worker_slot": worker_slot, "settled": False, **binding}
            if adjudication is not None:
                data["restart_adjudication"] = adjudication["reference"]
            self.store._append("RESERVE", data, "reserve-" + reservation_id)
            return data

    def settle(self, reservation_id: str, elapsed_ms: int | None, *, outcome: str):
        with self.store._read_transaction():
            return self._settle(reservation_id, elapsed_ms, outcome=outcome)

    def _settle(self, reservation_id: str, elapsed_ms: int | None, *, outcome: str):
        if elapsed_ms is not None and (type(elapsed_ms) is not int or elapsed_ms < 0):
            raise ResearchError("CONTRACT_MISMATCH", "invalid elapsed cost")
        reservations, _ = self._state()
        if reservation_id not in reservations:
            raise ResearchError("MISSING_INPUT", "reservation missing")
        reservation = reservations[reservation_id]
        if reservation["settled"]:
            if reservation["monotonic_elapsed_ms"] != elapsed_ms or reservation["outcome"] != outcome:
                raise ResearchError("IDENTITY_CONFLICT", "settlement replay changed cost/outcome")
            return reservation
        if elapsed_ms is None and reservation["reserved_ms"] is None:
            self.store._append("ARM_CLOSED", {"arm_id": reservation["arm_id"], "reason": "UNKNOWN_WORKER_COST"},
                               "unknown-resource-cost-" + reservation_id)
            raise ResearchError("RECOVERY_REQUIRED", "unbounded unknown cost retains the worker reservation")
        charged = reservation["reserved_ms"] if elapsed_ms is None else elapsed_ms
        data = {**reservation, "settled": True, "charged_ms": charged,
                "monotonic_elapsed_ms": elapsed_ms, "outcome": outcome}
        self.store._append("SETTLE", data, "settle-" + reservation_id)
        if (elapsed_ms is None or outcome in {"TIMEOUT", "BUDGET_EXHAUSTED", "INTERRUPTED"}
                or (reservation["reserved_ms"] is not None
                    and self._balance(reservation["arm_id"])["remaining_ms"] == 0)):
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
                if result["reserved_ms"] is None:
                    raise ResearchError("RECOVERY_REQUIRED", "unbounded recovery needs independently measured cost and stop evidence")
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
