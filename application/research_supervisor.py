"""Run one registered attempt inside the shared budget and process boundary."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

from infrastructure.process_tree import ProcessTree
from infrastructure.research_store import ResearchError, ResearchStore
from .research_budget import BudgetLedger, BudgetSpec
from .research_queue import ResearchQueue


class ResearchSupervisor:
    def __init__(self, store: ResearchStore, *, monotonic=time.monotonic):
        self.store = store
        self.budget = BudgetLedger(store)
        self.monotonic = monotonic

    def run(self, attempt_id: str, command_builder, budget: BudgetSpec, *, result_validator=None):
        reservation = self.budget.reserve(attempt_id, budget)
        if reservation["settled"]:
            raise ResearchError("IDENTITY_CONFLICT", "settled attempt cannot execute again")
        run = self.store.manifest("run-" + reservation["run_id"])
        queue = ResearchQueue(self.store)
        try:
            queue.enqueue(reservation, run["cell"].get("resource_class", "cpu"))
            while queue.claim(reservation["reservation_id"]) is None:
                time.sleep(0.05)
        except BaseException as exc:
            # A duplicate caller must not release the original worker's claim.
            if not isinstance(exc, ResearchError) or exc.code != "IDENTITY_CONFLICT":
                self.budget.settle(reservation["reservation_id"], 0, outcome="PREFLIGHT_FAILED")
                self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code="QUEUE_REJECTED")
            raise
        try:
            work = self.store.path / "artifacts" / (".attempt-" + attempt_id)
            work.mkdir(exist_ok=False)
            result_path = work / "result.json"
            heartbeat = work / "heartbeat.json"
            command = list(command_builder(result_path))
            if not command or any(not isinstance(arg, str) for arg in command):
                raise ResearchError("CONTRACT_MISMATCH", "worker command must be a string argument list")
        except BaseException as exc:
            self.budget.settle(reservation["reservation_id"], 0, outcome="PREFLIGHT_FAILED")
            self.store.transition(attempt_id, "PREFLIGHT_FAILED",
                                  error_code=exc.code if isinstance(exc, ResearchError) else "CONTRACT_MISMATCH")
            raise
        start = self.monotonic()
        deadline = start + budget.job_seconds
        wrapper = Path(__file__).resolve().parents[1] / "infrastructure/research_worker.py"
        process, tree, deadline_monitor = None, None, None
        expired = threading.Event()
        monitor_errors = []
        outcome, artifact_id, error_code = "FAILED", None, "WORKER_FAILED"
        control_close_error, control_close_recorded = False, False
        def stop_tree():
            try:
                if tree is not None:
                    tree.terminate()
                    if not tree.wait_stopped():
                        raise ResearchError("WORKER_ACTIVE", "whole process tree stop is unconfirmed")
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ResearchError("WORKER_ACTIVE", "whole process tree stop is unconfirmed") from exc

        def hard_stop():
            # No store lock or journal I/O before the native tree kill. This
            # remains live after the wrapper exits or the owner poll blocks.
            expired.set()
            try:
                tree.terminate()
            except OSError as exc:
                monitor_errors.append(exc)

        def past_deadline():
            return expired.is_set() or self.monotonic() >= deadline
        self.store.transition(attempt_id, "RUNNING")
        try:
            with (work / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [sys.executable, str(wrapper), str(heartbeat), str(deadline), *command],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                )
                tree = ProcessTree(process)
                deadline_monitor = threading.Timer(max(0, deadline - self.monotonic()), hard_stop)
                deadline_monitor.daemon = True
                deadline_monitor.start()
                with self.store.lock():
                    if self.store._attempts()[attempt_id]["state"] != "RUNNING":
                        raise ResearchError("IDENTITY_CONFLICT", "attempt reconciled before worker launch")
                    self.store._append("WORKER_STARTED", {"attempt_id": attempt_id, "pid": process.pid,
                                       "reservation_id": reservation["reservation_id"], "deadline_monotonic": deadline})
                    process.stdin.write(b"GO\n")
                    process.stdin.flush()
                warned = False
                last_logged = start
                while process.poll() is None:
                    now = self.monotonic()
                    if expired.is_set() or now >= deadline:
                        outcome, error_code = "TIMEOUT", "TIMEOUT"
                        break
                    if self.budget.balance(run["arm_id"])["closed"]:
                        outcome, error_code = "BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"
                        break
                    if now - start >= 15 and (not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 15):
                        outcome, error_code = "INTERRUPTED", "HEARTBEAT_LOST"
                        break
                    if now - start >= budget.job_seconds * 0.8 and not warned:
                        self.store.append("CHECKPOINT_REQUESTED", {"attempt_id": attempt_id, "remaining_seconds": max(0, deadline - now)})
                        warned = True
                    if now - last_logged >= 5:
                        self.store.append("HEARTBEAT", {"attempt_id": attempt_id, "monotonic_elapsed_ms": math.ceil((now - start) * 1000)})
                        last_logged = now
                    time.sleep(min(0.025, max(0, deadline - now)))
                completed_code = process.poll()
                # A successful wrapper is not proof its descendants exited.
                # Stop and confirm them before any owner-side result handling.
                stop_tree()
                self.store.append("WORKER_TREE_STOPPED", {"attempt_id": attempt_id,
                    "reservation_id": reservation["reservation_id"],
                    "observed_elapsed_ms": math.ceil((self.monotonic() - start) * 1000),
                    "confirmation": "native-job-or-process-group-no-running-descendants"})
                if completed_code is not None:
                    if past_deadline() or completed_code == 124:
                        outcome, error_code = "TIMEOUT", "TIMEOUT"
                    elif completed_code == 0 and result_path.is_file():
                        # Registered adapters must supply a finite JSON result; no pickle.
                        import json
                        content = result_path.read_bytes()
                        result = json.loads(content)
                        from infrastructure.research_store import encode
                        encode(result)
                        if not isinstance(result, dict):
                            raise ResearchError("CONTRACT_MISMATCH", "worker result must be an object")
                        if result_validator is not None:
                            result_validator(result)
                        # Supervisor-owned admission bindings added by the
                        # validator must be part of the published artifact.
                        content = encode(result)
                        from infrastructure.research_visibility import study_visibility, admission_visibility, combine_visibility
                        spec = self.store.manifest("study-" + run["study_id"])["spec"]
                        visibility = study_visibility(self.store.manifest, spec)
                        if result.get("admission_hash"):
                            receipt = self.store.manifest("admission-" + result["admission_hash"])
                            visibility = combine_visibility([visibility, admission_visibility(self.store.manifest, receipt)])
                        if past_deadline():
                            outcome, error_code = "TIMEOUT", "TIMEOUT"
                        else:
                            artifact = self.store.artifact(content, role="result", visibility=visibility,
                                block_ids=[run["cell"]["block_id"]], study_id=run["study_id"])
                            outcome, artifact_id, error_code = "SUCCEEDED", artifact["artifact_id"], None
        except BaseException as exc:
            # Even if recording is unavailable, containment cleanup always runs.
            if isinstance(exc, Exception) and past_deadline():
                outcome, artifact_id, error_code = "TIMEOUT", None, "TIMEOUT"
            else:
                outcome, error_code = "INTERRUPTED", "SUPERVISOR_ERROR"
                raise
        finally:
            stop_confirmed = True
            if deadline_monitor is not None:
                deadline_monitor.cancel()
                deadline_monitor.join(timeout=1)
                stop_confirmed = not deadline_monitor.is_alive()
            try:
                stop_tree()
            except (OSError, ResearchError, subprocess.TimeoutExpired):
                stop_confirmed = False
            if tree is not None:
                tree.close()
            if process is not None:
                try:
                    process.stdin.close()
                except OSError as exc:
                    # Buffered GO may flush into a pipe already invalidated by
                    # the native deadline kill. This is not proof of live work
                    # or permission to replace the stop/settlement outcome.
                    control_close_error = True
                    try:
                        self.store.append("WORKER_CONTROL_CLOSE_FAILED", {
                            "attempt_id": attempt_id, "reservation_id": reservation["reservation_id"],
                            "error_code": "PIPE_CLOSE_FAILED", "tree_stop_confirmed": stop_confirmed,
                            "errno": exc.errno if type(exc.errno) is int else None})
                    except OSError:
                        # Only this optional diagnostic is best-effort. The
                        # authoritative settlement/transition below must still
                        # propagate corruption or I/O failure, retaining costs.
                        pass
                    else:
                        control_close_recorded = True
            elapsed = math.ceil((self.monotonic() - start) * 1000)
            if outcome == "SUCCEEDED" and (past_deadline() or elapsed > reservation["reserved_ms"]):
                outcome, artifact_id, error_code = "TIMEOUT", None, "TIMEOUT"
            if stop_confirmed:
                self.budget.settle(reservation["reservation_id"], elapsed, outcome=outcome)
            else:
                # Do not release a possibly-live slot or invent measured cost.
                # Recovery still requires actual stop evidence for settlement.
                outcome, artifact_id, error_code = "INTERRUPTED", None, "WORKER_STOP_UNCONFIRMED"
                self.store.append("WORKER_STOP_UNCONFIRMED", {"attempt_id": attempt_id,
                    "reservation_id": reservation["reservation_id"], "observed_elapsed_ms": elapsed,
                    "watchdog_error": bool(monitor_errors)})
                self.store.append("ARM_CLOSED", {"arm_id": run["arm_id"], "reason": error_code})
                elapsed = None
            self.store.transition(attempt_id, outcome, error_code=error_code, artifact_id=artifact_id)
        result = {"attempt_id": attempt_id, "state": outcome, "artifact_id": artifact_id,
                  "exit_code": 0 if outcome == "SUCCEEDED" else 1, "elapsed_ms": elapsed}
        if control_close_error:
            result["cleanup"] = {"control_pipe_close_error": "PIPE_CLOSE_FAILED",
                                 "diagnostic_recorded": control_close_recorded}
        return result
