"""Run one registered attempt inside the shared budget and process boundary."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import time

from infrastructure.process_tree import ProcessTree
from infrastructure.research_store import ResearchError, ResearchStore
from .research_budget import BudgetLedger, BudgetSpec


class ResearchSupervisor:
    def __init__(self, store: ResearchStore, *, monotonic=time.monotonic):
        self.store = store
        self.budget = BudgetLedger(store)
        self.monotonic = monotonic

    def run(self, attempt_id: str, command_builder, budget: BudgetSpec):
        reservation = self.budget.reserve(attempt_id, budget)
        if reservation["settled"]:
            raise ResearchError("IDENTITY_CONFLICT", "settled attempt cannot execute again")
        run = self.store.manifest("run-" + reservation["run_id"])
        work = self.store.path / "artifacts" / (".attempt-" + attempt_id)
        work.mkdir(exist_ok=True)
        result_path = work / "result.json"
        if result_path.exists():
            raise ResearchError("IDENTITY_CONFLICT", "attempt output already exists; use explicit recovery")
        heartbeat = work / "heartbeat.json"
        try:
            command = list(command_builder(result_path))
            if not command or any(not isinstance(arg, str) for arg in command):
                raise ResearchError("CONTRACT_MISMATCH", "worker command must be a string argument list")
        except Exception:
            self.budget.settle(reservation["reservation_id"], 0, outcome="PREFLIGHT_FAILED")
            self.store.transition(attempt_id, "PREFLIGHT_FAILED", error_code="CONTRACT_MISMATCH")
            raise
        start = self.monotonic()
        deadline = start + budget.job_seconds
        wrapper = Path(__file__).resolve().parents[1] / "infrastructure/research_worker.py"
        process, tree = None, None
        outcome, artifact_id, error_code = "FAILED", None, "WORKER_FAILED"
        self.store.transition(attempt_id, "RUNNING")
        try:
            with (work / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [sys.executable, str(wrapper), str(heartbeat), str(deadline), *command],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                tree = ProcessTree(process)
                self.store.append("WORKER_STARTED", {"attempt_id": attempt_id, "pid": process.pid,
                                  "reservation_id": reservation["reservation_id"], "deadline_monotonic": deadline})
                process.stdin.write(b"GO\n")
                process.stdin.flush()
                warned = False
                last_logged = start
                while process.poll() is None:
                    now = self.monotonic()
                    if now >= deadline:
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
                if process.poll() is not None:
                    if self.monotonic() >= deadline or process.returncode == 124:
                        outcome, error_code = "TIMEOUT", "TIMEOUT"
                    elif process.returncode == 0 and result_path.is_file():
                        # Registered adapters must supply a finite JSON result; no pickle.
                        import json
                        content = result_path.read_bytes()
                        result = json.loads(content)
                        from infrastructure.research_store import encode
                        encode(result)
                        if not isinstance(result, dict):
                            raise ResearchError("CONTRACT_MISMATCH", "worker result must be an object")
                        artifact = self.store.artifact(content, role="result", visibility=run["cell"].get("visibility", "restricted"),
                            block_ids=[run["cell"]["block_id"]], study_id=run["study_id"])
                        outcome, artifact_id, error_code = "SUCCEEDED", artifact["artifact_id"], None
        except BaseException:
            # Even if recording is unavailable, containment cleanup always runs.
            outcome, error_code = "INTERRUPTED", "SUPERVISOR_ERROR"
            raise
        finally:
            if tree is not None:
                tree.terminate()
                tree.close()
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdin.close()
            elapsed = math.ceil((self.monotonic() - start) * 1000)
            self.budget.settle(reservation["reservation_id"], elapsed, outcome=outcome)
            self.store.transition(attempt_id, outcome, error_code=error_code, artifact_id=artifact_id)
        return {"attempt_id": attempt_id, "state": outcome, "artifact_id": artifact_id,
                "exit_code": 0 if outcome == "SUCCEEDED" else 1, "elapsed_ms": elapsed}
