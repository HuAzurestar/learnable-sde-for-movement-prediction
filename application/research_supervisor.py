"""Run one registered attempt inside the shared budget and process boundary."""

from __future__ import annotations

import math
from datetime import datetime, timezone
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

    def run(self, attempt_id: str, command_builder, budget: BudgetSpec, *, result_validator=None, resource_plan=None, checkpoint_handler=None):
        reservation = self.budget.reserve(attempt_id, budget)
        if reservation["settled"]:
            raise ResearchError("IDENTITY_CONFLICT", "settled attempt cannot execute again")
        run = self.store.manifest("run-" + reservation["run_id"])
        queue = ResearchQueue(self.store)
        try:
            queue.enqueue(reservation, run["cell"].get("resource_class", "cpu"))
            while queue.claim(reservation["reservation_id"]) is None:
                if (reservation.get("work_deadline") and datetime.fromisoformat(reservation["work_deadline"])
                        <= datetime.now(timezone.utc)):
                    raise ResearchError("UNAUTHORIZED_DATA", "approved work window expired while queued")
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
            from .research_registry import GLOBAL_LIMITS
            maximum_result_bytes = GLOBAL_LIMITS["result_bytes"] if resource_plan is None else resource_plan.get("maximum_result_bytes")
            if type(maximum_result_bytes) is not int or not 0 < maximum_result_bytes <= GLOBAL_LIMITS["result_bytes"]:
                raise ResearchError("RESOURCE_PLAN_REJECTED", "worker result byte quota is invalid")
            if not command or any(not isinstance(arg, str) for arg in command):
                raise ResearchError("CONTRACT_MISMATCH", "worker command must be a string argument list")
        except BaseException as exc:
            self.budget.settle(reservation["reservation_id"], 0, outcome="PREFLIGHT_FAILED")
            self.store.transition(attempt_id, "PREFLIGHT_FAILED",
                                  error_code=exc.code if isinstance(exc, ResearchError) else "CONTRACT_MISMATCH")
            raise
        start = self.monotonic()
        allocation_seconds = budget.job_seconds
        if reservation.get("work_deadline"):
            allocation_seconds = min(allocation_seconds, max(0,
                (datetime.fromisoformat(reservation["work_deadline"]) - datetime.now(timezone.utc)).total_seconds()))
        deadline = start + allocation_seconds
        from infrastructure.research_control import CheckpointExchange, ControlError, ENVIRONMENT
        channel = CheckpointExchange(work, attempt_id, deadline, maximum_result_bytes) if checkpoint_handler is not None else None
        saved_checkpoint = None
        budget_stop_requested = False
        wrapper = Path(__file__).resolve().parents[1] / "infrastructure/research_worker.py"
        process, tree, deadline_monitor, soft_monitor = None, None, None, None
        expired = threading.Event()
        monitor_errors = []
        soft_done, soft_cancelled = threading.Event(), threading.Event()
        soft_errors, soft_started = [], None
        warned = False
        outcome, artifact_id, error_code = "FAILED", None, "WORKER_FAILED"
        control_close_error, control_close_recorded = False, False
        tree_stop_recorded = False
        stopped_at, stopped_before_deadline = None, False
        def stop_tree():
            nonlocal stopped_at, stopped_before_deadline
            soft_cancelled.set()
            if soft_monitor is not None:
                soft_monitor.cancel()
            try:
                if tree is not None:
                    tree.terminate()
                    if not tree.wait_stopped():
                        raise ResearchError("WORKER_ACTIVE", "whole process tree stop is unconfirmed")
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=1)
                if stopped_at is None and (tree is not None or process is not None):
                    # Slot occupancy ends at the first actual whole-tree stop,
                    # before journalling, owner validation or control cleanup.
                    # A later watchdog tick cannot undo this native observation.
                    stopped_at = self.monotonic()
                    stopped_before_deadline = not expired.is_set() and stopped_at < deadline
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
        def signal_checkpoint():
            nonlocal soft_started
            # Like the hard fuse, the actual80% signal must not wait behind
            # owner journal/budget I/O. This writes only bounded worker control,
            # never authority or an ACK. The owner still journals the request
            # before handling its response and performs all fresh save checks.
            try:
                # A Timer wakeup is not clock evidence (notably with Windows
                # monotonic tick rounding). Wait cancellably until the actual
                # threshold; never move the original signal before80%.
                threshold = deadline - allocation_seconds * 0.2
                while True:
                    remaining = threshold - self.monotonic()
                    if remaining <= 0:
                        break
                    if soft_cancelled.wait(remaining):
                        return
                if (soft_cancelled.is_set() or past_deadline()
                        or process.poll() is not None):
                    return
                soft_started = self.monotonic()
                channel.request()
            except BaseException as exc:
                soft_errors.append(exc)
            finally:
                soft_done.set()
        def record_checkpoint_request():
            nonlocal warned
            if warned:
                return True
            if channel is not None:
                # request_id alone is not proof of completed frame publication.
                if not soft_done.is_set():
                    return False
                if soft_errors:
                    raise soft_errors[0]
                if soft_started is None:
                    return False
                issued = soft_started
                request_id = channel.request_id
            else:
                issued = self.monotonic()
                if issued - start < allocation_seconds * 0.8:
                    return False
                request_id = None
            self.store.append("CHECKPOINT_REQUESTED", {"attempt_id": attempt_id,
                "remaining_seconds": max(0, deadline - issued), "request_id": request_id,
                "supported": channel is not None})
            warned = True
            return True
        def record_tree_stopped():
            nonlocal tree_stop_recorded
            if not tree_stop_recorded:
                self.store.append("WORKER_TREE_STOPPED", {"attempt_id": attempt_id,
                    "reservation_id": reservation["reservation_id"],
                    "observed_elapsed_ms": math.ceil((stopped_at - start) * 1000),
                    "confirmation": "native-job-or-process-group-no-running-descendants"})
                tree_stop_recorded = True
        def collect_checkpoint():
            nonlocal saved_checkpoint, budget_stop_requested
            if budget_stop_requested or channel is None or saved_checkpoint is not None or past_deadline():
                return False
            if not soft_done.is_set():
                if self.monotonic() < deadline - allocation_seconds * 0.2:
                    return False
                # A published request can already have a response while the
                # sending call is still completing. Wait OUTSIDE the authority
                # lock, bounded by the original deadline and independent fuse.
                # A sender/other writer must remain able to publish a closure.
                soft_done.wait(max(0, deadline - self.monotonic()))
                if not soft_done.is_set() or past_deadline():
                    return False
            if soft_errors:
                raise soft_errors[0]
            if soft_started is None:
                return False
            # REQUEST, fresh budget and ready-response/save share one short
            # initial/final physical scope, never an idle/poll approval. Ordinary
            # REQUEST keeps its immediate durable head. The checkpoint-only
            # MANIFEST/CHECKPOINT/SAVED heads still finish before uncached final
            # checks. Every event fsync remains real; ACK stays outside.
            with self.store._checkpoint_publication():
                record_checkpoint_request()
                def check_current_budget():
                    nonlocal budget_stop_requested
                    budget_stop_requested |= self.budget.balance(run["arm_id"])["closed"]
                check_current_budget()
                # Hold/closure can change during actual publication. Recheck
                # after the final uncached physical pass, before any ACK.
                self.store._read_completion(check_current_budget, lambda: None)
                if budget_stop_requested or past_deadline():
                    return True
                try:
                    frame = channel.response()
                except ControlError as exc:
                    raise ResearchError("CONTRACT_MISMATCH", str(exc)) from exc
                # The real response read itself may publish a current closure
                # or hold. Recheck authority before the funded save handler.
                check_current_budget()
                if frame is None or budget_stop_requested or past_deadline():
                    return True
                progress = frame["progress"]
                if (type(progress) is not dict or set(progress) != {"completed_steps", "total_steps", "throughput_per_second", "eta_seconds"}
                        or type(progress["completed_steps"]) is not int or type(progress["total_steps"]) is not int
                        or not 0 <= progress["completed_steps"] <= progress["total_steps"]
                        or progress["total_steps"] <= 0
                        or any(type(progress[key]) not in {int, float} or not math.isfinite(progress[key]) or progress[key] < 0
                            for key in ("throughput_per_second", "eta_seconds"))):
                    raise ResearchError("CONTRACT_MISMATCH", "worker checkpoint progress is invalid")
                reference = checkpoint_handler(frame["state"], progress, deadline)
                if past_deadline():
                    return True
                if type(reference) is not dict or set(reference) != {"artifact_id", "resume_level"}:
                    raise ResearchError("CONTRACT_MISMATCH", "owner checkpoint receipt is invalid")
                metadata = self.store.manifest("artifact-" + reference["artifact_id"])
                if metadata["role"] != "checkpoint" or metadata["study_id"] != run["study_id"]:
                    raise ResearchError("CONTRACT_MISMATCH", "owner checkpoint artifact scope differs")
                if past_deadline():
                    return True
                candidate = {**reference, "progress": progress, "elapsed_ms": math.ceil((self.monotonic() - start) * 1000)}
                self.store.append("CHECKPOINT_SAVED", {"attempt_id": attempt_id, "request_id": channel.request_id, **candidate})
            if budget_stop_requested or past_deadline():
                return True
            channel.acknowledge(reference["artifact_id"])
            if not past_deadline():
                saved_checkpoint = candidate
            return True
        self.store.transition(attempt_id, "RUNNING")
        try:
            with (work / "worker.log").open("xb") as log:
                worker_environment = dict(os.environ)
                # An unsupported/new attempt must never inherit a parent or
                # previous session's checkpoint directory/token/identity.
                worker_environment.pop(ENVIRONMENT, None)
                worker_environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                if channel is not None:
                    worker_environment.update(channel.environment())
                process = subprocess.Popen(
                    [sys.executable, str(wrapper), str(heartbeat), str(deadline), *command],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    env=worker_environment,
                )
                tree = ProcessTree(process)
                deadline_monitor = threading.Timer(max(0, deadline - self.monotonic()), hard_stop)
                deadline_monitor.daemon = True
                deadline_monitor.start()
                if channel is not None:
                    soft_monitor = threading.Timer(
                        max(0, start + allocation_seconds * 0.8 - self.monotonic()), signal_checkpoint)
                    soft_monitor.daemon = True
                    soft_monitor.start()
                with self.store.lock():
                    if self.store._attempts()[attempt_id]["state"] != "RUNNING":
                        raise ResearchError("IDENTITY_CONFLICT", "attempt reconciled before worker launch")
                    self.store._append("WORKER_STARTED", {"attempt_id": attempt_id, "pid": process.pid,
                                       "reservation_id": reservation["reservation_id"], "deadline_monotonic": deadline})
                    process.stdin.write(b"GO\n")
                    process.stdin.flush()
                last_logged = start
                while process.poll() is None:
                    now = self.monotonic()
                    if expired.is_set() or now >= deadline:
                        outcome, error_code = "TIMEOUT", "TIMEOUT"
                        break
                    if saved_checkpoint is not None:
                        # ACK follows complete physical/save/budget validation.
                        # The worker is now closing, not admitted to another job.
                        # Observe its real exit before further budget I/O; retain
                        # the reservation, independent fuse and heartbeat veto.
                        # Fresh terminal budget authority is checked after stop.
                        if now - start >= 15 and (not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 15):
                            outcome, error_code = "INTERRUPTED", "HEARTBEAT_LOST"
                            break
                        try:
                            process.wait(timeout=min(0.025, deadline - now))
                        except subprocess.TimeoutExpired:
                            pass
                        continue
                    # A ready response checks current budget in its short fresh
                    # save scope. With no response, retain the ordinary fresh
                    # poll; no budget decision is borrowed by another iteration.
                    budget_checked = collect_checkpoint()
                    if past_deadline():
                        outcome, error_code = "TIMEOUT", "TIMEOUT"
                        break
                    current_budget_closed = budget_stop_requested or (
                        not budget_checked and self.budget.balance(run["arm_id"])["closed"])
                    if current_budget_closed:
                        # A completed signal remains trace evidence even when
                        # the fresh blocked poll closes authority. No save/ACK.
                        record_checkpoint_request()
                        outcome, error_code = "BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"
                        break
                    # Authority I/O may cross the soft or hard threshold. Do
                    # not defer the80% request using its pre-read timestamp.
                    now = self.monotonic()
                    if expired.is_set() or now >= deadline:
                        outcome, error_code = "TIMEOUT", "TIMEOUT"
                        break
                    if now - start >= 15 and (not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 15):
                        outcome, error_code = "INTERRUPTED", "HEARTBEAT_LOST"
                        break
                    if channel is None and now - start >= allocation_seconds * 0.8 and not warned:
                        record_checkpoint_request()
                    if not budget_checked:
                        collect_checkpoint()
                    if budget_stop_requested:
                        outcome, error_code = "BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"
                        break
                    now = self.monotonic()
                    if now - last_logged >= 5:
                        self.store.append("HEARTBEAT", {"attempt_id": attempt_id, "monotonic_elapsed_ms": math.ceil((now - start) * 1000)})
                        last_logged = now
                    remaining = deadline - self.monotonic()
                    if expired.is_set() or remaining <= 0:
                        outcome, error_code = "TIMEOUT", "TIMEOUT"
                        break
                    # Keep the same poll cadence, but wake on actual wrapper
                    # exit rather than idling while its checkpoint margin dies.
                    try:
                        process.wait(timeout=min(0.025, remaining))
                    except subprocess.TimeoutExpired:
                        pass
                completed_code = process.poll()
                collect_checkpoint()
                if budget_stop_requested:
                    outcome, error_code = "BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"
                # A successful wrapper is not proof its descendants exited.
                # Stop and confirm them before any owner-side result handling.
                stop_tree()
                record_tree_stopped()
                if saved_checkpoint is not None:
                    # A fast exit must not skip a closure/hold occurring after
                    # ACK. This fresh read also verifies the actual final chain;
                    # no poll approval or pre-ACK snapshot is reused here.
                    budget_stop_requested |= self.budget.balance(run["arm_id"])["closed"]
                if completed_code is not None:
                    checkpoint_stopped = (completed_code == 85 and saved_checkpoint is not None
                                          and stopped_before_deadline)
                    if (past_deadline() and not checkpoint_stopped) or completed_code == 124:
                        outcome, error_code = "TIMEOUT", "TIMEOUT"
                    elif budget_stop_requested:
                        outcome, error_code = "BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"
                    elif checkpoint_stopped:
                        outcome, error_code = "FAILED", "CHECKPOINT_SAVED"
                    elif completed_code == 0 and result_path.is_file():
                        # Registered adapters must supply a finite JSON result; no pickle.
                        import json
                        with result_path.open("rb") as stream:
                            content = stream.read(maximum_result_bytes + 1)
                        if len(content) > maximum_result_bytes:
                            raise ResearchError("RESOURCE_PLAN_REJECTED", "worker result exceeds admitted byte quota")
                        result = json.loads(content)
                        from infrastructure.research_store import encode
                        encode(result)
                        if not isinstance(result, dict):
                            raise ResearchError("CONTRACT_MISMATCH", "worker result must be an object")
                        # The tree is already confirmed stopped. Reuse only
                        # this short owner phase's verified prefix, then verify
                        # the actual final journal before independent settlement.
                        with self.store._read_transaction():
                            if result_validator is not None:
                                result_validator(result)
                            # Supervisor-owned admission bindings added by the
                            # validator must be part of the published artifact.
                            content = encode(result)
                            if len(content) > maximum_result_bytes:
                                raise ResearchError("RESOURCE_PLAN_REJECTED", "final result including owner provenance exceeds admitted byte quota")
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
                        if past_deadline():
                            outcome, artifact_id, error_code = "TIMEOUT", None, "TIMEOUT"
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
            if soft_monitor is not None:
                # Native containment comes first; do not wait for a blocked
                # control writer before killing/stopping the actual worker.
                soft_monitor.join(timeout=1)
                if soft_monitor.is_alive():
                    outcome, artifact_id, error_code = "INTERRUPTED", None, "CHECKPOINT_SIGNAL_UNCONFIRMED"
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
            # Control release or owner I/O can leave the body before its normal
            # stop journal. Native cleanup still proves the launched tree is
            # stopped; preserve that evidence before authoritative settlement,
            # including when settlement subsequently fails and retains cost.
            if stop_confirmed and tree is not None:
                record_tree_stopped()
            # Unknown stop retains the reservation and current diagnostic wall
            # time. Confirmed computation is never billed for subsequent owner
            # work; result validation still has its original deadline vetoes.
            end = stopped_at if stop_confirmed and stopped_at is not None else self.monotonic()
            elapsed = math.ceil((end - start) * 1000)
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
        if saved_checkpoint is not None:
            result["checkpoint"] = saved_checkpoint
        if control_close_error:
            result["cleanup"] = {"control_pipe_close_error": "PIPE_CLOSE_FAILED",
                                 "diagnostic_recorded": control_close_recorded}
        return result
