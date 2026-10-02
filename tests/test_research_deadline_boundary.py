"""Real stopped-leader trees and blocked-owner deadline counterexamples."""

import sys
import time
import pytest

from application.research_budget import BudgetSpec
from application.research_supervisor import ResearchSupervisor
from application.research_queue import ResearchQueue
from infrastructure.research_store import ResearchError, digest
from tests.test_research_budget import registered


def _worker(child, marker):
    source = ("import json,pathlib,subprocess,sys; "
              "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[3]]); "
              "pathlib.Path(sys.argv[1]).write_text(json.dumps({'fixture':True}))")
    return lambda output: [sys.executable, "-c", source, str(output), child, str(marker)]


def test_stopped_leader_descendants_are_dead_before_owner_validation(tmp_path):
    store, attempts = registered(tmp_path)
    marker = tmp_path / "descendant-wrote-during-validation"
    release = tmp_path / "owner-validation-started"
    child = ("import pathlib,sys,time; marker=pathlib.Path(sys.argv[1]); "
             "release=marker.with_name('owner-validation-started'); "
             "\nwhile not release.exists(): time.sleep(0.01)"
             "\nmarker.touch()")
    def inspect(value):
        release.touch()
        time.sleep(0.5)
        assert not marker.exists(), "descendant survived its successful leader into owner validation"
    result = ResearchSupervisor(store).run(attempts[0], _worker(child, marker), BudgetSpec(4), result_validator=inspect)
    assert result["state"] == "SUCCEEDED" and result["exit_code"] == 0
    assert result["elapsed_ms"] < 4000


def test_owner_validation_overrun_cannot_publish_success(tmp_path):
    store, attempts = registered(tmp_path)
    def late_validation(value):
        time.sleep(2)
        value["validated"] = True
    result = ResearchSupervisor(store).run(attempts[0],
        lambda output: [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{}')", str(output)],
        BudgetSpec(1), result_validator=late_validation)
    assert result["state"] == "TIMEOUT" and result["exit_code"] != 0, result
    assert result["artifact_id"] is None
    assert store.attempts()[attempts[0]]["state"] == "TIMEOUT"
    assert not any(store.manifest(e["payload"]["object_id"]).get("role") == "result"
                   for e in store.events() if e["event_kind"] == "MANIFEST")
    assert ResearchSupervisor(store).budget.balance("affine")["closed"] is True


def test_independent_watchdog_stops_tree_when_owner_polling_blocks(tmp_path, monkeypatch):
    store, attempts = registered(tmp_path)
    marker = tmp_path / "descendant-after-hard-deadline"
    child = "import pathlib,sys,time; time.sleep(2); pathlib.Path(sys.argv[1]).touch()"
    supervisor = ResearchSupervisor(store)
    original = supervisor.budget.balance
    calls = []
    def blocked_balance(arm_id):
        calls.append(True)
        time.sleep(3)
        return original(arm_id)
    monkeypatch.setattr(supervisor.budget, "balance", blocked_balance)
    result = supervisor.run(attempts[0], _worker(child, marker), BudgetSpec(1.5))
    assert calls, "counterexample must block the actual owner poll"
    assert result["state"] == "TIMEOUT" and result["exit_code"] != 0
    assert not marker.exists(), "deadline enforcement disappeared when the wrapper exited"


@pytest.mark.parametrize("observation", ["active", "unavailable"])
def test_unconfirmed_whole_tree_stop_retains_reservation_and_queue_slot(tmp_path, monkeypatch, observation):
    from infrastructure.process_tree import ProcessTree
    store, attempts = registered(tmp_path)
    def unknown(self, timeout=1):
        if observation == "unavailable":
            raise OSError("synthetic stop observation failure")
        return False
    monkeypatch.setattr(ProcessTree, "wait_stopped", unknown)
    supervisor = ResearchSupervisor(store)
    validated = []
    with pytest.raises(ResearchError, match="WORKER_ACTIVE"):
        supervisor.run(attempts[0], lambda output: [sys.executable, "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{}')", str(output)],
            BudgetSpec(4), result_validator=lambda value: validated.append(True))
    assert validated == []
    events = store.events()
    reservation = next(e["payload"] for e in events if e["event_kind"] == "RESERVE")
    assert not any(e["event_kind"] == "SETTLE" for e in events)
    assert store.attempts()[attempts[0]]["error_code"] == "WORKER_STOP_UNCONFIRMED"
    assert supervisor.budget.balance("affine")["committed_ms"] == reservation["reserved_ms"]
    assert supervisor.budget.balance("affine")["closed"]
    value = store.manifest("study-synthetic")["spec"]
    arm = {**value["arms"][0], "arm_id": "other", "model_family_id": "other", "method_family_id": "other"}
    other = {**value, "study_id": "other-study", "arms": [arm],
             "cells": [{**value["cells"][0], "arm_id": "other"}]}
    store.register(other, digest(other))
    attempt = store.new_attempt(store.register_run("other-study", other["cells"][0]))
    next_reservation = supervisor.budget.reserve(attempt, BudgetSpec(1))
    queue = ResearchQueue(store)
    queue.enqueue(next_reservation)
    assert queue.claim(next_reservation["reservation_id"]) is None


def test_numeric_process_group_kill_is_not_reissued_after_stop(monkeypatch):
    from types import SimpleNamespace
    import infrastructure.process_tree as containment
    tree = object.__new__(containment.ProcessTree)
    tree.process = SimpleNamespace(pid=12345)
    tree.job = None
    tree._termination_sent = False
    calls = []
    # Mock the module backend, not the host os.name or an actual unowned PID.
    monkeypatch.setattr(containment, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: calls.append((pid, sig))))
    monkeypatch.setattr(containment, "signal", SimpleNamespace(SIGKILL=9))
    tree.terminate()
    tree.terminate()
    assert calls == [(12345, 9)], "a reaped numeric group ID must not be signaled again"


@pytest.mark.parametrize("assign_succeeds", [True, False])
def test_wrapper_self_containment_uses_native_pseudo_handle_and_fails_closed(monkeypatch, assign_succeeds):
    import ctypes
    from types import SimpleNamespace
    import infrastructure.process_tree as containment
    calls = []
    def assign(job, process):
        calls.append(("assign", job, process.value))
        return int(assign_succeeds)
    def terminate(job, code):
        calls.append(("terminate", job, code))
        return 1
    def close(job):
        calls.append(("close", job))
        return 1
    kernel = SimpleNamespace(CreateJobObjectW=lambda *args: 1234,
        SetInformationJobObject=lambda *args: 1, AssignProcessToJobObject=assign,
        TerminateJobObject=terminate, CloseHandle=close,
        QueryInformationJobObject=lambda *args: 1)
    # Entire backend is synthetic: no real job, handle or foreign process.
    monkeypatch.setattr(containment, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    if assign_succeeds:
        tree = containment.ProcessTree.contain_current_process()
        tree.terminate()
        tree.terminate()
        tree.close()
        assert calls == [("assign", 1234, ctypes.c_void_p(-1).value),
                         ("terminate", 1234, 124), ("close", 1234)]
    else:
        with pytest.raises(OSError, match="cannot contain worker process tree"):
            containment.ProcessTree.contain_current_process()
        assert calls == [("assign", 1234, ctypes.c_void_p(-1).value), ("close", 1234)]


@pytest.mark.parametrize("stop_confirmed", [True, False])
def test_stale_control_pipe_close_cannot_mask_deadline_settlement(tmp_path, monkeypatch, stop_confirmed):
    import errno
    import application.research_supervisor as supervision
    from infrastructure.process_tree import ProcessTree
    store, attempts = registered(tmp_path)
    original_popen = supervision.subprocess.Popen
    class StaleStdin:
        def __init__(self, stream):
            self.stream = stream
        def write(self, value):
            return self.stream.write(value)
        def flush(self):
            return self.stream.flush()
        def close(self):
            try:
                self.stream.close()
            except OSError:
                pass
            raise OSError(errno.EINVAL, "synthetic stale worker control pipe")
    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        process.stdin = StaleStdin(process.stdin)
        return process
    monkeypatch.setattr(supervision.subprocess, "Popen", popen)
    if not stop_confirmed:
        monkeypatch.setattr(ProcessTree, "wait_stopped", lambda self, timeout=1: False)
    supervisor = ResearchSupervisor(store)
    command = lambda output: [sys.executable, "-c", "import time; time.sleep(20)"]
    if stop_confirmed:
        result = supervisor.run(attempts[0], command, BudgetSpec(0.2))
        assert result["state"] == "TIMEOUT" and result["exit_code"] != 0
        settlements = [event for event in store.events() if event["event_kind"] == "SETTLE"]
        assert len(settlements) == 1 and settlements[0]["payload"]["outcome"] == "TIMEOUT"
        assert result["elapsed_ms"] > 0
        assert store.attempts()[attempts[0]]["state"] == "TIMEOUT"
    else:
        with pytest.raises(ResearchError, match="WORKER_ACTIVE"):
            supervisor.run(attempts[0], command, BudgetSpec(0.2))
        assert not any(event["event_kind"] == "SETTLE" for event in store.events())
        assert store.attempts()[attempts[0]]["error_code"] == "WORKER_STOP_UNCONFIRMED"
        assert supervisor.budget.balance("affine")["committed_ms"] == 200
    assert supervisor.budget.balance("affine")["closed"]
