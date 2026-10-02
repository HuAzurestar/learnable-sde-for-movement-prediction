"""Real stopped-leader trees and blocked-owner deadline counterexamples."""

import sys
import time

from application.research_budget import BudgetSpec
from application.research_supervisor import ResearchSupervisor
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
