"""Actual blocking owner reads cannot leave checkpoint/idle clocks stale.

Use the separately registered six-second workload for observation margin;
original three/six-second continuation acceptance remains unchanged.
"""

import time
from types import SimpleNamespace

from application.research_budget import BudgetLedger, BudgetSpec
import application.research_supervisor as supervisor_module
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import digest
from tests.test_research_checkpoint_phase_work import live_owner


def test_budget_read_crossing_80_percent_requests_in_that_same_owner_poll(tmp_path, monkeypatch):
    store, value, runner = live_owner(tmp_path)
    channels, crossed, requests = [], [], []
    original_init, original_request, original_balance = (
        CheckpointExchange.__init__, CheckpointExchange.request, BudgetLedger.balance)

    def initialize(channel, *args, **kwargs):
        original_init(channel, *args, **kwargs)
        channels.append(channel)

    def request(channel):
        assert time.monotonic() >= channel.deadline - 1.2, "request moved before the original80% threshold"
        result = original_request(channel)
        requests.append(result)
        return result

    def balance(ledger, arm):
        if crossed:
            assert requests, "owner began another budget poll after crossing80% without issuing the request"
        else:
            assert channels
            threshold = channels[0].deadline - 1.2
            assert time.monotonic() < threshold, "observation window was missed"
            time.sleep(max(0, threshold - time.monotonic()) + 0.01)
            crossed.append(True)
        return original_balance(ledger, arm)

    monkeypatch.setattr(CheckpointExchange, "__init__", initialize)
    monkeypatch.setattr(CheckpointExchange, "request", request)
    monkeypatch.setattr(BudgetLedger, "balance", balance)
    result = runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))
    assert crossed and requests
    assert result["state"] == "FAILED" and "checkpoint" in result, result


def test_owner_idle_wait_responds_to_actual_acknowledged_worker_exit(tmp_path, monkeypatch):
    store, value, runner = live_owner(tmp_path)
    workers, acknowledged, idle_after_ack = [], [], []
    original_popen = supervisor_module.subprocess.Popen
    original_ack = CheckpointExchange.acknowledge

    def popen(command, *args, **kwargs):
        process = original_popen(command, *args, **kwargs)
        if len(command) > 1 and str(command[1]).endswith("research_worker.py"):
            workers.append(process)
        return process

    def acknowledge(channel, artifact):
        result = original_ack(channel, artifact)
        acknowledged.append(channel.deadline)
        return result

    def sleep(seconds):
        if acknowledged:
            # Observe actual exit85, not a fake poll/clock. The old idle sleep
            # cannot wake on it; the fixed actual Popen.wait never reaches here.
            code = workers[0].wait(timeout=max(0, acknowledged[0] - time.monotonic()))
            assert code == 85
            idle_after_ack.append(code)
        else:
            time.sleep(seconds)

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", popen)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    monkeypatch.setattr(supervisor_module, "time", SimpleNamespace(time=time.time, sleep=sleep))
    result = runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))
    assert acknowledged and result["state"] == "FAILED" and "checkpoint" in result, result
    assert not idle_after_ack, "owner idled instead of waiting responsively for real exit85"


def test_owner_never_enters_idle_after_actual_budget_read_crosses_hard_deadline(tmp_path, monkeypatch):
    store, value, runner = live_owner(tmp_path)
    channels, crossed, late_idle = [], [], []
    original_init, original_balance = CheckpointExchange.__init__, BudgetLedger.balance

    def initialize(channel, *args, **kwargs):
        original_init(channel, *args, **kwargs)
        channels.append(channel)

    def balance(ledger, arm):
        assert channels and not crossed, "actual first owner budget read was not selected"
        time.sleep(max(0, channels[0].deadline - time.monotonic()) + 0.05)
        crossed.append(True)
        return original_balance(ledger, arm)

    def sleep(seconds):
        if crossed and time.monotonic() >= channels[0].deadline:
            late_idle.append(seconds)
        time.sleep(seconds)

    monkeypatch.setattr(CheckpointExchange, "__init__", initialize)
    monkeypatch.setattr(BudgetLedger, "balance", balance)
    monkeypatch.setattr(supervisor_module, "time", SimpleNamespace(time=time.time, sleep=sleep))
    result = runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))
    assert crossed and result["state"] == "TIMEOUT", result
    assert result["elapsed_ms"] >= 6000 and "checkpoint" not in result
    assert not late_idle, "owner used a pre-read clock to idle after the actual hard deadline"
