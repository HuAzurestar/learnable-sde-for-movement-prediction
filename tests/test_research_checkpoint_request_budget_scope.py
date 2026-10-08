"""Real missing-response REQUEST shares fresh budget checks before native idle.

Use the already registered600-step/6s gated worker. No clock, authority result,
native wait/status or deadline is replaced; actual responses are released only
at the original native idle boundary. Original300-step acceptance is unchanged.
"""
import threading
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
import application.research_supervisor as supervisor
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import atomic_write, digest, encode
from tests.test_research_checkpoint_budget_phase import gated_owner


@pytest.mark.parametrize("change", [None, "closed-arm", "recovery-hold"])
def test_missing_response_request_budget_is_fresh_in_the_same_read_phase(
        tmp_path, monkeypatch, change):
    store, value, runner = gated_owner(tmp_path)
    channels, requests, balances, released, acknowledgements, workers = [], [], [], [], [], []
    owner = threading.get_ident()
    original_init, original_ack = CheckpointExchange.__init__, CheckpointExchange.acknowledge
    original_append, original_balance = store.append, BudgetLedger.balance
    original_popen = supervisor.subprocess.Popen

    def initialize(channel, *args, **kwargs):
        original_init(channel, *args, **kwargs)
        channels.append(channel)

    def append(kind, *args, **kwargs):
        result = original_append(kind, *args, **kwargs)
        if kind == "CHECKPOINT_REQUESTED" and not requests:
            requests.append(store._read_snapshot())
            assert not hasattr(store._read_scope, "checkpoint_head"), (
                "missing response entered checkpoint publication")
            if change == "closed-arm":
                original_append("ARM_CLOSED", {"arm_id": "affine", "reason": "actual request close"})
            elif change == "recovery-hold":
                atomic_write(store.path / "recovery-hold.json", encode({"reason": "actual request hold"}))
        return result

    def balance(ledger, *args, **kwargs):
        if ledger.store is store and requests and not released:
            balances.append(store._read_snapshot())
        return original_balance(ledger, *args, **kwargs)

    def acknowledge(channel, artifact_id):
        assert store._read_snapshot() is None
        result = original_ack(channel, artifact_id)
        acknowledgements.append((time.monotonic(), channel.deadline, artifact_id))
        return result

    def popen(command, *args, **kwargs):
        process = original_popen(command, *args, **kwargs)
        if len(command) > 1 and str(command[1]).endswith("research_worker.py"):
            workers.append(process)
            original_wait = process.wait

            def wait(*wait_args, **wait_kwargs):
                timeout = wait_kwargs.get("timeout")
                if (change is None and threading.get_ident() == owner
                        and timeout is not None and 0 < timeout <= .025
                        and requests and not released):
                    released.append(True)
                    atomic_write(channels[0].directory / "release-response", b"release")
                    while channels[0].response() is None:
                        assert time.monotonic() < channels[0].deadline
                        time.sleep(.001)
                return original_wait(*wait_args, **wait_kwargs)

            monkeypatch.setattr(process, "wait", wait)
        return process

    monkeypatch.setattr(CheckpointExchange, "__init__", initialize)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    monkeypatch.setattr(store, "append", append)
    monkeypatch.setattr(BudgetLedger, "balance", balance)
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    result = runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))

    # Assertions follow actual stop/settlement. Every observed budget call is
    # real; None exposes the old extra independent physical read before idle.
    assert len(requests) == 1 and requests[0] is not None
    if change is None:
        assert balances and all(snapshot is not None for snapshot in balances), (
            "missing-response REQUEST repeated an unscoped budget read before actual native idle")
        assert len(balances) >= 2 and balances[0] is requests[0], (
            "first REQUEST did not verify current budget in its owned read phase")
        assert balances[-1] is not requests[0], "final physical budget check reused the initial prefix"
        assert released == [True] and len(acknowledgements) == 1
        assert acknowledgements[0][0] < acknowledgements[0][1]
        assert workers[0].poll() == 85
        assert result["state"] == "FAILED" and "checkpoint" in result, result
    else:
        assert result["state"] == "BUDGET_EXHAUSTED" and "checkpoint" not in result, result
        assert not acknowledgements and not released
        assert not (channels[0].directory / "checkpoint-ack.json").exists()
    events = store.events()
    assert len([event for event in events if event["event_kind"] == "CHECKPOINT_REQUESTED"]) == 1
    assert len([event for event in events if event["event_kind"] == "WORKER_TREE_STOPPED"]) == 1
    settlements = [event["payload"] for event in events if event["event_kind"] == "SETTLE"]
    assert len(settlements) == 1 and settlements[0]["charged_ms"] > 0
    assert settlements[0]["monotonic_elapsed_ms"] == result["elapsed_ms"]
    assert store._read_snapshot() is None
