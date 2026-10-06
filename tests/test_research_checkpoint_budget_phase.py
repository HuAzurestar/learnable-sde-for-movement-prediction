"""Actual response/save budget authority is fresh, complete and not rescanned."""

import pytest
import time

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryPlugin, RecoveryRegistry
import application.research_supervisor as supervisor_module
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import ResearchStore, atomic_write, digest, encode
from tests.research_admission_fixtures import synthetic_plugin, admit_fixture
from tests.test_research_checkpoint_phase_work import live_owner, longer_command, longer_resume_command
from tests.test_research_store import spec


def gated_command(output, value, cell):
    command = longer_command(output, value, cell)
    command[2] = command[2].replace("control.save(saved, progress)",
        "while not (Path(output).parent / 'release-response').exists(): time.sleep(0.001)\n"
        "        control.save(saved, progress)")
    return command


def gated_owner(tmp_path):
    store = ResearchStore(tmp_path, "budget-phase", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="budget-phase", capability="generic-rollout",
                             visibility="synthetic", fixture_resume_level="exact")
    plugin = synthetic_plugin("budget-phase", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "exact", gated_command)
    plugin.registry_entry.resource_contract["counts"]["steps"] = {"constant": 600}
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(plugin)
    recovery.register(RecoveryPlugin(plugin.plugin_id, "exact", longer_resume_command, "1.0.0"))
    admit_fixture(store, value, plugin, tmp_path, recovery_command_builder=longer_resume_command)
    store.register(value, digest(value))
    return store, value, SharedRunner(store, registry, recovery_registry=recovery)


def test_ready_response_budget_and_save_use_two_actual_physical_passes(tmp_path, monkeypatch):
    store, value, runner = gated_owner(tmp_path)
    channels, active, reads, acknowledged, released = [], [], [], [], []
    original_init, original_response, original_ack = (
        CheckpointExchange.__init__, CheckpointExchange.response, CheckpointExchange.acknowledge)
    original_balance, original_json = BudgetLedger.balance, store._json
    original_popen = supervisor_module.subprocess.Popen

    def popen(command, *args, **kwargs):
        process = original_popen(command, *args, **kwargs)
        if len(command) > 1 and str(command[1]).endswith("research_worker.py"):
            original_wait = process.wait
            def wait(*args, **kwargs):
                if channels and channels[0].request_id is not None and not released:
                    # Release the actual worker only during owner idle. A real
                    # bounded response is then ready before the next poll.
                    released.append(True)
                    atomic_write(channels[0].directory / "release-response", b"release")
                    while channels[0].response() is None:
                        assert time.monotonic() < channels[0].deadline
                        time.sleep(0.001)
                return original_wait(*args, **kwargs)
            monkeypatch.setattr(process, "wait", wait)
        return process

    def initialize(channel, *args, **kwargs):
        original_init(channel, *args, **kwargs)
        channels.append(channel)

    def response(channel):
        frame = original_response(channel)
        if frame is not None and not active:
            active.append(True)
        return frame

    def balance(ledger, arm):
        # Old owner polls budget before discovering a ready response. Probe the
        # actual bounded frame to start observing that same phase, not fake it.
        if channels and not active:
            channels[0].response()
        return original_balance(ledger, arm)

    def read(path):
        if active and not acknowledged and path.parent == store.path / "events":
            reads.append(path)
        return original_json(path)

    def acknowledge(channel, artifact):
        count = len(reads)
        events = store.events()
        assert released and active and count and store._read_snapshot() is None
        assert count <= 2 * len(events), (
            f"ready-response budget/save read {count} actual event files for {len(events)} events")
        acknowledged.append(artifact)
        return original_ack(channel, artifact)

    monkeypatch.setattr(CheckpointExchange, "__init__", initialize)
    monkeypatch.setattr(CheckpointExchange, "response", response)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    monkeypatch.setattr(BudgetLedger, "balance", balance)
    monkeypatch.setattr(supervisor_module.subprocess, "Popen", popen)
    monkeypatch.setattr(store, "_json", read)
    result = runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))
    assert result["state"] == "FAILED" and "checkpoint" in result, result
    assert acknowledged == [result["checkpoint"]["artifact_id"]]


@pytest.mark.parametrize("changed_at", ["response", "saved-journal"])
@pytest.mark.parametrize("change", ["closed-arm", "recovery-hold"])
def test_actual_checkpoint_ack_refuses_current_arm_close_and_recovery_hold(
        tmp_path, monkeypatch, changed_at, change):
    store, value, runner = live_owner(tmp_path)
    changed, acknowledgements = [], []
    original_response, original_append, original_ack = (
        CheckpointExchange.response, store._append, CheckpointExchange.acknowledge)

    def change_authority():
        changed.append(True)
        if change == "closed-arm":
            store.append("ARM_CLOSED", {"arm_id": "affine", "reason": "synthetic actual close"})
        else:
            atomic_write(store.path / "recovery-hold.json", encode({"reason": "synthetic actual hold"}))

    def response(channel):
        frame = original_response(channel)
        if frame is not None and changed_at == "response" and not changed:
            change_authority()
        return frame

    def append(kind, *args, **kwargs):
        event = original_append(kind, *args, **kwargs)
        if kind == "CHECKPOINT_SAVED" and changed_at == "saved-journal" and not changed:
            change_authority()
        return event

    def acknowledge(channel, artifact):
        acknowledgements.append(artifact)
        return original_ack(channel, artifact)

    monkeypatch.setattr(CheckpointExchange, "response", response)
    monkeypatch.setattr(store, "_append", append)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    result = runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))
    assert changed == [True], "actual selected response/publication boundary was not reached"
    assert not acknowledgements, "owner acknowledged despite current closed arm or recovery hold"
    assert result["state"] == "BUDGET_EXHAUSTED" and "checkpoint" not in result, result
    balance = BudgetLedger(store).balance("affine")
    assert balance["closed"] and balance["committed_ms"] > 0
    events = store.events()
    assert len([event for event in events if event["event_kind"] == "WORKER_TREE_STOPPED"]) == 1
    assert [event["payload"]["outcome"] for event in events if event["event_kind"] == "SETTLE"] == ["BUDGET_EXHAUSTED"]
