"""Actual live save completion precedes ACK and bounds owner authority work.

The separate 600-step diagnostic gives the counterexample a stable observation
window. It does not change the original 300-step, 3s/6s acceptance fixtures.
"""

import pytest

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryPlugin, RecoveryRegistry
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.research_admission_fixtures import synthetic_plugin, admit_fixture
from tests.test_research_live_checkpoint import command, resume_command
from tests.test_research_store import spec


def longer_command(output, value, cell):
    result = command(output, value, cell)
    result[2] = result[2].replace("300", "600")
    return result


def longer_resume_command(output, value, cell, state):
    result = resume_command(output, value, cell, state)
    result[2] = result[2].replace("300", "600")
    return result


def live_owner(tmp_path):
    store = ResearchStore(tmp_path, "checkpoint-phase", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="checkpoint-phase", capability="generic-rollout",
                             visibility="synthetic", fixture_resume_level="exact")
    plugin = synthetic_plugin("checkpoint-phase", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "exact", longer_command)
    plugin.registry_entry.resource_contract["counts"]["steps"] = {"constant": 600}
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(plugin)
    recovery.register(RecoveryPlugin(plugin.plugin_id, "exact", longer_resume_command, "1.0.0"))
    admit_fixture(store, value, plugin, tmp_path, recovery_command_builder=longer_resume_command)
    store.register(value, digest(value))
    return store, value, SharedRunner(store, registry, recovery_registry=recovery)


def test_actual_controller_save_is_physically_verified_before_ack_without_repeat_scans(tmp_path, monkeypatch):
    store, value, runner = live_owner(tmp_path)
    original_json, original_response, original_ack = store._json, CheckpointExchange.response, CheckpointExchange.acknowledge
    active, reads, acknowledgements = [], [], []
    def response(exchange):
        frame = original_response(exchange)
        if frame is not None and not active:
            active.append(True)
        return frame
    def read(path):
        if active and not acknowledgements and path.parent == store.path / "events":
            reads.append(path)
        return original_json(path)
    def acknowledge(exchange, artifact_id):
        assert active and reads, "actual response/save must reach real event reads"
        assert store._read_snapshot() is None, "ACK preceded final physical scope verification"
        count = len(reads)
        events = store.events()
        assert count <= 2 * len(events), (
            f"controller save reread {count} actual event files for {len(events)} events")
        acknowledgements.append(artifact_id)
        return original_ack(exchange, artifact_id)
    monkeypatch.setattr(CheckpointExchange, "response", response)
    monkeypatch.setattr(store, "_json", read)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    result = runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))
    assert result["state"] == "FAILED" and "checkpoint" in result, result
    assert acknowledgements == [result["checkpoint"]["artifact_id"]]
    assert store.attempts()[result["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED"


def test_actual_controller_never_acknowledges_corruption_after_saved_journal(tmp_path, monkeypatch):
    store, value, runner = live_owner(tmp_path)
    original_append, original_ack = store._append, CheckpointExchange.acknowledge
    corrupted, acknowledgements = [], []
    def append(kind, *args, **kwargs):
        event = original_append(kind, *args, **kwargs)
        if kind == "CHECKPOINT_SAVED":
            corrupted.append(True)
            (store.path / "events/0000000000000001.json").write_bytes(b"{}")
        return event
    def acknowledge(exchange, artifact_id):
        acknowledgements.append(artifact_id)
        return original_ack(exchange, artifact_id)
    monkeypatch.setattr(store, "_append", append)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(6))
    assert corrupted == [True], "actual final saved journal was not reached"
    assert acknowledgements == [], "worker received ACK before final integrity verification"
    assert store._read_snapshot() is None
