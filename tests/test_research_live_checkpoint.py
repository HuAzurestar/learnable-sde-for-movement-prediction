"""Real shared-runner checkpoint delivery and continued RNG computation.

The bounded accumulator is an engineering fixture, not an SDE method or
scientific qualification. It must consume the same actual random stream after
reopening the store; a serialized-state roundtrip cannot satisfy these tests.
"""

import inspect
import json
from pathlib import Path
import sys
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryPlugin, RecoveryRegistry, SharedRecovery
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchStore, digest, encode, atomic_write
from tests.research_admission_fixtures import synthetic_plugin, admit_fixture
from tests.test_research_store import spec


PROGRAM = r'''
import importlib.util, json, math, os, random, sys, time
from pathlib import Path
output, value, cell, library, restored = sys.argv[1:]
value, cell = json.loads(value), json.loads(cell)
def tuples(value):
    return tuple(tuples(item) for item in value) if isinstance(value, list) else value
state = json.loads(Path(restored).read_text()) if restored else None
rng = random.Random(cell['seed'])
position = 0
accumulator = [0., 0., 0., 0.]
if state:
    rng.setstate(tuples(state['rng_state']['python']))
    position = state['data_position']
    accumulator = state['method_state']['accumulator']
control = None
if os.environ.get('PIRC25_WORKER_CONTROL'):
    module_spec = importlib.util.spec_from_file_location('research_control', library)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    control = module.WorkerControl.from_environment()
start = time.monotonic()
for step in range(position, 300):
    for index in range(4):
        accumulator[index] += math.sqrt(0.01) * rng.gauss(0., 1.)
    time.sleep(0.01)
    if control is not None and control.poll() is not None:
        saved = {'step': step + 1, 'data_position': step + 1,
            'method_state': {'accumulator': accumulator},
            'rng_state': {'python': rng.getstate(), 'stream_id': str(cell['seed'])},
            'chunk_complete': True}
        elapsed = time.monotonic() - start
        progress = {'completed_steps': step + 1, 'total_steps': 300,
            'throughput_per_second': (step + 1 - position) / elapsed,
            'eta_seconds': (300 - step - 1) * elapsed / max(1, step + 1 - position)}
        control.save(saved, progress)
        raise SystemExit(85)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
def digest(value):
    import hashlib
    return hashlib.sha256(canonical(value)).hexdigest()
payload = {'metrics': {'error': abs(accumulator[0])},
    'forecast': {'accumulator': accumulator, 'steps': 300, 'rng_state': rng.getstate()},
    'fit': {}, 'source_schema': 'pirc25-rng-continuation-fixture-v1'}
result = {'schema_version': 'pirc25-result-v1', 'status': 'SUCCEEDED',
    'spec_hash': digest(value), 'cell_hash': digest(cell), 'protocol_hash': value['protocol_hash'],
    'input_hash': value['data_hash'], 'output_hash': digest(payload),
    'state_order': ['x','y','vx','vy'], 'units': ['m','m','m/s','m/s'],
    'resume_level': cell['fixture_resume_level'], 'qualification': 'fixture',
    'metric_units': {'error': 'm'}, **payload}
Path(output).write_bytes(canonical(result))
'''


def command(output, value, cell):
    library = Path(__file__).resolve().parents[1] / "infrastructure" / "research_control.py"
    return [sys.executable, "-c", PROGRAM, str(output), encode(value).decode(), encode(cell).decode(), str(library), ""]


def resume_command(output, value, cell, state):
    path = Path(output).with_name("resume-state.json")
    atomic_write(path, encode(state))
    result = command(output, value, cell)
    result[-1] = str(path)
    return result


def prepared(root, level, study):
    root.mkdir()
    store = ResearchStore(root, "live-checkpoint", initialize=True)
    value = spec()
    value["study_id"] = study
    value["cells"][0].update(plugin_id="rng-fixture", capability="generic-rollout", visibility="synthetic",
        fixture_resume_level=level)
    plugin = synthetic_plugin("rng-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), level, command)
    plugin.registry_entry.resource_contract["counts"]["steps"] = {"constant": 300}
    registry = CapabilityRegistry()
    registry.register(plugin)
    adapters = RecoveryRegistry()
    adapters.register(RecoveryPlugin(plugin.plugin_id, level, resume_command, "1.0.0"))
    grant = admit_fixture(store, value, plugin, root, recovery_command_builder=resume_command)
    store.register(value, digest(value))
    return store, value, registry, adapters, grant


@pytest.mark.parametrize("level", ["exact", "chunk"])
def test_actual_worker_saves_before_deadline_and_reopened_resume_matches_continuous(tmp_path, level):
    continuous, baseline, baseline_registry, _, _ = prepared(tmp_path / "continuous", level, "continuous")
    baseline_cell = baseline["cells"][0]
    expected = SharedRunner(continuous, baseline_registry).run_cell("continuous", digest(baseline_cell), budget=BudgetSpec(6))
    assert expected["state"] == "SUCCEEDED"
    expected_output = json.loads((continuous.path / "artifacts" / expected["artifact_id"]).read_bytes())
    store, value, registry, adapters, grant = prepared(tmp_path / "resumed", level, "resumed")
    # Baseline compatibility is test-only: exercise the actual existing worker
    # even while the owner recovery hook is absent, not just a missing keyword.
    arguments = {"recovery_registry": adapters} if "recovery_registry" in inspect.signature(SharedRunner).parameters else {}
    runner = SharedRunner(store, registry, **arguments)
    interrupted = runner.run_cell("resumed", digest(value["cells"][0]), budget=BudgetSpec(3))
    assert interrupted["state"] == "FAILED", "a supported checkpoint did not stop before the hard fuse"
    attempt = store.attempts()[interrupted["attempt_id"]]
    assert attempt["error_code"] == "CHECKPOINT_SAVED"
    events = list(store.events())
    saved = [event["payload"] for event in events if event["event_kind"] == "CHECKPOINT_SAVED"]
    assert len(saved) == 1
    checkpoint = saved[0]
    assert 2400 <= checkpoint["elapsed_ms"] < 3000
    assert 0 < checkpoint["progress"]["completed_steps"] < 300
    assert checkpoint["progress"]["total_steps"] == 300
    assert checkpoint["progress"]["throughput_per_second"] > 0 and checkpoint["progress"]["eta_seconds"] > 0
    previous_cost = BudgetLedger(store).balance("affine")["committed_ms"]
    assert previous_cost > 0 and not BudgetLedger(store).balance("affine")["closed"]
    reopened = ResearchStore(tmp_path / "resumed", "live-checkpoint")
    resumed = SharedRecovery(reopened, registry, adapters).resume(interrupted["attempt_id"], checkpoint["artifact_id"],
        authorization=grant, budget=BudgetSpec(6))
    assert resumed["state"] == "SUCCEEDED"
    actual = json.loads((reopened.path / "artifacts" / resumed["artifact_id"]).read_bytes())
    assert actual["forecast"] == expected_output["forecast"]
    assert actual["metrics"] == expected_output["metrics"]
    assert BudgetLedger(reopened).balance("affine")["committed_ms"] > previous_cost
    assert reopened.attempts()[resumed["attempt_id"]]["parent_attempt_id"] == interrupted["attempt_id"]


@pytest.mark.parametrize("level", ["exact", "chunk"])
@pytest.mark.parametrize("delay_at", ["reference-read", "saved-event"])
def test_delayed_owner_cannot_confirm_checkpoint_after_hard_deadline(tmp_path, monkeypatch, level, delay_at):
    from infrastructure import research_control as control
    store, value, registry, adapters, _ = prepared(tmp_path / "resumed", level, "resumed")
    delayed = []
    original_manifest, original_append = store.manifest, store.append
    def manifest(object_id):
        result = original_manifest(object_id)
        if delay_at == "reference-read" and result.get("role") == "checkpoint" and not delayed:
            delayed.append(True)
            time.sleep(0.8)
        return result
    def append(kind, *args, **kwargs):
        if delay_at == "saved-event" and kind == "CHECKPOINT_SAVED" and not delayed:
            delayed.append(True)
            time.sleep(0.8)
        return original_append(kind, *args, **kwargs)
    monkeypatch.setattr(store, "manifest", manifest)
    monkeypatch.setattr(store, "append", append)
    original_ack = control.CheckpointExchange.acknowledge
    acknowledgements = []
    def acknowledge(exchange, artifact_id):
        acknowledgements.append((time.monotonic(), exchange.deadline))
        return original_ack(exchange, artifact_id)
    monkeypatch.setattr(control.CheckpointExchange, "acknowledge", acknowledge)
    result = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
        "resumed", digest(value["cells"][0]), budget=BudgetSpec(3))
    assert delayed, "actual owner checkpoint handling was not reached"
    assert result["state"] == "TIMEOUT" and result["exit_code"] != 0
    assert "checkpoint" not in result, "owner returned a checkpoint receipt after the hard deadline"
    assert not acknowledgements, "owner sent checkpoint acceptance after the hard deadline"
    assert not (store.path / "artifacts" / (".attempt-" + result["attempt_id"]) / "checkpoint-ack.json").exists()
    balance = BudgetLedger(store).balance("affine")
    assert balance["closed"] and balance["committed_ms"] >= 3000
