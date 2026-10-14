"""Synthetic numerical worker under real admission, deadline and recovery.

The input/authorization/upstream attestations are explicitly disposable test
controls. No production source, qualification or statistical result is claimed.
"""

import json
from pathlib import Path
import sys

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryPlugin, RecoveryRegistry, SharedRecovery
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchStore, atomic_write, digest, encode
from tests.research_admission_fixtures import admit_fixture, synthetic_plugin
from tests.test_research_store import spec


PROGRAM = r'''
import json, os, sys, time
print('synthetic-worker-boot', time.monotonic(), flush=True)
import torch
from pathlib import Path
from application.pirc26_training_control import managed_fit_o1, exit_managed_worker, ManagedTrainingControl
from estimation.phase_space import O1Plan
from inference.phase_space import forecast
from infrastructure.research_store import digest, encode
from tests.test_pirc26_dynamics import model
from tests.test_pirc26_training_forecast import batch, request
output, value, cell, restored, pause_step = sys.argv[1:]
value, cell = json.loads(value), json.loads(cell)
pause_step = int(pause_step)
torch.set_num_threads(1)
m = model('M2')
state = json.loads(Path(restored).read_bytes()) if restored else None
print('synthetic-inputs-ready', time.monotonic(), flush=True)
original_progress = ManagedTrainingControl.progress
def report(control, row):
    original_progress(control, row)
    if row['step'] % 100 == 0 or row['step'] == 1:
        print('synthetic-training-step', row['step'], time.monotonic(), flush=True)
    if state is None and row['step'] == pause_step:
        # A synthetic synchronization barrier, not simulated compute or a
        # numerical timing measurement. Keep the real owner clock, 80% signal,
        # hard fuse and save/ACK exchange; do not predict machine throughput.
        print('synthetic-awaiting-owner-checkpoint', row['step'], flush=True)
        while not control.requested():
            if time.monotonic() >= control.control.descriptor['deadline']:
                raise TimeoutError('owner checkpoint request did not arrive before the hard deadline')
            time.sleep(0.01)
ManagedTrainingControl.progress = report
try:
    result = managed_fit_o1(m, [batch(m, 4096)],
        O1Plan(max_steps=600, patience=600, tolerance=0., fit_diffusion=False), restored_state=state)
except SystemExit as stopped:
    # ACK is complete. Do not enter Windows numerical-library finalizers after
    # signalling the supported checkpoint exit; the owner still confirms stop.
    exit_managed_worker(stopped.code)
prediction = forecast(m, request(8, 8))
endpoint = prediction['samples'][:, -1, :2].mean(0).detach().tolist()
payload = {'metrics': {'position_norm': float(torch.tensor(endpoint).norm())},
           'forecast': {'mean_endpoint': endpoint},
           'fit': {key: result[key] for key in ('checkpoint','history','best_train_objective','steps')},
           'source_schema': 'pirc26-managed-training-synthetic-v1'}
document = {'schema_version': 'pirc25-result-v1', 'status': 'SUCCEEDED',
    'spec_hash': digest(value), 'cell_hash': digest(cell), 'protocol_hash': value['protocol_hash'],
    'input_hash': value['data_hash'], 'output_hash': digest(payload),
    'state_order': ['x','y','vx','vy'], 'units': ['m','m','m/s','m/s'],
    'resume_level': 'exact', 'qualification': 'fixture', 'metric_units': {'position_norm': 'm'}, **payload}
Path(output).write_bytes(encode(document))
# write_bytes closed the result. Mirror the shared internal wrapper's bounded
# native exit, rather than cancelling its watchdog before Python finalizers.
exit_managed_worker(0)
'''


def command(output, value, cell, *, pause_step=0):
    return [sys.executable, "-c", PROGRAM, str(output), encode(value).decode(), encode(cell).decode(), "", str(pause_step)]


def interruption_command(output, value, cell):
    return command(output, value, cell, pause_step=100)


def resume_command(output, value, cell, state):
    path = Path(output).with_name("training-resume.json")
    atomic_write(path, encode(state))
    result = command(output, value, cell)
    result[-2] = str(path)
    return result


def prepare(root, study_id, *, interrupt=False):
    root.mkdir()
    store = ResearchStore(root, "pirc26-synthetic-training", initialize=True)
    value = spec()
    value["study_id"] = study_id
    value["cells"][0].update(plugin_id="pirc26-training-control-fixture", capability="generic-rollout", visibility="synthetic")
    builder = interruption_command if interrupt else command
    plugin = synthetic_plugin("pirc26-training-control-fixture", frozenset({"generic-rollout"}),
                              ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "exact", builder)
    plugin.registry_entry.resource_contract["counts"].update(
        steps={"constant": 600}, paths={"constant": 8}, observations={"constant": 4096}, components={"constant": 128})
    plugin.registry_entry.resource_contract["limits"]["result_bytes"] = 4 * 1024 * 1024
    plugin.registry_entry.resource_contract["tensors"] = [
        {"name": "train_states_and_targets", "axes": ["observations", "state_dim"], "item_bytes": 16},
        {"name": "time_and_dt", "axes": ["observations"], "item_bytes": 16},
        *[{"name": "activation_workspace_" + str(i), "axes": ["observations", "components"], "item_bytes": 8}
          for i in range(4)],
        {"name": "parameter_optimizer_workspace", "axes": ["components", "components"], "item_bytes": 16},
        {"name": "prediction_states", "axes": ["paths", "steps", "state_dim"], "item_bytes": 8}]
    registry = CapabilityRegistry()
    registry.register(plugin)
    adapters = RecoveryRegistry()
    adapters.register(RecoveryPlugin(plugin.plugin_id, "exact", resume_command, "1.0.0"))
    grant = admit_fixture(store, value, plugin, root, recovery_command_builder=resume_command)
    store.register(value, digest(value))
    return store, value, registry, adapters, grant


def worker_log(store, result):
    return (store.path / 'artifacts' / ('.attempt-' + result['attempt_id']) / 'worker.log').read_text()


def test_owner_checkpoint_reopened_numerical_training_matches_continuous_and_charges_both_attempts(tmp_path):
    continuous, baseline, registry, adapters, _ = prepare(tmp_path / "continuous", "continuous-training")
    expected = SharedRunner(continuous, registry, recovery_registry=adapters).run_cell(
        baseline["study_id"], digest(baseline["cells"][0]), budget=BudgetSpec(70))
    assert expected["state"] == "SUCCEEDED", (expected, worker_log(continuous, expected))
    expected_output = json.loads((continuous.path / "artifacts" / expected["artifact_id"]).read_bytes())
    store, value, registry, adapters, grant = prepare(tmp_path / "resumed", "resumed-training", interrupt=True)
    interrupted = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
        value["study_id"], digest(value["cells"][0]), budget=BudgetSpec(70))
    assert interrupted["state"] == "FAILED", (interrupted, worker_log(store, interrupted))
    assert store.attempts()[interrupted["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED"
    saved = [event["payload"] for event in store.events() if event["event_kind"] == "CHECKPOINT_SAVED"]
    assert len(saved) == 1 and saved[0]["progress"]["completed_steps"] == 100
    requests = [event["payload"] for event in store.events() if event["event_kind"] == "CHECKPOINT_REQUESTED"]
    assert len(requests) == 1 and requests[0]["supported"]
    assert 0 < requests[0]["remaining_seconds"] <= 70 * 0.2 + 1e-6
    cost = BudgetLedger(store).balance("affine")["committed_ms"]
    assert cost > 0 and not BudgetLedger(store).balance("affine")["closed"]
    reopened = ResearchStore(tmp_path / "resumed", "pirc26-synthetic-training")
    resumed = SharedRecovery(reopened, registry, adapters).resume(interrupted["attempt_id"], saved[0]["artifact_id"],
                                                               authorization=grant, budget=BudgetSpec(70))
    assert resumed["state"] == "SUCCEEDED", resumed
    actual = json.loads((reopened.path / "artifacts" / resumed["artifact_id"]).read_bytes())
    assert actual["fit"] == expected_output["fit"]
    assert actual["forecast"] == expected_output["forecast"]
    assert actual["metrics"] == expected_output["metrics"]
    assert BudgetLedger(reopened).balance("affine")["committed_ms"] > cost
    assert reopened.attempts()[resumed["attempt_id"]]["parent_attempt_id"] == interrupted["attempt_id"]
