"""Real 80% save/ACK, reopened complete-population/selection continuation.

The disposable worker bootstrap ONLY synchronizes a known numerical boundary
with the real owner signal. All clocks, deadlines, source checks, numerical
updates, saves, process containment, settlement and retry remain original.
Barrier elapsed time is charged, not reported as numerical throughput evidence.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest
import torch

from application.pirc26_forecast_control import materialize_fit, unpack_json
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_recovery import SharedRecovery
from estimation.phase_space import O1Plan, fit_o1
from experiments.pirc25.runner import SharedRunner
from inference.phase_space import forecast
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_pirc26_dsde import contracts
from tests.test_pirc26_population_runtime import prepare, receipt_for, one_thread
from tests.test_pirc26_selection_runtime import selection_fixture


BOOTSTRAP = r'''
import sys, time
from pathlib import Path
from infrastructure.pirc26_worker_control import require_owned_worker
require_owned_worker(sys.argv[1])  # Ownership before numerical imports too.
from application.pirc26_training_control import ManagedTrainingControl
from application.pirc26_forecast_control import ManagedForecastControl
phase = sys.argv.pop()
def await_owner(control, poll):
    while not poll():
        if time.monotonic() >= control.control.descriptor['deadline']:
            raise TimeoutError('synthetic barrier reached original hard deadline')
        time.sleep(.01)
if phase == 'training':
    original = ManagedTrainingControl.progress
    def progress(self, row):
        original(self, row)
        if row['step'] == 1:
            print('synthetic-complete-population-first-batch-awaits-owner', flush=True)
            await_owner(self, self.requested)
    ManagedTrainingControl.progress = progress
else:
    original = ManagedForecastControl.requested
    def requested(self):
        self.synthetic_poll_count = getattr(self, 'synthetic_poll_count', 0) + 1
        if self.index == 1 and self.synthetic_poll_count == 3:
            print('synthetic-selection-second-origin-partial-awaits-owner', flush=True)
            await_owner(self, lambda: original(self))
        return original(self)
    ManagedForecastControl.requested = requested
from infrastructure.pirc26_worker import main
main()
'''


def synchronize_original_dispatch(monkeypatch, phase):
    import application.pirc26_runtime as runtime
    original = runtime._dispatch
    prefix = "pirc26-population-" if phase == "training" else "pirc26-selection-"
    def dispatch(output, spec, cell, state, recovery):
        command = original(output, spec, cell, state, recovery)
        if state is None and cell["plugin_id"].startswith(prefix):
            assert command[1:3] == ["-m", "infrastructure.pirc26_worker"]
            return [command[0], "-c", BOOTSTRAP, *command[3:], phase]
        return command  # Actual original resumed worker, no synchronization.
    monkeypatch.setattr(runtime, "_dispatch", dispatch)


def late_source_expiry_must_refuse_before_retry(store, recovery, parent, checkpoint, grant, monkeypatch):
    """Expiry after checkpoint validator but during its final verification I/O.

    Checkpoint grant remains valid. The original raw/model source authority
    must still be checked at completion of the WHOLE recovery preparation.
    """
    import application.pirc26_preparation as preparation
    expired = [False]
    class PermissionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100,1,1,tzinfo=timezone.utc) if expired[0] else datetime.now(tz)
    real_verify = store.verify_artifact_read
    def final_io(artifact_id, *, purpose, authorization):
        result = real_verify(artifact_id, purpose=purpose, authorization=authorization)
        if artifact_id == checkpoint and purpose == "resume":
            expired[0] = True
        return result
    attempts, balance = store.attempts(), BudgetLedger(store).balance("affine")
    with monkeypatch.context() as patch:
        patch.setattr(preparation, "datetime", PermissionClock)
        patch.setattr(store, "verify_artifact_read", final_io)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            recovery.prepare(parent, checkpoint, authorization=grant)
    assert expired[0]
    assert store.attempts() == attempts and BudgetLedger(store).balance("affine") == balance


def late_dispatch_expiry_must_refuse_before_worker(store, value, registry, adapters, monkeypatch):
    import application.pirc26_preparation as preparation
    import application.pirc26_runtime as runtime
    expired = [False]
    class PermissionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100,1,1,tzinfo=timezone.utc) if expired[0] else datetime.now(tz)
    original = runtime.read_block
    def after_derived_read(*args, **kwargs):
        block = original(*args, **kwargs)
        expired[0] = True
        return block
    before, balance, attempts = store.events(), BudgetLedger(store).balance("affine"), store.attempts()
    with monkeypatch.context() as patch:
        patch.setattr(preparation, "datetime", PermissionClock)
        patch.setattr(runtime, "read_block", after_derived_read)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            SharedRunner(store, registry, adapters).run_cell(value["study_id"], digest(value["cells"][0]),
                budget=BudgetSpec(60, category="smoke"))
    assert expired[0]
    events = store.events()[len(before):]
    assert any(e["event_kind"] == "READ_STARTED" for e in events)
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in events)
    assert BudgetLedger(store).balance("affine") == balance
    settlement = next(e["payload"] for e in events if e["event_kind"] == "SETTLE")
    assert settlement["outcome"] == "PREFLIGHT_FAILED" and settlement["charged_ms"] == 0
    denied = [a for key,a in store.attempts().items() if key not in attempts]
    assert len(denied) == 1 and denied[0]["state"] == "PREFLIGHT_FAILED"
    return denied[0]["attempt_id"]


@pytest.mark.parametrize("phase", ["training", "selection"])
def test_original_80_percent_checkpoint_reopens_complete_population_or_frozen_selection(tmp_path, contracts, monkeypatch, phase):
    if phase == "training":
        fixture = prepare(tmp_path, contracts)
        store, value, plugin, registry, adapters, grant, model, dto = fixture
        config = value["cells"][0]["execution"]["config"]
        expected_fit = fit_o1(model, dto.transitions(batch_size=4), O1Plan(**config["plan"]))
        # The direct oracle is disposable synthetic work, not production cost
        # or a second claimed supervised execution/qualification.
        package = store.manifest("package-" + value["admission"]["package_hash"])
        recipes = package["payload"]["pirc26_job"]["origins"]
        requests = [dto.forecast_request(r["segment_id"], r["origin_index"], r["time_grid"],
            sample_count=r["sample_count"], brownian_root_id=r["brownian_root_id"], chunk_size=r["chunk_size"])
            for r in recipes]
    else:
        store, value, plugin, registry, adapters, grant, fitted, model, requests = selection_fixture(tmp_path, contracts)
        expected_fit = {"checkpoint": fitted["checkpoint"]}
    expected_predictions = [forecast(model, request) for request in requests]
    denied = late_dispatch_expiry_must_refuse_before_worker(store, value, registry, adapters, monkeypatch)
    with monkeypatch.context() as patch:
        synchronize_original_dispatch(patch, phase)
        stopped = SharedRunner(store, registry, adapters).run_cell(value["study_id"], digest(value["cells"][0]),
            budget=BudgetSpec(60, category="smoke"), parent_attempt_id=denied,
            reason="explicit synthetic checkpoint proof after refused source-expiry preflight")
    assert stopped["state"] == "FAILED", stopped
    assert store.attempts()[stopped["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED"
    journal = store.events()
    requests_to_save = [e for e in journal if e["event_kind"] == "CHECKPOINT_REQUESTED"
        and e["payload"]["attempt_id"] == stopped["attempt_id"]]
    saved = [e for e in journal if e["event_kind"] == "CHECKPOINT_SAVED"
        and e["payload"]["attempt_id"] == stopped["attempt_id"]]
    assert len(requests_to_save) == len(saved) == 1
    assert requests_to_save[0]["payload"]["supported"]
    assert 0 < requests_to_save[0]["payload"]["remaining_seconds"] <= 12
    assert requests_to_save[0]["sequence"] < saved[0]["sequence"]
    checkpoint_id = saved[0]["payload"]["artifact_id"]
    checkpoint = json.loads(store.read_artifact(checkpoint_id, purpose="resume", authorization=grant))
    state, progress = checkpoint["state"], checkpoint["progress"]
    assert not {"budget", "remaining_seconds"} & set(state)
    if phase == "training":
        assert state["step"] == progress["completed_steps"] == 1 and progress["total_steps"] == 4
        assert state["method_state"]["schema_version"] == "pirc26-training-state-v2"
    else:
        assert state["method_state"]["schema_version"] == "pirc26-owned-forecast-state-v1"
        assert state["method_state"]["origin_index"] == 1 and state["method_state"]["active"]["step"] == 1
        assert len(unpack_json(state["method_state"]["finished"])) == 1
        fit = materialize_fit(state["method_state"]["fit"])
        assert fit["status"] == "FROZEN" and fit["checkpoint"] == fitted["checkpoint"]
        assert fit["selection_input"]["independent_block_id"] == "selection-unit"
    charged = BudgetLedger(store).balance("affine")
    settlement = next(e["payload"] for e in journal if e["event_kind"] == "SETTLE"
        and e["payload"]["attempt_id"] == stopped["attempt_id"])
    assert 0 < settlement["charged_ms"] == settlement["monotonic_elapsed_ms"] <= 60000
    assert not charged["closed"]
    reopened = ResearchStore(tmp_path / "ledger", store.store_id)
    recovery = SharedRecovery(reopened, registry, adapters)
    late_source_expiry_must_refuse_before_retry(reopened, recovery, stopped["attempt_id"], checkpoint_id, grant, monkeypatch)
    original_receipt = receipt_for(reopened, stopped["attempt_id"])
    changed = deepcopy(original_receipt)
    binding = changed["documents"]["package"]["payload"]["pirc26_population" if phase == "training" else "pirc26_selection"]
    binding["population_hash"] = "f" * 64
    with monkeypatch.context() as patch:
        patch.setattr(reopened, "_verified_artifact_content", lambda *a,**k: pytest.fail("metadata hook opened payload"))
        with pytest.raises(ResearchError):
            plugin.checkpoint_validator(changed, state, progress, store=reopened)
    resumed = recovery.resume(stopped["attempt_id"], checkpoint_id, authorization=grant,
        budget=BudgetSpec(60, category="smoke"))
    assert resumed["state"] == "SUCCEEDED", resumed
    actual = json.loads(reopened.read_artifact(resumed["artifact_id"], purpose="evaluate", authorization=grant))
    assert reopened.attempts()[resumed["attempt_id"]]["parent_attempt_id"] == stopped["attempt_id"]
    assert BudgetLedger(reopened).balance("affine")["committed_ms"] > charged["committed_ms"]
    for key in (("checkpoint", "history", "best_train_objective", "steps") if phase == "training" else ("checkpoint",)):
        assert actual["fit"][key] == expected_fit[key]
    for output, expected in zip(actual["forecast"]["origins"], expected_predictions):
        assert torch.equal(torch.tensor(output["samples"], dtype=torch.float64), expected["samples"])
        assert output["moments"] == expected["moments"] and output["sample_ids"] == expected["sample_ids"]
    evidence = actual["fit"]["training_population" if phase == "training" else "selection_input"]
    assert evidence["observations"] == 10
    if phase == "training":
        assert evidence["independent_block_ids"] == ["train-unit"] and evidence["transitions"] == 6
    else:
        assert evidence["independent_block_id"] == "selection-unit" and len(evidence["segment_ids"]) == 2
    observation = actual["fit"]["worker_resource_observation"]
    assert observation["attempt_id"] == resumed["attempt_id"] and not observation["includes_previous_attempts"]
    before, attempts = BudgetLedger(reopened).balance("affine"), reopened.attempts()
    reused = SharedRunner(reopened, registry, adapters).run_cell(value["study_id"], digest(value["cells"][0]),
        budget=BudgetSpec(60, category="smoke"))
    assert reused["reused"] and reopened.attempts() == attempts and BudgetLedger(reopened).balance("affine") == before
