"""Actual versioned workers, real owner gates; disposable synthetic data only."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from application.pirc26_components import component_bindings
from application.pirc26_data import decode_block
from application.pirc26_runtime import execution_plugin, command, resume_command, validate_job
from application.research_admission import AdmissionGate
from application.research_budget import BudgetSpec, BudgetLedger
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryRegistry
from experiments.pirc25.runner import SharedRunner
from infrastructure.pirc26_worker import run
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_pirc26_components_data import declarations, document, admitted
from tests.test_pirc26_dynamics import model
from tests.test_research_store import spec


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prepare(tmp_path, *, operation="fit-and-forecast", role="train", corrupt=False, long_training=False):
    m = model("M2" if long_training else "M0")
    doc = document(m)
    doc["block_id"] = "fixture-1"
    if long_training:
        segment = doc["segments"][0]
        segment.update(time=[i / 100 for i in range(4098)], position=[[i / 100, i / 200] for i in range(4098)],
                       condition=[[] for _ in range(4098)], condition_available_at=[i / 100 for i in range(4098)])
    dto = decode_block(admitted(doc), m)
    grid = (.02, .03, .04) if long_training else (2., 3., 4.)
    req = dto.forecast_request("synthetic-segment", 2, grid, sample_count=8, brownian_root_id="a" * 64, chunk_size=8)
    cfg, profile, _, _, _, _ = declarations(m)
    cfg["forecast_request_hashes"] = [digest(asdict(req))]
    profile["steps"] = 4
    if long_training:
        cfg["plan"].update(max_steps=600, patience=600, tolerance=0.)
        profile.update(steps=600, observations=4098)
    plugin = execution_plugin("O1", cfg["family"])
    components = component_bindings(cfg, profile, matrix_cells=1, registries=plugin.component_registries)
    root = tmp_path.absolute()
    store = ResearchStore(root, "pirc26-synthetic-runtime", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id=plugin.plugin_id, capability="generic-rollout", seed=cfg["seed"], visibility="synthetic")
    job = {"schema_version": "pirc26-worker-job-v1", "operation": operation, "initial_checkpoint": m.checkpoint(),
        "o1_result": None, "batch_size": 4096 if long_training else 4,
        "origins": [{"segment_id": "synthetic-segment", "origin_index": 2,
            "time_grid": list(grid), "sample_count": 8, "brownian_root_id": "a" * 64, "chunk_size": 8}]}
    if corrupt:
        doc["state_units"] = ["unknown"] * 4  # hash-admitted but semantically invalid, worker must refuse.
    grant = admit_fixture(store, value, plugin, root, execution_config=cfg,
        execution_inputs={**profile, "runtime_root": str(root), "store_id": store.store_id},
        execution_components=components, input_content=encode(doc), package_payload={"pirc26_job": job},
        recovery_command_builder=resume_command, split_role=role)
    store.register(value, digest(value))
    registry = CapabilityRegistry()
    registry.register(plugin)
    recovery = RecoveryRegistry()
    from application.pirc26_runtime import recovery_plugin
    recovery.register(recovery_plugin("O1", cfg["family"]))
    return store, value, plugin, job, registry, recovery, grant


def owner_admit(store, value, plugin):
    cell = value["cells"][0]
    run_id = store.register_run(value["study_id"], cell)
    attempt = store.new_attempt(run_id)
    output = store.path / "artifacts" / (".attempt-" + attempt) / "result.json"
    output.parent.mkdir()
    receipt = AdmissionGate(store).prepare(value, cell, plugin, attempt)
    return output, receipt


def test_actual_owner_command_reads_only_admitted_block_and_requires_attempt_scope(tmp_path):
    store, value, plugin, _, _, _, _ = prepare(tmp_path)
    output, receipt = owner_admit(store, value, plugin)
    args = command(output, value, value["cells"][0])
    assert args[1:3] == ["-m", "infrastructure.pirc26_worker"]
    handoff = json.loads((output.parent / "pirc26-handoff.json").read_bytes())
    assert args[-1] == digest(handoff) and handoff["admission_hash"] == receipt["admission_hash"]
    assert "content_utf8" not in handoff["transport"] and handoff["transport"]["content_chunks"]
    reads = [e for e in store.events() if e["event_kind"] == "READ_COMPLETED"]
    assert len(reads) == 3  # central admission verify + provider verify + provider read.
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        command(tmp_path / "result.json", value, value["cells"][0])
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        run(str(output), args[-1])  # cannot run outside the actual owner control channel.


def test_actual_decode_enforces_declared_capacity_before_tensor_allocation(monkeypatch):
    m = model("M0")
    dto = admitted(document(m))
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("capacity refusal allocated tensors"))
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        decode_block(dto, m, max_observations=4)


@pytest.mark.parametrize("fault", ["selection-fit", "final-unqualified", "lineage", "model", "extra", "basis-adam"])
def test_job_refusal_precedes_new_physical_read(tmp_path, fault):
    store, value, plugin, job, _, _, _ = prepare(tmp_path)
    _, receipt = owner_admit(store, value, plugin)
    job, receipt = deepcopy(job), deepcopy(receipt)
    if fault == "selection-fit":
        receipt["documents"]["protocol"]["blocks"][0]["split_role"] = "selection"
    elif fault == "final-unqualified":
        receipt["documents"]["protocol"]["blocks"][0]["split_role"] = "final-eval"
        job["operation"] = "forecast"
    elif fault == "lineage":
        job["o1_result"] = {}
    elif fault == "model":
        job["initial_checkpoint"]["sha256"] = "f" * 64
    elif fault == "basis-adam":
        receipt["cell"]["execution"]["config"]["family"] = "M1-S"
    else:
        job["raw_source_path"] = "not-authority"
    before = len(store.events())
    with pytest.raises(ResearchError):
        validate_job(job, receipt)
    assert len(store.events()) == before


@pytest.mark.parametrize("operation,role,corrupt", [
    ("fit-and-forecast", "train", False), ("forecast", "validation", False),
    ("fit-and-forecast", "train", True)])
def test_actual_shared_runner_versioned_worker_publishes_or_fails_and_settles(tmp_path, operation, role, corrupt):
    store, value, _, _, registry, recovery, _ = prepare(tmp_path, operation=operation, role=role, corrupt=corrupt)
    result = SharedRunner(store, registry, recovery_registry=recovery).run_cell(value["study_id"],
        digest(value["cells"][0]), budget=BudgetSpec(70))
    log = (store.path / "artifacts" / (".attempt-" + result["attempt_id"]) / "worker.log").read_text()
    if corrupt:
        assert result["state"] == "FAILED", (result, log)
        assert store.attempts()[result["attempt_id"]]["error_code"] == "WORKER_FAILED"
        failure = json.loads((store.path / "artifacts" / (".attempt-" + result["attempt_id"]) / "pirc26-failure.json").read_bytes())
        assert failure["error_code"] == "CONTRACT_MISMATCH"
        assert not result.get("artifact_id")
    else:
        assert result["state"] == "SUCCEEDED", (result, log)
        actual = json.loads((store.path / "artifacts" / result["artifact_id"]).read_bytes())
        assert actual["qualification"] == "fixture" and actual["metrics"]["energy_score"] >= 0
        assert actual["component_plan_hash"] == value["cells"][0]["execution"]["component_plan_hash"]
        assert actual["fit"]["status"] == ("MAX_STEPS" if operation == "fit-and-forecast" else "FROZEN")
        assert actual["source_identity"]["dataset_id"] == "synthetic"
    assert BudgetLedger(store).balance("affine")["committed_ms"] > 0
    assert not any(e["event_kind"] == "READ_STARTED" and e["payload"].get("split_role") == "final-eval" for e in store.events())


def test_actual_versioned_worker_reopens_exact_training_and_forecasts_under_new_budget(tmp_path):
    from application.research_recovery import SharedRecovery
    baseline, spec1, _, _, registry1, recovery1, _ = prepare(tmp_path / "baseline", long_training=True)
    expected = SharedRunner(baseline, registry1, recovery_registry=recovery1).run_cell(
        spec1["study_id"], digest(spec1["cells"][0]), budget=BudgetSpec(100))
    assert expected["state"] == "SUCCEEDED", expected
    target = json.loads((baseline.path / "artifacts" / expected["artifact_id"]).read_bytes())
    store, value, _, _, registry, recovery, grant = prepare(tmp_path / "resumed", long_training=True)
    stopped = SharedRunner(store, registry, recovery_registry=recovery).run_cell(
        value["study_id"], digest(value["cells"][0]), budget=BudgetSpec(40))
    assert stopped["state"] == "FAILED", stopped
    assert store.attempts()[stopped["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED"
    saves = [event["payload"] for event in store.events() if event["event_kind"] == "CHECKPOINT_SAVED"]
    assert len(saves) == 1 and 0 < saves[0]["progress"]["completed_steps"] < 600
    prior_cost = BudgetLedger(store).balance("affine")["committed_ms"]
    reopened = ResearchStore(tmp_path / "resumed", store.store_id)
    result = SharedRecovery(reopened, registry, recovery).resume(stopped["attempt_id"], saves[0]["artifact_id"],
        authorization=grant, budget=BudgetSpec(100))
    assert result["state"] == "SUCCEEDED", result
    actual = json.loads((reopened.path / "artifacts" / result["artifact_id"]).read_bytes())
    assert actual["forecast"] == target["forecast"] and actual["metrics"] == target["metrics"]
    for key in ("checkpoint", "history", "best_train_objective", "steps"):
        assert actual["fit"][key] == target["fit"][key]
    assert reopened.attempts()[result["attempt_id"]]["parent_attempt_id"] == stopped["attempt_id"]
    assert BudgetLedger(reopened).balance("affine")["committed_ms"] > prior_cost > 0
