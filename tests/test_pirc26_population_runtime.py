"""Real supervised complete-population fit, disposable synthetic grants only."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json

import pytest
import torch

from application.pirc26_components import component_bindings
from application.pirc26_data import decode_block
from application.pirc26_population import population_source
from application.pirc26_population_runtime import (BINDING_SCHEMA, execution_plugin, recovery_plugin,
    resume_command, source, read_fitted, result_validate)
from application.research_admission import plugin_binding
from application.research_budget import BudgetSpec, BudgetLedger
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryRegistry
from estimation.phase_space import fit_o1, O1Plan
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, digest, encode
from models.phase_space import DynamicsSpec, AffineAccelerationDrift, PhaseSpaceSDE
from tests.research_admission_fixtures import admit_fixture
from tests.test_pirc26_components_data import declarations, admitted
from tests.test_pirc26_dsde import contracts
from tests.test_pirc26_preparation import prepared_fixture, run
from tests.test_research_store import spec


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prepare(tmp_path, contracts, *, permit_consumer=True):
    study_id = "population-training-fixture"
    arm = {**spec()["arms"][0], "model_family_id": "M0", "method_family_id": "observed-o1"}
    fixture = prepared_fixture(tmp_path, contracts, arm=arm,
        consumer_study_ids=[study_id, "selection-consumer"] if permit_consumer else [])
    store = fixture[0]
    prepared = run(fixture)
    assert prepared["state"] == "SUCCEEDED", prepared
    population = population_source(store, prepared["preparation_ref"])
    n = population["normalizer"]
    dynamics = DynamicsSpec(population["provenance"]["coordinate_frame"], n["train_binding_hash"], digest(n),
        n["context_hash"], 56, tuple(n["means"]), tuple(n["scales"]))
    model = PhaseSpaceSDE(AffineAccelerationDrift(56), [[.1, 0.], [0., .1]], dynamics).to(dtype=torch.float64)
    document = prepared["prepared"]["training_population"]["document"]
    dto = decode_block(admitted(document), model)
    segment = document["segments"][0]
    request = dto.forecast_request(segment["segment_id"], 2, tuple(segment["time"][2:]),
        sample_count=8, brownian_root_id="a" * 64, chunk_size=8)
    cfg, profile, _, _, _, _ = declarations(model)
    cfg["forecast_request_hashes"] = [digest(asdict(request))]
    cfg["plan"].update(max_steps=4, patience=4, tolerance=0.)
    profile.update(observations=10, batches=2, components=60)
    plugin = execution_plugin("M0")
    value = spec()
    value.update(study_id=study_id, arms=[arm])
    value["cells"][0].update(block_id=document["block_id"], plugin_id=plugin.plugin_id,
        capability="generic-rollout", seed=cfg["seed"], visibility="synthetic")
    job = {"schema_version": "pirc26-worker-job-v1", "operation": "fit-and-forecast",
        "initial_checkpoint": model.checkpoint(), "o1_result": None, "batch_size": 4,
        "origins": [{"segment_id": segment["segment_id"], "origin_index": 2,
            "time_grid": list(request.time_grid), "sample_count": 8, "brownian_root_id": "a" * 64, "chunk_size": 8}]}
    bindings = component_bindings(cfg, profile, matrix_cells=1, registries=plugin.component_registries)
    grant = admit_fixture(store, value, plugin, tmp_path.absolute(), execution_config=cfg,
        execution_inputs={**profile, "runtime_root": str(tmp_path / "ledger"), "store_id": store.store_id},
        execution_components=bindings, input_content=encode(document),
        input_block_metadata=population["protocol_entry"], input_data_root=population["data_root"],
        recovery_command_builder=resume_command, package_payload={"pirc26_job": job,
            "pirc26_population": {"schema_version": BINDING_SCHEMA, "preparation_ref": prepared["preparation_ref"],
                "population_hash": digest(population["provenance"])}})
    store.register(value, digest(value))
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(plugin)
    recovery.register(recovery_plugin("M0"))
    return store, value, plugin, registry, recovery, grant, model, dto


def execute(fixture):
    store, value, _, registry, recovery, *_ = fixture
    return SharedRunner(store, registry, recovery).run_cell(value["study_id"], digest(value["cells"][0]),
        budget=BudgetSpec(60, category="smoke"))


def receipt_for(store, attempt_id):
    admission = next(e["payload"] for e in store.events() if e["event_kind"] == "ADMISSION"
        and e["payload"].get("attempt_id") == attempt_id)
    return store.manifest("admission-" + admission["admission_hash"])


def test_actual_supervised_fit_visits_all_members_and_reuses_without_new_job(tmp_path, contracts, monkeypatch):
    fixture = prepare(tmp_path, contracts)
    store, value, plugin, _, _, grant, model, dto = fixture
    outcome = execute(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    attempt = store.attempts()[outcome["attempt_id"]]
    result = json.loads(store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant))
    fit = result["fit"]
    evidence = fit["training_population"]
    assert evidence["observations"] == 10 and evidence["transitions"] == 6 and evidence["batch_count"] == 2
    assert evidence["independent_block_ids"] == ["train-unit"]
    assert evidence["use"] == "training-diagnostics-not-held-out"
    direct = fit_o1(model, dto.transitions(batch_size=4), O1Plan(**value["cells"][0]["execution"]["config"]["plan"]))
    assert fit["checkpoint"] == direct["checkpoint"] and fit["history"] == direct["history"]
    reads = [e for e in store.events() if e["event_kind"] == "READ_STARTED"
        and e["payload"].get("attempt_id") == outcome["attempt_id"]]
    assert reads and all(e["payload"]["block_id"] == value["cells"][0]["block_id"] for e in reads)
    balance, attempts = BudgetLedger(store).balance("affine"), store.attempts()
    reused = execute(fixture)
    assert reused["reused"] and BudgetLedger(store).balance("affine") == balance and store.attempts() == attempts
    receipt = receipt_for(store, outcome["attempt_id"])
    forged = deepcopy(result)
    forged["fit"]["training_population"]["transitions"] -= 1
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        result_validate(receipt, forged)
    disclosure = {**grant, "authorization_id": "fixture-model-disclosure", "consumer_study_ids": ["selection-consumer"]}
    store.authorize(disclosure)
    actual = read_fitted(store, outcome["attempt_id"], consumer_study_id="selection-consumer", authorization=disclosure)
    assert actual["checkpoint"] == fit["checkpoint"] and actual["scientific_qualification"] == "not-established"
    denied = {**disclosure, "consumer_study_ids": []}
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read_fitted(store, outcome["attempt_id"], consumer_study_id="selection-consumer", authorization=denied)
    before = len([e for e in store.events() if e["event_kind"] == "READ_STARTED"])
    for fault in ("hash", "membership", "batch-quota", "context-quota", "early-stop", "arm", "normalizer"):
        forged_receipt = deepcopy(receipt)
        if fault == "hash":
            forged_receipt["documents"]["package"]["payload"]["pirc26_population"]["population_hash"] = "f" * 64
        elif fault == "membership":
            forged_receipt["documents"]["protocol"]["blocks"][0]["independent_block_ids"] = ["invented-unit"]
        elif fault in ("batch-quota", "context-quota"):
            forged_receipt["cell"]["execution"]["inputs"]["batches" if fault == "batch-quota" else "components"] = 1
        elif fault == "early-stop":
            forged_receipt["cell"]["execution"]["config"]["plan"]["patience"] = 1
        elif fault == "arm":
            forged_receipt["spec"]["arms"][0]["budget_seconds"] -= 1
        else:
            job = forged_receipt["documents"]["package"]["payload"]["pirc26_job"]
            checkpoint = job["initial_checkpoint"]
            checkpoint["model_card"]["spec"]["scales"][0] *= 2
            checkpoint["sha256"] = digest({k: v for k, v in checkpoint.items() if k != "sha256"})
            config = forged_receipt["cell"]["execution"]["config"]
            config.update(initial_model_hash=checkpoint["sha256"],
                dynamics_spec_hash=digest(checkpoint["model_card"]["spec"]))
        with pytest.raises(ResearchError):
            source(forged_receipt, store=store)
    assert len([e for e in store.events() if e["event_kind"] == "READ_STARTED"]) == before
    # Only permission-clock injection, after the numerical worker is stopped.
    # Actual supervisor clocks, deadline, costs and success remain untouched.
    import application.pirc26_preparation as preparation
    import application.pirc26_population_runtime as population_runtime
    import application.research_data as data
    import infrastructure.research_store as store_module
    expired = [False]
    class PermissionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100, 1, 1, tzinfo=timezone.utc) if expired[0] else datetime.now(tz)
    for module in (preparation, population_runtime, data, store_module):
        monkeypatch.setattr(module, "datetime", PermissionClock)
    completion = store._read_completion
    def expire_after_authority(authority, expiry):
        def checked():
            authority()
            expired[0] = True
        completion(checked, expiry)
    monkeypatch.setattr(store, "_read_completion", expire_after_authority)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read_fitted(store, outcome["attempt_id"], consumer_study_id="selection-consumer", authorization=disclosure)
    assert expired[0] and store.attempts()[outcome["attempt_id"]]["state"] == "SUCCEEDED"


def test_original_raw_grant_must_allow_the_actual_training_consumer(tmp_path, contracts):
    fixture = prepare(tmp_path, contracts, permit_consumer=False)
    store = fixture[0]
    before = len([e for e in store.events() if e["event_kind"] == "READ_STARTED"])
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        execute(fixture)
    assert len([e for e in store.events() if e["event_kind"] == "READ_STARTED"]) == before


def test_result_hook_is_callable_and_part_of_immutable_plugin_binding():
    plugin = execution_plugin("M0")
    assert plugin_binding(plugin) != plugin_binding(replace(plugin, result_validator=None))
    with pytest.raises(ResearchError):
        CapabilityRegistry().register(replace(plugin, result_validator="not-callable"))
