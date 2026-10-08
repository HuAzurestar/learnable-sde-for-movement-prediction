"""Actual complete fit -> model-only attachment -> separate owned selection."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest
import torch

from application.pirc26_components import component_bindings
from application.pirc26_data import decode_block
from application.pirc26_fitted_model import publish_fitted, read_model
from application.pirc26_preparation import load_prepared
from application.pirc26_selection import selection_source, selection_populations
from application.pirc26_selection_runtime import (BINDING_SCHEMA, execution_plugin, recovery_plugin,
    resume_command, source, result_validate)
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryRegistry
from experiments.pirc25.runner import SharedRunner
from inference.phase_space import forecast
from infrastructure.research_store import ResearchError, digest, encode
from models.phase_space import PhaseSpaceSDE
from tests.research_admission_fixtures import admit_fixture
from tests.test_pirc26_components_data import declarations, admitted
from tests.test_pirc26_dsde import contracts
from tests.test_pirc26_population_runtime import prepare, execute, receipt_for, one_thread
from tests.test_research_store import spec


def selection_fixture(tmp_path, contracts):
    training = prepare(tmp_path, contracts, selection_files=2)
    store, training_spec, *_ = training
    trained = execute(training)
    assert trained["state"] == "SUCCEEDED", trained
    model_ref = publish_fitted(store, trained["attempt_id"], authorization=training[5])
    disclosure = {**training[5], "authorization_id": "selection-model-disclosure",
        "consumer_study_ids": ["selection-consumer"]}
    store.authorize(disclosure)
    fitted = read_model(store, model_ref, consumer_study_id="selection-consumer", authorization=disclosure)
    prep_ref = fitted["training_population"]["preparation_ref"]
    selected = selection_source(store, prep_ref, "selection-unit", consumer_study_id="selection-consumer",
        arm=training_spec["arms"][0])
    # Explicit disposable owner oracle only. Not the consumer's preflight;
    # never a new normalizer/model fit or a blind research-data qualification.
    prepared = load_prepared(store, prep_ref)
    document = selection_populations(prepared["blocks"], prepared["normalizer"],
        study_id=selected["provenance"]["source_study_id"])[0]["document"]
    model = PhaseSpaceSDE.from_checkpoint(fitted["checkpoint"])
    transport = {**admitted(document), "split_role":"selection", "purpose":"select"}
    dto = decode_block(transport, model)
    recipes, requests = [], []
    for segment in document["segments"]:
        brownian_root = digest(["disposable-selection-origin", segment["segment_id"]])
        request = dto.forecast_request(segment["segment_id"], 2, tuple(segment["time"][2:]),
            sample_count=8, brownian_root_id=brownian_root, chunk_size=8)
        requests.append(request)
        recipes.append({"segment_id":segment["segment_id"], "origin_index":2, "time_grid":list(request.time_grid),
            "sample_count":8, "brownian_root_id":brownian_root, "chunk_size":8})
    config, profile, *_ = declarations(model)
    config["forecast_request_hashes"] = [digest(asdict(r)) for r in requests]
    profile.update(observations=10, components=60, origins=2)
    plugin = execution_plugin("M0")
    value = spec()
    value.update(study_id="selection-consumer", arms=deepcopy(training_spec["arms"]))
    value["cells"][0].update(block_id=document["block_id"], plugin_id=plugin.plugin_id,
        capability="generic-rollout", seed=config["seed"], visibility="synthetic")
    job = {"schema_version":"pirc26-worker-job-v1", "operation":"forecast",
        "initial_checkpoint":fitted["checkpoint"], "o1_result":None, "batch_size":4, "origins":recipes}
    binding = {"schema_version":BINDING_SCHEMA, "preparation_ref":prep_ref, "independent_block_id":"selection-unit",
        "population_hash":digest(selected["provenance"]), "model_ref":model_ref,
        "model_authorization":{"authorization_id":disclosure["authorization_id"], "authorization_version":None}}
    grant = admit_fixture(store, value, plugin, tmp_path.absolute(), execution_config=config,
        execution_inputs={**profile, "runtime_root":str(tmp_path / "ledger"), "store_id":store.store_id},
        execution_components=component_bindings(config, profile, matrix_cells=1, registries=plugin.component_registries),
        recovery_command_builder=resume_command, input_content=encode(document),
        input_block_metadata=selected["protocol_entry"], input_data_root=selected["data_root"], split_role="selection",
        protocol_id="owned-selection", authorization_id="owned-selection-grant",
        package_payload={"pirc26_job":job, "pirc26_selection":binding})
    store.register(value, digest(value))
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(plugin)
    recovery.register(recovery_plugin("M0"))
    return store, value, plugin, registry, recovery, grant, fitted, model, requests


def test_actual_owned_selection_uses_only_fitted_attachment_and_entire_original_unit(tmp_path, contracts, monkeypatch):
    fixture = selection_fixture(tmp_path, contracts)
    store, value, plugin, registry, recovery, grant, fitted, model, requests = fixture
    runner = SharedRunner(store, registry, recovery)
    attempts_before = store.attempts()
    balance_before = BudgetLedger(store).balance("affine")
    events_before = store.events()
    outcome = runner.run_cell(value["study_id"], digest(value["cells"][0]), budget=BudgetSpec(60, category="smoke"))
    assert outcome["state"] == "SUCCEEDED", outcome
    attempt = store.attempts()[outcome["attempt_id"]]
    result = json.loads(store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant))
    fit = result["fit"]
    assert fit["status"] == "FROZEN" and fit["checkpoint"] == fitted["checkpoint"]
    assert not {"history", "steps", "objective", "training_population"} & set(fit)
    evidence = fit["selection_input"]
    assert evidence["independent_block_id"] == "selection-unit" and evidence["observations"] == 10
    assert len(evidence["segment_ids"]) == 2 and evidence["scientific_qualification"] == "not-established"
    assert evidence["use"] == "selection-only-not-final-evidence"
    for actual, request in zip(result["forecast"]["origins"], requests):
        expected = forecast(model, request)
        assert torch.equal(torch.tensor(actual["samples"], dtype=torch.float64), expected["samples"])
    assert result["source_identity"]["source_block_id"] == "selection-unit"
    assert len(store.attempts()) == len(attempts_before) + 1
    assert BudgetLedger(store).balance("affine")["committed_ms"] > balance_before["committed_ms"]
    journal = store.events()[len(events_before):]
    attachment = evidence["model_ref"]["artifact_id"]
    model_reads = [e for e in journal if e["event_kind"] == "READ_STARTED" and e["payload"].get("artifact_id") == attachment]
    assert len(model_reads) == 1
    reservation = next(e for e in journal if e["event_kind"] == "RESERVE" and e["payload"]["attempt_id"] == attempt["attempt_id"])
    started = next(e for e in journal if e["event_kind"] == "WORKER_STARTED" and e["payload"]["attempt_id"] == attempt["attempt_id"])
    settled = next(e for e in journal if e["event_kind"] == "SETTLE" and e["payload"]["attempt_id"] == attempt["attempt_id"])
    assert reservation["sequence"] < model_reads[0]["sequence"] < started["sequence"] < settled["sequence"]
    source_reads = [e for e in journal if e["event_kind"] == "READ_STARTED" and "block_id" in e["payload"]]
    assert source_reads and all(e["payload"]["block_id"] == value["cells"][0]["block_id"] for e in source_reads)
    receipt = receipt_for(store, attempt["attempt_id"])
    with monkeypatch.context() as patch:
        patch.setattr(store, "_verified_artifact_content", lambda *a,**k: pytest.fail("preflight read payload"))
        selected, expected = source(receipt, store=store)
        assert expected == evidence
        assert selected["protocol_entry"]["source_block_id"] == "selection-unit"
    before = store.events()
    for fault in ("fit", "formal", "membership", "wrong-model", "omit-segment", "small-profile", "arm", "visibility", "grant"):
        forged = deepcopy(receipt)
        job = forged["documents"]["package"]["payload"]["pirc26_job"]
        binding = forged["documents"]["package"]["payload"]["pirc26_selection"]
        if fault == "fit":
            job["operation"] = "fit-and-forecast"
        elif fault == "formal":
            forged["mode"] = "formal"
        elif fault == "membership":
            binding["population_hash"] = "f"*64
        elif fault == "wrong-model":
            binding["model_ref"]["receipt_hash"] = "f"*64
        elif fault == "omit-segment":
            job["origins"].pop()
        elif fault == "small-profile":
            forged["cell"]["execution"]["inputs"]["observations"] = 1
        elif fault == "arm":
            forged["spec"]["arms"][0]["budget_seconds"] -= 1
        elif fault == "visibility":
            forged["cell"]["visibility"] = "public"
        else:
            binding["model_authorization"]["authorization_id"] = grant["authorization_id"]
        with monkeypatch.context() as patch:
            patch.setattr(store, "_verified_artifact_content", lambda *a,**k: pytest.fail("denied preflight read payload"))
            with pytest.raises(ResearchError):
                source(forged, store=store)
    assert not any(e["event_kind"] == "READ_STARTED" for e in store.events()[len(before):])
    # Even a legacy derived artifact claiming a different visibility cannot
    # turn restricted geometry/model inputs into synthetic/public output.
    original_manifest = store._manifest
    artifact_id = selected["protocol_entry"]["path"]
    def restricted_selection(object_id):
        value = original_manifest(object_id)
        if object_id == "artifact-" + artifact_id:
            value = {**value, "visibility":"restricted"}
        return value
    with monkeypatch.context() as patch:
        patch.setattr(store, "_manifest", restricted_selection)
        patch.setattr(store, "_verified_artifact_content", lambda *a,**k: pytest.fail("visibility denial read payload"))
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            source(receipt, store=store)
    for fault in ("checkpoint", "unit", "refit"):
        changed = deepcopy(result)
        if fault == "checkpoint":
            changed["fit"]["checkpoint"]["sha256"] = "f"*64
        elif fault == "unit":
            changed["fit"]["selection_input"]["independent_block_id"] = "invented-unit"
        else:
            changed["fit"]["steps"] = 1
        with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
            result_validate(receipt, changed, store=store)
    balance, attempts = BudgetLedger(store).balance("affine"), store.attempts()
    reused = runner.run_cell(value["study_id"], digest(value["cells"][0]), budget=BudgetSpec(60, category="smoke"))
    assert reused["reused"] and store.attempts() == attempts and BudgetLedger(store).balance("affine") == balance
