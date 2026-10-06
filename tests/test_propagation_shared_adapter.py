"""Synthetic integration with real admission, process and shared budget APIs."""

from dataclasses import asdict
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from domain.propagation import PropagationRequest
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.oracles import oracle_suite
from experiments.pirc27.plugin import propagation_plugin, execution_config, execution_inputs
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture


def prepare(tmp_path, method="euler"):
    # A disposable test store, not a new scientific ledger or protected input.
    store = ResearchStore(tmp_path, "propagation-unit", initialize=True)
    case = oracle_suite()[0]
    request = PropagationRequest("endpoint-fixture", case.package.package_hash, case.initial_mean,
        case.initial_covariance, 0.0, 0.0, (1.0,), "endpoint-halfspace" if method == "importance" else "endpoint-x", 11, "paired-root", "affine-method",
        samples=16, steps=4, chunk_size=8)
    plugin = propagation_plugin()
    config = execution_config(request, method, level_samples=(8, 8) if method == "mlmc" else (),
                              proposal=(1.0, 0.0) if method == "importance" else (0.0, 0.0))
    inputs = execution_inputs(request)
    cell = {"arm_id": request.arm_id, "block_id": "generator-v1", "seed": request.seed, "horizon": 1.0,
            "plugin_id": plugin.plugin_id, "capability": {"mlmc": "coupled-level", "exact": "exact-transition", "importance": "rare-event"}.get(method, "generic-rollout"),
            "visibility": "synthetic", "frozen_dynamics": case.package.manifest(),
            "propagation_request": json.loads(encode(asdict(request)))}
    cell["resource_class"] = "cpu"
    cell["execution"] = execution_binding(plugin.registry_entry, config, inputs, matrix_cells=1)
    spec = {"schema_version": "pirc25-contract-v1", "study_id": "propagation-unit", "experiment_id": "oracle-unit",
        "comparison_family": "synthetic-engineering", "code_hash": code_hash(),
        "protocol_hash": digest("placeholder"), "data_hash": digest("placeholder"),
        "feature_hash": digest("no-terrain"), "selection_hash": digest("none"),
        "arms": [{"arm_id": request.arm_id, "model_family_id": "affine-stable-v1",
                  "method_family_id": method, "objective_id": request.functional, "budget_seconds": 86400}],
        "cells": [cell], "runtime_binding": {"root": str(tmp_path.resolve()), "store_id": store.store_id}}
    admit_fixture(store, spec, plugin, tmp_path, execution_config=config, execution_inputs=inputs, legacy_upstream=False)
    registry = CapabilityRegistry()
    registry.register(plugin)
    return store, spec, registry


@pytest.mark.parametrize("method", ["exact", "euler", "heun", "gaussian", "mlmc", "importance"])
def test_method_uses_shared_supervisor_and_retains_error_and_budget_provenance(tmp_path, method):
    store, spec, registry = prepare(tmp_path, method)
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60, category="smoke"))
    assert result["state"] == "SUCCEEDED", result
    artifact = json.loads((store.path / "artifacts" / result["artifact_id"]).read_bytes())
    functional = artifact["forecast"]["functional"]
    assert functional["error_budget"]["model"]["value"] is None
    assert artifact["qualification"] == "fixture" and artifact["resume_level"] == "restart-only"
    assert artifact["admission_hash"]
    events = store.events()
    settlements = [event for event in events if event["event_kind"] == "SETTLE"]
    assert len(settlements) == 1 and settlements[0]["payload"]["charged_ms"] > 0
    charged = BudgetLedger(store).balance("affine-method")["committed_ms"]
    reused = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert reused["reused"] and reused["artifact_id"] == result["artifact_id"]
    assert BudgetLedger(store).balance("affine-method")["committed_ms"] == charged


def test_plugin_counts_and_hashes_are_checked_before_worker_allocation(tmp_path):
    store, spec, registry = prepare(tmp_path)
    spec["cells"][0]["execution"]["config"]["samples"] = 1_000_001
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert not any(event["event_kind"] == "WORKER_STARTED" for event in store.events())


def test_missing_execution_grant_does_not_run_the_numerical_method(tmp_path):
    store, spec, registry = prepare(tmp_path)
    spec["admission"]["authorization_id"] = "unavailable-grant"
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert not any(event["event_kind"] == "WORKER_STARTED" for event in store.events())


def test_closed_shared_arm_refuses_plugin_before_any_worker_can_start(tmp_path):
    store, spec, registry = prepare(tmp_path)
    store.register(spec, digest(spec))
    store.append("ARM_CLOSED", {"arm_id": "affine-method", "reason": "synthetic timeout injection"})
    with pytest.raises(ResearchError, match="BUDGET_EXHAUSTED"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert not any(event["event_kind"] == "WORKER_STARTED" for event in store.events())
    assert next(iter(store.attempts().values()))["state"] == "BUDGET_EXHAUSTED"
