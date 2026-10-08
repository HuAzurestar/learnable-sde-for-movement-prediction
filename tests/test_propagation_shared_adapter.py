"""Synthetic integration with real admission, process and shared budget APIs."""

from dataclasses import asdict, replace
import json
import math
import subprocess
import sys
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from domain.propagation import PropagationRequest
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.oracles import oracle_suite
from experiments.pirc27.plugin import (propagation_plugin, execution_config, execution_inputs,
    propagation_resume_command, propagation_recovery_plugin)
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture


def _mlmc_checkpoint_allocation(calibration_seconds):
    # Engineering-only sizing before creating/registering/reserving the store.
    # Three levels at 128 base steps cost 2048 real fine+coarse updates per unit.
    units = max(16, min(1_000_000 // 2048, math.ceil(16.0 * 16 / calibration_seconds)))
    return (4*units, 2*units, units)


def _path_checkpoint_samples(calibration_seconds):
    # The same pre-store sizing for each actual method, not a budget extension.
    return max(1024, min(1_000_000 // 128, math.ceil(16.0 * 128 / calibration_seconds)))


def _worker_startup_seconds():
    # Read-only engineering calibration before a store/grant/reservation exists.
    # Parent imports are warm; a real fresh worker must import these packages.
    started = time.monotonic()
    subprocess.run([sys.executable, "-c", "from experiments.pirc27.worker import execute_propagation; "
        "from inference.propagation_methods import mlmc_estimate; "
        "from experiments.pirc27.oracles import oracle_suite; oracle_suite()"],
        check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return time.monotonic()-started


def _checkpoint_job_seconds(compute_seconds, startup_seconds):
    # One predeclared job, never a live deadline extension or save grace period.
    return max(3., min(20., .5*compute_seconds+startup_seconds))


@pytest.mark.parametrize("compute,startup,expected", [(12., 6., 12.), (16., .5, 8.5),
    (1., .1, 3.), (100., 10., 20.)])
def test_checkpoint_job_sizing_includes_fresh_startup_with_existing_fixture_cap(compute, startup, expected):
    assert _checkpoint_job_seconds(compute, startup) == expected


@pytest.mark.parametrize("seconds", [0.01, 0.5, 1.0, 4.0, 10.0])
def test_path_checkpoint_fixture_calibration_preserves_total_work_quota(seconds):
    samples = _path_checkpoint_samples(seconds)
    assert 1024 <= samples <= 1_000_000 // 128
    assert samples*128 <= 1_000_000


@pytest.mark.parametrize("seconds", [0.01, 0.5, 1.0, 4.0, 10.0])
def test_mlmc_checkpoint_fixture_calibration_preserves_all_levels_and_work_quota(seconds):
    allocation = _mlmc_checkpoint_allocation(seconds)
    assert len(allocation) == 3 and min(allocation) >= 16
    work = sum(n*(128*2**level + (128*2**(level-1) if level else 0))
               for level, n in enumerate(allocation))
    assert work <= 1_000_000 and sum(allocation) <= 1_000_000


def prepare(tmp_path, method="euler", *, recovery=False, changes=None, level_samples=None, unbound_steps=None,
            synthetic=False, formal=False, pilot_changes=None, cubature_qualification=False):
    # A disposable test store, not a new scientific ledger or protected input.
    store = ResearchStore(tmp_path, "propagation-unit", initialize=True)
    case = oracle_suite()[0]
    from experiments.pirc27.nonlinear import nonlinear_package
    package = nonlinear_package() if synthetic else case.package
    request = PropagationRequest("endpoint-fixture", package.package_hash, case.initial_mean,
        case.initial_covariance, 0.0, 0.0, (1.0,), "endpoint-halfspace" if method == "importance" else "endpoint-x", 11, "paired-root", "synthetic-method" if synthetic else "affine-method",
        samples=16, steps=4, chunk_size=8)
    request = replace(request, **(changes or {}))
    if method == "cubature" and not synthetic:
        from experiments.pirc27.cubature_plugin import cubature_plugin, cubature_config
        plugin = cubature_plugin(qualification=cubature_qualification)
        policy = None
        if cubature_qualification:
            from domain.cubature_qualification import CubatureQualificationPolicy
            policy = CubatureQualificationPolicy(request.request_hash, package.package_hash,
                code_hash(), 1e-8, 1e-8, .5, 100., (10., 10., 1., 1.), 400001, 60.)
        config = cubature_config(request, policy=policy)
    else:
        plugin = propagation_plugin(recovery=recovery, synthetic=synthetic)
        config = execution_config(request, method, level_samples=(level_samples or (8, 8)) if method in {"mlmc", "mlmc-pilot"} else (),
                                  proposal=(1.0, 0.0) if method == "importance" else (0.0, 0.0), recovery=recovery, synthetic=synthetic)
    inputs = execution_inputs(request)
    cell = {"arm_id": request.arm_id, "block_id": "generator-v1", "seed": request.seed, "horizon": 1.0,
            "plugin_id": plugin.plugin_id, "capability": {"mlmc": "coupled-level", "mlmc-pilot": "coupled-level", "exact": "exact-transition", "importance": "rare-event"}.get(method, "generic-rollout"),
            "visibility": "synthetic", "frozen_dynamics": package.manifest(),
            "propagation_request": json.loads(encode(asdict(request)))}
    if unbound_steps is not None:
        # Register this malformed pairing as the original matrix, so the test
        # reaches method accounting rather than an earlier snapshot hash veto.
        cell["propagation_request"]["steps"] = unbound_steps
    cell["resource_class"] = "cpu"
    cell["execution"] = execution_binding(plugin.registry_entry, config, inputs, matrix_cells=1)
    if cubature_qualification:
        cell.update(execution_role="qualification", cubature_qualification_policy=policy.manifest())
    if method == "mlmc-pilot":
        from tests.test_mlmc_pilot import policy_for
        cell.update(study_role="secondary", execution_role="pilot", mlmc_pilot_policy=asdict(policy_for(request,
            maximum_samples=1_000_000, maximum_work_steps=1_000_000)))
        cell.update(pilot_changes or {})
    spec = {"schema_version": "pirc25-contract-v1", "study_id": "propagation-unit", "experiment_id": "oracle-unit",
        "comparison_family": "synthetic-engineering", "code_hash": code_hash(),
        "protocol_hash": digest("placeholder"), "data_hash": digest("placeholder"),
        "feature_hash": digest("no-terrain"), "selection_hash": digest("none"),
        "arms": [{"arm_id": request.arm_id, "model_family_id": "tanh-stress-v1" if synthetic else "affine-stable-v1",
                  "method_family_id": "mlmc" if method == "mlmc-pilot" else method,
                  "objective_id": request.functional, "budget_seconds": 86400}],
        "cells": [cell], "runtime_binding": {"root": str(tmp_path.resolve()), "store_id": store.store_id}}
    admit_fixture(store, spec, plugin, tmp_path, formal=formal, execution_config=config, execution_inputs=inputs, legacy_upstream=False,
        recovery_command_builder=propagation_resume_command if recovery else None)
    if method == "mlmc-pilot":
        spec["admission"]["mode"] = "pilot"
    if cubature_qualification and not formal:
        spec["admission"]["mode"] = "pilot"
    registry = CapabilityRegistry()
    registry.register(plugin)
    return store, spec, registry


@pytest.mark.parametrize("method", ["exact", "euler", "heun", "reversible-heun", "gaussian", "mlmc", "importance", "cubature"])
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


@pytest.mark.parametrize("method", ["euler", "heun", "reversible-heun", "mlmc", "importance"])
def test_actual_chunk_worker_stops_and_reopened_resume_preserves_computation_and_cost(tmp_path, method):
    from application.propagation_execution import execute_propagation
    from application.research_recovery import RecoveryRegistry, SharedRecovery
    # Real numerical work, with tiny chunks to make the owner's real 80% signal
    # reachable at a completed boundary. No fake clock, sleep-only worker,
    # budget extension, forced checkpoint file or state-only roundtrip.
    allocation = None
    samples = 2048
    if method == "mlmc":
        from inference.propagation_methods import mlmc_estimate
        case = oracle_suite()[0]
        calibration = PropagationRequest("timing-fixture", case.package.package_hash, case.initial_mean,
            case.initial_covariance, 0.0, 0.0, (1.0,), "endpoint-x", 11, "paired-root", "affine-method",
            samples=112, steps=128, chunk_size=1)
        started = time.monotonic()
        mlmc_estimate(case.package, calibration, level_samples=(64, 32, 16))
        allocation = _mlmc_checkpoint_allocation(time.monotonic()-started)
    else:
        from inference.propagation_methods import monte_carlo, importance_sampling
        case = oracle_suite()[0]
        calibration = PropagationRequest("timing-fixture", case.package.package_hash, case.initial_mean,
            case.initial_covariance, 0.0, 0.0, (1.0,), "endpoint-halfspace" if method == "importance" else "endpoint-x",
            11, "paired-root", "affine-method", samples=128, steps=128, chunk_size=1)
        started = time.monotonic()
        if method == "importance":
            importance_sampling(case.package, calibration, proposal=(1.0, 0.0))
        else:
            monte_carlo(case.package, calibration, solver={"heun": "additive-heun", "reversible-heun": "reversible-heun"}.get(method, "euler"))
        samples = _path_checkpoint_samples(time.monotonic()-started)
    changes = {"samples": sum(allocation) if allocation else samples, "steps": 128, "chunk_size": 1}
    startup_seconds = _worker_startup_seconds()
    store, spec, registry = prepare(tmp_path, method, recovery=True, changes=changes,
        level_samples=allocation)
    cell = spec["cells"][0]
    started = time.monotonic()
    expected = execute_propagation(spec, cell)
    # Calibrate only this synthetic test's job duration, not a research arm.
    # Faster hosts must still exercise a real save before workload completion;
    # no clock or deadline is altered after the reservation is made.
    job_seconds = _checkpoint_job_seconds(time.monotonic()-started, startup_seconds)
    store.register(spec, digest(spec))
    adapters = RecoveryRegistry()
    adapters.register(propagation_recovery_plugin())
    interrupted = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
        spec["study_id"], digest(cell), budget=BudgetSpec(job_seconds, category="smoke"))
    assert interrupted["state"] == "FAILED", interrupted
    assert store.attempts()[interrupted["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED"
    saved = interrupted["checkpoint"]
    assert saved["resume_level"] == "chunk"
    assert 0 < saved["progress"]["completed_steps"] < saved["progress"]["total_steps"]
    assert saved["progress"]["total_steps"] == cell["execution"]["config"]["work_steps"]
    before = BudgetLedger(store).balance("affine-method")
    assert before["committed_ms"] > 0 and not before["closed"]
    reopened = ResearchStore(tmp_path, "propagation-unit")
    grant = reopened.manifest("authorization-" + spec["admission"]["authorization_id"])
    resumed = SharedRecovery(reopened, registry, adapters).resume(interrupted["attempt_id"], saved["artifact_id"],
        authorization=grant, budget=BudgetSpec(60, category="smoke"))
    assert resumed["state"] == "SUCCEEDED", resumed
    actual = json.loads((reopened.path / "artifacts" / resumed["artifact_id"]).read_bytes())
    assert actual["forecast"] == json.loads(encode(expected["forecast"]))
    assert actual["metrics"] == expected["metrics"] and actual["output_hash"] == expected["output_hash"]
    after = BudgetLedger(reopened).balance("affine-method")
    assert after["committed_ms"] > before["committed_ms"] and not after["closed"]
    assert reopened.attempts()[resumed["attempt_id"]]["parent_attempt_id"] == interrupted["attempt_id"]
    assert actual["resume_level"] == "chunk" and actual["admission_hash"]


def test_chunk_resource_plan_refuses_total_work_not_just_finest_grid(tmp_path):
    from application.propagation_execution import request_from_manifest
    from application.research_registry import plan_resources
    store, spec, registry = prepare(tmp_path, recovery=True)
    plugin = propagation_plugin(recovery=True)
    config = dict(spec["cells"][0]["execution"]["config"], work_steps=1_000_001)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        plan_resources(plugin.registry_entry, config, spec["cells"][0]["execution"]["inputs"], matrix_cells=1)
    # Every individual bound is legal; the true product is not. Refuse it when
    # building the immutable config rather than understating admitted progress.
    request = replace(request_from_manifest(spec["cells"][0]["propagation_request"]), samples=1024, steps=8192)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        execution_config(request, "euler", recovery=True)
    assert not any(event["event_kind"] == "WORKER_STARTED" for event in store.events())


@pytest.mark.parametrize("recovery", [False, True])
def test_request_cannot_understate_registered_work_before_process_launch(tmp_path, recovery):
    store, spec, registry = prepare(tmp_path, recovery=recovery, unbound_steps=8192)
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert not any(event["event_kind"] == "WORKER_STARTED" for event in store.events())
