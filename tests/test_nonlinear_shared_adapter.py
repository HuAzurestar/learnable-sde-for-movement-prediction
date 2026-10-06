"""Explicit synthetic managed adapters; no trained data or official ledger."""

import json
import math
import time

import pytest

from application.propagation_execution import execute_propagation, request_from_manifest
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_recovery import RecoveryRegistry, SharedRecovery
from application.research_registry import plan_resources
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.plugin import execution_config, propagation_plugin, propagation_recovery_plugin
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.test_propagation_shared_adapter import prepare


def _checkpoint_fixture_samples(calibration_seconds):
    # Size this engineering-only workload BEFORE registration/reservation. Fast
    # CI otherwise finishes before the old ten-second minimum can request save.
    # Keep actual work under the shared one-million-update declaration.
    return max(1024, min(1_000_000 // 128, math.ceil(16.0 * 128 / calibration_seconds)))


@pytest.mark.parametrize("seconds", [0.01, 0.5, 1.0, 4.0, 10.0])
def test_checkpoint_fixture_calibration_stays_within_the_frozen_work_quota(seconds):
    samples = _checkpoint_fixture_samples(seconds)
    assert 1024 <= samples and samples*128 <= 1_000_000


@pytest.mark.parametrize("method", ["euler", "heun", "reversible-heun", "mlmc", "importance", "cubature"])
def test_synthetic_method_has_shared_admission_cost_and_no_fabricated_oracle_error(tmp_path, method):
    store, spec, registry = prepare(tmp_path, method, synthetic=True)
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60, category="smoke"))
    assert outcome["state"] == "SUCCEEDED", outcome
    result = json.loads((store.path / "artifacts" / outcome["artifact_id"]).read_bytes())
    assert set(result["metrics"]) == {"functional_estimate"}
    assert result["source_schema"] == "synthetic-endpoint-propagation-v1"
    assert result["qualification"] == "fixture" and result["admission_hash"]
    errors = result["forecast"]["functional"]["error_budget"]
    assert errors["reference"]["value"] is None and errors["time_discretization"]["value"] is None
    if method == "cubature":
        assert errors["propagation_approximation"]["value"] is None
        assert result["forecast"]["functional"]["status"] == "APPROXIMATION_ONLY"
    charged = BudgetLedger(store).balance("synthetic-method")["committed_ms"]
    assert charged > 0
    reused = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell))
    assert reused["reused"] and reused["artifact_id"] == outcome["artifact_id"]
    assert BudgetLedger(store).balance("synthetic-method")["committed_ms"] == charged


def test_cubature_workspace_reserves_eight_points_even_when_path_chunk_is_one(tmp_path):
    store, spec, registry = prepare(tmp_path, "cubature", synthetic=True, changes={"chunk_size": 1})
    cell = spec["cells"][0]
    plan = plan_resources(propagation_plugin(synthetic=True).registry_entry, cell["execution"]["config"],
        cell["execution"]["inputs"], matrix_cells=1)
    assert plan["counts"]["observations"] == 8
    assert plan["counts"]["steps"] == 8*cell["propagation_request"]["steps"]
    assert any(tensor["name"] == "cubature-points" and tensor["shape"] == [8, 4] for tensor in plan["tensors"])
    assert "exact-transition" not in propagation_plugin(synthetic=True).capabilities
    request = request_from_manifest(cell["propagation_request"])
    for unsupported in ("exact", "gaussian"):
        with pytest.raises(ResearchError):
            execution_config(request, unsupported, synthetic=True)
    with pytest.raises(ResearchError):
        execution_config(request, "cubature", synthetic=True, recovery=True)


@pytest.mark.parametrize("method", ["euler", "reversible-heun"])
def test_real_nonlinear_chunk_worker_restores_actual_statistics_and_original_arm_cost(tmp_path, method):
    from domain.propagation import PropagationRequest
    from experiments.pirc27.nonlinear import nonlinear_package
    from experiments.pirc27.oracles import oracle_suite
    from inference.propagation_methods import monte_carlo
    from tests.test_propagation_shared_adapter import _worker_startup_seconds, _checkpoint_job_seconds
    case = oracle_suite()[0]
    package = nonlinear_package()
    calibration = PropagationRequest("timing-fixture", package.package_hash, case.initial_mean,
        case.initial_covariance, 0.0, 0.0, (1.0,), "endpoint-x", 11, "paired-root", "synthetic-method",
        samples=128, steps=128, chunk_size=1)
    started = time.monotonic()
    monte_carlo(package, calibration, solver=method)
    samples = _checkpoint_fixture_samples(time.monotonic()-started)
    startup_seconds = _worker_startup_seconds()
    store, spec, registry = prepare(tmp_path, method, synthetic=True, recovery=True,
        changes={"samples": samples, "steps": 128, "chunk_size": 1})
    cell = spec["cells"][0]
    started = time.monotonic()
    expected = execute_propagation(spec, cell)
    # Real owner deadlines and soft-save condition are unchanged. No fake
    # clocks, sleeps, live reservation extension, or weakened restore assertions.
    job_seconds = _checkpoint_job_seconds(time.monotonic()-started, startup_seconds)
    store.register(spec, digest(spec))
    adapters = RecoveryRegistry()
    adapters.register(propagation_recovery_plugin(synthetic=True))
    stopped = SharedRunner(store, registry, recovery_registry=adapters).run_cell(spec["study_id"], digest(cell),
        budget=BudgetSpec(job_seconds, category="smoke"))
    assert stopped["state"] == "FAILED" and store.attempts()[stopped["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED", stopped
    checkpoint = stopped["checkpoint"]
    assert 0 < checkpoint["progress"]["completed_steps"] < checkpoint["progress"]["total_steps"]
    before = BudgetLedger(store).balance("synthetic-method")
    assert before["committed_ms"] > 0 and not before["closed"]
    reopened = ResearchStore(tmp_path, "propagation-unit")
    grant = reopened.authorization(spec["admission"]["authorization_id"])
    resumed = SharedRecovery(reopened, registry, adapters).resume(stopped["attempt_id"], checkpoint["artifact_id"],
        authorization=grant, budget=BudgetSpec(120, category="smoke"))
    assert resumed["state"] == "SUCCEEDED", resumed
    result = json.loads((reopened.path / "artifacts" / resumed["artifact_id"]).read_bytes())
    assert result["forecast"] == json.loads(encode(expected["forecast"]))
    assert result["output_hash"] == expected["output_hash"] and result["metrics"] == expected["metrics"]
    assert reopened.attempts()[resumed["attempt_id"]]["parent_attempt_id"] == stopped["attempt_id"]
    assert BudgetLedger(reopened).balance("synthetic-method")["committed_ms"] > before["committed_ms"]
