"""Bounded affine MLMC reference and actual shared pilot controls only."""

from dataclasses import asdict, replace
from fractions import Fraction
import json

import pytest

from application.propagation_execution import request_from_manifest, validate_propagation_cell
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from application.research_recovery import RecoveryRegistry
from application.research_registry import plan_resources
from domain.affine_mlmc_qualification import AffineMLMCReferencePolicy
from domain.errors import DataValidationError
from domain.frozen_dynamics import FrozenDynamicsPackage
from domain.mlmc_pilot import MLMCPilotPolicy
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.mlmc_qualification_plugin import (mlmc_qualification_plugin,
    mlmc_qualification_config, mlmc_qualification_recovery_plugin)
from experiments.pirc27.plugin import execution_inputs, propagation_resume_command
from inference.affine_mlmc_qualification import analyze_bounded_mlmc_pilot
from inference.affine_reference import _parse_interval
from inference.propagation_methods import mlmc_estimate
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_mlmc_pilot import policy_for
from tests.test_propagation_methods import inputs
from tests.test_propagation_shared_adapter import prepare


COUNTS = (8, 8, 8)


def reference_for(package, request, pilot, counts=COUNTS, **changes):
    # Explicit engineering fixture caps, no automatic research policy defaults.
    return replace(AffineMLMCReferencePolicy(request.request_hash, package.package_hash, code_hash(),
        pilot.policy_hash, counts, 1e-20, 10., (1., 1., 1., 1.), 400_001, 60.), **changes)


@pytest.fixture(scope="module")
def actual():
    package, request = inputs(samples=sum(COUNTS), steps=4)
    pilot = policy_for(request)
    reference = reference_for(package, request, pilot)
    result = mlmc_estimate(package, request, level_samples=COUNTS, phase=2, pilot=True)
    return package, request, pilot, reference, result


def test_real_empirical_levels_get_a_bounded_finest_grid_bias_not_a_float_difference(actual):
    package, request, pilot, reference, result = actual
    analysis = analyze_bounded_mlmc_pilot(package, request, result, pilot, reference)
    assert analysis["checks"]["resolved_reference"] and analysis["checks"]["finest_grid_bias"]
    assert analysis["finest_grid"] == {"solver": "euler", "base_steps": 4, "level_count": 3, "steps": 16}
    bias = _parse_interval(analysis["signed_finest_grid_bias_bounds"])
    cb = _parse_interval(analysis["continuous_certificate"]["functional_bounds"])
    fb = _parse_interval(analysis["finest_certificate"]["functional_bounds"])
    assert (bias.lo, bias.hi) == (fb.lo-cb.hi, fb.hi-cb.lo)
    assert analysis["reference_width_upper"] >= cb.hi-cb.lo
    assert analysis["finest_grid_bias_absolute_upper"] >= max(abs(bias.lo), abs(bias.hi))
    assert analysis["observed_pilot_error_upper_vs_continuous_law"] >= max(abs(Fraction(result.estimate)-cb.lo), abs(Fraction(result.estimate)-cb.hi))
    assert analysis["qualification"] == "UNQUALIFIED" and not analysis["scientific_qualification"]
    assert analysis["sampler_roundoff"] == analysis["model_error"] == {"value": None, "status": "NOT_IDENTIFIABLE"}
    assert analysis["sampling_error"]["status"] == "ESTIMATED"
    assert analysis["cost_status"] == "OWNER_SETTLEMENT_REQUIRED" and not analysis["automatic_execution"]
    assert analysis["empirical_pilot_analysis"]["reference_uncertainty"] is None  # Original evidence retained.
    assert analysis["analysis_hash"] == digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    assert len(encode(analysis)) <= 384*1024


@pytest.mark.parametrize("changes,check", [({"maximum_reference_width": 1e-100}, "reference_width"),
    ({"maximum_scaled_transition_norm": 1e-4}, "scaled_transition_growth"),
    ({"maximum_operations": 1}, "arithmetic_operations")])
def test_actual_bounded_numeric_checks_fail_without_loosening_thresholds(actual, changes, check):
    package, request, pilot, reference, result = actual
    analysis = analyze_bounded_mlmc_pilot(package, request, result, pilot, replace(reference, **changes))
    assert analysis["status"] == "NUMERICAL_FAILED" and analysis["checks"][check] is False
    assert analysis["qualification"] == "UNQUALIFIED"


def test_frozen_bias_tolerance_uses_absolute_interval_upper_not_float_or_last_level_mean(actual):
    package, request, pilot, reference, result = actual
    strict = replace(pilot, bias_tolerance=1e-100)
    reference = replace(reference, pilot_policy_hash=strict.policy_hash)
    analysis = analyze_bounded_mlmc_pilot(package, request, result, strict, reference)
    assert not analysis["checks"]["finest_grid_bias"] and analysis["status"] == "NUMERICAL_FAILED"


@pytest.mark.parametrize("changes", [{"request_hash": "0"*64}, {"model_package_hash": "0"*64},
    {"code_hash": "0"*64}, {"pilot_policy_hash": "0"*64}, {"maximum_job_seconds": 1801},
    {"maximum_reference_width": float("nan")}, {"maximum_scaled_transition_norm": 10**1000},
    {"state_scales": (1, 0, 1, 1)}, {"level_samples": [8, 8, 8]},
    {"maximum_operations": True}, {"maximum_operations": 400002}])
def test_unbound_or_unbounded_reference_policy_refuses(actual, changes):
    package, request, pilot, reference, _ = actual
    with pytest.raises(DataValidationError):
        replace(reference, **changes).validate(package, request, pilot, code_hash())


def test_policy_detaches_and_requires_complete_canonical_manifest(actual):
    _, _, _, reference, _ = actual
    value = reference.manifest()
    restored = AffineMLMCReferencePolicy.from_manifest(value)
    value["level_samples"][0] = 100
    assert restored == reference
    value = reference.manifest()
    value.pop("maximum_job_seconds")
    with pytest.raises(DataValidationError):
        AffineMLMCReferencePolicy.from_manifest(value)


@pytest.mark.parametrize("fault", ["scalar", "se", "interval", "interval-kind", "kind", "status", "phase", "sample-count"])
def test_reference_cannot_repair_false_sampler_identity_or_statistics(actual, fault):
    package, request, pilot, reference, result = actual
    if fault == "scalar":
        result = replace(result, estimate=result.estimate+1)
    elif fault == "se":
        result = replace(result, standard_error=0)
    elif fault == "interval":
        result = replace(result, interval=(0, 0))
    elif fault == "interval-kind":
        result = replace(result, interval_kind="exact-binomial-one-sided-95")
    elif fault == "kind":
        result = replace(result, kind="analytic")
    elif fault == "status":
        result = replace(result, status="SUCCEEDED")
    elif fault == "sample-count":
        result = replace(result, sample_count=result.sample_count+1)
    else:
        result = replace(result, diagnostics=tuple((k, 1 if k == "pilot_phase" else v) for k, v in result.diagnostics))
    with pytest.raises(DataValidationError):
        analyze_bounded_mlmc_pilot(package, request, result, pilot, reference)


def test_zero_observed_rare_corrections_remain_unresolved_even_with_certified_reference():
    package, request = inputs(samples=sum(COUNTS), steps=4, functional="endpoint-halfspace", threshold=100)
    pilot = policy_for(request)
    result = mlmc_estimate(package, request, level_samples=COUNTS, phase=2, pilot=True)
    analysis = analyze_bounded_mlmc_pilot(package, request, result, pilot, reference_for(package, request, pilot))
    assert analysis["checks"]["resolved_reference"]
    assert not analysis["checks"]["empirical_sampling_and_cost"] and analysis["status"] == "NUMERICAL_FAILED"
    assert "SAMPLING_UNRESOLVED" in analysis["empirical_failures"]
    assert analysis["sampling_error"]["value"] is None
    assert analysis["empirical_pilot_analysis"]["sampling_only_proposal"] is None


def managed(tmp_path, *, reference_changes=None, bias_tolerance=1., sampling_tolerance=1.,
            counts=(64, 64, 64), request_changes=None, formal=False):
    store, spec, _ = prepare(tmp_path, "mlmc-pilot", recovery=True,
        changes={"samples": sum(counts), **(request_changes or {})}, level_samples=counts, formal=formal)
    cell = spec["cells"][0]
    request = request_from_manifest(cell["propagation_request"])
    package = FrozenDynamicsPackage.from_manifest(cell["frozen_dynamics"], expected_hash=request.model_package_hash)
    pilot = replace(MLMCPilotPolicy(**cell["mlmc_pilot_policy"]), bias_tolerance=bias_tolerance,
        sampling_tolerance=sampling_tolerance)
    reference = reference_for(package, request, pilot, counts, **(reference_changes or {}))
    plugin = mlmc_qualification_plugin()
    config, parameters = mlmc_qualification_config(request, reference), execution_inputs(request)
    cell.update(plugin_id=plugin.plugin_id, mlmc_pilot_policy=asdict(pilot), affine_mlmc_reference_policy=reference.manifest(),
        execution=execution_binding(plugin.registry_entry, config, parameters, matrix_cells=1))
    admit_fixture(store, spec, plugin, tmp_path, execution_config=config, execution_inputs=parameters,
        legacy_upstream=False, recovery_command_builder=propagation_resume_command, fixture_prefix="bounded-pilot-", formal=formal)
    spec["admission"]["mode"] = "pilot"
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(plugin)
    recovery.register(mlmc_qualification_recovery_plugin())
    return store, spec, registry, recovery


def test_resource_plan_keeps_physical_dimension_four_and_reserves_fraction_pools(tmp_path):
    _, spec, _, _ = managed(tmp_path)
    cell = spec["cells"][0]
    _, _, config, plugin = validate_propagation_cell(spec, cell)
    plan = plan_resources(plugin.registry_entry, config, cell["execution"]["inputs"], matrix_cells=1)
    pools = [t for t in plan["tensors"] if "-pool-" in t["name"]]
    assert len(pools) == 64 and sum(t["bytes"] for t in pools) == 64*1024*1024
    assert plan["counts"]["state_dim"] == 4 and plugin.resume_level == "chunk"


@pytest.mark.parametrize("bias_tolerance,expected", [(1., "NUMERICAL_READY"), (1e-100, "NUMERICAL_FAILED")])
def test_actual_shared_worker_computes_bounded_pilot_under_same_original_arm(tmp_path, bias_tolerance, expected):
    store, spec, registry, recovery = managed(tmp_path, bias_tolerance=bias_tolerance)
    store.register(spec, digest(spec))
    runner = SharedRunner(store, registry, recovery_registry=recovery)
    outcome = runner.run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60, category="pilot"))
    assert outcome["state"] == "SUCCEEDED", outcome
    result = json.loads((store.path/"artifacts"/outcome["artifact_id"]).read_bytes())
    analysis = result["forecast"]["bounded_mlmc_pilot_analysis"]
    assert analysis["status"] == expected and analysis["qualification"] == "UNQUALIFIED"
    assert result["qualification"] == "fixture" and result["forecast"]["functional"]["status"] == "PILOT_ONLY"
    assert analysis["finest_grid"]["steps"] == 16 and spec["arms"][0]["method_family_id"] == "mlmc"
    balance = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])
    assert 0 < balance["committed_ms"] <= 60000
    events = store.events()
    stopped = [e for e in events if e["event_kind"] == "WORKER_TREE_STOPPED"]
    settled = [e for e in events if e["event_kind"] == "SETTLE"]
    assert len(stopped) == len(settled) == 1 and stopped[0]["sequence"] < settled[0]["sequence"]
    assert settled[0]["payload"]["charged_ms"] == balance["committed_ms"] == stopped[0]["payload"]["observed_elapsed_ms"]
    reused = runner.run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60, category="pilot"))
    assert reused["reused"] and BudgetLedger(store).balance(spec["arms"][0]["arm_id"]) == balance


@pytest.mark.parametrize("budget", [BudgetSpec(60, category="smoke"), BudgetSpec(61, category="pilot")])
def test_wrong_job_stage_or_cap_refuses_before_reservation_and_read(tmp_path, budget):
    store, spec, registry, recovery = managed(tmp_path)
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="frozen pilot job budget"):
        SharedRunner(store, registry, recovery_registry=recovery).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=budget)
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "READ_STARTED"} for e in store.events())


def test_reference_pilot_rejects_heldout_block_even_with_a_synthetic_test_grant(tmp_path, monkeypatch):
    store, spec, registry, recovery = managed(tmp_path, formal=True)
    store.register(spec, digest(spec))
    def forbidden(*args, **kwargs):
        pytest.fail("qualification pilot read heldout artifact/input")
    monkeypatch.setattr(store, "read_artifact", forbidden)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        SharedRunner(store, registry, recovery_registry=recovery).run_cell(spec["study_id"], digest(spec["cells"][0]),
            budget=BudgetSpec(60, category="pilot"))
    assert not any(e["event_kind"] in {"WORKER_STARTED", "READ_STARTED", "EXPOSURE_ALLOWED"} for e in store.events())


def test_nilpotent_acceleration_bias_matches_independent_closed_form_at_finest_level():
    from experiments.pirc27.oracles import affine_package
    from tests.test_propagation_oracles import integrated_brownian
    base = integrated_brownian().manifest()["parameters"]
    package = affine_package("unit-accelerated-brownian", base["A"], [0, 0, 0.5, 0], base["L"])
    _, request = inputs(samples=sum(COUNTS), steps=4)
    request = replace(request, model_package_hash=package.package_hash, horizons=(2.,))
    pilot = policy_for(request)
    result = mlmc_estimate(package, request, level_samples=COUNTS, phase=2, pilot=True)
    analysis = analyze_bounded_mlmc_pilot(package, request, result, pilot, reference_for(package, request, pilot))
    bias = _parse_interval(analysis["signed_finest_grid_bias_bounds"])
    # Sum deterministic Euler acceleration increments vs integral of b*t.
    # Finest grid has16 steps, not base4, even though request hash stays base-bound.
    expected = -Fraction(1, 2)*Fraction(0.5)*Fraction(2)**2/16
    assert bias.lo == bias.hi == expected
    assert analysis["finest_grid_bias_absolute_upper"] == float(abs(expected))


@pytest.mark.parametrize("fault", ["missing-policy", "wrong-policy", "wrong-role", "formal", "wrong-family", "reused-seed"])
def test_malformed_unfrozen_or_nonpilot_reference_cell_refuses_before_input(tmp_path, monkeypatch, fault):
    store, spec, registry, recovery = managed(tmp_path)
    cell = spec["cells"][0]
    if fault == "missing-policy":
        cell.pop("affine_mlmc_reference_policy")
    elif fault == "wrong-policy":
        cell["affine_mlmc_reference_policy"]["request_hash"] = "0"*64
    elif fault == "wrong-role":
        cell["execution_role"] = "production"
    elif fault == "formal":
        spec["admission"]["mode"] = "formal"
    elif fault == "wrong-family":
        spec["arms"][0]["method_family_id"] = "new-pilot"
    else:
        cell["mlmc_pilot_policy"]["production_seed"] = cell["seed"]
    store.register(spec, digest(spec))
    def forbidden(*args, **kwargs):
        pytest.fail("unqualified pilot read artifacts/input")
    monkeypatch.setattr(store, "read_artifact", forbidden)
    with pytest.raises(ResearchError):
        SharedRunner(store, registry, recovery_registry=recovery).run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60, category="pilot"))
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "READ_STARTED"} for e in store.events())
