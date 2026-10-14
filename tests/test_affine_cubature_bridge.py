"""Affine cubature composition and pilot evidence, never formal qualification."""

from dataclasses import replace
import json

import pytest

from application.propagation_execution import validate_propagation_cell
from application.research_budget import BudgetSpec
from domain.cubature_qualification import CubatureQualificationPolicy
from domain.errors import DataValidationError
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.cubature_plugin import cubature_plugin, cubature_config
from experiments.pirc27.design import StudyMethod, freeze_design
from experiments.pirc27.oracles import oracle_suite
from inference.cubature_qualification import analyze_affine_cubature
from inference.nonlinear_propagation import cubature_estimate
from infrastructure.research_store import ResearchError, digest
from tests.test_propagation_methods import inputs
from tests.test_propagation_shared_adapter import prepare
from tests.test_propagation_study_design import fixture


def policy_for(package, request, **changes):
    from experiments.pirc25.affine import code_hash
    return replace(CubatureQualificationPolicy(request.request_hash, package.package_hash,
        code_hash(), 1e-8, 1e-8, .5, 100., (10., 10., 1., 1.), 400001, 60.), **changes)


def test_affine_cubature_compiles_without_relabeling_the_frozen_model_or_budget():
    design = fixture(methods=(StudyMethod("cubature", samples=8, steps=4, chunk_size=1),),
        horizons=(1.,), seeds=(11,))
    frozen = freeze_design(design)
    row = frozen.manifest()["matrix"][0]
    assert row["disposition"] == "PLANNED"
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    package, request, config, plugin = validate_propagation_cell(spec, spec["cells"][0])
    assert package.manifest()["schema_version"] == "frozen-affine-dynamics-v1"
    assert package.package_hash == design.models[0].package.package_hash
    assert plugin.plugin_id == "affine-cubature" and plugin.resume_level == "restart-only"
    assert plugin.capabilities == frozenset({"generic-rollout"})
    assert config["work_steps"] == 8*request.steps and config["workspace_rows"] == 8
    assert spec["arms"][0]["budget_seconds"] == 86400
    assert spec["arms"][0]["method_family_id"] == "cubature"
    assert "admission" not in spec


def test_cubature_configuration_never_claims_chunk_recovery():
    _, request = inputs(chunk_size=1)
    config = cubature_config(request)
    assert config["workspace_rows"] == 8 and config["work_steps"] == 8*request.steps
    with pytest.raises(DataValidationError):
        freeze_design(fixture(methods=(StudyMethod("cubature", recovery=True),)))


def test_qualification_resource_proxy_counts_certifiers_and_both_output_replays():
    package, request = inputs(samples=2, steps=16)
    policy = policy_for(package, request)
    assert cubature_config(request, policy=policy)["work_steps"] == 16*16+policy.maximum_operations


def test_halfspace_cubature_is_bounded_as_a_probability_not_a_position_mean():
    package, request = inputs(samples=2, steps=8, functional="endpoint-halfspace", threshold=.6)
    result = cubature_estimate(package, request)
    analysis = analyze_affine_cubature(package, request, policy_for(package, request), result)
    assert analysis["status"] == "PASSED" and 0 <= result.estimate <= 1
    assert result.error_budget.reference.units == "1"
    assert analysis["functional_roundoff_upper"] <= 1e-8


@pytest.mark.parametrize("index", range(4))
def test_affine_cubature_actual_output_has_independent_discrete_and_continuous_bounds(index):
    case = oracle_suite()[index]
    _, base = inputs(samples=2, steps=16)
    request = replace(base, model_package_hash=case.package.package_hash,
        initial_mean=case.initial_mean, initial_covariance=case.initial_covariance)
    policy = policy_for(case.package, request)
    result = cubature_estimate(case.package, request)
    analysis = analyze_affine_cubature(case.package, request, policy, result)
    assert analysis["status"] == "PASSED" and all(analysis["checks"].values())
    assert analysis["scientific_qualification"] is False
    assert analysis["target_grid"] == {"solver": "euler", "steps": 16}
    assert analysis["cubature_point_updates"] == 128
    assert analysis["functional_roundoff_upper"] <= policy.maximum_functional_roundoff
    assert analysis["continuous_certificate_hash"] != analysis["target_certificate_hash"]
    assert analysis["model_error"] == {"value": None, "status": "NOT_IDENTIFIABLE"}
    assert result.status == "APPROXIMATION_ONLY"


def test_cubature_analysis_keeps_a_failed_time_bias_policy_without_refinement():
    package, request = inputs(samples=2, steps=2)
    policy = policy_for(package, request, maximum_time_bias=1e-20)
    analysis = analyze_affine_cubature(package, request, policy, cubature_estimate(package, request))
    assert analysis["status"] == "FAILED" and analysis["checks"]["time_bias"] is False
    assert analysis["target_grid"]["steps"] == 2


@pytest.mark.parametrize("field,value", [("code_hash", "0"*64), ("request_hash", "0"*64),
    ("maximum_reference_width", 0.), ("maximum_job_seconds", 1801.),
    ("maximum_operations", 400002), ("state_scales", (1., 1.)),
    ("schema_version", "affine-analytic-qualification-policy-v1")])
def test_cubature_policy_refuses_changed_bindings_and_unbounded_thresholds(field, value):
    from experiments.pirc25.affine import code_hash
    package, request = inputs(samples=2, steps=2)
    policy = policy_for(package, request, **{field: value})
    with pytest.raises(DataValidationError):
        policy.validate(package, request, code_hash())


def test_nonlinear_and_forged_estimates_cannot_inherit_an_affine_cube_proof():
    from experiments.pirc27.nonlinear import nonlinear_package
    from experiments.pirc25.affine import code_hash
    nonlinear = nonlinear_package()
    package, request = inputs(samples=2, steps=2)
    other = replace(request, model_package_hash=nonlinear.package_hash)
    with pytest.raises(DataValidationError):
        policy_for(nonlinear, other).validate(nonlinear, other, code_hash())
    actual = cubature_estimate(package, request)
    with pytest.raises(DataValidationError):
        analyze_affine_cubature(package, request, policy_for(package, request),
            replace(actual, estimate=actual.estimate+1.))


def test_actual_cubature_pilot_runs_under_original_arm_and_retains_bound_analysis(tmp_path):
    store, spec, registry = prepare(tmp_path, "cubature", cubature_qualification=True)
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    before = store.events()
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in before)
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell),
        budget=BudgetSpec(60., category="pilot"))
    assert result["state"] == "SUCCEEDED", result
    artifact = json.loads((store.path/"artifacts"/result["artifact_id"]).read_bytes())
    analysis = artifact["forecast"]["cubature_qualification_analysis"]
    assert analysis["status"] == "PASSED" and artifact["qualification"] == "fixture"
    assert artifact["forecast"]["functional"]["sample_count"] == 0
    assert artifact["forecast"]["functional"]["error_budget"]["reference"]["value"] is None
    events = store.events()
    assert any(e["event_kind"] == "READ_COMPLETED" for e in events)
    charges = [e["payload"] for e in events if e["event_kind"] == "SETTLE"]
    assert len(charges) == 1 and charges[0]["charged_ms"] > 0
    reused = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell))
    assert reused["reused"] and reused["artifact_id"] == result["artifact_id"]
    assert len([e for e in store.events() if e["event_kind"] == "SETTLE"]) == 1


@pytest.mark.parametrize("qualification", [False, True])
def test_formal_cubature_is_refused_before_any_protected_read_or_worker(tmp_path, qualification):
    store, spec, registry = prepare(tmp_path, "cubature", formal=True,
        cubature_qualification=qualification)
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="UNQUALIFIED|UNAUTHORIZED_DATA"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
            budget=BudgetSpec(60., category="pilot" if qualification else "job"))
    events = store.events()
    assert not any(e["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for e in events)


def test_cubature_resource_policy_is_checked_before_reservation(tmp_path):
    store, spec, registry = prepare(tmp_path, "cubature", cubature_qualification=True)
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
            budget=BudgetSpec(61., category="pilot"))
    assert not any(e["event_kind"] in {"RESERVE", "READ_STARTED", "WORKER_STARTED"} for e in store.events())


def test_downgrading_qualification_to_ordinary_eight_point_work_is_refused(tmp_path):
    from application.research_execution import execution_binding
    from experiments.pirc27.plugin import execution_inputs
    store, spec, _ = prepare(tmp_path, "cubature", cubature_qualification=True)
    cell = spec["cells"][0]
    package, request, config, plugin = validate_propagation_cell(spec, cell)
    changed = {**config, "work_steps": 8*request.steps}
    cell["execution"] = execution_binding(plugin.registry_entry, changed, execution_inputs(request), matrix_cells=1)
    with pytest.raises(ResearchError, match="UNQUALIFIED|CONTRACT_MISMATCH"):
        validate_propagation_cell(spec, cell)
    assert not any(e["event_kind"] in {"RESERVE", "READ_STARTED", "WORKER_STARTED"} for e in store.events())
