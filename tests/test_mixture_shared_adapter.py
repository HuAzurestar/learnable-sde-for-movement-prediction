"""Disposable shared-worker controls, never an official research ledger."""

from dataclasses import replace
import json

import pytest

from application.propagation_execution import request_from_manifest, validate_propagation_cell
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_plan
from domain.mixture import MixturePolicy, MixtureSettings
from domain.errors import DataValidationError
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.design import StudyMethod, freeze_design
from experiments.pirc27.mixture_plugin import mixture_config, mixture_plugin, mixture_recovery_plugin
from experiments.pirc27.plugin import execution_inputs, propagation_resume_command
from infrastructure.research_store import ResearchError, digest
from tests.research_admission_fixtures import admit_fixture
from tests.test_propagation_shared_adapter import prepare
from tests.test_propagation_study_design import fixture


def prepared(root, *, synthetic=False, formal=False, maximum_job_seconds=60):
    changes = ({"steps": 1, "initial_mean": (0.,)*4,
        "initial_covariance": ((.25, 0., 0., 0.), (0.,)*4, (0.,)*4, (0.,)*4)} if synthetic else {})
    store, spec, _ = prepare(root, recovery=True, synthetic=synthetic, changes=changes)
    cell = spec["cells"][0]
    request = request_from_manifest(cell["propagation_request"])
    plugin = mixture_plugin(synthetic=synthetic)
    policy = MixturePolicy(request.request_hash, request.model_package_hash, code_hash(),
        8 if synthetic else 1, 0., 0., (1.,)*4, 0., 1_000_000, maximum_job_seconds)
    cell.update(plugin_id=plugin.plugin_id, mixture_policy=policy.manifest())
    spec["arms"][0]["method_family_id"] = "mixture"
    config = mixture_config(request, policy, synthetic=synthetic)
    admit_fixture(store, spec, plugin, root, formal=formal, execution_config=config,
        execution_inputs=execution_inputs(request), recovery_command_builder=propagation_resume_command,
        legacy_upstream=False, fixture_prefix="mixture-")
    registry = CapabilityRegistry()
    registry.register(plugin)
    return store, spec, registry


@pytest.mark.parametrize("synthetic", [False, True])
def test_actual_mixture_worker_uses_shared_owner_cost_and_reuse_without_qualification(tmp_path, synthetic):
    store, spec, registry = prepared(tmp_path, synthetic=synthetic)
    cell = spec["cells"][0]
    package, request, config, plugin = validate_propagation_cell(spec, cell)
    plan = execution_plan(spec, cell, plugin)
    assert plan["counts"]["state_dim"] == 4
    assert plan["counts"]["paths"] == 1
    assert plan["counts"]["components"] == config["component_cap"]
    assert plan["counts"]["observations"] == 8*config["component_cap"]
    store.register(spec, digest(spec))
    runner = SharedRunner(store, registry)
    outcome = runner.run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60, category="smoke"))
    assert outcome["state"] == "SUCCEEDED", outcome
    artifact = json.loads((store.path / "artifacts" / outcome["artifact_id"]).read_bytes())
    functional = artifact["forecast"]["functional"]
    assert artifact["qualification"] == "fixture" and artifact["resume_level"] == "chunk"
    assert functional["status"] == "APPROXIMATION_ONLY"
    assert functional["error_budget"]["propagation_approximation"]["value"] is None
    assert functional["sample_count"] == 0 and functional["interval"] is None
    assert artifact["admission_hash"]
    charged = BudgetLedger(store).balance(request.arm_id)["committed_ms"]
    assert charged > 0
    reused = runner.run_cell(spec["study_id"], digest(cell))
    assert reused["reused"] and reused["artifact_id"] == outcome["artifact_id"]
    assert BudgetLedger(store).balance(request.arm_id)["committed_ms"] == charged
    recovery = mixture_recovery_plugin(synthetic=synthetic)
    assert recovery.plugin_id == plugin.plugin_id and recovery.version == "1.0.0"


@pytest.mark.parametrize("synthetic", [False, True])
def test_generic_formal_report_cannot_promote_mixture_or_read_held_out_inputs(tmp_path, synthetic):
    store, spec, registry = prepared(tmp_path, synthetic=synthetic, formal=True)
    store.register(spec, digest(spec))
    before = len(store.events())
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
            budget=BudgetSpec(60, category="smoke"))
    events = store.events()[before:]
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "READ_STARTED", "READ_COMPLETED"} for e in events)
    assert next(iter(store.attempts().values()))["state"] == "PREFLIGHT_FAILED"


def test_frozen_job_cap_refuses_before_reservation_or_worker(tmp_path):
    store, spec, registry = prepared(tmp_path, maximum_job_seconds=1)
    store.register(spec, digest(spec))
    before = len(store.events())
    with pytest.raises(ResearchError, match="frozen job budget"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
            budget=BudgetSpec(60, category="smoke"))
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "READ_STARTED", "READ_COMPLETED"}
        for e in store.events()[before:])


@pytest.mark.parametrize("nonlinear", [False, True])
def test_frozen_compiler_binds_explicit_mixture_per_cell_without_new_arms_or_admission(nonlinear):
    settings = MixtureSettings(8, 0., 0., (1.,)*4, 0., 1_000_000, 60.)
    design = fixture(nonlinear=nonlinear, methods=(StudyMethod("mixture", steps=2, samples=8,
        recovery=True, mixture_settings=settings), StudyMethod("euler", steps=2, samples=8)))
    frozen = freeze_design(design)
    document = frozen.manifest()
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert "admission" not in spec and "runtime_binding" not in spec
    assert len(document["matrix"]) == 8 and len(spec["arms"]) == 2
    mixture_cells = [cell for cell in spec["cells"] if "mixture_policy" in cell]
    assert len({cell["arm_id"] for cell in mixture_cells}) == 1
    assert len({cell["mixture_policy"]["request_hash"] for cell in mixture_cells}) == 4
    for cell in spec["cells"]:
        validate_propagation_cell(spec, cell)
    for horizon in design.horizons:
        for seed in design.seeds:
            assert len({cell["propagation_request"]["coupling_id"] for cell in spec["cells"]
                if cell["horizon"] == horizon and cell["seed"] == seed}) == 1


@pytest.mark.parametrize("changes", [{"maximum_work_units": 1}, {"state_scales": [1., 1., 1., 1.]},
    {"component_cap": 33}, {"maximum_job_seconds": 7201}])
def test_compiler_refuses_invalid_settings_not_silently_smaller_axes(changes):
    settings = replace(MixtureSettings(8, 0., 0., (1.,)*4, 0., 1_000_000, 60.), **changes)
    design = fixture(methods=(StudyMethod("mixture", steps=2, samples=8, recovery=True, mixture_settings=settings),))
    with pytest.raises(DataValidationError):
        freeze_design(design)
