"""Frozen horizon grids change requests, never budgets or scientific status."""

from dataclasses import replace

import pytest

from application.propagation_execution import validate_propagation_cell
from application.research_budget import BudgetLedger
from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash
from domain.mixture import MixtureSettings
from experiments.pirc27 import design as module
from experiments.pirc27.design import StudyFunctional, StudyMethod, freeze_design
from experiments.pirc27.preparation import reload_design
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.test_propagation_study_design import fixture


def grid(method):
    return replace(method, horizon_steps=((1., method.steps), (10., method.steps*4)))


@pytest.mark.parametrize("nonlinear", [False, True])
def test_horizon_grids_reach_actual_owner_requests_and_resources(nonlinear):
    names = ("exact", "gaussian", "euler", "heun", "reversible-heun", "mlmc",
             "importance", "cubature", "mixture", "pde")
    methods = tuple(grid(StudyMethod(name, steps=2, samples=8, chunk_size=4,
        level_samples=(4, 4) if name == "mlmc" else (),
        recovery=name in {"euler", "heun", "reversible-heun", "mlmc", "importance", "mixture"},
        mixture_settings=MixtureSettings(2, .1, 0., (1.,)*4, 0., 100000, 60.) if name == "mixture" else None))
        for name in names)
    design = fixture(nonlinear=nonlinear, methods=methods,
        functionals=(StudyFunctional("region", "endpoint-halfspace"),))
    frozen = freeze_design(design)
    document = frozen.manifest()
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert len(document["matrix"]) == 40 and len(spec["arms"]) == 10
    assert document["qualification"] == "preparation-only-not-scientific"
    assert "admission" not in spec
    for row, cell in zip(document["matrix"], spec["cells"]):
        expected_steps = 2 if row["horizon"] == 1. else 8
        assert cell["propagation_request"]["steps"] == expected_steps
        if row["disposition"] == "PLANNED":
            _, request, config, _ = validate_propagation_cell(spec, cell)
            assert request.steps == expected_steps
            # Shared MLMC resources bind the finest level, not its base grid.
            assert config["steps"] == expected_steps*(2 if row["method"] == "mlmc" else 1)
        else:
            assert "execution" not in cell
    for horizon in design.horizons:
        for seed in design.seeds:
            pair = [c["propagation_request"] for c in spec["cells"]
                if c["horizon"] == horizon and c["seed"] == seed]
            assert len({r["coupling_id"] for r in pair}) == 1
    assert all(arm["budget_seconds"] == 86400 for arm in spec["arms"])


@pytest.mark.parametrize("table", [None, [], [[1., 4], [10., 16]],
    ((1., 4),), ((1., 4), (100., 16)), ((10., 4), (1., 16)),
    ((1., 4), (1., 16)), ((True, 4), (10., 16)),
    ((1., 4), (10., True)), ((1., 4), (10., 16.)),
    ((1., 4), (10., 0)), ((1., 4), (10., 8193)),
    ((1., 4), (float("nan"), 16)), ((1., 8), (10., 16)),
    ((1., 4, 8), (10., 16))])
def test_invalid_or_incomplete_grids_refused_before_copy_or_input_evaluation(monkeypatch, table):
    design = fixture()
    design = replace(design, methods=(replace(design.methods[0], horizon_steps=table),))
    monkeypatch.setattr(module, "asdict", lambda *args: pytest.fail("invalid grid copied"))
    monkeypatch.setattr(module, "validate_oracle_input", lambda *a, **kw: pytest.fail("invalid grid evaluated"))
    with pytest.raises(DataValidationError, match="horizon"):
        freeze_design(design)


@pytest.mark.parametrize("method", [
    StudyMethod("euler", steps=1, samples=256, horizon_steps=((1., 1), (10., 8192))),
    StudyMethod("mlmc", steps=1, samples=8, level_samples=(4, 4), horizon_steps=((1., 1), (10., 8192))),
])
def test_later_horizon_overwork_refused_before_full_matrix_expansion(monkeypatch, method):
    original = module.product
    def bounded_product(*args):
        if len(args) == 5:
            pytest.fail("overwork matrix expanded")
        return original(*args)
    monkeypatch.setattr(module, "product", bounded_product)
    with pytest.raises(DataValidationError):
        freeze_design(fixture(methods=(method,)))


def test_implicit_and_explicit_constant_grids_cannot_disguise_duplicate_configs():
    method = fixture().methods[0]
    methods = (replace(method, configuration_id="implicit"),
        replace(method, configuration_id="explicit", horizon_steps=((1., 4), (10., 4))))
    with pytest.raises(DataValidationError, match="duplicate numerical"):
        freeze_design(fixture(methods=methods))


def test_legacy_grids_remain_omitted_and_first_horizon_bound():
    design = fixture()
    old = freeze_design(design).manifest()
    assert all("horizon_steps" not in m for m in old["axis_manifest"]["methods"])
    explicit = replace(design, methods=tuple(replace(m, horizon_steps=((1., m.steps), (10., m.steps)))
        for m in design.methods))
    new = freeze_design(explicit).manifest()
    assert old["arms"] == new["arms"]
    assert [r["cell"]["propagation_request"]["steps"] for r in old["matrix"]] == [
        r["cell"]["propagation_request"]["steps"] for r in new["matrix"]]
    assert old["matrix"][0]["request_hash"] != new["matrix"][0]["request_hash"]


def test_roundtrip_complete_grid_and_rehashed_changed_request_refusal(tmp_path):
    design = fixture()
    frozen = freeze_design(replace(design, methods=tuple(grid(m) for m in design.methods)))
    path = tmp_path/"design.json"
    path.write_bytes(encode(frozen.manifest()))
    assert reload_design(path, expected_hash=frozen.manifest_hash).manifest() == frozen.manifest()
    changed = frozen.manifest()
    changed["matrix"][-1]["cell"]["propagation_request"]["steps"] = 4
    path.write_bytes(encode(changed))
    with pytest.raises(ResearchError, match="regenerated"):
        reload_design(path, expected_hash=content_hash(changed))


def test_horizon_expansion_keeps_closed_original_arm(tmp_path):
    store = ResearchStore(tmp_path, "horizon-grid-unit", initialize=True)
    design = fixture()
    original = freeze_design(design)
    spec = original.study_spec(expected_hash=original.manifest_hash)
    store.register(spec, digest(spec))
    arm = spec["arms"][0]["arm_id"]
    store.append("ARM_CLOSED", {"arm_id": arm, "reason": "engineering closure"})
    expanded = freeze_design(replace(design, study_id="horizon-expanded",
        methods=tuple(grid(m) for m in design.methods)))
    next_spec = expanded.study_spec(expected_hash=expanded.manifest_hash)
    store.register(next_spec, digest(next_spec))
    assert next_spec["arms"] == spec["arms"] and BudgetLedger(store).balance(arm)["closed"]
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "SETTLE"} for e in store.events())
