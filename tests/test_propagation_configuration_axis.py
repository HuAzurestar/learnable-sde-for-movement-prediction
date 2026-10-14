"""Explicit numerical axes share original arms; no automatic research launch."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.propagation_execution import validate_propagation_cell
from application.research_budget import BudgetLedger
from application.research_contracts import CapabilityRegistry
from application.research_dimensions import comparison_dimensions
from application.research_evidence import export_evidence
from domain.errors import DataValidationError
from domain.mixture import MixtureSettings
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27 import design as module
from experiments.pirc27.design import StudyArm, StudyFunctional, StudyMethod, freeze_design
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.test_propagation_study_design import fixture


def configurations(names=("euler", "heun")):
    return tuple(StudyMethod(name, steps=steps, samples=samples, chunk_size=8,
        recovery=True, configuration_id=f"grid-{steps}-samples-{samples}")
        for name in names for steps in (2, 4) for samples in (8, 16))


@pytest.mark.parametrize("nonlinear", [False, True])
def test_full_grid_count_axis_has_unique_cells_and_shared_original_method_arms(nonlinear):
    design = fixture(nonlinear=nonlinear, methods=configurations())
    frozen = freeze_design(design)
    doc = frozen.manifest()
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert len(doc["matrix"]) == doc["expected_cells"] == 32
    assert len(doc["arms"]) == 2
    assert all(arm["budget_seconds"] == 86400 for arm in doc["arms"])
    assert len({row["request_hash"] for row in doc["matrix"]}) == 32
    for row, cell in zip(doc["matrix"], spec["cells"]):
        _, request, config, _ = validate_propagation_cell(spec, cell)
        assert request.steps in (2, 4) and request.samples in (8, 16)
        assert row["configuration_id"] == f"grid-{request.steps}-samples-{request.samples}"
        assert comparison_dimensions(cell)["numerical_configuration"] == row["configuration_id"]
        assert config["samples"] == request.samples
    for horizon in design.horizons:
        for seed in design.seeds:
            rows = [row for row in doc["matrix"] if row["horizon"] == horizon and row["seed"] == seed]
            assert len({row["cell"]["propagation_request"]["coupling_id"] for row in rows}) == 1
            for label in {row["configuration_id"] for row in rows}:
                pair = [row for row in rows if row["configuration_id"] == label]
                assert len(pair) == 2
                assert comparison_dimensions(pair[0]["cell"]) == comparison_dimensions(pair[1]["cell"])
    assert "runtime_binding" not in spec and "admission" not in spec


@pytest.mark.parametrize("case", ["missing", "mixed", "duplicate-label", "duplicate-values", "invalid-label", "split-arm"])
def test_ambiguous_configurations_and_fresh_arm_aliases_are_refused(case):
    methods = configurations(("euler",))
    if case == "missing":
        methods = tuple(replace(method, configuration_id=None) for method in methods)
    elif case == "mixed":
        methods = (replace(methods[0], configuration_id=None),)+methods[1:]
    elif case == "duplicate-label":
        methods = (methods[0], replace(methods[1], configuration_id=methods[0].configuration_id))
    elif case == "duplicate-values":
        methods = (methods[0], replace(methods[0], configuration_id="different-label"))
    elif case == "invalid-label":
        methods = (replace(methods[0], configuration_id="invalid/label"),)
    design = fixture(methods=methods)
    if case == "split-arm":
        design = replace(design, arms=tuple(StudyArm("new-"+str(i), design.models[0].family_id,
            method.configuration_id, "endpoint-x") for i, method in enumerate(methods)))
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(design)


@pytest.mark.parametrize("label", [True, [], {"nested": "value"}, "", "x"*129])
def test_invalid_configuration_primitives_refused_before_copying(monkeypatch, label):
    design = fixture(methods=(replace(configurations()[0], configuration_id=label),))
    monkeypatch.setattr(module, "asdict", lambda *args: pytest.fail("invalid label copied"))
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(design)


def test_legacy_single_configuration_shape_and_pairing_are_not_relabelled():
    doc = freeze_design(fixture()).manifest()
    assert all("configuration_id" not in method for method in doc["axis_manifest"]["methods"])
    assert all("configuration_id" not in row for row in doc["matrix"])
    assert all("numerical_configuration" not in row["cell"]["comparison_dimensions"] for row in doc["matrix"])


@pytest.mark.parametrize("proposal", [(0, 0), (-0., 0.)])
def test_scalar_aliases_cannot_disguise_duplicate_numerical_configuration(proposal):
    original = configurations(("euler",))[0]
    duplicate = replace(original, configuration_id="alias-label", proposal=proposal)
    with pytest.raises(DataValidationError, match="duplicate numerical configuration"):
        freeze_design(fixture(methods=(original, duplicate)))


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("name", ["exact", "gaussian", "cubature", "mixture"])
def test_unused_path_counts_cannot_disguise_deterministic_configuration(monkeypatch, nonlinear, name):
    original = StudyMethod(name, steps=2, samples=8, chunk_size=8,
        recovery=name == "mixture",
        mixture_settings=MixtureSettings(2, .1, 0., (1.,)*4, 0., 100000, 60.) if name == "mixture" else None,
        configuration_id="count-8")
    alias = replace(original, samples=16, configuration_id="count-16")
    design = fixture(nonlinear=nonlinear, methods=(original, alias))
    monkeypatch.setattr(module, "validate_oracle_input", lambda *a, **kw: pytest.fail("alias evaluated"))
    monkeypatch.setattr(module, "validate_nonlinear_input", lambda *a, **kw: pytest.fail("alias evaluated"))
    monkeypatch.setattr(module, "product", lambda *a: pytest.fail("alias expanded"))
    with pytest.raises(DataValidationError, match="duplicate numerical configuration"):
        freeze_design(design)


@pytest.mark.parametrize("name", ["euler", "heun", "reversible-heun", "importance", "mlmc"])
def test_actual_path_count_changes_remain_distinct_original_arm_configs(name):
    original = StudyMethod(name, steps=2, samples=8, chunk_size=8,
        recovery=True, level_samples=(4, 4) if name == "mlmc" else (), configuration_id="count-8")
    changed = replace(original, samples=16, level_samples=(8, 8) if name == "mlmc" else (),
        configuration_id="count-16")
    doc = freeze_design(fixture(methods=(original, changed),
        functionals=(StudyFunctional("probability", "endpoint-halfspace"),),
        horizons=(1.,), seeds=(11,))).manifest()
    assert doc["expected_cells"] == 2 and len(doc["arms"]) == 1
    assert [row["cell"]["propagation_request"]["samples"] for row in doc["matrix"]] == [8, 16]
    assert len({row["request_hash"] for row in doc["matrix"]}) == 2
    assert doc["arms"][0]["budget_seconds"] == 86400


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("name", ["exact", "gaussian", "cubature", "mixture"])
def test_single_deterministic_config_keeps_original_request_and_resource_fields(nonlinear, name):
    method = StudyMethod(name, steps=2, samples=16, chunk_size=8,
        recovery=name == "mixture",
        mixture_settings=MixtureSettings(2, .1, 0., (1.,)*4, 0., 100000, 60.) if name == "mixture" else None)
    doc = freeze_design(fixture(nonlinear=nonlinear, methods=(method,), horizons=(1.,), seeds=(11,))).manifest()
    assert doc["axis_manifest"]["methods"][0]["samples"] == 16
    row = doc["matrix"][0]
    assert row["cell"]["propagation_request"]["samples"] == 16
    assert "configuration_id" not in row
    assert "numerical_configuration" not in row["cell"]["comparison_dimensions"]
    if row["disposition"] == "PLANNED":
        assert row["cell"]["execution"]["config"]["samples"] == 16
    else:
        assert row["disposition"] == "INELIGIBLE" and "execution" not in row["cell"]


@pytest.mark.parametrize("nonlinear", [False, True])
def test_all_method_configs_keep_full_resource_or_nonexecution_bindings(nonlinear):
    names = ("exact", "gaussian", "euler", "heun", "reversible-heun", "mlmc",
             "importance", "cubature", "mixture", "pde")
    methods = tuple(StudyMethod(name, steps=steps, samples=samples, chunk_size=8,
        level_samples=(samples//2, samples//2) if name == "mlmc" else (),
        proposal=(0.3, 0.) if name == "importance" else (0., 0.),
        recovery=name in {"euler", "heun", "reversible-heun", "mlmc", "importance", "mixture"},
        mixture_settings=MixtureSettings(4, 0., 0., (1.,)*4, 0., 1_000_000, 60.) if name == "mixture" else None,
        configuration_id=f"grid-{steps}-samples-{samples}")
        for name in names for steps, samples in ((2, 8), (4, 16)))
    frozen = freeze_design(fixture(nonlinear=nonlinear, methods=methods,
        functionals=(StudyFunctional("probability", "endpoint-halfspace"),), horizons=(1.,), seeds=(11,)))
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert len(spec["arms"]) == 10 and len(spec["cells"]) == 20
    for row, cell in zip(frozen.manifest()["matrix"], spec["cells"]):
        if row["disposition"] == "PLANNED":
            validate_propagation_cell(spec, cell)
        else:
            assert cell["execution_disposition"]["status"] == row["disposition"]
            assert "execution" not in cell
        assert cell["comparison_dimensions"]["numerical_configuration"] == row["configuration_id"]
    assert sum(row["disposition"] == "NOT_IMPLEMENTED" for row in frozen.manifest()["matrix"]) == 2


def test_expanded_configuration_axes_keep_cell_limit_before_materialization(monkeypatch):
    design = fixture(methods=configurations(), horizons=tuple(float(n) for n in range(1, 65)), seeds=tuple(range(64)))
    monkeypatch.setattr(module, "product", lambda *args: pytest.fail("expanded over-cap matrix"))
    with pytest.raises(DataValidationError, match="cell limit"):
        freeze_design(design)


def test_registered_configuration_expansion_keeps_closed_arm_and_refuses_reset(tmp_path):
    store = ResearchStore(tmp_path, "configuration-unit", initialize=True)
    original = fixture(horizons=(1.,), seeds=(11,))
    frozen = freeze_design(original)
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    store.register(spec, digest(spec))
    arm = original.arms[0].arm_id
    store.append("ARM_CLOSED", {"arm_id": arm, "reason": "engineering closure"})
    expanded = replace(original, study_id="expanded", methods=configurations())
    next_frozen = freeze_design(expanded)
    next_spec = next_frozen.study_spec(expected_hash=next_frozen.manifest_hash)
    assert next_spec["arms"] == spec["arms"]
    store.register(next_spec, digest(next_spec))
    assert BudgetLedger(store).balance(arm)["closed"]
    reset = replace(expanded, study_id="reset", arms=(replace(expanded.arms[0], arm_id="fresh-config-arm"), expanded.arms[1]))
    reset_frozen = freeze_design(reset)
    reset_spec = reset_frozen.study_spec(expected_hash=reset_frozen.manifest_hash)
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        store.register(reset_spec, digest(reset_spec))


@pytest.mark.parametrize("nonlinear", [False, True])
def test_authorized_full_export_and_independent_paper_keep_each_configuration_stratum(tmp_path, nonlinear):
    methods = configurations() if not nonlinear else tuple(StudyMethod(name, steps=steps, samples=samples,
        configuration_id=f"grid-{steps}-samples-{samples}")
        for name in ("euler", "exact", "pde") for steps, samples in ((2, 8), (4, 16)))
    frozen = freeze_design(fixture(nonlinear=nonlinear, methods=methods))
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    store = ResearchStore(tmp_path/"runtime", "configuration-export", initialize=True)
    store.register(spec, digest(spec))
    runner = SharedRunner(store, CapabilityRegistry())
    for cell in spec["cells"]:
        if "execution_disposition" in cell:
            refusal = runner.run_cell(spec["study_id"], digest(cell))
            assert refusal["state"] == "PREFLIGHT_FAILED"
            assert refusal["error_code"] == cell["execution_disposition"]["status"]
    grant = {"authorization_id": "export-fixture", "study_id": spec["study_id"],
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("engineering export"),
        "purposes": ["export"], "visibilities": ["synthetic"], "block_ids": [spec["cells"][0]["block_id"]]}
    store.authorize(grant)
    bundle = export_evidence(store, spec["study_id"], grant)
    expected_cells = 24 if nonlinear else 32
    assert len(bundle["expected_cells"]) == len(bundle["cells"]) == expected_cells
    assert sum(row["status"] == "MISSING" for row in bundle["cells"]) == (8 if nonlinear else 32)
    assert sum(row["status"] == "PREFLIGHT_FAILED" for row in bundle["cells"]) == (16 if nonlinear else 0)
    assert len({(row["arm_id"], row["block_id"], row["seed"], digest(row["comparison_dimensions"]))
        for row in bundle["cells"]}) == expected_cells
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    source = tmp_path/"bundle.json"
    source.write_bytes(encode(bundle))
    output = tmp_path/"paper"
    result = subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/aggregate.py"), str(source),
        "--expected-hash", bundle["bundle_hash"], "--output", str(output)], cwd=paper,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((output/"aggregate.json").read_bytes())
    assert aggregate["expected_cell_count"] == expected_cells and aggregate["successful_cell_count"] == 0
    assert len(aggregate["arms"]) == (12 if nonlinear else 16)
    assert all(arm["expected_cells"] == 2 and arm["independent_n"] == 0 for arm in aggregate["arms"])
    assert all(arm["status_rates"]["denominator"] == 2 for arm in aggregate["arms"])
    assert all(arm["status"] == "incomplete" for arm in aggregate["arms"])
    assert not any(event["event_kind"] in {"RESERVE", "WORKER_STARTED", "SETTLE"} for event in store.events())
