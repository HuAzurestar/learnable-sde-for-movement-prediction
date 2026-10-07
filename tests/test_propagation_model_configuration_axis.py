"""Model scales/initializations are strata, not new arms or independent blocks."""

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
from experiments.pirc27 import design as module
from experiments.pirc27.design import StudyMethod, freeze_design
from experiments.pirc27.nonlinear import nonlinear_package
from experiments.pirc27.oracles import affine_package
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.test_propagation_configuration_axis import configurations
from tests.test_propagation_study_design import fixture


def model_configurations(design):
    base = design.models[0]
    if base.package.manifest()["family"] == "synthetic-nonlinear":
        scaled = nonlinear_package("length-scale-unit", length_scale=(2., 0.5))
    else:
        parameters = base.package.manifest()["parameters"]
        scaled = affine_package("diffusion-scale-unit", parameters["A"], parameters["b"],
            [[2*x for x in row] for row in parameters["L"]])
    covariance = tuple(tuple(4. if i == j else 0. for j in range(4)) for i in range(4))
    return (replace(base, configuration_id="base"),
        replace(base, configuration_id="position-shift", initial_mean=(2., -1., 1., -0.5)),
        replace(base, configuration_id="initial-spread", initial_covariance=covariance),
        replace(base, configuration_id="model-scale", package=scaled))


@pytest.mark.parametrize("nonlinear", [False, True])
def test_model_scale_and_initial_axes_share_original_arms_and_physical_blocks(nonlinear):
    design = fixture(nonlinear=nonlinear, methods=configurations())
    design = replace(design, models=model_configurations(design))
    frozen = freeze_design(design)
    document = frozen.manifest()
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert document["expected_cells"] == len(spec["cells"]) == 128
    assert len(spec["arms"]) == 2 and all(arm["budget_seconds"] == 86400 for arm in spec["arms"])
    assert len({row["request_hash"] for row in document["matrix"]}) == 128
    assert {cell["block_id"] for cell in spec["cells"]} == {design.models[0].family_id}
    assert document["axis_manifest"]["state_names"] == ["x", "y", "vx", "vy"]
    assert "admission" not in spec and "runtime_binding" not in spec
    for row, cell in zip(document["matrix"], spec["cells"]):
        package, request, _, _ = validate_propagation_cell(spec, cell)
        model = next(model for model in design.models if model.configuration_id == row["model_configuration_id"])
        assert package.package_hash == model.package.package_hash
        assert request.initial_mean == model.initial_mean
        assert request.initial_covariance == model.initial_covariance
        assert comparison_dimensions(cell)["model_configuration"] == model.configuration_id
    for horizon in design.horizons:
        for seed in design.seeds:
            for model in design.models:
                rows = [row for row in document["matrix"] if row["model_configuration_id"] == model.configuration_id
                    and row["horizon"] == horizon and row["seed"] == seed]
                assert len(rows) == 8
                assert len({row["cell"]["propagation_request"]["coupling_id"] for row in rows}) == 1
                for numerical in {row["configuration_id"] for row in rows}:
                    pair = [row["cell"] for row in rows if row["configuration_id"] == numerical]
                    assert len(pair) == 2 and comparison_dimensions(pair[0]) == comparison_dimensions(pair[1])
    before = frozen.manifest_hash
    document["axis_manifest"]["models"][0]["configuration_id"] = "changed"
    document["matrix"].clear()
    assert frozen.manifest_hash == before and len(frozen.manifest()["matrix"]) == 128


@pytest.mark.parametrize("case", ["unlabelled", "mixed", "duplicate-label", "duplicate-values"])
def test_ambiguous_same_family_model_axes_are_refused(case):
    design = fixture()
    models = model_configurations(design)[:2]
    if case == "unlabelled":
        models = tuple(replace(model, configuration_id=None) for model in models)
    elif case == "mixed":
        models = (replace(models[0], configuration_id=None), models[1])
    elif case == "duplicate-label":
        models = (models[0], replace(models[1], configuration_id="base"))
    else:
        models = (models[0], replace(models[0], configuration_id="disguised"))
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(replace(design, models=models))


@pytest.mark.parametrize("label", [True, [], {"label": "nested"}, "", "x"*129])
def test_model_labels_refuse_invalid_later_values_before_copy_or_package_evaluation(monkeypatch, label):
    design = fixture()
    models = model_configurations(design)[:2]
    models = (models[0], replace(models[1], configuration_id=label))
    monkeypatch.setattr(module, "asdict", lambda *a: pytest.fail("invalid label copied"))
    monkeypatch.setattr(module, "validate_oracle_input", lambda *a, **kw: pytest.fail("invalid label evaluated"))
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(replace(design, models=models))


@pytest.mark.parametrize("zero", [0, -0.])
def test_initial_scalar_aliases_cannot_disguise_duplicate_same_package_model_configuration(zero):
    design = fixture()
    base = model_configurations(design)[0]
    alias = replace(base, configuration_id="alias", initial_mean=(zero, zero, 1., -0.5),
        initial_covariance=tuple((zero,)*4 for _ in range(4)))
    with pytest.raises(DataValidationError, match="duplicate model configuration"):
        freeze_design(replace(design, models=(base, alias)))


def test_legacy_model_shape_and_request_identity_remain_unlabelled():
    design = fixture()
    document = freeze_design(design).manifest()
    assert all("configuration_id" not in model for model in document["axis_manifest"]["models"])
    assert all("model_configuration_id" not in row for row in document["matrix"])
    assert all("model_configuration" not in row["cell"]["comparison_dimensions"] for row in document["matrix"])
    labelled = replace(design, models=(replace(design.models[0], configuration_id="base"),))
    other = freeze_design(labelled).manifest()
    assert [row["request_hash"] for row in other["matrix"]] == [row["request_hash"] for row in document["matrix"]]
    assert [row["cell"]["propagation_request"]["coupling_id"] for row in other["matrix"]] == [
        row["cell"]["propagation_request"]["coupling_id"] for row in document["matrix"]]
    assert other["arms"] == document["arms"]


def test_model_axis_cell_cap_is_checked_before_expansion(monkeypatch):
    design = fixture(methods=configurations(), horizons=tuple(float(i) for i in range(1, 65)), seeds=tuple(range(64)))
    design = replace(design, models=model_configurations(design))
    monkeypatch.setattr(module, "product", lambda *a: pytest.fail("over-cap expansion"))
    with pytest.raises(DataValidationError, match="cell limit"):
        freeze_design(design)


def test_registered_scale_expansion_cannot_reset_closed_original_arms(tmp_path):
    store = ResearchStore(tmp_path, "model-axis-unit", initialize=True)
    original = fixture(horizons=(1.,), seeds=(11,))
    frozen = freeze_design(original)
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    store.register(spec, digest(spec))
    arm = original.arms[0].arm_id
    store.append("ARM_CLOSED", {"arm_id": arm, "reason": "engineering closure"})
    expanded = replace(original, study_id="expanded-models", models=model_configurations(original))
    next_frozen = freeze_design(expanded)
    next_spec = next_frozen.study_spec(expected_hash=next_frozen.manifest_hash)
    store.register(next_spec, digest(next_spec))
    assert next_spec["arms"] == spec["arms"] and BudgetLedger(store).balance(arm)["closed"]
    renamed = replace(expanded, study_id="renamed-models",
        arms=(replace(expanded.arms[0], arm_id="fresh-scale-arm"), expanded.arms[1]))
    reset_frozen = freeze_design(renamed)
    reset_spec = reset_frozen.study_spec(expected_hash=reset_frozen.manifest_hash)
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        store.register(reset_spec, digest(reset_spec))


@pytest.mark.parametrize("nonlinear", [False, True])
def test_full_authorized_missing_refused_export_never_counts_model_configs_as_independent_blocks(tmp_path, nonlinear):
    methods = tuple(StudyMethod(name, steps=2, samples=8, chunk_size=8)
        for name in (("euler", "exact", "pde") if nonlinear else ("euler", "heun")))
    design = fixture(nonlinear=nonlinear, methods=methods, horizons=(1.,), seeds=(11, 19))
    design = replace(design, models=model_configurations(design)[::3])
    frozen = freeze_design(design)
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    store = ResearchStore(tmp_path/"runtime", "model-axis-export", initialize=True)
    store.register(spec, digest(spec))
    runner = SharedRunner(store, CapabilityRegistry())
    for cell in spec["cells"]:
        if "execution_disposition" in cell:
            assert runner.run_cell(spec["study_id"], digest(cell))["state"] == "PREFLIGHT_FAILED"
    grant = {"authorization_id": "export-unit", "study_id": spec["study_id"],
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("engineering export"),
        "purposes": ["export"], "visibilities": ["synthetic"], "block_ids": [spec["cells"][0]["block_id"]]}
    store.authorize(grant)
    bundle = export_evidence(store, spec["study_id"], grant)
    count = 12 if nonlinear else 8
    assert len(bundle["expected_cells"]) == len(bundle["cells"]) == count
    assert {row["block_id"] for row in bundle["cells"]} == {design.models[0].family_id}
    assert sum(row["status"] == "MISSING" for row in bundle["cells"]) == (4 if nonlinear else 8)
    assert sum(row["status"] == "PREFLIGHT_FAILED" for row in bundle["cells"]) == (8 if nonlinear else 0)
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    source = tmp_path/"bundle.json"
    source.write_bytes(encode(bundle))
    output = tmp_path/"paper"
    result = subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/aggregate.py"), str(source),
        "--expected-hash", bundle["bundle_hash"], "--output", str(output)], cwd=paper,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((output/"aggregate.json").read_bytes())
    assert aggregate["expected_cell_count"] == count and aggregate["successful_cell_count"] == 0
    assert len(aggregate["arms"]) == (6 if nonlinear else 4)
    assert all(arm["expected_cells"] == 2 and arm["independent_n"] == 0 for arm in aggregate["arms"])
    assert all(arm["status_rates"]["denominator"] == 2 and arm["status"] == "incomplete" for arm in aggregate["arms"])
    assert not any(event["event_kind"] in {"RESERVE", "WORKER_STARTED", "SETTLE"} for event in store.events())
