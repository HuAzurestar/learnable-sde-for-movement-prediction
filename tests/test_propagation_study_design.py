"""Pure bounded preparation: no research ledger, grants or estimates are made."""

from dataclasses import FrozenInstanceError, replace
import copy

import pytest

from application.propagation_execution import validate_propagation_cell
from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash
from experiments.pirc27 import design as module
from experiments.pirc27.design import (PropagationStudyDesign, StudyModel, StudyMethod,
    StudyFunctional, StudyArm, freeze_design)
from experiments.pirc27.nonlinear import nonlinear_package
from experiments.pirc27.oracles import oracle_suite
from infrastructure.research_store import ResearchError


def fixture(*, nonlinear=False, methods=None, functionals=None, horizons=(1.0, 10.0), seeds=(11, 19)):
    case = oracle_suite()[0]
    package = nonlinear_package() if nonlinear else case.package
    models = (StudyModel("stress-v1" if nonlinear else "stable-v1", package,
                        case.initial_mean, case.initial_covariance),)
    methods = methods or (StudyMethod("euler", samples=16, steps=4, chunk_size=8, recovery=True),
                          StudyMethod("heun", samples=16, steps=4, chunk_size=8, recovery=True))
    functionals = functionals or (StudyFunctional("mean-x", "endpoint-x"),)
    families = sorted({(model.family_id, method.method, f.kind) for model in models
                       for method in methods for f in functionals})
    # Operator-provided fixture identities, not derived from study/config/seed.
    arms = tuple(StudyArm("existing-" + str(i), *family) for i, family in enumerate(families))
    return PropagationStudyDesign("design-unit", "endpoint-unit", "synthetic-engineering", models,
        methods, functionals, horizons, seeds, arms, *(content_hash(key) for key in
            ("protocol", "synthetic-input", "context", "no-selection")))


@pytest.mark.parametrize("nonlinear", [False, True])
def test_compilation_is_immutable_and_all_cells_pass_actual_owner_validation(nonlinear):
    design = fixture(nonlinear=nonlinear)
    frozen = freeze_design(design)
    document = frozen.manifest()
    assert document["expected_cells"] == len(document["matrix"]) == 8
    assert document["qualification"] == "preparation-only-not-scientific"
    assert len(document["arms"]) == 2
    assert all(a["budget_seconds"] == 86400 for a in document["arms"])
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert "admission" not in spec and "runtime_binding" not in spec
    assert len(spec["cells"]) == 8
    for cell in spec["cells"]:
        package, request, config, plugin = validate_propagation_cell(spec, cell)
        assert package.package_hash == design.models[0].package.package_hash
        assert request.horizons == (cell["horizon"],)
        assert config["method"] in {"euler", "heun"}
        assert plugin.resume_level == "chunk"
    before = frozen.manifest_hash
    document["matrix"].clear()
    spec["cells"][0]["frozen_dynamics"]["parameters"]["b"][0] = 90
    assert frozen.manifest_hash == before and len(frozen.manifest()["matrix"]) == 8
    with pytest.raises(FrozenInstanceError):
        design.seeds = (99,)
    assert freeze_design(design).manifest_hash == before


def test_comparators_pair_inputs_not_method_configuration_and_never_multiply_budget_arms():
    design = fixture()
    doc = freeze_design(design).manifest()
    cells = [r["cell"] for r in doc["matrix"]]
    for horizon in design.horizons:
        for seed in design.seeds:
            pair = [c["propagation_request"] for c in cells if c["horizon"] == horizon and c["seed"] == seed]
            assert len({r["coupling_id"] for r in pair}) == 1
            assert len({r["request_id"] for r in pair}) == 2
            for key in ("initial_mean", "initial_covariance", "origin", "history_cutoff", "horizons",
                        "normal", "threshold", "closed", "functional_version", "tolerance"):
                assert pair[0][key] == pair[1][key]
    changed = replace(design, study_id="other-study", seeds=(29,), horizons=(100.0,),
                      methods=tuple(replace(m, chunk_size=1, steps=8) for m in design.methods))
    new = freeze_design(changed).manifest()
    assert doc["arms"] == new["arms"]
    assert doc["matrix"][0]["request_hash"] != new["matrix"][0]["request_hash"]


def test_multiple_regions_share_the_same_objective_arm_but_not_requests():
    functions = (StudyFunctional("right", "endpoint-halfspace"),
                 StudyFunctional("far-right", "endpoint-halfspace", threshold=3.0))
    doc = freeze_design(fixture(functionals=functions)).manifest()
    assert len(doc["arms"]) == 2 and len(doc["matrix"]) == 16
    assert len({r["request_hash"] for r in doc["matrix"]}) == 16
    assert len({r["arm_id"] for r in doc["matrix"] if r["method"] == "euler"}) == 1


def test_unsupported_and_affine_only_rows_are_retained_in_the_registered_draft():
    methods = tuple(StudyMethod(method, samples=8, steps=2) for method in
                    ("euler", "exact", "gaussian", "mixture", "reversible-heun", "pde"))
    frozen = freeze_design(fixture(nonlinear=True, methods=methods))
    doc = frozen.manifest()
    assert doc["expected_cells"] == len(doc["matrix"]) == 24
    assert {r["method"]: r["disposition"] for r in doc["matrix"]} == {
        "euler": "PLANNED", "exact": "INELIGIBLE", "gaussian": "INELIGIBLE",
        "mixture": "NOT_IMPLEMENTED", "reversible-heun": "PLANNED", "pde": "NOT_IMPLEMENTED"}
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert len(spec["cells"]) == 24
    for row, cell in zip(doc["matrix"], spec["cells"]):
        assert cell == row["cell"]
        if row["disposition"] == "PLANNED":
            validate_propagation_cell(spec, cell)
        else:
            assert "execution" not in cell
            assert cell["execution_disposition"]["status"] == row["disposition"]


def test_matrix_cap_precedes_package_validation_and_cartesian_materialization(monkeypatch):
    design = fixture(horizons=tuple(float(i) for i in range(1, 65)), seeds=tuple(range(64)))
    design = replace(design, methods=design.methods + (StudyMethod("gaussian"),))
    def forbidden(*args, **kwargs):
        pytest.fail("matrix expanded or package processed before the cell limit")
    monkeypatch.setattr(module, "product", forbidden)
    monkeypatch.setattr(module, "validate_oracle_input", forbidden)
    with pytest.raises(DataValidationError, match="cell limit"):
        freeze_design(design)


def test_byte_quota_precedes_cartesian_materialization(monkeypatch):
    design = fixture(horizons=tuple(float(i) for i in range(1, 65)), seeds=tuple(range(64)))
    monkeypatch.setattr(module, "product", lambda *args: pytest.fail("expanded before byte quota"))
    with pytest.raises(DataValidationError, match="byte quota"):
        freeze_design(design)


@pytest.mark.parametrize("changes", [
    {"horizons": (0.0,)}, {"horizons": (1.0, 1.0)}, {"horizons": (float("inf"),)},
    {"seeds": (True,)}, {"seeds": (11, 11)}, {"seeds": [11]},
    {"history_cutoff": 1.0}, {"origin": float("nan")}, {"protocol_hash": "missing"},
    {"stopping_rule": "extend-until-significant"}, {"primary_metrics": ("cost", "cost")},
])
def test_invalid_freeze_metadata_is_rejected(changes):
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(replace(fixture(), **changes))


@pytest.mark.parametrize("method", [
    StudyMethod("unknown"), StudyMethod("exact", recovery=True),
    StudyMethod("euler", samples=1), StudyMethod("euler", steps=8193),
    StudyMethod("euler", proposal=(1.0, 0.0)), StudyMethod("euler", level_samples=(8,)),
    StudyMethod("importance", proposal=[0.0, 0.0]),
    StudyMethod("mlmc", samples=10, level_samples=(8, 8)),
    StudyMethod("mlmc", samples=16, steps=8192, level_samples=(8, 8)),
    StudyMethod("euler", samples=1024, steps=8192),
])
def test_invalid_or_oversized_method_configuration_is_rejected(method):
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(fixture(methods=(method,)))


def test_importance_requires_a_probability_objective():
    with pytest.raises(DataValidationError, match="probability objective"):
        freeze_design(fixture(methods=(StudyMethod("importance"),)))


@pytest.mark.parametrize("change", ["duplicate-id", "duplicate-family", "missing", "extra", "config-family"])
def test_arm_family_mapping_is_explicit_exact_and_cannot_split_by_configuration(change):
    design = fixture()
    a, b = design.arms
    arms = {"duplicate-id": (a, replace(b, arm_id=a.arm_id)), "duplicate-family": (a, a),
            "missing": (a,), "extra": (a, b, StudyArm("extra", "unused", "euler", "endpoint-x")),
            "config-family": (replace(a, method_family_id="euler-new-steps"), b)}[change]
    with pytest.raises(DataValidationError):
        freeze_design(replace(design, arms=arms))


def test_changed_package_or_initial_distribution_changes_frozen_identity_and_pairing():
    design = fixture(nonlinear=True)
    frozen = freeze_design(design)
    model = design.models[0]
    changed = replace(model, package=nonlinear_package(amplitude=(4.0, 0.0)))
    other = freeze_design(replace(design, models=(changed,)))
    assert other.manifest_hash != frozen.manifest_hash
    assert other.manifest()["arms"] == frozen.manifest()["arms"]
    assert other.manifest()["matrix"][0]["cell"]["propagation_request"]["coupling_id"] != (
        frozen.manifest()["matrix"][0]["cell"]["propagation_request"]["coupling_id"])
    invalid = replace(model, initial_covariance=((-1.0, 0.0, 0.0, 0.0),) + model.initial_covariance[1:])
    with pytest.raises(DataValidationError):
        freeze_design(replace(design, models=(invalid,)))


def test_frozen_draft_requires_exact_hash_and_current_source(monkeypatch):
    frozen = freeze_design(fixture())
    with pytest.raises(DataValidationError, match="source differs"):
        frozen.study_spec(expected_hash="0" * 64)
    from experiments.pirc25 import affine
    monkeypatch.setattr(affine, "code_hash", lambda: "0" * 64)
    with pytest.raises(DataValidationError, match="source differs"):
        frozen.study_spec(expected_hash=frozen.manifest_hash)


def test_all_supported_method_cells_have_frozen_shared_resource_bindings():
    functions = (StudyFunctional("probability", "endpoint-halfspace"),)
    for nonlinear, names in ((False, ("exact", "gaussian", "euler", "heun", "reversible-heun", "mlmc", "importance")),
                             (True, ("cubature", "euler", "heun", "reversible-heun", "mlmc", "importance"))):
        methods = tuple(StudyMethod(name, steps=2, samples=16,
            level_samples=(8, 8) if name == "mlmc" else (),
            proposal=(1.0, 0.0) if name == "importance" else (0.0, 0.0)) for name in names)
        frozen = freeze_design(fixture(nonlinear=nonlinear, methods=methods, functionals=functions))
        spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
        for cell in spec["cells"]:
            validate_propagation_cell(spec, cell)
        modified = copy.deepcopy(spec)
        modified["cells"][0]["execution"]["resource_plan_hash"] = "0" * 64
        with pytest.raises(ResearchError):
            validate_propagation_cell(modified, modified["cells"][0])


def test_prepared_specs_preserve_shared_closed_arm_and_reject_renamed_identity(tmp_path):
    from application.research_budget import BudgetLedger
    from infrastructure.research_store import ResearchStore, digest
    # Disposable engineering store, registration only: no launch/admission/grant.
    store = ResearchStore(tmp_path, "design-unit", initialize=True)
    design = fixture(horizons=(1.0,), seeds=(11,))
    prepared = freeze_design(design)
    spec = prepared.study_spec(expected_hash=prepared.manifest_hash)
    store.register(spec, digest(spec))
    arm_id = design.arms[0].arm_id
    store.append("ARM_CLOSED", {"arm_id": arm_id, "reason": "unit closure"})
    changed = replace(design, study_id="changed-design", horizons=(10.0,), seeds=(19,),
                      methods=tuple(replace(m, steps=8) for m in design.methods))
    next_prepared = freeze_design(changed)
    next_spec = next_prepared.study_spec(expected_hash=next_prepared.manifest_hash)
    store.register(next_spec, digest(next_spec))
    assert BudgetLedger(store).balance(arm_id)["closed"]
    renamed = replace(changed, study_id="renamed-design",
                      arms=(replace(changed.arms[0], arm_id="fresh-name"), changed.arms[1]))
    renamed_prepared = freeze_design(renamed)
    renamed_spec = renamed_prepared.study_spec(expected_hash=renamed_prepared.manifest_hash)
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        store.register(renamed_spec, digest(renamed_spec))
