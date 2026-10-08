"""Frozen physical probability regions, not research or source qualification."""

from dataclasses import replace
from copy import deepcopy

import pytest

from domain.errors import DataValidationError
from domain.probability_calibration import AffineHalfspaceCalibrationPolicy, halfspace_geometry
from domain.propagation import PropagationRequest
from experiments.pirc25.affine import code_hash
from experiments.pirc27.calibration_bindings import StudyCalibration, MAX_DOCUMENT_BYTES
from experiments.pirc27.design import StudyFunctional, StudyMethod, freeze_design
from experiments.pirc27.preparation import reload_design
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_propagation_study_design import fixture


def declared_design():
    # Explicit unverified proof identities exercise preparation only. No claim
    # that these hashes are genuine native source/owner evidence.
    functional = StudyFunctional("tail", "endpoint-halfspace", target_probability=1e-6)
    design = fixture(functionals=(functional,), methods=(StudyMethod("exact", samples=2),
        StudyMethod("euler", samples=8, steps=4), StudyMethod("pde", samples=2)), seeds=(11, 19))
    model = design.models[0]
    entries = []
    for horizon in design.horizons:
        request = PropagationRequest("declared-template", model.package.package_hash,
            model.initial_mean, model.initial_covariance, design.origin, design.history_cutoff,
            (horizon,), "endpoint-halfspace", 0, "declared", "existing-reference", samples=2)
        policy = AffineHalfspaceCalibrationPolicy(request.request_hash, model.package.package_hash,
            code_hash(), 1e-6, 1e-8, 1e-10, 200000, 60.)
        pointer = {"schema_version": "managed-affine-halfspace-calibration-v1", "policy": policy.manifest(),
            "source_attempt_id": "declared-source", "source_artifact_id": "a"*64,
            "source_authorization_id": "declared-grant", "source_authorization_version": None}
        geometry = halfspace_geometry(model.package, replace(request, threshold=horizon+2), causal_input_hash="b"*64)
        entries.append(StudyCalibration("tail", model.family_id, model.package.package_hash, horizon,
            status="CALIBRATED", geometry_document=encode(geometry), source_pointer_document=encode(pointer),
            source_evidence_hash="c"*64, consumer_study_id=design.study_id,
            reason="declared engineering control, no native proof"))
    return replace(design, calibrations=tuple(entries))


def test_every_method_and_seed_share_one_frozen_region_per_physical_geometry():
    design = declared_design()
    frozen = freeze_design(design)
    document = frozen.manifest()
    assert document["expected_cells"] == len(document["matrix"]) == 12
    assert len(document["arms"]) == 3 and len(document["axis_manifest"]["calibrations"]) == 2
    for horizon in design.horizons:
        rows = [r for r in document["matrix"] if r["horizon"] == horizon]
        assert {r["cell"]["propagation_request"]["threshold"] for r in rows} == {horizon+2}
        assert len({r["cell"]["calibration_binding"]["geometry_hash"] for r in rows}) == 1
        assert len({r["cell"]["calibration_binding"]["binding_hash"] for r in rows}) == 1
    assert all(r["disposition"] == "NOT_IMPLEMENTED" for r in document["matrix"] if r["method"] == "pde")
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert len(spec["cells"]) == 12 and "admission" not in spec
    assert document["qualification"] == "preparation-only-not-scientific"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "horizon", "probability", "mean", "normal", "consumer"])
def test_incomplete_or_substituted_table_cannot_be_a_frozen_comparison(fault):
    design = declared_design()
    first, second = design.calibrations
    if fault == "missing":
        entries = (first,)
    elif fault == "duplicate":
        entries = (first, first)
    elif fault == "extra":
        entries = (first, second, replace(first, horizon=100.))
    elif fault == "horizon":
        entries = (replace(first, horizon=2.), second)
    elif fault == "consumer":
        entries = (replace(first, consumer_study_id="other-study"), second)
    elif fault == "probability":
        pointer = deepcopy(first.manifest()["source_pointer"])
        pointer["policy"]["target_probability"] = .01
        entries = (replace(first, source_pointer_document=encode(pointer)), second)
    else:
        geometry = deepcopy(first.manifest()["geometry"])
        geometry["initial_mean" if fault == "mean" else "normal"][0] += 1
        entries = (replace(first, geometry_document=encode(geometry)), second)
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(replace(design, calibrations=entries))


def test_missing_or_failed_calibration_retains_all_rows_without_scalar_fallback():
    design = declared_design()
    first, second = design.calibrations
    unavailable = replace(first, status="UNAVAILABLE", geometry_document=None, source_pointer_document=None,
        source_evidence_hash=None, consumer_study_id=None, reason="no qualified source")
    failed = replace(second, status="FAILED", geometry_document=None, source_evidence_hash=None,
        consumer_study_id=None, reason="original bounded numerical attempt failed")
    doc = freeze_design(replace(design, calibrations=(unavailable, failed))).manifest()
    assert len(doc["matrix"]) == doc["expected_cells"] == 12
    for row in doc["matrix"]:
        assert "execution" not in row["cell"]
        assert row["disposition"] == ("NOT_IMPLEMENTED" if row["method"] == "pde" else "INELIGIBLE")
        assert row["cell"]["calibration_binding"]["status"] in {"FAILED", "UNAVAILABLE"}


def test_table_reload_regenerates_all_rows_and_refuses_resealed_region_changes(tmp_path):
    frozen = freeze_design(declared_design())
    path = tmp_path/"calibrated-design.json"
    path.write_bytes(encode(frozen.manifest()))
    assert reload_design(path, expected_hash=frozen.manifest_hash).manifest_hash == frozen.manifest_hash
    document = frozen.manifest()
    document["matrix"][0]["cell"]["propagation_request"]["threshold"] += 1
    path.write_bytes(encode(document))
    with pytest.raises(ResearchError):
        reload_design(path, expected_hash=digest(document))


def test_legacy_design_omits_new_optional_fields_and_retains_its_old_pairing():
    design = fixture()
    doc = freeze_design(design).manifest()
    assert "calibrations" not in doc["axis_manifest"]
    assert "target_probability" not in doc["axis_manifest"]["functionals"][0]
    assert all("calibration_binding" not in row["cell"] for row in doc["matrix"])


def test_cyclic_table_inputs_refuse_before_serialization():
    value = {}
    value["geometry"] = value
    with pytest.raises((DataValidationError, ResearchError)):
        StudyCalibration.from_manifest(value)


@pytest.mark.parametrize("field,value", [("horizon", True), ("horizon", 2**2000),
    ("horizon", float("nan")), ("status", ["CALIBRATED"]), ("reason", "x"*257),
    ("geometry_document", b"x"*(MAX_DOCUMENT_BYTES+1)),
    ("geometry_document", b'{"threshold":0,"threshold":1}')])
def test_scalar_shape_and_canonical_metadata_bounds(field, value):
    design = declared_design()
    with pytest.raises((DataValidationError, ResearchError)):
        replace(design.calibrations[0], **{field: value}).manifest()


def test_entry_roundtrip_is_canonical_immutable_and_refuses_resealed_bad_hash():
    entry = declared_design().calibrations[0]
    document = entry.manifest()
    assert StudyCalibration.from_manifest(document).manifest() == document
    document["geometry"]["threshold"] += 1
    with pytest.raises(DataValidationError):
        StudyCalibration.from_manifest(document)
    assert entry.manifest()["geometry"]["threshold"] == 3.


@pytest.mark.parametrize("strip_binding", [False, True])
@pytest.mark.parametrize("builtin", [False, True])
def test_declared_calibration_cannot_fall_through_generic_admission_before_store_access(strip_binding, builtin):
    from application.research_admission import AdmissionGate
    frozen = freeze_design(declared_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    cell = spec["cells"][0]
    if strip_binding:
        cell.pop("calibration_binding")
    # No store/plugin/attempt exists: refusal must precede even plan evaluation,
    # fixture admission, registry writes, permission lookups and protected I/O.
    with pytest.raises(ResearchError, match="settled consumer"):
        AdmissionGate.prepare(None, spec, cell, None, "not-created", builtin_fixture=builtin)


@pytest.mark.parametrize("private", [False, True])
def test_case_regions_use_actual_laws_share_rule_stratum_and_preserve_source_pairing(private):
    from tests.test_propagation_input_cases import cases_design
    from experiments.pirc27.input_cases import input_binding, validate_cell_input_binding
    base = declared_design()
    cases = cases_design(private=private)
    design = replace(base, input_cases=cases.input_cases, input_policy=cases.input_policy)
    model = design.models[0]
    entries = []
    for original in base.calibrations:
        for case in design.input_cases:
            functional = design.functionals[0]
            request = PropagationRequest("case-template", model.package.package_hash, case.initial_mean,
                case.initial_covariance, case.origin, case.history_cutoff, (original.horizon,),
                functional.kind, 0, "declared", "existing-reference", normal=functional.normal,
                threshold=original.horizon+case.origin+2)
            geometry = halfspace_geometry(model.package, request,
                causal_input_hash=input_binding(case, design.input_policy)["binding_hash"])
            entries.append(replace(original, input_case_id=case.instance_id, geometry_document=encode(geometry)))
    frozen = freeze_design(replace(design, calibrations=tuple(entries)))
    rows = frozen.manifest()["matrix"]
    assert len(rows) == 24
    assert len({digest(r["cell"]["comparison_dimensions"]) for r in rows}) == 1
    for row in rows:
        cell = row["cell"]
        validate_cell_input_binding(cell)
        case = next(c for c in cases.input_cases if c.instance_id == cell["instance_id"])
        assert cell["propagation_request"]["threshold"] == row["horizon"]+case.origin+2
        if private:
            assert "execution" not in cell and cell["visibility"] == "restricted"


def test_missing_nonlinear_and_extreme_probability_slots_remain_full_unavailable_rows():
    design = fixture(nonlinear=True, functionals=(StudyFunctional("tail", "endpoint-halfspace", target_probability=1e-12),))
    model = design.models[0]
    entries = tuple(StudyCalibration("tail", model.family_id, model.package.package_hash, h) for h in design.horizons)
    document = freeze_design(replace(design, calibrations=entries)).manifest()
    assert document["expected_cells"] == len(document["matrix"]) == 8
    assert all(r["disposition"] == "INELIGIBLE" and "execution" not in r["cell"] for r in document["matrix"])


def test_deterministic_alias_refusal_precedes_even_calibrated_table_expansion(monkeypatch):
    from experiments.pirc27 import design as module
    design = declared_design()
    methods = (StudyMethod("exact", samples=2, configuration_id="two"),
        StudyMethod("exact", samples=4, configuration_id="four"))
    design = replace(design, methods=methods, arms=fixture(methods=methods,
        functionals=design.functionals).arms)
    monkeypatch.setattr(module, "product", lambda *a: pytest.fail("calibrated alias expanded"))
    with pytest.raises(DataValidationError, match="duplicate numerical configuration"):
        freeze_design(design)
