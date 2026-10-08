"""Disposable method consumers; none of these grants authorize research data."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_evidence import export_evidence
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.calibration_bindings import StudyCalibration, require_consumer_support
from experiments.pirc27.design import StudyFunctional, StudyMethod, freeze_design
from experiments.pirc27.plugin import propagation_plugin
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_calibrated_study_matrix import declared_design
from tests.test_managed_probability_calibration import prepared
from tests.test_probability_calibration_owner import pointer
from tests.test_propagation_study_design import fixture


def test_complete_declaration_reaches_owner_boundary_without_claiming_authority():
    frozen = freeze_design(declared_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    cell = next(c for c in spec["cells"] if "execution" in c)
    assert require_consumer_support(spec, cell) == cell["calibration_binding"]


@pytest.fixture(scope="module")
def consumers(tmp_path_factory):
    from application.probability_calibration_admission import prepare_probability_calibration
    from experiments.pirc27.preparation import _design_from_document

    root = tmp_path_factory.mktemp("calibrated-method-consumers")
    store, source_spec, registry = prepared(root)
    store.register(source_spec, digest(source_spec))
    source_result = SharedRunner(store, registry).run_cell(source_spec["study_id"],
        digest(source_spec["cells"][0]), budget=BudgetSpec(60, category="pilot"))
    assert source_result["state"] == "SUCCEEDED", source_result
    source_pointer = pointer(source_spec, source_result)
    consumer = "calibrated-method-unit"
    grant = store.authorization("execution-fixture")
    store.authorize({**grant, "version": "consumer-v1", "consumer_study_ids": [consumer]})
    source_pointer["source_authorization_version"] = "consumer-v1"
    proof = prepare_probability_calibration(store, source_pointer, consumer_study_id=consumer)
    design = replace(fixture(methods=(StudyMethod("exact", samples=8, steps=4),
        StudyMethod("euler", samples=8, steps=4)),
        functionals=(StudyFunctional("tail", "endpoint-halfspace", target_probability=1e-6),),
        horizons=(1.,)), study_id=consumer)
    model = design.models[0]
    entry = StudyCalibration("tail", model.family_id, model.package.package_hash, 1.,
        status="CALIBRATED", reason="actual settled disposable source, not scientific qualification",
        geometry_document=encode(proof["geometry"]), source_pointer_document=encode(source_pointer),
        source_evidence_hash=proof["evidence_hash"], consumer_study_id=consumer)
    frozen = freeze_design(replace(design, calibrations=(entry,)))
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    plugin = propagation_plugin()
    target_grant = admit_fixture(store, spec, plugin, root, fixture_prefix="consumer-",
        preserve_execution=True, legacy_upstream=False)
    # This fixture registers its own explicit protocol. Recompile with those
    # actual identities; do not retain an old design hash or alter any cell.
    document = frozen.manifest()
    document.update({k: spec[k] for k in ("protocol_hash", "data_hash")})
    rebound = freeze_design(_design_from_document(document))
    assert rebound.study_spec(expected_hash=rebound.manifest_hash)["cells"] == spec["cells"]
    spec["propagation_design_hash"] = rebound.manifest_hash
    spec["runtime_binding"] = {"root": str(root.resolve()), "store_id": store.store_id}
    store.register(spec, digest(spec))
    registry.register(plugin)
    source_balance = BudgetLedger(store).balance(source_spec["arms"][0]["arm_id"])
    return store, spec, registry, target_grant, source_spec, source_balance, proof


def test_actual_methods_share_paid_region_and_reopened_reuse_does_not_recalibrate(consumers, monkeypatch):
    store, spec, registry, _, source_spec, source_balance, proof = consumers
    import inference.affine_probability_calibration as calibration
    monkeypatch.setattr(calibration, "calibrate_affine_halfspace",
        lambda *a, **k: pytest.fail("consumer recalibrated outside the original charged source"))
    runner = SharedRunner(store, registry)
    for cell in spec["cells"]:
        result = runner.run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60, category="smoke"))
        assert result["state"] == "SUCCEEDED", result
        artifact = json.loads(store.read_artifact(result["artifact_id"], purpose="evaluate",
            authorization=store.authorization("consumer-execution-fixture")))
        receipt = store.manifest("admission-" + artifact["admission_hash"])
        assert receipt["documents"]["probability_calibration"] == proof
        assert "propagation_qualification" not in receipt["documents"]
        assert artifact["qualification"] == "fixture"
        assert artifact["forecast"]["request_hash"] == digest(cell["propagation_request"])
        assert artifact["forecast"]["functional"]["status"] in {"SUCCEEDED", "UNRESOLVED", "LOW_ESS"}
        before = BudgetLedger(store).balance(cell["arm_id"])
        reused = SharedRunner(ResearchStore(store.path.parent, store.store_id), registry).run_cell(
            spec["study_id"], digest(cell))
        assert reused["reused"] and reused["artifact_id"] == result["artifact_id"]
        assert BudgetLedger(store).balance(cell["arm_id"]) == before
    assert BudgetLedger(store).balance(source_spec["arms"][0]["arm_id"]) == source_balance
    source_workers = [e for e in store.events() if e["event_kind"] == "WORKER_STARTED"
        and e["payload"]["attempt_id"] == proof["source_attempt"]["attempt_id"]]
    assert len(source_workers) == 1
    assert len({c["propagation_request"]["threshold"] for c in spec["cells"]}) == 1
    assert len({c["arm_id"] for c in spec["cells"]}) == 2


@pytest.mark.parametrize("fault", ["stripped", "table", "other-cell", "orphan", "subset", "geometry"])
def test_declaration_substitutions_refuse_before_store_access(fault):
    frozen = freeze_design(declared_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    cell = next(c for c in spec["cells"] if "execution" in c)
    if fault == "stripped":
        cell.pop("calibration_binding")
    elif fault == "table":
        spec["propagation_design"]["axis_manifest"]["calibrations"].pop()
    elif fault == "other-cell":
        cell = deepcopy(cell)
        cell["seed"] += 1
    elif fault == "orphan":
        spec["propagation_design"]["axis_manifest"]["functionals"][0].pop("target_probability")
    elif fault == "subset":
        spec["cells"].pop()
    else:
        cell["propagation_request"]["threshold"] += 1
    with pytest.raises(ResearchError):
        AdmissionGate.prepare(None, spec, cell, None, "not-created")


def test_calibrated_study_cannot_disclose_without_independent_export_support(consumers, monkeypatch):
    store, spec, _, grant, *_ = consumers
    monkeypatch.setattr(store, "read_artifact", lambda *a, **k: pytest.fail("export read before refusal"))
    before = store.events()
    with pytest.raises(ResearchError, match="independent calibration export"):
        export_evidence(store, spec["study_id"], grant)
    assert not any(e["event_kind"] == "MANIFEST" and e["payload"]["object_id"].startswith("bundle-")
        for e in store.events()[len(before):])


@pytest.mark.parametrize("fault", ["consumer", "purpose", "expiry", "version"])
def test_current_selected_source_permission_is_required_before_any_artifact_read(consumers, monkeypatch, fault):
    from application.probability_calibration_consumption import prepare_calibrated_consumer
    store, spec, *_ = consumers
    original = store.authorization
    def permission(authorization_id, *, version=None):
        grant = original(authorization_id, version=version)
        if version == "consumer-v1":
            grant = deepcopy(grant)
            key, value = {"consumer": ("consumer_study_ids", []), "purpose": ("purposes", []),
                "expiry": ("expires_at", "2000-01-01T00:00:00+00:00"), "version": ("version", "other-version")}[fault]
            grant[key] = value
        return grant
    monkeypatch.setattr(store, "authorization", permission)
    monkeypatch.setattr(store, "read_artifact", lambda *a, **k: pytest.fail("read before source permission"))
    with pytest.raises(ResearchError, match="permission"):
        prepare_calibrated_consumer(store, spec, spec["cells"][0])


@pytest.mark.parametrize("fault", ["missing", "hash-only", "subset", "hash"])
def test_formal_region_table_must_be_frozen_in_full_before_source_disclosure(consumers, monkeypatch, fault):
    from application.probability_calibration_consumption import prepare_calibrated_consumer
    store, spec, *_ = consumers
    table = spec["propagation_design"]["axis_manifest"]["calibrations"]
    prereg = {"probability_calibration_bindings": deepcopy(table),
        "probability_calibration_bindings_hash": digest(table)}
    if fault == "missing":
        prereg = {}
    elif fault == "hash-only":
        prereg.pop("probability_calibration_bindings")
    elif fault == "subset":
        prereg["probability_calibration_bindings"].pop()
    else:
        prereg["probability_calibration_bindings_hash"] = "0"*64
    monkeypatch.setattr(store, "read_artifact", lambda *a, **k: pytest.fail("source read before table freeze"))
    with pytest.raises(ResearchError, match="preregistration"):
        prepare_calibrated_consumer(store, spec, spec["cells"][0], preregistration=prereg)


@pytest.mark.parametrize("fault", ["evidence", "geometry"])
def test_resealed_declaration_is_not_a_substitute_for_actual_settled_owner(consumers, fault):
    from application.probability_calibration_consumption import prepare_calibrated_consumer
    from experiments.pirc27.preparation import _design_from_document
    store, spec, *_ = consumers
    document = {**spec, **spec["propagation_design"]}
    design = _design_from_document(document)
    entry = design.calibrations[0]
    if fault == "evidence":
        entry = replace(entry, source_evidence_hash="a"*64)
    else:
        geometry = deepcopy(entry.manifest()["geometry"])
        geometry["threshold"] += 1
        entry = replace(entry, geometry_document=encode(geometry))
    frozen = freeze_design(replace(design, calibrations=(entry,)))
    substituted = frozen.study_spec(expected_hash=frozen.manifest_hash)
    # Every derived request/hash is legitimately regenerated, but native source
    # evidence still owns the threshold. No target data or worker is consumed.
    with pytest.raises(ResearchError, match="fresh settled owner"):
        prepare_calibrated_consumer(store, substituted, substituted["cells"][0])


@pytest.mark.parametrize("fault", ["missing-proof", "request", "model", "horizon-type"])
def test_saved_receipt_and_current_output_cannot_strip_or_swap_geometry_provenance(consumers, fault):
    store, spec, registry, grant, *_ = consumers
    cell = spec["cells"][0]
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell))
    assert outcome["state"] == "SUCCEEDED", outcome
    result = json.loads(store.read_artifact(outcome["artifact_id"], purpose="evaluate", authorization=grant))
    receipt = store.manifest("admission-"+result["admission_hash"])
    if fault == "missing-proof":
        receipt["documents"].pop("probability_calibration")
    elif fault == "horizon-type":
        result["forecast"]["horizons"] = [True]
        result["output_hash"] = digest({k: result[k] for k in ("metrics", "forecast", "fit", "source_schema")})
    else:
        result["forecast"]["request_hash" if fault == "request" else "model_package_hash"] = "a"*64
        result["output_hash"] = digest({k: result[k] for k in ("metrics", "forecast", "fit", "source_schema")})
    with pytest.raises(ResearchError, match="geometry admission|current output"):
        AdmissionGate(store).result_validator(receipt, spec, cell, propagation_plugin())(result)


def test_calibration_proof_does_not_replace_formal_method_qualification(consumers, monkeypatch):
    from application.propagation_qualification_admission import prepare_managed_qualification
    store, spec, _, _, _, _, proof = consumers
    package = store.manifest("package-"+spec["admission"]["package_hash"])
    package["payload"]["probability_calibration"] = deepcopy(proof)
    monkeypatch.setattr(store, "read_artifact", lambda *a, **k: pytest.fail("read without method-specific source"))
    with pytest.raises(ResearchError):
        prepare_managed_qualification(store, spec, spec["cells"][0], package, {})


@pytest.mark.parametrize("fault", ["axes", "functionals", "cycle", "too-many-functionals"])
def test_malformed_or_unbounded_consumer_axes_refuse_before_store_access(fault):
    frozen = freeze_design(declared_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    axes = spec["propagation_design"]["axis_manifest"]
    if fault == "axes":
        spec["propagation_design"]["axis_manifest"] = None
    elif fault == "functionals":
        axes["functionals"] = None
    elif fault == "cycle":
        axes["calibrations"].append(axes)
    else:
        axes["functionals"] *= 65
    with pytest.raises(ResearchError):
        AdmissionGate.prepare(None, spec, spec["cells"][0], None, "not-created")


def test_preregistered_table_cannot_substitute_equal_python_boolean_for_physical_horizon(monkeypatch):
    from application import probability_calibration_consumption as consumption
    frozen = freeze_design(declared_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    table = spec["propagation_design"]["axis_manifest"]["calibrations"]
    substituted = deepcopy(table)
    substituted[0]["horizon"] = True  # True == 1., but not canonical physical metadata.
    prereg = {"probability_calibration_bindings": substituted,
        "probability_calibration_bindings_hash": digest(table)}
    monkeypatch.setattr(consumption, "prepare_probability_calibration",
        lambda *a, **k: pytest.fail("source owner reached before exact preregistration"))
    with pytest.raises(ResearchError, match="preregistration"):
        consumption.prepare_calibrated_consumer(None, spec, spec["cells"][0], preregistration=prereg)


def test_saved_proof_cannot_replace_false_qualification_with_integer_zero(consumers):
    from application.probability_calibration_consumption import validate_calibrated_result
    store, spec, _, _, _, _, proof = consumers
    cell = spec["cells"][0]
    saved = deepcopy(proof)
    saved["scientific_qualification"] = 0  # 0 == False does not preserve the proof hash/type.
    receipt = {"documents": {"probability_calibration": saved}}
    result = {"forecast": {"model_package_hash": proof["geometry"]["model_package_hash"],
        "request_hash": digest(cell["propagation_request"]), "horizons": [cell["horizon"]]}}
    with pytest.raises(ResearchError, match="saved calibration proof"):
        validate_calibrated_result(store, receipt, spec, cell, result)


def test_legacy_bundle_cannot_hide_an_unrecognized_embedded_calibration_during_publication(consumers):
    from application.research_evidence import authorize_evidence_publication
    store, _, _, _, source_spec, *_ = consumers
    grant = store.authorization("execution-fixture")
    bundle = export_evidence(store, source_spec["study_id"], grant)
    bundle["cells"][0]["admission"]["documents"]["probability_calibration"] = {
        "kind": "explicit opaque synthetic control, not authorized source proof"}
    bundle["bundle_hash"] = digest({k: v for k, v in bundle.items() if k != "bundle_hash"})
    # Explicit disposable operator publication reaches the additional scope
    # guard. An unpublished/tampered object is already refused by old checks.
    store.publish("bundle-"+bundle["bundle_hash"], bundle)
    with pytest.raises(ResearchError, match="independent calibration export"):
        authorize_evidence_publication(store, bundle, grant)


def test_exact_preregistered_full_table_reuses_actual_geometry_without_method_promotion(consumers):
    from application.probability_calibration_consumption import prepare_calibrated_consumer
    store, spec, _, _, source_spec, _, proof = consumers
    table = spec["propagation_design"]["axis_manifest"]["calibrations"]
    prereg = {"probability_calibration_bindings": deepcopy(table),
        "probability_calibration_bindings_hash": digest(table)}
    before = BudgetLedger(store).balance(source_spec["arms"][0]["arm_id"])
    accepted = prepare_calibrated_consumer(store, spec, spec["cells"][0], preregistration=prereg)
    assert accepted == proof
    assert accepted["scientific_qualification"] is accepted["method_qualification"] is False
    assert BudgetLedger(store).balance(source_spec["arms"][0]["arm_id"]) == before


@pytest.mark.parametrize("fault", ["null", "cycle", "oversized"])
def test_preregistration_table_is_bounded_before_hashing_or_source_access(monkeypatch, fault):
    from application import probability_calibration_consumption as consumption
    frozen = freeze_design(declared_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    table = spec["propagation_design"]["axis_manifest"]["calibrations"]
    declared = None if fault == "null" else [] if fault == "cycle" else [table[0]]*10001
    if fault == "cycle":
        declared.append(declared)
    prereg = {"probability_calibration_bindings": declared,
        "probability_calibration_bindings_hash": digest(table)}
    monkeypatch.setattr(consumption, "prepare_probability_calibration",
        lambda *a, **k: pytest.fail("source owner called before bounded metadata checks"))
    with pytest.raises(ResearchError):
        consumption.prepare_calibrated_consumer(None, spec, spec["cells"][0], preregistration=prereg)
