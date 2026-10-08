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
