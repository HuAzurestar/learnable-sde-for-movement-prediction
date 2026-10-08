"""Actual disposable source ownership, never formal research qualification."""

from copy import deepcopy
from dataclasses import replace

import pytest

from application.probability_calibration_admission import prepare_probability_calibration, saved_calibration_request
from application.research_budget import BudgetLedger, BudgetSpec
from domain.errors import DataValidationError
from domain.probability_calibration import halfspace_geometry
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_affine_probability_calibration import example
from tests.test_managed_probability_calibration import prepared


def pointer(spec, result):
    return {"schema_version": "managed-affine-halfspace-calibration-v1",
        "policy": spec["cells"][0]["probability_calibration_policy"],
        "source_attempt_id": result["attempt_id"], "source_artifact_id": result["artifact_id"],
        "source_authorization_id": "execution-fixture", "source_authorization_version": None}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    store, spec, registry = prepared(tmp_path_factory.mktemp("calibration-owner"))
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert result["state"] == "SUCCEEDED", result
    return store, spec, pointer(spec, result)


@pytest.mark.parametrize("changes", [{"request_id": "another"}, {"seed": 42},
    {"coupling_id": "other"}, {"arm_id": "euler"}, {"steps": 64}, {"samples": 100},
    {"chunk_size": 32}, {"tolerance": .1}])
def test_geometry_does_not_partition_a_physical_event_by_method_configuration(changes):
    package, request, _ = example()
    assert halfspace_geometry(package, request, causal_input_hash="a"*64) == halfspace_geometry(
        package, replace(request, **changes), causal_input_hash="a"*64)


@pytest.mark.parametrize("changes", [{"initial_mean": (1., 2., 3., 4.)},
    {"initial_covariance": tuple(tuple(.1 if i == j else 0. for j in range(4)) for i in range(4))},
    {"origin": 1.}, {"history_cutoff": -1.}, {"horizons": (2.,)},
    {"normal": (0., 1., 0., 0.)}, {"threshold": 1.}, {"closed": False}])
def test_geometry_preserves_law_time_and_actual_event(changes):
    package, request, _ = example()
    assert halfspace_geometry(package, request, causal_input_hash="a"*64) != halfspace_geometry(
        package, replace(request, **changes), causal_input_hash="a"*64)


def test_geometry_requires_position_halfspace_and_explicit_causal_identity():
    package, request, _ = example()
    with pytest.raises(DataValidationError):
        halfspace_geometry(package, request, causal_input_hash=True)
    with pytest.raises(DataValidationError):
        halfspace_geometry(package, replace(request, normal=(0., 0., 1., 0.)), causal_input_hash="a"*64)
    with pytest.raises(DataValidationError):
        halfspace_geometry(package, replace(request, functional="endpoint-x"), causal_input_hash="a"*64)
    first = halfspace_geometry(package, request, causal_input_hash="a"*64)
    first["initial_mean"][0] = 999
    assert first != halfspace_geometry(package, request, causal_input_hash="a"*64)
    assert halfspace_geometry(package, request, causal_input_hash="a"*64) != halfspace_geometry(
        package, request, causal_input_hash="b"*64)


def test_actual_owner_binds_settled_source_and_reopened_reuse_without_free_numerics(source, monkeypatch):
    store, spec, ref = source
    import inference.affine_reference as reference
    import inference.affine_probability_calibration as calibration
    def forbidden(*args, **kwargs):
        pytest.fail("owner executed worker-side numerical reference")
    for name in ("Arithmetic", "_bound_affine_reference", "_cdf"):
        monkeypatch.setattr(reference, name, forbidden)
    for name in ("calibrate_affine_halfspace", "_quantile"):
        monkeypatch.setattr(calibration, name, forbidden)
    before = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])
    evidence = prepare_probability_calibration(store, ref, consumer_study_id=spec["study_id"])
    assert evidence["cost_status"] == "OWNER_SETTLED"
    assert evidence["admission_status"] == "NOT_ADMITTED"
    assert evidence["scientific_qualification"] is evidence["method_qualification"] is False
    assert evidence["geometry_hash"] == digest(evidence["geometry"])
    cost = evidence["source_cost"]
    assert cost["source_id"] == digest([store.store_id, ref["source_attempt_id"]])
    assert 0 < cost["charged_ms"] == evidence["settlement_event"]["payload"]["charged_ms"]
    reopened = ResearchStore(store.path.parent, store.store_id)
    again = prepare_probability_calibration(reopened, ref, consumer_study_id=spec["study_id"])
    assert again == evidence
    assert BudgetLedger(store).balance(spec["arms"][0]["arm_id"]) == before
    assert len([e for e in store.events() if e["event_kind"] == "WORKER_STARTED"]) == 1


@pytest.mark.parametrize("fault", ["policy", "artifact", "consumer", "extra"])
def test_pointer_and_consumer_substitutions_refuse_before_artifact_read(source, monkeypatch, fault):
    store, spec, original = source
    ref = deepcopy(original)
    consumer = spec["study_id"]
    if fault == "policy":
        ref["policy"]["target_probability"] = .01
    elif fault == "artifact":
        ref["source_artifact_id"] = "0"*64
    elif fault == "consumer":
        consumer = "another-study"
    else:
        ref["trusted"] = True
    monkeypatch.setattr(store, "read_artifact", lambda *args, **kwargs: pytest.fail("source read before pointer/consumer refusal"))
    with pytest.raises(ResearchError):
        prepare_probability_calibration(store, ref, consumer_study_id=consumer)


@pytest.mark.parametrize("fault", ["expired", "purpose", "protocol", "version"])
def test_current_source_read_authority_is_not_inferred(source, monkeypatch, fault):
    store, spec, ref = source
    grant = deepcopy(store.authorization(ref["source_authorization_id"]))
    if fault == "expired":
        grant["expires_at"] = "2000-01-01T00:00:00+00:00"
    elif fault == "purpose":
        grant["purposes"] = ["execute"]
    elif fault == "protocol":
        grant["protocol_hash"] = "0"*64
    else:
        grant["version"] = "not-selected"
    monkeypatch.setattr(store, "authorization", lambda *args, **kwargs: grant)
    monkeypatch.setattr(store, "read_artifact", lambda *args, **kwargs: pytest.fail("protected source read before grant refusal"))
    with pytest.raises(ResearchError):
        prepare_probability_calibration(store, ref, consumer_study_id=spec["study_id"])


@pytest.mark.parametrize("fault", ["missing-stop", "confirmation", "elapsed", "charged", "reserved", "order"])
def test_original_native_stop_order_and_budget_are_required(source, monkeypatch, fault):
    store, spec, ref = source
    events = deepcopy(store.events())
    stop = next(e for e in events if e["event_kind"] == "WORKER_TREE_STOPPED")
    settlement = next(e for e in events if e["event_kind"] == "SETTLE")
    if fault == "missing-stop":
        events.remove(stop)
    elif fault == "confirmation":
        stop["payload"]["confirmation"] = "not-stopped"
    elif fault == "elapsed":
        stop["payload"]["observed_elapsed_ms"] += 1
    elif fault == "charged":
        settlement["payload"]["charged_ms"] = 0
    elif fault == "reserved":
        settlement["payload"]["reserved_ms"] = 60001
    else:
        stop["sequence"] = settlement["sequence"]+1
    monkeypatch.setattr(store, "events", lambda: events)
    with pytest.raises(ResearchError):
        prepare_probability_calibration(store, ref, consumer_study_id=spec["study_id"])


@pytest.mark.parametrize("fault", ["threshold", "relative-error", "operations", "quantile", "scope", "numeric-flag", "certificate-request"])
def test_resealed_saved_scalar_substitutions_do_not_pass_owner_checks(source, fault):
    store, spec, ref = source
    from application.propagation_execution import validate_propagation_cell
    from domain.probability_calibration import AffineHalfspaceCalibrationPolicy
    import json
    package, request, _, _ = validate_propagation_cell(spec, spec["cells"][0])
    analysis = json.loads(store.read_artifact(ref["source_artifact_id"], purpose="evaluate",
        authorization=store.authorization("execution-fixture")))["forecast"]["probability_calibration_analysis"]
    if fault == "threshold":
        analysis["threshold"] += 1
    elif fault == "relative-error":
        analysis["relative_probability_error_upper"] = 0
    elif fault == "operations":
        analysis["operation_counts"]["projection_and_quantile"] += 1
    elif fault == "quantile":
        analysis["executed_quantile_bisections"] = 47
    elif fault == "scope":
        analysis["scope"] = "qualified-all-regions"
    elif fault == "numeric-flag":
        analysis["scientific_qualification"] = True
    else:
        analysis["probability_certificate"]["request_hash"] = "0"*64
        analysis["probability_certificate_hash"] = digest(analysis["probability_certificate"])
    analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    with pytest.raises(ResearchError):
        saved_calibration_request(analysis, AffineHalfspaceCalibrationPolicy.from_manifest(ref["policy"]), package, request)


def test_actual_numerical_failure_remains_charged_and_cannot_create_geometry(tmp_path):
    store, spec, registry = prepared(tmp_path, policy_changes={"maximum_relative_probability_error": 1e-30,
        "maximum_relative_probability_width": 1e-50})
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert result["state"] == "SUCCEEDED", result
    with pytest.raises(ResearchError, match="saved numerical checks"):
        prepare_probability_calibration(store, pointer(spec, result), consumer_study_id=spec["study_id"])
    assert BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"] > 0


@pytest.mark.parametrize("scale", [1e-100, 1e100, -1e-100])
def test_saved_owner_keeps_actual_normalized_candidate_and_coarse_physical_envelope(scale, monkeypatch):
    from inference.affine_probability_calibration import calibrate_affine_halfspace
    package, request, policy = example()
    request = replace(request, normal=(scale, 0., 0., 0.))
    policy = replace(policy, source_request_hash=request.request_hash)
    analysis = calibrate_affine_halfspace(package, request, policy)
    assert analysis["status"] == "PASSED"
    import inference.affine_reference as reference
    monkeypatch.setattr(reference, "Arithmetic", lambda *args, **kwargs: pytest.fail("owner replayed interval reference"))
    assert saved_calibration_request(analysis, policy, package, request).threshold == analysis["threshold"]


def test_saved_owner_rejects_cycles_before_encoding(monkeypatch):
    import application.probability_calibration_admission as owner
    package, request, policy = example()
    analysis = {}
    analysis["cycle"] = analysis
    monkeypatch.setattr(owner, "encode", lambda *args: pytest.fail("unbounded caller document encoded"))
    with pytest.raises(ResearchError, match="structure"):
        saved_calibration_request(analysis, policy, package, request)


def test_actual_settled_source_can_be_frozen_with_its_selected_string_grant_version(source):
    from experiments.pirc27.calibration_bindings import StudyCalibration
    from infrastructure.research_store import encode
    store, spec, original = source
    consumer = "versioned-geometry-consumer"
    original_grant = store.authorization(original["source_authorization_id"])
    grant = {**original_grant,
        "version": "calibration-consumer-v1", "consumer_study_ids": [consumer]}
    # Disposable synthetic test store only. This creates a distinct immutable
    # version, not an overwrite, official grant, source budget or new worker.
    store.authorize(grant)
    ref = {**original, "source_authorization_version": grant["version"]}
    before = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])
    evidence = prepare_probability_calibration(store, ref, consumer_study_id=consumer)
    geometry = evidence["geometry"]
    entry = StudyCalibration("tail", "affine-stable-v1", geometry["model_package_hash"], geometry["horizon"],
        status="CALIBRATED", geometry_document=encode(geometry), source_pointer_document=encode(ref),
        source_evidence_hash=evidence["evidence_hash"], consumer_study_id=consumer)
    document = entry.manifest()
    assert StudyCalibration.from_manifest(document).manifest() == document
    assert document["source_pointer"]["source_authorization_version"] == grant["version"]
    assert evidence["authorization"] == grant
    assert store.authorization(original["source_authorization_id"]) == original_grant
    reopened = ResearchStore(store.path.parent, store.store_id)
    assert prepare_probability_calibration(reopened, ref, consumer_study_id=consumer) == evidence
    assert BudgetLedger(store).balance(spec["arms"][0]["arm_id"]) == before
    assert len([e for e in store.events() if e["event_kind"] == "WORKER_STARTED"]) == 1


@pytest.mark.parametrize("version", ["unselected-calibration-version", 1, True, ""])
def test_source_owner_does_not_fall_back_from_missing_or_invalid_selected_grant(source, monkeypatch, version):
    store, spec, original = source
    monkeypatch.setattr(store, "read_artifact", lambda *a, **kw: pytest.fail("read before selected grant refusal"))
    with pytest.raises(ResearchError):
        prepare_probability_calibration(store, {**original, "source_authorization_version": version},
            consumer_study_id=spec["study_id"])
