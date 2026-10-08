"""Complete isolated preparation export; no formal research permission."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.research_evidence import export_evidence
from experiments.pirc27.calibration_bindings import StudyCalibration
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_calibrated_method_consumption import consumers
from tests.test_calibrated_study_matrix import declared_design


def test_failed_slot_can_retain_an_original_attempt_without_a_result_artifact():
    original = declared_design().calibrations[0]
    pointer = original.manifest()["source_pointer"]
    pointer["source_artifact_id"] = None
    entry = replace(original, status="FAILED", geometry_document=None,
        source_pointer_document=encode(pointer), source_evidence_hash=None, consumer_study_id=None,
        reason="original native TIMEOUT; no result artifact")
    document = entry.manifest()
    assert StudyCalibration.from_manifest(document).manifest() == document
    assert document["geometry"] is document["source_evidence_hash"] is None
    assert document["source_pointer"]["source_artifact_id"] is None


@pytest.fixture(scope="module")
def complete_export(consumers):
    store, spec, _, grant, *_ = consumers
    return export_evidence(store, spec["study_id"], grant)


def test_actual_complete_calibrated_export_retains_one_source_cost_and_all_cells(consumers, complete_export):
    store, spec, _, grant, _, _, proof = consumers
    bundle = complete_export
    header = bundle["probability_calibration"]
    assert header["table_hash"] == digest(spec["propagation_design"]["axis_manifest"]["calibrations"])
    assert header["cost"]["charged_ms"] == proof["source_cost"]["charged_ms"]
    assert header["cost"]["unique_attempts"] == 1
    assert len(header["slots"]) == 1 and len(header["sources"]) == 1
    assert len(bundle["cells"]) == len(bundle["expected_cells"]) == len(spec["cells"])
    assert all(row["status"] == "MISSING" for row in bundle["cells"])
    assert header["sources"][0]["proof"]["evidence_hash"] == proof["evidence_hash"]


def paper_check(tmp_path, bundle):
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    path = tmp_path/"calibrated-bundle.json"
    path.write_bytes(encode(bundle))
    return subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/validate_calibrated_study.py"),
        str(path), "--expected-hash", bundle["bundle_hash"]], cwd=paper,
        capture_output=True, text=True, timeout=30)


def test_independent_complete_matrix_inspection_does_not_require_a_successful_target(consumers, complete_export, tmp_path):
    store, spec, _, grant, *_ = consumers
    bundle = complete_export
    checked = paper_check(tmp_path, bundle)
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["expected_cells"] == len(spec["cells"])
    assert result["calibration_slots"] == 1 and result["cost"]["unique_attempts"] == 1
    assert result["formal_comparison"] is False


@pytest.mark.parametrize("fault", ["export-purpose", "consumer", "version", "expiry", "visibility"])
def test_source_export_authority_precedes_any_artifact_disclosure(consumers, monkeypatch, fault):
    store, spec, _, grant, *_ = consumers
    original = store.authorization
    def authorization(authorization_id, *, version=None):
        value = original(authorization_id, version=version)
        if version == "consumer-v1":
            value = deepcopy(value)
            field, replacement = {"export-purpose": ("purposes", ["evaluate"]),
                "consumer": ("consumer_study_ids", []), "version": ("version", "other"),
                "expiry": ("expires_at", "2000-01-01T00:00:00+00:00"),
                "visibility": ("visibilities", [])}[fault]
            value[field] = replacement
        return value
    monkeypatch.setattr(store, "authorization", authorization)
    monkeypatch.setattr(store, "read_artifact", lambda *a, **k: pytest.fail("read before source export permission"))
    from infrastructure.research_store import ResearchError
    with pytest.raises(ResearchError):
        export_evidence(store, spec["study_id"], grant)


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    from application.probability_calibration_admission import prepare_probability_calibration
    from application.research_budget import BudgetSpec
    from experiments.pirc25.runner import SharedRunner
    from experiments.pirc27.design import StudyFunctional, StudyMethod, freeze_design
    from experiments.pirc27.plugin import propagation_plugin
    from experiments.pirc27.preparation import _design_from_document
    from tests.test_managed_probability_calibration import prepared
    from tests.test_probability_calibration_owner import pointer
    from tests.test_propagation_study_design import fixture
    from tests.research_admission_fixtures import admit_fixture
    root = tmp_path_factory.mktemp("calibration-outcomes")
    consumer, entries, sources = "calibration-outcomes-unit", [], []
    family = fixture().models[0].family_id
    for name, probability, changes in [("passed", 1e-6, {}),
            ("numerical-failure", 1e-4, {"maximum_relative_probability_error": 1e-30,
                "maximum_relative_probability_width": 1e-50}),
            ("execution-failure", .01, {"maximum_operations": 100})]:
        prefix = name+"-"
        store, source, registry = prepared(root, study_id="source-"+name,
            fixture_prefix=prefix, policy_changes={"target_probability": probability, **changes})
        store.register(source, digest(source))
        runner = SharedRunner(store, registry)
        retry = {}
        if name == "passed":
            with pytest.raises(ResearchError, match="frozen pilot job budget"):
                runner.run_cell(source["study_id"], digest(source["cells"][0]), budget=BudgetSpec(61, category="pilot"))
            parent = next(a for a in store.attempts().values() if a["state"] == "PREFLIGHT_FAILED")
            retry = {"parent_attempt_id": parent["attempt_id"], "reason": "use original frozen pilot budget after explicit preflight refusal"}
        attempt = runner.run_cell(source["study_id"], digest(source["cells"][0]), budget=BudgetSpec(60, category="pilot"), **retry)
        assert attempt["state"] == ("FAILED" if name == "execution-failure" else "SUCCEEDED"), attempt
        grant = store.authorization(prefix+"execution-fixture")
        store.authorize({**grant, "version": "export-v1", "consumer_study_ids": [consumer]})
        ref = pointer(source, attempt)
        ref.update(source_authorization_id=grant["authorization_id"], source_authorization_version="export-v1")
        if name == "passed":
            proof = prepare_probability_calibration(store, ref, consumer_study_id=consumer)
            entries.append(StudyCalibration(name, family, ref["policy"]["model_package_hash"], 1.,
                status="CALIBRATED", reason="actual paid synthetic preparation",
                geometry_document=encode(proof["geometry"]), source_pointer_document=encode(ref),
                source_evidence_hash=proof["evidence_hash"], consumer_study_id=consumer))
        else:
            entries.append(StudyCalibration(name, family, ref["policy"]["model_package_hash"], 1.,
                status="FAILED", reason="original numerical or execution failure", source_pointer_document=encode(ref)))
        sources.append((source, attempt))
    entries.append(StudyCalibration("unavailable", family, entries[0].model_package_hash, 1.,
        status="UNAVAILABLE", reason="no source has been authorized or run"))
    design = replace(fixture(methods=(StudyMethod("exact", samples=8, steps=4), StudyMethod("euler", samples=8, steps=4)),
        functionals=tuple(StudyFunctional(name, "endpoint-halfspace", target_probability=p)
            for name, p in [("passed", 1e-6), ("numerical-failure", 1e-4), ("execution-failure", .01), ("unavailable", .001)]),
        horizons=(1.,)), study_id=consumer, calibrations=tuple(entries))
    frozen = freeze_design(design)
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    grant = admit_fixture(store, spec, propagation_plugin(), root, fixture_prefix="consumer-",
        preserve_execution=True, legacy_upstream=False)
    document = frozen.manifest()
    document.update({k: spec[k] for k in ("protocol_hash", "data_hash")})
    rebound = freeze_design(_design_from_document(document))
    spec["propagation_design_hash"] = rebound.manifest_hash
    spec["runtime_binding"] = {"root": str(root.resolve()), "store_id": store.store_id}
    store.register(spec, digest(spec))
    return store, spec, grant, sources


@pytest.fixture(scope="module")
def outcome_export(outcomes):
    store, spec, grant, _ = outcomes
    return export_evidence(store, spec["study_id"], grant)


def test_complete_failed_unavailable_and_abrupt_sources_retain_original_history_and_cost(outcomes, outcome_export, tmp_path):
    store, spec, grant, sources = outcomes
    bundle = outcome_export
    header = bundle["probability_calibration"]
    assert [slot["status"] for slot in header["slots"]] == ["CALIBRATED", "FAILED", "FAILED", "UNAVAILABLE"]
    assert len(bundle["cells"]) == 16 and all(row["status"] == "MISSING" for row in bundle["cells"])
    assert len(header["sources"]) == 3 and header["cost"]["unique_attempts"] == 4
    assert header["cost"]["charged_ms"] is None  # preflight failure has no reservation, not free success
    assert [len(record["history"]) for record in header["sources"]] == [2, 1, 1]
    assert header["sources"][1]["outcome"] == "recorded-numerical-analysis-failure"
    assert header["sources"][2]["outcome"] == "recorded-execution-failure-FAILED"
    assert header["sources"][2]["source_artifact"] is None
    measured = [s for s in header["cost"]["sources"] if s["status"] == "MEASURED"]
    assert len(measured) == 3 and all(s["charged_ms"] > 0 for s in measured)
    checked = paper_check(tmp_path, bundle)
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["calibration_slots"] == 4 and result["cost"] == header["cost"]


def test_later_failed_source_denial_precedes_even_the_first_successful_source_bytes(outcomes, monkeypatch):
    store, spec, grant, _ = outcomes
    original = store.authorization
    def authorization(identifier, *, version=None):
        value = original(identifier, version=version)
        return {**value, "purposes": ["evaluate"]} if identifier.startswith("numerical-failure-") else value
    monkeypatch.setattr(store, "authorization", authorization)
    monkeypatch.setattr(store, "read_artifact", lambda *a, **k: pytest.fail("partial source disclosure before all-source preflight"))
    with pytest.raises(ResearchError, match="permission"):
        export_evidence(store, spec["study_id"], grant)


@pytest.mark.parametrize("fault", ["slot", "source", "matrix", "history", "native-cost", "visibility",
    "secondary-authority", "qualification", "expiry", "time", "table"])
def test_independent_reader_rejects_resealed_incomplete_or_substituted_evidence(outcome_export, tmp_path, fault):
    from application.probability_calibration_export import calibration_cost
    from application.research_cost import frozen_cost
    bundle = deepcopy(outcome_export)
    header, record = bundle["probability_calibration"], bundle["probability_calibration"]["sources"][1]
    if fault == "slot":
        header["slots"].pop()
    elif fault == "source":
        header["sources"].pop()
    elif fault == "matrix":
        bundle["cells"].pop()
    elif fault == "history":
        record = header["sources"][0]
        omitted = next(a["attempt_id"] for a in record["history"] if a["state"] == "PREFLIGHT_FAILED")
        record["history"] = [a for a in record["history"] if a["attempt_id"] != omitted]
        record["events"] = [e for e in record["events"] if e["payload"]["attempt_id"] != omitted]
        record["cost"] = frozen_cost(record["history"], record["events"])
    elif fault == "native-cost":
        old = record["cost"]["charged_ms"]
        for events in (record["events"], record["cost"]["sources"]):
            for saved in events:
                if saved["event_kind"] == "SETTLE":
                    saved["payload"].update(charged_ms=old-1, monotonic_elapsed_ms=old-1)
                    saved["hash"] = digest({k: v for k, v in saved.items() if k != "hash"})
        record["cost"] = frozen_cost(record["history"], record["events"])
    elif fault == "visibility":
        record["visibility"] = "public"
        record["authorization"]["visibilities"].append("public")
    elif fault == "secondary-authority":
        record["extra_authorizations"].append(deepcopy(record["authorization"]))
    elif fault == "qualification":
        record["method_qualification"] = True
    elif fault == "expiry":
        record["authorization"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    elif fault == "time":
        bundle["recorded_at"] = "2000-01-01T00:00:00+00:00"
    else:
        bundle["registered_spec"]["propagation_design"]["axis_manifest"]["calibrations"][1]["reason"] = "replacement"
        bundle["spec_hash"] = digest(bundle["registered_spec"])
    for source in header["sources"]:
        source["record_hash"] = digest({k: v for k, v in source.items() if k != "record_hash"})
    header["cost"] = calibration_cost(header["sources"])
    header["evidence_hash"] = digest({k: v for k, v in header.items() if k != "evidence_hash"})
    bundle["bundle_hash"] = digest({k: v for k, v in bundle.items() if k != "bundle_hash"})
    checked = paper_check(tmp_path, bundle)
    assert checked.returncode != 0, checked.stdout


def test_repeat_export_retains_identical_frozen_bytes_and_original_cost(consumers, complete_export):
    store, spec, _, grant, *_ = consumers
    assert encode(export_evidence(store, spec["study_id"], grant)) == encode(complete_export)


def test_actual_cli_source_permission_change_after_target_fsync_blocks_rename(consumers, tmp_path, monkeypatch, capsys):
    import infrastructure.research_store as store_module
    from experiments.pirc25.__main__ import main
    store, spec, _, grant, *_ = consumers
    output = tmp_path/"expired-source-export.json"
    original_write, original_fsync = store_module.atomic_write, store_module.os.fsync
    original_authorization, expired = type(store).authorization, [False]
    def authorization(self, identifier, *, version=None):
        value = original_authorization(self, identifier, version=version)
        return {**value, "expires_at": "2000-01-01T00:00:00+00:00"} if expired[0] and version == "consumer-v1" else value
    def write(path, *args, **kwargs):
        if path != output:
            return original_write(path, *args, **kwargs)
        with monkeypatch.context() as context:
            def fsync(fd):
                original_fsync(fd)
                expired[0] = True
            context.setattr(store_module.os, "fsync", fsync)
            return original_write(path, *args, **kwargs)
    monkeypatch.setattr(type(store), "authorization", authorization)
    monkeypatch.setattr(store_module, "atomic_write", write)
    status = main(["--root", str(store.path.parent), "--store-id", store.store_id, "export", spec["study_id"],
        "--authorization-id", grant["authorization_id"], "--output", str(output)])
    response = json.loads(capsys.readouterr().out)
    assert expired[0] and status == 1 and "error" in response
    assert not output.exists() and not list(tmp_path.glob(".expired-source-export.json.*.staging"))
