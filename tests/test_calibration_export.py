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
from infrastructure.research_store import digest, encode
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


def test_actual_complete_calibrated_export_retains_one_source_cost_and_all_cells(consumers):
    store, spec, _, grant, _, _, proof = consumers
    bundle = export_evidence(store, spec["study_id"], grant)
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


def test_independent_complete_matrix_inspection_does_not_require_a_successful_target(consumers, tmp_path):
    store, spec, _, grant, *_ = consumers
    bundle = export_evidence(store, spec["study_id"], grant)
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
