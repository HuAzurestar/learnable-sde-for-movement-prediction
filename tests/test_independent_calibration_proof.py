"""Actual charged disposable proof -> separate stdlib reader, never export."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from infrastructure.research_store import digest, encode
from tests.test_calibrated_method_consumption import consumers


def paper_check(tmp_path, proof, pointer, consumer, *, references=1):
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    document = {"schema_version": "saved-calibration-inspection-v1", "consumer_study_id": consumer,
        "sources": [{"evidence": proof, "pointer": pointer} for _ in range(references)]}
    path = tmp_path/"saved-proof.json"
    path.write_bytes(encode(document))
    return subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/validate_calibration.py"),
        str(path), "--expected-hash", digest(document)], cwd=paper, capture_output=True, text=True, timeout=30)


def test_actual_paid_geometry_is_verified_without_source_reexecution(consumers, tmp_path):
    store, spec, _, _, source_spec, source_balance, proof = consumers
    before = store.events()
    pointer = spec["cells"][0]["calibration_binding"]["source_pointer"]
    checked = paper_check(tmp_path, proof, pointer, spec["study_id"])
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["cost"]["charged_ms"] == proof["source_cost"]["charged_ms"]
    assert result["cost"]["unique_sources"] == 1
    assert result["scientific_qualification"] is result["method_qualification"] is False
    assert "no export authorization" in result["scope"]
    assert store.events() == before
    from application.research_budget import BudgetLedger
    assert BudgetLedger(store).balance(source_spec["arms"][0]["arm_id"]) == source_balance


def test_repeated_references_count_one_actual_original_charge(consumers, tmp_path):
    _, spec, _, _, _, _, proof = consumers
    pointer = spec["cells"][0]["calibration_binding"]["source_pointer"]
    checked = paper_check(tmp_path, proof, pointer, spec["study_id"], references=4)
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["verified_references"] == 4
    assert result["cost"]["unique_sources"] == 1
    assert result["cost"]["charged_ms"] == proof["source_cost"]["charged_ms"]
    assert "independent_n" not in result and "independent_n" not in result["cost"]


def reseal_result(proof, pointer):
    """Keep all transport hashes coherent to test actual saved-value checks."""
    result = proof["source_result"]
    analysis = result["forecast"]["probability_calibration_analysis"]
    for key in ("moment_certificate", "probability_certificate"):
        analysis[key+"_hash"] = digest(analysis[key])
    analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    result["output_hash"] = digest({k: result[k] for k in ("metrics", "forecast", "fit", "source_schema")})
    artifact_id = digest(result)
    proof["source_artifact"].update(artifact_id=artifact_id, sha256=artifact_id, size_bytes=len(encode(result)))
    proof["source_attempt"].update(artifact_id=artifact_id, artifact_manifest_hash=digest(proof["source_artifact"]))
    pointer["source_artifact_id"] = artifact_id
    proof["completion_event"]["payload"] = deepcopy(proof["source_attempt"])
    proof["completion_event"]["hash"] = digest({k: v for k, v in proof["completion_event"].items() if k != "hash"})


@pytest.mark.parametrize("fault", ["flag", "cost", "consumer", "geometry", "threshold", "analysis",
    "interval", "operations", "native-stop", "event-order", "grant-version", "source-artifact",
    "source-receipt", "source-heldout", "orphan-field"])
def test_resealed_saved_proof_cannot_hide_source_or_numeric_changes(consumers, tmp_path, fault):
    _, spec, _, _, _, _, original = consumers
    proof = deepcopy(original)
    pointer = deepcopy(spec["cells"][0]["calibration_binding"]["source_pointer"])
    analysis = proof["source_result"]["forecast"]["probability_calibration_analysis"]
    if fault == "flag":
        proof["scientific_qualification"] = 0
    elif fault == "cost":
        proof["source_cost"]["charged_ms"] = 0
    elif fault == "consumer":
        proof["consumer_study_id"] = "other"
    elif fault == "geometry":
        proof["geometry"]["closed"] = not proof["geometry"]["closed"]
        proof["geometry_hash"] = digest(proof["geometry"])
    elif fault == "threshold":
        analysis["threshold"] += 1.
    elif fault == "analysis":
        analysis["checks"]["relative_probability_error"] = 1
    elif fault == "interval":
        analysis["probability_certificate"]["functional_bounds"][0] = ["0", "1"]
    elif fault == "operations":
        analysis["operation_counts"]["projection_and_quantile"] += 1
    elif fault == "native-stop":
        proof["stop_event"]["payload"]["confirmation"] = "parent-reported"
    elif fault == "event-order":
        proof["settlement_event"]["sequence"] = proof["worker_event"]["sequence"]
    elif fault == "grant-version":
        proof["authorization"]["version"] = "other"
    elif fault == "source-artifact":
        proof["source_artifact"]["size_bytes"] += 1
    elif fault == "source-receipt":
        proof["source_admission"]["mode"] = "formal"
    elif fault == "source-heldout":
        proof["source_admission"]["documents"]["protocol"]["blocks"][0]["split_role"] = "test"
    else:
        proof["opaque_pass"] = True
    if fault in {"threshold", "analysis", "interval", "operations"}:
        reseal_result(proof, pointer)
    if fault in {"native-stop", "event-order"}:
        key = "stop_event" if fault == "native-stop" else "settlement_event"
        proof[key]["hash"] = digest({k: v for k, v in proof[key].items() if k != "hash"})
    proof["evidence_hash"] = digest({k: v for k, v in proof.items() if k != "evidence_hash"})
    checked = paper_check(tmp_path, proof, pointer, spec["study_id"])
    assert checked.returncode != 0 and "calibration" in checked.stderr
