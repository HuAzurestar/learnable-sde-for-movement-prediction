"""Actual charged disposable proof -> separate stdlib reader, never export."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from infrastructure.research_store import digest, encode
from tests.test_calibrated_method_consumption import consumers


def paper_check(tmp_path, proof, pointer, consumer):
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    document = {"schema_version": "saved-calibration-inspection-v1", "consumer_study_id": consumer,
        "sources": [{"evidence": proof, "pointer": pointer}]}
    path = tmp_path/"saved-proof.json"
    path.write_bytes(encode(document))
    return subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/validate_calibration.py"),
        str(path), "--expected-hash", digest(document)], cwd=paper, capture_output=True, text=True)


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
    proof["evidence_hash"] = digest({k: v for k, v in proof.items() if k != "evidence_hash"})
    checked = paper_check(tmp_path, proof, pointer, spec["study_id"])
    assert checked.returncode != 0 and "calibration" in checked.stderr
