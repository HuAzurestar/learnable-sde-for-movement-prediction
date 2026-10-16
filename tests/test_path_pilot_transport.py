"""Actual disposable pilot results; no canonical-store reads or scientific retry."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, digest, encode
from tests.path_paper_helpers import _load_paper_module, independent_reader
from tests.test_path_qualification import prepared


@pytest.fixture(scope="module", params=[("euler", None, None, "PASSED"),
    ("heun", None, None, "PASSED"), ("reversible-heun", None, None, "PASSED"),
    ("importance", None, None, "PASSED"),
    ("euler", {"maximum_sampling_uncertainty": 1e-10}, None, "FAILED"),
    ("importance", None, {"threshold": 100.}, "FAILED")])
def pilot(request, tmp_path_factory):
    method, changes, request_changes, expected = request.param
    store, spec, registry = prepared(tmp_path_factory.mktemp("path-pilot-proof"),
        method=method, changes=changes, request_changes=request_changes, resource_observation=True)
    store.register(spec, digest(spec))
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert outcome["state"] == "SUCCEEDED", outcome
    grant = store.authorization(spec["admission"]["authorization_id"])
    before = len(store.attempts())
    cost = BudgetLedger(store).balance(spec["cells"][0]["arm_id"])
    legacy = export_evidence(store, spec["study_id"], grant)
    bundle = export_evidence(store, spec["study_id"], grant, include_path_pilot_evidence=True)
    assert export_evidence(store, spec["study_id"], grant) == legacy
    assert len(store.attempts()) == before and BudgetLedger(store).balance(spec["cells"][0]["arm_id"]) == cost
    return store, spec, grant, outcome, expected, legacy, bundle


def reseal(bundle):
    row = bundle["cells"][0]
    receipt = row["admission"]
    receipt["admission_hash"] = digest({k: v for k, v in receipt.items() if k != "admission_hash"})
    row["admission_hash"] = receipt["admission_hash"]
    row["result"]["admission_hash"] = receipt["admission_hash"]
    analysis = row["result"]["forecast"]["path_qualification_analysis"]
    analysis["completed_statistics_hash"] = digest(analysis["completed_statistics"])
    for prefix in ("continuous", "target"):
        analysis[prefix+"_certificate_hash"] = digest(analysis[prefix+"_certificate"])
    analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    result = row["result"]
    result["output_hash"] = digest({k: result[k] for k in ("metrics", "forecast", "fit", "source_schema")})
    row["artifact_id"] = digest(result)
    row["result_artifact"].update(artifact_id=digest(result), sha256=digest(result), size_bytes=len(encode(result)))
    proof = row["path_pilot_execution"]
    proof["admission_event"]["payload"]["admission_hash"] = receipt["admission_hash"]
    proof["attempt"].update(artifact_id=digest(result), artifact_manifest_hash=digest(row["result_artifact"]))
    proof["completion_event"]["payload"] = deepcopy(proof["attempt"])
    for key in ("reservation_event", "worker_event", "stop_event", "settlement_event", "admission_event", "completion_event"):
        proof[key]["hash"] = digest({k: v for k, v in proof[key].items() if k != "hash"})
    proof["proof_hash"] = digest({k: v for k, v in proof.items() if k != "proof_hash"})
    for source in row.get("resource_usage", {}).get("attempts", []):
        publication = source["publication_event"]
        if source["attempt_id"] == row["attempt_id"] and publication is not None:
            publication["payload"]["result_artifact_id"] = row["artifact_id"]
            publication["hash"] = digest({k: v for k, v in publication.items() if k != "hash"})
    bundle["bundle_hash"] = digest({k: v for k, v in bundle.items() if k != "bundle_hash"})


def test_owned_saved_bytes_and_independent_pass_fail_are_not_production_admission(pilot):
    store, spec, grant, outcome, expected, legacy, bundle = pilot
    row = bundle["cells"][0]
    assert "result" not in legacy["cells"][0] and "path_pilot_evidence" not in legacy
    assert bundle["bundle_hash"] != legacy["bundle_hash"]
    actual = store.read_artifact(outcome["artifact_id"], purpose="export", authorization=grant)
    assert encode(row["result"]) == actual and row["artifact_id"] == outcome["artifact_id"]
    assert row["result_artifact"] == store.manifest("artifact-"+outcome["artifact_id"])
    assert _load_paper_module("path_pilots").validate_path_pilot(row["admission"], row) == expected
    report = _load_paper_module("validate_path_pilots").validate(bundle, bundle["bundle_hash"])
    assert report["verified_pilot_cells"] == 1 and report["noncompleted_rows"] == 0
    assert report["passed_cells"] == (expected == "PASSED") and report["failed_cells"] == (expected == "FAILED")
    assert row["qualification"] == "fixture" and report["scientific_qualification"] is False
    if expected == "FAILED":
        analysis = row["result"]["forecast"]["path_qualification_analysis"]
        with pytest.raises(ValueError, match="source must actually pass"):
            independent_reader().analysis_values(analysis, row["registered_cell"]["path_qualification_policy"],
                row["registered_cell"]["propagation_request"], row["result"]["forecast"]["functional"],
                analysis["continuous_certificate"], analysis["target_certificate"])


@pytest.mark.parametrize("fault", ["classification", "sampling", "statistics", "grid", "unknown", "scientific",
    "native-stop", "cost", "order", "policy", "role", "missing-result", "missing-proof"])
def test_resealed_pilot_meaning_or_native_source_substitution_is_refused(pilot, fault):
    bundle = deepcopy(pilot[-1])
    row = bundle["cells"][0]
    analysis = row["result"]["forecast"]["path_qualification_analysis"]
    proof = row["path_pilot_execution"]
    if fault == "classification":
        analysis["status"] = "FAILED" if analysis["status"] == "PASSED" else "PASSED"
    elif fault == "sampling":
        analysis["sampling_error"]["uncertainty_radius"] = 0.
    elif fault == "statistics":
        analysis["completed_statistics"]["method_state"]["statistics"][0]["n"] -= 1
    elif fault == "grid":
        analysis["target_certificate"]["grid"]["steps"] *= 2
    elif fault == "unknown":
        analysis["model_error"] = {"value": 0., "status": "BOUNDED"}
    elif fault == "scientific":
        analysis["scientific_qualification"] = True
    elif fault == "native-stop":
        proof["stop_event"]["payload"]["confirmation"] = "not-stopped"
    elif fault == "cost":
        proof["settlement_event"]["payload"]["charged_ms"] = 0
    elif fault == "order":
        proof["completion_event"]["sequence"] = proof["worker_event"]["sequence"]
    elif fault == "policy":
        row["admission"]["cell"]["path_qualification_policy"]["maximum_job_seconds"] = 7200.
    elif fault == "role":
        row["admission"]["mode"] = "formal"
    if fault not in {"missing-result", "missing-proof"}:
        reseal(bundle)
    else:
        row.pop("result" if fault == "missing-result" else "path_pilot_execution")
    meaning = {"classification": "all saved numeric", "sampling": "all saved numeric",
        "statistics": "actual sample/event counts", "grid": "declared grid",
        "unknown": "all saved numeric", "scientific": "all saved numeric"}
    with pytest.raises(ValueError, match=meaning.get(fault)):
        _load_paper_module("path_pilots").validate_path_pilot(row["admission"], row)


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_export_opt_in_is_strict_boolean_before_any_owner_io(flag):
    with pytest.raises(ResearchError, match="boolean"):
        export_evidence(None, "unused", {}, include_path_pilot_evidence=flag)


def test_actual_independent_cli_uses_trusted_hash_and_does_not_promote_pilots(pilot, tmp_path):
    bundle = pilot[-1]
    path = tmp_path/"saved-pilot.json"
    path.write_bytes(encode(bundle))
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    command = [sys.executable, "-B", str(paper/"scripts/pirc25/validate_path_pilots.py"), str(path),
        "--expected-hash", bundle["bundle_hash"]]
    result = subprocess.run(command, cwd=paper, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == _load_paper_module("validate_path_pilots").validate(bundle, bundle["bundle_hash"])
    wrong = subprocess.run([*command[:-1], "0"*64], cwd=paper, capture_output=True, text=True, timeout=30)
    assert wrong.returncode != 0 and "trusted expected bundle hash differs" in wrong.stderr
    with pytest.raises(ValueError):
        _load_paper_module("validate_paths").validate(bundle, bundle["bundle_hash"])


def test_opted_export_still_requires_current_original_export_authority(pilot):
    store, spec, grant, *_ = pilot
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        export_evidence(store, spec["study_id"], {**grant, "purposes": ["preview"]}, include_path_pilot_evidence=True)


def test_absent_native_stop_cannot_be_packaged_as_a_source_proof(pilot):
    from application.path_pilot_export import PilotEventIndex
    store, _, _, _, _, _, bundle = pilot
    proof = bundle["cells"][0]["path_pilot_execution"]
    index = PilotEventIndex([e for e in store.events() if e["event_kind"] != "WORKER_TREE_STOPPED"])
    with pytest.raises(ResearchError, match="unique native execution"):
        index.proof(proof["attempt"], proof["run"])
