"""Complete-cell export preserves failures and denies unauthorized disclosure."""

import pytest

from application.research_evidence import (export_evidence, accept_aggregate, accept_evidence_package,
                                           expected_metrics_csv, expected_paper_index)
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_research_store import spec


def setup(tmp_path):
    store = ResearchStore(tmp_path, "evidence", initialize=True)
    value = spec()
    value["cells"] += [{"arm_id": "affine", "seed": 2, "block_id": "fixture-1"}]
    for cell in value["cells"]:
        cell["visibility"] = "synthetic"
    store.register(value, digest(value))
    grant = {"authorization_id": "export", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
             "evidence_hash": digest("fixture permission"), "purposes": ["export"],
             "visibilities": ["synthetic"], "block_ids": ["fixture-1"]}
    store.authorize(grant)
    return store, value, grant


def test_export_keeps_every_registered_cell_and_immutable_source(tmp_path):
    store, value, grant = setup(tmp_path)
    cell = value["cells"][0]
    run = store.register_run("synthetic", cell)
    attempt = store.new_attempt(run)
    store.transition(attempt, "RUNNING")
    result = {"spec_hash": digest(value), "cell_hash": digest(cell), "protocol_hash": value["protocol_hash"],
              "metrics": {"error": 1}, "metric_units": {"error": "m"}, "qualification": "fixture",
              "state_order": ["x", "y", "vx", "vy"], "units": ["m", "m", "m/s", "m/s"]}
    artifact = store.artifact(encode(result), role="result", visibility="synthetic", block_ids=["fixture-1"], study_id="synthetic")
    store.transition(attempt, "SUCCEEDED", artifact_id=artifact["artifact_id"])
    bundle = export_evidence(store, "synthetic", grant)
    assert len(bundle["expected_cells"]) == len(bundle["cells"]) == 2
    assert [c["status"] for c in bundle["cells"]] == ["SUCCEEDED", "MISSING"]
    assert bundle["cells"][0]["artifact_id"] == artifact["artifact_id"]
    assert "forecast" not in bundle["cells"][0]
    assert bundle == export_evidence(store, "synthetic", grant)


def test_export_cannot_use_preview_only_or_partial_block_grants(tmp_path):
    store, value, grant = setup(tmp_path)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        export_evidence(store, "synthetic", {**grant, "purposes": ["preview"]})
    partial = {**grant, "authorization_id": "partial", "block_ids": []}
    store.authorize(partial)
    with pytest.raises(ResearchError, match="complete study matrix"):
        export_evidence(store, "synthetic", partial)


def test_export_rejects_restricted_registered_cells_before_reading_results(tmp_path):
    store = ResearchStore(tmp_path, "visibility", initialize=True)
    value = spec()
    value["cells"][0]["visibility"] = "restricted"
    store.register(value, digest(value))
    grant = {"authorization_id": "synthetic-only", "study_id": "synthetic",
             "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("visibility test"),
             "purposes": ["export"], "visibilities": ["synthetic"], "block_ids": ["fixture-1"]}
    store.authorize(grant)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        export_evidence(store, "synthetic", grant)
    assert not any(event["payload"].get("object_id", "").startswith("bundle-") for event in store.events())
    assert any(event["event_kind"] == "DISCLOSURE_DENIED" for event in store.events())


def test_export_rejects_restricted_admission_attachment(tmp_path):
    store, value, grant = setup(tmp_path)
    cell = value["cells"][0]
    run = store.register_run("synthetic", cell)
    attempt = store.new_attempt(run)
    store.transition(attempt, "RUNNING")
    admission = {"spec_hash": digest(value), "cell_hash": digest(cell), "attempt_id": attempt,
                 "run_id": run, "qualification": "fixture",
                 "documents": {"package": {"visibility": "restricted"}}}
    admission["admission_hash"] = digest(admission)
    store.publish("admission-" + admission["admission_hash"], admission)
    result = {"spec_hash": digest(value), "cell_hash": digest(cell), "protocol_hash": value["protocol_hash"],
              "metrics": {"error": 1}, "metric_units": {"error": "m"}, "qualification": "fixture",
              "state_order": ["x", "y", "vx", "vy"], "units": ["m", "m", "m/s", "m/s"],
              "admission_hash": admission["admission_hash"]}
    artifact = store.artifact(encode(result), role="result", visibility="synthetic",
                              block_ids=[cell["block_id"]], study_id="synthetic")
    store.transition(attempt, "SUCCEEDED", artifact_id=artifact["artifact_id"])
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        export_evidence(store, "synthetic", grant)


def test_export_preserves_registered_comparison_dimensions_for_missing_cells(tmp_path):
    store = ResearchStore(tmp_path, "dimensions", initialize=True)
    value = spec()
    base = value["cells"][0]
    value["cells"] = [{**base, "horizon": horizon, "region": "whole",
                       "scenario": "synthetic", "visibility": "synthetic"}
                      for horizon in (1, 2)]
    store.register(value, digest(value))
    grant = {"authorization_id": "export", "study_id": "synthetic",
             "expires_at": "2099-01-01T00:00:00+00:00",
             "evidence_hash": digest("synthetic permission"), "purposes": ["export"],
             "visibilities": ["synthetic"], "block_ids": ["fixture-1"]}
    store.authorize(grant)
    bundle = export_evidence(store, "synthetic", grant)
    assert len({row["cell_hash"] for row in bundle["cells"]}) == 2
    assert {row["arm_id"] for row in bundle["cells"]} == {"affine"}
    for expected, row, cell in zip(bundle["expected_cells"], bundle["cells"], value["cells"]):
        dimensions = {key: cell[key] for key in ("horizon", "region", "scenario")}
        assert row["comparison_dimensions"] == expected["comparison_dimensions"] == dimensions
        assert row["status"] == "MISSING"


def test_frozen_cost_keeps_failed_retry_and_unknown_charge(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    from application.research_budget import BudgetLedger, BudgetSpec
    store, value, grant = setup(tmp_path)
    run = store.register_run("synthetic", value["cells"][0])
    ledger = BudgetLedger(store)
    first = store.new_attempt(run)
    reservation = ledger.reserve(first, BudgetSpec(1))
    ledger.settle(reservation["reservation_id"], 250, outcome="FAILED")
    store.transition(first, "FAILED", error_code="FIXTURE_FAILURE")
    second = store.new_attempt(run, parent_attempt_id=first, reason="synthetic retry")
    reservation = ledger.reserve(second, BudgetSpec(2))
    ledger.settle(reservation["reservation_id"], None, outcome="INTERRUPTED")
    store.transition(second, "INTERRUPTED", error_code="UNKNOWN_WORKER_COST")
    bundle = export_evidence(store, "synthetic", grant)
    cost = bundle["cells"][0]["cost"]
    assert cost["charged_ms"] == 2250 and cost["measured_ms"] is None
    assert cost["unknown_attempt_ids"] == [second]
    assert len(cost["sources"]) == 2 and cost["unit"] == "slot-ms"
    aggregator = Path(__file__).resolve().parents[2] / "TSDE-SDE/scripts/pirc25/aggregate.py"
    if not aggregator.is_file():
        pytest.skip("cross-repository cost check requires sibling TSDE checkout")
    path = tmp_path / "cost-bundle.json"
    path.write_bytes(encode(bundle))
    result = subprocess.run([sys.executable, "-B", str(aggregator), str(path), "--expected-hash", bundle["bundle_hash"],
                             "--output", str(tmp_path / "cost-package")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    arm = json.loads((tmp_path / "cost-package/aggregate.json").read_bytes())["arms"][0]
    assert arm["cost"]["charged_ms"] == 2250 and arm["cost"]["measured_ms"] is None
    assert arm["metrics"] == {} and arm["expected_cells"] == 2


def test_cost_export_is_immutable_across_later_settlement(tmp_path):
    from application.research_budget import BudgetLedger, BudgetSpec
    store, value, grant = setup(tmp_path)
    run = store.register_run("synthetic", value["cells"][0])
    attempt = store.new_attempt(run)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempt, BudgetSpec(10))
    before = export_evidence(store, "synthetic", grant)
    assert before["cells"][0]["cost"]["reserved_ms"] == 10000
    ledger.settle(reservation["reservation_id"], 123, outcome="FAILED")
    store.transition(attempt, "FAILED", error_code="FIXTURE_FAILURE")
    after = export_evidence(store, "synthetic", grant)
    assert after["cells"][0]["cost"]["charged_ms"] == 123
    assert after["cells"][0]["cost"]["reserved_ms"] == 0
    assert before["bundle_hash"] != after["bundle_hash"]
    assert store.manifest("bundle-" + before["bundle_hash"]) == before


def test_aggregate_import_binds_hash_and_registered_spec(tmp_path):
    store, value, grant = setup(tmp_path)
    aggregate = {"schema_version": "pirc25-aggregate-v1", "study_id": "synthetic", "spec_hash": digest(value),
                 "protocol_hash": value["protocol_hash"], "arms": []}
    with pytest.raises(ResearchError):
        accept_aggregate(store, {**aggregate, "aggregate_hash": "0" * 64}, "synthetic")


def test_frozen_package_import_validates_all_attached_hashes(tmp_path):
    import hashlib

    store, value, grant = setup(tmp_path)
    bundle = export_evidence(store, "synthetic", grant)
    aggregate = {"schema_version": "pirc25-aggregate-v1", "study_id": "synthetic", "spec_hash": digest(value),
                 "protocol_hash": value["protocol_hash"], "source_bundle_hash": bundle["bundle_hash"],
                 "expected_cell_count": 2, "cell_dispositions": bundle["cells"], "arms": [],
                 "code_hash": value["code_hash"], "disclosure_scope": bundle["disclosure_scope"]}
    aggregate["aggregate_hash"] = digest(aggregate)
    table = expected_metrics_csv(aggregate)
    files = {"aggregate.json": encode(aggregate), "metrics.csv": table,
             "PaperEvidenceIndex.json": encode(expected_paper_index(aggregate, table))}
    manifest = {"schema_version": "pirc25-evidence-package-v1", "aggregate_hash": aggregate["aggregate_hash"],
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    folder = tmp_path / "package"
    folder.mkdir()
    for name, content in {**files, "manifest.json": encode(manifest)}.items():
        (folder / name).write_bytes(content)
    imported = accept_evidence_package(store, folder, aggregate["aggregate_hash"])
    assert store.manifest("comparison-" + aggregate["aggregate_hash"]) == imported
    assert (store.path / "artifacts" / imported["table"]).read_bytes() == table
    (folder / "metrics.csv").write_bytes(b"changed")
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        accept_evidence_package(store, folder, aggregate["aggregate_hash"])


@pytest.mark.parametrize("tamper", ["csv", "index"])
def test_rehashed_package_cannot_change_frozen_table_or_claims(tmp_path, tamper):
    import hashlib

    store, value, grant = setup(tmp_path)
    bundle = export_evidence(store, "synthetic", grant)
    aggregate = {"schema_version": "pirc25-aggregate-v1", "study_id": "synthetic", "spec_hash": digest(value),
                 "protocol_hash": value["protocol_hash"], "source_bundle_hash": bundle["bundle_hash"],
                 "expected_cell_count": 2, "cell_dispositions": bundle["cells"], "arms": [],
                 "code_hash": value["code_hash"], "disclosure_scope": bundle["disclosure_scope"]}
    aggregate["aggregate_hash"] = digest(aggregate)
    table = expected_metrics_csv(aggregate)
    index = expected_paper_index(aggregate, table)
    if tamper == "csv":
        table = b"aggregate_hash,metric,value\nforged,accuracy,1\n"
        index["table_sha256"] = hashlib.sha256(table).hexdigest()
    else:
        index["claims"] = [{"claim_id": "forged", "value": 1}]
    files = {"aggregate.json": encode(aggregate), "metrics.csv": table,
             "PaperEvidenceIndex.json": encode(index)}
    manifest = {"schema_version": "pirc25-evidence-package-v1", "aggregate_hash": aggregate["aggregate_hash"],
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    folder = tmp_path / "rehashed-package"
    folder.mkdir()
    for name, content in {**files, "manifest.json": encode(manifest)}.items():
        (folder / name).write_bytes(content)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        accept_evidence_package(store, folder, aggregate["aggregate_hash"])
