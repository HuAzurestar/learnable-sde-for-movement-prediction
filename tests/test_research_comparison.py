"""Managed statistical jobs share arm budgets but never enlarge the matrix."""

import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import accept_evidence_package, export_evidence
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_research_adjudication_binding import policy
from tests.test_research_store import spec


@pytest.fixture
def source(tmp_path):
    return comparison_source(tmp_path)


def comparison_source(tmp_path, *, horizons=None):
    """Freeze optional strata before registration; never mutate an executed study."""
    store = ResearchStore(tmp_path / "runtime", "comparison", initialize=True)
    value = spec()
    value["arms"].append({**value["arms"][0], "arm_id": "candidate", "model_family_id": "candidate"})
    value["comparison_plan"] = {"reference_arm_id": "affine", "candidate_arm_ids": ["candidate"],
        "failure_policy": "retain-and-exclude-incomplete-blocks", "adjudication_spec": policy(), "adjudication_hash": digest(policy())}
    value["cells"] = [{"arm_id": arm, "seed": seed, "block_id": block, "visibility": "synthetic"}
                      for arm in ("affine", "candidate") for block in ("block-1", "block-2") for seed in (1, 2)]
    if horizons is not None:
        assert horizons and len(set(horizons)) == len(horizons)
        value['cells'] = [{**cell, 'horizon': horizon} for cell in value['cells'] for horizon in horizons]
        frozen = value['comparison_plan']['adjudication_spec']
        frozen['contrasts'][0]['stratum_weights'] = [
            {'comparison_dimensions': {'horizon': horizon}, 'weight': 1 / len(horizons)}
            for horizon in horizons]
        value['comparison_plan']['adjudication_hash'] = digest(frozen)
    store.register(value, digest(value))
    grant = {"authorization_id": "viewer", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
        "evidence_hash": digest("synthetic comparison permission"), "purposes": ["preview", "export"],
        "visibilities": ["synthetic"], "block_ids": ["block-1", "block-2"]}
    store.authorize(grant)
    ledger = BudgetLedger(store)
    for cell in value["cells"]:
        attempt = store.new_attempt(store.register_run("synthetic", cell))
        reserved = ledger.reserve(attempt, BudgetSpec(1))
        store.transition(attempt, "RUNNING")
        result = {"spec_hash": digest(value), "cell_hash": digest(cell), "protocol_hash": value["protocol_hash"],
            "metrics": {"error": 10 if cell["arm_id"] == "affine" else 9}, "metric_units": {"error": "m"},
            "metric_definitions": {"error": "admitted-synthetic-error-v1"}, "qualification": "fixture",
            "state_order": ["x", "y", "vx", "vy"], "units": ["m", "m", "m/s", "m/s"]}
        artifact = store.artifact(encode(result), role="result", visibility="synthetic",
                                  block_ids=[cell["block_id"]], study_id="synthetic")
        ledger.settle(reserved["reservation_id"], 100, outcome="SUCCEEDED")
        store.transition(attempt, "SUCCEEDED", artifact_id=artifact["artifact_id"])
    frozen = export_evidence(store, "synthetic", grant)
    paper = Path(__file__).resolve().parents[2] / "TSDE-SDE"
    assert (paper / "scripts/pirc25/aggregate.py").is_file(), "paired checkout required"
    bundle_path = tmp_path / "source.json"
    bundle_path.write_bytes(encode(frozen))
    subprocess.run([sys.executable, "-B", str(paper / "scripts/pirc25/aggregate.py"), str(bundle_path),
        "--expected-hash", frozen["bundle_hash"], "--output", str(tmp_path / "source-evidence")],
        check=True, capture_output=True, timeout=30)
    aggregate = json.loads((tmp_path / "source-evidence/aggregate.json").read_bytes())
    package = accept_evidence_package(store, tmp_path / "source-evidence", aggregate["aggregate_hash"])
    return store, value, grant, paper, package


def compute(source, **kwargs):
    from application.research_comparison import ComparisonRunner
    store, _, _, paper, package = source
    return ComparisonRunner(store, paper).run("synthetic", package["aggregate_hash"],
        authorization_id=kwargs.pop("authorization_id", "viewer"),
        budget=kwargs.pop("budget", BudgetSpec(10)), **kwargs)


def test_actual_worker_comparison_cost_and_frozen_evidence(source):
    store, value, grant, _, _ = source
    before = BudgetLedger(store).balance("affine")["committed_ms"]
    result = compute(source)
    assert result["state"] == "SUCCEEDED" and result["exit_code"] == 0
    viewed = ResearchQuery(store, "viewer").comparison(result["comparison"]["aggregate_hash"])
    aggregate = viewed["aggregate"]
    assert aggregate["expected_cell_count"] == len(value["cells"]) == 8
    record = aggregate["adjudication"]["records"][0]
    assert record["verdict"] == "GAIN" and record["independent_n"] == 2
    assert aggregate["adjudication"]["qualification"] == "engineering-fixture"
    proof = store.manifest(aggregate["computation_ref"]["manifest_id"])
    assert proof["attempt_id"] == result["attempt_id"]
    assert proof["cost"]["charged_ms"] > 0 and proof["cost"]["unit"] == "slot-ms"
    assert BudgetLedger(store).balance("affine")["committed_ms"] == before + proof["cost"]["charged_ms"]
    assert BudgetLedger(store).balance("candidate")["committed_ms"] == 400
    assert any(e["event_kind"] == "WORKER_STARTED" and e["payload"]["attempt_id"] == result["attempt_id"] for e in store.events())
    csv_content = store.read_artifact(result["comparison"]["table"], purpose="export", authorization=grant)
    rows = list(csv.DictReader(io.StringIO(csv_content.decode())))
    assert all(json.loads(row["adjudication"]) == aggregate["adjudication"] for row in rows)
    assert all(json.loads(row["computation_ref"]) == aggregate["computation_ref"] for row in rows)
    index = json.loads(store.read_artifact(result["comparison"]["evidence-index"], purpose="export", authorization=grant))
    assert index["adjudication"] == aggregate["adjudication"] and index["computation_ref"] == aggregate["computation_ref"]
    assert store.manifest("study-synthetic")["spec"] == value
    assert len(export_evidence(store, "synthetic", grant)["cells"]) == 8


def test_successful_computation_reuse_does_not_reset_or_charge_again(source):
    store = source[0]
    first = compute(source)
    before = BudgetLedger(store).balance("affine")
    count = len(store.attempts())
    second = compute(source)
    assert second["reused"] is True and second["attempt_id"] == first["attempt_id"]
    assert second["comparison"] == first["comparison"]
    assert BudgetLedger(store).balance("affine") == before and len(store.attempts()) == count


def test_computation_cannot_escape_original_cumulative_arm_cap(source):
    store, value, _, _, _ = source
    ledger = BudgetLedger(store)
    # Spend the same arm on another immutable study, not by rewriting success.
    spending = {**value, "study_id": "synthetic-spending", "cells": [value["cells"][0]]}
    store.register(spending, digest(spending))
    attempt = store.new_attempt(store.register_run(spending["study_id"], spending["cells"][0]))
    reservation = ledger.reserve(attempt, BudgetSpec(1))
    used = ledger.balance("affine")["committed_ms"] - reservation["reserved_ms"]
    ledger.settle(reservation["reservation_id"], 86_400_000 - used, outcome="FAILED")
    store.transition(attempt, "FAILED", error_code="SYNTHETIC_SPENT_BUDGET")
    starts = sum(e["event_kind"] == "WORKER_STARTED" for e in store.events())
    with pytest.raises(ResearchError, match="BUDGET_EXHAUSTED"):
        compute(source)
    assert sum(e["event_kind"] == "WORKER_STARTED" for e in store.events()) == starts


def test_preview_permission_does_not_authorize_computation_export(source):
    store, _, grant, _, _ = source
    store.authorize({**grant, "authorization_id": "preview", "purposes": ["preview"]})
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        compute(source, authorization_id="preview")
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())


def test_comparison_preflight_rejects_impossible_operation_plan(source):
    store = source[0]
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        compute(source, max_operations=1)
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())


def test_actual_deadline_and_failed_retry_history_are_not_silently_restarted(source):
    store = source[0]
    timed = compute(source, budget=BudgetSpec(0.001))
    assert timed["state"] == "TIMEOUT" and timed["exit_code"] != 0
    assert BudgetLedger(store).balance("affine")["closed"] is True
    with pytest.raises(ResearchError):
        compute(source)
    assert store.attempts()[timed["attempt_id"]]["state"] == "TIMEOUT"


def test_corrupt_managed_result_cannot_be_reused_or_republished(source):
    store = source[0]
    result = compute(source)
    assert result["state"] == "SUCCEEDED", result
    attempt = store.attempts()[result["attempt_id"]]
    (store.path / "artifacts" / attempt["artifact_id"]).write_bytes(b"damaged disposable synthetic result")
    before = BudgetLedger(store).balance("affine")
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        compute(source)
    assert BudgetLedger(store).balance("affine") == before


def test_adjudication_import_requires_actual_supervised_output(source, tmp_path):
    from application.research_evidence import accept_aggregate, expected_metrics_csv, expected_paper_index
    import hashlib
    store = source[0]
    result = compute(source, output=tmp_path / "managed")
    view = ResearchQuery(store, "viewer").comparison(result["comparison"]["aggregate_hash"])
    original = view["aggregate"]
    altered = json.loads(encode(original))
    altered["adjudication"]["records"][0]["verdict"] = "NO_GAIN"
    decision = altered["adjudication"]
    decision["compare_hash"] = digest({k: v for k, v in decision.items() if k != "compare_hash"})
    altered["aggregate_hash"] = digest({k: v for k, v in altered.items() if k != "aggregate_hash"})
    table = expected_metrics_csv(altered)
    files = {"aggregate.json": encode(altered), "metrics.csv": table,
             "PaperEvidenceIndex.json": encode(expected_paper_index(altered, table)),
             "ComputationReceipt.json": (tmp_path / "managed/ComputationReceipt.json").read_bytes()}
    files["manifest.json"] = encode({"schema_version": "pirc25-evidence-package-v1",
        "aggregate_hash": altered["aggregate_hash"], "files": {n: hashlib.sha256(c).hexdigest() for n, c in files.items()}})
    forged = tmp_path / "forged"
    forged.mkdir()
    for name, content in files.items():
        (forged / name).write_bytes(content)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        accept_evidence_package(store, forged, altered["aggregate_hash"])
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        accept_aggregate(store, altered, "synthetic")
    # Removing the reference must yield a typed rejection, not a raw KeyError.
    altered.pop("computation_ref")
    altered["aggregate_hash"] = digest({k: v for k, v in altered.items() if k != "aggregate_hash"})
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        accept_aggregate(store, altered, "synthetic")
    assert len(store.attempts()) == 9


def test_publication_interruption_recovers_same_success_without_new_charge(source, monkeypatch):
    store = source[0]
    original = store.publish
    def stop_receipt(object_id, value):
        if object_id.startswith("computation-") and not object_id.startswith("computation-request-"):
            raise OSError("synthetic receipt publication interruption")
        return original(object_id, value)
    monkeypatch.setattr(store, "publish", stop_receipt)
    with pytest.raises(OSError, match="publication interruption"):
        compute(source)
    success = [a for a in store.attempts().values() if a["state"] == "SUCCEEDED"][-1]
    before = BudgetLedger(store).balance("affine")
    monkeypatch.setattr(store, "publish", original)
    recovered = compute(source)
    assert recovered["reused"] is True and recovered["attempt_id"] == success["attempt_id"]
    assert BudgetLedger(store).balance("affine") == before and len(store.attempts()) == 9


def test_independent_compare_cli_preserves_nonzero_timeout(source):
    store, _, _, paper, package = source
    command = [sys.executable, "-B", "-m", "experiments.pirc25", "--root", str(store.path.parent),
        "--store-id", store.store_id, "compare", "synthetic", package["aggregate_hash"],
        "--authorization-id", "viewer", "--paper-root", str(paper), "--seconds", "0.001"]
    process = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert process.returncode != 0, process.stdout + process.stderr
    result = json.loads(process.stdout)
    assert result.get("state") == "TIMEOUT" and result.get("exit_code") != 0, result
    assert store.attempts()[result["attempt_id"]]["state"] == "TIMEOUT"
    assert BudgetLedger(store).balance("affine")["closed"] is True


@pytest.mark.parametrize("policy", [[], {"interval": ["bad"]}, {"contrasts": ["bad"]},
    {"contrasts": [{"stratum_weights": "bad"}]}, {"interval": {"replicates": True}}])
def test_malformed_computation_plan_fails_with_typed_diagnostic(policy):
    from application.research_computation import comparison_plan
    bundle = {"cells": [{"block_id": "synthetic", "metrics": None}],
              "comparison_plan": {"adjudication_spec": policy}}
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        comparison_plan(bundle, 20_000_000)


def test_independent_compare_cli_success_is_reconstructible_and_reusable(source, tmp_path):
    store, _, _, paper, package = source
    process = subprocess.run([sys.executable, "-B", "-m", "experiments.pirc25", "--root", str(store.path.parent),
        "--store-id", store.store_id, "compare", "synthetic", package["aggregate_hash"],
        "--authorization-id", "viewer", "--paper-root", str(paper), "--seconds", "10",
        "--output", str(tmp_path / "cli-evidence")], capture_output=True, text=True, timeout=40)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout)
    assert result["state"] == "SUCCEEDED"
    imported = accept_evidence_package(store, tmp_path / "cli-evidence", result["comparison"]["aggregate_hash"])
    assert imported == result["comparison"]
    before = BudgetLedger(store).balance("affine")
    assert compute(source)["attempt_id"] == result["attempt_id"]
    assert BudgetLedger(store).balance("affine") == before


def test_actual_admitted_formal_fixture_uses_managed_comparison(tmp_path, monkeypatch):
    from tests.test_research_adjudication_binding import prepare
    from experiments.pirc25.runner import SharedRunner
    from application.research_comparison import ComparisonRunner
    store, value, registry, grant = prepare(tmp_path, monkeypatch)
    store.register(value, digest(value))
    for cell in value["cells"]:
        assert SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    bundle = export_evidence(store, "synthetic", grant)
    paper = Path(__file__).resolve().parents[2] / "TSDE-SDE"
    source_path = tmp_path / "formal.json"
    source_path.write_bytes(encode(bundle))
    subprocess.run([sys.executable, "-B", str(paper / "scripts/pirc25/aggregate.py"), str(source_path),
        "--formal", "--expected-hash", bundle["bundle_hash"], "--output", str(tmp_path / "formal-evidence")],
        capture_output=True, check=True, timeout=30)
    aggregate = json.loads((tmp_path / "formal-evidence/aggregate.json").read_bytes())
    assert aggregate["qualification"] == "descriptive", "source admission is not a computation proof"
    package = accept_evidence_package(store, tmp_path / "formal-evidence", aggregate["aggregate_hash"])
    result = ComparisonRunner(store, paper).run("synthetic", package["aggregate_hash"],
        authorization_id=grant["authorization_id"], budget=BudgetSpec(10))
    assert result["state"] == "SUCCEEDED", result
    viewed = ResearchQuery(store, grant["authorization_id"]).comparison(result["comparison"]["aggregate_hash"])
    assert viewed["aggregate"]["adjudication"]["qualification"] == "formal"
    assert viewed["aggregate"]["adjudication"]["records"][0]["verdict"] == "INSUFFICIENT_DATA"
    assert viewed["aggregate"]["adjudication"]["records"][0]["independent_n"] == 1
    # This tests admission plumbing with synthetic inputs, not research qualification.
