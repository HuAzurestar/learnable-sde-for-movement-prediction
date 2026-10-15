"""Paid first analysis from an immutable bundle, never an unfunded bootstrap."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_comparison import ComparisonRunner
from application.research_computation import verified_computation
from application.research_evidence import export_evidence
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_research_store import spec as source_spec

PAPER = Path(__file__).resolve().parents[2]/"TSDE-SDE"


def missing_source(root):
    store = ResearchStore(root, "bundle-comparison", initialize=True)
    value = source_spec()
    value["cells"] = [{**cell, "visibility": "synthetic"} for cell in value["cells"]]
    value["cells"][0]["seed"] = 1
    store.register(value, digest(value))
    grant = {"authorization_id": "bundle-viewer", "study_id": value["study_id"],
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("disposable bundle export"),
        "purposes": ["preview", "export"], "visibilities": ["synthetic"],
        "block_ids": sorted({c["block_id"] for c in value["cells"]})}
    store.authorize(grant)
    bundle = export_evidence(store, value["study_id"], grant)
    return store, value, grant, bundle


@pytest.mark.parametrize("input_kind", [None, True, "unknown", []])
def test_invalid_input_kind_refuses_before_any_owner_io(input_kind):
    with pytest.raises(ResearchError, match="input kind"):
        ComparisonRunner(None, PAPER).run("unused", "0"*64, authorization_id="unused", input_kind=input_kind)


@pytest.mark.parametrize("fault", ["wrong-hash", "clipped", "foreign-study", "source-spec", "boolean-cell"])
def test_frozen_bundle_binding_refuses_before_reservation_or_worker(tmp_path, monkeypatch, fault):
    store, value, grant, bundle = missing_source(tmp_path)
    forged = deepcopy(bundle)
    if fault == "wrong-hash":
        forged["bundle_hash"] = "0"*64
    elif fault == "clipped":
        forged["cells"].pop()
        forged["expected_cells"].pop()
    elif fault == "foreign-study":
        forged["study_id"] = "foreign-study"
    elif fault == "source-spec":
        forged["registered_spec"]["cells"][0]["seed"] += 1
    else:
        forged["cells"][0]["seed"] = True  # Python True==1 must not change the typed frozen identity.
    if fault != "wrong-hash":
        forged["bundle_hash"] = digest({k: v for k, v in forged.items() if k != "bundle_hash"})
    input_hash = bundle["bundle_hash"] if fault == "wrong-hash" else forged["bundle_hash"]
    original = store.manifest
    monkeypatch.setattr(store, "manifest", lambda key: deepcopy(forged) if key == "bundle-"+input_hash else original(key))
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT|UNAUTHORIZED_DATA"):
        ComparisonRunner(store, PAPER).run_bundle(value["study_id"], input_hash,
            authorization_id=grant["authorization_id"], budget=BudgetSpec(20))
    assert not store.attempts()
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED"} for e in store.events())


def test_bundle_comparison_does_not_turn_preview_into_compute_export_authority(tmp_path):
    store, value, grant, bundle = missing_source(tmp_path)
    store.authorize({**grant, "authorization_id": "preview-only", "purposes": ["preview"]})
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ComparisonRunner(store, PAPER).run_bundle(value["study_id"], bundle["bundle_hash"],
            authorization_id="preview-only", budget=BudgetSpec(20))
    assert not store.attempts()


def test_bundle_comparison_cannot_escape_original_arm_cap(tmp_path):
    store, value, grant, bundle = missing_source(tmp_path)
    # Disposable ledger oracle spends the original 24H arm; do not invent a
    # smaller/different arm or rewrite the earlier immutable missing snapshot.
    ledger = BudgetLedger(store)
    attempt = store.new_attempt(store.register_run(value["study_id"], value["cells"][0]))
    reservation = ledger.reserve(attempt, BudgetSpec(1))
    ledger.settle(reservation["reservation_id"], 86_400_000, outcome="FAILED")
    store.transition(attempt, "FAILED", error_code="SYNTHETIC_SPENT_BUDGET")
    with pytest.raises(ResearchError, match="BUDGET_EXHAUSTED"):
        ComparisonRunner(store, PAPER).run_bundle(value["study_id"], bundle["bundle_hash"],
            authorization_id=grant["authorization_id"], budget=BudgetSpec(2))
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())
    assert ledger.balance(value["arms"][0]["arm_id"])["committed_ms"] == 86_400_000


def test_bundle_comparison_operation_cap_refuses_before_attempt_registration(tmp_path):
    store, value, grant, bundle = missing_source(tmp_path)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        ComparisonRunner(store, PAPER).run_bundle(value["study_id"], bundle["bundle_hash"],
            authorization_id=grant["authorization_id"], max_operations=1, budget=BudgetSpec(20))
    assert not store.attempts()


def test_bundle_api_cannot_be_overridden_into_aggregate_mode():
    with pytest.raises(ResearchError, match="input kind is fixed"):
        ComparisonRunner(None, PAPER).run_bundle("unused", "0"*64, input_kind="aggregate")


@pytest.mark.parametrize("fault", ["mixed", "missing-kind", "source-substitution"])
def test_raw_reference_cannot_impersonate_a_legacy_aggregate(fault):
    from application.research_computation import computation_input
    reference = {"input_kind": "bundle", "input_bundle_hash": "1"*64, "source_bundle_hash": "1"*64}
    if fault == "mixed":
        reference["input_aggregate_hash"] = "1"*64
    elif fault == "missing-kind":
        reference.pop("input_kind")
    else:
        reference["input_bundle_hash"] = "0"*64
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        computation_input(reference)


def test_actual_resource_bundle_first_analysis_is_paid_bound_and_exactly_reusable(tmp_path, monkeypatch):
    from tests.test_propagation_resource_evidence import run
    store, value, _, grant, source = run(tmp_path)
    assert source["state"] == "SUCCEEDED", source
    bundle = export_evidence(store, value["study_id"], grant)
    assert not any(e["event_kind"] == "MANIFEST" and e["payload"]["object_id"].startswith("comparison-")
        for e in store.events())
    original_run = subprocess.run

    def no_unfunded_bootstrap(command, *args, **kwargs):
        assert not any(str(a).replace("\\", "/").endswith("/aggregate.py") for a in command)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", no_unfunded_bootstrap)
    owner = ComparisonRunner(store, PAPER)
    arm = value["arms"][0]["arm_id"]
    before = BudgetLedger(store).balance(arm)["committed_ms"]
    options = {"authorization_id": grant["authorization_id"], "budget": BudgetSpec(30), "include_resource_summaries": True}
    result = owner.run_bundle(value["study_id"], bundle["bundle_hash"], **options)
    assert result["state"] == "SUCCEEDED" and result["exit_code"] == 0, result
    view = ResearchQuery(store, grant["authorization_id"]).comparison(result["comparison"]["aggregate_hash"])
    aggregate = view["aggregate"]
    reference = aggregate["computation_ref"]
    assert reference["input_kind"] == "bundle" and reference["input_bundle_hash"] == bundle["bundle_hash"]
    assert "input_aggregate_hash" not in reference and aggregate["cell_dispositions"] == bundle["cells"]
    request = store.manifest(reference["request_manifest_id"])
    origin_id = "study-"+request["computation_study_id"]
    origin = store.manifest(origin_id)["spec"]["computation_origin"]
    assert origin["input_kind"] == request["input_kind"] == "bundle"
    assert origin["input_bundle_hash"] == bundle["bundle_hash"] and "input_aggregate_hash" not in origin
    receipt, _ = verified_computation(store, reference, aggregate=aggregate)
    assert receipt["cost"]["charged_ms"] > 0
    assert BudgetLedger(store).balance(arm)["committed_ms"] == before+receipt["cost"]["charged_ms"]
    assert aggregate["arms"][0]["resource_usage_summary"]["observed_attempt_count"] == 1
    assert aggregate["qualification"] == "engineering-fixture"
    assert store.manifest("study-"+value["study_id"])["spec"] == value
    count, balance = len(store.attempts()), BudgetLedger(store).balance(arm)
    again = owner.run_bundle(value["study_id"], bundle["bundle_hash"], **options)
    assert again["reused"] and again["attempt_id"] == result["attempt_id"]
    assert len(store.attempts()) == count and BudgetLedger(store).balance(arm) == balance
    cli = subprocess.run([sys.executable, "-B", "-m", "experiments.pirc25", "--root", str(store.path.parent),
        "--store-id", store.store_id, "compare-bundle", value["study_id"], bundle["bundle_hash"],
        "--paper-root", str(PAPER), "--authorization-id", grant["authorization_id"], "--seconds", "30",
        "--include-resource-summaries"], capture_output=True, text=True, timeout=120)
    assert cli.returncode == 0, cli.stderr+cli.stdout
    cli_result = json.loads(cli.stdout)
    assert cli_result["reused"] and cli_result["attempt_id"] == result["attempt_id"]
    assert len(store.attempts()) == count and BudgetLedger(store).balance(arm) == balance

    original_manifest = store._manifest
    for location in ("origin", "request", "mixed-reference"):
        copied = {}
        if location == "origin":
            copied[origin_id] = deepcopy(original_manifest(origin_id))
            copied[origin_id]["spec"]["computation_origin"].pop("input_kind")
        else:
            request_id = reference["request_manifest_id"]
            copied[request_id] = deepcopy(original_manifest(request_id))
            if location == "request":
                copied[request_id].pop("input_kind")
            else:
                copied[request_id]["computation_ref"]["input_aggregate_hash"] = bundle["bundle_hash"]
            copied[reference["manifest_id"]] = deepcopy(receipt)
            copied[reference["manifest_id"]]["request_hash"] = digest(copied[request_id])
        with monkeypatch.context() as patch:
            patch.setattr(store, "_manifest", lambda key, docs=copied: docs[key] if key in docs else original_manifest(key))
            with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT|CONTRACT_MISMATCH"):
                verified_computation(store, reference, aggregate=aggregate)
    assert len(store.attempts()) == count and BudgetLedger(store).balance(arm) == balance
