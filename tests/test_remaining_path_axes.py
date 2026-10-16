"""New-cell compiler and native controls, exclusively in disposable stores."""
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
operator = importlib.import_module("pirc27_remaining_path_axes")
from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger
from application.research_evidence import export_evidence
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_propagation_resource_evidence import paper_check


@pytest.fixture(scope="module")
def compiled():
    arms = {m: {"arm_id": "pirc27-affine-"+m+"-mean-x", "model_family_id": operator.original.FAMILY,
        "method_family_id": m, "objective_id": "endpoint-x", "budget_seconds": 86400} for m in operator.METHODS}
    return arms, operator.frozen(arms)


def source_store(path, arms):
    store = ResearchStore(path, "remaining-path-control", initialize=True)
    store.publish(operator.original.ORIGINAL, {"arms": list(arms.values())})
    return store


def test_exact_new_inventory_preserves_old_failure_and_physical_axes(compiled):
    arms, (cases, cells, plugins) = compiled
    assert len(cases) == 4 and len(cells) == 16 and len(plugins) == 1
    assert len({digest(c) for c in cells}) == 16
    dims = [c["comparison_dimensions"] for c in cells]
    assert {(d["method"], d["model_configuration"], d["horizon"]) for d in dims if d["method"] != "euler"} == {
        (m, case, h) for m in ("heun", "reversible-heun") for case, h in (
            ("stable-v1", 10.), ("stable-v1", 100.), ("stable-v1", 1000.),
            ("near-critical-v1", 1.), ("anisotropic-v1", 1.), ("uncertain-initial-v1", 1.))}
    assert {(d["steps"], d["samples"]) for d in dims if d["method"] == "euler"} == {
        (16, 256), (16, 1024), (64, 256), (64, 1024)}
    for c in cells:
        d, q = c["comparison_dimensions"], c["path_qualification_policy"]
        assert c["arm_id"] == arms[d["method"]]["arm_id"]
        assert c["dimensions"] == 4 and c["frozen_dynamics"]["state_names"] == ["x", "y", "vx", "vy"]
        assert q["code_hash"] == operator.code_hash() and q["maximum_job_seconds"] == 300
        assert q["maximum_total_observed_functional_error"] == .05 and q["maximum_sampling_uncertainty"] == .025
        assert c["resource_observation_policy"]["required"] is True
        assert c["propagation_request"]["seed"] == 1147 and c["propagation_request"]["chunk_size"] == 128
        if d["method"] == "euler":
            assert d["model_configuration"] == "stable-v1" and d["horizon"] == 1. and d["steps"] != 32


def test_all_cell_packages_admit_only_own_source_bound_pilot(tmp_path, compiled):
    arms, (_, _, plugins) = compiled
    store = source_store(tmp_path, arms)
    with store._read_transaction():
        report = operator.prepare(store)
        assert report["cells"] == 16 and report["prospective_slot_seconds"] == 4800
        spec = store.manifest("study-"+operator.STUDY)["spec"]
        assert spec["scientific_independent_n"] == 0 and spec["maximum_new_research_slot_seconds"] == 14400
        for cell in spec["cells"]:
            attempt = store.new_attempt(store.register_run(operator.STUDY, cell))
            receipt = AdmissionGate(store).prepare(spec, cell, plugins[cell["plugin_id"]], attempt)
            assert receipt["mode"] == "pilot" and receipt["qualification"] == "fixture"
        assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED"} for e in store.events())


@pytest.mark.parametrize("fault", ["foreign", "duplicate", "closed", "over-cap", "pending"])
def test_bad_identity_closed_original_or_unfunded_prepare_never_starts(tmp_path, compiled, monkeypatch, fault):
    from copy import deepcopy
    arms = deepcopy(compiled[0])
    if fault == "foreign":
        arms["heun"]["model_family_id"] = "not-original"
    if fault == "duplicate":
        arms["duplicate"] = arms["heun"]
    store = source_store(tmp_path, arms)
    if fault == "closed":
        store.append("ARM_CLOSED", {"arm_id": arms["heun"]["arm_id"], "reason": "control"})
    if fault in {"over-cap", "pending"}:
        monkeypatch.setattr(operator, "accounting", lambda s: {"charged_ms": 9600001 if fault == "over-cap" else 0,
            "reserved_ms": 1 if fault == "pending" else 0})
    before = store.events()
    with pytest.raises(ResearchError):
        operator.prepare(store)
    assert store.events() == before


def test_closed_arm_is_never_reopened_and_does_not_block_other_methods(compiled):
    arms, (_, cells, _) = compiled
    balances = {a["arm_id"]: {"closed": m == "heun"} for m, a in arms.items()}
    first = cells[0]
    attempts = {"a": {"attempt_id": "a", "run_id": "r", "state": "TIMEOUT", "error_code": "TIMEOUT"}}
    runs = {"r": {"study_id": operator.STUDY, "cell_hash": digest(first)}}
    own, eligible, blocked = operator.next_cell({"cells": cells}, attempts, runs, balances)
    assert len(own) == 1 and len(eligible) == 10 and len(blocked) == 5
    assert eligible[0]["comparison_dimensions"]["method"] == "reversible-heun"
    assert digest(first) not in blocked and all(c["arm_id"] != arms["heun"]["arm_id"] for c in eligible)


@pytest.mark.parametrize("state,error", [("RUNNING", None), ("PREFLIGHT_FAILED", "UNAUTHORIZED_DATA"),
    ("FAILED", "INVALID_RESULT"), ("INTERRUPTED", "INTERRUPTED")])
def test_live_admission_control_and_ambiguous_interruption_stop_without_retry(compiled, state, error):
    arms, (_, cells, _) = compiled
    a = {"attempt_id": "a", "run_id": "r", "state": state, "error_code": error}
    with pytest.raises(ResearchError):
        operator.next_cell({"cells": cells}, {"a": a}, {"r": {"study_id": operator.STUDY,
            "cell_hash": digest(cells[0])}}, {v["arm_id"]: {"closed": False} for v in arms.values()})


@pytest.mark.parametrize("fault", ["source", "clipped", "unconfirmed-stop"])
def test_resealed_recipe_or_missing_native_proof_never_dispatch(tmp_path, compiled, monkeypatch, fault):
    from copy import deepcopy
    store = source_store(tmp_path, compiled[0])
    operator.prepare(store)
    key = "study-"+operator.STUDY
    frozen = store.manifest(key)
    if fault == "unconfirmed-stop":
        attempt = store.new_attempt(store.register_run(operator.STUDY, frozen["spec"]["cells"][0]))
        store.transition(attempt, "FAILED", error_code="WORKER_FAILED")
    else:
        changed = deepcopy(frozen)
        if fault == "source":
            changed["spec"]["orchestrator_source_sha256"] = "0"*64
        else:
            changed["spec"]["cells"] = changed["spec"]["cells"][:-1]
        original_manifest = store.manifest
        monkeypatch.setattr(store, "manifest", lambda name: changed if name == key else original_manifest(name))
    before = store.events()
    with pytest.raises(ResearchError):
        operator.run_next(store)
    assert store.events() == before
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED"} for e in before)


def test_actual_native_first_new_cell_resources_and_complete_partial_export(tmp_path, compiled):
    store = source_store(tmp_path, compiled[0])
    operator.prepare(store)
    result = operator.run_next(store)
    assert result["state"] == "SUCCEEDED", result
    assert result["dimensions"]["method"] == "heun" and result["dimensions"]["horizon"] == 10.
    grant = store.authorization(operator.AUTHORIZATION)
    payload = json.loads(store.read_artifact(result["artifact_id"], purpose="evaluate", authorization=grant))
    assert payload["resource_observation"]["execution_state"] == "SUCCEEDED"
    assert payload["forecast"]["path_qualification_analysis"]["scientific_qualification"] is False
    balance = BudgetLedger(store).balance("pirc27-affine-"+result["dimensions"]["method"]+"-mean-x")
    assert balance["committed_ms"] == result["elapsed_ms"] > 0
    bundle = export_evidence(store, operator.STUDY, grant, include_path_pilot_evidence=True)
    assert len(bundle["cells"]) == 16 and sum(c["status"] == "MISSING" for c in bundle["cells"]) == 15
    observed = next(c for c in bundle["cells"] if c["status"] == "SUCCEEDED")
    assert observed["resource_usage"]["attempts"][0]["status"] == "RECORDED" and "path_pilot_execution" in observed
    checked = paper_check(bundle)
    assert checked.returncode == 0, checked.stderr
    assert BudgetLedger(store).balance(balance["arm_id"]) == balance
