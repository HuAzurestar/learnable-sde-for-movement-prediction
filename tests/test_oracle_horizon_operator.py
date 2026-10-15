"""Fixed operator controls in disposable stores; never the shared pilot ledger."""
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
operator = importlib.import_module("pirc27_oracle_horizon")
from application.research_budget import BudgetLedger
from application.research_evidence import export_evidence
from infrastructure.research_store import ResearchError, digest
from tests.test_propagation_resource_evidence import paper_check
from tests.test_propagation_shared_adapter import prepare


def original_arm():
    return {"arm_id": "affine-method", "model_family_id": operator.original.FAMILY,
            "method_family_id": "euler", "objective_id": "endpoint-x", "budget_seconds": 86400}


def source_store(root, *, wrong_family=False):
    store, base, _ = prepare(root)
    base["arms"] = [original_arm()]
    store.register(base, digest(base))  # Register only the existing semantic budget.
    source_arm = dict(original_arm())
    if wrong_family:
        source_arm["model_family_id"] = "another-source"
    store.publish(operator.original.ORIGINAL, {"arms": [source_arm]})
    return store


def test_all_four_physical_case_horizon_pairs_are_frozen_before_results():
    cases, cells, plugins = operator.frozen(original_arm())
    assert len(cases) == 4 and len(cells) == 16 and len(plugins) == 1
    assert {(c["comparison_dimensions"]["model_configuration"], c["horizon"]) for c in cells} == {
        (case.case_id, horizon) for case in cases for horizon in (1., 10., 100., 1000.)}
    assert len({digest(c) for c in cells}) == 16
    for c in cells:
        request = c["propagation_request"]
        assert request["horizons"] == [c["horizon"]] and request["samples"] == 512 and request["steps"] == 32
        assert request["arm_id"] == c["arm_id"] == "affine-method" and c["dimensions"] == 4
        assert c["resource_observation_policy"]["required"] is True
        assert c["path_qualification_policy"]["code_hash"] == operator.code_hash()


@pytest.mark.parametrize("costs,pending", [({"charged_ms": True, "reserved_ms": 0}, 16),
    ({"charged_ms": -1, "reserved_ms": 0}, 16), ({"charged_ms": 0, "reserved_ms": 1}, 16),
    ({"charged_ms": 0, "reserved_ms": 0}, True), ({"charged_ms": 0, "reserved_ms": 0}, -1),
    ({"charged_ms": 0, "reserved_ms": 0}, 17), ({"charged_ms": 9600001, "reserved_ms": 0}, 16)])
def test_no_aggregate_budget_reset_negative_boolean_or_pending_reservation(costs, pending):
    with pytest.raises(ResearchError):
        operator.budget_guard(costs, pending)


def test_exact_remaining_budget_boundary_retains_history():
    operator.budget_guard({"charged_ms": 1474985, "reserved_ms": 0}, 16)
    operator.budget_guard({"charged_ms": 9600000, "reserved_ms": 0}, 16)


def test_wrong_original_family_refuses_before_any_registration(tmp_path):
    store = source_store(tmp_path, wrong_family=True)
    before = store.events()
    with pytest.raises(ResearchError, match="original Euler"):
        operator.prepare(store)
    assert store.events() == before


def test_closed_original_arm_is_not_reopened_or_replaced(tmp_path):
    store = source_store(tmp_path)
    store.append("ARM_CLOSED", {"arm_id": "affine-method", "reason": "explicit engineering closure control"})
    before = store.events()
    with pytest.raises(ResearchError, match="original Euler arm closed"):
        operator.prepare(store)
    assert store.events() == before


@pytest.mark.parametrize("fault", ["source", "cells", "preflight"])
def test_changed_frozen_source_or_recipe_and_failed_admission_never_dispatch(tmp_path, monkeypatch, fault):
    from copy import deepcopy
    store = source_store(tmp_path)
    operator.prepare(store)
    reference = "study-"+operator.STUDY
    published = store.manifest(reference)
    if fault == "preflight":
        run = store.register_run(operator.STUDY, published["spec"]["cells"][0])
        attempt = store.new_attempt(run)
        store.transition(attempt, "PREFLIGHT_FAILED", error_code="UNAUTHORIZED_DATA")
    else:
        altered = deepcopy(published)
        if fault == "source":
            altered["spec"]["orchestrator_source_sha256"] = "0"*64
        else:
            altered["spec"]["cells"] = altered["spec"]["cells"][:-1]
        original_manifest = store.manifest
        monkeypatch.setattr(store, "manifest", lambda key: altered if key == reference else original_manifest(key))
    before = store.events()
    with pytest.raises(ResearchError):
        operator.run_next(store)
    assert store.events() == before
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED"} for e in store.events())


def test_real_isolated_registration_worker_resource_and_independent_complete_export(tmp_path):
    store = source_store(tmp_path)
    prepared = operator.prepare(store)
    assert prepared["state"] == "REGISTERED_NOT_STARTED" and prepared["cells"] == 16
    assert prepared["prospective_slot_seconds"] == 4800 and prepared["charged_ms"] == prepared["reserved_ms"] == 0
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())
    assert operator.prepare(store)["spec_hash"] == prepared["spec_hash"]
    result = operator.run_next(store)
    assert result["state"] == "SUCCEEDED" and result["horizon"] == 1., result
    grant = store.authorization(operator.AUTHORIZATION)
    payload = json.loads(store.read_artifact(result["artifact_id"], purpose="evaluate", authorization=grant))
    assert payload["resource_observation"]["execution_state"] == "SUCCEEDED"
    balance = BudgetLedger(store).balance("affine-method")
    assert balance["committed_ms"] == result["elapsed_ms"] > 0
    bundle = export_evidence(store, operator.STUDY, grant)
    assert len(bundle["cells"]) == 16 and sum(c["status"] == "MISSING" for c in bundle["cells"]) == 15
    source = bundle["cells"][0]["resource_usage"]["attempts"][0]
    assert source["status"] == "RECORDED" and source["packet"]["attempt_id"] == result["attempt_id"]
    checked = paper_check(bundle)
    assert checked.returncode == 0, checked.stderr
    assert BudgetLedger(store).balance("affine-method") == balance
