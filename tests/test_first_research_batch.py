"""Pure first-batch bounds; never grants, registers or runs research."""
import importlib.util
from pathlib import Path

import pytest

from infrastructure.research_store import ResearchError, digest


@pytest.fixture(scope="module")
def batch():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pirc27_first_batch.py"
    spec = importlib.util.spec_from_file_location("pirc27_first_batch_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arms():
    return {method: {"arm_id": "pirc27-affine-" + method + "-mean-x",
        "model_family_id": "pirc27-affine-oracle-v1", "method_family_id": method,
        "objective_id": "endpoint-x", "budget_seconds": 86400}
        for method in ("exact", "gaussian")}


@pytest.fixture(scope="module")
def cells(batch, arms):
    return batch.frozen_cells(arms)


def test_exact_prospective_work_stays_inside_total_not_per_arm(batch, cells):
    assert len(cells) == 8
    assert len(cells) * batch.JOB_SECONDS == 2400 < batch.TOTAL_APPROVED_SECONDS == 14400
    assert all(c["affine_qualification_policy"]["maximum_job_seconds"] == 300 for c in cells)


def test_original_arms_and_generator_block_never_become_seed_replicates(cells):
    assert {c["arm_id"] for c in cells} == {
        "pirc27-affine-exact-mean-x", "pirc27-affine-gaussian-mean-x"}
    assert {c["block_id"] for c in cells} == {"pirc27-affine-oracle-v1"}
    assert {c["seed"] for c in cells} == {1147}
    assert len({digest(c) for c in cells}) == 8


def test_all_four_cases_and_identical_requests_within_method_pairs(cells):
    by_case = {}
    for c in cells:
        by_case.setdefault(c["comparison_dimensions"]["model_configuration"], []).append(c)
    assert set(by_case) == {"stable-v1", "near-critical-v1", "anisotropic-v1", "uncertain-initial-v1"}
    for pair in by_case.values():
        assert len(pair) == 2
        for key in ("model_package_hash", "initial_mean", "initial_covariance", "origin",
                    "history_cutoff", "horizons", "functional", "coupling_id", "steps", "tolerance"):
            assert pair[0]["propagation_request"][key] == pair[1]["propagation_request"][key]


def test_actual_policies_are_request_source_and_method_bound(cells):
    for c in cells:
        from application.propagation_execution import request_from_manifest
        request = request_from_manifest(c["propagation_request"])
        policy = c["affine_qualification_policy"]
        assert policy["request_hash"] == request.request_hash
        assert policy["model_package_hash"] == request.model_package_hash
        assert policy["method"] == c["execution"]["config"]["method"]
        assert c["execution_role"] == "qualification" and c["dimensions"] == 4


def test_excess_prospective_work_is_refused_without_store(batch, arms, monkeypatch):
    monkeypatch.setattr(batch, "TOTAL_APPROVED_SECONDS", 2399)
    with pytest.raises(ResearchError, match="exceeds 4H"):
        batch.frozen_cells(arms)


def test_source_recipe_is_not_a_fitted_or_scientifically_qualified_model(cells):
    for c in cells:
        model = c["frozen_dynamics"]
        assert model["training_source"]["training_data_hash"] is None
        assert model["qualification"]["scientific_qualification"] is False
        assert model["state_names"] == ["x", "y", "vx", "vy"]


def test_authority_keeps_actual_user_words_and_forbids_heldout(batch):
    assert batch.DECISION["execution_decision"] == "同意开始实验，请执行，GOAL 模式完成。"
    assert batch.DECISION["maximum_new_research_slot_seconds"] == 14400
    assert "not final-eval" in batch.DECISION["scope"]


def test_actual_registration_and_pre_read_admission_in_disposable_store(batch, arms, tmp_path):
    # Test-only store; production connect never initializes a store.
    from application.research_admission import AdmissionGate
    from infrastructure.research_store import ResearchStore
    from experiments.pirc27.qualification_plugin import qualification_plugin
    store = ResearchStore(tmp_path, "first-batch-control", initialize=True)
    store.publish(batch.ORIGINAL, {"arms": list(arms.values())})
    result = batch.prepare(store)
    assert result["state"] == "REGISTERED_NOT_STARTED"
    spec = store.manifest("study-" + batch.STUDY)["spec"]
    assert spec["scientific_independent_n"] == 0
    assert spec["admission"]["mode"] == "pilot"
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(batch.STUDY, cell))
    receipt = AdmissionGate(store).prepare(spec, cell, qualification_plugin(), attempt)
    assert receipt["mode"] == "pilot" and receipt["qualification"] == "fixture"
    assert receipt["documents"]["protocol"]["blocks"][0]["split_role"] == "validation"
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED"} for e in store.events())
