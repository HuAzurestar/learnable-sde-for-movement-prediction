"""New no-time-cap policy boundary, disposable stores only; no execution."""

from copy import deepcopy

import pytest

from application.research_execution_policy import (SCHEMA, validate_policy,
    resolve_policy, check_resource_observation)
from application.research_budget import BudgetLedger, BudgetSpec
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_budget import registered


def policy(store_id="policy-fixture"):
    return {"schema_version": SCHEMA, "store_id": store_id, "feature_id": "PIRC-28",
        "arm_ids": ["pirc28-k1", "pirc28-k2"], "decision": "approved",
        "budget_mode": "resource-guarded-no-time-cap", "job_seconds": None,
        "arm_seconds": None, "max_infrastructure_retries": 2, "source_code_hash": "a"*64,
        "resources": {"cpu_slots": 1, "gpu_slots": 0,
            "minimum_system_free_bytes": 10*2**30, "minimum_runtime_free_bytes": 10*2**30,
            "minimum_available_memory_bytes": 2*2**30, "maximum_worker_rss_bytes": 4*2**30},
        "decision_basis": {"authority": "local-human-decision-record", "task_id": "fixture-task",
            "session_id": "fixture-session", "instructions": ["synthetic fixture decision, not a human signature"]},
        "real_data_authorized": False, "test_authorization": False, "public_export": False}


def validate(value, **changes):
    return validate_policy(value, **{"store_id": "policy-fixture", "arm_id": "pirc28-k1",
                                    "source_code_hash": "a"*64, **changes})


def test_nullable_limits_are_explicit_and_detached_from_the_input():
    value = policy()
    checked = validate(value)
    assert checked["job_seconds"] is None and checked["arm_seconds"] is None
    checked["resources"]["cpu_slots"] = 9
    assert value["resources"]["cpu_slots"] == 1


@pytest.mark.parametrize("field,value", [
    ("schema_version", "pirc25-contract-v1"), ("decision", "proposed"),
    ("feature_id", "PIRC-26"), ("store_id", "another-store"),
    ("arm_ids", ["pirc28-k1"]), ("arm_ids", ["pirc26-m0", "pirc28-k2"]),
    ("budget_mode", "cumulative-only-hard"), ("job_seconds", float("inf")),
    ("job_seconds", 7200), ("arm_seconds", 86400), ("max_infrastructure_retries", 3),
    ("max_infrastructure_retries", True), ("source_code_hash", "b"*64),
    ("real_data_authorized", True), ("test_authorization", True), ("public_export", True),
])
def test_policy_refuses_scope_permission_and_implicit_time_changes(field, value):
    candidate = policy()
    candidate[field] = value
    with pytest.raises(ResearchError):
        validate(candidate)


@pytest.mark.parametrize("arm", ["affine", "pirc26-m0", "pirc27-mc"])
def test_no_sibling_or_default_arm_inherits_new_policy(arm):
    with pytest.raises(ResearchError):
        validate(policy(), arm_id=arm)


@pytest.mark.parametrize("field,value", [
    ("cpu_slots", 2), ("cpu_slots", True), ("gpu_slots", 1),
    ("minimum_system_free_bytes", 0), ("minimum_runtime_free_bytes", -1),
    ("minimum_available_memory_bytes", None), ("maximum_worker_rss_bytes", float("inf")),
])
def test_time_policy_keeps_finite_single_worker_resource_guards(field, value):
    candidate = policy()
    candidate["resources"][field] = value
    with pytest.raises(ResearchError):
        validate(candidate)


@pytest.mark.parametrize("fault", ["missing", "extra", "empty-basis", "fake-signature", "huge-basis"])
def test_complete_bounded_local_decision_provenance_required(fault):
    candidate = policy()
    if fault == "missing":
        del candidate["job_seconds"]
    elif fault == "extra":
        candidate["automatic_reopen"] = True
    elif fault == "empty-basis":
        candidate["decision_basis"]["instructions"] = []
    elif fault == "fake-signature":
        candidate["decision_basis"]["authority"] = "externally-signed"
    else:
        candidate["decision_basis"]["instructions"] = ["x"*4097]
    with pytest.raises(ResearchError):
        validate(candidate)


def test_resolver_is_read_only_and_requires_exact_published_hash(tmp_path):
    store = ResearchStore(tmp_path, "policy-fixture", initialize=True)
    value = policy()
    reference = {"manifest_id": "policy-"+digest(value), "sha256": digest(value)}
    store.publish(reference["manifest_id"], value)
    before = store.events()
    checked = resolve_policy(store, reference, arm_id="pirc28-k2", source_code_hash="a"*64)
    assert checked == value and store.events() == before
    changed = {**reference, "sha256": "f"*64}
    with pytest.raises(ResearchError):
        resolve_policy(store, changed, arm_id="pirc28-k2", source_code_hash="a"*64)
    assert store.events() == before


def test_file_without_manifest_event_is_not_a_published_decision(tmp_path):
    store = ResearchStore(tmp_path, "policy-fixture", initialize=True)
    value = policy()
    store._publish("unregistered-policy", value)
    reference = {"manifest_id": "unregistered-policy", "sha256": digest(value)}
    with pytest.raises(ResearchError) as caught:
        resolve_policy(store, reference, arm_id="pirc28-k1", source_code_hash="a"*64)
    assert caught.value.code == "MISSING_INPUT"


def test_final_physical_completion_rechecks_recovery_hold(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path, "policy-fixture", initialize=True)
    value = policy()
    reference = {"manifest_id": "policy-"+digest(value), "sha256": digest(value)}
    store.publish(reference["manifest_id"], value)
    original = store._read_completion
    def inject(callback, finish):
        (store.path / "recovery-hold.json").write_text("{}", encoding="utf-8")
        return original(callback, finish)
    monkeypatch.setattr(store, "_read_completion", inject)
    with pytest.raises(ResearchError, match="recovery hold"):
        resolve_policy(store, reference, arm_id="pirc28-k1", source_code_hash="a"*64)


def sample():
    return {"system_free_bytes": 10*2**30, "runtime_free_bytes": 10*2**30,
            "available_memory_bytes": 2*2**30, "worker_rss_bytes": 4*2**30}


def test_resource_sample_at_exact_boundaries_never_authorizes_execution():
    assert check_resource_observation(validate(policy()), sample()) == {
        "state": "RESOURCE_SAMPLE_WITHIN_LIMITS", "execution_permitted": False}


@pytest.mark.parametrize("field", ["system_free_bytes", "runtime_free_bytes", "available_memory_bytes", "worker_rss_bytes"])
def test_resource_boundaries_refuse_one_byte_beyond_guard(field):
    observation = sample()
    observation[field] += 1 if field == "worker_rss_bytes" else -1
    with pytest.raises(ResearchError, match="RESOURCE_LIMIT"):
        check_resource_observation(validate(policy()), observation)


@pytest.mark.parametrize("value", [None, -1, True, float("inf")])
def test_unknown_resource_measurements_fail_closed(value):
    observation = sample()
    observation["available_memory_bytes"] = value
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        check_resource_observation(validate(policy()), observation)


def test_policy_validation_does_not_change_legacy_funding_or_caps(tmp_path):
    store, attempts = registered(tmp_path)
    before = store.events()
    validate(policy())
    with pytest.raises(ResearchError):
        BudgetSpec(7201).validate()
    with pytest.raises(ResearchError):
        BudgetSpec(1, arm_seconds=86401).validate()
    assert store.events() == before
    assert BudgetLedger(store).reserve(attempts[0], BudgetSpec(1))["reserved_ms"] == 1000
