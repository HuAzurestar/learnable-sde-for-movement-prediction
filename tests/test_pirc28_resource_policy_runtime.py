"""End-to-end PIRC-28 pilot funding through the opt-in resource policy."""

import pytest

from application.research_budget import BudgetLedger, ResourceBudgetSpec
from experiments.pirc25.runner import SharedRunner
from experiments.pirc28.evidence import collect_qualification
from experiments.pirc28.runtime import registries
from infrastructure.research_store import digest
from tests import test_pirc28_study as study_tests
from tests.test_pirc28_approximation import v2_master
from tests.test_pirc28_synthetic import fixture_poll
from tests.test_research_execution_policy import policy


@pytest.fixture(scope="module")
def resource_pilot(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study_tests, "definition_value", v2_master)
        store, master, grants, _, registrations = study_tests.prepared(
            tmp_path_factory.mktemp("resource-policy-pilot") / "store")
    study = store.manifest("study-" + registrations[0]["study_id"])["spec"]
    cell = next(cell for cell in study["cells"] if cell["fit_K"] == 2)
    decision = policy(store.store_id)
    decision["source_code_hash"] = study["code_hash"]
    decision["resources"].update(minimum_system_free_bytes=1,
        minimum_runtime_free_bytes=1, minimum_available_memory_bytes=1,
        maximum_worker_rss_bytes=8 * 2**30)
    reference = {"manifest_id": "policy-" + digest(decision), "sha256": digest(decision)}
    store.publish(reference["manifest_id"], decision)
    registry, recovery = registries()
    result = SharedRunner(store, registry, recovery).run_cell(study["study_id"], digest(cell),
        budget=ResourceBudgetSpec(reference, study["code_hash"]))
    logs = list((store.path / "artifacts").glob(".attempt-*/worker.log"))
    if result["state"] not in {"SUCCEEDED", "FAILED"} or result.get("artifact_id") is None:
        pytest.fail(str(result) + "\n" + "\n".join(path.read_text(errors="replace") for path in logs))
    return store, master, grants, study, cell, result, reference


def test_native_pilot_uses_nullable_deadline_and_measured_cost(resource_pilot):
    store, _, _, study, cell, result, reference = resource_pilot
    events = [event for event in store.events()
              if event["payload"].get("attempt_id") == result["attempt_id"]]
    reservation = next(event["payload"] for event in events if event["event_kind"] == "RESERVE")
    started = next(event["payload"] for event in events if event["event_kind"] == "WORKER_STARTED")
    settlement = next(event["payload"] for event in events if event["event_kind"] == "SETTLE")
    assert reservation["reserved_ms"] is None and reservation["execution_policy"] == reference
    assert started["deadline_monotonic"] is None and started["resource_observation"]["worker_rss_bytes"] >= 0
    assert settlement["charged_ms"] == result["elapsed_ms"] > 0
    balance = BudgetLedger(store).balance(cell["arm_id"], execution_policy=reference,
        source_code_hash=study["code_hash"])
    assert balance["remaining_ms"] is None and balance["committed_ms"] == result["elapsed_ms"]


def test_resource_policy_cost_is_accepted_by_scientific_evidence(resource_pilot):
    store, master, grants, _, _, result, reference = resource_pilot
    from experiments.pirc28.qualification import IdentifiabilitySpec
    definition = IdentifiabilitySpec.from_dict(master.to_dict()["identifiability_spec"])
    report = collect_qualification(store, definition, master.study_ids, grants, check_budget=fixture_poll)
    row = next(item for item in report["lineage"]
               if item["selected_attempt_id"] == result["attempt_id"])
    cost = next(item["cost"] for item in row["attempts"]
                if item["attempt_id"] == result["attempt_id"])
    assert cost["reserved_ms"] is None and cost["charged_ms"] == result["elapsed_ms"]
    assert cost["budget_mode"] == "resource-guarded-no-time-cap"
    assert cost["execution_policy"] == reference
