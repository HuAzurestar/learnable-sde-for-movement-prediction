"""Cross-source restart adjudication controls in disposable stores only."""

from copy import deepcopy

import pytest

from application.research_budget import BudgetLedger, BudgetSpec, ResourceBudgetSpec
from application.research_restart_adjudication import (adjudicate_resource_retry,
    adjudicate_restart, promote_resource_arm, resolve_restart_adjudication,
    resolve_retry_adjudication, resource_arm_continuation, scientific_cell_identity)
from experiments.pirc28.release import execute_first_restart, execute_next_source_cell
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc28.runtime import registries
from experiments.pirc28.study import SyntheticStudyDefinition
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc28_study import prepared
from tests.test_research_execution_policy import policy


def epoch_fixture(tmp_path):
    store, current, grants, _, _ = prepared(tmp_path / "epoch")
    current_value = current.to_dict()
    historical_value = deepcopy(current_value)
    historical_value["runtime_code_hash"] = "b" * 64
    historical = SyntheticStudyDefinition.from_dict(historical_value)
    store.publish("pirc28-study-definition-" + historical.definition_hash, historical_value)
    binding = {"root": str(store.path.parent), "store_id": store.store_id}
    old_specs = {}
    for proposal, current_id in zip(historical.proposals(binding), current.study_ids):
        current_spec = store.manifest("study-" + current_id)["spec"]
        old_id = proposal["study_id"]
        old_grant = {**deepcopy(grants[current_id]), "authorization_id": "execute-" + old_id,
                     "study_id": old_id, "protocol_hash": digest(["historical", old_id])}
        store.authorize(old_grant)
        proposal.update(schema_version=store.SCHEMA, protocol_hash=old_grant["protocol_hash"],
            data_hash=current_spec["data_hash"], admission={"mode": "pilot", "purpose": "fit",
                "protocol_id": "input-" + old_id, "authorization_id": old_grant["authorization_id"],
                "package_hash": digest(["historical-package", old_id])},
            comparison_plan=deepcopy(current_spec["comparison_plan"]))
        store.register(proposal, digest(proposal))
        old_specs[old_id] = proposal

    decision = policy(store.store_id)
    decision["source_code_hash"] = code_hash()
    decision["resources"].update(minimum_system_free_bytes=1,
        minimum_runtime_free_bytes=1, minimum_available_memory_bytes=1,
        maximum_worker_rss_bytes=2**40)
    policy_ref = {"manifest_id": "policy-" + digest(decision), "sha256": digest(decision)}
    store.publish(policy_ref["manifest_id"], decision)

    pairs = []
    for old_id, current_id in zip(historical.study_ids, current.study_ids):
        old_spec = old_specs[old_id]
        new_spec = store.manifest("study-" + current_id)["spec"]
        current_by_identity = {digest(scientific_cell_identity(cell)): cell for cell in new_spec["cells"]}
        cells = []
        for old_cell in old_spec["cells"]:
            identity_hash = digest(scientific_cell_identity(old_cell))
            new_cell = current_by_identity[identity_hash]
            cells.append({"scientific_cell_hash": identity_hash,
                "historical_cell_hash": digest(old_cell), "current_cell_hash": digest(new_cell)})
        pairs.append({"historical_study_id": old_id, "current_study_id": current_id,
            "historical_spec_hash": digest(old_spec), "current_spec_hash": digest(new_spec),
            "historical_grant_hash": digest(store.authorization("execute-" + old_id)),
            "current_grant_hash": digest(grants[current_id]), "cell_pairs": cells})
    bridge = {"schema_version": "pirc28-source-epoch-bridge-v1",
        "execution_policy": policy_ref, "historical_definition_hash": historical.definition_hash,
        "current_definition_hash": current.definition_hash,
        "historical_code_hash": historical_value["runtime_code_hash"],
        "current_code_hash": current_value["runtime_code_hash"],
        "changed_fields": ["runtime_code_hash"], "study_pairs": pairs}
    bridge_ref = {"manifest_id": "bridge-" + digest(bridge), "sha256": digest(bridge)}
    store.publish(bridge_ref["manifest_id"], bridge)
    return store, historical, current, policy_ref, bridge_ref


def interrupted_parent(store, historical):
    study_id = historical.study_ids[0]
    study = store.manifest("study-" + study_id)["spec"]
    cell = next(cell for cell in study["cells"] if cell["fit_K"] == 1)
    run_id = store.register_run(study_id, cell)
    attempt = store.new_attempt(run_id)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempt, BudgetSpec(900, category="smoke"))
    store.transition(attempt, "RUNNING")
    store.append("WORKER_STARTED", {"attempt_id": attempt,
        "reservation_id": reservation["reservation_id"], "pid": 123})
    store.append("WORKER_TREE_STOPPED", {"attempt_id": attempt,
        "reservation_id": reservation["reservation_id"], "observed_elapsed_ms": 47892,
        "confirmation": "native-job-or-process-group-no-running-descendants"})
    ledger.settle(reservation["reservation_id"], 47892, outcome="INTERRUPTED")
    store.transition(attempt, "INTERRUPTED", error_code="SUPERVISOR_ERROR")
    return attempt, cell


def current_resource_interruption(store, current, policy_ref, *, attempt_id=None, cost=37):
    if attempt_id is None:
        study = store.manifest("study-" + current.study_ids[0])["spec"]
        cell = next(cell for cell in study["cells"] if cell["fit_K"] == 2)
        run_id = store.register_run(study["study_id"], cell)
        attempt = store.new_attempt(run_id)
    else:
        attempt = attempt_id
        run = store.manifest("run-" + store.attempts()[attempt]["run_id"])
        run_id, cell = run["run_id"], run["cell"]
        study = store.manifest("study-" + run["study_id"])["spec"]
    reservation = BudgetLedger(store).reserve(attempt, ResourceBudgetSpec(policy_ref, code_hash()))
    store.transition(attempt, "RUNNING")
    store.append("WORKER_STARTED", {"attempt_id": attempt, "run_id": run_id,
        "reservation_id": reservation["reservation_id"], "pid": 123})
    store.append("WORKER_TREE_STOPPED", {"attempt_id": attempt,
        "reservation_id": reservation["reservation_id"], "observed_elapsed_ms": cost,
        "confirmation": "native-job-or-process-group-no-running-descendants"})
    BudgetLedger(store).settle(reservation["reservation_id"], cost, outcome="INTERRUPTED")
    store.transition(attempt, "INTERRUPTED", error_code="RESOURCE_LIMIT")
    return attempt, study, cell


def test_exact_adjudication_preserves_closure_and_funds_only_registered_child(tmp_path, monkeypatch):
    store, historical, current, policy_ref, bridge_ref = epoch_fixture(tmp_path)
    parent, old_cell = interrupted_parent(store, historical)
    before_cost = BudgetLedger(store).balance("pirc28-k1")
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: False)
    reference = adjudicate_restart(store, parent, source_bridge=bridge_ref,
        execution_policy=policy_ref, source_code_hash=code_hash())
    receipt = resolve_restart_adjudication(store, reference,
        execution_policy=policy_ref, source_code_hash=code_hash())
    assert receipt["retained_cost_ms"] == 47892 and receipt["retry_ordinal"] == 1
    assert receipt["native_identity"]["status"] == "recorded-worker-absent"
    assert BudgetLedger(store).balance("pirc28-k1") == before_cost
    assert scientific_cell_identity(old_cell) == scientific_cell_identity(
        store.manifest("run-" + receipt["current_run_id"])["cell"])

    budget = ResourceBudgetSpec(policy_ref, code_hash())
    reservation = BudgetLedger(store).reserve(receipt["retry_attempt_id"], budget)
    assert reservation["restart_adjudication"] == reference
    other_study = store.manifest("study-" + current.study_ids[0])["spec"]
    other_cell = next(cell for cell in other_study["cells"]
                      if cell["fit_K"] == 1 and digest(cell) != receipt["current_cell_hash"])
    other = store.new_attempt(store.register_run(other_study["study_id"], other_cell))
    with pytest.raises(ResearchError) as error:
        BudgetLedger(store).reserve(other, budget)
    assert error.value.code == "BUDGET_EXHAUSTED"
    store.transition(receipt["retry_attempt_id"], "RUNNING")
    BudgetLedger(store).settle(reservation["reservation_id"], 7, outcome="INTERRUPTED")
    store.transition(receipt["retry_attempt_id"], "INTERRUPTED", error_code="RESOURCE_LIMIT")
    assert resolve_restart_adjudication(store, reference,
        execution_policy=policy_ref, source_code_hash=code_hash())["retry_attempt_id"] == receipt["retry_attempt_id"]


def test_adjudication_is_idempotent_and_refuses_live_or_changed_lineage(tmp_path, monkeypatch):
    store, historical, _, policy_ref, bridge_ref = epoch_fixture(tmp_path)
    parent, _ = interrupted_parent(store, historical)
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: False)
    first = adjudicate_restart(store, parent, source_bridge=bridge_ref,
        execution_policy=policy_ref, source_code_hash=code_hash())
    before = store.events()
    assert adjudicate_restart(store, parent, source_bridge=bridge_ref,
        execution_policy=policy_ref, source_code_hash=code_hash()) == first
    assert store.events() == before
    with pytest.raises(ResearchError):
        resolve_restart_adjudication(store, first, execution_policy=policy_ref,
                                     source_code_hash="f" * 64)

    second_root = tmp_path / "live"
    second_root.mkdir()
    other, old, _, other_policy, other_bridge = epoch_fixture(second_root)
    live_parent, _ = interrupted_parent(other, old)
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: True)
    monkeypatch.setattr("application.research_restart_adjudication._pid_reuse_observation",
                        lambda *args: None)
    with pytest.raises(ResearchError) as error:
        adjudicate_restart(other, live_parent, source_bridge=other_bridge,
            execution_policy=other_policy, source_code_hash=code_hash())
    assert error.value.code == "WORKER_ACTIVE"
    assert not any(event["event_kind"] == "RESTART_ADJUDICATED" for event in other.events())


def test_verified_restart_promotes_policy_arm_without_erasing_old_closure(tmp_path, monkeypatch):
    store, historical, current, policy_ref, bridge_ref = epoch_fixture(tmp_path)
    parent, _ = interrupted_parent(store, historical)
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: False)
    reference = adjudicate_restart(store, parent, source_bridge=bridge_ref,
        execution_policy=policy_ref, source_code_hash=code_hash())
    receipt = resolve_restart_adjudication(store, reference,
        execution_policy=policy_ref, source_code_hash=code_hash())
    run = store.manifest("run-" + receipt["current_run_id"])
    budget = ResourceBudgetSpec(policy_ref, code_hash())
    release = execute_first_restart(store, restart_adjudication=reference)
    assert release["result"]["state"] == "SUCCEEDED"
    assert release["verified"]["reused"] is True
    assert release["resource_arm_continuation"] == execute_first_restart(
        store, restart_adjudication=reference)["resource_arm_continuation"]
    ledger = BudgetLedger(store)
    assert ledger.balance("pirc28-k1")["closed"] is True
    assert ledger.balance("pirc28-k1", execution_policy=policy_ref,
                          source_code_hash=code_hash())["closed"] is False
    study = store.manifest("study-" + current.study_ids[0])["spec"]
    other_cell = next(cell for cell in study["cells"]
                      if cell["fit_K"] == 1 and digest(cell) != run["cell_hash"])
    other = store.new_attempt(store.register_run(study["study_id"], other_cell))
    assert ledger.reserve(other, budget)["reserved_ms"] is None
    ledger.settle(digest([store.store_id, other]), 0, outcome="PREFLIGHT_FAILED")
    store.transition(other, "PREFLIGHT_FAILED", error_code="fixture-release")
    current_parent, _, _ = current_resource_interruption(store, current, policy_ref)
    def fake_run(self, study_id, cell_hash, **kwargs):
        restart = kwargs.get("restart_adjudication")
        assert restart is not None
        receipt = resolve_retry_adjudication(store, restart, execution_policy=policy_ref,
                                             source_code_hash=code_hash())
        return {"state": "INTERRUPTED", "attempt_id": receipt["retry_attempt_id"]}
    monkeypatch.setattr("experiments.pirc25.runner.SharedRunner.run_cell", fake_run)
    step = execute_next_source_cell(store, study_definition_hash=current.definition_hash,
                                    execution_policy=policy_ref)
    assert step["state"] == "CELL_DISPATCHED" and step["total_cells"] == 8
    assert step["result"]["attempt_id"] != current_parent
    assert step["retry_adjudication"] is not None
    assert step["resource_arm_continuation"] is None


def test_current_resource_retries_are_chained_bounded_and_old_receipts_stay_valid(
        tmp_path, monkeypatch):
    store, _, current, policy_ref, _ = epoch_fixture(tmp_path)
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: False)
    parent, _, cell = current_resource_interruption(store, current, policy_ref)
    first = adjudicate_resource_retry(store, parent, execution_policy=policy_ref,
                                      source_code_hash=code_hash())
    first_receipt = resolve_retry_adjudication(store, first, execution_policy=policy_ref,
                                               source_code_hash=code_hash())
    assert first_receipt["retry_ordinal"] == 1
    assert first_receipt["parent_cost_ms"] == 37
    assert BudgetLedger(store).balance(cell["arm_id"], execution_policy=policy_ref,
                                      source_code_hash=code_hash())["closed"]

    first_child = first_receipt["retry_attempt_id"]
    current_resource_interruption(store, current, policy_ref, attempt_id=first_child, cost=41)
    second = adjudicate_resource_retry(store, first_child, execution_policy=policy_ref,
                                       source_code_hash=code_hash())
    second_receipt = resolve_retry_adjudication(store, second, execution_policy=policy_ref,
                                                source_code_hash=code_hash())
    assert second_receipt["retry_ordinal"] == 2
    assert resolve_retry_adjudication(store, first, execution_policy=policy_ref,
                                      source_code_hash=code_hash()) == first_receipt

    second_child = second_receipt["retry_attempt_id"]
    current_resource_interruption(store, current, policy_ref, attempt_id=second_child, cost=43)
    with pytest.raises(ResearchError) as error:
        adjudicate_resource_retry(store, second_child, execution_policy=policy_ref,
                                  source_code_hash=code_hash())
    assert error.value.code == "RETRY_EXHAUSTED"
    assert len([event for event in store.events()
                if event["event_kind"] == "RESOURCE_RETRY_ADJUDICATED"]) == 2


def test_cross_epoch_child_can_use_two_current_run_retries(tmp_path, monkeypatch):
    store, historical, current, policy_ref, bridge_ref = epoch_fixture(tmp_path)
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: False)
    historical_parent, _ = interrupted_parent(store, historical)
    historical_reference = adjudicate_restart(store, historical_parent, source_bridge=bridge_ref,
        execution_policy=policy_ref, source_code_hash=code_hash())
    historical_receipt = resolve_restart_adjudication(store, historical_reference,
        execution_policy=policy_ref, source_code_hash=code_hash())
    first_parent, _, _ = current_resource_interruption(store, current, policy_ref,
        attempt_id=historical_receipt["retry_attempt_id"])
    first = adjudicate_resource_retry(store, first_parent, execution_policy=policy_ref,
                                      source_code_hash=code_hash())
    first_receipt = resolve_retry_adjudication(store, first, execution_policy=policy_ref,
                                               source_code_hash=code_hash())
    assert first_receipt["retry_ordinal"] == 1
    current_resource_interruption(store, current, policy_ref,
                                  attempt_id=first_receipt["retry_attempt_id"], cost=41)
    second = adjudicate_resource_retry(store, first_receipt["retry_attempt_id"],
                                       execution_policy=policy_ref, source_code_hash=code_hash())
    assert resolve_retry_adjudication(store, second, execution_policy=policy_ref,
                                      source_code_hash=code_hash())["retry_ordinal"] == 2
    assert resolve_restart_adjudication(store, historical_reference,
        execution_policy=policy_ref, source_code_hash=code_hash()) == historical_receipt


def test_current_resource_retry_runs_verifies_and_reopens_only_its_arm(tmp_path, monkeypatch):
    store, _, current, policy_ref, _ = epoch_fixture(tmp_path)
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: False)
    parent, study, cell = current_resource_interruption(store, current, policy_ref)
    reference = adjudicate_resource_retry(store, parent, execution_policy=policy_ref,
                                          source_code_hash=code_hash())
    registry, recovery = registries()
    runner = SharedRunner(store, registry, recovery)
    budget = ResourceBudgetSpec(policy_ref, code_hash())
    result = runner.run_cell(study["study_id"], digest(cell), budget=budget,
                             restart_adjudication=reference)
    assert result["state"] == "SUCCEEDED"
    assert runner.run_cell(study["study_id"], digest(cell), budget=budget)["reused"] is True
    continuation = promote_resource_arm(store, reference)
    assert resource_arm_continuation(store, cell["arm_id"], execution_policy=policy_ref,
        source_code_hash=code_hash())["reference"] == continuation
    assert not BudgetLedger(store).balance(cell["arm_id"], execution_policy=policy_ref,
                                          source_code_hash=code_hash())["closed"]
    assert BudgetLedger(store).balance("pirc28-k1", execution_policy=policy_ref,
                                      source_code_hash=code_hash())["closed"] is False


@pytest.mark.parametrize("damage", ["missing-stop", "changed-cost", "checkpoint"])
def test_current_resource_retry_refuses_incomplete_or_changed_evidence(
        tmp_path, monkeypatch, damage):
    store, _, current, policy_ref, _ = epoch_fixture(tmp_path)
    parent, _, _ = current_resource_interruption(store, current, policy_ref)
    if damage == "checkpoint":
        store.append("CHECKPOINT", {"attempt_id": parent, "checkpoint_id": digest("unexpected")})
    else:
        events = deepcopy(store.events())
        if damage == "missing-stop":
            events = [event for event in events if event["event_kind"] != "WORKER_TREE_STOPPED"]
        else:
            next(event for event in events if event["event_kind"] == "SETTLE")["payload"][
                "charged_ms"] += 1
        monkeypatch.setattr(store, "_events", lambda: events)
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: False)
    with pytest.raises(ResearchError) as error:
        adjudicate_resource_retry(store, parent, execution_policy=policy_ref,
                                  source_code_hash=code_hash())
    assert error.value.code == "RECOVERY_REQUIRED"
    assert not any(event["event_kind"] == "RESOURCE_RETRY_ADJUDICATED"
                   for event in store.events())


def test_current_resource_retry_refuses_possibly_live_worker(tmp_path, monkeypatch):
    store, _, current, policy_ref, _ = epoch_fixture(tmp_path)
    parent, _, _ = current_resource_interruption(store, current, policy_ref)
    monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: True)
    monkeypatch.setattr("application.research_restart_adjudication._pid_reuse_observation",
                        lambda *args: None)
    with pytest.raises(ResearchError) as error:
        adjudicate_resource_retry(store, parent, execution_policy=policy_ref,
                                  source_code_hash=code_hash())
    assert error.value.code == "WORKER_ACTIVE"
