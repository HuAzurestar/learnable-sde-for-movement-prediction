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


def load_dispatch(batch):
    import sys
    sys.modules["pirc27_first_batch"] = batch
    path = Path(batch.__file__).with_name("pirc27_first_batch_dispatch.py")
    spec = importlib.util.spec_from_file_location("pirc27_dispatch_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_shared_cpu_supplement(batch):
    import sys
    sys.modules["pirc27_first_batch"] = batch
    path = Path(batch.__file__).with_name("pirc27_shared_cpu_supplement.py")
    spec = importlib.util.spec_from_file_location("pirc27_shared_cpu_supplement_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_cpu_supplement_is_two_new_gaussian_cells_without_budget_reset(
        batch, arms, tmp_path):
    from application.research_admission import AdmissionGate
    from application.research_queue import SHARED_CPU_POLICY
    from infrastructure.research_store import ResearchStore
    from experiments.pirc27.qualification_plugin import qualification_plugin

    supplement = load_shared_cpu_supplement(batch)
    store = ResearchStore(tmp_path, "shared-cpu-supplement-control", initialize=True)
    store.publish(batch.ORIGINAL, {"arms": list(arms.values())})
    store.append("ARM_CLOSED", {"arm_id": arms["exact"]["arm_id"], "reason": "prior interrupted attempt"})
    result = supplement.prepare(store)

    assert result["state"] == "REGISTERED_NOT_STARTED"
    assert result["settled_pirc27_ms_before"] == 0
    assert result["prospective_slot_seconds"] == 600
    spec = store.manifest("study-" + supplement.STUDY)["spec"]
    assert spec["arms"] == [arms["gaussian"]]
    assert spec["scientific_independent_n"] == 0
    assert {cell["comparison_dimensions"]["model_configuration"] for cell in spec["cells"]} == {
        "anisotropic-v1", "uncertain-initial-v1"}
    assert all(cell["execution"]["config"]["method"] == "gaussian" for cell in spec["cells"])
    assert all(cell["resource_sharing"] == SHARED_CPU_POLICY for cell in spec["cells"])
    from application.research_execution import execution_plan
    assert all(execution_plan(spec, cell, qualification_plugin())["matrix_cells"] == 2
               for cell in spec["cells"])

    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(supplement.STUDY, cell))
    receipt = AdmissionGate(store).prepare(spec, cell, qualification_plugin(), attempt)
    assert receipt["mode"] == "pilot" and receipt["qualification"] == "fixture"
    assert not any(event["event_kind"] in {"RESERVE", "WORKER_STARTED"} for event in store.events())


def test_dispatch_run_identity_is_literal_actual_owner_identity(batch, arms, tmp_path):
    from infrastructure.research_store import ResearchStore
    store = ResearchStore(tmp_path, "dispatch-identity-control", initialize=True)
    store.publish(batch.ORIGINAL, {"arms": list(arms.values())})
    batch.prepare(store)
    record = store.manifest("study-" + batch.STUDY)
    dispatch = load_dispatch(batch)
    for cell in record["spec"]["cells"]:
        assert dispatch.run_identity(record, cell) == store.register_run(batch.STUDY, cell)


def test_dispatch_inspects_only_owned_records_without_foreign_manifest_reads(
        batch, arms, tmp_path, monkeypatch):
    from infrastructure.research_store import ResearchStore
    store = ResearchStore(tmp_path, "dispatch-inspection-control", initialize=True)
    store.publish(batch.ORIGINAL, {"arms": list(arms.values())})
    result = batch.prepare(store)
    dispatch = load_dispatch(batch)
    monkeypatch.setattr(dispatch, "SPEC_HASH", result["spec_hash"])  # Disposable root only.
    original = store.manifest
    def exact_study_only(object_id):
        assert object_id == "study-" + batch.STUDY
        return original(object_id)
    monkeypatch.setattr(store, "manifest", exact_study_only)
    _, attempts, pending, balances = dispatch.inspect_owned(store)
    assert attempts == [] and len(pending) == 8
    assert all(b["committed_ms"] == 0 and not b["closed"] for b in balances)
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())


def test_closed_arm_cells_remain_visible_while_other_original_arm_can_dispatch(
        batch, cells, monkeypatch):
    from types import SimpleNamespace
    dispatch = load_dispatch(batch)
    spec = {"orchestrator_source_sha256": "planner"}
    balances = [{"arm_id": "pirc27-affine-" + method + "-mean-x", "closed": method == "exact"}
                for method in ("exact", "gaussian")]
    store = SimpleNamespace(TERMINAL={"SUCCEEDED"}, publish=lambda *args: None)
    monkeypatch.setattr(dispatch, "inspect_owned", lambda owner: (spec, [], cells, balances))
    selected = []
    def observe_dispatch(owner, study, cell_hash, *, budget):
        selected.append(cell_hash)
        assert budget.job_seconds == 300 and budget.category == "pilot"
        return {"state": "SUCCEEDED"}  # Selector control only, no research worker.
    monkeypatch.setattr(batch.SharedRunner, "run_cell", observe_dispatch)
    result = dispatch.next_cell(store)
    assert selected == [digest(cells[1])]
    assert result["blocked_cell_hashes"] == [digest(c) for c in cells if
                                           c["execution"]["config"]["method"] == "exact"]
    balances[1]["closed"] = True
    result = dispatch.next_cell(store)
    assert result["state"] == "REMAINING_CELLS_BLOCKED"
    assert len(result["blocked_cell_hashes"]) == 8
    assert selected == [digest(cells[1])]


@pytest.mark.parametrize("durable_success", [False, True])
def test_stopped_recovery_preserves_original_verdict_cost_and_terminal_identity(
        batch, arms, tmp_path, monkeypatch, durable_success):
    from application.research_admission import AdmissionGate
    from application.research_budget import BudgetLedger, BudgetSpec
    from infrastructure.research_store import ResearchStore, encode
    store = ResearchStore(tmp_path, "dispatch-reconcile-control", initialize=True)
    store.publish(batch.ORIGINAL, {"arms": list(arms.values())})
    prepared = batch.prepare(store)
    dispatch = load_dispatch(batch)
    monkeypatch.setattr(dispatch, "SPEC_HASH", prepared["spec_hash"])
    spec = store.manifest("study-" + batch.STUDY)["spec"]
    cell = spec["cells"][0]
    run_id = store.register_run(batch.STUDY, cell)
    attempt = store.new_attempt(run_id)
    reservation = BudgetLedger(store).reserve(attempt, BudgetSpec(batch.JOB_SECONDS, category="pilot"))
    receipt = AdmissionGate(store).prepare(spec, cell, batch.qualification_plugin(), attempt)
    store.transition(attempt, "RUNNING")
    store.append("WORKER_STARTED", {"attempt_id": attempt, "pid": 99999999,
        "reservation_id": reservation["reservation_id"], "deadline_monotonic": 1.0})
    stop = store.append("WORKER_TREE_STOPPED", {"attempt_id": attempt,
        "reservation_id": reservation["reservation_id"], "observed_elapsed_ms": 17,
        "confirmation": "native-job-or-process-group-no-running-descendants"})
    # Produce the same schema through the registered plugin, then apply the
    # real owner validator before standard artifact publication.
    import subprocess
    output = tmp_path / "worker-result.json"
    command = batch.qualification_plugin().command_builder(output, spec, cell)
    subprocess.run(command, check=True)
    value = __import__("json").loads(output.read_bytes())
    AdmissionGate(store).result_validator(receipt, spec, cell, batch.qualification_plugin())(value)
    artifact = store.artifact(encode(value), role="result", visibility="synthetic",
                              block_ids=[cell["block_id"]], study_id=batch.STUDY)
    monkeypatch.setattr(dispatch, "process_may_be_alive", lambda pid: False)
    def no_launch(*args, **kwargs):
        raise AssertionError("recovery must never launch another worker")
    monkeypatch.setattr(batch.SharedRunner, "run_cell", no_launch)
    if durable_success:
        BudgetLedger(store).settle(reservation["reservation_id"], 17, outcome="SUCCEEDED")
        with pytest.raises(ResearchError, match="settlement cannot be replaced"):
            dispatch.recover_stopped(store, attempt_id=attempt, stop_hash=stop["hash"])
        result = dispatch.reconcile_published(store)
        assert result == {"study_id": batch.STUDY, "attempt_id": attempt,
            "state": "SUCCEEDED", "artifact_id": artifact["artifact_id"],
            "elapsed_ms": 17, "reconciled": True}
        assert store.attempts()[attempt]["artifact_id"] == artifact["artifact_id"]
        assert not BudgetLedger(store).balance(cell["arm_id"])["closed"]
    else:
        with pytest.raises(ResearchError, match="durable supervisor success settlement required"):
            dispatch.reconcile_published(store)
        assert not any(e["event_kind"] == "SETTLE" for e in store.events())
        with pytest.raises(ResearchError, match="original native whole-tree stop"):
            dispatch.recover_stopped(store, attempt_id=attempt, stop_hash="0" * 64)
        monkeypatch.setattr(dispatch, "process_may_be_alive", lambda pid: True)
        with pytest.raises(ResearchError, match="may still be alive"):
            dispatch.recover_stopped(store, attempt_id=attempt, stop_hash=stop["hash"])
        assert not any(e["event_kind"] == "SETTLE" for e in store.events())
        monkeypatch.setattr(dispatch, "process_may_be_alive", lambda pid: False)
        original_transition = store.transition
        def interrupted_transition(*args, **kwargs):
            raise OSError("synthetic terminal-write outage")
        monkeypatch.setattr(store, "transition", interrupted_transition)
        with pytest.raises(OSError, match="terminal-write outage"):
            dispatch.recover_stopped(store, attempt_id=attempt, stop_hash=stop["hash"])
        assert store.attempts()[attempt]["state"] == "RUNNING"
        monkeypatch.setattr(store, "transition", original_transition)
        result = dispatch.recover_stopped(store, attempt_id=attempt, stop_hash=stop["hash"])
        assert result["state"] == "INTERRUPTED" and result["arm_closed"]
        assert result["elapsed_ms"] == 17 and result["reconciled"]
        assert dispatch.recover_stopped(store, attempt_id=attempt, stop_hash=stop["hash"]) == result
        assert store.attempts()[attempt]["artifact_id"] is None
        assert store._verified_artifact_content(artifact) == encode(value)
        assert len([e for e in store.events() if e["event_kind"] == "SETTLE"]) == 1
        assert len([e for e in store.events() if e["event_kind"] == "STOPPED_RECONCILIATION"]) == 1
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] == 17
