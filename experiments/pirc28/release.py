"""Reviewed operator entry points for the PIRC-28 resource-policy release.

Preparation publishes only immutable policy/source metadata and one registered
restart child.  Execution is a separate explicit call.  Neither path grants
real-data, test, GPU or public-export authority.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from application.research_budget import ResourceBudgetSpec
from application.research_execution_policy import SCHEMA, resolve_policy
from application.research_restart_adjudication import (adjudicate_resource_retry,
    adjudicate_restart, promote_resource_arm, resolve_restart_adjudication,
    resource_arm_was_promoted)
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, digest
from .plan import load_plan, prepare_definition
from .runtime import registries, switching_plugin
from .source_epoch import register_source_epoch


def _require(condition, message, code="CONTRACT_MISMATCH"):
    if not condition:
        raise ResearchError(code, message)


def publish_resource_policy(store, *, task_id, session_id, instructions):
    """Publish the exact PIRC-28-only decision with finite resource guards."""
    _require(type(task_id) is str and type(session_id) is str
             and type(instructions) is list and instructions,
             "task/session/instructions decision provenance required")
    source = code_hash()
    value = {"schema_version": SCHEMA, "store_id": store.store_id, "feature_id": "PIRC-28",
        "arm_ids": ["pirc28-k1", "pirc28-k2"], "decision": "approved",
        "budget_mode": "resource-guarded-no-time-cap", "job_seconds": None, "arm_seconds": None,
        "max_infrastructure_retries": 2, "source_code_hash": source,
        "resources": {"cpu_slots": 1, "gpu_slots": 0,
            "minimum_system_free_bytes": 10 * 2**30,
            "minimum_runtime_free_bytes": 10 * 2**30,
            "minimum_available_memory_bytes": 2 * 2**30,
            "maximum_worker_rss_bytes": 4 * 2**30},
        "decision_basis": {"authority": "local-human-decision-record", "task_id": task_id,
            "session_id": session_id, "instructions": deepcopy(instructions)},
        "real_data_authorized": False, "test_authorization": False, "public_export": False}
    reference = {"manifest_id": "pirc28-execution-policy-" + digest(value), "sha256": digest(value)}
    store.publish(reference["manifest_id"], value)
    resolve_policy(store, reference, arm_id="pirc28-k1", source_code_hash=source)
    return reference


def prepare_restart_release(store, *, historical_definition_hash, parent_attempt_id,
                            task_id, session_id, instructions, plan_path=None,
                            prelaunch_cancellations=None, unfunded_supersession=None):
    """Register the current source bridge and one non-running K1 retry child."""
    plan_path = Path(plan_path or Path(__file__).parent / "plans" / "identifiability-v1.json")
    definition = prepare_definition(load_plan(plan_path))
    policy = publish_resource_policy(store, task_id=task_id, session_id=session_id,
                                     instructions=instructions)
    source = register_source_epoch(store, historical_definition_hash, definition,
                                   execution_policy=policy)
    restart = adjudicate_restart(store, parent_attempt_id,
        source_bridge=source["source_bridge"], execution_policy=policy,
        source_code_hash=definition.to_dict()["runtime_code_hash"],
        prelaunch_cancellations=prelaunch_cancellations, unfunded_supersession=unfunded_supersession)
    receipt = resolve_restart_adjudication(store, restart, execution_policy=policy,
                                           source_code_hash=definition.to_dict()["runtime_code_hash"])
    return {"execution_policy": policy, **source, "restart_adjudication": restart,
            "retry_attempt_id": receipt["retry_attempt_id"], "execution_started": False,
            "real_data_authorized": False}


def execute_first_restart(store, *, restart_adjudication):
    """Run, verify, then promote the single reviewed K1 restart."""
    reference = store.manifest(restart_adjudication["manifest_id"])
    policy = reference["execution_policy"]
    source = reference["current_code_hash"]
    _require(source == code_hash(), "restart release source changed", "IDENTITY_CONFLICT")
    receipt = resolve_restart_adjudication(store, restart_adjudication,
                                           execution_policy=policy, source_code_hash=source)
    run = store.manifest("run-" + receipt["current_run_id"])
    registry, recovery = registries()
    runner = SharedRunner(store, registry, recovery)
    budget = ResourceBudgetSpec(policy, source)
    attempt = store.attempts()[receipt["retry_attempt_id"]]
    if attempt["state"] == "REGISTERED":
        result = runner.run_cell(run["study_id"], run["cell_hash"], budget=budget,
                                 restart_adjudication=restart_adjudication)
    elif attempt["state"] == "SUCCEEDED":
        result = runner.run_cell(run["study_id"], run["cell_hash"], budget=budget)
    else:
        raise ResearchError("RECOVERY_REQUIRED", "reviewed first restart did not succeed")
    _require(result["state"] == "SUCCEEDED", "reviewed first restart did not succeed", "RECOVERY_REQUIRED")
    # A second idempotent lookup performs full result/cost/admission verification
    # and records REUSE_VERIFIED before the arm-wide continuation is published.
    verified = runner.run_cell(run["study_id"], run["cell_hash"], budget=budget)
    _require(verified.get("reused") is True, "successful restart reuse verification failed", "UNQUALIFIED")
    continuation = promote_resource_arm(store, restart_adjudication)
    return {"result": result, "verified": verified, "resource_arm_continuation": continuation,
            "historical_cost_retained_ms": receipt["retained_cost_ms"],
            "real_data_authorized": False}


def execute_next_source_cell(store, *, study_definition_hash, execution_policy):
    """Execute at most one pending current-epoch cell; safe to call repeatedly."""
    from .study import SyntheticStudyDefinition

    definition = SyntheticStudyDefinition.from_dict(
        store.manifest("pirc28-study-definition-" + study_definition_hash))
    source = definition.to_dict()["runtime_code_hash"]
    _require(source == code_hash(), "source matrix runtime changed", "IDENTITY_CONFLICT")
    from application.research_restart_adjudication import prime_policy_definitions
    prime_policy_definitions(store, execution_policy, source)
    for arm in ("pirc28-k1", "pirc28-k2"):
        resolve_policy(store, execution_policy, arm_id=arm, source_code_hash=source)
    _require(resource_arm_was_promoted(store, "pirc28-k1", execution_policy=execution_policy,
        source_code_hash=source),
        "verified first K1 continuation required before matrix dispatch", "RECOVERY_REQUIRED")
    registry, recovery = registries()
    runner = SharedRunner(store, registry, recovery)
    budget = ResourceBudgetSpec(execution_policy, source)
    attempts = store.attempts()
    total_cells = sum(len(store.manifest("study-" + study_id)["spec"]["cells"])
                      for study_id in definition.study_ids)
    terminal, successful, scientific_failures = 0, 0, 0
    for study_id in definition.study_ids:
        study = store.manifest("study-" + study_id)["spec"]
        for cell in sorted(study["cells"], key=lambda value: (value["replicate_id"], value["fit_K"])):
            run_id = store.register_run(study_id, cell)
            related = sorted((value for value in attempts.values() if value["run_id"] == run_id),
                             key=lambda value: value.get("started_at") or "")
            success = next((value for value in related if value["state"] == "SUCCEEDED"), None)
            if success is not None:
                terminal += 1
                successful += 1
                continue
            failed = next((value for value in reversed(related)
                           if value["state"] == "FAILED" and value.get("artifact_id") is not None), None)
            if failed is not None:
                terminal += 1
                scientific_failures += 1
                continue
            if any(value["state"] not in store.TERMINAL for value in related):
                return {"state": "CELL_ACTIVE", "study_id": study_id, "cell_hash": digest(cell),
                        "attempt_id": next(value["attempt_id"] for value in related
                                           if value["state"] not in store.TERMINAL),
                        "terminal_cells": terminal, "total_cells": total_cells}
            parent = related[-1] if related else None
            restart = None
            if parent is not None:
                _require(len(related) < 3, "source cell infrastructure retries exhausted", "RETRY_EXHAUSTED")
                balance = store.path / "recovery-hold.json"
                _require(not balance.exists(), "source matrix held for recovery", "RECOVERY_REQUIRED")
                from application.research_budget import BudgetLedger
                closed = BudgetLedger(store).balance(cell["arm_id"],
                    execution_policy=execution_policy, source_code_hash=source)["closed"]
                if closed:
                    restart = adjudicate_resource_retry(store, parent["attempt_id"],
                        execution_policy=execution_policy, source_code_hash=source)
            if restart is None:
                result = runner.run_cell(study_id, digest(cell), budget=budget,
                    parent_attempt_id=None if parent is None else parent["attempt_id"],
                    reason=None if parent is None else "PIRC-28 explicit infrastructure retry")
            else:
                result = runner.run_cell(study_id, digest(cell), budget=budget,
                                         restart_adjudication=restart)
                attempt = store.attempts()[result["attempt_id"]]
                if attempt["state"] == "SUCCEEDED":
                    verified = runner.run_cell(study_id, digest(cell), budget=budget)
                    _require(verified.get("reused") is True,
                             "resource retry reuse verification failed", "UNQUALIFIED")
                    continuation = promote_resource_arm(store, restart)
                elif attempt["state"] == "FAILED" and attempt.get("artifact_id"):
                    from application.research_reuse import verified_failure
                    verified = verified_failure(store, attempt, study, cell, switching_plugin())
                    continuation = promote_resource_arm(store, restart)
                else:
                    verified, continuation = None, None
            response = {"state": "CELL_DISPATCHED", "study_id": study_id,
                "cell_hash": digest(cell), "result": result,
                "terminal_cells_before": terminal, "total_cells": total_cells}
            if restart is not None:
                response.update(retry_adjudication=restart, verified=verified,
                                resource_arm_continuation=continuation)
            return response
    return {"state": "SOURCE_MATRIX_TERMINAL", "terminal_cells": terminal,
            "successful_cells": successful, "scientific_failure_cells": scientific_failures,
            "total_cells": total_cells, "real_data_authorized": False}
