"""Immutable declared non-execution, not a qualification or budget authority.

Keep unavailable/ineligible rows in the registered matrix. Recording a refusal
creates a shared PREFLIGHT_FAILED attempt, never a grant, reservation or worker.
"""

from infrastructure.research_store import ResearchError, digest


def declared_execution_disposition(cell):
    if "execution_disposition" not in cell:
        return None
    value = cell["execution_disposition"]
    if (type(value) is not dict or set(value) != {"schema_version", "status", "reason"}
            or value["schema_version"] != "pirc25-execution-disposition-v1"
            or type(value["status"]) is not str
            or value["status"] not in {"NOT_IMPLEMENTED", "INELIGIBLE", "MISSING_INPUT", "UNQUALIFIED"}
            or type(value["reason"]) is not str or not 0 < len(value["reason"]) <= 512
            or any(ord(char) < 32 for char in value["reason"])
            or "execution" in cell):
        raise ResearchError("CONTRACT_MISMATCH", "invalid non-executable cell declaration")
    return dict(value)


def require_executable_cell(cell):
    declaration = declared_execution_disposition(cell)
    if declaration is not None:
        raise ResearchError(declaration["status"], "registered cell is declared non-executable")


def record_declared_refusal(store, study_id, cell, *, parent_attempt_id=None, reason=None):
    """Idempotent shared terminal metadata; do not execute or retry this cell."""
    declaration = declared_execution_disposition(cell)
    if declaration is None:
        raise ResearchError("CONTRACT_MISMATCH", "non-execution declaration required")
    if parent_attempt_id is not None or reason is not None:
        raise ResearchError("CONTRACT_MISMATCH", "immutable non-executable cell cannot be retried")
    # Share the existing thread/PID-owned verified scope. A raw nested writer
    # lock is not reentrant on Windows; this scope retains the actual OS lock
    # and performs fresh physical validation before returning the metadata.
    with store._read_transaction():
        # Membership/immutable study checks still use the original shared store.
        run_id = store.register_run(study_id, cell)
        prior = [attempt for attempt in store.attempts().values() if attempt["run_id"] == run_id]
        if prior:
            if (len(prior) != 1 or prior[0]["state"] != "PREFLIGHT_FAILED"
                    or prior[0]["error_code"] != declaration["status"]):
                raise ResearchError("IDENTITY_CONFLICT", "declared refusal has incompatible attempt history")
            attempt_id = prior[0]["attempt_id"]
        else:
            attempt_id = store.new_attempt(run_id)
            store.transition(attempt_id, "PREFLIGHT_FAILED", error_code=declaration["status"])
    return {"state": "PREFLIGHT_FAILED", "run_id": run_id, "attempt_id": attempt_id,
            "cell_hash": digest(cell), "error_code": declaration["status"],
            "reason": declaration["reason"], "reused_preflight": bool(prior)}
