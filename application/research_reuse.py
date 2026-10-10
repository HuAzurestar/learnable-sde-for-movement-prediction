"""Verify completed output and authoritative provenance without re-execution.

This is an owner-side integrity check, not a disclosure API. It returns only
the attempt reference; result bytes remain subject to the normal read grants.
"""

import json

from application.research_admission import AdmissionGate, command_binding, plugin_binding
from application.research_execution import execution_plan
from infrastructure.research_store import ResearchError, digest, encode


def verified_success_metadata(store, attempt, spec, cell, plugin):
    """Original reuse identity and measured-cost proof, without artifact I/O.

    Not result validation, reuse success, read permission or qualification.
    Callers must separately validate/read the result or a source-bound model
    attachment through original owner/disclosure authority.
    """
    with store._read_transaction():
        execution_plan(spec, cell, plugin)
        return _success_metadata(store, attempt, spec, cell, plugin=plugin)


def verified_historical_success_metadata(store, attempt, spec, cell, *, admission_hash):
    """Verify an immutable past execution, not current executable qualification.

    The caller must bind this exact admission hash from the original published
    result/attachment and separately verify current permission/code compatibility.
    This path never runs an old plugin or upgrades its scientific qualification.
    """
    with store._read_transaction():
        return _success_metadata(store, attempt, spec, cell, historical_admission=admission_hash)


def _success_metadata(store, attempt, spec, cell, *, plugin=None, historical_admission=None):
    with store._read_transaction():
        current = store._attempts().get(attempt["attempt_id"])
        run = store._manifest("run-" + attempt["run_id"])
        if (current != attempt or attempt["state"] != "SUCCEEDED"
                or run["spec_hash"] != digest(spec) or run["cell_hash"] != digest(cell)
                or run["study_id"] != spec["study_id"] or run["cell"] != cell):
            raise ResearchError("CONTRACT_MISMATCH", "reused attempt/run identity differs")
        metadata = store._manifest("artifact-" + attempt["artifact_id"])
        if (metadata["role"] != "result" or metadata["media_type"] != "application/json"
                or metadata["study_id"] != spec["study_id"]
                or metadata["block_ids"] != [cell["block_id"]]
                or digest(metadata) != attempt.get("artifact_manifest_hash")):
            raise ResearchError("CONTRACT_MISMATCH", "reused result metadata differs from completion")
        admissions = [e for e in store._events() if e["event_kind"] == "ADMISSION"
            and e["payload"].get("attempt_id") == attempt["attempt_id"]]
        if len(admissions) != 1:
            raise ResearchError("UNQUALIFIED", "unique successful admission absent")
        reference = admissions[0]["payload"].get("admission_hash")
        if not isinstance(reference, str):
            raise ResearchError("UNQUALIFIED", "successful admission reference absent")
        try:
            receipt = store._manifest("admission-" + reference)
        except ResearchError as exc:
            if exc.code != "MISSING_INPUT":
                raise
            raise ResearchError("UNQUALIFIED", "successful admission reference is not published") from exc
        if (receipt.get("schema_version") != "pirc25-admission-v1"
                or receipt.get("admission_hash") != reference
                or digest({key: value for key, value in receipt.items() if key != "admission_hash"}) != reference
                or receipt.get("attempt_id") != attempt["attempt_id"] or receipt.get("run_id") != run["run_id"]
                or receipt.get("spec") != spec or receipt.get("cell") != cell
                or receipt.get("spec_hash") != digest(spec) or receipt.get("cell_hash") != digest(cell)
                or (plugin is not None and receipt.get("plugin_hash") != plugin_binding(plugin))
                or (plugin is None and (reference != historical_admission
                    or receipt.get("documents", {}).get("package", {}).get("code_hash") != spec["code_hash"]))):
            raise ResearchError("UNQUALIFIED", "reused admission bindings differ")
        execution_kind = receipt.get("execution_kind")
        if plugin is None:
            expected_command = receipt.get("command_hash") if execution_kind in {"run", "resume"} else None
        elif execution_kind == "run":
            expected_command = command_binding(plugin.command_builder)
        elif execution_kind == "resume":
            expected_command = receipt.get("documents", {}).get("package", {}).get("recovery_command_hash")
        else:
            expected_command = None
        if not expected_command or expected_command != receipt.get("command_hash"):
            raise ResearchError("UNQUALIFIED", "reused command binding differs")
        events = store._events()
        admission_events = [event for event in events if event["event_kind"] == "ADMISSION"
            and event["payload"] == {"attempt_id": attempt["attempt_id"], "run_id": run["run_id"],
                                     "admission_hash": reference}]
        reservations = [event for event in events if event["event_kind"] == "RESERVE"
            and event["payload"].get("attempt_id") == attempt["attempt_id"]]
        settlements = [event for event in events if event["event_kind"] == "SETTLE"
            and event["payload"].get("attempt_id") == attempt["attempt_id"]]
        workers = [event for event in events if event["event_kind"] == "WORKER_STARTED"
            and event["payload"].get("attempt_id") == attempt["attempt_id"]]
        completions = [event for event in events if event["event_kind"] == "ATTEMPT" and event["payload"] == attempt]
        if not admission_events or len(reservations) != 1 or len(settlements) != 1 or len(workers) != 1 or not completions:
            raise ResearchError("UNQUALIFIED", "reused success has no complete execution/cost provenance")
        reservation, settlement, worker = reservations[0], settlements[0], workers[0]
        cost = settlement["payload"]
        expected_id = digest([store.store_id, attempt["attempt_id"]])
        elapsed, reserved = cost.get("monotonic_elapsed_ms"), cost.get("reserved_ms")
        if (any(event["payload"].get("reservation_id") != expected_id for event in (reservation, settlement, worker))
                or any(cost.get(key) != value for key, value in {
                    "attempt_id": attempt["attempt_id"], "run_id": run["run_id"], "study_id": spec["study_id"],
                    "arm_id": cell["arm_id"], "settled": True, "outcome": "SUCCEEDED"}.items())
                or any(reservation["payload"].get(key) != cost.get(key) for key in
                    ("attempt_id", "run_id", "study_id", "arm_id", "reserved_ms", "worker_slot"))
                or isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0
                or cost.get("charged_ms") != elapsed
                or isinstance(reserved, bool) or not isinstance(reserved, int) or not 0 < reserved <= 7200000
                or not reservation["sequence"] < worker["sequence"] < settlement["sequence"] < completions[0]["sequence"]
                or not any(event["sequence"] < worker["sequence"] for event in admission_events)):
            raise ResearchError("UNQUALIFIED", "reused execution/cost bindings differ")
        return {"receipt": receipt, "result_artifact": metadata, "settlement": settlement}


def verified_reuse(store, attempt, spec, cell, plugin):
    # Hooks may need fresh authority under this same owned scope. Preserve the
    # final uncached journal and permission guards; do not reacquire its OS lock.
    with store._read_transaction():
        request = {"attempt_id": attempt["attempt_id"], "run_id": attempt["run_id"],
                   "artifact_id": attempt.get("artifact_id")}
        store._append("REUSE_CHECK_STARTED", request)
        try:
            proof = verified_success_metadata(store, attempt, spec, cell, plugin)
            content = store._verified_artifact_content(proof["result_artifact"])
            result = json.loads(content)
            if not isinstance(result, dict) or content != encode(result):
                raise ResearchError("CORRUPT_ARTIFACT", "reused result must be a canonical object")
            receipt = proof["receipt"]
            if result.get("admission_hash") != receipt["admission_hash"]:
                raise ResearchError("UNQUALIFIED", "reused result lacks admission evidence")
            AdmissionGate(store).result_validator(receipt, spec, cell, plugin)(result)
        except (ResearchError, KeyError, TypeError, ValueError, OSError) as exc:
            error = exc if isinstance(exc, ResearchError) else ResearchError("CORRUPT_ARTIFACT", "reused evidence is malformed")
            store._append("REUSE_CHECK_FAILED", {**request, "error_code": error.code})
            raise error from (None if error is exc else exc)
        store._append("REUSE_VERIFIED", request)
        return {**attempt, "reused": True, "exit_code": 0}
