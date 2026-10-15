"""Opt-in disclosure of existing owned pilot bytes, never an admission token."""
from infrastructure.research_store import ResearchError, digest, encode


class PilotEventIndex:
    def __init__(self, events):
        self.events = {}
        for event in events:
            payload, kind = event["payload"], event["event_kind"]
            if (kind in {"RESERVE", "WORKER_STARTED", "WORKER_TREE_STOPPED", "SETTLE", "ADMISSION"}
                    or kind == "ATTEMPT" and payload.get("state") == "SUCCEEDED"):
                self.events.setdefault(payload.get("attempt_id"), {}).setdefault(kind, []).append(event)

    def proof(self, attempt, run):
        found = self.events.get(attempt["attempt_id"], {})
        keys = {"reservation_event": "RESERVE", "worker_event": "WORKER_STARTED",
            "stop_event": "WORKER_TREE_STOPPED", "settlement_event": "SETTLE",
            "admission_event": "ADMISSION", "completion_event": "ATTEMPT"}
        if any(len(found.get(kind, [])) != 1 for kind in keys.values()):
            raise ResearchError("CORRUPT_ARTIFACT", "pilot export lacks unique native execution sources")
        body = {"schema_version": "owned-path-pilot-execution-v1", "attempt": attempt, "run": run,
            **{key: found[kind][0] for key, kind in keys.items()}}
        if body["completion_event"]["payload"] != attempt:
            raise ResearchError("CORRUPT_ARTIFACT", "pilot completion differs from actual terminal attempt")
        return {**body, "proof_hash": digest(body)}


def attach_pilot_result(store, row, result, attempt, index):
    receipt = row.get("admission", {})
    cell = row["registered_cell"]
    if (receipt.get("mode") != "pilot" or receipt.get("qualification") != "fixture"
            or cell.get("execution_role") != "qualification" or cell.get("visibility") != "synthetic"
            or "path_qualification_analysis" not in result.get("forecast", {})
            or len(encode(result)) > 512*1024):
        raise ResearchError("UNQUALIFIED", "explicit bounded synthetic path pilot required for proof export")
    metadata = store.manifest("artifact-" + attempt["artifact_id"])
    if (metadata["artifact_id"] != digest(result) or digest(metadata) != attempt["artifact_manifest_hash"]
            or metadata["size_bytes"] != len(encode(result))):
        raise ResearchError("CORRUPT_ARTIFACT", "pilot export differs from owned artifact bytes")
    row.update(result=result, result_artifact=metadata,
        path_pilot_execution=index.proof(attempt, store.manifest("run-" + attempt["run_id"])))
