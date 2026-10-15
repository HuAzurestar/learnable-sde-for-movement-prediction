"""Freeze native-stopped per-attempt observations; never re-execute research."""
import json

from .propagation_resources import observation_enabled, validate_packet, require
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError, digest, encode
from infrastructure.research_visibility import study_visibility, admission_visibility, combine_visibility


EVENT = "PROPAGATION_RESOURCE_RECORDED"
ROLE = "propagation-resource-observation"
PEAK_SCOPE = "not-additive;not-a-phase-delta;not-a-tree-peak"


class ResourceEventIndex:
    def __init__(self, events):
        self.by_hash = {}
        self.by_attempt = {}
        for event in events:
            self.by_hash[event["hash"]] = event
            if event["event_kind"] in {"WORKER_STARTED", "WORKER_TREE_STOPPED", "SETTLE", "RESUME", EVENT}:
                self.by_attempt.setdefault(event["payload"].get("attempt_id"), []).append(event)

    def proof(self, attempt_id):
        events = self.by_attempt.get(attempt_id, [])
        starts = [e for e in events if e["event_kind"] == "WORKER_STARTED"]
        stops = [e for e in events if e["event_kind"] == "WORKER_TREE_STOPPED"]
        settled = [e for e in events if e["event_kind"] == "SETTLE"]
        if len(starts) != 1:
            return None, None, "NOT_LAUNCHED" if not starts else "UNCONFIRMED_STOP"
        reservation = starts[0]["payload"]["reservation_id"]
        stops = [e for e in stops if e["payload"].get("reservation_id") == reservation
                 and e["payload"].get("confirmation") == "native-job-or-process-group-no-running-descendants"]
        settled = [e for e in settled if e["payload"].get("reservation_id") == reservation and e["payload"].get("settled") is True]
        if len(stops) != 1 or not starts[0]["sequence"] < stops[0]["sequence"]:
            return None, None, "UNCONFIRMED_STOP"
        if len(settled) != 1 or not stops[0]["sequence"] < settled[0]["sequence"]:
            return stops[0], None, "UNSETTLED_COST"
        return stops[0], settled[0], None


def _binding(spec, cell, attempt):
    return {"schema_version": "pirc27-resource-record-v1", "attempt_id": attempt["attempt_id"],
        "run_id": attempt["run_id"], "spec_hash": digest(spec), "cell_hash": digest(cell),
        "code_hash": spec["code_hash"], "policy_hash": digest(cell["resource_observation_policy"]),
        "attempt_state": attempt["state"], "result_artifact_id": attempt["artifact_id"]}


def record_observation(store, attempt_id, spec, cell, receipt, *, validated_packet=None):
    """Called after supervisor cleanup/settlement, including linked resumes.

    Successful packets are detached by the owner result validator. This does
    not reopen a protected prediction artifact under a new implicit purpose.
    Only stopped failures/checkpoints may read their bounded private sidecar.
    """
    if not observation_enabled(cell):
        return None
    with store._read_transaction():
        attempt = store._attempts()[attempt_id]
        if attempt["state"] not in store.TERMINAL:
            return None  # No observer publication can invent a terminal state.
        run = store.manifest("run-"+attempt["run_id"])
        require(run["study_id"] == spec["study_id"] and run["cell_hash"] == digest(cell), "owner run binding differs")
        index = ResourceEventIndex(store.events())
        prior = [e for e in index.by_attempt.get(attempt_id, []) if e["event_kind"] == EVENT]
        binding = _binding(spec, cell, attempt)
        if prior:
            require(len(prior) == 1 and encode({k: prior[0]["payload"].get(k) for k in binding}) == encode(binding),
                    "resource publication replay changed binding")
            return prior[0]
        stop, settlement, reason = index.proof(attempt_id)
        packet = validated_packet if attempt["state"] == "SUCCEEDED" else None
        if reason is None:
            try:
                if packet is None:
                    path = store.path/"artifacts"/(".attempt-"+attempt_id)/"propagation-resources.json"
                    with opened_regular_file(store.path/"artifacts", path, maximum_bytes=16384) as (stream, _, _):
                        content = stream.read(16385)
                    packet = json.loads(content)
                    require(content == encode(packet), "sidecar is not canonical JSON")
                validate_packet(packet, spec, cell, attempt_id,
                    execution_state="SUCCEEDED" if attempt["state"] == "SUCCEEDED" else
                                    "CHECKPOINTED" if attempt["error_code"] == "CHECKPOINT_SAVED" else None)
                resumed = any(e["event_kind"] == "RESUME" for e in index.by_attempt.get(attempt_id, []))
                require(packet["resumed"] is resumed, "resource packet substituted resume history")
            except FileNotFoundError:
                packet, reason = None, "MISSING_OBSERVATION"
            except (OSError, ValueError, TypeError, KeyError):
                packet, reason = None, "INVALID_OBSERVATION"
        artifact = None
        if reason is None:
            visibility = study_visibility(store.manifest, spec)
            if receipt:
                visibility = combine_visibility([visibility, admission_visibility(store.manifest, receipt)])
            artifact = store.artifact(encode(packet), role=ROLE, visibility=visibility,
                block_ids=[cell["block_id"]], study_id=spec["study_id"])
        payload = {**binding, "status": "RECORDED" if artifact else "UNAVAILABLE", "reason": reason,
            "artifact_id": artifact["artifact_id"] if artifact else None,
            "packet_hash": digest(packet) if artifact else None,
            "worker_stop_event_hash": stop["hash"] if stop else None,
            "settlement_event_hash": settlement["hash"] if settlement else None}
        return store.append(EVENT, payload, "pirc27-resources-"+attempt_id)


def resource_history(store, spec, cell, attempts, authorization, *, index=None):
    """Copy every attempt under fresh export authority; never sum RSS peaks."""
    index = index or ResourceEventIndex(store.events())
    rows = []
    for attempt in attempts:
        sources = [e for e in index.by_attempt.get(attempt["attempt_id"], []) if e["event_kind"] == EVENT]
        require(len(sources) <= 1, "duplicate resource publications")
        row = {"attempt_id": attempt["attempt_id"], "status": "UNAVAILABLE", "reason": "NOT_RECORDED",
               "artifact": None, "packet": None, "publication_event": None,
               "worker_stop_event": None, "settlement_event": None}
        if sources:
            event = sources[0]
            payload = event["payload"]
            binding = _binding(spec, cell, attempt)
            require(encode({k: payload.get(k) for k in binding}) == encode(binding), "published resource source changed")
            stop, settled, proof_reason = index.proof(attempt["attempt_id"])
            require(payload["worker_stop_event_hash"] == (stop["hash"] if stop else None)
                    and payload["settlement_event_hash"] == (settled["hash"] if settled else None), "published stop/cost source changed")
            row.update(status=payload["status"], reason=payload["reason"], publication_event=event,
                       worker_stop_event=stop, settlement_event=settled)
            if payload["status"] == "RECORDED":
                require(proof_reason is None and settled["sequence"] < event["sequence"]
                        and payload["reason"] is None, "observation published before native stop/settlement")
                metadata = store.manifest("artifact-"+payload["artifact_id"])
                require(metadata["role"] == ROLE and metadata["study_id"] == spec["study_id"]
                        and metadata["size_bytes"] <= 16384, "resource artifact scope/size differs")
                content = store.read_artifact(metadata["artifact_id"], purpose="export", authorization=authorization)
                packet = json.loads(content)
                require(content == encode(packet) and digest(packet) == payload["packet_hash"] == metadata["artifact_id"],
                        "resource packet bytes changed")
                validate_packet(packet, spec, cell, attempt["attempt_id"],
                    execution_state="SUCCEEDED" if attempt["state"] == "SUCCEEDED" else
                                    "CHECKPOINTED" if attempt["error_code"] == "CHECKPOINT_SAVED" else None)
                resumed = any(e["event_kind"] == "RESUME" for e in index.by_attempt.get(attempt["attempt_id"], []))
                require(packet["resumed"] is resumed, "resource resume history changed")
                row.update(artifact=metadata, packet=packet)
            else:
                require(payload["status"] == "UNAVAILABLE" and payload["artifact_id"] is None
                        and payload["packet_hash"] is None and type(payload["reason"]) is str,
                        "unknown observation became a measurement")
        rows.append(row)
    return {"schema_version": "pirc27-resource-history-v1", "peak_aggregation": PEAK_SCOPE, "attempts": rows}
