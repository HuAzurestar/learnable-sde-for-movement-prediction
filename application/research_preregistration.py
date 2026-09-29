"""Frozen test plans and imported exposure evidence, independent of grants.

The operator imports external history; software verifies its content bindings
and never claims to detect reads made outside the controlled ledger.
"""

from __future__ import annotations

import re

from infrastructure.research_store import ResearchError, digest, identifier


def hash_reference(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def protocol_binding(protocol):
    """Exclude only evidence pointers to avoid a circular content hash."""
    return digest({key: value for key, value in protocol.items()
                   if key not in {"preregistration_hash", "history_hash", "history_status"}})


def source_identity(block):
    return {"dataset_id": block["dataset_id"], "release_id": block["release_id"],
            "source_block_id": block.get("source_block_id", block["block_id"]),
            "sha256": block["sha256"]}


def same_source(left, right):
    # Renaming a window/study/release cannot wash the same content or source.
    return (left.get("dataset_id") == right.get("dataset_id") and
            (not hash_reference(left.get("sha256")) or
             left.get("sha256") == right.get("sha256") or
             left.get("source_block_id", left.get("block_id")) ==
             right.get("source_block_id", right.get("block_id"))))


def validate_preregistration(value):
    if not isinstance(value, dict) or value.get("schema_version") != "pirc25-preregistration-v1":
        raise ResearchError("CONTRACT_MISMATCH", "preregistration schema")
    if value.get("test_mode") not in {"blind", "exploratory"}:
        raise ResearchError("CONTRACT_MISMATCH", "explicit blind/exploratory test mode required")
    for key in ("study_ids", "protocol_bindings", "primary_metrics"):
        items = value.get(key)
        if (not isinstance(items, list) or not items or
                any(not isinstance(item, str) or not item.strip() for item in items) or len(set(items)) != len(items)):
            raise ResearchError("CONTRACT_MISMATCH", "missing or duplicate preregistration " + key)
    for study in value["study_ids"]:
        identifier(study)
    if not all(hash_reference(item) for item in value["protocol_bindings"]):
        raise ResearchError("CONTRACT_MISMATCH", "invalid frozen protocol bindings")
    for key in ("selection_rule", "stopping_rule"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ResearchError("CONTRACT_MISMATCH", "missing preregistration " + key)
    if not isinstance(value.get("comparisons"), list) or not value["comparisons"]:
        raise ResearchError("CONTRACT_MISMATCH", "preregistered comparisons required")


def validate_history(value):
    if (not isinstance(value, dict) or value.get("schema_version") != "pirc25-exposure-history-v1"
            or not isinstance(value.get("source"), str) or not value["source"].strip()
            or not hash_reference(value.get("source_evidence_hash"))):
        raise ResearchError("CONTRACT_MISMATCH", "historical exposure source/evidence missing")
    # Retain the imported evidence, not just a caller's 'known' flag or hash.
    report = value.get("source_evidence")
    if not isinstance(report, dict) or digest(report) != value["source_evidence_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "historical evidence content/hash mismatch")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ResearchError("CONTRACT_MISMATCH", "historical evidence needs scoped records")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ResearchError("CONTRACT_MISMATCH", "invalid historical record")
        for key in ("dataset_id", "release_id", "source_block_id"):
            identifier(record.get(key))
        if not hash_reference(record.get("sha256")) or record.get("status") not in {"unexposed", "exposed", "unknown"}:
            raise ResearchError("CONTRACT_MISMATCH", "historical data identity/status missing")
        identity = tuple(record[key] for key in ("dataset_id", "release_id", "source_block_id", "sha256"))
        if identity in seen:
            raise ResearchError("CONTRACT_MISMATCH", "duplicate historical evidence identity")
        seen.add(identity)


class PreregistrationGate:
    def __init__(self, store):
        self.store = store

    def register_preregistration(self, value, expected_hash):
        validate_preregistration(value)
        return self._register("preregistration", value, expected_hash)

    def import_history(self, value, expected_hash):
        validate_history(value)
        return self._register("exposure-history", value, expected_hash)

    def _register(self, kind, value, expected_hash):
        if not hash_reference(expected_hash) or digest(value) != expected_hash:
            raise ResearchError("CONTRACT_MISMATCH", "frozen evidence hash mismatch")
        return self.store.publish(kind + "-" + expected_hash, value)

    def _resolve(self, kind, reference, validator):
        if not hash_reference(reference):
            raise ResearchError("UNAUTHORIZED_DATA", "valid frozen " + kind + " reference required")
        value = self.store._manifest(kind + "-" + reference)
        if digest(value) != reference:
            raise ResearchError("UNAUTHORIZED_DATA", "frozen evidence content binding differs")
        validator(value)
        return value

    def _validate(self, protocol, block):
        """Caller holds the store lock through validation and READ_STARTED."""
        prereg = self._resolve("preregistration", protocol.get("preregistration_hash"), validate_preregistration)
        history = self._resolve("exposure-history", protocol.get("history_hash"), validate_history)
        if (protocol["study_id"] not in prereg["study_ids"] or
                protocol_binding(protocol) not in prereg["protocol_bindings"]):
            raise ResearchError("UNAUTHORIZED_DATA", "preregistration does not bind this study/protocol")
        identity = source_identity(block)
        matching = [record for record in history["source_evidence"]["records"]
                    if all(record[key] == value for key, value in identity.items())]
        if len(matching) != 1 or matching[0]["status"] == "unknown":
            raise ResearchError("UNAUTHORIZED_DATA", "historical coverage unknown or mismatched")
        events = self.store._events()
        frozen = next(event for event in events if event["event_kind"] == "MANIFEST" and
                      event["payload"]["object_id"] == "preregistration-" + protocol["preregistration_hash"])
        if prereg["test_mode"] == "blind":
            if matching[0]["status"] != "unexposed":
                raise ResearchError("UNAUTHORIZED_DATA", "previously exposed data cannot be blind test")
            for event in events:
                payload = event["payload"]
                if event["event_kind"] == "MANIFEST" and payload["object_id"].startswith("exposure-history-"):
                    prior = self.store._manifest(payload["object_id"])
                    validate_history(prior)
                    if any(same_source(record, identity) and record["status"] in {"exposed", "unknown"}
                           for record in prior["source_evidence"]["records"]):
                        raise ResearchError("UNAUTHORIZED_DATA", "prior imported exposure cannot be reset")
                if event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "READ_FAILED"}:
                    # Legacy reads without a content identity cannot prove a
                    # renamed window from that dataset is an untouched block.
                    if same_source(payload, identity) and event["sequence"] < frozen["sequence"]:
                        raise ResearchError("UNAUTHORIZED_DATA", "test plan was frozen after data exposure")
                    if payload.get("artifact_id") and event["sequence"] < frozen["sequence"]:
                        artifact = self.store._manifest("artifact-" + payload["artifact_id"])
                        if block["block_id"] in artifact["block_ids"]:
                            raise ResearchError("UNAUTHORIZED_DATA", "test plan was frozen after result disclosure")
        return {"preregistration_hash": protocol["preregistration_hash"],
                "history_hash": protocol["history_hash"], "protocol_binding": protocol_binding(protocol),
                "test_mode": prereg["test_mode"], "frozen_sequence": frozen["sequence"]}
