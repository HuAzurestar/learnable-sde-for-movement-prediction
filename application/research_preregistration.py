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
    # A renamed dataset cannot wash a controlled read of identical content.
    # Within a dataset, retain the conservative legacy/source-block check.
    if hash_reference(left.get("sha256")) and left["sha256"] == right.get("sha256"):
        return True
    return (left.get("dataset_id") == right.get("dataset_id") and
            (not hash_reference(left.get("sha256")) or
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

    def _history_facts(self, object_id, lookup, *, expected=None):
        # Always consume/hash actual metadata, including when facts were already
        # compiled under this short owned prefix. Never reuse file bytes or an
        # authorization result. The key is the actual freshly verified content.
        value = self.store._manifest(object_id)
        content_hash = digest(value)
        if expected is not None and content_hash != expected:
            raise ResearchError('UNAUTHORIZED_DATA', 'frozen evidence content binding differs')
        report_key = (object_id, content_hash)
        if not lookup.contains(report_key):
            validate_history(value)
            lookup.add(report_key, value['source_evidence']['records'])
        return report_key

    def _validate(self, protocol, block):
        """Caller holds the store lock through validation and READ_STARTED."""
        prereg = self._resolve("preregistration", protocol.get("preregistration_hash"), validate_preregistration)
        history_hash = protocol.get('history_hash')
        if not hash_reference(history_hash):
            raise ResearchError('UNAUTHORIZED_DATA', 'valid frozen exposure-history reference required')
        history = self.store._event_lookup().source_history()
        report_key = self._history_facts('exposure-history-' + history_hash, history, expected=history_hash)
        if (protocol["study_id"] not in prereg["study_ids"] or
                protocol_binding(protocol) not in prereg["protocol_bindings"]):
            raise ResearchError("UNAUTHORIZED_DATA", "preregistration does not bind this study/protocol")
        identity = source_identity(block)
        status = history.coverage(report_key, identity)
        if status is None or status == 'unknown':
            raise ResearchError("UNAUTHORIZED_DATA", "historical coverage unknown or mismatched")
        frozen = self.store._manifest_event("preregistration-" + protocol["preregistration_hash"])
        if prereg["test_mode"] == "blind":
            if status != "unexposed":
                raise ResearchError("UNAUTHORIZED_DATA", "previously exposed data cannot be blind test")
            for event in self.store._manifest_events("exposure-history-"):
                payload = event["payload"]
                self._history_facts(payload['object_id'], history)
            if history.prior_exposure(identity):
                raise ResearchError("UNAUTHORIZED_DATA", "prior imported exposure cannot be reset")
            for event in self.store._prior_read_events(identity, frozen["sequence"]):
                payload = event["payload"]
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
