"""Pre-read data protocol enforcement using the shared physical exposure log."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

from infrastructure.research_store import ResearchError, ResearchStore, digest, identifier


class EvaluationExposureLedger:
    def __init__(self, store: ResearchStore):
        self.store = store

    def register_protocol(self, protocol: dict, expected_hash: str):
        if protocol.get("schema_version") != "pirc25-data-protocol-v1" or digest(protocol) != expected_hash:
            raise ResearchError("CONTRACT_MISMATCH", "data protocol hash or schema")
        identifier(protocol.get("protocol_id"))
        identifier(protocol.get("study_id"))
        blocks = protocol.get("blocks", [])
        if not blocks or len({b["block_id"] for b in blocks}) != len(blocks):
            raise ResearchError("CONTRACT_MISMATCH", "missing or overlapping independent blocks")
        for block in blocks:
            if block.get("split_role") not in {"train", "selection", "validation", "test", "final-eval"}:
                raise ResearchError("CONTRACT_MISMATCH", "unknown source role; explicit legacy mapping required")
            if block.get("fit_scope") and block["split_role"] != "train":
                raise ResearchError("CONTRACT_MISMATCH", "only train blocks may fit preprocessing")
            if not block.get("dataset_id") or not block.get("release_id") or len(block.get("sha256", "")) != 64:
                raise ResearchError("CONTRACT_MISMATCH", "data identity incomplete")
        self.store.publish("protocol-" + protocol["protocol_id"], protocol)

    def read(self, protocol_id: str, block_id: str, *, purpose: str,
             authorization_id: str, data_root: Path) -> bytes:
        with self.store.lock():
            protocol = self.store._manifest("protocol-" + identifier(protocol_id))
            selected = [b for b in protocol["blocks"] if b["block_id"] == block_id]
            if len(selected) != 1:
                raise ResearchError("UNAUTHORIZED_DATA", "block is not in frozen protocol")
            block = selected[0]
            allowed = False
            try:
                grant = self.store._manifest("authorization-" + identifier(authorization_id))
                purposes = {"train": {"fit"}, "selection": {"select"}, "validation": {"validate"},
                            "test": {"evaluate"}, "final-eval": {"evaluate"}}[block["split_role"]]
                allowed = (purpose in purposes and purpose in grant["purposes"]
                           and protocol["study_id"] == grant["study_id"]
                           and block_id in grant["block_ids"]
                           and grant.get("protocol_hash") == digest(protocol)
                           and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc))
                if block["split_role"] in {"test", "final-eval"}:
                    allowed = (allowed and protocol.get("preregistration_hash") is not None
                               and protocol.get("history_status") == "known"
                               and grant.get("test_authorization") is True)
                if purpose == "fit" and not block.get("fit_scope"):
                    allowed = False
            except (ResearchError, KeyError, ValueError, TypeError):
                allowed = False
            request = {"study_id": protocol["study_id"], "protocol_hash": digest(protocol),
                       "block_id": block_id, "dataset_id": block["dataset_id"],
                       "release_id": block["release_id"], "purpose": purpose,
                       "authorization_id": authorization_id, "allowed": bool(allowed)}
            self.store._append("EXPOSURE_ALLOWED" if allowed else "EXPOSURE_DENIED", request)
            if not allowed:
                raise ResearchError("UNAUTHORIZED_DATA", "data purpose, protocol or grant mismatch")
            root = Path(data_root).resolve()
            path = (root / block["path"]).resolve()
            if Path(block["path"]).is_absolute() or not path.is_relative_to(root):
                raise ResearchError("UNAUTHORIZED_DATA", "data path escapes authorized root")
            self.store._append("READ_STARTED", request)
            try:
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != block["sha256"]:
                    raise ResearchError("CORRUPT_ARTIFACT", "data content differs from protocol")
            except (OSError, ResearchError):
                self.store._append("READ_FAILED", request)
                raise
            self.store._append("READ_COMPLETED", request)
            return content
