"""Pre-read data protocol enforcement using the shared physical exposure log."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

from infrastructure.research_store import ResearchError, ResearchStore, digest, identifier
from infrastructure.research_files import opened_regular_file
from .research_preregistration import PreregistrationGate, source_identity


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

    def _read_authority(self, protocol, block, purpose, authorization_id, expected=None):
        evidence, grant = {}, {}
        try:
            grant_id = "authorization-" + identifier(authorization_id)
            protocol_id = "protocol-" + protocol["protocol_id"]
            grant = self.store._manifest(grant_id)
            purposes = {"train": {"fit"}, "selection": {"select"}, "validation": {"validate"},
                        "test": {"evaluate"}, "final-eval": {"evaluate"}}[block["split_role"]]
            allowed = (self.store._manifest(protocol_id) == protocol
                       and (expected is None or grant == expected)
                       and purpose in purposes and purpose in grant["purposes"]
                       and protocol["study_id"] == grant["study_id"]
                       and block["block_id"] in grant["block_ids"]
                       and grant.get("protocol_hash") == digest(protocol))
            if block["split_role"] in {"test", "final-eval"}:
                allowed = allowed and grant.get("test_authorization") is True
                if allowed:
                    evidence = PreregistrationGate(self.store)._validate(protocol, block)
            if purpose == "fit" and not block.get("fit_scope"):
                allowed = False
            # Rehash after preregistration/history I/O, not just before it.
            allowed = (allowed and self.store._manifest(protocol_id) == protocol
                       and self.store._manifest(grant_id) == grant
                       and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc))
        except (ResearchError, KeyError, ValueError, TypeError):
            allowed = False
        return bool(allowed), evidence, grant

    def _require_read_authority(self, protocol, block, purpose, authorization_id, grant, request):
        allowed, _, _ = self._read_authority(protocol, block, purpose, authorization_id, expected=grant)
        if not allowed:
            self.store._append("EXPOSURE_DENIED", {**request, "allowed": False})
            raise ResearchError("UNAUTHORIZED_DATA", "data purpose, protocol or grant mismatch")

    def _require_read_expiry(self, grant, request):
        try:
            unexpired = datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc)
        except (KeyError, TypeError, ValueError):
            unexpired = False
        if not unexpired:
            self.store.append("EXPOSURE_DENIED", {**request, "allowed": False})
            raise ResearchError("UNAUTHORIZED_DATA", "data purpose, protocol or grant mismatch")

    def read(self, protocol_id: str, block_id: str, *, purpose: str,
             authorization_id: str, data_root: Path, consumer=None) -> bytes:
        return self._read(protocol_id, block_id, purpose=purpose, authorization_id=authorization_id,
                          data_root=data_root, consumer=consumer, materialize=True)

    def verify(self, protocol_id: str, block_id: str, *, purpose: str,
               authorization_id: str, data_root: Path, consumer=None) -> dict:
        """Verify actual admitted bytes without retaining an unused whole input."""
        return self._read(protocol_id, block_id, purpose=purpose, authorization_id=authorization_id,
                          data_root=data_root, consumer=consumer, materialize=False)

    def _read(self, protocol_id, block_id, *, purpose, authorization_id, data_root, consumer, materialize):
        with self.store._read_transaction():
            protocol = self.store._manifest("protocol-" + identifier(protocol_id))
            selected = [b for b in protocol["blocks"] if b["block_id"] == block_id]
            if len(selected) != 1:
                raise ResearchError("UNAUTHORIZED_DATA", "block is not in frozen protocol")
            block = selected[0]
            consumer = consumer or {}
            if consumer:
                attempt = self.store._attempts().get(consumer.get("attempt_id"))
                if not attempt or attempt["run_id"] != consumer.get("run_id"):
                    raise ResearchError("CONTRACT_MISMATCH", "data consumer attempt/run mismatch")
                run = self.store._manifest("run-" + attempt["run_id"])
                if run["study_id"] != protocol["study_id"] or run["cell"]["block_id"] != block_id:
                    raise ResearchError("CONTRACT_MISMATCH", "data consumer study/block mismatch")
            allowed, evidence, grant = self._read_authority(protocol, block, purpose, authorization_id)
            request = {"study_id": protocol["study_id"], "protocol_hash": digest(protocol),
                       "block_id": block_id, "dataset_id": block["dataset_id"],
                       "release_id": block["release_id"], "purpose": purpose,
                       **source_identity(block), **evidence,
                       **{key: consumer[key] for key in ("attempt_id", "run_id", "entrypoint") if key in consumer},
                       "authorization_id": authorization_id, "allowed": bool(allowed)}
            self.store._append("EXPOSURE_ALLOWED" if allowed else "EXPOSURE_DENIED", request)
            if not allowed:
                raise ResearchError("UNAUTHORIZED_DATA", "data purpose, protocol or grant mismatch")
            self._require_read_authority(protocol, block, purpose, authorization_id, grant, request)
            root = Path(data_root).resolve()
            path = root / block["path"]
            if Path(block["path"]).is_absolute() or not path.resolve().is_relative_to(root):
                raise ResearchError("UNAUTHORIZED_DATA", "data path escapes authorized root")
            self.store._append("READ_STARTED", request)
            self._require_read_authority(protocol, block, purpose, authorization_id, grant, request)
            try:
                with opened_regular_file(root, path) as (stream, size, verify_identity):
                    # Legacy protocols freeze a digest, not necessarily a size.
                    # Verify with bounded memory before materializing any input;
                    # legitimate large byte-return reads remain supported.
                    remaining, hasher = size, hashlib.sha256()
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ResearchError("CORRUPT_ARTIFACT", "data shrank during verification")
                        remaining -= len(chunk)
                        hasher.update(chunk)
                    if stream.read(1) or hasher.hexdigest() != block["sha256"]:
                        raise ResearchError("CORRUPT_ARTIFACT", "data content differs from protocol")
                    verify_identity()
                    content = {"sha256": hasher.hexdigest(), "size_bytes": size}
                    if materialize:
                        self._require_read_authority(protocol, block, purpose, authorization_id, grant, request)
                        verify_identity()
                        stream.seek(0)
                        content = stream.read(size + 1)
                        if len(content) != size or hashlib.sha256(content).hexdigest() != block["sha256"]:
                            raise ResearchError("CORRUPT_ARTIFACT", "data changed during materialization")
            except (OSError, ResearchError):
                self.store._append("READ_FAILED", request)
                raise
            self.store._append("READ_COMPLETED", request)
            self._require_read_authority(protocol, block, purpose, authorization_id, grant, request)
            self.store._read_completion(
                lambda: self._require_read_authority(protocol, block, purpose, authorization_id, grant, request),
                lambda: self._require_read_expiry(grant, request))
        # Physical scope verification is I/O too; do not cache an earlier clock.
        self._require_read_expiry(grant, request)
        return content
