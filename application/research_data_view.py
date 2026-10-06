"""Authorized protocol/scope/exposure metadata; never opens source data bytes.

Checks describe this request's frozen scope, not an execution token, scientific
qualification or proof that other tools never accessed a block.
"""
from .research_admission import data_binding
from .research_data import EvaluationExposureLedger
from .research_preregistration import source_identity, same_source, validate_history
from infrastructure.research_store import ResearchError, digest
from infrastructure.research_artifact_scope import source_scope_keys, query_scope_keys

_READS = {"READ_STARTED", "READ_COMPLETED", "READ_FAILED"}


def require(condition):
    if not condition:
        raise ResearchError("CONTRACT_MISMATCH", "data access view differs from frozen protocol")


def event_ref(event):
    return {key: event[key] for key in ("event_id", "sequence", "hash", "created_at")} if event else None


def exposure_watermark(store):
    # Metadata disclosure journals are not raw/result byte reads. A successful
    # refresh must not invalidate its own cursor; a real new read must do so.
    return max((e["sequence"] for e in store.events() if e["event_kind"] in
                _READS | {"MANIFEST", "UPSTREAM_VALIDATION", "UPSTREAM_REFUSED"}), default=0)


class DataAccessView:
    def __init__(self, store, spec, events, preview_grant):
        self.store, self.spec, self.events = store, spec, events
        self.preview_grant = preview_grant
        self.settings = spec.get("admission", {})
        self.protocol, self.grant, self.grant_error = None, None, None
        self.metadata = {}
        self.checks = []
        self.histories, self.history_flags, self.artifacts, self.protocol_sources = {}, {}, {}, {}
        self.result_reads = {}
        if not self.settings.get("protocol_id"):
            return
        protocol = self._manifest("protocol-" + self.settings["protocol_id"])
        require(protocol.get("schema_version") == "pirc25-data-protocol-v1"
                and protocol.get("study_id") == spec["study_id"] and digest(protocol) == spec["protocol_hash"]
                and data_binding(protocol) == spec["data_hash"] and isinstance(protocol.get("blocks"), list))
        blocks = protocol["blocks"]
        require(len({b["block_id"] for b in blocks}) == len(blocks))
        self.protocol = protocol
        try:
            self.grant = store.authorization(self.settings["authorization_id"],
                                            version=self.settings.get("authorization_version"))
        except ResearchError as exc:
            if exc.code != "MISSING_INPUT":
                raise
            self.grant_error = exc.code

    def _manifest(self, object_id, *, frozen_exposure=False):
        value = (self.store._frozen_exposure_scope(object_id) if frozen_exposure
                 else self.store.manifest(object_id))
        content_hash = digest(value)
        require(object_id not in self.metadata or self.metadata[object_id] == content_hash)
        self.metadata[object_id] = content_hash
        return value

    def _history(self):
        lookup = self.store._event_lookup().source_history()
        for event in self.store._manifest_events("exposure-history-"):
            object_id = event["payload"]["object_id"]
            report = self._manifest(object_id)
            require(object_id == "exposure-history-" + digest(report))
            validate_history(report)
            key = (object_id, digest(report))
            lookup.add(key, report["source_evidence"]["records"])
            self.histories[key] = report["source_evidence"]["records"]
            for record in report["source_evidence"]["records"]:
                for scope in (("hash", record["sha256"]), ("source", record["dataset_id"], record["source_block_id"])):
                    self.history_flags.setdefault(scope, set()).add(record["status"])
        return lookup

    def _artifact_scopes(self):
        # Original legacy artifact reads only carry an artifact ID. Resolve
        # metadata, not result bytes. Frozen protocols retain source aliases
        # across studies and renamed block IDs. Do not return foreign payloads.
        for event in self.store._manifest_events("protocol-"):
            protocol = self._manifest(event["payload"]["object_id"], frozen_exposure=True)
            require(protocol.get("schema_version") == "pirc25-data-protocol-v1"
                    and isinstance(protocol.get("blocks"), list))
            for block in protocol["blocks"]:
                self.protocol_sources.setdefault((protocol["study_id"], block["block_id"]), set()).update(
                    source_scope_keys(source_identity(block)))
        for event in self.events:
            artifact_id = event["payload"].get("artifact_id") if isinstance(event["payload"], dict) else None
            if event["event_kind"] in _READS and artifact_id and artifact_id not in self.artifacts:
                metadata = self._manifest("artifact-" + artifact_id, frozen_exposure=True)
                require(metadata.get("artifact_id") == artifact_id and isinstance(metadata.get("block_ids"), list)
                        and all(isinstance(block, str) and block for block in metadata["block_ids"]))
                self.artifacts[artifact_id] = {"study_id": metadata["study_id"],
                                               "block_ids": tuple(metadata["block_ids"])}
            if event["event_kind"] in _READS and artifact_id:
                artifact = self.artifacts[artifact_id]
                keys = {("block", name) for name in artifact["block_ids"]}
                for name in artifact["block_ids"]:
                    keys.update(self.protocol_sources.get((artifact["study_id"], name), ()))
                for key in keys:
                    self.result_reads.setdefault(key, []).append(event)

    def _exposure(self, block, history):
        identity = source_identity(block)
        reference = self.protocol.get("history_hash")
        report_key = ("exposure-history-" + str(reference), reference)
        coverage = history.coverage(report_key, identity) if report_key in self.histories else None
        unknown = coverage is None or coverage == "unknown"
        # A same-source unknown report is conservative even when another report
        # says unexposed; importing/re-authorizing must not erase uncertainty.
        statuses = set().union(*(self.history_flags.get(scope, set()) for scope in (
            ("hash", identity["sha256"]), ("source", identity["dataset_id"], identity["source_block_id"]))))
        unknown = unknown or "unknown" in statuses
        raw = []
        for event in self.store._source_read_events(identity, len(self.events) + 1):
            payload = event["payload"]
            if not payload.get("artifact_id") and same_source(payload, identity):
                raw.append(event)
        matched = {event["sequence"]: event
                   for key in query_scope_keys(block["block_id"], identity)
                   for event in self.result_reads.get(key, ())}
        results = [matched[sequence] for sequence in sorted(matched)]
        related = sorted(raw + results, key=lambda event: event["sequence"])
        exposed = bool(related) or "exposed" in statuses
        def first(rows, kind):
            return event_ref(next((event for event in rows if event["event_kind"] == kind), None))
        return {"status": "EXPOSED" if exposed else "UNKNOWN" if unknown else "NO_RECORDED_EXPOSURE",
                "external_history_hash": reference, "external_coverage": coverage or "unknown",
                "raw_read_started": first(raw, "READ_STARTED"), "raw_read_completed": first(raw, "READ_COMPLETED"),
                "result_disclosure_started": first(results, "READ_STARTED"),
                "result_read_completed": first(results, "READ_COMPLETED"),
                "read_failed_count": sum(e["event_kind"] == "READ_FAILED" for e in related),
                "read_event_count": len(related)}

    def rows(self, cells):
        history = self._history() if self.protocol else None
        if self.protocol:
            self._artifact_scopes()
        result = []
        for cell in cells:
            row = {"cell_id": digest(cell), "study_id": self.spec["study_id"], "arm_id": cell["arm_id"],
                   "block_id": cell["block_id"], "purpose": None,
                   "authorization": {"status": "NOT_CONFIGURED", "reason_codes": []},
                   "exposure": {"status": "UNKNOWN"}}
            if self.protocol:
                blocks = [b for b in self.protocol["blocks"] if b["block_id"] == cell["block_id"]]
                require(len(blocks) == 1)
                block = blocks[0]
                purpose = self.settings.get("purpose")
                row["purpose"] = {"split_role": block["split_role"], "requested": purpose,
                                  "fit_scope": bool(block.get("fit_scope"))}
                allowed = False
                if self.grant is not None:
                    allowed, _, current = EvaluationExposureLedger(self.store)._read_authority(
                        self.protocol, block, purpose, self.grant["authorization_id"], expected=self.grant,
                        authorization_version=self.grant.get("version"))
                    allowed = (allowed and "execute" in current["purposes"]
                               and cell.get("visibility", "restricted") in current["visibilities"])
                self.checks.append((block, cell, bool(allowed)))
                row["authorization"] = {"status": "SCOPE_CHECK_PASSED" if allowed else "SCOPE_CHECK_DENIED",
                    "reason_codes": [] if allowed else [self.grant_error or "UNAUTHORIZED_DATA"],
                    "authorization_id": self.settings.get("authorization_id"),
                    "version": self.settings.get("authorization_version"),
                    "authorization_hash": digest(self.grant) if self.grant is not None else None,
                    "evidence_hash": self.grant.get("evidence_hash") if self.grant is not None else None,
                    "expires_at": self.grant.get("expires_at") if self.grant is not None else None}
                row["exposure"] = self._exposure(block, history)
            result.append(row)
        return result

    def verify_authority(self):
        from infrastructure.research_visibility import study_visibility
        require(self.store.manifest("study-" + self.spec["study_id"])["spec"] == self.spec)
        if study_visibility(self._manifest, self.spec) not in self.preview_grant["visibilities"]:
            raise ResearchError("UNAUTHORIZED_DATA", "data metadata source visibility changed")
        # These hashes are only request-local bindings, not reusable metadata or
        # permission caches. Rehash every actual consumed source after the final
        # physical chain pass, including when data access itself was denied.
        for object_id, expected_hash in self.metadata.items():
            require(digest(self.store.manifest(object_id)) == expected_hash)
        if self.grant is None:
            return
        require(self.store.authorization(self.grant["authorization_id"], version=self.grant.get("version")) == self.grant)
        ledger = EvaluationExposureLedger(self.store)
        for block, cell, expected in self.checks:
            allowed, _, grant = ledger._read_authority(self.protocol, block, self.settings.get("purpose"),
                self.grant["authorization_id"], expected=self.grant, authorization_version=self.grant.get("version"))
            allowed = (allowed and "execute" in grant["purposes"]
                       and cell.get("visibility", "restricted") in grant["visibilities"])
            if bool(allowed) != expected:
                raise ResearchError("UNAUTHORIZED_DATA", "data scope changed during metadata disclosure")

    def verify_expiry(self):
        if self.grant is not None and any(allowed for _, _, allowed in self.checks):
            EvaluationExposureLedger(self.store)._require_read_expiry(self.grant, {
                "study_id": self.spec["study_id"], "protocol_hash": digest(self.protocol),
                "purpose": self.settings.get("purpose"), "authorization_hash": digest(self.grant),
                "entrypoint": "data-access-metadata-view"})
