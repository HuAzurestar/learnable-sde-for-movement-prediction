"""Authorized read-only result views; all numbers come from frozen artifacts."""

from __future__ import annotations

import base64
import json

from .research_evidence import authorize_study, require_unexpired_disclosure
from .research_budget import BudgetLedger
from .research_dimensions import comparison_dimensions
from infrastructure.research_index import ResearchIndex
from infrastructure.research_store import ResearchError, digest, identifier

MAX_RESPONSE = 2 * 1024 * 1024


class ResearchQuery:
    def __init__(self, store, authorization_id, *, authorization_version=None):
        self.store = store
        self.authorization_id = identifier(authorization_id)
        self.authorization_version = identifier(authorization_version) if authorization_version is not None else None

    def _grant(self, purpose="preview"):
        grant = self.store.authorization(self.authorization_id, version=self.authorization_version)
        authorize_study(self.store, grant["study_id"], grant, purpose)
        spec = self.store.manifest("study-" + grant["study_id"])["spec"]
        if not {c["block_id"] for c in spec["cells"]} <= set(grant["block_ids"]):
            raise ResearchError("UNAUTHORIZED_DATA", "session grant must cover the study matrix")
        if not {c.get("visibility", "restricted") for c in spec["cells"]} <= set(grant["visibilities"]):
            raise ResearchError("UNAUTHORIZED_DATA", "session cannot disclose this study visibility")
        from infrastructure.research_visibility import study_visibility
        if study_visibility(self.store.manifest, spec) not in grant["visibilities"]:
            raise ResearchError("UNAUTHORIZED_DATA", "session cannot disclose the study source lineage")
        return grant

    def _objects(self, kind, grant):
        # Rebuild and every page must see the same physical authority chain.
        # Another read's disclosure journal is not a data change, but must not
        # land between these index operations and invalidate their watermark.
        with self.store._read_transaction():
            authorize_study(self.store, grant['study_id'], grant, 'preview')
            return self._project_objects(kind, grant)

    def _data_watermark(self):
        return max((event['sequence'] for event in self.store.events()
                    if event['event_kind'] in {'MANIFEST', 'ATTEMPT', 'RESERVE', 'SETTLE',
                                              'ARM_CLOSED', 'RECOVERY_HOLD', 'UPSTREAM_VALIDATION',
                                              'UPSTREAM_REFUSED'}), default=0)

    def _authorized_response(self, assemble, *, purpose='preview'):
        # This snapshot belongs only to this request and the current lock owner.
        # It is not a permission cache: actual reads and final disclosure both
        # recheck the immutable grant and its current validity, journalling any
        # denial before a value can leave the query.
        with self.store._read_transaction():
            grant = self._grant(purpose)
            watermark = self._data_watermark()
            result = assemble(grant)
            authorize_study(self.store, grant['study_id'], grant, purpose)
            if self._data_watermark() != watermark:
                raise ResearchError('INDEX_STALE', 'authority changed while assembling query snapshot; retry the request')
        # The outer scope revalidates the physical chain before releasing its
        # lock. That I/O may take time, so expiry must be checked afterwards,
        # with no further valid-path I/O before returning this frozen snapshot.
        require_unexpired_disclosure(self.store, grant['study_id'], grant, purpose)
        return result

    def _project_objects(self, kind, grant):
        index = ResearchIndex(self.store)
        index.rebuild()
        items, after = [], ""
        while True:
            page = index.list(kind=kind, after=after, limit=200)
            items.extend(item for item in page["items"] if (
                item["manifest"].get("study_id", item["manifest"].get("spec", {}).get("study_id")) == grant["study_id"]
                and (kind != "comparison" or self._comparison_visible(item["manifest"], grant))))
            if not page["next_cursor"]:
                break
            after = page["next_cursor"]
        return items

    def _comparison_visible(self, package, grant):
        with self.store._read_transaction():
            return self._visible_comparison(package, grant)

    def _visible_comparison(self, package, grant):
        # A package manifest itself contains hashes and artifact IDs. Do not expose it
        # unless every referenced artifact is within the preview grant's scope.
        references = [package[key] for key in ("aggregate_id", "table", "evidence-index")]
        references.extend(package[key] for key in ("computation-receipt", "figure-index") if key in package)
        references.extend(entry["artifact_id"] for entry in package.get("figures", []))
        for artifact_id in references:
            metadata = self.store.manifest("artifact-" + identifier(artifact_id))
            if (metadata["study_id"] != grant["study_id"]
                    or metadata["visibility"] not in grant["visibilities"]
                    or not set(metadata["block_ids"]) <= set(grant["block_ids"])):
                return False
        return True

    def _run_metadata(self, row, spec, events, balances=None):
        manifest = row["manifest"]
        arm = next(a for a in spec["arms"] if a["arm_id"] == manifest["arm_id"])
        cell = manifest["cell"]
        dimensions = comparison_dimensions(cell)
        row["selectors"] = {"model": arm["model_family_id"], "version": digest(spec),
                            "trainer": arm.get("trainer_id", spec.get("trainer_id")),
                            "predictor": cell.get("plugin_id"), "seed": cell["seed"],
                            "horizon": dimensions.get("horizon")}
        row["comparison_dimensions"] = dimensions
        sources = {}
        for event in events:
            payload = event["payload"]
            if (event["event_kind"] in {"RESERVE", "SETTLE"} and
                    payload.get("study_id") == spec["study_id"] and payload.get("run_id") == manifest["run_id"]):
                basis = ("reservation" if not payload["settled"] else "measured-monotonic" if
                         payload["monotonic_elapsed_ms"] is not None else "unknown-conservative-reservation")
                sources[payload["reservation_id"]] = {**payload, "event_hash": event["hash"],
                    "event_sequence": event["sequence"], "cost_basis": basis}
        balance = BudgetLedger(self.store).balance(arm["arm_id"]) if balances is None else balances[arm["arm_id"]]
        row["budget"] = {**balance, "unit": "slot-ms",
                         "limit_ms": 86400000, "scope": "cumulative-arm-all-attempts",
                         "sources": list(sources.values())}
        return row

    def list(self, kind, *, arm_id=None, state=None, model=None, version=None, horizon=None,
             seed=None, trainer=None, predictor=None, limit=50, cursor=None):
        if kind not in {"study", "run", "comparison"} or not 1 <= limit <= 200:
            raise ResearchError("CONTRACT_MISMATCH", "invalid query kind or page limit")
        return self._authorized_response(lambda grant: self._list(kind, grant,
            arm_id=arm_id, state=state, model=model, version=version, horizon=horizon,
            seed=seed, trainer=trainer, predictor=predictor, limit=limit, cursor=cursor))

    def _list(self, kind, grant, *, arm_id, state, model, version, horizon,
              seed, trainer, predictor, limit, cursor):
        snapshot_start = self._data_watermark()
        filters = {"model": model, "version": version, "horizon": horizon, "seed": seed,
                   "trainer": trainer, "predictor": predictor}
        selector = digest([kind, grant["study_id"], arm_id, state, filters])
        rows = self._objects(kind, grant)
        attempts = self.store.attempts()
        for row in rows:
            if kind == "run":
                history = [a for a in attempts.values() if a["run_id"] == row["manifest"]["run_id"]]
                row["attempts"] = history
                row["state"] = history[-1]["state"] if history else "REGISTERED"
        if kind == "run":
            spec = self.store.manifest("study-" + grant["study_id"])["spec"]
            present = {row["manifest"]["cell_hash"] for row in rows}
            for cell in spec["cells"]:
                if digest(cell) not in present:
                    rows.append({"object_id": "expected-" + digest(cell), "expected_cell": True,
                                 "manifest": {"run_id": None, "study_id": grant["study_id"], "cell_hash": digest(cell),
                                              "cell": cell, "arm_id": cell["arm_id"], "seed": cell["seed"]},
                                 "attempts": [], "state": "MISSING"})
            events = self.store.events()
            # Query-local only: new requests always read current authority. The
            # watermark below rejects concurrent cost/closure/recovery changes.
            balances = {name: BudgetLedger(self.store).balance(name)
                        for name in dict.fromkeys(row['manifest']['arm_id'] for row in rows)}
            rows = [self._run_metadata(row, spec, events, balances) for row in rows]
            rows = [row for row in rows if all(selected is None or
                    (row["selectors"][key] is not None and str(row["selectors"][key]) == str(selected))
                    for key, selected in filters.items())]
        rows = [r for r in rows if (arm_id is None or r["manifest"].get("arm_id") == arm_id)
                and (state is None or r.get("state") == state)]
        # Disclosure logging must not invalidate its own continuation cursor.
        watermark = self._data_watermark()
        if watermark != snapshot_start:
            raise ResearchError("INDEX_STALE", "authority changed while assembling query snapshot; retry the request")
        after = ""
        if cursor:
            try:
                saved = json.loads(base64.urlsafe_b64decode(cursor.encode()))
            except (ValueError, TypeError) as exc:
                raise ResearchError("CURSOR_STALE", "invalid cursor") from exc
            if (not isinstance(saved, dict) or saved.get("selector") != selector
                    or saved.get("watermark") != watermark or not isinstance(saved.get("after"), str)):
                raise ResearchError("CURSOR_STALE", "query version or filters changed")
            after = saved["after"]
        rows = sorted((r for r in rows if r["object_id"] > after), key=lambda r: r["object_id"])
        page = rows[:limit]
        next_cursor = None
        if len(rows) > limit:
            next_cursor = base64.urlsafe_b64encode(json.dumps({"selector": selector, "watermark": watermark,
                                                             "after": page[-1]["object_id"]}).encode()).decode()
        result = {"items": page, "next_cursor": next_cursor, "watermark": watermark, "schema_version": self.store.SCHEMA}
        if len(json.dumps(result).encode()) > MAX_RESPONSE:
            raise ResearchError("TOO_LARGE", "query response exceeds 2 MiB; lower page limit")
        return result

    def upstream(self, *, limit=50, cursor=None):
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ResearchError("CONTRACT_MISMATCH", "invalid upstream page limit")
        return self._authorized_response(lambda grant: self._upstream(grant, limit, cursor))

    def _upstream(self, grant, limit, cursor):
        from .research_upstream_view import recorded_cells
        spec = self.store.manifest("study-" + grant["study_id"])["spec"]
        watermark = self._data_watermark()
        spec_hash = digest(spec)
        selector = digest({"kind": "upstream", "study_id": spec["study_id"], "spec_hash": spec_hash})
        cells = sorted(spec["cells"], key=digest)
        after = ""
        if cursor:
            try:
                saved = json.loads(base64.urlsafe_b64decode(cursor.encode()))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ResearchError("CURSOR_STALE", "invalid upstream cursor") from exc
            if (not isinstance(saved, dict) or saved.get("selector") != selector
                    or saved.get("watermark") != watermark or saved.get("after") not in {digest(cell) for cell in cells}):
                raise ResearchError("CURSOR_STALE", "upstream facts or cursor scope changed")
            after = saved["after"]
        remaining = [cell for cell in cells if digest(cell) > after]
        page = remaining[:limit]
        rows = recorded_cells(self.store, spec, self.store.events(), page, spec_hash=spec_hash)
        next_cursor = None
        if len(remaining) > limit:
            next_cursor = base64.urlsafe_b64encode(json.dumps({"selector": selector, "watermark": watermark,
                "after": digest(page[-1])}).encode()).decode()
        settings = spec.get("admission", {})
        result = {"schema_version": "pirc25-upstream-view-v1", "study_id": spec["study_id"],
                  "spec_hash": spec_hash, "snapshot_hash": settings.get("upstream_snapshot_hash"),
                  "acceptance_catalog_hash": settings.get("upstream_acceptance_hash"), "data_authorization": "none",
                  "items": rows, "watermark": watermark, "next_cursor": next_cursor}
        if len(json.dumps(result).encode()) > MAX_RESPONSE:
            raise ResearchError("TOO_LARGE", "upstream view exceeds 2 MiB; lower page limit")
        return result

    def run(self, run_id):
        return self._authorized_response(lambda grant: self._run(run_id, grant))

    def data_access(self, *, limit=50, cursor=None):
        from .research_data_view import DataAccessView, exposure_watermark
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ResearchError("CONTRACT_MISMATCH", "invalid data access page limit")
        views = []
        def assemble(grant):
            spec = self.store.manifest("study-" + grant["study_id"])["spec"]
            watermark = exposure_watermark(self.store)
            selector = digest({"kind": "data-access", "study_id": spec["study_id"], "spec_hash": digest(spec)})
            cells = sorted(spec["cells"], key=digest)
            after = ""
            if cursor:
                try:
                    saved = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ResearchError("CURSOR_STALE", "invalid data access cursor") from exc
                if (not isinstance(saved, dict) or saved.get("selector") != selector
                        or saved.get("watermark") != watermark or saved.get("after") not in {digest(c) for c in cells}):
                    raise ResearchError("CURSOR_STALE", "data access scope or exposure changed")
                after = saved["after"]
            remaining = [c for c in cells if digest(c) > after]
            projection = DataAccessView(self.store, spec, self.store.events(), grant)
            views.append(projection)
            rows = projection.rows(remaining[:limit])
            self.store._read_completion(projection.verify_authority, projection.verify_expiry)
            next_cursor = None
            if len(remaining) > limit:
                next_cursor = base64.urlsafe_b64encode(json.dumps({"selector": selector, "watermark": watermark,
                    "after": rows[-1]["cell_id"]}).encode()).decode()
            if exposure_watermark(self.store) != watermark:
                raise ResearchError("INDEX_STALE", "exposure changed while assembling data access view")
            result = {"schema_version": "pirc25-data-access-view-v1", "study_id": spec["study_id"],
                "spec_hash": digest(spec), "permission_decision": "not-provided", "items": rows,
                "watermark": watermark, "next_cursor": next_cursor}
            if len(json.dumps(result).encode()) > MAX_RESPONSE:
                raise ResearchError("TOO_LARGE", "data access view exceeds 2 MiB; lower page limit")
            return result
        result = self._authorized_response(assemble)
        # No valid-path I/O may follow this final data-grant clock guard. The
        # preview grant and the data grant are different immutable authorities.
        for projection in views:
            projection.verify_expiry()
        return result

    def _run(self, run_id, grant):
        value = self.store.manifest("run-" + identifier(run_id))
        if value["study_id"] != grant["study_id"]:
            raise ResearchError("UNAUTHORIZED_DATA", "run is outside session scope")
        history = [a for a in self.store.attempts().values() if a["run_id"] == run_id]
        spec = self.store.manifest("study-" + grant["study_id"])["spec"]
        events = self.store.events()
        row = self._run_metadata({"manifest": value}, spec, events)
        attempt_ids = {a["attempt_id"] for a in history}
        checkpoints = [event["payload"] for event in events if event["event_kind"] == "CHECKPOINT"
                       and event["payload"].get("attempt_id") in attempt_ids]
        provenance = {key: spec[key] for key in ("schema_version", "code_hash", "data_hash", "protocol_hash", "feature_hash", "selection_hash")}
        paper_evidence = [entry["manifest"] for entry in self._objects("comparison", grant)]
        return {"run": value, "attempts": history, "schema_version": self.store.SCHEMA,
                "provenance": provenance, "checkpoints": checkpoints, "paper_evidence": paper_evidence,
                "selectors": row["selectors"], "comparison_dimensions": row["comparison_dimensions"], "budget": row["budget"]}

    def artifact(self, artifact_id, *, export=False):
        return self._authorized_response(lambda grant: self._artifact(artifact_id, grant, export=export),
                                         purpose='export' if export else 'preview')

    def _artifact(self, artifact_id, grant, *, export=False):
        # One comparison is one study disclosure, not three repeated whole-chain
        # study checks. Every actual read below still revalidates the immutable
        # grant (including expiry) and journals its own exposure/read events.
        metadata = self.store.manifest("artifact-" + identifier(artifact_id))
        svg = metadata["media_type"] == "image/svg+xml" and metadata["role"] in {"comparison-figure", "case-figure"}
        if metadata["media_type"] not in {"application/json", "text/csv"} and not svg:
            raise ResearchError("UNAUTHORIZED_DATA", "artifact media type is not safe for this read service")
        if metadata["study_id"] != grant["study_id"] or metadata["size_bytes"] > MAX_RESPONSE:
            raise ResearchError("UNAUTHORIZED_DATA" if metadata["study_id"] != grant["study_id"] else "TOO_LARGE",
                                "artifact outside scope or larger than 2 MiB")
        content = self.store.read_artifact(artifact_id, purpose="export" if export else "preview", authorization=grant)
        if svg:
            from application.research_figures import validate_figure_svg
            validate_figure_svg(content, expected_kind='case' if metadata['role'] == 'case-figure' else 'comparison')
        if metadata["media_type"] == "application/json" and not export:
            value = json.loads(content)
            samples = value.get("forecast", {}).get("samples") if isinstance(value, dict) else None
            if samples is not None and (len(samples) > 64 or any(len(trajectory) > 512 for trajectory in samples)):
                raise ResearchError("TOO_LARGE", "preview exceeds 64 trajectories or 512 points")
        return content, metadata["media_type"]

    def comparison(self, aggregate_hash):
        return self._authorized_response(lambda grant: self._comparison(aggregate_hash, grant))

    def _comparison(self, aggregate_hash, grant):
        package = self.store.manifest("comparison-" + identifier(aggregate_hash))
        if package["study_id"] != grant["study_id"] or not self._comparison_visible(package, grant):
            raise ResearchError("UNAUTHORIZED_DATA", "comparison outside session scope")
        content, _ = self._artifact(package["aggregate_id"], grant)
        result = {"aggregate": json.loads(content), "package": package}
        if "computation-receipt" in package:
            receipt, _ = self._artifact(package["computation-receipt"], grant)
            result["computation_receipt"] = json.loads(receipt)
        if "figure-index" in package:
            index, _ = self._artifact(package["figure-index"], grant)
            result["figure_index"] = json.loads(index)
        return result

    def case(self, artifact_id):
        from application.research_cases import case_view
        return self._authorized_response(lambda grant: case_view(self.store, artifact_id, grant))

    def result_manifest(self, artifact_id):
        return self._authorized_response(lambda grant: self._result_manifest(artifact_id, grant), purpose='export')

    def _result_manifest(self, artifact_id, grant):
        content, media = self._artifact(artifact_id, grant, export=True)
        metadata = self.store.manifest("artifact-" + identifier(artifact_id))
        value = json.loads(content) if media == "application/json" else {}
        if not isinstance(value, dict):
            raise ResearchError("CONTRACT_MISMATCH", "result manifest requires an object artifact")
        keys = ("schema_version", "spec_hash", "cell_hash", "protocol_hash", "input_hash", "output_hash",
                "admission_hash", "aggregate_hash", "source_bundle_hash", "code_hash", "data_hash")
        return {"schema_version": "pirc25-result-manifest-v1", "artifact": metadata,
                "bindings": {key: value[key] for key in keys if key in value}}
