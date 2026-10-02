"""Authorized read-only result views; all numbers come from frozen artifacts."""

from __future__ import annotations

import base64
import json

from .research_evidence import authorize_study
from .research_budget import BudgetLedger
from .research_dimensions import comparison_dimensions
from infrastructure.research_index import ResearchIndex
from infrastructure.research_store import ResearchError, digest, identifier

MAX_RESPONSE = 2 * 1024 * 1024


class ResearchQuery:
    def __init__(self, store, authorization_id):
        self.store = store
        self.authorization_id = identifier(authorization_id)

    def _grant(self, purpose="preview"):
        grant = self.store.manifest("authorization-" + self.authorization_id)
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
        # A package manifest itself contains hashes and artifact IDs. Do not expose it
        # unless every referenced artifact is within the preview grant's scope.
        for key in ("aggregate_id", "table", "evidence-index", *(["computation-receipt"] if "computation-receipt" in package else [])):
            metadata = self.store.manifest("artifact-" + identifier(package[key]))
            if (metadata["study_id"] != grant["study_id"]
                    or metadata["visibility"] not in grant["visibilities"]
                    or not set(metadata["block_ids"]) <= set(grant["block_ids"])):
                return False
        return True

    def _run_metadata(self, row, spec, events):
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
        row["budget"] = {**BudgetLedger(self.store).balance(arm["arm_id"]), "unit": "slot-ms",
                         "limit_ms": 86400000, "scope": "cumulative-arm-all-attempts",
                         "sources": list(sources.values())}
        return row

    def list(self, kind, *, arm_id=None, state=None, model=None, version=None, horizon=None,
             seed=None, trainer=None, predictor=None, limit=50, cursor=None):
        if kind not in {"study", "run", "comparison"} or not 1 <= limit <= 200:
            raise ResearchError("CONTRACT_MISMATCH", "invalid query kind or page limit")
        grant = self._grant()
        def data_watermark():
            return max((e["sequence"] for e in self.store.events()
                        if e["event_kind"] in {"MANIFEST", "ATTEMPT", "RESERVE", "SETTLE", "ARM_CLOSED"}), default=0)

        snapshot_start = data_watermark()
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
            rows = [self._run_metadata(row, spec, events) for row in rows]
            rows = [row for row in rows if all(selected is None or
                    (row["selectors"][key] is not None and str(row["selectors"][key]) == str(selected))
                    for key, selected in filters.items())]
        rows = [r for r in rows if (arm_id is None or r["manifest"].get("arm_id") == arm_id)
                and (state is None or r.get("state") == state)]
        # Disclosure logging must not invalidate its own continuation cursor.
        watermark = data_watermark()
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

    def run(self, run_id):
        grant = self._grant()
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
        grant = self._grant("export" if export else "preview")
        metadata = self.store.manifest("artifact-" + identifier(artifact_id))
        if metadata["media_type"] not in {"application/json", "text/csv"}:
            raise ResearchError("UNAUTHORIZED_DATA", "artifact media type is not safe for this read service")
        if metadata["study_id"] != grant["study_id"] or metadata["size_bytes"] > MAX_RESPONSE:
            raise ResearchError("UNAUTHORIZED_DATA" if metadata["study_id"] != grant["study_id"] else "TOO_LARGE",
                                "artifact outside scope or larger than 2 MiB")
        content = self.store.read_artifact(artifact_id, purpose="export" if export else "preview", authorization=grant)
        if metadata["media_type"] == "application/json" and not export:
            value = json.loads(content)
            samples = value.get("forecast", {}).get("samples") if isinstance(value, dict) else None
            if samples is not None and (len(samples) > 64 or any(len(trajectory) > 512 for trajectory in samples)):
                raise ResearchError("TOO_LARGE", "preview exceeds 64 trajectories or 512 points")
        return content, metadata["media_type"]

    def comparison(self, aggregate_hash):
        grant = self._grant()
        package = self.store.manifest("comparison-" + identifier(aggregate_hash))
        if package["study_id"] != grant["study_id"] or not self._comparison_visible(package, grant):
            raise ResearchError("UNAUTHORIZED_DATA", "comparison outside session scope")
        content, _ = self.artifact(package["aggregate_id"])
        result = {"aggregate": json.loads(content), "package": package}
        if "computation-receipt" in package:
            receipt, _ = self.artifact(package["computation-receipt"])
            result["computation_receipt"] = json.loads(receipt)
        return result

    def result_manifest(self, artifact_id):
        content, media = self.artifact(artifact_id, export=True)
        metadata = self.store.manifest("artifact-" + identifier(artifact_id))
        value = json.loads(content) if media == "application/json" else {}
        if not isinstance(value, dict):
            raise ResearchError("CONTRACT_MISMATCH", "result manifest requires an object artifact")
        keys = ("schema_version", "spec_hash", "cell_hash", "protocol_hash", "input_hash", "output_hash",
                "admission_hash", "aggregate_hash", "source_bundle_hash", "code_hash", "data_hash")
        return {"schema_version": "pirc25-result-manifest-v1", "artifact": metadata,
                "bindings": {key: value[key] for key in keys if key in value}}
