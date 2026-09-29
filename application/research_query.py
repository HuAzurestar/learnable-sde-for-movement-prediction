"""Authorized read-only result views; all numbers come from frozen artifacts."""

from __future__ import annotations

import base64
import json

from .research_evidence import authorize_study
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
        return grant

    def _objects(self, kind, grant):
        index = ResearchIndex(self.store)
        index.rebuild()
        items, after = [], ""
        while True:
            page = index.list(kind=kind, after=after, limit=200)
            items.extend(item for item in page["items"] if (
                item["manifest"].get("study_id", item["manifest"].get("spec", {}).get("study_id")) == grant["study_id"]))
            if not page["next_cursor"]:
                break
            after = page["next_cursor"]
        return items

    def list(self, kind, *, arm_id=None, state=None, limit=50, cursor=None):
        if kind not in {"study", "run", "comparison"} or not 1 <= limit <= 200:
            raise ResearchError("CONTRACT_MISMATCH", "invalid query kind or page limit")
        grant = self._grant()
        selector = digest([kind, grant["study_id"], arm_id, state])
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
        rows = [r for r in rows if (arm_id is None or r["manifest"].get("arm_id") == arm_id)
                and (state is None or r.get("state") == state)]
        # Disclosure logging must not invalidate its own continuation cursor.
        watermark = max((e["sequence"] for e in self.store.events() if e["event_kind"] in {"MANIFEST", "ATTEMPT"}), default=0)
        after = ""
        if cursor:
            try:
                saved = json.loads(base64.urlsafe_b64decode(cursor.encode()))
            except (ValueError, TypeError) as exc:
                raise ResearchError("CURSOR_STALE", "invalid cursor") from exc
            if not isinstance(saved, dict) or saved.get("selector") != selector or saved.get("watermark") != watermark:
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
        return {"run": value, "attempts": history, "schema_version": self.store.SCHEMA}

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
        if package["study_id"] != grant["study_id"]:
            raise ResearchError("UNAUTHORIZED_DATA", "comparison outside session scope")
        content, _ = self.artifact(package["aggregate_id"])
        return {"aggregate": json.loads(content), "package": package}
