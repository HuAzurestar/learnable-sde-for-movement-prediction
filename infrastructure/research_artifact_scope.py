"""Source-bound historical block/first-read facts for one physical owner phase.

No manifest bytes, grants, clocks or approvals are retained. The owning store
rebuilds from actual metadata after its final physical prefix validation. New
durable artifact/read events update these facts on the write side, not by
re-enumerating the old global history on every block query.
"""

from bisect import bisect_left
import re


def source_scope_keys(identity):
    """Exact left-side same_source scopes, without pairwise window searches."""
    sha, dataset, source = identity.get("sha256"), identity.get("dataset_id"), identity.get("source_block_id")
    known = isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None
    if known:
        keys = {("hash", sha)}
        if isinstance(dataset, str) and isinstance(source, str):
            keys.add(("source", dataset, source))
        return keys
    if not isinstance(dataset, str):
        raise ValueError("historical source dataset is unresolved")
    return {("dataset", dataset)}


def query_scope_keys(block_id, identity):
    keys = {("block", block_id), ("source", identity["dataset_id"], identity["source_block_id"]),
            ("dataset", identity["dataset_id"])}
    if isinstance(identity.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]):
        keys.add(("hash", identity["sha256"]))
    return keys


class ArtifactExposureFacts:
    def __init__(self, events, manifest_loader, protocol_events=()):
        self._loader = manifest_loader
        self._first_reads = {}
        self._sources = {}
        self._blocks = {}
        self._errors = {}
        self._protocols = {}
        self._dependants = {}
        for event in events:
            artifact_id = event["payload"].get("artifact_id")
            if isinstance(artifact_id, str):
                self._first_reads.setdefault(artifact_id, event["sequence"])
            else:
                self._errors[("malformed", event["sequence"])] = event["sequence"]
        if self._first_reads:
            for event in protocol_events:
                self._load_protocol(event["payload"]["object_id"], initial=True)
        for artifact_id in self._first_reads:
            self._load(artifact_id, initial=True)
        # One initial sort, not a sorted-list insertion for every old window.
        self._keys = sorted(self._blocks)
        self._first_error = min(self._errors.values(), default=None)

    def _add_keys(self, keys, sequence, *, initial=False):
        for key in keys:
            if key not in self._blocks:
                self._blocks[key] = sequence
                if not initial:
                    self._keys.insert(bisect_left(self._keys, key), key)
            else:
                self._blocks[key] = min(sequence, self._blocks[key])

    def _load_protocol(self, object_id, *, initial=False):
        from .research_store import ResearchError
        try:
            value = self._loader(object_id)
            if (not isinstance(value, dict) or value.get("schema_version") != "pirc25-data-protocol-v1"
                    or not isinstance(value.get("study_id"), str) or not isinstance(value.get("blocks"), list)):
                raise ResearchError("CORRUPT_ARTIFACT", "historical artifact source protocol malformed")
            additions = {}
            for block in value["blocks"]:
                if (not isinstance(block, dict)
                        or any(not isinstance(block.get(field), str) or not block[field]
                               for field in ("dataset_id", "release_id", "block_id", "sha256"))
                        or not isinstance(block.get("source_block_id", block["block_id"]), str)):
                    raise ResearchError("CORRUPT_ARTIFACT", "historical artifact source identity unresolved")
                point = (value["study_id"], block["block_id"])
                additions.setdefault(point, set()).update(source_scope_keys({
                    "dataset_id": block["dataset_id"], "sha256": block["sha256"],
                    "source_block_id": block.get("source_block_id", block["block_id"])}))
        except (ResearchError, OSError, TypeError, ValueError, KeyError):
            self._errors[("protocol", object_id)] = min(self._first_reads.values())
            return
        for point, keys in additions.items():
            self._protocols.setdefault(point, set()).update(keys)
            for artifact_id in self._dependants.get(point, ()):
                self._add_keys(keys, self._first_reads[artifact_id], initial=initial)
        self._errors.pop(("protocol", object_id), None)

    def _load(self, artifact_id, *, initial=False):
        from .research_store import ResearchError, digest
        sequence = self._first_reads[artifact_id]
        try:
            value = self._loader("artifact-" + artifact_id)
            blocks = value.get("block_ids") if isinstance(value, dict) else None
            if (not isinstance(blocks, list)
                    or any(not isinstance(block, str) or not block for block in blocks)):
                raise ResearchError("CORRUPT_ARTIFACT", "historical artifact block scope malformed")
            study = value.get("study_id")
            if study is not None and not isinstance(study, str):
                raise ResearchError("CORRUPT_ARTIFACT", "historical artifact study scope malformed")
            source_hash = digest(value)
            previous = self._sources.get(artifact_id)
            if previous is not None and previous[0] != source_hash:
                raise ResearchError("CORRUPT_ARTIFACT", "immutable historical artifact scope changed")
        except (ResearchError, OSError, TypeError, ValueError):
            # Keep uncertainty scoped to its ORIGINAL read time. An unresolved
            # artifact first read after this plan's freeze did not precede it.
            # This is a failed source fact, never an unexposed/allowed result.
            self._errors[artifact_id] = sequence
            return
        keys = {("block", block) for block in blocks}
        for block in blocks:
            point = (study, block)
            self._dependants.setdefault(point, set()).add(artifact_id)
            keys.update(self._protocols.get(point, ()))
        self._add_keys(keys, sequence, initial=initial)
        self._sources[artifact_id] = (source_hash, tuple(blocks))
        self._errors.pop(artifact_id, None)

    def append(self, event):
        """Called only after a real durable event; maintenance is write-side."""
        kind, payload = event["event_kind"], event["payload"]
        if kind in {"READ_STARTED", "READ_COMPLETED", "READ_FAILED"} and isinstance(payload, dict):
            artifact_id = payload.get("artifact_id")
            if not artifact_id:
                return
            if isinstance(artifact_id, str):
                self._first_reads.setdefault(artifact_id, event["sequence"])
                self._load(artifact_id)
            else:
                self._errors[("malformed", event["sequence"])] = event["sequence"]
        elif kind == "MANIFEST" and isinstance(payload, dict):
            object_id = payload.get("object_id")
            if isinstance(object_id, str) and object_id.startswith("protocol-") and self._first_reads:
                self._load_protocol(object_id)
                self._first_error = min(self._errors.values(), default=None)
                return
            if not isinstance(object_id, str) or not object_id.startswith("artifact-"):
                return
            artifact_id = object_id[len("artifact-"):]
            if artifact_id not in self._first_reads:
                return  # Publication alone is not exposure.
            self._load(artifact_id)
        else:
            return
        self._first_error = min(self._errors.values(), default=None)

    def before(self, block_id, identity, sequence):
        if self._first_error is not None and self._first_error < sequence:
            from .research_store import ResearchError
            raise ResearchError("UNAUTHORIZED_DATA", "historical artifact exposure scope unresolved")
        for key in query_scope_keys(block_id, identity):
            position = bisect_left(self._keys, key)
            if (position < len(self._keys) and self._keys[position] == key
                    and self._blocks[key] < sequence):
                return True
        return False
