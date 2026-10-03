"""Ephemeral keyed projection of an already physically verified event prefix.

No file contents, approvals or clocks live here. The owning store must discard
this projection at each physical boundary, on uncertain writes and recovery.
Key lookup is logarithmic; matching-row output and projection construction have
their own costs. This is not an authoritative or persistent replacement log.
"""

from bisect import bisect_left
import re


_READS = {"READ_STARTED", "READ_COMPLETED", "READ_FAILED"}


def _keys(event):
    kind, payload = event["event_kind"], event["payload"]
    yield ("event", event["event_id"])
    if not isinstance(kind, str):
        return
    yield ("kind", kind)
    if not isinstance(payload, dict):
        if kind in _READS:
            yield ("read-malformed",)
        elif kind == "MANIFEST":
            yield ("manifest-malformed",)
        return  # Unrelated legacy events are not manifest/read authority.
    if kind == "MANIFEST":
        if isinstance(payload.get("object_id"), str):
            yield ("manifest", payload["object_id"])
        else:
            yield ("manifest-malformed",)
    if kind in _READS:
        sha, dataset = payload.get("sha256"), payload.get("dataset_id")
        known_hash = isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None
        if known_hash:
            yield ("read-hash", sha)
        if isinstance(dataset, str):
            source = payload.get("source_block_id", payload.get("block_id"))
            if isinstance(source, str):
                yield ("read-source", dataset, source)
            if not known_hash:
                yield ("read-unknown", dataset)
        if payload.get("artifact_id"):
            yield ("read-artifact",)


class VerifiedEventLookup:
    def __init__(self, events):
        groups = {}
        for event in events:
            for key in _keys(event):
                groups.setdefault(key, []).append(event)
        self._keys = sorted(groups)
        self._rows = [groups[key] for key in self._keys]
        self.size = len(events)
        self.last_hash = events[-1]["hash"] if events else "0" * 64

    def _group(self, key):
        position = bisect_left(self._keys, key)
        return self._rows[position] if position < len(self._keys) and self._keys[position] == key else []

    def get(self, key, *, last=False):
        if key[0] == "manifest":
            self._require_manifest_shape()
        rows = self._group(key)
        return rows[-1 if last else 0] if rows else None

    def _require_manifest_shape(self):
        if self._group(("manifest-malformed",)):
            from .research_store import ResearchError
            raise ResearchError("CORRUPT_ARTIFACT", "authoritative manifest event is malformed")

    def matching(self, key, *, before=None):
        rows = self._group(key)
        if before is None:
            return rows[:]
        low, high = 0, len(rows)
        while low < high:
            middle = (low + high) // 2
            if rows[middle]["sequence"] < before:
                low = middle + 1
            else:
                high = middle
        return rows[:low]

    def manifests(self, prefix):
        self._require_manifest_shape()
        start = bisect_left(self._keys, ("manifest", prefix))
        stop = bisect_left(self._keys, ("manifest", prefix + "\uffff"))
        rows = [event for group in self._rows[start:stop] for event in group]
        return sorted(rows, key=lambda event: event["sequence"])

    def prior_reads(self, identity, before):
        # Preserve same_source's content alias, same source-block and unknown
        # legacy-dataset rules, plus legacy artifact disclosures. The caller
        # still applies the original predicate and rehashes actual manifests.
        if self.get(("read-malformed",)) is not None:
            # Unknown scope cannot be silently discarded as unrelated, even
            # after freeze. A damaged history is never proof of an untouched
            # block. The actual data gate records a denial before opening it.
            from .research_store import ResearchError
            raise ResearchError("UNAUTHORIZED_DATA", "historical read scope is malformed")
        keys = [("read-hash", identity["sha256"]),
                ("read-source", identity["dataset_id"], identity["source_block_id"]),
                ("read-unknown", identity["dataset_id"]), ("read-artifact",)]
        rows = {event["sequence"]: event for key in keys
                for event in self.matching(key, before=before)}
        return [rows[sequence] for sequence in sorted(rows)]

    def append(self, event):
        # Only called after the actual event file is durably published. Lists
        # preserve sequence order; insertion cost is write-side, not lookup.
        for key in _keys(event):
            position = bisect_left(self._keys, key)
            if position == len(self._keys) or self._keys[position] != key:
                self._keys.insert(position, key)
                self._rows.insert(position, [])
            self._rows[position].append(event)
        self.size += 1
        self.last_hash = event["hash"]
