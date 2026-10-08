"""Exclusive, per-run commit records for a separately versioned resume attempt.

Storage integrity only: callers must validate the scientific contract and prove
the previous process is closed before inheriting its records. Every new process
uses a NEW directory. Never reopen a crashed writer or overwrite its orphans.
File fsync and no-replace publication protect process-interruption boundaries;
they are not a promise of zero loss across arbitrary hardware/power failures.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .precision_check import KEYS, load_particle_evidence
from .qualification import _hash
from .seed_resume import read_closed

VERSION = "pirc17-resume-journal-v1"
MANIFEST_FIELDS = {"schema_version", "workloads", "inherited_count", "ancestry", "contract",
                   "certified", "formal_training_accepted", "final_eval_label_prediction_metric_reads"}
RECORD_FIELDS = {"schema_version", "index", "manifest_sha256", "previous_record_sha256", "row"}


def _encode(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))+"\n").encode("utf-8")


def _publish(path, writer):
    """Publish a fully fsynced file with atomic no-overwrite hard-link creation.

    Temp and target are in the same directory/filesystem. os.replace/rename is
    deliberately not used: POSIX rename could clobber another attempt's record.
    If hard links are unsupported we fail closed, with no unsafe fallback.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="."+path.name+".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            writer(target)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        # Only this invocation's exact temporary file, never a glob or orphan.
        Path(temporary).unlink(missing_ok=True)


def _publish_json(path, value):
    encoded = _encode(value)  # Reject unserializable/nonfinite input before IO.
    _publish(path, lambda target: target.write(encoded))
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(manifest):
    if (set(manifest) != MANIFEST_FIELDS or manifest["schema_version"] != VERSION
            or any(manifest[key] is not False for key in ("certified", "formal_training_accepted"))
            or type(manifest["final_eval_label_prediction_metric_reads"]) is not int
            or manifest["final_eval_label_prediction_metric_reads"] != 0
            or not isinstance(manifest["ancestry"], dict) or not manifest["ancestry"]
            or not isinstance(manifest["contract"], dict) or not manifest["contract"]):
        raise ValueError("explicit unsealed journal ancestry and execution contract required")
    workloads = manifest["workloads"]
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("full ordered nonempty workload denominator required")
    keys = []
    for row in workloads:
        if (not isinstance(row, dict) or set(row) != set(KEYS)
                or any(not isinstance(row[key], str) or not row[key] for key in ("sample_id", "configuration"))
                or type(row["seed"]) is not int or row["seed"] < 0
                or type(row["particles"]) is not int or row["particles"] < 3
                or isinstance(row["max_step_seconds"], bool)
                or not isinstance(row["max_step_seconds"], (int, float))
                or not 0 < row["max_step_seconds"] < float("inf")):
            raise ValueError("invalid journal workload key")
        keys.append(tuple(row[key] for key in KEYS))
    inherited = manifest["inherited_count"]
    if len(set(keys)) != len(keys) or type(inherited) is not int or not 0 <= inherited <= len(keys):
        raise ValueError("unique workload keys and bounded inherited prefix count required")
    return keys


def _validate_row(row, workload):
    if (not isinstance(row, dict) or row.get("type") != "run"
            or {key: row.get(key) for key in KEYS} != workload
            or type(row.get("seed")) is not int or type(row.get("particles")) is not int
            or isinstance(row.get("max_step_seconds"), bool)
            or row.get("status") not in {"success", "failure"}):
        raise ValueError("record must match the exact next registered workload")
    if row["status"] == "failure" and (not isinstance(row.get("error_type"), str) or not row["error_type"]
                                      or not isinstance(row.get("error_message"), str)):
        raise ValueError("failed work must retain its error, never become missing work")


@dataclass
class Snapshot:
    manifest: dict
    records: list
    tip: dict
    uncommitted_files: list

    @property
    def failure_count(self):
        return sum(record["row"]["status"] == "failure" for record in self.records)


def read(directory, manifest_sha256, *, expected_tip=None):
    """Read a strict committed prefix; use a pinned tip when inheriting it.

    Initial inventory is not proof that a writer has stopped. The supervisor /
    admission layer must close, pin, and bind this tip into the NEXT manifest.
    A pinned tip detects deletion of even the last record, not just chain gaps.
    """
    directory = Path(directory).resolve()
    manifest = read_closed(directory/"manifest.json", manifest_sha256)
    _validate_manifest(manifest)
    index = manifest["inherited_count"]
    records, committed = [], {directory/"manifest.json"}
    bound = {directory/"manifest.json": manifest_sha256}
    previous = manifest_sha256
    for path in sorted((directory/"records").glob("*.json")):
        if index >= len(manifest["workloads"]) or path.name != f"{index:06d}.json":
            raise ValueError("journal has a missing, reordered or out-of-range commit")
        digest = _hash(path)
        record = read_closed(path, digest)
        if (set(record) != RECORD_FIELDS or record["schema_version"] != VERSION+"-record"
                or type(record["index"]) is not int or record["index"] != index
                or record["manifest_sha256"] != manifest_sha256
                or record["previous_record_sha256"] != previous):
            raise ValueError("journal record breaks its manifest/index/hash chain")
        row = record["row"]
        _validate_row(row, manifest["workloads"][index])
        if row["status"] == "success":
            expected = f"particles/{index:06d}.npz"
            if row.get("particle_artifact", {}).get("path") != expected:
                raise ValueError("journal particle path does not match its committed index")
            load_particle_evidence(row, directory)
            committed.add(directory/expected)
            bound[directory/expected] = row["particle_artifact"]["sha256"]
        elif "particle_artifact" in row:
            raise ValueError("failure record cannot claim committed successful particles")
        committed.add(path)
        bound[path] = digest
        records.append(record)
        previous = digest
        index += 1
    tip = {"manifest_sha256": manifest_sha256, "record_count": len(records), "last_record_sha256": previous}
    if expected_tip is not None and tip != expected_tip:
        raise ValueError("journal no longer matches its pinned closed-attempt tip")
    uncommitted = [{"path": path.relative_to(directory).as_posix(), "sha256": _hash(path)}
        for path in sorted(directory.rglob("*")) if path.is_file() and path not in committed]
    if any(_hash(path) != digest for path, digest in bound.items()):
        raise ValueError("committed journal evidence changed during the snapshot read")
    return Snapshot(manifest, records, tip, uncommitted)


class Writer:
    """One held writer per NEW attempt directory. No reopening/repair in place."""

    def __init__(self, directory, *, workloads, inherited_count, ancestry, contract):
        manifest = {"schema_version": VERSION, "workloads": deepcopy(workloads), "inherited_count": inherited_count,
            "ancestry": deepcopy(ancestry), "contract": deepcopy(contract), "certified": False,
            "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0}
        _validate_manifest(manifest)
        _encode(manifest)
        self.directory = Path(directory).resolve()
        self.directory.mkdir()  # Atomic reservation; do not silently reuse parents.
        (self.directory/"records").mkdir()
        (self.directory/"particles").mkdir()
        self.manifest_sha256 = _publish_json(self.directory/"manifest.json", manifest)
        self._manifest = manifest
        self._index = inherited_count
        self._previous = self.manifest_sha256
        self._failed = False

    @property
    def tip(self):
        return {"manifest_sha256": self.manifest_sha256,
                "record_count": self._index-self._manifest["inherited_count"], "last_record_sha256": self._previous}

    def append(self, row, arrays=None):
        if self._failed:
            raise RuntimeError("writer publication failed; close this attempt and use a new directory")
        index = self._index
        if index >= len(self._manifest["workloads"]):
            raise ValueError("original workload denominator already recorded")
        _validate_row(row, self._manifest["workloads"][index])
        if "particle_artifact" in row:
            raise ValueError("new run cannot import a particle path through the writer")
        row = deepcopy(row)
        _encode(row)
        if row["status"] == "success":
            if not isinstance(arrays, tuple) or len(arrays) != 3:
                raise ValueError("successful checkpoint requires positions, targets and elapsed times")
            x, y, t = arrays
            if (any(not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number)
                    or np.iscomplexobj(value) or not np.isfinite(value).all() for value in arrays)
                    or t.ndim != 1 or x.shape != (row["particles"], len(t), 2) or y.shape != (len(t), 2)
                    or not np.array_equal(t, row.get("actual_horizons_seconds"))):
                raise ValueError("invalid checkpoint particle dimensions, times or values")
        elif arrays is not None:
            raise ValueError("failed run must not publish successful particle evidence")
        try:
            if row["status"] == "success":
                path = self.directory/"particles"/f"{index:06d}.npz"
                _publish(path, lambda target: np.savez_compressed(target, positions_m=x,
                    target_positions_m=y, elapsed_seconds=t))
                row["particle_artifact"] = {"path": path.relative_to(self.directory).as_posix(), "sha256": _hash(path)}
            record = {"schema_version": VERSION+"-record", "index": index, "manifest_sha256": self.manifest_sha256,
                      "previous_record_sha256": self._previous, "row": row}
            digest = _publish_json(self.directory/"records"/f"{index:06d}.json", record)
        except BaseException:
            self._failed = True
            raise
        self._index += 1
        self._previous = digest
        return deepcopy(record)
