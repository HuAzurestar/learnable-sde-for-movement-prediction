"""Bounded, hash-bound real method preparation; NO fitting or forecasting.

The full release is used only for split identities and global adaptation roles.
Only explicitly selected, previously qualified train/validation samples can be
materialized. Region comes from the bound trajectory's city/region metadata;
coordinates/times come from exact condition/snapshot source-index joins. Source
directories called 'eval' never override the frozen PIRC20 split assignment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import duckdb
import numpy as np
import pyarrow.parquet as pq

from data.pirc20 import load_pirc20_cohort
from .method_inputs import (FRAME_POLICY, SOLAR_POLICY, assign_development_samples,
                            bind_method_prefix, development_training_segment,
                            resample_training_segment)

VERSION = "pirc17-bounded-method-development-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bound_file(root, relative, expected, max_bytes=None):
    root = Path(root).resolve()
    candidate = root / relative
    path = candidate.resolve()
    if not path.is_relative_to(root) or candidate.is_symlink() or not path.is_file():
        raise ValueError("bound source path missing, symlinked or outside root")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ValueError("source file exceeds preparation byte cap")
    if _hash(path) != expected:
        raise ValueError("bound source hash mismatch")
    return path


@dataclass(frozen=True)
class PreparationBudget:
    max_samples: int
    max_total_points: int
    max_file_bytes: int
    max_file_rows: int
    wall_seconds: float

    def __post_init__(self):
        for value in (self.max_samples, self.max_total_points, self.max_file_bytes, self.max_file_rows):
            if type(value) is not int or value <= 0:
                raise ValueError("explicit positive integer preparation caps required")
        if type(self.wall_seconds) not in (int, float) or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("positive finite preparation wall cap required")


def _deadline(started, budget):
    if time.perf_counter() - started >= budget.wall_seconds:
        raise TimeoutError("preparation cooperative wall cap reached; no automatic retry")


def load_source_regions(trajectory_path, expected_sha256, assignments):
    """Read ONLY file/city/region columns for already admitted development files.

    Whole-file byte hashing is an integrity operation, not final-eval label
    decoding. No position, velocity, timestamp or outcome column is requested.
    Ambiguous per-file region metadata fails instead of inventing task labels.
    """
    assignments = tuple(assignments)
    if not assignments or any(a.sample.split not in {"train", "validation"}
                              or a.method_role not in {"train", "adapt", "validation"} for a in assignments):
        raise ValueError("region loading requires admitted development assignments")
    path = Path(trajectory_path)
    if path.is_symlink() or not path.is_file() or _hash(path) != expected_sha256:
        raise ValueError("original trajectory source hash mismatch")
    files = sorted({a.sample.file_id for a in assignments})
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='512MB'")
        db.execute("CREATE TABLE wanted(file_id VARCHAR)")
        db.executemany("INSERT INTO wanted VALUES (?)", [(f,) for f in files])
        rows = db.execute("""SELECT cast(t.file_id AS VARCHAR), cast(t.city AS VARCHAR),
            cast(t.region AS VARCHAR) FROM read_parquet(?) t
            JOIN wanted w ON cast(t.file_id AS VARCHAR)=w.file_id
            GROUP BY 1,2,3 ORDER BY 1,2,3""", [str(path)]).fetchall()
    result = {}
    for file_id, city, region in rows:
        if file_id in result:
            raise ValueError("ambiguous source city/region within selected file")
        city = str(city or "").strip()
        value = city if city and city.lower() != "nan" else str(region or "unknown")
        if not value.strip() or value.strip().lower() in {"nan", "unknown"}:
            raise ValueError("missing real source region; no fabricated Reptile task")
        result[file_id] = value
    if set(result) != set(files):
        raise ValueError("selected file absent from bound region source")
    return result, {"trajectory_sha256": expected_sha256,
                    "columns_read": ["file_id", "city", "region"],
                    "mapping_sha256": _digest(result), "selected_files": len(files)}


@dataclass(frozen=True)
class MethodDevelopment:
    prefixes: tuple
    segments: tuple
    identity: dict

    def resample(self, interval_seconds):
        """No fit: return prepared role tuples and complete interval accounting."""
        roles = {r: [] for r in ("train", "adapt", "validation")}
        reports = []
        for prefix, segment in zip(self.prefixes, self.segments):
            sampled, row = resample_training_segment(prefix, segment, interval_seconds)
            role = prefix.assignment.method_role
            roles[role].append(sampled)
            reports.append({"sample_id": prefix.assignment.sample.sample_id, "role": role, **row})
        return {k: tuple(v) for k, v in roles.items()}, {
            "input_sha256": self.identity["sha256"], "nominal_interval_seconds": float(interval_seconds),
            "per_sample": reports, "transitions_by_role": {role: sum(r["transition_count"] for r in reports if r["role"] == role)
                                                            for role in roles},
            "short_tails_by_role": {role: sum(r["short_tail_count"] for r in reports if r["role"] == role)
                                     for role in roles},
            "minimum_interval_seconds": min(r["minimum_interval_seconds"] for r in reports),
            "noise_embedding_qualified": False, "formal_training_accepted": False}


def load_method_development(eligibility_path, eligibility_sha256, release, snapshot,
                            condition_root, trajectory_path, *, sample_ids, budget):
    """Load an exact explicit subset; never default to every release sample."""
    started = time.perf_counter()
    if not isinstance(budget, PreparationBudget):
        raise ValueError("explicit preparation budget required")
    release, snapshot = Path(release), Path(snapshot)
    if _hash(eligibility_path) != eligibility_sha256:
        raise ValueError("eligibility hash mismatch")
    eligibility = _json(eligibility_path)
    dataset = _json(release / "dataset.json")
    for name in ("cohort.json", "samples.jsonl", "condition_file_manifest.jsonl"):
        _bound_file(release, name, dataset["artifacts"][name]["sha256"])
    if dataset["artifacts"]["samples.jsonl"]["sha256"] != eligibility["inputs_sha256"]["samples.jsonl"]:
        raise ValueError("eligibility refers to different sample source")
    cohort = load_pirc20_cohort(release / "cohort.json")
    all_samples = tuple(cohort.iter_samples())
    assignments = assign_development_samples(all_samples, sample_ids)
    selected = {a.sample.sample_id: a for a in assignments}
    eligible = {r["sample_id"]: r for r in eligibility["eligibility"]["rows"]}
    if len(eligible) != len(eligibility["eligibility"]["rows"]):
        raise ValueError("duplicate eligibility identity")
    fields = ("split", "file_id", "segment_id", "independent_block_id", "history_start", "history_end", "target_start", "target_end")
    for a in assignments:
        row = eligible.get(a.sample.sample_id)
        if row is None or any(row[k] != getattr(a.sample, k) for k in fields):
            raise ValueError("selected sample is not identically qualified")
    total_points = sum(a.sample.target_end-a.sample.history_start+1 for a in assignments)
    if len(assignments) > budget.max_samples or total_points > budget.max_total_points:
        raise ValueError("selected population exceeds preparation sample/point caps")
    manifest = _json(snapshot / "manifest.json")
    if not dataset["dataset_id"] == cohort.cohort_id == eligibility["dataset_id"] == manifest["dataset_id"]:
        raise ValueError("development dataset identities differ")
    inventory = hashlib.sha256()
    for entry in sorted(manifest["files"], key=lambda e: e["path"]):
        inventory.update(f"{entry['path']}\0{entry['sha256']}\0{entry['row_count']}\n".encode())
    if inventory.hexdigest() != eligibility["snapshot"]["content_inventory_sha256"]:
        raise ValueError("feature content inventory differs from eligibility")
    _deadline(started, budget)
    regions, region_identity = load_source_regions(trajectory_path, dataset["source"]["trajectory"]["sha256"], assignments)
    _deadline(started, budget)
    conditions_list = list(map(json.loads, (release/"condition_file_manifest.jsonl").read_text(encoding="utf-8").splitlines()))
    conditions = {entry["file_id"]: entry for entry in conditions_list}
    features = {(e["split"], e["file_id"]): e for e in manifest["files"]}
    if len(conditions) != len(conditions_list) or len(features) != len(manifest["files"]):
        raise ValueError("duplicate bound file identity")
    groups = defaultdict(list)
    for a in assignments:
        groups[a.sample.split, a.sample.file_id].append(a)
    prepared, sources = {}, {}
    for (split, file_id), group in sorted(groups.items()):
        _deadline(started, budget)
        entry = conditions[file_id]
        cp = _bound_file(condition_root, entry["relative_path"], entry["sha256"], budget.max_file_bytes)
        feature = features[split, file_id]
        fp = _bound_file(snapshot, feature["path"], feature["sha256"], budget.max_file_bytes)
        source = pq.ParquetFile(cp)
        source_meta = pq.ParquetFile(fp)
        if source.metadata.num_rows > budget.max_file_rows or source_meta.metadata.num_rows > budget.max_file_rows:
            raise ValueError("bound file exceeds preparation row cap")
        table = source.read(columns=["file_id", "t", "lon", "lat"])
        if set(table["file_id"].to_pylist()) != {file_id}:
            raise ValueError("condition source file identity differs")
        raw_times = table["t"].to_numpy()
        if raw_times.dtype.kind != "M":
            raise ValueError("condition source must carry exact datetime UTC timestamps")
        epochs = raw_times.astype("datetime64[ns]").astype(np.int64)
        lonlat = np.column_stack((table["lon"].to_numpy(), table["lat"].to_numpy()))
        rows = pq.read_table(fp, columns=["file_id", "segment_id", "point_index", "absolute_epoch_ns", "split", "independent_block_id"],
                             filters=[("segment_id", "in", [a.sample.segment_id for a in group])]).to_pylist()
        by_segment = defaultdict(list)
        for row in rows:
            if row["file_id"] != file_id or row["split"] != split:
                raise ValueError("feature source file/split identity differs")
            by_segment[row["segment_id"]].append(row)
        if set(by_segment) != {a.sample.segment_id for a in group}:
            raise ValueError("selected segments missing/extra in feature source")
        for a in group:
            sample = a.sample
            metadata = sorted(by_segment[sample.segment_id], key=lambda r: r["point_index"])
            if len(metadata) != sample.target_end + 1 or any(r["independent_block_id"] != sample.independent_block_id for r in metadata):
                raise ValueError("feature segment count/block differs")
            metadata = metadata[sample.history_start:sample.target_end+1]
            indexes = np.array([r["point_index"] for r in metadata], dtype=np.int64)
            if np.any(indexes < 0) or np.any(indexes >= len(epochs)) or np.any(np.diff(indexes) <= 0):
                raise ValueError("source point indexes out of range or not strictly ordered")
            times = np.array([r["absolute_epoch_ns"] for r in metadata], dtype=np.int64)
            if not np.array_equal(times, epochs[indexes]):
                raise ValueError("exact source UTC timestamp join differs")
            visible_count = sample.history_end-sample.history_start+1
            prefix = bind_method_prefix(a, times[:visible_count], lonlat[indexes[:visible_count]],
                source_identity=_digest({"condition": entry["sha256"], "feature": feature["sha256"]}))
            segment, _ = development_training_segment(prefix, times, lonlat[indexes],
                region=regions[file_id], region_identity=region_identity["mapping_sha256"])
            prepared[sample.sample_id] = prefix, segment
        sources[f"{split}:{file_id}"] = {"condition_sha256": entry["sha256"], "feature_sha256": feature["sha256"]}
    _deadline(started, budget)
    ordered = [prepared[key] for key in sorted(selected)]
    identity = {"version": VERSION, "dataset_id": dataset["dataset_id"], "dataset_sha256": _hash(release/"dataset.json"),
        "eligibility_sha256": eligibility_sha256,
        "population_identity": assignments[0].population_identity, "sample_ids": sorted(selected),
        "frame_policy": FRAME_POLICY, "solar_policy": SOLAR_POLICY, "source_regions": region_identity,
        "sources": sources, "observed_points": total_points,
        "sample_counts": dict(Counter(a.method_role for a in assignments)),
        "region_counts": {role: len({regions[a.sample.file_id] for a in assignments if a.method_role == role})
                          for role in ("train", "adapt", "validation")},
        "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0}
    identity["sha256"] = _digest(identity)
    return MethodDevelopment(tuple(p for p, _ in ordered), tuple(s for _, s in ordered), identity)
