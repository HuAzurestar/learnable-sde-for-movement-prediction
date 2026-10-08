"""Bounded exact map-query experiment, never a production forecast backend.

The candidate retains decoded geometry within an original immutable cell.
All nearest/tie/raster/source semantics remain the reference's responsibility.
Endpoint query equality is not full-path or numerical qualification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pyarrow as pa
import shapely

from trajectory.linear_materialization import _SEARCH_RADII_M
from trajectory.multicell_terrain import MultiCellRawMapQuery

from .qualification import _hash

PROBE_VERSION = "pirc17-resident-geometry-transport-probe-v2"
MINIMUM_FREE_BYTES = 2 * 1024**3


class ResidentLineIndex:
    """Experiment-owned temporary table; the original cell owns the connection."""

    def __init__(self, reference):
        self.reference = reference
        self.connection = db = reference.connection
        self.projection = reference.projection
        self.offsets = {}
        blobs, texts = [], []
        for family, (_, family_blobs, family_texts) in reference.families.items():
            self.offsets[family] = len(blobs)
            blobs.extend(family_blobs.tolist())
            texts.extend(family_texts.tolist())
        names = ("pirc17_probe_geometries", "pirc17_probe_geometry_input", "pirc17_probe_refs")
        if any(db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?",
                          [name]).fetchone()[0] for name in names):
            raise ValueError("probe names already owned by another object")
        table = pa.table({"line_id": pa.array(range(len(blobs)), type=pa.int64()),
                          "line_wkb": pa.array(blobs, type=pa.binary()),
                          "tie_text": pa.array(texts, type=pa.string())})
        self.geometry_count = len(blobs)
        self.initial_arrow_bytes = table.nbytes
        db.register("pirc17_probe_geometry_input", table)
        try:
            db.execute("""CREATE TEMP TABLE pirc17_probe_geometries AS
                SELECT line_id, ST_GeomFromWKB(line_wkb) line_geom, tie_text
                FROM pirc17_probe_geometry_input""")
        finally:
            db.unregister("pirc17_probe_geometry_input")

    def query(self, lonlat, pids):
        values = np.asarray(lonlat, dtype=float)
        pids = np.asarray(pids, dtype=np.int64)
        if values.shape != (len(pids), 2) or not np.isfinite(values).all():
            raise ValueError("finite lon/lat rows with matching point IDs required")
        if len(set(pids.tolist())) != len(pids):
            raise ValueError("unique point IDs required")
        if not len(values):
            return []
        x, y, lon0, lat0 = self.projection
        projected = shapely.points(values[:, 0]*x-lon0*x, values[:, 1]*y-lat0*y)
        maximum = float(max(_SEARCH_RADII_M))
        candidates = defaultdict(list)
        for family, (tree, _, _) in self.reference.families.items():
            indexes, distances = tree.query_nearest(projected, return_distance=True, all_matches=False)
            admitted = distances <= maximum + 1e-6
            indexes, distances = indexes[:, admitted], distances[admitted]
            if not len(distances):
                continue
            radius = distances + np.maximum(1e-7, np.abs(distances)*1e-12)
            nearby = tree.query(projected[indexes[0]], predicate="dwithin", distance=radius)
            query_rows, line_rows = indexes[0][nearby[0]], nearby[1]
            candidates["pid"].extend(pids[query_rows].tolist())
            candidates["family"].extend([family]*len(query_rows))
            candidates["lon"].extend(values[query_rows, 0].tolist())
            candidates["lat"].extend(values[query_rows, 1].tolist())
            candidates["line_id"].extend((line_rows+self.offsets[family]).tolist())
        if not candidates:
            return []
        db = self.connection
        db.register("pirc17_probe_refs", pa.table(candidates))
        try:
            return db.execute("""
                WITH inputs AS (
                    SELECT refs.*, geometries.line_geom, geometries.tie_text,
                           ST_Point(lon,lat) wgs84_geom,
                           ST_Affine(ST_Point(lon,lat),?,0,0,?,?,?) point_geom
                    FROM pirc17_probe_refs refs
                    JOIN pirc17_probe_geometries geometries USING (line_id)
                ), distances AS (
                    SELECT *, ST_Distance(point_geom,line_geom) metric_distance FROM inputs
                ), ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY pid,family ORDER BY metric_distance,tie_text
                    ) nearest_rank FROM distances WHERE metric_distance <= ?
                ), winners AS (
                    SELECT *, ST_ClosestPoint(line_geom,point_geom) closest_geom
                    FROM ranked WHERE nearest_rank=1
                )
                SELECT pid,family,
                    ST_Distance_Sphere(ST_FlipCoordinates(wgs84_geom),
                        ST_FlipCoordinates(ST_Affine(closest_geom,?,0,0,?,?,?))),
                    CASE WHEN metric_distance>1e-9 THEN
                        (ST_X(closest_geom)-ST_X(point_geom))/metric_distance ELSE 0.0 END,
                    CASE WHEN metric_distance>1e-9 THEN
                        (ST_Y(closest_geom)-ST_Y(point_geom))/metric_distance ELSE 0.0 END
                FROM winners ORDER BY pid,family
            """, [x, y, -lon0*x, -lat0*y, maximum, 1/x, 1/y, lon0, lat0]).fetchall()
        finally:
            db.unregister("pirc17_probe_refs")


class ProbeMap(MultiCellRawMapQuery):
    def __init__(self, *args, **kwargs):
        self.resident_builds = []
        super().__init__(*args, **kwargs)

    def _linear_cell(self, west, south):
        super()._linear_cell(west, south)
        if not isinstance(self.line_index, ResidentLineIndex):
            started = time.perf_counter()
            index = ResidentLineIndex(self.line_index)
            self.line_index = self.geometry_cells[west, south].line_index = index
            self.resident_builds.append({"cell": [west, south],
                "wall_seconds": time.perf_counter()-started,
                "geometry_count": index.geometry_count, "initial_arrow_bytes": index.initial_arrow_bytes})

    @property
    def identity(self):
        original = super().identity
        return {**original, "query_version": PROBE_VERSION,
                "reference_query_version": original["query_version"],
                "scope": "isolated experiment only; not a registered forecast backend"}


class SharedProbeMap(ProbeMap):
    """Two synchronous query modes borrowing one original cell/raster owner.

    The cached cell retains its resident wrapper (and original index). Only
    the parent's borrowed alias changes for a reference call. No query answers
    are retained, and the inherited source checks/eviction remain in force.
    """

    def __init__(self, *args, **kwargs):
        self._active_backend = None
        super().__init__(*args, **kwargs)

    def query_mode(self, backend, lonlat):
        if backend not in ("reference", "candidate"):
            raise ValueError("explicit reference or candidate mode required")
        if self._active_backend is not None:
            raise RuntimeError("shared probe permits one synchronous query at a time")
        self._active_backend = backend
        try:
            return self(lonlat)
        finally:
            self._active_backend = None

    def __call__(self, lonlat):
        if self._active_backend is None:
            raise RuntimeError("use query_mode with an explicit backend")
        return super().__call__(lonlat)

    def _linear_cell(self, west, south):
        if self._active_backend == "candidate":
            super()._linear_cell(west, south)
        elif self._active_backend == "reference":
            MultiCellRawMapQuery._linear_cell(self, west, south)
            if isinstance(self.line_index, ResidentLineIndex):
                self.line_index = self.line_index.reference
        else:
            raise RuntimeError("explicit query mode required before loading a cell")

    @property
    def identity(self):
        return {**super().identity, "provider_layout": "shared",
                "shared_state": "original cell connection/STRtree/WKB/WKT and raster cache",
                "resident_build_policy": "lazy, once per retained cell on first candidate call",
                "query_answer_cache": False}


@contextmanager
def scratch_working_directory(path):
    """Own a fresh scratch directory; close providers before restoring cwd."""
    previous = Path.cwd()
    path.mkdir(exist_ok=False)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def output_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def benchmark(batches, reference, candidate, emit, *, warmups=2, repeats=6,
              available_memory=lambda: float("inf"), memory_snapshot=lambda: {},
              clock=time.perf_counter):
    """Retain the complete planned denominator, including mismatches/failures."""
    if (not batches or type(warmups) is not int or warmups < 1
            or type(repeats) is not int or repeats < 1):
        raise ValueError("nonempty batches and positive integer warmups/repeats required")
    planned = len(batches)*2*(warmups+repeats)
    attempted = failures = mismatches = 0
    stopped = False
    for batch in batches:
        expected = expected_hash = None
        for phase, count in (("warmup", warmups), ("timed", repeats)):
            for repetition in range(count):
                order = ("reference", "candidate")
                if phase == "timed" and repetition % 2:
                    order = order[::-1]
                for name in order:
                    if available_memory() < MINIMUM_FREE_BYTES:
                        stopped = True
                        break
                    record = {"type": "call", "batch": batch["identity"], "phase": phase,
                              "repetition": repetition, "backend": name, "status": "failure"}
                    started = clock()
                    try:
                        rows = (reference if name == "reference" else candidate)(batch["lonlat"])
                        record["query_wall_seconds"] = clock()-started
                        if not isinstance(rows, list) or len(rows) != len(batch["lonlat"]):
                            raise ValueError("provider changed the registered query row count")
                        digest = output_digest(rows)
                        if expected is None:
                            if name != "reference":
                                raise ValueError("no successful reference output for this batch")
                            expected, expected_hash = rows, digest
                        exact = rows == expected and digest == expected_hash
                        mismatches += int(not exact)
                        record.update(status="success", output_sha256=digest,
                                      reference_output_sha256=expected_hash, exactly_equal=exact,
                                      output_row_count=len(rows))
                    except Exception as exc:
                        failures += 1
                        record.setdefault("query_wall_seconds", clock()-started)
                        record.update(error_type=type(exc).__name__, error_message=str(exc))
                    attempted += 1
                    record["memory"] = memory_snapshot()
                    emit(record)
                if stopped:
                    break
            if stopped:
                break
        if stopped:
            break
    return {"type": "completion", "expected_call_count": planned, "attempted_call_count": attempted,
            "failure_count": failures, "mismatch_count": mismatches, "resource_stopped": stopped,
            "all_calls_exact": attempted == planned and not failures and not mismatches,
            "certified": False, "scope": "registered endpoint map queries only; not full forecast equivalence"}


def select_probe_rows(rows):
    from .numerical_check import audit
    checked = audit(rows, tolerance_m=10.27506475)
    header = rows[0]
    expected = {"configurations": ["base", "all-terrain"], "seeds": [20260814],
                "particle_counts": [512, 1024], "max_steps_seconds": [.625, .3125],
                "map_backend": "multicell", "selection_policy": "lexical_independent_blocks",
                "save_particles": True}
    if (checked["failures"] or checked["independent_block_count"] != 3
            or any(header.get(k) != v for k, v in expected.items())
            or len(header["sample_ids"]) != 3 or header["expected_run_count"] != 24):
        raise ValueError("probe requires the full registered three-block v11 workload")
    selected = [row for row in rows[1:-1] if row["configuration"] == "all-terrain"
                and row["particles"] == 1024 and row["max_step_seconds"] == .3125]
    if (len(selected) != 3 or [r["sample_id"] for r in selected] != header["sample_ids"]
            or any(len(r["actual_horizons_seconds"]) != 4 for r in selected)):
        raise ValueError("probe requires all three origins and four registered scoring slots")
    return selected


def endpoint_batches(selected, directory, windows):
    from .precision_check import load_particle_evidence
    by_id = {window.sample_id: window for window in windows}
    batches = []
    for row in selected:
        window = by_id[row["sample_id"]]
        particles, targets, times = load_particle_evidence(row, directory)
        if (window.role != "validation" or window.block_id != row["independent_block_id"]
                or not np.array_equal(times, window.horizon_seconds)
                or not np.array_equal(targets, window.target_positions_m)):
            raise ValueError("saved particles differ from the bound validation frame/window")
        for slot, elapsed in enumerate(times):
            lonlat = window.frame.to_lonlat(particles[:, slot])
            batches.append({"identity": {"sample_id": window.sample_id,
                "independent_block_id": window.block_id, "scoring_slot": slot,
                "elapsed_seconds": float(elapsed), "point_count": len(lonlat),
                "particle_artifact_sha256": row["particle_artifact"]["sha256"],
                "query_lonlat_sha256": hashlib.sha256(lonlat.astype('<f8').tobytes()).hexdigest()},
                "lonlat": lonlat})
    return batches


def prepare_batches(args):
    from .development import load_development
    from .development_rollout import resolve_map_backend
    from .features import CanonicalEncoder
    from .precision_check import load_ledger, replay
    from .pilot_selection import select_windows

    rows = load_ledger(args.ledger, args.ledger_sha256)
    selected = select_probe_rows(rows)
    _, modules = resolve_map_backend("multicell")
    source_names = ("development_rollout.py", "rollout.py", "brownian.py", "dynamics.py",
                    "checkpoints.py", "development.py", "features.py", "configurations.py",
                    "metrics.py", "pilot_selection.py", "precision.py")
    if (rows[0]["map_source_sha256"] != {m.__name__: _hash(Path(m.__file__)) for m in modules}
            or rows[0]["source_sha256"] != {n: _hash(Path(__file__).with_name(n)) for n in source_names}):
        raise ValueError("original predictor/map source bindings changed")
    replay(rows, args.ledger.parent, tolerance_m=10.27506475)
    print("Replayed all 24 original particle workloads", flush=True)
    encoder = CanonicalEncoder.frozen_pirc22(args.snapshot)
    windows, parents, identity = load_development(args.eligibility, args.eligibility_sha256,
                                                args.release, args.data_root, encoder)
    if identity["sha256"] != rows[0]["development_identity"]:
        raise ValueError("development identity changed")
    chosen = select_windows(windows["validation"], count=3, policy="lexical_independent_blocks")
    if [w.sample_id for w in chosen] != rows[0]["sample_ids"]:
        raise ValueError("registered independent-block selection changed")
    batches = endpoint_batches(selected, args.ledger.parent, chosen)
    if len(batches) != 12 or any(len(b["lonlat"]) != 1024 for b in batches):
        raise ValueError("probe requires twelve complete 1024-point batches")
    return batches, parents, rows


def main():
    import psutil

    parser = argparse.ArgumentParser(description=__doc__)
    path_names = ("ledger", "eligibility", "release", "snapshot", "data-root", "output")
    for name in path_names:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--eligibility-sha256", required=True)
    parser.add_argument("--provider-layout", choices=("independent", "shared"), default="independent")
    parser.add_argument("--scratch-directory", type=Path,
                        help="new direct child of output's directory; required for shared layout")
    args = parser.parse_args()
    for name in path_names:
        attr = name.replace("-", "_")
        setattr(args, attr, getattr(args, attr).resolve())
    if args.output.exists():
        parser.error("output must be new; original evidence is never overwritten")
    if args.provider_layout == "shared" and args.scratch_directory is None:
        parser.error("shared layout requires an explicit new scratch directory")
    if args.scratch_directory is not None:
        args.scratch_directory = args.scratch_directory.resolve()
        if (args.scratch_directory.exists() or args.scratch_directory == args.output
                or args.scratch_directory.parent != args.output.parent):
            parser.error("scratch directory must be a new direct child of output's directory, distinct from output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    shared_layout = args.provider_layout == "shared"
    layout = {"provider_layout": args.provider_layout,
              "physical_map_owner_count": 1 if shared_layout else 2,
              "geometry_cells_per_owner": 4, "maximum_retained_geometry_cells": 4 if shared_layout else 8,
              "cache_scope": "shared cells and raster cache" if shared_layout else "independent provider caches",
              "memory_scope": "combined process including preflight; not independent per-backend memory",
              "scratch_directory": str(args.scratch_directory) if args.scratch_directory else None}

    def memory():
        value = process.memory_info()
        return {"rss_bytes": value.rss, "peak_working_set_bytes": getattr(value, "peak_wset", None),
                "available_system_bytes": psutil.virtual_memory().available}

    calls = []
    with args.output.open("x", encoding="utf-8") as target:
        def emit(value):
            target.write(json.dumps(value, allow_nan=False)+"\n")
            target.flush()
            if value.get("type") == "call":
                calls.append(value)

        emit({"type": "initialization", "schema_version": PROBE_VERSION, **layout,
              "ledger_sha256": args.ledger_sha256, "eligibility_sha256": args.eligibility_sha256,
              "probe_source_sha256": _hash(Path(__file__)), "expected_call_count": 192,
              "final_eval_label_prediction_metric_reads": 0, "certified": False})
        started = time.perf_counter()
        try:
            if psutil.virtual_memory().available < MINIMUM_FREE_BYTES:
                raise MemoryError("system free memory is below the registered 2 GiB floor")
            batches, parents, rows = prepare_batches(args)
            receipts = [args.data_root/name for name in (
                "registry/terrain_expansion_receipt.json", "registry/linear_expansion_receipt.json",
                "runs/pirc18-production-001/terrain-receipt.json",
                "runs/pirc18-production-001/remaining-terrain-receipt.json",
                "runs/pirc18-production-001/linear-receipt.json")]
            with ExitStack() as stack:
                if args.scratch_directory is not None:
                    stack.enter_context(scratch_working_directory(args.scratch_directory))
                if shared_layout:
                    shared = SharedProbeMap(args.data_root, receipts)
                    stack.callback(shared.close)
                    providers = (shared,)
                    reference, candidate = (partial(shared.query_mode, name)
                                            for name in ("reference", "candidate"))
                else:
                    reference = MultiCellRawMapQuery(args.data_root, receipts)
                    stack.callback(reference.close)
                    candidate = ProbeMap(args.data_root, receipts)
                    stack.callback(candidate.close)
                    providers = (reference, candidate)
                for provider in providers:
                    for parent in parents:
                        provider.register_snapshot_parent(parent)
                    if provider.identity["receipt_sha256"] != rows[-1]["maps"]["receipt_sha256"]:
                        raise ValueError("map receipts changed since the complete original ledger")
                emit({"type": "header", "schema_version": PROBE_VERSION, **layout, "batch_count": 12,
                      "warmups_per_backend_batch": 2, "timed_repeats_per_backend_batch": 6,
                      "expected_call_count": 192, "source_sha256": rows[0]["source_sha256"],
                      "map_source_sha256": rows[0]["map_source_sha256"],
                      "development_identity": rows[0]["development_identity"],
                      "preflight_wall_seconds": time.perf_counter()-started,
                      "memory_before_queries": memory(), "platform": platform.platform(),
                      "processor": platform.processor(), "logical_cpu_count": psutil.cpu_count(),
                      "duckdb_threads_per_connection": 2,
                      "timing_scope": "complete provider call; first-call geometry setup included; excludes output comparison",
                      "cold_scope": "provider initialization only; OS page cache not controlled",
                      "batches": [b["identity"] for b in batches]})
                print("Starting twelve fixed map-query batches (192 calls)", flush=True)
                result = benchmark(batches, reference, candidate, emit,
                                   available_memory=lambda: psutil.virtual_memory().available,
                                   memory_snapshot=memory)
                if shared_layout:
                    resources = {"shared_identity": shared.identity,
                                 "shared_cells": shared.geometry_cache_measurements,
                                 "candidate_resident_builds": shared.resident_builds}
                else:
                    resources = {"reference_identity": reference.identity, "candidate_identity": candidate.identity,
                                 "reference_cells": reference.geometry_cache_measurements,
                                 "candidate_cells": candidate.geometry_cache_measurements,
                                 "candidate_resident_builds": candidate.resident_builds}
                emit({"type": "resources", **layout, **resources, "memory_after_queries": memory()})
        except Exception as exc:
            result = {"type": "completion", "expected_call_count": 192, "attempted_call_count": len(calls),
                      "failure_count": sum(r["status"] == "failure" for r in calls),
                      "all_calls_exact": False, "certified": False,
                      "error_type": type(exc).__name__, "error_message": str(exc)}
        emit(result)
    print(json.dumps({"output": str(args.output), **result}, allow_nan=False), flush=True)
    if not result["all_calls_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
