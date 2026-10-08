"""Observe invalid feature provenance in one hash-bound original v11 forecast.

This is not a numerical qualification subset or a replacement map backend.
Private predicted coordinates stay in the exclusive local diagnostic ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .brownian import BrownianPath, integration_grid
from .checkpoints import restore_dynamics
from .configurations import configuration_encoder, terrain_configurations
from .development import load_development
from .development_rollout import resolve_map_backend
from .features import CanonicalEncoder, PredictedPositionFeatures
from . import geometry_transport_probe as probe
from .pilot_selection import select_windows
from .precision_check import KEYS, load_ledger, load_particle_evidence, replay
from .qualification import _hash
from .rollout import rollout

REPLAY_VERSION = "pirc17-invalid-feature-replay-v1"
SAMPLE_ID = "0924849bf85b7c13eb7fc8378d0725b63e50b46c3e074e08f204a2303fdd6149"
EXPECTED_INVALID_ROWS = 51
EXPECTED_QUERY_ROWS = 5898240
EXPECTED_QUERY_CALLS = 5760


def diagnostic_scalar(value):
    """Preserve nonfinite invalid inputs explicitly, without JSON NaN or imputation."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("canonical diagnostic fields must be scalar")


class FeatureObserver:
    """Delegate exactly once and return the original Features object unchanged."""

    def __init__(self, encoder, map_query, frame, emit, *, available_memory=lambda: float("inf")):
        self.original_encoder = encoder
        self.columns = tuple(encoder.columns)
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("unique feature columns required for provenance")
        self.frame, self.emit, self.available_memory = frame, emit, available_memory
        self.provider = PredictedPositionFeatures(self, map_query, frame)
        self.state = None
        self.encoded = False
        self.query_times = []
        self.query_rows = self.invalid_rows = self.event_count = self.completed_calls = 0
        self.invalid_columns = Counter()

    def __call__(self, state):
        if self.state is not None:
            raise RuntimeError("one synchronous observed feature query required")
        if self.available_memory() < probe.MINIMUM_FREE_BYTES:
            raise MemoryError("system free memory is below the registered 2 GiB floor")
        self.state, self.encoded = state, False
        try:
            features = self.provider(state)
            if not self.encoded:
                raise RuntimeError("feature query did not invoke the observed encoder")
            return features
        finally:
            self.state = None

    def encode(self, rows):
        if self.state is None or self.encoded:
            raise RuntimeError("exactly one encode inside each observed state query required")
        self.encoded = True
        features = self.original_encoder.encode(rows)
        valid, values = np.asarray(features.valid), np.asarray(features.values)
        if (valid.dtype != np.bool_ or valid.shape != (len(rows), len(self.columns))
                or values.shape != valid.shape or len(rows) != len(self.state.positions_m)):
            raise ValueError("observed feature columns, mask or row count changed")
        query_index = self.completed_calls
        self.completed_calls += 1
        self.query_times.append(float(self.state.elapsed_seconds))
        self.query_rows += len(rows)
        invalid = np.flatnonzero(~valid.all(axis=1))
        self.invalid_rows += len(invalid)
        for index in invalid:
            column_indexes = np.flatnonzero(~valid[index]).tolist()
            columns = [self.columns[i] for i in column_indexes]
            event = {"type": "invalid_feature", "query_index": query_index,
                     "elapsed_seconds": float(self.state.elapsed_seconds), "particle_index": int(index),
                     "invalid_column_indexes": column_indexes, "invalid_columns": columns,
                     "encoded_values": [diagnostic_scalar(v) for v in values[index]],
                     "canonical_row": {k: diagnostic_scalar(v) for k, v in rows[index].items()},
                     "predicted_position_m": self.state.positions_m[index].tolist(),
                     "predicted_history_positions_m": self.state.history_positions_m[index].tolist(),
                     "history_elapsed_seconds": self.state.history_elapsed_seconds.tolist()}
            self.emit(event)
            self.event_count += 1
            self.invalid_columns.update(columns)
        if self.completed_calls % 128 == 0:
            self.emit({"type": "progress", "last_query_elapsed_seconds": self.query_times[-1], **self.counts})
        return features

    @property
    def counts(self):
        return {"completed_feature_calls": self.completed_calls, "feature_query_rows": self.query_rows,
                "invalid_feature_rows": self.invalid_rows, "recorded_invalid_events": self.event_count,
                "invalid_column_counts": dict(sorted(self.invalid_columns.items()))}


def select_workload(rows):
    selected = probe.select_probe_rows(rows)  # Audit all original 24 runs, not just this selected case.
    matches = [r for r in selected if r["sample_id"] == SAMPLE_ID]
    if len(matches) != 1:
        raise ValueError("registered invalid-feature workload is missing")
    chosen = matches[0]
    if (chosen.get("invalid_feature_rows") != EXPECTED_INVALID_ROWS
            or chosen.get("feature_query_rows") != EXPECTED_QUERY_ROWS
            or chosen["actual_horizons_seconds"] != [60., 300., 900., 1800.]
            or any(r.get("invalid_feature_rows") != 0 for r in rows[1:-1] if r is not chosen)
            or rows[0].get("physical_history_step_seconds") != 5.):
        raise ValueError("original invalid-feature counts or physical-time workload changed")
    return chosen


def prepare(args):
    rows = load_ledger(args.ledger, args.ledger_sha256)
    row = select_workload(rows)
    query_type, modules = resolve_map_backend("multicell")
    names = ("development_rollout.py", "rollout.py", "brownian.py", "dynamics.py", "checkpoints.py",
             "development.py", "features.py", "configurations.py", "metrics.py", "pilot_selection.py", "precision.py")
    if (rows[0]["source_sha256"] != {n: _hash(Path(__file__).with_name(n)) for n in names}
            or rows[0]["map_source_sha256"] != {m.__name__: _hash(Path(m.__file__)) for m in modules}):
        raise ValueError("original predictor/map source bindings changed")
    audited = replay(rows, args.ledger.parent, tolerance_m=10.27506475)
    if len(audited["runs"]) != 24:
        raise ValueError("complete original 24-workload particle replay required")
    if _hash(args.fit) != args.fit_sha256 or args.fit_sha256 != rows[0]["fit_sha256"]:
        raise ValueError("original fit artifact changed")
    fitted = json.loads(args.fit.read_text(encoding="utf-8"))
    configs = terrain_configurations()
    if fitted["schema_version"] != "pirc17-development-fit-v1" or fitted["configurations"] != configs:
        raise ValueError("fit configuration registry differs")
    model = restore_dynamics(fitted["models"][row["configuration"]][str(row["seed"])])
    if (model.identity["configuration_identity"] != configs[row["configuration"]]["sha256"]
            or model.identity["training_identity"] != fitted["development_identity"]["sha256"]
            or model.identity["seed"] != row["seed"]):
        raise ValueError("checkpoint mislabeled by configuration, population or seed")
    encoder = CanonicalEncoder.frozen_pirc22(args.snapshot)
    windows, parents, identity = load_development(args.eligibility, args.eligibility_sha256,
                                                args.release, args.data_root, encoder)
    if identity != fitted["development_identity"] or identity["sha256"] != rows[0]["development_identity"]:
        raise ValueError("original fit and rollout development populations differ")
    selected = select_windows(windows["validation"], count=3, policy="lexical_independent_blocks")
    if [w.sample_id for w in selected] != rows[0]["sample_ids"]:
        raise ValueError("registered independent-block selection changed")
    window = next(w for w in selected if w.sample_id == SAMPLE_ID)
    reference_positions, targets, times = load_particle_evidence(row, args.ledger.parent)
    if (window.role != "validation" or window.block_id != row["independent_block_id"]
            or not np.array_equal(times, window.horizon_seconds)
            or not np.array_equal(targets, window.target_positions_m)):
        raise ValueError("saved particles differ from the bound validation frame/window")
    driver = BrownianPath(times, rows[0]["max_steps_seconds"], history_step_seconds=5.,
                          particles=1024, seed=row["seed"], stream_id=window.sample_id)
    if driver.identity != row["brownian_identity"]:
        raise ValueError("original complete Brownian identity changed")
    query_times = integration_grid(times, row["max_step_seconds"], 5.)[:-1]
    if len(query_times) != EXPECTED_QUERY_CALLS:
        raise ValueError("original complete query grid changed")
    return SimpleNamespace(rows=rows, row=row, window=window, model=model, parents=parents, driver=driver,
                           encoder=configuration_encoder(encoder, row["configuration"]), query_type=query_type,
                           reference_positions=reference_positions, reference_times=times, query_times=query_times)


def array_digest(values):
    return hashlib.sha256(np.asarray(values).astype('<f8').tobytes()).hexdigest()


def compare_prediction(prediction, prepared, observer):
    expected, actual = prepared.reference_positions, prediction.positions_m
    position_hash, reference_hash = array_digest(actual), array_digest(expected)
    same_shape = actual.shape == expected.shape
    finite = bool(np.isfinite(actual).all())
    positions_exact = bool(same_shape and np.array_equal(actual, expected) and position_hash == reference_hash)
    times_exact = bool(np.array_equal(prediction.elapsed_seconds, prepared.reference_times))
    calls_exact = observer.query_times == prepared.query_times
    counts_exact = (prediction.invalid_feature_rows == observer.invalid_rows == observer.event_count
                    == prepared.row["invalid_feature_rows"]
                    and prediction.feature_query_rows == observer.query_rows == prepared.row["feature_query_rows"]
                    and observer.completed_calls == len(prepared.query_times))
    return {"type": "comparison", "positions_exact": positions_exact, "times_exact": times_exact,
            "query_grid_exact": calls_exact, "counts_exact": counts_exact,
            "prediction_positions_sha256": position_hash, "reference_positions_sha256": reference_hash,
            "maximum_absolute_position_difference_m": float(np.max(np.abs(actual-expected))) if same_shape and finite else None,
            "exact_replay": bool(positions_exact and times_exact and calls_exact and counts_exact), "certified": False}


def main():
    import psutil

    parser = argparse.ArgumentParser(description=__doc__)
    paths = ("ledger", "fit", "eligibility", "release", "snapshot", "data-root", "output", "scratch-directory")
    for name in paths:
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("ledger", "fit", "eligibility"):
        parser.add_argument("--"+name+"-sha256", required=True)
    args = parser.parse_args()
    for name in paths:
        name = name.replace("-", "_")
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        parser.error("output must be new; original evidence is never overwritten")
    if (args.scratch_directory.exists() or args.scratch_directory == args.output
            or args.scratch_directory.parent != args.output.parent):
        parser.error("scratch directory must be a new direct child of output's directory, distinct from output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()

    def memory():
        info = process.memory_info()
        return {"rss_bytes": info.rss, "peak_working_set_bytes": getattr(info, "peak_wset", None),
                "available_system_bytes": psutil.virtual_memory().available}

    attempted = 0
    observer = None
    with args.output.open("x", encoding="utf-8") as target:
        def emit(row):
            target.write(json.dumps(row, allow_nan=False)+"\n")
            target.flush()

        emit({"type": "initialization", "schema_version": REPLAY_VERSION, "expected_diagnostic_runs": 1,
              "original_run_denominator": 24, "expected_feature_calls": EXPECTED_QUERY_CALLS,
              "expected_feature_query_rows": EXPECTED_QUERY_ROWS, "expected_invalid_feature_rows": EXPECTED_INVALID_ROWS,
              "ledger_sha256": args.ledger_sha256, "fit_sha256": args.fit_sha256,
              "eligibility_sha256": args.eligibility_sha256, "diagnostic_source_sha256": _hash(Path(__file__)),
              "preflight_support_source_sha256": {n: _hash(Path(__file__).with_name(n)) for n in
                  ("geometry_transport_probe.py", "precision_check.py", "numerical_check.py")},
              "scratch_directory": str(args.scratch_directory), "certified": False,
              "final_eval_label_prediction_metric_reads": 0})
        started = time.perf_counter()
        try:
            if psutil.virtual_memory().available < probe.MINIMUM_FREE_BYTES:
                raise MemoryError("system free memory is below the registered 2 GiB floor")
            prepared = prepare(args)
            receipts = [args.data_root/name for name in (
                "registry/terrain_expansion_receipt.json", "registry/linear_expansion_receipt.json",
                "runs/pirc18-production-001/terrain-receipt.json",
                "runs/pirc18-production-001/remaining-terrain-receipt.json",
                "runs/pirc18-production-001/linear-receipt.json")]
            with ExitStack() as stack:
                stack.enter_context(probe.scratch_working_directory(args.scratch_directory))
                maps = prepared.query_type(args.data_root, receipts)
                stack.callback(maps.close)
                for parent in prepared.parents:
                    maps.register_snapshot_parent(parent)
                if maps.identity["receipt_sha256"] != prepared.rows[-1]["maps"]["receipt_sha256"]:
                    raise ValueError("map receipts changed since the complete original ledger")
                observer = FeatureObserver(prepared.encoder, maps, prepared.window.frame, emit,
                                           available_memory=lambda: psutil.virtual_memory().available)
                emit({"type": "header", "schema_version": REPLAY_VERSION,
                      "workload": {k: prepared.row[k] for k in KEYS},
                      "independent_block_id": prepared.row["independent_block_id"],
                      "particle_artifact": prepared.row["particle_artifact"],
                      "brownian_identity": prepared.driver.identity, "feature_columns": list(observer.columns),
                      "source_sha256": prepared.rows[0]["source_sha256"],
                      "map_source_sha256": prepared.rows[0]["map_source_sha256"],
                      "development_identity": prepared.rows[0]["development_identity"],
                      "preflight_wall_seconds": time.perf_counter()-started, "memory_before_replay": memory(),
                      "scope": "one original invalid-feature provenance replay; not 24-run numerical qualification"})
                attempted = 1
                replay_started = time.perf_counter()
                prediction = rollout(prepared.window.origin, prepared.window.horizon_seconds,
                    particles=prepared.row["particles"], seed=prepared.row["seed"],
                    max_step_seconds=prepared.row["max_step_seconds"], history_step_seconds=5.,
                    base_drift=prepared.model.base_drift, diffusion=prepared.model.diffusion,
                    terrain=observer, conditioner=prepared.model.correction, brownian_increments=prepared.driver)
                elapsed = time.perf_counter()-replay_started
                comparison = compare_prediction(prediction, prepared, observer)
                emit(comparison)
                emit({"type": "resources", "maps": maps.identity, "geometry_cells": maps.geometry_cache_measurements,
                      "rollout_wall_seconds_including_observation": elapsed, "memory_after_replay": memory(),
                      **observer.counts})
                result = {"type": "completion", "expected_diagnostic_runs": 1, "attempted_diagnostic_runs": attempted,
                          "exact_replay": comparison["exact_replay"], "resource_stopped": False,
                          "certified": False, **observer.counts}
        except Exception as exc:
            result = {"type": "completion", "expected_diagnostic_runs": 1, "attempted_diagnostic_runs": attempted,
                      "exact_replay": False, "certified": False, "error_type": type(exc).__name__,
                      "error_message": str(exc)[:300], "resource_stopped": isinstance(exc, MemoryError),
                      "partial_observation": observer.counts if observer is not None else None}
        emit(result)
    print(json.dumps({"output": str(args.output), **result}, allow_nan=False), flush=True)
    if not result["exact_replay"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
