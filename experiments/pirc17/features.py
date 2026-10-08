"""Frozen PIRC-22 transforms applied to newly queried canonical map rows.

No snapshot trajectory rows are loaded by encode(). The snapshot adapter is used
only for its verified specification and the exact registered transform kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pyarrow as pa

from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
from experiments.pirc22.consumer import load_benchmark_selection_binding
from .origins import HISTORY_POINTS
from .rollout import Features, PredictedState

ENCODER_VERSION = "pirc17-online-frozen-transforms-v1"
EARTH_RADIUS_M = 6371008.8


@dataclass(frozen=True)
class LocalFrame:
    """Origin-anchored equirectangular frame; no full-route centering."""
    longitude: float
    latitude: float

    def __post_init__(self):
        if not np.isfinite([self.longitude, self.latitude]).all():
            raise ValueError("finite geographical origin required")
        if not -180 <= self.longitude <= 180 or not -89 < self.latitude < 89:
            raise ValueError("frame is outside supported nonpolar geographic extent")

    def to_lonlat(self, positions_m):
        values = np.asarray(positions_m, dtype=float)
        if values.shape[-1:] != (2,) or not np.isfinite(values).all():
            raise ValueError("finite east/north positions required")
        scale = math.pi * EARTH_RADIUS_M / 180
        result = values / [scale * math.cos(math.radians(self.latitude)), scale]
        return result + [self.longitude, self.latitude]

    def from_lonlat(self, positions):
        values = np.asarray(positions, dtype=float)
        if values.shape[-1:] != (2,) or not np.isfinite(values).all():
            raise ValueError("finite lon/lat positions required")
        scale = math.pi * EARTH_RADIUS_M / 180
        return (values - [self.longitude, self.latitude]) * [scale * math.cos(math.radians(self.latitude)), scale]


class CanonicalEncoder:
    def __init__(self, adapter: FeatureSnapshotAdapter):
        if adapter.selection.interaction_ids:
            raise ValueError("online encoder does not support unregistered interactions")
        for variant_id in adapter.selection.variant_ids:
            variant = adapter._variant_by_id[variant_id]
            if variant["fit_scope"] != "none" or variant["transform"] == "visible_time_difference":
                raise ValueError("online encoder requires pointwise, unfitted transforms")
        self.adapter = adapter
        self.columns = adapter.output_columns

    @classmethod
    def frozen_pirc22(cls, snapshot):
        binding = load_benchmark_selection_binding()
        selected = binding["selected_configuration"]
        adapter = FeatureSnapshotAdapter(snapshot, FeatureSelection(
            variant_ids=tuple(selected["variant_ids"]),
            composition_ids=tuple(selected["composition_ids"]),
            include_validity_indicators=True))
        if adapter.manifest.get("history_window_points") != HISTORY_POINTS:
            raise ValueError("frozen history window differs from rollout contract")
        result = cls(adapter)
        if 2 * len(result.columns) != selected["model_input_dim"]:
            raise ValueError("frozen feature dimension mismatch")
        return result

    def encode(self, rows):
        if not rows:
            raise ValueError("nonempty queried rows required")
        # Stored canonical numeric values are float32. Match that representation
        # before invoking the identical offline transform kernels.
        numeric = {name for name in self.adapter._column_factor}
        columns = set()
        for variant_id in self.adapter.selection.variant_ids:
            parameters = self.adapter._variant_by_id[variant_id]["parameters"]
            names = list(parameters["input_columns"]) + list(parameters.get("reference_columns", ()))
            columns.update(names)
            columns.update(self.adapter._factor_by_id[self.adapter._column_factor[n]]["status_column"] for n in names)
        arrays = {}
        for name in columns:
            # Missing canonical fields are explicit missing values/statuses.
            values = [r.get(name, None if name in numeric else "source_missing") for r in rows]
            arrays[name] = pa.array(values, type=pa.float32() if name in numeric else pa.string())
        table = pa.table(arrays)
        variants = {v: self.adapter._variant(table, v) for v in self.adapter.selection.variant_ids}
        results = list(variants.values()) + [self.adapter._composition(c, variants) for c in self.adapter.selection.composition_ids]
        if not results:
            return Features(np.empty((len(rows), 0)), np.empty((len(rows), 0), dtype=bool))
        return Features(np.concatenate([r.values for r in results], axis=1),
                        np.concatenate([r.valid for r in results], axis=1))


class PredictedPositionFeatures:
    def __init__(self, encoder: CanonicalEncoder, map_query, frame: LocalFrame):
        self.encoder, self.map_query, self.frame = encoder, map_query, frame

    def __call__(self, state: PredictedState):
        lonlat = self.frame.to_lonlat(state.positions_m)
        rows = [dict(r) for r in self.map_query(lonlat)]
        if len(rows) != len(lonlat):
            raise ValueError("map provider changed query row count")
        history = self.frame.to_lonlat(state.history_positions_m)
        if history.shape[1] >= 2:
            # Match PIRC-21's cosine at the first point of the history window,
            # not the origin frame's cosine at a potentially different latitude.
            delta = history[:, -1] - history[:, 0]
            delta[:, 0] *= np.cos(np.radians(history[:, 0, 1]))
            delta *= math.pi * EARTH_RADIUS_M / 180
            norms = np.linalg.norm(delta, axis=1)
            valid = norms > 1e-9
            direction = delta / np.where(valid, norms, 1)[:, None]
        else:
            # Explicit known/prior velocity origin semantics, versioned separately.
            direction, valid = state.history_direction
        for i, row in enumerate(rows):
            row["history_direction_east"] = float(direction[i, 0]) if valid[i] else None
            row["history_direction_north"] = float(direction[i, 1]) if valid[i] else None
            row["historical_motion_status"] = "valid" if valid[i] else "insufficient_history"
        return self.encoder.encode(rows)
