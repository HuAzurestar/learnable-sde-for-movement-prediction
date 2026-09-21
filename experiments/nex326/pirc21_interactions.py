"""Causal mathematical interactions between PIRC-21 fields and SDE state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from .cohort import Segment


INTERACTION_REGISTRY_VERSION = "pirc21-kinematic-interactions-v1"
_EPSILON = 1e-12


class PIRC21InteractionError(ValueError):
    """A requested interaction is unknown, unavailable, or numerically invalid."""


@dataclass(frozen=True)
class InteractionColumn:
    name: str
    unit: str
    role: str


@dataclass(frozen=True)
class InteractionDefinition:
    interaction_id: str
    factor_ids: tuple[str, ...]
    operator: str
    formula: str
    required_columns: tuple[str, ...]
    output_columns: tuple[InteractionColumn, ...]
    validity_rule: str
    parameters: tuple[tuple[str, str], ...] = ()

    def record(self) -> dict[str, object]:
        return {
            "interaction_id": self.interaction_id,
            "factor_ids": list(self.factor_ids),
            "operator": self.operator,
            "formula": self.formula,
            "required_columns": list(self.required_columns),
            "output_columns": [
                {"name": item.name, "unit": item.unit, "role": item.role}
                for item in self.output_columns
            ],
            "validity_rule": self.validity_rule,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class InteractionBatch:
    columns: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray


def _column(name: str, unit: str, role: str) -> InteractionColumn:
    return InteractionColumn(name=name, unit=unit, role=role)


def _geometry_definition(
    prefix: str,
    factor_id: str,
    distance_column: str,
    direction_prefix: str | None = None,
) -> InteractionDefinition:
    direction_prefix = direction_prefix or prefix
    direction_east = f"v_{direction_prefix}_direction_east"
    direction_north = f"v_{direction_prefix}_direction_north"
    return InteractionDefinition(
        interaction_id=f"{prefix}.motion",
        factor_ids=(factor_id,),
        operator="geometry_motion",
        formula=(
            "d=expm1(log_distance); approach=dot(v,n); "
            "tangent=abs(cross(v,n)); alignment=approach/||v||; "
            "distance_rate=delta(d)/delta(t); "
            "time_to_contact=d/max(approach,epsilon); "
            "normalized_distance=d/(60s*||v||)"
        ),
        required_columns=(distance_column, direction_east, direction_north),
        output_columns=(
            _column(f"k_{prefix}_approach_speed_mps", "m/s", "approach_speed"),
            _column(f"k_{prefix}_tangent_speed_mps", "m/s", "tangential_speed"),
            _column(f"k_{prefix}_heading_alignment", "1", "heading_alignment"),
            _column(f"k_{prefix}_distance_rate_mps", "m/s", "distance_rate"),
            _column(f"k_{prefix}_time_to_contact_s", "s", "time_to_contact"),
            _column(
                f"k_{prefix}_distance_over_60s_travel",
                "1",
                "speed_normalized_distance",
            ),
        ),
        validity_rule=(
            "causal velocity and current geometry must be valid; rate also requires "
            "previous distance; alignment/normalized distance require nonzero speed; "
            "time-to-contact requires positive approach speed"
        ),
        parameters=(("distance_encoding", "log1p_m"), ("travel_horizon_s", "60")),
    )


_BASE_DEFINITIONS = (
    InteractionDefinition(
        interaction_id="history.motion",
        factor_ids=("historical_motion",),
        operator="history_motion",
        formula="v_i=(x_i-x_(i-1))/(t_i-t_(i-1)); speed=||v_i||",
        required_columns=(),
        output_columns=(
            _column("k_velocity_east_mps", "m/s", "causal_velocity_east"),
            _column("k_velocity_north_mps", "m/s", "causal_velocity_north"),
            _column("k_speed_mps", "m/s", "causal_speed"),
        ),
        validity_rule="requires one visible previous point and positive finite delta(t)",
    ),
    InteractionDefinition(
        interaction_id="history.acceleration",
        factor_ids=("historical_motion",),
        operator="history_acceleration",
        formula=(
            "a_i=(v_i-v_(i-1))/(t_i-t_(i-1)); "
            "turn_rate=atan2(cross(v_(i-1),v_i),dot(v_(i-1),v_i))/delta(t)"
        ),
        required_columns=(),
        output_columns=(
            _column("k_acceleration_east_mps2", "m/s2", "causal_acceleration_east"),
            _column("k_acceleration_north_mps2", "m/s2", "causal_acceleration_north"),
            _column("k_turn_rate_radps", "rad/s", "causal_turn_rate"),
        ),
        validity_rule=(
            "acceleration requires two causal velocities; turn rate also requires "
            "both velocities to have nonzero magnitude"
        ),
    ),
    InteractionDefinition(
        interaction_id="surface.gradient",
        factor_ids=("dem_surface",),
        operator="surface_gradient",
        formula="gradient=-tan(slope)*downslope_unit",
        required_columns=(
            "v_surface_slope_rad",
            "v_surface_aspect_sin",
            "v_surface_aspect_cos",
        ),
        output_columns=(
            _column("k_surface_gradient_east", "1", "elevation_gradient_east"),
            _column("k_surface_gradient_north", "1", "elevation_gradient_north"),
        ),
        validity_rule="all current surface orientation columns must be valid",
    ),
    InteractionDefinition(
        interaction_id="surface.velocity",
        factor_ids=("dem_surface",),
        operator="surface_velocity",
        formula=(
            "uphill_speed=dot(v,gradient/||gradient||); "
            "contour_speed=abs(cross(v,gradient/||gradient||)); "
            "height_rate=dot(v,gradient)"
        ),
        required_columns=(
            "v_surface_slope_rad",
            "v_surface_aspect_sin",
            "v_surface_aspect_cos",
        ),
        output_columns=(
            _column("k_surface_uphill_speed_mps", "m/s", "uphill_speed"),
            _column("k_surface_contour_speed_mps", "m/s", "contour_speed"),
            _column("k_surface_height_rate_mps", "m/s", "directional_height_rate"),
        ),
        validity_rule="causal velocity and current surface orientation must be valid",
    ),
    InteractionDefinition(
        interaction_id="surface.directional_curvature",
        factor_ids=("dem_surface",),
        operator="surface_directional_curvature",
        formula=(
            "speed^2*dot(gradient_i-gradient_(i-1),unit(v_i))/"
            "||x_i-x_(i-1)||, a causal finite-difference approximation of v^T H_h v"
        ),
        required_columns=(
            "v_surface_slope_rad",
            "v_surface_aspect_sin",
            "v_surface_aspect_cos",
        ),
        output_columns=(
            _column(
                "k_surface_directional_curvature_mps2",
                "m/s2",
                "directional_hessian_quadratic",
            ),
        ),
        validity_rule=(
            "causal velocity, positive travelled distance, and current/previous "
            "surface orientation must be valid"
        ),
    ),
    InteractionDefinition(
        interaction_id="worldcover.grouped_speed",
        factor_ids=("worldcover",),
        operator="categorical_speed",
        formula="grouped_one_hot(class)*||v||",
        required_columns=(
            "v_worldcover_built",
            "v_worldcover_vegetated",
            "v_worldcover_water_wetland",
            "v_worldcover_bare_snow_unknown",
        ),
        output_columns=(
            _column("k_worldcover_built_speed_mps", "m/s", "category_speed_gate"),
            _column(
                "k_worldcover_vegetated_speed_mps", "m/s", "category_speed_gate"
            ),
            _column("k_worldcover_water_speed_mps", "m/s", "category_speed_gate"),
            _column("k_worldcover_other_speed_mps", "m/s", "category_speed_gate"),
        ),
        validity_rule="causal velocity and every grouped category indicator must be valid",
    ),
    InteractionDefinition(
        interaction_id="worldcover.transition",
        factor_ids=("worldcover",),
        operator="categorical_transition",
        formula="1[argmax(group_i) != argmax(group_(i-1))]",
        required_columns=(
            "v_worldcover_built",
            "v_worldcover_vegetated",
            "v_worldcover_water_wetland",
            "v_worldcover_bare_snow_unknown",
        ),
        output_columns=(
            _column("k_worldcover_transition", "1", "category_transition"),
        ),
        validity_rule="current and previous grouped category indicators must be valid",
    ),
    InteractionDefinition(
        interaction_id="jrc.attribute_rates",
        factor_ids=("jrc_surface_water",),
        operator="field_time_difference",
        formula="delta(field)/delta(t) over the visible trajectory prefix",
        required_columns=(
            "v_jrc_occurrence_percent",
            "v_jrc_seasonality_months",
        ),
        output_columns=(
            _column(
                "k_jrc_occurrence_rate_pctps",
                "percent/s",
                "water_occurrence_directional_rate",
            ),
            _column(
                "k_jrc_seasonality_rate_monthps",
                "month/s",
                "water_seasonality_directional_rate",
            ),
        ),
        validity_rule="positive delta(t) and current/previous field values must be valid",
    ),
)

_GEOMETRY_COLUMNS = {
    "road": ("overture_road", "v_road_distance_log1p", "road"),
    "path": ("overture_path", "v_path_distance_log1p", "path"),
    "rail": ("overture_rail", "v_rail_distance_log1p", "rail"),
    "river": ("hydrorivers_river", "v_river_distance_log1p", "river"),
    "jrc": ("jrc_surface_water", "v_jrc_water_distance_log1p", "jrc_water"),
    "navigable_water": (
        "overture_navigable_water",
        "v_navigable_water_distance_log1p",
        "navigable_water",
    ),
    "ridge": ("osm_ridge", "v_ridge_distance_log1p", "ridge"),
    "cliff": ("osm_cliff", "v_cliff_distance_log1p", "cliff"),
}

INTERACTION_DEFINITIONS = {
    item.interaction_id: item
    for item in (
        *_BASE_DEFINITIONS,
        *(
            _geometry_definition(prefix, factor_id, distance_column, direction_prefix)
            for prefix, (
                factor_id,
                distance_column,
                direction_prefix,
            ) in _GEOMETRY_COLUMNS.items()
        ),
    )
}


def interaction_registry_record() -> dict[str, object]:
    return {
        "registry_version": INTERACTION_REGISTRY_VERSION,
        "interactions": [
            INTERACTION_DEFINITIONS[key].record()
            for key in sorted(INTERACTION_DEFINITIONS)
        ],
    }


def interaction_registry_fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(
            interaction_registry_record(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def validate_interaction_selection(
    interaction_ids: Sequence[str], available_columns: Sequence[str]
) -> None:
    ids = tuple(str(value) for value in interaction_ids)
    if len(ids) != len(set(ids)):
        raise PIRC21InteractionError("interaction IDs must be ordered and unique")
    unknown = set(ids) - set(INTERACTION_DEFINITIONS)
    if unknown:
        raise PIRC21InteractionError(f"unknown interactions: {sorted(unknown)}")
    available = set(available_columns)
    for interaction_id in ids:
        missing = set(INTERACTION_DEFINITIONS[interaction_id].required_columns) - available
        if missing:
            raise PIRC21InteractionError(
                f"interaction {interaction_id} requires unavailable columns: "
                f"{sorted(missing)}"
            )


def interaction_output_columns(interaction_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        column.name
        for interaction_id in interaction_ids
        for column in INTERACTION_DEFINITIONS[str(interaction_id)].output_columns
    )


def _causal_velocity(segment: Segment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(segment.time)
    velocity = np.full((count, 2), np.nan, dtype=float)
    valid = np.zeros(count, dtype=bool)
    elapsed = np.diff(np.asarray(segment.time, dtype=float))
    displacement = np.diff(np.asarray(segment.state, dtype=float), axis=0)
    usable = (
        np.isfinite(elapsed)
        & (elapsed > 0)
        & np.isfinite(displacement).all(axis=1)
    )
    velocity[1:][usable] = displacement[usable] / elapsed[usable, None]
    valid[1:] = usable
    return velocity, valid, elapsed


def _surface_gradient(
    values: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    names = (
        "v_surface_slope_rad",
        "v_surface_aspect_sin",
        "v_surface_aspect_cos",
    )
    slope, down_east, down_north = (np.asarray(values[name], dtype=float) for name in names)
    usable = np.logical_and.reduce([np.asarray(valid[name], dtype=bool) for name in names])
    usable &= np.isfinite(slope) & np.isfinite(down_east) & np.isfinite(down_north)
    magnitude = np.tan(slope)
    gradient = np.column_stack((-magnitude * down_east, -magnitude * down_north))
    gradient[~usable] = np.nan
    return gradient, usable


def _history_motion(segment: Segment) -> InteractionBatch:
    velocity, velocity_valid, _ = _causal_velocity(segment)
    speed = np.linalg.norm(velocity, axis=1)
    values = np.column_stack((velocity, speed))
    valid = np.repeat(velocity_valid[:, None], 3, axis=1)
    return InteractionBatch((), values, valid)


def _history_acceleration(segment: Segment) -> InteractionBatch:
    velocity, velocity_valid, elapsed = _causal_velocity(segment)
    count = len(segment.time)
    values = np.full((count, 3), np.nan, dtype=float)
    valid = np.zeros((count, 3), dtype=bool)
    for index in range(2, count):
        dt = elapsed[index - 1]
        if velocity_valid[index] and velocity_valid[index - 1] and dt > 0:
            values[index, :2] = (velocity[index] - velocity[index - 1]) / dt
            valid[index, :2] = True
            previous_speed = float(np.linalg.norm(velocity[index - 1]))
            current_speed = float(np.linalg.norm(velocity[index]))
            if previous_speed > _EPSILON and current_speed > _EPSILON:
                cross = (
                    velocity[index - 1, 0] * velocity[index, 1]
                    - velocity[index - 1, 1] * velocity[index, 0]
                )
                dot = float(np.dot(velocity[index - 1], velocity[index]))
                values[index, 2] = np.arctan2(cross, dot) / dt
                valid[index, 2] = True
    return InteractionBatch((), values, valid)


def _surface_values(
    definition: InteractionDefinition,
    segment: Segment,
    source_values: Mapping[str, np.ndarray],
    source_valid: Mapping[str, np.ndarray],
) -> InteractionBatch:
    gradient, surface_valid = _surface_gradient(source_values, source_valid)
    if definition.operator == "surface_gradient":
        return InteractionBatch(
            (), gradient, np.repeat(surface_valid[:, None], 2, axis=1)
        )
    velocity, velocity_valid, _ = _causal_velocity(segment)
    if definition.operator == "surface_velocity":
        magnitude = np.linalg.norm(gradient, axis=1)
        uphill = np.divide(
            gradient,
            magnitude[:, None],
            out=np.zeros_like(gradient),
            where=magnitude[:, None] > _EPSILON,
        )
        # Flat terrain has a valid zero gradient but no preferred uphill direction.
        uphill_speed = np.einsum("ij,ij->i", velocity, uphill)
        contour_speed = np.abs(
            velocity[:, 0] * uphill[:, 1] - velocity[:, 1] * uphill[:, 0]
        )
        height_rate = np.einsum("ij,ij->i", velocity, gradient)
        result = np.column_stack((uphill_speed, contour_speed, height_rate))
        height_valid = surface_valid & velocity_valid
        direction_valid = height_valid & (magnitude > _EPSILON)
        result[~direction_valid, :2] = np.nan
        result[~height_valid, 2] = np.nan
        return InteractionBatch(
            (),
            result,
            np.column_stack((direction_valid, direction_valid, height_valid)),
        )
    count = len(segment.time)
    result = np.full((count, 1), np.nan, dtype=float)
    result_valid = np.zeros((count, 1), dtype=bool)
    displacement = np.diff(np.asarray(segment.state, dtype=float), axis=0)
    for index in range(1, count):
        distance = float(np.linalg.norm(displacement[index - 1]))
        if (
            surface_valid[index]
            and surface_valid[index - 1]
            and velocity_valid[index]
            and distance > _EPSILON
        ):
            speed = float(np.linalg.norm(velocity[index]))
            direction = displacement[index - 1] / distance
            directional_curvature = float(
                np.dot(gradient[index] - gradient[index - 1], direction) / distance
            )
            result[index, 0] = speed * speed * directional_curvature
            result_valid[index, 0] = True
    return InteractionBatch((), result, result_valid)


def _geometry_motion(
    definition: InteractionDefinition,
    segment: Segment,
    source_values: Mapping[str, np.ndarray],
    source_valid: Mapping[str, np.ndarray],
) -> InteractionBatch:
    distance_log, east, north = (
        np.asarray(source_values[name], dtype=float)
        for name in definition.required_columns
    )
    distance_valid = np.asarray(
        source_valid[definition.required_columns[0]], dtype=bool
    ) & np.isfinite(distance_log)
    direction_valid = (
        np.asarray(source_valid[definition.required_columns[1]], dtype=bool)
        & np.asarray(source_valid[definition.required_columns[2]], dtype=bool)
        & np.isfinite(east)
        & np.isfinite(north)
    )
    distance = np.expm1(np.maximum(distance_log, 0.0))
    direction = np.column_stack((east, north))
    direction_norm = np.linalg.norm(direction, axis=1)
    direction_valid &= direction_norm > _EPSILON
    direction = np.divide(
        direction,
        direction_norm[:, None],
        out=np.zeros_like(direction),
        where=direction_norm[:, None] > _EPSILON,
    )
    velocity, velocity_valid, elapsed = _causal_velocity(segment)
    speed = np.linalg.norm(velocity, axis=1)
    approach = np.einsum("ij,ij->i", velocity, direction)
    tangent = np.abs(
        velocity[:, 0] * direction[:, 1] - velocity[:, 1] * direction[:, 0]
    )
    count = len(segment.time)
    values = np.full((count, 6), np.nan, dtype=float)
    valid = np.zeros((count, 6), dtype=bool)
    current = distance_valid & direction_valid & velocity_valid
    values[current, 0] = approach[current]
    values[current, 1] = tangent[current]
    valid[current, :2] = True
    moving = current & (speed > _EPSILON)
    values[moving, 2] = approach[moving] / speed[moving]
    valid[moving, 2] = True
    normalized = distance_valid & velocity_valid & (speed > _EPSILON)
    values[normalized, 5] = distance[normalized] / (60.0 * speed[normalized])
    valid[normalized, 5] = True
    approaching = current & (approach > _EPSILON)
    values[approaching, 4] = distance[approaching] / approach[approaching]
    valid[approaching, 4] = True
    for index in range(1, count):
        dt = elapsed[index - 1]
        if distance_valid[index] and distance_valid[index - 1] and dt > 0:
            values[index, 3] = (distance[index] - distance[index - 1]) / dt
            valid[index, 3] = True
    return InteractionBatch((), values, valid)


def _categorical_speed(
    definition: InteractionDefinition,
    segment: Segment,
    source_values: Mapping[str, np.ndarray],
    source_valid: Mapping[str, np.ndarray],
) -> InteractionBatch:
    categories = np.column_stack(
        [np.asarray(source_values[name], dtype=float) for name in definition.required_columns]
    )
    category_valid = np.logical_and.reduce(
        [np.asarray(source_valid[name], dtype=bool) for name in definition.required_columns]
    )
    velocity, velocity_valid, _ = _causal_velocity(segment)
    speed = np.linalg.norm(velocity, axis=1)
    usable = category_valid & velocity_valid & np.isfinite(categories).all(axis=1)
    values = categories * speed[:, None]
    values[~usable] = np.nan
    return InteractionBatch(
        (), values, np.repeat(usable[:, None], categories.shape[1], axis=1)
    )


def _categorical_transition(
    definition: InteractionDefinition,
    source_values: Mapping[str, np.ndarray],
    source_valid: Mapping[str, np.ndarray],
) -> InteractionBatch:
    categories = np.column_stack(
        [np.asarray(source_values[name], dtype=float) for name in definition.required_columns]
    )
    row_valid = np.logical_and.reduce(
        [np.asarray(source_valid[name], dtype=bool) for name in definition.required_columns]
    ) & np.isfinite(categories).all(axis=1)
    values = np.full((len(categories), 1), np.nan, dtype=float)
    valid = np.zeros((len(categories), 1), dtype=bool)
    groups = np.argmax(categories, axis=1)
    usable = row_valid[1:] & row_valid[:-1]
    values[1:, 0][usable] = (groups[1:][usable] != groups[:-1][usable]).astype(float)
    valid[1:, 0] = usable
    return InteractionBatch((), values, valid)


def _field_time_difference(
    definition: InteractionDefinition,
    segment: Segment,
    source_values: Mapping[str, np.ndarray],
    source_valid: Mapping[str, np.ndarray],
) -> InteractionBatch:
    raw = np.column_stack(
        [np.asarray(source_values[name], dtype=float) for name in definition.required_columns]
    )
    raw_valid = np.column_stack(
        [np.asarray(source_valid[name], dtype=bool) for name in definition.required_columns]
    )
    values = np.full_like(raw, np.nan, dtype=float)
    valid = np.zeros_like(raw_valid, dtype=bool)
    elapsed = np.diff(np.asarray(segment.time, dtype=float))
    for index in range(1, len(raw)):
        dt = elapsed[index - 1]
        usable = raw_valid[index] & raw_valid[index - 1] & (dt > 0)
        values[index, usable] = (raw[index, usable] - raw[index - 1, usable]) / dt
        valid[index, usable] = True
    return InteractionBatch((), values, valid)


def evaluate_interactions(
    segment: Segment,
    source_values: Mapping[str, np.ndarray],
    source_valid: Mapping[str, np.ndarray],
    interaction_ids: Sequence[str],
) -> InteractionBatch:
    """Evaluate ordered interactions using only the visible trajectory prefix."""

    ids = tuple(str(value) for value in interaction_ids)
    validate_interaction_selection(ids, tuple(source_values))
    columns = interaction_output_columns(ids)
    if not ids:
        empty = np.empty((len(segment.time), 0), dtype=float)
        return InteractionBatch(columns, empty, empty.astype(bool))
    batches: list[InteractionBatch] = []
    for interaction_id in ids:
        definition = INTERACTION_DEFINITIONS[interaction_id]
        if definition.operator == "history_motion":
            batch = _history_motion(segment)
        elif definition.operator == "history_acceleration":
            batch = _history_acceleration(segment)
        elif definition.operator.startswith("surface_"):
            batch = _surface_values(definition, segment, source_values, source_valid)
        elif definition.operator == "geometry_motion":
            batch = _geometry_motion(definition, segment, source_values, source_valid)
        elif definition.operator == "categorical_speed":
            batch = _categorical_speed(definition, segment, source_values, source_valid)
        elif definition.operator == "categorical_transition":
            batch = _categorical_transition(definition, source_values, source_valid)
        elif definition.operator == "field_time_difference":
            batch = _field_time_difference(
                definition, segment, source_values, source_valid
            )
        else:  # pragma: no cover - registry construction owns this invariant
            raise PIRC21InteractionError(
                f"unsupported interaction operator: {definition.operator}"
            )
        expected = len(definition.output_columns)
        if batch.values.shape != (len(segment.time), expected) or batch.valid.shape != (
            len(segment.time),
            expected,
        ):
            raise PIRC21InteractionError(
                f"interaction output shape mismatch: {interaction_id}"
            )
        batches.append(batch)
    return InteractionBatch(
        columns,
        np.concatenate([item.values for item in batches], axis=1),
        np.concatenate([item.valid for item in batches], axis=1),
    )


__all__ = [
    "INTERACTION_DEFINITIONS",
    "INTERACTION_REGISTRY_VERSION",
    "InteractionBatch",
    "InteractionColumn",
    "InteractionDefinition",
    "PIRC21InteractionError",
    "evaluate_interactions",
    "interaction_output_columns",
    "interaction_registry_fingerprint",
    "interaction_registry_record",
    "validate_interaction_selection",
]
