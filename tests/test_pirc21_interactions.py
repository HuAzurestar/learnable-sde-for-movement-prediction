from __future__ import annotations

import numpy as np
import pytest

from experiments.nex326.cohort import Segment
from experiments.nex326.pirc21_interactions import (
    PIRC21InteractionError,
    evaluate_interactions,
    interaction_registry_fingerprint,
    interaction_registry_record,
    validate_interaction_selection,
)


def _segment() -> Segment:
    segment = Segment(
        segment_id="interaction-fixture",
        source_domain="human",
        region="fixture",
        time=np.asarray([0.0, 1.0, 2.0, 4.0]),
        state=np.asarray(
            [
                [0.0, 0.0],
                [3.0, 4.0],
                [6.0, 4.0],
                [6.0, 8.0],
            ]
        ),
        conditions={},
        has_terrain=False,
    )
    segment.validate()
    return segment


def _source() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = {
        "v_surface_slope_rad": np.arctan([0.5, 0.5, 1.0, 1.0]),
        "v_surface_aspect_sin": np.asarray([-1.0, -1.0, -1.0, -1.0]),
        "v_surface_aspect_cos": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "v_road_distance_log1p": np.log1p([10.0, 5.0, 2.0, 1.0]),
        "v_road_direction_east": np.asarray([0.6, 0.6, 1.0, 0.0]),
        "v_road_direction_north": np.asarray([0.8, 0.8, 0.0, 1.0]),
        "v_worldcover_built": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "v_worldcover_vegetated": np.asarray([0.0, 0.0, 1.0, 1.0]),
        "v_worldcover_water_wetland": np.zeros(4),
        "v_worldcover_bare_snow_unknown": np.zeros(4),
        "v_jrc_occurrence_percent": np.asarray([20.0, 25.0, 35.0, 35.0]),
        "v_jrc_seasonality_months": np.asarray([2.0, 2.0, 4.0, 8.0]),
    }
    valid = {name: np.ones(4, dtype=bool) for name in values}
    return values, valid


def test_registry_records_formula_dependencies_units_and_stable_identity():
    record = interaction_registry_record()
    by_id = {item["interaction_id"]: item for item in record["interactions"]}

    assert by_id["surface.velocity"]["formula"].endswith("dot(v,gradient)")
    assert by_id["road.motion"]["required_columns"] == [
        "v_road_distance_log1p",
        "v_road_direction_east",
        "v_road_direction_north",
    ]
    assert by_id["surface.directional_curvature"]["output_columns"][0]["unit"] == (
        "m/s2"
    )
    assert len(interaction_registry_fingerprint()) == 64


def test_selection_fails_closed_when_a_mathematical_dependency_is_absent():
    with pytest.raises(PIRC21InteractionError, match="requires unavailable columns"):
        validate_interaction_selection(
            ("road.motion",),
            ("v_road_distance_log1p",),
        )


def test_history_and_surface_interactions_use_only_causal_velocity():
    values, valid = _source()
    result = evaluate_interactions(
        _segment(),
        values,
        valid,
        (
            "history.motion",
            "history.acceleration",
            "surface.gradient",
            "surface.velocity",
            "surface.directional_curvature",
        ),
    )
    by_name = {
        name: (result.values[:, index], result.valid[:, index])
        for index, name in enumerate(result.columns)
    }

    assert not by_name["k_speed_mps"][1][0]
    assert by_name["k_speed_mps"][0][1:] == pytest.approx([5.0, 3.0, 2.0])
    assert by_name["k_acceleration_east_mps2"][0][2] == pytest.approx(0.0)
    assert by_name["k_acceleration_north_mps2"][0][2] == pytest.approx(-4.0)
    assert by_name["k_turn_rate_radps"][0][2] == pytest.approx(
        -np.arctan2(12.0, 9.0)
    )
    assert by_name["k_surface_gradient_east"][0] == pytest.approx(
        [0.5, 0.5, 1.0, 1.0]
    )
    assert by_name["k_surface_uphill_speed_mps"][0][1:] == pytest.approx(
        [3.0, 3.0, 0.0]
    )
    assert by_name["k_surface_contour_speed_mps"][0][1:] == pytest.approx(
        [4.0, 0.0, 2.0]
    )
    assert by_name["k_surface_height_rate_mps"][0][1:] == pytest.approx(
        [1.5, 3.0, 0.0]
    )
    # Gradient changes from (0.5, 0) to (1, 0) over a 3 m eastward step.
    assert by_name["k_surface_directional_curvature_mps2"][0][2] == pytest.approx(
        1.5
    )


def test_geometry_category_and_raster_rates_have_distinct_validity_masks():
    values, valid = _source()
    result = evaluate_interactions(
        _segment(),
        values,
        valid,
        (
            "road.motion",
            "worldcover.grouped_speed",
            "worldcover.transition",
            "jrc.attribute_rates",
        ),
    )
    by_name = {
        name: (result.values[:, index], result.valid[:, index])
        for index, name in enumerate(result.columns)
    }

    assert by_name["k_road_approach_speed_mps"][0][1] == pytest.approx(5.0)
    assert by_name["k_road_tangent_speed_mps"][0][1] == pytest.approx(0.0)
    assert by_name["k_road_heading_alignment"][0][1] == pytest.approx(1.0)
    assert by_name["k_road_distance_rate_mps"][0][1] == pytest.approx(-5.0)
    assert by_name["k_road_time_to_contact_s"][0][1] == pytest.approx(1.0)
    assert by_name["k_road_distance_over_60s_travel"][0][1] == pytest.approx(
        1.0 / 60.0
    )
    assert by_name["k_road_time_to_contact_s"][0][3] == pytest.approx(0.5)
    assert by_name["k_worldcover_built_speed_mps"][0][1] == pytest.approx(5.0)
    assert by_name["k_worldcover_transition"][0].tolist() == pytest.approx(
        [np.nan, 0.0, 1.0, 0.0], nan_ok=True
    )
    assert by_name["k_jrc_occurrence_rate_pctps"][0][1:] == pytest.approx(
        [5.0, 10.0, 0.0]
    )
    assert by_name["k_jrc_seasonality_rate_monthps"][0][1:] == pytest.approx(
        [0.0, 2.0, 2.0]
    )


def test_flat_surface_keeps_height_rate_but_has_no_uphill_or_contour_direction():
    values, valid = _source()
    values["v_surface_slope_rad"][1] = 0.0
    values["v_surface_aspect_sin"][1] = 0.0
    values["v_surface_aspect_cos"][1] = 0.0
    result = evaluate_interactions(
        _segment(), values, valid, ("surface.velocity",)
    )

    assert result.valid[1].tolist() == [False, False, True]
    assert np.isnan(result.values[1, 0])
    assert result.values[1, 2] == pytest.approx(0.0)


def test_missing_source_status_propagates_without_becoming_numeric_zero():
    values, valid = _source()
    valid["v_road_direction_east"][2] = False
    result = evaluate_interactions(
        _segment(), values, valid, ("road.motion",)
    )

    assert not result.valid[2, 0]
    assert np.isnan(result.values[2, 0])
    assert result.valid[3, 3]
    assert result.values[3, 3] == pytest.approx(-0.5)
