from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.nex326.pirc21_ablation import (
    PIRC21AblationError,
    build_ablation_plan,
)


def _column(name: str) -> dict[str, object]:
    return {"name": name, "dtype": "float32", "unit": "1", "role": name}


def _variant(
    variant_id: str, factor_id: str, outputs: tuple[str, ...]
) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "factor_id": factor_id,
        "output_columns": [_column(name) for name in outputs],
        "output_dim": len(outputs),
    }


def _spec() -> dict[str, object]:
    return {
        "feature_spec_id": "ablation-fixture-v1",
        "processing_version": "fixture-v1",
        "factors": [
            {"factor_id": "dem_surface"},
            {"factor_id": "overture_road"},
            {"factor_id": "historical_motion"},
        ],
        "variants": [
            _variant(
                "surface.orientation",
                "dem_surface",
                (
                    "v_surface_slope_rad",
                    "v_surface_aspect_sin",
                    "v_surface_aspect_cos",
                ),
            ),
            _variant(
                "road.distance_log1p",
                "overture_road",
                ("v_road_distance_log1p",),
            ),
            _variant(
                "road.direction",
                "overture_road",
                ("v_road_direction_east", "v_road_direction_north"),
            ),
            _variant(
                "history.direction",
                "historical_motion",
                ("v_history_direction_east", "v_history_direction_north"),
            ),
        ],
    }


FULL_VARIANTS = (
    "surface.orientation",
    "road.distance_log1p",
    "road.direction",
    "history.direction",
)
FULL_INTERACTIONS = ("history.motion", "surface.velocity", "road.motion")


def test_plan_contains_paired_interactions_and_leave_one_factor_out_arms():
    plan = build_ablation_plan(
        _spec(),
        full_variant_ids=FULL_VARIANTS,
        full_interaction_ids=FULL_INTERACTIONS,
    )
    arms = {item["arm_id"]: item for item in plan["arms"]}

    assert plan["counts"] == {
        "factors": 3,
        "registered_variants": 4,
        "selected_full_variants": 4,
        "selected_full_interactions": 3,
        "arms": 14,
    }
    full = arms["F00-full-mathematical-inputs"]
    assert full["numeric_dim"] == 20
    assert full["model_condition_dim"] == 40

    road = next(
        item
        for item in plan["arms"]
        if item["kind"] == "single_interaction"
        and item["interaction_ids"] == ["road.motion"]
    )
    reference = arms[road["comparison_arm_id"]]
    assert road["variant_ids"] == ["road.distance_log1p", "road.direction"]
    assert reference["variant_ids"] == road["variant_ids"]
    assert reference["interaction_ids"] == []

    without_road = next(
        item
        for item in plan["arms"]
        if item.get("omitted_factor_id") == "overture_road"
    )
    assert not {"road.distance_log1p", "road.direction"}.intersection(
        without_road["variant_ids"]
    )
    assert "road.motion" not in without_road["interaction_ids"]
    assert without_road["comparison_arm_id"] == "F00-full-mathematical-inputs"


def test_plan_identity_and_order_are_deterministic():
    first = build_ablation_plan(
        _spec(),
        full_variant_ids=FULL_VARIANTS,
        full_interaction_ids=FULL_INTERACTIONS,
    )
    second = build_ablation_plan(
        deepcopy(_spec()),
        full_variant_ids=FULL_VARIANTS,
        full_interaction_ids=FULL_INTERACTIONS,
    )

    assert first == second
    assert len(first["plan_sha256"]) == 64
    assert first["arms"][0]["arm_id"] == "B00-no-pirc21"
    assert first["arms"][-1]["omitted_factor_id"] == "historical_motion"


def test_plan_rejects_missing_or_ambiguous_interaction_dependencies():
    spec = _spec()
    spec["variants"] = [
        item for item in spec["variants"] if item["variant_id"] != "road.direction"
    ]
    with pytest.raises(PIRC21AblationError, match="unavailable outputs"):
        build_ablation_plan(
            spec,
            full_variant_ids=("road.distance_log1p",),
            full_interaction_ids=("road.motion",),
        )

    duplicate = _spec()
    duplicate["variants"].append(
        _variant(
            "road.direction.copy",
            "overture_road",
            ("v_road_direction_east",),
        )
    )
    with pytest.raises(PIRC21AblationError, match="multiple variant owners"):
        build_ablation_plan(
            duplicate,
            full_variant_ids=FULL_VARIANTS,
            full_interaction_ids=FULL_INTERACTIONS,
        )
