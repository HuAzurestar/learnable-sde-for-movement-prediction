"""Validated PIRC-17 handoff for the immutable PIRC-22 selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .conditioners import CONDITIONER_BY_ID, TrainingConfig
from .representations import RepresentationMatrix, load_representation_matrix


CONSUMER_SCHEMA_VERSION = "pirc22-benchmark-selection-consumer-v1"
ABLATION_SCHEMA_VERSION = "pirc17-terrain-ablation-configs-v1"
DEFAULT_BINDING_PATH = Path(__file__).with_name(
    "benchmark_selection.consumer.json"
)

FACTOR_GROUPS = {
    "road": ("road.distance_log1p", "road.direction"),
    "river": ("river.distance_log1p", "river.direction"),
    "worldcover": ("worldcover.grouped",),
    "surface": ("surface.orientation",),
    "history": ("history.direction",),
}
COMPOSITION_DEPENDENCIES = {
    "road.log_distance_x_direction": (
        "road.distance_log1p",
        "road.direction",
    ),
    "river.log_distance_x_direction": (
        "river.distance_log1p",
        "river.direction",
    ),
    "worldcover.grouped_x_road_distance": (
        "worldcover.grouped",
        "road.distance_log1p",
    ),
    "surface.orientation_x_history": (
        "surface.orientation",
        "history.direction",
    ),
}


class BenchmarkConsumerError(ValueError):
    """A frozen selection binding is changed or cannot drive PIRC-17."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_benchmark_selection_binding(
    path: str | Path = DEFAULT_BINDING_PATH,
    *,
    matrix: RepresentationMatrix | None = None,
) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkConsumerError("cannot read BenchmarkSelection binding") from error
    return validate_benchmark_selection_binding(payload, matrix=matrix)


def validate_benchmark_selection_binding(
    payload: object,
    *,
    matrix: RepresentationMatrix | None = None,
) -> dict[str, object]:
    """Apply the original frozen contract to an already admitted JSON value."""
    if not isinstance(payload, dict):
        raise BenchmarkConsumerError("BenchmarkSelection binding is not an object")
    identity_payload = dict(payload)
    identity = identity_payload.pop("consumer_identity_sha256", None)
    if (
        payload.get("schema_version") != CONSUMER_SCHEMA_VERSION
        or identity != _canonical_hash(identity_payload)
    ):
        raise BenchmarkConsumerError("BenchmarkSelection binding identity mismatch")
    if payload.get("status") != "selected" or payload.get("final_eval_read_count") != 0:
        raise BenchmarkConsumerError("BenchmarkSelection is not consumable")
    resolved_matrix = matrix or load_representation_matrix()
    if payload.get("matrix_identity_sha256") != resolved_matrix.matrix_identity_sha256:
        raise BenchmarkConsumerError("BenchmarkSelection matrix identity mismatch")
    configuration = payload.get("selected_configuration")
    if not isinstance(configuration, Mapping):
        raise BenchmarkConsumerError("selected configuration is absent")
    candidate = resolved_matrix.candidate(str(configuration.get("candidate_id", "")))
    conditioner_id = str(configuration.get("conditioner_id", ""))
    conditioner = CONDITIONER_BY_ID.get(conditioner_id)
    if conditioner is None:
        raise BenchmarkConsumerError("selected conditioner is unknown")
    if (
        list(candidate.variant_ids) != configuration.get("variant_ids")
        or list(candidate.composition_ids) != configuration.get("composition_ids")
        or list(candidate.interaction_ids) != configuration.get("interaction_ids")
        or candidate.model_input_dim != configuration.get("model_input_dim")
        or list(conditioner.hidden_widths) != configuration.get("hidden_widths")
        or conditioner.layer_count != configuration.get("layer_count")
    ):
        raise BenchmarkConsumerError("selected configuration changed from its matrix")
    training = configuration.get("training_config")
    if not isinstance(training, Mapping):
        raise BenchmarkConsumerError("selected training budget is absent")
    try:
        TrainingConfig(**training).validate()
    except (TypeError, ValueError) as error:
        raise BenchmarkConsumerError("selected training budget is invalid") from error
    return payload


def _configuration(
    *,
    configuration_id: str,
    kind: str,
    variants: tuple[str, ...],
    selected: Mapping[str, object],
    factor_id: str | None = None,
) -> dict[str, object]:
    selected_compositions = tuple(str(value) for value in selected["composition_ids"])
    variant_set = set(variants)
    compositions = tuple(
        value
        for value in selected_compositions
        if set(COMPOSITION_DEPENDENCIES[value]) <= variant_set
    )
    return {
        "configuration_id": configuration_id,
        "kind": kind,
        "factor_id": factor_id,
        "variant_ids": list(variants),
        "composition_ids": list(compositions),
        "interaction_ids": list(selected["interaction_ids"]),
        "conditioner_id": selected["conditioner_id"],
        "hidden_widths": list(selected["hidden_widths"]),
        "training_config": dict(selected["training_config"]),
    }


def build_pirc17_ablation_configs(
    binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    frozen = dict(binding or load_benchmark_selection_binding())
    selected = frozen.get("selected_configuration")
    if not isinstance(selected, Mapping):
        raise BenchmarkConsumerError("selected configuration is absent")
    all_variants = tuple(str(value) for value in selected["variant_ids"])
    configurations = [
        _configuration(
            configuration_id="base",
            kind="base",
            variants=(),
            selected=selected,
        ),
        _configuration(
            configuration_id="all-terrain",
            kind="all_terrain",
            variants=all_variants,
            selected=selected,
        ),
    ]
    for factor_id, variants in FACTOR_GROUPS.items():
        configurations.append(
            _configuration(
                configuration_id=f"loo-{factor_id}",
                kind="leave_one_out",
                factor_id=factor_id,
                variants=tuple(value for value in all_variants if value not in variants),
                selected=selected,
            )
        )
    for factor_id, variants in FACTOR_GROUPS.items():
        configurations.append(
            _configuration(
                configuration_id=f"lio-{factor_id}",
                kind="leave_one_in",
                factor_id=factor_id,
                variants=tuple(value for value in all_variants if value in variants),
                selected=selected,
            )
        )
    result: dict[str, object] = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "source_selection_identity_sha256": frozen[
            "source_selection_identity_sha256"
        ],
        "selection_version": frozen["selection_version"],
        "configuration_count": len(configurations),
        "configurations": configurations,
        "retuning_allowed": False,
    }
    result["configuration_identity_sha256"] = _canonical_hash(result)
    return result


__all__ = [
    "ABLATION_SCHEMA_VERSION",
    "BenchmarkConsumerError",
    "CONSUMER_SCHEMA_VERSION",
    "DEFAULT_BINDING_PATH",
    "FACTOR_GROUPS",
    "build_pirc17_ablation_configs",
    "load_benchmark_selection_binding",
    "validate_benchmark_selection_binding",
]
