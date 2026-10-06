"""Immutable, schema-validated PIRC-22 representation preregistration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.nex326.pirc21_adapter import FeatureSelection


MATRIX_SCHEMA_VERSION = "pirc22-representation-matrix-v1"
LOCK_SCHEMA_VERSION = "pirc22-representation-matrix-lock-v1"
SOURCE_FEATURE_SPEC_ID = "pirc21-p0-feature-spec-v1"
SOURCE_FEATURE_SPEC_SHA256 = (
    "e53857fd2749da76c3bf856537d3ff8e30c7c2989d2d585dcc068ac913058616"
)
FROZEN_MATRIX_SHA256 = (
    "94e91ad3bfe9a5421a697dd8c7f6d964f61bdf95478a7a4a68175707abaa9d4f"
)
DEFAULT_MATRIX_PATH = Path(__file__).with_name("representation_matrix.json")

REQUIRED_COVERAGE = frozenset(
    {
        "no_terrain",
        "legacy_scalar",
        "local_statistics",
        "source_specific_distance",
        "direction_vector",
        "surface_normal",
        "motion_projection",
        "worldcover_one_hot",
        "worldcover_embedding",
        "registered_combination",
        "transform_raw",
        "transform_standardized",
        "transform_log1p",
        "transform_clipped",
        "transform_fixed_rbf",
    }
)

_VARIANT_DIMENSIONS = {
    "elevation.absolute": 1,
    "elevation.local": 2,
    "elevation.absolute_standardized": 1,
    "elevation.local_standardized": 2,
    "surface.orientation": 6,
    "surface.shape": 2,
    "worldcover.grouped": 4,
    "worldcover.embedding": 1,
    "road.distance_raw": 1,
    "road.distance_log1p": 1,
    "road.direction": 2,
    "path.distance_raw": 1,
    "path.distance_log1p": 1,
    "path.direction": 2,
    "rail.distance_raw": 1,
    "rail.distance_log1p": 1,
    "rail.direction": 2,
    "river.distance_raw": 1,
    "river.distance_log1p": 1,
    "river.direction": 2,
    "jrc.distance_log1p": 1,
    "jrc.direction": 2,
    "navigable_water.distance_raw": 1,
    "navigable_water.distance_log1p": 1,
    "navigable_water.direction": 2,
    "ridge.distance_raw": 1,
    "ridge.distance_log1p": 1,
    "ridge.direction": 2,
    "cliff.distance_raw": 1,
    "cliff.distance_log1p": 1,
    "cliff.direction": 2,
    "history.direction": 2,
}

_COMPOSITIONS = {
    "road.log_distance_x_direction": (
        2,
        ("road.distance_log1p", "road.direction"),
    ),
    "river.log_distance_x_direction": (
        2,
        ("river.distance_log1p", "river.direction"),
    ),
    "worldcover.grouped_x_road_distance": (
        4,
        ("worldcover.grouped", "road.distance_log1p"),
    ),
    "surface.orientation_x_history": (
        2,
        ("surface.orientation", "history.direction"),
    ),
}

_INTERACTIONS = {
    "road.motion": (6, ("road.distance_log1p", "road.direction")),
    "river.motion": (6, ("river.distance_log1p", "river.direction")),
    "surface.velocity": (3, ("surface.orientation",)),
}

_AGGREGATIONS = frozenset({"mean", "std", "min", "max", "last", "valid_fraction"})


class RepresentationMatrixError(ValueError):
    """The representation preregistration is missing, changed, or inconsistent."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _read_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RepresentationMatrixError(f"cannot read JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise RepresentationMatrixError(f"JSON root must be an object: {path}")
    return payload


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RepresentationMatrixError(f"{field} must be a string list")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise RepresentationMatrixError(f"{field} must be ordered and unique")
    return result


@dataclass(frozen=True)
class DerivedTransform:
    transform: str
    source_variant_ids: tuple[str, ...]
    output_dim: int
    parameters: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class CategoricalEmbedding:
    source_variant_id: str
    vocabulary: str
    embedding_dim: int


@dataclass(frozen=True)
class RepresentationCandidate:
    candidate_id: str
    order: int
    family: str
    coverage_tags: tuple[str, ...]
    variant_ids: tuple[str, ...]
    composition_ids: tuple[str, ...]
    interaction_ids: tuple[str, ...]
    segment_aggregations: tuple[str, ...]
    legacy_condition_names: tuple[str, ...]
    include_base_outputs: bool
    derived_transforms: tuple[DerivedTransform, ...]
    categorical_embedding: CategoricalEmbedding | None
    feature_output_dim: int
    validity_indicator_dim: int
    model_input_dim: int

    def feature_selection(self) -> FeatureSelection:
        return FeatureSelection(
            variant_ids=self.variant_ids,
            composition_ids=self.composition_ids,
            interaction_ids=self.interaction_ids,
            segment_aggregations=self.segment_aggregations,
            include_validity_indicators=self.validity_indicator_dim > 0,
        )


@dataclass(frozen=True)
class RepresentationMatrix:
    matrix_id: str
    source_feature_spec_id: str
    source_feature_spec_sha256: str
    candidates: tuple[RepresentationCandidate, ...]
    matrix_identity_sha256: str

    def candidate(self, candidate_id: str) -> RepresentationCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise RepresentationMatrixError(f"unknown representation: {candidate_id}")

    def validate_feature_spec(self, feature_spec: Mapping[str, object]) -> None:
        if feature_spec.get("feature_spec_id") != self.source_feature_spec_id:
            raise RepresentationMatrixError("feature spec identity does not match matrix")
        if _canonical_hash(feature_spec) != self.source_feature_spec_sha256:
            raise RepresentationMatrixError("feature spec content does not match matrix")


def _derived_transform(
    raw: object, candidate_id: str, selected: set[str]
) -> DerivedTransform:
    if not isinstance(raw, Mapping):
        raise RepresentationMatrixError(
            f"derived transform must be an object: {candidate_id}"
        )
    transform = str(raw.get("transform", ""))
    source_ids = _strings(
        raw.get("source_variant_ids"),
        f"{candidate_id}.derived_transform.source_variant_ids",
    )
    if not set(source_ids) <= selected:
        raise RepresentationMatrixError(
            f"derived transform has unselected sources: {candidate_id}"
        )
    parameters = raw.get("parameters")
    if not isinstance(parameters, Mapping):
        raise RepresentationMatrixError(
            f"derived transform parameters are invalid: {candidate_id}"
        )
    output_dim = int(raw.get("output_dim", -1))
    source_dim = sum(_VARIANT_DIMENSIONS[value] for value in source_ids)
    if transform == "clipped_standardize":
        clip_sigma = parameters.get("clip_sigma")
        if not isinstance(clip_sigma, (int, float)) or isinstance(clip_sigma, bool) or clip_sigma <= 0:
            raise RepresentationMatrixError(f"invalid clip_sigma: {candidate_id}")
        expected = source_dim
    elif transform == "fixed_rbf":
        knots = parameters.get("knots_m")
        if (
            not isinstance(knots, list)
            or len(knots) < 3
            or any(not isinstance(value, (int, float)) for value in knots)
            or list(knots) != sorted(set(knots))
        ):
            raise RepresentationMatrixError(f"invalid fixed RBF knots: {candidate_id}")
        width = parameters.get("width_m")
        if not isinstance(width, (int, float)) or isinstance(width, bool) or width <= 0:
            raise RepresentationMatrixError(f"invalid fixed RBF width: {candidate_id}")
        expected = source_dim * len(knots)
    else:
        raise RepresentationMatrixError(
            f"unsupported derived transform {transform!r}: {candidate_id}"
        )
    if output_dim != expected:
        raise RepresentationMatrixError(
            f"derived transform dimension mismatch: {candidate_id}"
        )
    return DerivedTransform(
        transform=transform,
        source_variant_ids=source_ids,
        output_dim=output_dim,
        parameters=tuple(
            sorted((str(key), _freeze(value)) for key, value in parameters.items())
        ),
    )


def _candidate(raw: object) -> RepresentationCandidate:
    if not isinstance(raw, Mapping):
        raise RepresentationMatrixError("candidate must be an object")
    candidate_id = str(raw.get("candidate_id", ""))
    family = str(raw.get("family", ""))
    order = raw.get("order")
    if not candidate_id or not family or isinstance(order, bool) or not isinstance(order, int):
        raise RepresentationMatrixError("candidate identity/order is invalid")
    variants = _strings(raw.get("variant_ids"), f"{candidate_id}.variant_ids")
    compositions = _strings(
        raw.get("composition_ids"), f"{candidate_id}.composition_ids"
    )
    interactions = _strings(
        raw.get("interaction_ids"), f"{candidate_id}.interaction_ids"
    )
    aggregations = _strings(
        raw.get("segment_aggregations"), f"{candidate_id}.segment_aggregations"
    )
    legacy = _strings(
        raw.get("legacy_condition_names"), f"{candidate_id}.legacy_condition_names"
    )
    coverage = _strings(raw.get("coverage_tags"), f"{candidate_id}.coverage_tags")
    unknown_variants = set(variants) - set(_VARIANT_DIMENSIONS)
    unknown_compositions = set(compositions) - set(_COMPOSITIONS)
    unknown_interactions = set(interactions) - set(_INTERACTIONS)
    unknown_aggregations = set(aggregations) - _AGGREGATIONS
    if unknown_variants or unknown_compositions or unknown_interactions or unknown_aggregations:
        raise RepresentationMatrixError(
            f"candidate references unknown registry entries: {candidate_id}"
        )
    selected = set(variants)
    for composition_id in compositions:
        if not set(_COMPOSITIONS[composition_id][1]) <= selected:
            raise RepresentationMatrixError(
                f"composition has unselected dependencies: {candidate_id}"
            )
    for interaction_id in interactions:
        if not set(_INTERACTIONS[interaction_id][1]) <= selected:
            raise RepresentationMatrixError(
                f"interaction has unselected dependencies: {candidate_id}"
            )

    include_base = raw.get("include_base_outputs")
    if not isinstance(include_base, bool):
        raise RepresentationMatrixError(
            f"include_base_outputs must be boolean: {candidate_id}"
        )
    raw_derived = raw.get("derived_transforms")
    if not isinstance(raw_derived, list):
        raise RepresentationMatrixError(f"derived_transforms must be a list: {candidate_id}")
    derived = tuple(
        _derived_transform(item, candidate_id, selected) for item in raw_derived
    )
    raw_embedding = raw.get("categorical_embedding")
    embedding: CategoricalEmbedding | None = None
    if raw_embedding is not None:
        if not isinstance(raw_embedding, Mapping):
            raise RepresentationMatrixError(f"invalid categorical embedding: {candidate_id}")
        source = str(raw_embedding.get("source_variant_id", ""))
        vocabulary = str(raw_embedding.get("vocabulary", ""))
        embedding_dim = raw_embedding.get("embedding_dim")
        if (
            source != "worldcover.embedding"
            or source not in selected
            or not vocabulary
            or isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim < 1
        ):
            raise RepresentationMatrixError(f"invalid categorical embedding: {candidate_id}")
        embedding = CategoricalEmbedding(source, vocabulary, embedding_dim)

    base_dim = (
        sum(
            dimension
            for variant_id, dimension in _VARIANT_DIMENSIONS.items()
            if variant_id in selected and variant_id != "worldcover.embedding"
        )
        if include_base
        else 0
    )
    computed_feature_dim = (
        len(legacy)
        + base_dim
        + sum(_COMPOSITIONS[value][0] for value in compositions)
        + sum(_INTERACTIONS[value][0] for value in interactions)
        + sum(value.output_dim for value in derived)
        + (embedding.embedding_dim if embedding else 0)
    )
    computed_validity_dim = (
        base_dim
        + sum(_COMPOSITIONS[value][0] for value in compositions)
        + sum(_INTERACTIONS[value][0] for value in interactions)
        + sum(value.output_dim for value in derived)
        + (1 if embedding else 0)
    )
    recorded_feature_dim = raw.get("feature_output_dim")
    recorded_validity_dim = raw.get("validity_indicator_dim")
    recorded_model_dim = raw.get("model_input_dim")
    if (
        recorded_feature_dim != computed_feature_dim
        or recorded_validity_dim != computed_validity_dim
        or recorded_model_dim != computed_feature_dim + computed_validity_dim
    ):
        raise RepresentationMatrixError(
            f"recorded dimensions do not match registry calculation: {candidate_id}"
        )
    return RepresentationCandidate(
        candidate_id=candidate_id,
        order=order,
        family=family,
        coverage_tags=coverage,
        variant_ids=variants,
        composition_ids=compositions,
        interaction_ids=interactions,
        segment_aggregations=aggregations,
        legacy_condition_names=legacy,
        include_base_outputs=include_base,
        derived_transforms=derived,
        categorical_embedding=embedding,
        feature_output_dim=computed_feature_dim,
        validity_indicator_dim=computed_validity_dim,
        model_input_dim=computed_feature_dim + computed_validity_dim,
    )


def load_representation_matrix(
    path: str | Path = DEFAULT_MATRIX_PATH,
) -> RepresentationMatrix:
    source = Path(path).resolve()
    payload = _read_object(source)
    lock = _read_object(source.with_suffix(".lock.json"))
    return validate_representation_matrix(payload, lock)


def validate_representation_matrix(payload: object, lock: object) -> RepresentationMatrix:
    """Apply the same frozen matrix/lock contract without reopening inputs."""
    if not isinstance(payload, dict) or not isinstance(lock, dict):
        raise RepresentationMatrixError("matrix and lock JSON roots must be objects")
    if payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise RepresentationMatrixError("unsupported representation matrix schema")
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise RepresentationMatrixError("unsupported representation matrix lock schema")
    matrix_id = str(payload.get("matrix_id", ""))
    if (
        not matrix_id
        or payload.get("status") != "FROZEN"
        or payload.get("mutation_policy") != "new_matrix_id_required"
    ):
        raise RepresentationMatrixError("matrix must be frozen and versioned")
    identity = _canonical_hash(payload)
    if (
        lock.get("matrix_id") != matrix_id
        or lock.get("matrix_identity_sha256") != identity
        or identity != FROZEN_MATRIX_SHA256
    ):
        raise RepresentationMatrixError("representation matrix content lock mismatch")
    if (
        payload.get("source_feature_spec_id") != SOURCE_FEATURE_SPEC_ID
        or payload.get("source_feature_spec_sha256") != SOURCE_FEATURE_SPEC_SHA256
    ):
        raise RepresentationMatrixError("matrix is not bound to the frozen feature spec")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise RepresentationMatrixError("representation matrix has no candidates")
    candidates = tuple(_candidate(item) for item in raw_candidates)
    ids = [candidate.candidate_id for candidate in candidates]
    orders = [candidate.order for candidate in candidates]
    if len(ids) != len(set(ids)) or orders != list(range(len(candidates))):
        raise RepresentationMatrixError("candidate IDs or complexity order are invalid")
    coverage = {tag for candidate in candidates for tag in candidate.coverage_tags}
    missing = REQUIRED_COVERAGE - coverage
    if missing:
        raise RepresentationMatrixError(
            f"representation matrix misses mandatory coverage: {sorted(missing)}"
        )
    no_terrain = [
        candidate for candidate in candidates if "no_terrain" in candidate.coverage_tags
    ]
    if len(no_terrain) != 1 or no_terrain[0].model_input_dim != 0:
        raise RepresentationMatrixError("matrix requires one zero-width no-terrain row")
    return RepresentationMatrix(
        matrix_id=matrix_id,
        source_feature_spec_id=SOURCE_FEATURE_SPEC_ID,
        source_feature_spec_sha256=SOURCE_FEATURE_SPEC_SHA256,
        candidates=candidates,
        matrix_identity_sha256=identity,
    )


__all__ = [
    "DEFAULT_MATRIX_PATH",
    "FROZEN_MATRIX_SHA256",
    "MATRIX_SCHEMA_VERSION",
    "REQUIRED_COVERAGE",
    "RepresentationCandidate",
    "RepresentationMatrix",
    "RepresentationMatrixError",
    "SOURCE_FEATURE_SPEC_ID",
    "SOURCE_FEATURE_SPEC_SHA256",
    "load_representation_matrix",
    "validate_representation_matrix",
]
