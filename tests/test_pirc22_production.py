from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.nex326.pirc20_adapter import PIRC20Segment
from experiments.nex326.pirc21_interactions import interaction_output_columns
from experiments.pirc22 import production
from experiments.pirc22.representations import (
    _COMPOSITIONS,
    _VARIANT_DIMENSIONS,
    load_representation_matrix,
)
from experiments.terrain_benchmark import BenchmarkFold


PRODUCTION_PROVIDER_SHA256 = (
    "3fd386c4fa746af876ec4f09bbae2152d44bfa47025ba2096b68576e8a08e971"
)


def _provider_and_segments():
    matrix = load_representation_matrix()
    provider = object.__new__(production.ProductionProvider)
    selected_variants = production.ordered_union(
        [candidate.variant_ids for candidate in matrix.candidates]
    )
    selected_compositions = production.ordered_union(
        [candidate.composition_ids for candidate in matrix.candidates]
    )
    provider._variant_columns = {
        variant_id: tuple(
            f"variant__{variant_id.replace('.', '_')}__{index}"
            for index in range(_VARIANT_DIMENSIONS[variant_id])
        )
        for variant_id in selected_variants
    }
    provider._composition_columns = {
        composition_id: tuple(
            f"composition__{composition_id.replace('.', '_')}__{index}"
            for index in range(_COMPOSITIONS[composition_id][0])
        )
        for composition_id in selected_compositions
    }
    feature_names = {
        name
        for names in (
            *provider._variant_columns.values(),
            *provider._composition_columns.values(),
        )
        for name in names
    }
    feature_names.update(
        interaction_output_columns(
            production.ordered_union(
                [candidate.interaction_ids for candidate in matrix.candidates]
            )
        )
    )

    def segment(segment_id: str, block_id: str, offset: float) -> PIRC20Segment:
        time = np.arange(4, dtype=float)
        conditions = {
            "terrain_elevation": np.arange(4, dtype=float) + 100.0 + offset,
            "terrain_slope": np.arange(4, dtype=float) / 10.0 + offset,
        }
        for index, name in enumerate(sorted(feature_names), start=1):
            conditions[f"pirc21:{name}"] = (
                np.arange(4, dtype=float) + index + offset
            )
            conditions[f"pirc21:{name}__valid"] = np.ones(4, dtype=bool)
        return PIRC20Segment(
            segment_id=segment_id,
            source_domain="human",
            region="fixture",
            time=time,
            state=np.column_stack((time + offset, 2.0 * time - offset)),
            conditions=conditions,
            has_terrain=True,
            independent_block_id=block_id,
        )

    train = (segment("train", "train-block", 0.0),)
    validation = (segment("validation", "validation-block", 0.5),)
    provider.train_states, provider.train_targets, provider.train_row_blocks = (
        production.transition_targets(train)
    )
    return provider, matrix, train, validation


def test_versioned_provider_is_the_exact_source_used_by_production_run():
    assert production.file_hash(Path(production.__file__).resolve()) == (
        PRODUCTION_PROVIDER_SHA256
    )


def test_union_selection_and_every_registered_candidate_dimension_are_constructible():
    provider, matrix, train, validation = _provider_and_segments()
    selection = production.union_selection(matrix)

    assert selection.variant_ids == production.ordered_union(
        [candidate.variant_ids for candidate in matrix.candidates]
    )
    assert selection.composition_ids == production.ordered_union(
        [candidate.composition_ids for candidate in matrix.candidates]
    )
    assert selection.interaction_ids == production.ordered_union(
        [candidate.interaction_ids for candidate in matrix.candidates]
    )
    assert selection.include_validity_indicators

    for candidate in matrix.candidates:
        train_features, validation_features, train_tokens, validation_tokens = (
            provider._candidate_features(candidate, train, validation)
        )
        embedding_dim = (
            candidate.categorical_embedding.embedding_dim
            if candidate.categorical_embedding
            else 0
        )
        expected_continuous = candidate.model_input_dim - embedding_dim
        assert train_features.shape == (3, expected_continuous)
        assert validation_features.shape == (3, expected_continuous)
        if candidate.categorical_embedding:
            assert train_tokens is not None and train_tokens.shape == (3,)
            assert validation_tokens is not None and validation_tokens.shape == (3,)
        else:
            assert train_tokens is None
            assert validation_tokens is None


def test_prepare_binds_control_and_problem_identities_to_fold_and_features():
    provider, matrix, train, validation = _provider_and_segments()
    provider.train_segments = train
    provider.validation_by_id = {validation[0].segment_id: validation[0]}
    provider.boundary = SimpleNamespace(benchmark_data_identity_sha256="b" * 64)
    provider.base_identity = "a" * 64
    provider.base_weights = np.zeros((3, 2), dtype=float)
    fold = BenchmarkFold(
        fold_id="fold-00",
        validation_block_ids=("validation-block",),
        validation_segment_ids=("validation",),
        identity_sha256="f" * 64,
    )

    prepared = provider.prepare(
        matrix.candidate("R12-registered-compositions"), fold
    )

    assert prepared.train_features.shape == (3, 56)
    assert prepared.validation_features.shape == (3, 56)
    assert prepared.train_block_ids == ("train-block",)
    assert prepared.validation_block_ids == ("validation-block",)
    assert prepared.validation_row_block_ids == ("validation-block",) * 3
    assert len(prepared.control_identity_sha256) == 64
    assert len(prepared.problem_identity_sha256) == 64
    assert prepared.problem_identity_sha256 != prepared.control_identity_sha256
