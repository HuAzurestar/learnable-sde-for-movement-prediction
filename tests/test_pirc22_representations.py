from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.pirc22.representations import (
    DEFAULT_MATRIX_PATH,
    FROZEN_MATRIX_SHA256,
    REQUIRED_COVERAGE,
    RepresentationMatrixError,
    SOURCE_FEATURE_SPEC_ID,
    load_representation_matrix,
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _copy_with_lock(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "representation_matrix.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lock = {
        "schema_version": "pirc22-representation-matrix-lock-v1",
        "matrix_id": payload["matrix_id"],
        "matrix_identity_sha256": _canonical_hash(payload),
    }
    path.with_suffix(".lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    return path


def test_frozen_matrix_is_complete_ordered_and_dimensioned():
    matrix = load_representation_matrix()

    assert matrix.matrix_identity_sha256 == FROZEN_MATRIX_SHA256
    assert matrix.source_feature_spec_id == SOURCE_FEATURE_SPEC_ID
    assert len(matrix.candidates) == 14
    assert [candidate.order for candidate in matrix.candidates] == list(range(14))
    assert {
        tag for candidate in matrix.candidates for tag in candidate.coverage_tags
    } >= REQUIRED_COVERAGE
    assert matrix.candidate("R00-no-terrain").model_input_dim == 0
    assert matrix.candidate("R06-source-distances-fixed-rbf").model_input_dim == 64
    assert matrix.candidate("R11-worldcover-embedding").model_input_dim == 5
    assert matrix.candidate("R13-all-registered-terrain").model_input_dim == 130
    for candidate in matrix.candidates:
        candidate.feature_selection().validate()


def test_same_matrix_id_cannot_delete_or_mutate_a_row_even_with_a_rewritten_lock(
    tmp_path,
):
    payload = json.loads(DEFAULT_MATRIX_PATH.read_text(encoding="utf-8"))
    payload["candidates"] = payload["candidates"][:-1]
    deleted = _copy_with_lock(tmp_path / "deleted", payload)
    with pytest.raises(RepresentationMatrixError, match="content lock mismatch"):
        load_representation_matrix(deleted)

    payload = json.loads(DEFAULT_MATRIX_PATH.read_text(encoding="utf-8"))
    payload["candidates"][1]["model_input_dim"] = 999
    mutated = _copy_with_lock(tmp_path / "mutated", payload)
    with pytest.raises(RepresentationMatrixError, match="content lock mismatch"):
        load_representation_matrix(mutated)


def test_matrix_rejects_a_different_feature_spec_identity():
    matrix = load_representation_matrix()

    with pytest.raises(RepresentationMatrixError, match="identity"):
        matrix.validate_feature_spec(
            {"feature_spec_id": "different", "variants": [], "compositions": []}
        )
