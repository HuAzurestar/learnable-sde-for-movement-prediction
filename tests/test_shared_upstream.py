"""Public metadata admission and existing adapter compatibility."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pirc25.upstream import (
    AdmissionError, PUBLIC_BINDINGS, affine_state, audit_inputs,
    canonical_hash, validate_binding,
)
from experiments.nex326.cohort import Segment
from experiments.nex326.phase_space import phase_space_state

ROOT = Path(__file__).resolve().parents[1]


def test_public_contracts_are_frozen_and_not_authorization():
    manifest = audit_inputs(ROOT, tuple(b.object_id for b in PUBLIC_BINDINGS))
    assert manifest["data_authorization"] == "none"
    assert manifest["fit_scope"] == "train-only"
    assert manifest["feature_adapter"] == "pirc21-psde-adapter-v2"
    assert len(manifest["objects"]) == 4
    assert manifest == audit_inputs(ROOT, tuple(b.object_id for b in PUBLIC_BINDINGS))


def test_missing_optional_terrain_does_not_block_affine(tmp_path):
    binding = next(b for b in PUBLIC_BINDINGS if b.object_id == "affine-4d")
    target = tmp_path / binding.path
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / binding.path).read_bytes())
    assert len(audit_inputs(tmp_path, ("affine-4d",))["objects"]) == 1
    with pytest.raises(AdmissionError, match="MISSING_INPUT"):
        audit_inputs(tmp_path, ("affine-4d", "terrain-selection"))


@pytest.mark.parametrize("change", ["hash", "schema", "status", "units", "role", "path"])
def test_reject_tampered_or_unauthorized_bindings(change):
    binding = next(b for b in PUBLIC_BINDINGS if b.object_id == "affine-4d")
    updates = {"hash": {"sha256": "0" * 64}, "schema": {"schema_version": "unknown"},
               "status": {"status": "qualified"}, "units": {"units": ("km",)},
               "role": {"role": "final-eval"}, "path": {"path": "../outside.json"}}
    with pytest.raises(AdmissionError):
        validate_binding(ROOT, replace(binding, **updates[change]))


def test_canonical_hash_survives_checkout_line_endings():
    assert canonical_hash(json.loads('{"a":1}\r\n')) == canonical_hash({"a": 1})
    with pytest.raises(ValueError):
        canonical_hash({"value": float("nan")})


def test_four_dimensional_adapter_matches_existing_irregular_time_fixture():
    time = np.array([0., 1., 3., 6., 10.])
    position = np.column_stack([time ** 2, 2 * time])
    segment = Segment("synthetic", "synthetic", "fixture", time, position, {}, False)
    expected = phase_space_state(segment)
    np.testing.assert_array_equal(affine_state(segment), expected)
    np.testing.assert_array_equal(expected[:, :2], position)
    np.testing.assert_array_equal(expected[1:, 2], [1., 4., 9., 16.])
    np.testing.assert_array_equal(expected[:, 3], np.full(5, 2.))
    with pytest.raises(AdmissionError):
        affine_state(segment, units=("km", "km", "km/h", "km/h"))


def test_unknown_and_duplicate_inputs_are_rejected():
    for ids in (("unknown",), ("affine-4d", "affine-4d")):
        with pytest.raises(AdmissionError):
            audit_inputs(ROOT, ids)
