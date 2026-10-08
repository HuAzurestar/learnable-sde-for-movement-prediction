"""Tiny array/metadata fixtures only; no model fitting or forecasts."""
import numpy as np
import pytest

from experiments.pirc17.feature_design_description import _OriginTransformReader, describe


def test_constant_masks_produce_raw_rank_deficiency():
    x = np.array([[0., 1.], [1., 0.], [2., -1.]])
    result = describe(x, np.ones_like(x, dtype=bool), ["a", "b"])
    assert result["constant_one_mask_columns"] == 2
    assert result["raw_columns_with_masks_and_intercept"] == 5
    assert result["raw_design_rank"] == 2  # b = 1-a, three identical constants
    assert result["raw_design_condition_number"] is None
    assert result["nonzero_spectrum_condition_number"] > 1


def test_mask_zero_placeholder_and_validity_fraction():
    x = np.array([[np.nan, 2.], [4., 2.], [6., 2.]])
    masks = np.array([[False, True], [True, True], [True, True]])
    result = describe(x, masks, ["a", "b"])
    a, b = result["columns"]
    assert a["valid_count"] == 2
    assert a["valid_fraction"] == pytest.approx(2/3)
    assert a["numeric_minimum"] == 0
    assert a["numeric_mean"] == pytest.approx(10/3)
    assert b["constant_numeric_column"]


def test_raw_full_rank_has_finite_condition():
    # Both masks vary independently; values are not mask duplicates.
    x = np.array([[0.,0.], [1.,0.], [2.,0.], [0.,1.], [0.,3.], [2.,4.]])
    masks = np.array([[False,False], [True,False], [True,False],
                      [False,True], [False,True], [True,True]])
    result = describe(x, masks, ["a", "b"])
    assert result["raw_design_rank"] == 5
    assert result["raw_design_condition_number"] > 1


@pytest.mark.parametrize("kind", ["empty", "wrong_shape", "non_boolean", "duplicate_names", "valid_nan"])
def test_invalid_design_rejected(kind):
    x = np.ones((3,2)); masks = np.ones((3,2), dtype=bool); names = ["a", "b"]
    if kind == "empty": x, masks = x[:0], masks[:0]
    if kind == "wrong_shape": masks = masks[:2]
    if kind == "non_boolean": masks = masks.astype(float)
    if kind == "duplicate_names": names = ["a", "a"]
    if kind == "valid_nan": x[0,0] = np.nan
    with pytest.raises(ValueError): describe(x, masks, names)


def test_diagnostic_reader_cannot_load_whole_snapshot_or_fit():
    # These methods must refuse before touching uninitialized metadata.
    reader = object.__new__(_OriginTransformReader)
    with pytest.raises(RuntimeError, match="general snapshot loader"):
        reader._load("final_eval")
    with pytest.raises(RuntimeError, match="cannot fit"):
        reader.fit()
