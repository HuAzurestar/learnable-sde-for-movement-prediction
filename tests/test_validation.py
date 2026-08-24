"""P1 数据 fail-fast 校验层测试。

运行：python -m pytest tests/test_validation.py
"""

import pytest
import torch

from data.loader import Segment
from data.validation import (
    DataValidationError,
    ensure_finite,
    ensure_monotonic_increasing,
    ensure_shapes_match,
    ensure_positive,
    validate_segment,
)


def _seg(t=None, x=None, dt=60.0):
    t = torch.tensor([0.0, 60.0, 120.0]) if t is None else t
    x = torch.zeros(3, 2) if x is None else x
    return Segment(t=t, x=x, dt=dt)


def test_data_validation_error_is_valueerror():
    assert issubclass(DataValidationError, ValueError)


def test_ensure_finite_raises_on_nan():
    with pytest.raises(DataValidationError):
        ensure_finite(torch.tensor([[0.0], [float("nan")]]), name="x")


def test_ensure_finite_raises_on_inf():
    with pytest.raises(DataValidationError):
        ensure_finite(torch.tensor([[0.0], [float("inf")]]), name="x")


def test_ensure_finite_passes_clean():
    ensure_finite(torch.tensor([[0.0], [1.0]]), name="x")


def test_ensure_monotonic_raises_on_duplicate():
    with pytest.raises(DataValidationError):
        ensure_monotonic_increasing(torch.tensor([0.0, 60.0, 60.0]), name="t")


def test_ensure_monotonic_raises_on_decrease():
    with pytest.raises(DataValidationError):
        ensure_monotonic_increasing(torch.tensor([0.0, 60.0, 30.0]), name="t")


def test_ensure_shapes_match_raises_on_mismatch():
    with pytest.raises(DataValidationError):
        ensure_shapes_match(torch.zeros(3, 2), torch.zeros(4), name_a="x", name_b="t")


def test_ensure_positive_raises_on_zero():
    with pytest.raises(DataValidationError):
        ensure_positive(torch.tensor([1.0, 0.0]), name="dt")


def test_validate_segment_passes_clean():
    validate_segment(_seg())


def test_validate_segment_raises_nan_x():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [float("nan"), 0.0]])
    with pytest.raises(DataValidationError):
        validate_segment(_seg(x=x))


def test_validate_segment_raises_nonmonotonic_t():
    with pytest.raises(DataValidationError):
        validate_segment(_seg(t=torch.tensor([0.0, 120.0, 60.0])))


def test_validate_segment_raises_shape_mismatch():
    with pytest.raises(DataValidationError):
        validate_segment(Segment(t=torch.zeros(3), x=torch.zeros(4, 2)))


def test_validate_segment_raises_too_short():
    with pytest.raises(DataValidationError):
        validate_segment(_seg(t=torch.zeros(1), x=torch.zeros(1, 2)))


def test_validate_segment_raises_nonpositive_dt():
    with pytest.raises(DataValidationError):
        validate_segment(_seg(dt=0.0))


def test_em_fit_raises_on_empty_segments():
    from estimation.base import FitContext
    from estimation.em import SegmentEM
    from estimation.em import SegmentEMData
    from models.segment_constant import SegmentConstantSDE

    with pytest.raises(DataValidationError):
        SegmentEM().fit(
            SegmentConstantSDE(n_modes=1),
            SegmentEMData((), ()),
            FitContext(
                torch.Generator().manual_seed(1),
                torch.device("cpu"),
                torch.float64,
            ),
        )
