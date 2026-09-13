from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.nex326.dispersion_calibration import (
    DispersionCalibrationError,
    calibration_factor,
    scale_predictions,
)
from experiments.nex326.runner import Prediction, compute_metrics


def _prediction(index: int, target_radius: float) -> Prediction:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    samples = np.column_stack((np.cos(angles), np.sin(angles)))
    return Prediction(
        segment_id=f"segment-{index}",
        target=np.asarray([target_radius, 0.0]),
        samples=samples,
        prior_mean=None,
        prior_source=None,
        unbridged_prior_distance=None,
        bridged_prior_distance=None,
        bridge_path=None,
        bridge_diagnostics=None,
    )


def test_validation_factor_attains_target_coverage_without_moving_centers():
    predictions = tuple(
        _prediction(index, target_radius)
        for index, target_radius in enumerate(np.linspace(0.5, 1.4, 10))
    )
    before_centers = [prediction.samples.mean(axis=0) for prediction in predictions]
    factor = calibration_factor(predictions)
    scaled = scale_predictions(predictions, factor)
    metrics = compute_metrics(scaled, "d2_mc")

    assert factor == pytest.approx(1.4)
    assert metrics["hdr90_coverage"] >= 0.9
    assert all(
        np.allclose(before, after.samples.mean(axis=0))
        for before, after in zip(before_centers, scaled)
    )


def test_validation_factor_rejects_degenerate_forecast():
    prediction = _prediction(0, 1.0)
    degenerate = Prediction(
        **{
            **prediction.__dict__,
            "samples": np.zeros_like(prediction.samples),
        }
    )
    with pytest.raises(DispersionCalibrationError, match="zero HDR radius"):
        calibration_factor((degenerate,))


def test_checked_in_calibration_receipt_is_audit_bound_and_not_promoted():
    root = Path(__file__).parents[1] / "experiments" / "nex326"
    receipt_path = root / "pirc20_dispersion_calibration_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    audit = json.loads((root / "pirc20_runtime_audit.json").read_text(encoding="utf-8"))

    assert receipt["assessment"]["passed_candidate_count"] == 0
    assert {row["strict_zero_calibration_gate"] for row in receipt["comparisons"]} == {
        "failed"
    }
    assert receipt["integrity"]["implementation_sha256"] == hashlib.sha256(
        (root / "dispersion_calibration.py").read_bytes()
    ).hexdigest()
    assert audit["future_confirmation"]["dispersion_calibration"]["portable_receipt"][
        "sha256"
    ] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
