"""Reproduce known legacy defects, NOT passes of method qualification.

These cheap software-only counterexamples pin why the historical NEX326 path
cannot be used as the PIRC-17 causal/numerical adapter. Historical sources stay
unchanged; a new adapter must independently pass the opposite, causal and
fixed-noise-law properties. No research observations or fitted models are used.
"""
from dataclasses import replace

import numpy as np
import pytest

from experiments.nex326.cohort import Segment
from experiments.nex326.model import ModelState
from experiments.nex326.runner import _propagate_gaussian, predict_segments


def software_fixture():
    time = np.arange(8, dtype=float)
    state = np.column_stack((time, np.zeros(8)))
    segment = Segment("software-only", "fixture", "none", time, state, {}, False)
    weights = np.zeros((6, 2))
    weights[3, 0] = 1.0  # The explicit speed feature drives x velocity.
    model = ModelState("explicit_decomp", (), (weights,), (np.eye(2) * .01,),
                       np.array([1.]), 8, 0., 0., 1.)
    segment.validate()
    return segment, model


@pytest.mark.parametrize("integrator", ["exact", "split", "euler_maruyama"])
@pytest.mark.parametrize("poa", ["fp", "mc", "crn"])
def test_known_legacy_failure_post_origin_truth_changes_forecast(integrator, poa):
    segment, model = software_fixture()
    changed_state = segment.state.copy()
    changed_state[4, 0] += 100.
    changed = replace(segment, state=changed_state)
    changed.validate()
    start = max(1, len(segment.time) // 2 - 1)
    assert start == 3
    np.testing.assert_array_equal(segment.state[:start + 1], changed.state[:start + 1])
    np.testing.assert_array_equal(segment.state[-1], changed.state[-1])
    np.testing.assert_array_equal(segment.time, changed.time)
    before = segment.state.copy()
    config = {"poa": poa, "integrator": integrator, "bridge": "none"}
    original = predict_segments(model, [segment], config, seed=11, n_samples=16)[0]
    altered = predict_segments(model, [changed], config, seed=11, n_samples=16)[0]
    np.testing.assert_array_equal(original.target, altered.target)
    np.testing.assert_array_equal(segment.state, before)
    # This asserts a known defect, not a desirable property of a new predictor.
    assert np.max(np.abs(original.samples - altered.samples)) > 1.


@pytest.mark.parametrize("integrator,substeps", [("exact", 1), ("split", 2), ("euler_maruyama", 4)])
def test_known_legacy_failure_substeps_change_constant_noise_law(integrator, substeps):
    segment, model = software_fixture()
    model = replace(model, model_kind="single_gaussian", weights=(np.zeros((3, 2)),),
                    covariances=(np.eye(2),))
    mean, covariance = _propagate_gaussian(model, 0, segment, 3, integrator)
    np.testing.assert_array_equal(mean, segment.state[3])
    # Four unit intervals: legacy noise dt**2 / substeps**2 per interval.
    # Therefore the variances 4, 1, .25 cannot be an h-refinement certificate
    # for one fixed constant-diffusion SDE. Unit conversion remains unqualified.
    expected = 4. / substeps**2 + 1e-10
    np.testing.assert_allclose(covariance, np.eye(2) * expected, rtol=0, atol=1e-15)
