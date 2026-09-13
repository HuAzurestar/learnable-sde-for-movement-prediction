from __future__ import annotations

import numpy as np
import pytest

from experiments.nex326 import runner as runner_module
from experiments.nex326.cohort import Segment
from experiments.nex326.pirc20_runtime import (
    PIRC20NEX326Runner,
    predict_segments_without_endpoint_start,
)


def _segment(points: int) -> Segment:
    return Segment(
        segment_id=f"segment-{points}",
        source_domain="fixture",
        region="fixture",
        time=np.linspace(0.0, 600.0, points),
        state=np.column_stack(
            (np.linspace(0.0, 10.0, points), np.linspace(0.0, -5.0, points))
        ),
        conditions={},
        has_terrain=False,
    )


@pytest.mark.parametrize(("points", "expected_start"), [(2, 0), (3, 1), (4, 1)])
def test_pirc20_prediction_never_starts_from_endpoint(
    monkeypatch, points, expected_start
):
    starts: list[int] = []

    def fake_rollout(model, segment, start, integrator, n_samples, rng, field):
        starts.append(start)
        return np.zeros((n_samples, 2))

    monkeypatch.setattr(runner_module, "_fp_rollout", fake_rollout)
    predictions = predict_segments_without_endpoint_start(
        object(),
        [_segment(points)],
        {"poa": "fp", "integrator": "split", "bridge": "none"},
        seed=7,
        n_samples=4,
    )

    assert starts == [expected_start]
    assert expected_start < points - 1
    assert len(predictions) == 1


def test_pirc20_runner_restores_core_prediction_function_on_error(monkeypatch):
    original = runner_module.predict_segments

    def fail_after_check(self, arm, subconfig):
        assert runner_module.predict_segments is predict_segments_without_endpoint_start
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(runner_module.NEX326Runner, "run_one", fail_after_check)
    instance = object.__new__(PIRC20NEX326Runner)

    with pytest.raises(RuntimeError, match="fixture failure"):
        instance.run_one(object(), {})
    assert runner_module.predict_segments is original
