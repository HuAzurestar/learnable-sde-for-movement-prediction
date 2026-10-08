import numpy as np
import pytest

from experiments.pirc17.origins import Origin, VelocityPrior, causal_prefix, known_velocity, motion


def test_prefix_speed_is_not_unit_direction_and_copies_input():
    positions = np.array([[0., 0.], [6., 8.], [9., 12.]])
    origin = causal_prefix(positions, [0, 2, 3])
    np.testing.assert_allclose(origin.velocity_mps, [3, 4])
    speed, direction, valid = motion(origin.velocity_mps)
    assert speed == 5 and valid
    np.testing.assert_allclose(direction, [.6, .8])
    positions[-1] = 999
    np.testing.assert_array_equal(origin.position_m, [9, 12])
    with pytest.raises(ValueError):
        origin.position_m[0] = 1


def test_stationary_and_known_velocity_provenance():
    stationary = causal_prefix([[2, 3], [2, 3]], [0, 1])
    speed, heading, valid = motion(stationary.velocity_mps)
    assert speed == 0 and not valid
    np.testing.assert_array_equal(heading, [0, 0])
    with pytest.raises(ValueError):
        known_velocity([0, 0], 1, [1, 0], source="")
    origin = known_velocity([0, 0], 1, [1, 0], source="test-measured-at-origin")
    assert len(origin.history_times_seconds) == 1
    assert origin.velocity_observed_at_seconds == 1 and origin.velocity_error_mps is None
    with pytest.raises(ValueError, match="no later than origin"):
        known_velocity([0,0], 1, [1,0], source="fixture", observed_at_seconds=2)
    with pytest.raises(ValueError, match="nonnegative"):
        known_velocity([0,0], 1, [1,0], source="fixture", error_mps=-1)


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, 61], [0, np.nan]])
def test_bad_prefix_times_fail(times):
    with pytest.raises(ValueError):
        causal_prefix([[0, 0], [1, 1]], times)


def test_point_only_uses_train_prior_without_borrowing_history():
    origins = [causal_prefix([[0, 0], [v, 0]], [0, 1]) for v in [-2, 2]]
    with pytest.raises(ValueError, match="only be fitted on train"):
        VelocityPrior.fit(origins, split="validation", training_identity="test")
    prior = VelocityPrior.fit(origins, split="train", training_identity="train-hash")
    origin = prior.at([8, 9], 100)
    assert origin.mode == "point_only" and len(origin.history_times_seconds) == 1
    draws = origin.sample_velocities(100, np.random.default_rng(1))
    assert set(draws[:, 0]) == {-2, 2}
    assert np.all(draws[:, 1] == 0)
    assert prior.identity != VelocityPrior(prior.velocities_mps, "other-train").identity


def test_origin_cannot_hide_future_or_history_in_wrong_mode():
    with pytest.raises(ValueError, match="end at the origin"):
        Origin("causal_prefix", 1, [1, 1], [1, 1], [[0, 0], [1, 1]], [0, 2], "test")
    with pytest.raises(ValueError, match="hide extra history"):
        Origin("point_only", 1, [1, 1], [1, 1], [[0, 0], [1, 1]], [0, 1], "test")
