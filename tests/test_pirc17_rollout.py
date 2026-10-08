import numpy as np
import pytest

from experiments.pirc17.origins import VelocityPrior, causal_prefix, known_velocity
from experiments.pirc17.rollout import Features, inertial, rollout


def zero_noise(state):
    return np.zeros((len(state.positions_m), 2, 2))


def test_irregular_horizons_match_equal_information_inertial():
    origin = causal_prefix([[0, 0], [6, 8]], [100, 102])
    result = rollout(origin, [1.2, 4.1], particles=4, seed=7, max_step_seconds=.7,
                     base_drift=lambda s: s.velocities_mps, diffusion=zero_noise)
    baseline = inertial(origin, [1.2, 4.1], particles=4, seed=7)
    np.testing.assert_allclose(result.positions_m, baseline.positions_m)
    np.testing.assert_array_equal(result.elapsed_seconds, [1.2, 4.1])


def test_terrain_queries_follow_predictions_and_history_updates():
    origin = known_velocity([0, 0], 200, [1, 0], source="fixture-known")
    seen = []
    def terrain(state):
        seen.append(state)
        assert not state.positions_m.flags.writeable
        return Features(state.positions_m[:, :1], np.ones((2, 1), dtype=bool))
    def correction(state, features):
        return np.column_stack((features[:, 0], np.ones(2)))
    result = rollout(origin, [3], particles=2, seed=1, max_step_seconds=1,
                     history_step_seconds=1,
                     base_drift=lambda s: np.tile([1, 0], (2, 1)), diffusion=zero_noise,
                     terrain=terrain, conditioner=correction)
    np.testing.assert_allclose(result.positions_m[:, 0], [[7, 3], [7, 3]])
    np.testing.assert_allclose([s.positions_m[0].tolist() for s in seen], [[0, 0], [1, 1], [3, 2]])
    np.testing.assert_allclose(seen[-1].history_direction[0][0], [3/np.sqrt(13), 2/np.sqrt(13)])
    np.testing.assert_allclose(seen[-1].velocities_mps[0], [1.5, 1.0])
    assert result.feature_query_rows == 6 and result.invalid_feature_rows == 0


def test_missing_map_queries_are_masked_and_not_dropped():
    origin = known_velocity([0, 0], 0, [0, 0], source="fixture")
    matrices = []
    def correction(state, values):
        matrices.append(values)
        return np.zeros((3, 2))
    result = rollout(origin, [2], particles=3, seed=1, max_step_seconds=1,
                     base_drift=lambda s: s.velocities_mps, diffusion=zero_noise,
                     terrain=lambda s: Features(np.full((3, 1), np.nan), np.zeros((3, 1), dtype=bool)),
                     conditioner=correction)
    assert result.invalid_feature_rows == 6
    assert np.all(np.stack(matrices) == 0) and result.positions_m.shape == (3, 1, 2)


def test_brownian_scaling_and_seed_replay():
    origin = known_velocity([0, 0], 0, [0, 0], source="fixture")
    def predict(seed):
        return rollout(origin, [4], particles=20000, seed=seed, max_step_seconds=4,
            base_drift=lambda s: np.zeros_like(s.positions_m),
            diffusion=lambda s: np.broadcast_to(np.eye(2)*3, (20000, 2, 2)))
    first = predict(71)
    np.testing.assert_array_equal(first.positions_m, predict(71).positions_m)
    np.testing.assert_allclose(first.positions_m[:, 0].var(axis=0), [36, 36], rtol=.04)


def test_point_only_baseline_and_model_start_with_same_prior_draws():
    prior = VelocityPrior([[1, 0], [-1, 2]], "fixture-train")
    origin = prior.at([4, 5], 100)
    baseline = inertial(origin, [1, 2], particles=12, seed=8)
    result = rollout(origin, [1, 2], particles=12, seed=8, max_step_seconds=.5,
                     base_drift=lambda s: s.velocities_mps, diffusion=zero_noise)
    np.testing.assert_allclose(result.positions_m, baseline.positions_m)


@pytest.mark.parametrize("horizons", [[], [0], [2, 1], [1, 1], [np.inf]])
def test_invalid_elapsed_time_rejected(horizons):
    origin = known_velocity([0, 0], 0, [1, 1], source="fixture")
    with pytest.raises(ValueError):
        inertial(origin, horizons, particles=1, seed=1)


def test_invalid_feature_mask_does_not_silently_accept_nan():
    with pytest.raises(ValueError):
        Features(np.array([[np.nan]]), np.array([[True]])).model_matrix(1)
    with pytest.raises(ValueError):
        Features(np.array([[1.]]), np.array([[1.]])).model_matrix(1)


def test_halving_step_reduces_known_linear_drift_error():
    origin = known_velocity([1,1], 0, [0,0], source="analytic-fixture")
    errors = []
    for step in (.2,.1,.05):
        result = rollout(origin, [1], particles=1, seed=1, max_step_seconds=step,
                         base_drift=lambda s:-s.positions_m, diffusion=zero_noise)
        errors.append(abs(result.positions_m[0,0,0] - np.exp(-1)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[1] < .55 * errors[0] and errors[2] < .55 * errors[1]


def test_inertial_diagnostic_freezes_origin_velocity_on_curved_prefix():
    origin = causal_prefix([[0,0],[1,2],[4,3]],[0,1,2])
    result = rollout(origin,[1,2,3],particles=3,seed=1,max_step_seconds=.5,
        base_drift=lambda s:np.broadcast_to(origin.velocity_mps,(3,2)),diffusion=zero_noise)
    np.testing.assert_allclose(result.positions_m,inertial(origin,[1,2,3],particles=3,seed=1).positions_m)


@pytest.mark.parametrize("step", [1., .5, .25, .125])
def test_noisy_velocity_history_has_step_independent_analytic_variance(step):
    # With cadence 5, X(10)=2 W(5)+(W(10)-W(5)), so Var X(10)=25.
    # Updating history every numerical step instead produced unbounded variance.
    origin = known_velocity([0,0], 0, [0,0], source="analytic-software-fixture")
    result = rollout(origin,[10],particles=30000,seed=1,max_step_seconds=step,
        history_step_seconds=5.,base_drift=lambda s:s.velocities_mps,
        diffusion=lambda s:np.broadcast_to(np.eye(2),(30000,2,2)))
    np.testing.assert_allclose(result.positions_m[:,0].var(axis=0), [25,25], rtol=.04)


def test_history_ticks_do_not_follow_irregular_outputs_or_integrator_steps():
    origin = causal_prefix([[0,0],[1,0],[3,0]],[-3,-2,0])
    seen = []
    def drift(state):
        seen.append(state)
        return np.broadcast_to([1.,0.],state.positions_m.shape)
    rollout(origin,[.7,1.3,3.1,5.2,10.2,16.],particles=1,seed=1,max_step_seconds=.6,
        history_step_seconds=5.,base_drift=drift,diffusion=zero_noise)
    for state in seen:
        ticks = np.arange(1,int(state.elapsed_seconds//5)+1)*5.
        expected = np.concatenate(([-3,-2,0],ticks))[-3:]
        np.testing.assert_array_equal(state.history_elapsed_seconds,expected)
        assert state.history_elapsed_seconds[-1] <= state.elapsed_seconds
        np.testing.assert_allclose(state.history_positions_m[0,:,0],expected+3)


@pytest.mark.parametrize("cadence", [0,-1,np.nan,np.inf])
def test_invalid_history_cadence_is_rejected(cadence):
    origin = known_velocity([0,0],0,[0,0],source="fixture")
    with pytest.raises(ValueError,match="history cadence"):
        rollout(origin,[1],particles=1,seed=1,max_step_seconds=1,
            history_step_seconds=cadence,base_drift=lambda s:s.velocities_mps,diffusion=zero_noise)
