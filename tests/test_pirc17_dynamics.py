import numpy as np
import pytest

from experiments.pirc17.dynamics import TransitionRows, baseline_design, diffusion_covariance, fit_dynamics
from experiments.pirc17.origins import known_velocity
from experiments.pirc17.rollout import Features, rollout
from experiments.pirc17.checkpoints import restore_dynamics


def rows(role, *, feature_dim=0):
    velocities = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]] * 4)
    n = len(velocities)
    return TransitionRows(role, (role+"-block",)*n, velocities, velocities, np.ones(n, dtype=bool),
        velocities.copy(), np.ones(n), np.zeros((n, feature_dim)))


def test_diffusion_units_use_dt_not_velocity_covariance():
    covariance = diffusion_covariance([[2,0],[-4,0],[0,6],[0,-8]], np.zeros((4,2)), [1,4,9,16])
    np.testing.assert_allclose(covariance, [[2,0],[0,2]])


def test_base_information_identical_across_terrain_widths():
    np.testing.assert_array_equal(rows("train", feature_dim=0).design(), rows("train", feature_dim=56).design())
    stationary = baseline_design([[0,0]])
    np.testing.assert_array_equal(stationary, [[1,0,0,0,0,0,0]])


def test_fitted_drift_conditioner_and_diffusion_run_closed_loop():
    fitted = fit_dynamics(rows("train"), rows("validation"), seed=7,
        training_identity="synthetic-training-fixture", configuration_identity="fixture-no-terrain")
    origin = known_velocity([0,0], 0, [1,0], source="synthetic-test")
    result = rollout(origin, [1,2], particles=8, seed=11, max_step_seconds=.5,
        base_drift=fitted.base_drift, diffusion=fitted.diffusion,
        terrain=lambda s:Features(np.empty((8,0)),np.empty((8,0),dtype=bool)), conditioner=fitted.correction)
    np.testing.assert_allclose(result.positions_m.mean(axis=0), [[1,0],[2,0]], atol=.03)
    assert fitted.identity["diffusion_estimator"] == "train_brownian_residual_mle_v1"
    assert fitted.base_weights.shape == (7,2)
    restored=restore_dynamics(fitted.identity)
    np.testing.assert_array_equal(restored.base_weights,fitted.base_weights)
    np.testing.assert_array_equal(restored.diffusion_root,fitted.diffusion_root)
    np.testing.assert_array_equal(restored.correction(None,np.empty((3,0))),fitted.correction(None,np.empty((3,0))))
    changed=dict(fitted.identity,base_weights=np.zeros((7,2)).tolist())
    with pytest.raises(ValueError,match="hash"):
        restore_dynamics(changed)


def test_fit_rejects_role_mixup_and_overlap():
    with pytest.raises(ValueError, match="roles reversed"):
        fit_dynamics(rows("validation"), rows("train"), seed=1, training_identity="x", configuration_identity="x")
    validation = rows("validation")
    overlapping = TransitionRows("validation", rows("train").independent_blocks,
        validation.velocities_mps, validation.history_direction, validation.direction_valid,
        validation.displacement_m, validation.elapsed_seconds, validation.feature_matrix)
    with pytest.raises(ValueError, match="blocks overlap"):
        fit_dynamics(rows("train"), overlapping, seed=1, training_identity="x", configuration_identity="x")


def test_frozen_full_conditioner_keeps_56_inputs_and_114_parameters():
    fitted = fit_dynamics(rows("train", feature_dim=56), rows("validation", feature_dim=56),
        seed=9, training_identity="synthetic-full-width", configuration_identity="fixture-56")
    assert fitted.conditioner_model.continuous_input_dim == 56
    assert sum(p.numel() for p in fitted.conditioner_model.parameters()) == 114
    zero = fitted.correction(None, np.zeros((2,56)))
    changed = fitted.correction(None, np.ones((2,56)))
    assert np.isfinite(changed).all() and not np.allclose(zero, changed)
