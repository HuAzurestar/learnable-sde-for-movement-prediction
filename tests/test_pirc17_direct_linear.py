"""Software oracles for the distinct optimizer; not research observations."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from experiments.pirc17 import direct_linear as direct
from experiments.pirc17.checkpoints import restore_dynamics
from experiments.pirc17.dynamics import TransitionRows, diffusion_covariance
from experiments.pirc17.inference import SEEDS
from experiments.pirc17.origins import known_velocity, motion
from experiments.pirc17.rollout import Features, rollout


def rows(role, configuration="base"):
    rng = np.random.default_rng(8 if role == "train" else 19)
    velocity = rng.normal(size=(24, 2))
    _, direction, valid = motion(velocity)
    dt = np.linspace(1, 4, len(velocity))
    displacement = (velocity + .2*np.sin(velocity*3)) * dt[:, None]
    features = rng.normal(size=(len(velocity), direct.configuration_width(configuration)))
    return TransitionRows(role, tuple(role+str(i//4) for i in range(len(velocity))),
        velocity, direction, valid, displacement, dt, features)


def fitted(configuration="base", seed=SEEDS[0]):
    return direct.fit_direct_dynamics(rows("train", configuration), rows("validation", configuration),
        seed=seed, training_identity="a"*64, configuration=configuration)


def rehash(identity):
    cp = identity["conditioner_checkpoint"]
    cp.pop("sha256", None)
    cp["sha256"] = direct.digest(cp)
    identity.pop("sha256", None)
    identity["sha256"] = direct.digest(identity)
    return identity


def test_normalization_and_bias_match_diagonal_analytic_solution():
    x = np.array([[-2.], [-1.], [1.], [2.]])
    y = x * [4., -1.] + [2., 3.]
    theta, report = direct.solve_linear(x, y, weight_decay=.2)
    expected = np.array([[40/10.8, -10/10.8], [8/4.8, 12/4.8]])
    np.testing.assert_allclose(theta, expected, rtol=1e-14, atol=1e-14)
    assert report["normal_equation_penalty"] == .8
    assert report["augmented_rank"] == 2
    assert report["objective"] < report["zero_correction_objective"]
    assert not theta.flags.writeable


@pytest.mark.parametrize("width", [0, 1, 4, 56])
def test_augmented_solver_matches_normal_equations_and_autograd(width):
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(80, width)), rng.normal(size=(80, 2))
    before = x.copy(), y.copy()
    theta, report = direct.solve_linear(x, y)
    design = np.column_stack((x, np.ones(len(x))))
    oracle = np.linalg.solve(design.T@design + len(x)*1e-4*np.eye(width+1), design.T@y)
    np.testing.assert_allclose(theta, oracle, rtol=1e-10, atol=1e-12)
    parameter = torch.tensor(theta.copy(), dtype=torch.float64, requires_grad=True)
    loss = ((torch.tensor(design)@parameter-torch.tensor(y))**2).mean() + .5e-4*(parameter**2).sum()
    loss.backward()
    assert parameter.grad.abs().max().item() < 1e-11
    assert loss.item() == pytest.approx(report["objective"], abs=1e-13)
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(y, before[1])


@pytest.mark.parametrize("case", ["constant", "zero", "duplicate", "underdetermined"])
def test_rank_deficient_feature_design_has_unique_regularized_solution(case):
    rng = np.random.default_rng(91)
    n, p = (4, 12) if case == "underdetermined" else (12, 4)
    x = rng.normal(size=(n, p))
    if case == "constant": x[:] = 1.
    if case == "zero": x[:] = 0.
    if case == "duplicate": x[:, 1:] = x[:, :1]
    theta, report = direct.solve_linear(x, rng.normal(size=(n, 2)))
    assert np.isfinite(theta).all() and report["augmented_rank"] == p+1
    assert report["relative_gradient_norm"] <= direct.STATIONARITY_RTOL
    assert report["objective"] <= report["zero_correction_objective"] + 1e-12


@pytest.mark.parametrize("x,y,decay", [
    ([[1.]], [[1., 2.]], 1e-4),
    ([[1.], [2.]], [[1.], [2.]], 1e-4),
    ([[1.], [2.]], [[1., 2.]]*3, 1e-4),
    ([[np.nan], [2.]], [[1., 2.]]*2, 1e-4),
    ([[1.], [2.]], [[np.inf, 2.]]*2, 1e-4),
    ([[1.], [2.]], [[1., 2.]]*2, 0),
    ([[1.], [2.]], [[1., 2.]]*2, -1),
    ([[1.], [2.]], [[1., 2.]]*2, True),
    ([[1.], [2.]], [[1., 2.]]*2, np.nan),
])
def test_solver_rejects_invalid_inputs(x, y, decay):
    with pytest.raises(ValueError):
        direct.solve_linear(x, y, weight_decay=decay)


def test_solver_rejects_extreme_condition_instead_of_silent_rank_truncation():
    with pytest.raises(ValueError, match="ill-conditioned"):
        direct.solve_linear([[1e12, 0], [0, 1e-8], [2e12, 0]], [[1, 0], [0, 1], [2, 0]])


def test_exact_frozen_widths_and_policy_are_explicit_not_old_epochs():
    assert {name:direct.configuration_width(name) for name in direct.terrain_configurations()} == {
        "base":4, "all-terrain":56, "loo-road":38, "loo-river":46,
        "loo-worldcover":40, "loo-surface":40, "lio-road":14,
        "lio-river":14, "lio-worldcover":12, "lio-surface":20}
    policy = direct.training_policy()
    assert policy["superseded_optimizer_settings"]["maximum_epochs"] == 40
    assert policy["data_roles"]["selection"] == "none"
    assert not policy["final_eval_authorized"]
    assert policy["sha256"] == direct.digest({k:v for k,v in policy.items() if k != "sha256"})


def test_validation_and_seed_do_not_change_fitted_parameters():
    train, validation = rows("train"), rows("validation")
    first = fitted()
    altered = replace(validation, displacement_m=validation.displacement_m*20,
                      feature_matrix=validation.feature_matrix*5)
    for seed in SEEDS:
        other = direct.fit_direct_dynamics(train, altered, seed=seed,
            training_identity="a"*64, configuration="base")
        np.testing.assert_array_equal(first.base_weights, other.base_weights)
        np.testing.assert_array_equal(first.diffusion_root, other.diffusion_root)
        np.testing.assert_array_equal(first.correction(None, train.feature_matrix),
                                      other.correction(None, train.feature_matrix))
        assert other.identity["diagnostics"]["validation"] != first.identity["diagnostics"]["validation"]


def test_restore_and_rollout_use_full_dynamics_and_reject_old_reader():
    model = fitted("all-terrain")
    restored = direct.restore_direct_dynamics(json.loads(json.dumps(model.identity, allow_nan=False)))
    train = rows("train", "all-terrain")
    correction = restored.correction(None, train.feature_matrix)
    covariance = diffusion_covariance(train.displacement_m, train.design()@restored.base_weights+correction,
                                     train.elapsed_seconds)
    np.testing.assert_allclose(covariance, model.identity["diffusion_covariance_m2_per_s"], atol=1e-14)
    np.testing.assert_array_equal(restored.correction(None, train.feature_matrix), model.correction(None, train.feature_matrix))
    np.testing.assert_array_equal(restored.base_weights, model.base_weights)
    np.testing.assert_array_equal(restored.diffusion_root, model.diffusion_root)
    origin = known_velocity([0.,0.], 0., [1.,0.], source="software-only")
    result = rollout(origin, [1.,2.], particles=4, seed=SEEDS[0], max_step_seconds=.5,
        base_drift=restored.base_drift, diffusion=restored.diffusion,
        terrain=lambda s: Features(np.zeros((4,28)), np.ones((4,28), dtype=bool)),
        conditioner=restored.correction)
    assert result.positions_m.shape == (4,2,2) and np.isfinite(result.positions_m).all()
    with pytest.raises(ValueError, match="version/hash"):
        restore_dynamics(model.identity)
    assert "best_epoch" not in model.conditioner_checkpoint and "learning_curve" not in model.conditioner_checkpoint


@pytest.mark.parametrize("fault", ["roles", "blocks", "width", "seed", "bool_seed", "identity", "configuration"])
def test_fit_rejects_bad_roles_pairing_and_bindings(fault):
    train, validation = rows("train"), rows("validation")
    kwargs = dict(seed=SEEDS[0], training_identity="a"*64, configuration="base")
    if fault == "roles": train,validation = validation,train
    if fault == "blocks": validation = replace(validation, independent_blocks=train.independent_blocks)
    if fault == "width": validation = replace(validation, feature_matrix=np.zeros((24,6)))
    if fault == "seed": kwargs["seed"] = 1
    if fault == "bool_seed": kwargs["seed"] = True
    if fault == "identity": kwargs["training_identity"] = "not-sha256"
    if fault == "configuration": kwargs["configuration"] = "loo-history"
    with pytest.raises(ValueError):
        direct.fit_direct_dynamics(train, validation, **kwargs)


@pytest.mark.parametrize("fault", ["outer_hash", "old_version", "policy", "config_hash", "base_columns",
    "count", "seed", "conditioner_hash", "dimension", "penalty", "rank", "condition", "gradient",
    "objective", "coefficient", "loss", "covariance", "infinite", "weights", "extra", "epochs"])
def test_restore_rejects_tampering_even_if_semantic_changes_are_rehashed(fault):
    identity = deepcopy(fitted().identity)
    cp = identity["conditioner_checkpoint"]
    report = cp["solver_report"]
    if fault == "outer_hash": identity["sha256"] = "0"*64
    if fault == "old_version": identity["version"] = "pirc17-causal-kinematic-drift-v1"
    if fault == "policy": identity["training_policy"]["data_roles"]["solve"] = "validation"
    if fault == "config_hash": identity["configuration_identity"] = "b"*64
    if fault == "base_columns": identity["base_columns"] = []
    if fault == "count": identity["train_transition_count"] = 1
    if fault == "seed": identity["seed"] = 1
    if fault == "conditioner_hash": cp["sha256"] = "0"*64
    if fault == "dimension": cp["input_dim"] = 5
    if fault == "penalty": report["normal_equation_penalty"] = 1e-4
    if fault == "rank": report["augmented_rank"] = 1
    if fault == "condition": report["augmented_condition"] = 1e12
    if fault == "gradient": report["relative_gradient_norm"] = 1.
    if fault == "objective": report["objective"] = report["zero_correction_objective"]+1.
    if fault == "coefficient": cp["theta_feature_rows_then_bias"][0][0] += 10
    if fault == "loss": identity["diagnostics"]["train"]["fitted_mse_m2_per_s2"] += 1
    if fault == "covariance": identity["diffusion_covariance_m2_per_s"] = [[-1,0],[0,1]]
    if fault == "infinite": identity["diagnostics"]["validation"]["fitted_mse_m2_per_s2"] = float("nan")
    if fault == "weights": identity["base_weights"] = [[0,0]]
    if fault == "extra": identity["certified"] = True
    if fault == "epochs": cp["best_epoch"] = 40
    if fault not in {"outer_hash", "conditioner_hash", "infinite"}:
        identity = rehash(identity)
    elif fault == "conditioner_hash":
        identity.pop("sha256")
        identity["sha256"] = direct.digest(identity)
    with pytest.raises(ValueError):
        direct.restore_direct_dynamics(identity)
