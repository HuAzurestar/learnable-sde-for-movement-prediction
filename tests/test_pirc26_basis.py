"""Independent dense/derivative oracles for the bounded production QR engine."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.interpolate import BSpline
import torch

from domain import ModelContext
from estimation.phase_space import TransitionBatch, fit_residual_basis
from estimation.phase_space_basis import BasisPlan, fit_basis, free_coefficients, spline_curvature_factor
from models.phase_space import AffineAccelerationDrift, ModelContractError, PhaseSpaceSDE, SplineResidualDrift
from tests.test_pirc26_dynamics import model, spec


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic(m, *, count=211):
    generator = torch.Generator().manual_seed(941)
    z = 3.6 * torch.rand(count, 4, dtype=torch.float64, generator=generator) - 1.8
    t = torch.linspace(0., 10., count, dtype=torch.float64)
    dt = .05 + .1 * torch.rand(count, dtype=torch.float64, generator=generator)
    d = m.acceleration_model
    identity = "full-rbf-v1" if d.family == "M1-R" else "reference-coded-additive-v1"
    free, constrained = free_coefficients(d, identity)
    truth = .2 * torch.randn(len(d.coefficients), 2, dtype=torch.float64, generator=generator)
    truth[constrained] = 0
    c = z.new_empty((count, 0))
    target = z.clone()
    target[:, 2:] += (d.affine(t, z, c) + d.basis(d.selected_features(z, c)) @ truth).detach() * dt[:, None]
    return TransitionBatch(t, z, target, dt, ModelContext(), m.spec.train_binding_hash), truth


def partition(b, size):
    return [TransitionBatch(b.time[i:i+size], b.state[i:i+size], b.next_state[i:i+size], b.dt[i:i+size],
                            b.context, b.train_binding_hash, b.split_role) for i in range(0, len(b.time), size)]


@pytest.mark.parametrize("family", ["M1-S", "M1-R"])
def test_qr_recovers_coefficients_freezes_base_and_matches_dense_augmented_least_squares(family):
    m = model(family)
    b, truth = synthetic(m)
    plan = BasisPlan(ridge=.03, curvature_penalty=.08 if family == "M1-S" else 0.,
                     max_batch_rows=31, identifiability="full-rbf-v1" if family == "M1-R" else "reference-coded-additive-v1")
    initial = m.checkpoint()
    d = m.acceleration_model
    design = d.basis(d.selected_features(b.state, b.state.new_empty((len(b.time), 0)))) * b.dt.sqrt()[:, None]
    response = (b.next_state[:, 2:] - b.state[:, 2:]) / b.dt[:, None] - d.affine(b.time, b.state, b.state.new_empty((len(b.time), 0)))
    penalties = [np.sqrt(plan.ridge) * np.eye(len(truth))]
    if plan.curvature_penalty:
        penalties.append(np.sqrt(plan.curvature_penalty) * spline_curvature_factor(d).numpy())
    penalty = np.concatenate(penalties)
    expected = np.linalg.lstsq(np.concatenate((design.detach().numpy(), penalty)),
                              np.concatenate(((response * b.dt.sqrt()[:, None]).detach().numpy(), np.zeros((len(penalty), 2)))), rcond=None)[0]
    rows = []
    result = fit_basis(m, partition(b, 31), plan, progress=rows.append)
    assert np.allclose(d.coefficients.detach().numpy(), expected, atol=1e-11)
    assert result["steps"] == len(rows) == 7 and rows[-1]["observations"] == 211
    assert result["resume_level"] == "restart-only" and result["gradient_route"] == "G0"
    assert result["penalized_train_objective"] >= result["train_objective"]
    assert 0 < result["effective_degrees_of_freedom"] <= len(truth)
    assert result["scope"]["initial_model_hash"] == initial["sha256"]
    fitted = PhaseSpaceSDE.from_checkpoint(result["checkpoint"])
    baseline = PhaseSpaceSDE.from_checkpoint(initial)
    for name, value in baseline.acceleration_model.affine.state_dict().items():
        assert torch.equal(value, fitted.acceleration_model.affine.state_dict()[name])
    assert torch.equal(baseline.velocity_factor, fitted.velocity_factor)
    fresh = PhaseSpaceSDE.from_checkpoint(initial)
    unregularized = fit_residual_basis(fresh, [b], ridge=0., condition_number_max=1e6)
    assert torch.allclose(fresh.acceleration_model.coefficients, truth, atol=1e-10)
    assert unregularized["effective_degrees_of_freedom"] == len(truth)


@pytest.mark.parametrize("degree", [2, 3])
def test_curvature_factor_matches_independent_bspline_second_derivative_integrals(degree):
    knots = [-2.] * (degree+1) + [-.7, .2, 1.1] + [2.] * (degree+1)
    d = SplineResidualDrift(AffineAccelerationDrift(), spec(), [2], [knots], degree=degree).double()
    factor = spline_curvature_factor(d)
    t = d.knots[0].numpy()
    oracle = BSpline(t, np.eye(len(d.coefficients)), degree).derivative(2)
    expected, _ = quad_vec(lambda x: np.outer(oracle(x), oracle(x)), t[0], t[-1], points=np.unique(t)[1:-1])
    assert np.allclose((factor.T @ factor).numpy(), expected, atol=2e-11)
    # Constants/linear functions have zero curvature, including nonuniform knots.
    assert torch.allclose(factor @ torch.ones(len(d.coefficients), dtype=torch.float64), torch.zeros(len(factor), dtype=torch.float64), atol=1e-12)
    greville = torch.stack([d.knots[0][i+1:i+degree+1].mean() for i in range(len(d.coefficients))])
    assert torch.allclose(factor @ greville, torch.zeros(len(factor), dtype=torch.float64), atol=1e-12)


def test_multiple_additive_splines_remove_only_redundant_constants_and_recover_functions():
    d = SplineResidualDrift(AffineAccelerationDrift(), spec(), [2, 3],
        [[-2., -2., -2., -2., 0., 2., 2., 2., 2.]] * 2).double()
    m = PhaseSpaceSDE(d, [[.4, 0.], [.1, .3]], spec()).double()
    b, truth = synthetic(m)
    result = fit_basis(m, partition(b, 29), BasisPlan(max_batch_rows=29))
    assert result["basis_count"] == 10 and result["free_basis_count"] == 9
    assert result["constrained_coefficient_indices"] == [9]
    assert torch.allclose(d.coefficients, truth, atol=1e-10)
    u = d.selected_features(b.state, b.state.new_empty((len(b.time), 0)))
    arbitrary = truth.clone()
    arbitrary[5:] += torch.tensor([.3, -.2])
    arbitrary[:5] -= torch.tensor([.3, -.2])
    assert torch.allclose(d.basis(u) @ arbitrary, d.basis(u) @ truth, atol=1e-12)
    regularized = fit_basis(m, [b], BasisPlan(ridge=.02, curvature_penalty=.03))
    assert regularized["free_basis_count"] == 9 and 0 < regularized["effective_degrees_of_freedom"] < 9


@pytest.mark.parametrize("fault", ["selection", "covariance", "rank", "condition", "curvature", "cancel", "observer"])
def test_refusals_leave_the_entire_initial_model_unchanged(fault):
    m = model("M1-R")
    b, _ = synthetic(m)
    plan = BasisPlan(max_batch_rows=29, identifiability="full-rbf-v1")
    if fault == "selection":
        b = replace(b, split_role="selection")
    elif fault == "covariance":
        m.velocity_factor.zero_()
    elif fault == "rank":
        b = replace(b, state=b.state * 0)
        plan = replace(plan, ridge=10.)  # do not silently rescue data rank.
    elif fault == "condition":
        plan = replace(plan, condition_number_max=1.)
    elif fault == "curvature":
        plan = replace(plan, curvature_penalty=.1)
    initial = m.checkpoint()
    seen = []
    def observe(row):
        seen.append(row)
        if fault == "observer":
            raise RuntimeError("explicit fixture observer refusal")
    with pytest.raises((ModelContractError, RuntimeError)):
        fit_basis(m, partition(b, 29), plan, progress=observe, cancellation=lambda: fault == "cancel" and bool(seen))
    assert m.checkpoint() == initial


@pytest.mark.parametrize("degree,knots", [(1, [-2., -2., 0., 2., 2.]),
    (3, [-2., -2., -2., -2., 0., 0., 0., 2., 2., 2., 2.])])
def test_active_curvature_refuses_non_c1_profiles(degree, knots):
    d = SplineResidualDrift(AffineAccelerationDrift(), spec(), [2], [knots], degree=degree).double()
    with pytest.raises(ModelContractError, match="OBJECTIVE_INCOMPATIBLE"):
        spline_curvature_factor(d)


def test_basis_scope_binds_data_and_regularization_and_batch_quota_is_checked_before_qr(monkeypatch):
    m = model("M1-R")
    b, _ = synthetic(m)
    plan = BasisPlan(identifiability="full-rbf-v1")
    initial = m.checkpoint()
    first = fit_basis(m, [b], plan)
    altered = replace(b, next_state=b.next_state + .001)
    other = fit_basis(PhaseSpaceSDE.from_checkpoint(initial), [altered], plan)
    different_plan = fit_basis(PhaseSpaceSDE.from_checkpoint(initial), [b], replace(plan, ridge=.1))
    assert len({first["objective_hash"], other["objective_hash"], different_plan["objective_hash"]}) == 3
    monkeypatch.setattr(torch.linalg, "qr", lambda *a, **k: pytest.fail("quota refusal reached QR"))
    with pytest.raises(ModelContractError, match="RESOURCE_PLAN_REJECTED"):
        fit_basis(m, [b], replace(plan, max_batch_rows=30))
