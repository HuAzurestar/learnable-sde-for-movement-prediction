"""O1, causal common rollout and independently calculated proper scores."""

from dataclasses import replace
import math

import pytest
import torch

from application.pirc26_dynamics import AffinePhaseSpaceOracle
from domain import ModelContext
from estimation.phase_space import O1Plan, TransitionBatch, VelocityCholesky, fit_o1, local_velocity_nll, fit_residual_basis
from evaluation.phase_space import energy_score_value, evaluate_forecast
from inference.phase_space import ForecastRequest, brownian_increments, forecast, forecast_coupled_levels
from models.phase_space import AffineAccelerationDrift, ModelContractError, PhaseSpaceSDE
from tests.test_pirc26_dynamics import inputs, model, spec


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch(m, count=256):
    generator = torch.Generator().manual_seed(29)
    state = torch.randn(count, 4, generator=generator, dtype=torch.float64)
    time = torch.linspace(0., 20., count, dtype=torch.float64)
    dt = torch.linspace(.05, .2, count, dtype=torch.float64)
    target = state.clone()
    target[:, :2] += state[:, 2:] * dt[:, None]
    # Known coupled drift and diffusion, independent of the fitted model.
    acceleration = state[:, 2:] @ torch.tensor([[-.3, .1], [-.2, -.4]], dtype=torch.float64).T
    noise = torch.randn(count, 2, generator=generator, dtype=torch.float64) * .05
    target[:, 2:] += acceleration * dt[:, None] + noise * dt.sqrt()[:, None]
    return TransitionBatch(time, state, target, dt, ModelContext(), m.spec.train_binding_hash)


def request(sample_count=128, chunk_size=32, times=(0., .1, .2, .3)):
    return ForecastRequest((0., 0., .3, -.2), times, 0., sample_count, "a" * 64,
                           chunk_size=chunk_size)


def test_velocity_objective_matches_manual_density_and_ignores_position_target():
    m = model("M0")
    b = batch(m, 16)
    covariance = m.velocity_factor @ m.velocity_factor.T
    residual = b.next_state[:, 2:] - b.state[:, 2:] - m.acceleration(b.time, b.state, b.context) * b.dt[:, None]
    expected = .5 * (2 * math.log(2 * math.pi) + torch.linalg.slogdet(covariance).logabsdet
                    + 2 * b.dt.log() + (residual @ torch.linalg.inv(covariance) * residual).sum(-1) / b.dt)
    assert torch.allclose(local_velocity_nll(m, b), expected.mean(), atol=1e-12)
    altered = b.next_state.clone()
    altered[:, :2] = 100000.
    assert torch.equal(local_velocity_nll(m, b), local_velocity_nll(m, replace(b, next_state=altered)))
    with pytest.raises(ModelContractError, match="OBJECTIVE_INCOMPATIBLE"):
        local_velocity_nll(m, b, torch.zeros_like(m.velocity_factor))
    with pytest.raises(ModelContractError, match="UNAUTHORIZED_DATA"):
        fit_o1(m, [replace(b, split_role="selection")], O1Plan())


def test_train_only_fit_recovers_affine_with_frozen_diffusion_and_stops():
    m = PhaseSpaceSDE(AffineAccelerationDrift(), torch.eye(2) * .05, spec()).double()
    b = batch(m, 2048)
    before = local_velocity_nll(m, b).item()
    result = fit_o1(m, [b], O1Plan(max_steps=200, patience=40, learning_rate=.02, fit_diffusion=False))
    expected = torch.tensor([[-.3, .1], [-.2, -.4]], dtype=torch.float64)
    assert torch.allclose(m.acceleration_model.A, expected, atol=.03)
    assert result["best_train_objective"] < before - 1
    assert result["objective"] == "O1" and result["gradient_route"] == "G0"
    assert 0 < result["steps"] <= 200 and result["resume_level"] == "exact"
    interrupted = fit_o1(m, [b], O1Plan(), cancellation=lambda: True)
    assert interrupted["status"] == "INTERRUPTED" and interrupted["steps"] == 0
    assert interrupted["best_train_objective"] is None


@pytest.mark.parametrize("family", ["M0", "M2"])
def test_finite_float32_gradients_with_overflowing_global_norm_fail_before_update(family, monkeypatch):
    m = model(family, torch.float32)
    state = torch.full((2, 4), 1e10, dtype=torch.float32)
    b = TransitionBatch(torch.zeros(2), state, state.clone(), torch.ones(2),
                        ModelContext(), m.spec.train_binding_hash)
    loss = local_velocity_nll(m, b)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in m.acceleration_model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert not torch.isfinite(torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients])))
    assert torch.isfinite(torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients])))
    for p in m.acceleration_model.parameters():
        p.grad = None
    before, rows, saved = m.checkpoint(), [], []
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *_a, **_k: pytest.fail("nonfinite norm reached Adam"))
    with pytest.raises(ModelContractError, match="NONFINITE: training gradient norm"):
        fit_o1(m, [b], O1Plan(max_steps=1, patience=1, fit_diffusion=False), progress=rows.append,
               checkpoint_requested=lambda: True, checkpoint_handler=lambda *args: saved.append(args))
    assert m.checkpoint() == before and not rows and not saved


def test_positive_diffusion_parameterization_and_gradient():
    factor = torch.tensor([[.2, 0.], [.03, .4]], dtype=torch.float64)
    fitted = VelocityCholesky(factor, .01)
    assert torch.allclose(fitted.factor() @ fitted.factor().T, factor @ factor.T, atol=1e-14)
    with torch.no_grad():
        fitted.raw_diagonal.fill_(-1e3)
    assert (torch.diag(fitted.factor()) >= .01).all()
    with pytest.raises(ModelContractError):
        VelocityCholesky(factor, .5)
    m = model("M2")
    value = local_velocity_nll(m, batch(m, 8), factor)
    value.backward()
    assert m.acceleration_model.layers[-1].weight.grad.norm() > 0


def test_brownian_sample_ids_replay_across_chunks_and_models():
    m = model("M0")
    req = request()
    whole = brownian_increments(req, m, 0, 128)
    pieces = torch.cat([brownian_increments(req, m, first, min(first + 17, 128)) for first in range(0, 128, 17)])
    assert torch.equal(whole, pieces)
    assert torch.equal(whole, brownian_increments(req, model("M2"), 0, 128))
    first = forecast(m, req)
    second = forecast(m, replace(req, chunk_size=17))
    assert first["sample_ids"] == second["sample_ids"] == list(range(128))
    assert torch.allclose(first["samples"], second["samples"], atol=1e-14, rtol=1e-14)
    assert first["failure_rate"] == 0


def test_causal_request_rejects_future_cutoff_and_unbounded_allocations():
    m = model("M0")
    for bad in (replace(request(), history_cutoff=1.), replace(request(), time_grid=(0., 0.)),
                replace(request(), chunk_size=257), replace(request(), sample_count=100001)):
        with pytest.raises(ModelContractError):
            forecast(m, bad)
    with pytest.raises(ModelContractError, match="INTERRUPTED"):
        forecast(m, request(), cancellation=lambda: True)


def test_path_failures_keep_original_denominator_and_sample_identity():
    m = model("M0")
    result = forecast(m, replace(request(), maximum_state_norm=.01))
    assert result["valid_paths"] == 0 and result["failure_rate"] == 1
    assert result["failed_sample_ids"] == list(range(128))
    report = evaluate_forecast(result, torch.zeros((4, 4), dtype=torch.float64))
    assert report["status"] == "UNAVAILABLE" and report["metrics"] is None
    successful = forecast(m, request())
    partial = {**successful, "samples": successful["samples"][:-1], "valid_paths": 127,
               "sample_ids": list(range(127)), "failed_sample_ids": [127], "failure_rate": 1 / 128}
    report = evaluate_forecast(partial, torch.zeros((4, 4), dtype=torch.float64))
    assert report["status"] == "PARTIAL" and report["requested_paths"] == 128
    assert report["evaluation_population"] == "surviving-paths"


def test_em_moments_agree_with_affine_oracle_at_fine_step():
    m = model("M0")
    times = tuple(i * .01 for i in range(51))
    req = request(4096, 128, times)
    result = forecast(m, req)
    initial = torch.tensor(req.initial_state, dtype=torch.float64)
    exact = AffinePhaseSpaceOracle(m).exact_transition(initial, .5, ModelContext())
    endpoint = result["samples"][:, -1]
    se = torch.diag(exact.covariance).sqrt() / math.sqrt(4096)
    assert ((endpoint.mean(0) - exact.mean).abs() < 6 * se + .003).all()
    assert torch.allclose(torch.cov(endpoint.T), exact.covariance, atol=.004, rtol=.06)


def test_energy_u_statistic_matches_enumerated_pairs_and_has_gradient():
    samples = torch.tensor([[0., 0.], [1., 0.], [0., 2.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([.2, .3], dtype=torch.float64)
    expected = sum((x - target).norm() for x in samples) / 3
    expected -= sum((samples[i] - samples[j]).norm() for i in range(3) for j in range(3) if i != j) / 12
    actual, metadata = energy_score_value(samples, target)
    assert torch.allclose(actual, expected)
    assert metadata == {"estimator_id": "energy-u-exact-v1", "pair_count": 6}
    assert torch.autograd.gradcheck(lambda values: energy_score_value(values, target)[0], (samples,))


def test_large_sample_score_requires_registered_pair_recipe_and_replays():
    samples = torch.linspace(-1., 1., 300, dtype=torch.float64)[:, None]
    target = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ModelContractError):
        energy_score_value(samples, target)
    first, meta = energy_score_value(samples, target, pair_seed=21)
    second, meta2 = energy_score_value(samples, target, pair_seed=21)
    assert torch.equal(first, second) and meta == meta2
    assert meta["pair_sampling_se"] > 0 and meta["pair_count"] == 65536


def test_evaluator_reports_probability_and_point_metrics_without_fabricated_density():
    result = forecast(model("M0"), request())
    truth = result["samples"].mean(0).detach()
    report = evaluate_forecast(result, truth)
    assert report["status"] == "SUCCEEDED"
    assert report["ade"] == pytest.approx(0., abs=1e-15) and report["fde"] == pytest.approx(0., abs=1e-15)
    assert report["requested_paths"] == 128
    for row in report["rows"]:
        assert row["metrics"]["held_out_nll"] is None
        assert len(row["metrics"]["marginal_crps"]) == 2
        assert set(row["intervals"]) == {"0.5", "0.8", "0.95"}
        assert row["metrics"]["energy_score_jackknife_se"] >= 0


def test_coupled_refinement_improves_deterministic_affine_error():
    m = model("M0")
    m.velocity_factor.zero_()
    req = request(4, 4, (0., .2, .4, .6))
    levels = forecast_coupled_levels(m, req)
    exact = AffinePhaseSpaceOracle(m).exact_transition(torch.tensor(req.initial_state, dtype=torch.float64), .6, ModelContext())
    coarse = levels["coarse"]["samples"][0, -1]
    fine = levels["fine"]["samples"][0, -1]
    assert (fine - exact.mean).norm() < (coarse - exact.mean).norm()
    assert levels["paired_sample_ids"] == [0, 1, 2, 3]
    assert levels["coarse"]["error_budget"]["time_discretization_sensitivity"] is not None
    assert levels["coarse"]["error_budget"]["mean_state_standard_error"][-1] == [0., 0., 0., 0.]


def test_brownian_refinement_preserves_terminal_free_velocity():
    m = PhaseSpaceSDE(AffineAccelerationDrift(), torch.eye(2) * .2, spec()).double()
    levels = forecast_coupled_levels(m, request(32, 16))
    assert torch.allclose(levels["coarse"]["samples"][:, -1, 2:], levels["fine"]["samples"][:, -1, 2:], atol=1e-15)
    assert levels["coarse"]["brownian_reference_grid"] == levels["fine"]["time_grid"]


def test_common_rollout_energy_gradient_matches_paired_finite_difference():
    m = model("M2")
    req = request(16, 16)
    target = torch.tensor([.3, -.1], dtype=torch.float64)
    bias = m.acceleration_model.layers[-1].bias
    value, _ = energy_score_value(forecast(m, req)["samples"][:, -1, :2], target)
    value.backward()
    gradient = bias.grad[0].item()
    with torch.no_grad():
        bias[0] += 1e-5
        upper = energy_score_value(forecast(m, req)["samples"][:, -1, :2], target)[0].item()
        bias[0] -= 2e-5
        lower = energy_score_value(forecast(m, req)["samples"][:, -1, :2], target)[0].item()
        bias[0] += 1e-5
    assert gradient == pytest.approx((upper - lower) / 2e-5, abs=1e-7)
    assert abs(gradient) > 1e-6


@pytest.mark.parametrize("family", ["M1-R", "M1-S"])
def test_streaming_basis_fit_recovers_same_coefficients_for_different_chunks(family):
    m = model(family)
    drift = m.acceleration_model
    t = torch.linspace(0., 10., 101, dtype=torch.float64)
    z = torch.zeros((101, 4), dtype=torch.float64)
    z[:, 2] = torch.linspace(-1.8, 1.8, 101, dtype=torch.float64)
    z[:, 3] = .4 * torch.cos(2 * z[:, 2])
    context = ModelContext()
    c = z.new_empty((len(z), 0))
    basis = drift.basis(drift.selected_features(z, c))
    truth = torch.linspace(-.2, .3, len(drift.coefficients) * 2, dtype=torch.float64).reshape(-1, 2)
    acceleration = drift.affine(t, z, c) + basis @ truth
    dt = torch.linspace(.05, .15, 101, dtype=torch.float64)
    target = z.clone()
    target[:, 2:] += acceleration.detach() * dt[:, None]
    b = TransitionBatch(t, z, target, dt, context, m.spec.train_binding_hash)
    result = fit_residual_basis(m, [b], ridge=0., condition_number_max=1e6)
    assert torch.allclose(drift.coefficients, truth, atol=1e-10)
    first = drift.coefficients.detach().clone()
    pieces = [TransitionBatch(t[start:start+13], z[start:start+13], target[start:start+13], dt[start:start+13],
                             context, m.spec.train_binding_hash) for start in range(0, 101, 13)]
    fit_residual_basis(m, pieces, ridge=0., condition_number_max=1e6)
    assert torch.allclose(drift.coefficients, first, atol=1e-10)
    assert result["effective_degrees_of_freedom"] == len(truth)
    with pytest.raises(ModelContractError, match="ILL_CONDITIONED"):
        fit_residual_basis(m, [replace(b, state=z * 0)], ridge=0., condition_number_max=1e6)
