"""Analytic and software-only oracles; no empirical fit/data/qualification."""
from dataclasses import replace

import numpy as np
import pytest

from experiments.nex326.cohort import Segment
from experiments.nex326.model import ModelState, feature_vector
from experiments.pirc17.method_rollout import (
    INTEGRATORS, MODEL_KINDS, MethodDynamics, affine_kernel, causal_features, forecast_method,
)
from experiments.pirc17 import method_rollout
from experiments.pirc17.origins import VelocityPrior, causal_prefix, known_velocity
from experiments.pirc17.workload import method_inventory


def fitted(kind="single_gaussian", *, names=(), noise=1., linear=None, drift=(.3, -.2)):
    count = 1 if kind in {"single_gaussian", "explicit_decomp"} else 3
    features = (6 if kind == "explicit_decomp" else 3) + len(names)
    weights = np.zeros((features, 2))
    weights[0] = drift
    weights[1:3] = np.zeros((2, 2)) if linear is None else np.asarray(linear).T
    return ModelState(kind, tuple(names), tuple(weights.copy() for _ in range(count)),
                      tuple(np.eye(2) * noise for _ in range(count)),
                      np.full(count, 1 / count), 8, 0., 0., 1.)


def bound(model=None, tau=1.):
    return MethodDynamics.bind(fitted() if model is None else model,
        reference_interval_seconds=tau, fit_identity="software-fixture-not-research",
        noise_binding_rationale="software test: match R*tau**2 over the stated reference interval")


def origin():
    return causal_prefix([[-1., -.5], [0., 0.], [1., .5]], [-2., -1., 0.])


def forecast(model=None, start=None, horizons=(.25, 1., 1.5, 2.), **changes):
    settings = dict(propagation="fp", integrator="split", particles=16, seed=11,
        max_step_seconds=.5, history_step_seconds=1., max_steps=128, max_particle_steps=4096,
        origin_id="software-origin", run_id="software-run")
    settings.update(changes)
    return forecast_method(bound() if model is None else model, origin() if start is None else start,
                           horizons, **settings)


@pytest.mark.parametrize("integrator", sorted(INTEGRATORS))
@pytest.mark.parametrize("step", [.25, .5, 2.])
def test_fixed_diffusion_not_rescaled_by_numerical_substeps(integrator, step):
    # tau=2 is model calibration, h is numerical. Exact constant-drift oracle.
    dynamics = bound(fitted(noise=3.), tau=2.)
    result = forecast(dynamics, integrator=integrator, max_step_seconds=step)
    times = result.forecast.elapsed_seconds
    expected_mean = origin().position_m + times[:, None] * np.array([.3, -.2])
    expected_cov = times[:, None, None] * (3. * 2. * np.eye(2))
    np.testing.assert_allclose(result.conditional_means_m,
        np.broadcast_to(expected_mean, result.conditional_means_m.shape), atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.conditional_covariances_m2,
        np.broadcast_to(expected_cov, result.conditional_covariances_m2.shape), atol=2e-12, rtol=0)
    assert result.diagnostics["dynamics"]["reference_interval_seconds"] == 2.
    assert not result.diagnostics["dynamics"]["embedding_is_continuous_time_fit_qualification"]
    assert not result.diagnostics["numerically_qualified"]
    assert not result.diagnostics["scientific_claim_authorized"]


def test_exact_affine_kernel_matches_correlated_diagonal_ou_oracle():
    rates = np.array([-.2, -.6])
    noise = np.array([[2., .3], [.3, 1.]])
    time = 1.7
    f, j, covariance = affine_kernel(np.diag(rates), noise, time, "exact")
    sums = rates[:, None] + rates[None, :]
    np.testing.assert_allclose(f, np.diag(np.exp(rates * time)), atol=2e-14, rtol=0)
    np.testing.assert_allclose(j, np.diag(np.expm1(rates * time) / rates), atol=2e-14, rtol=0)
    np.testing.assert_allclose(covariance, noise * np.expm1(sums * time) / sums, atol=2e-14, rtol=0)


def test_exact_affine_kernel_matches_integrated_brownian_shear_not_euler():
    time = 1.7
    linear = np.array([[0., 1.], [0., 0.]])
    noise = np.diag([0., 1.])
    f, j, covariance = affine_kernel(linear, noise, time, "exact")
    np.testing.assert_allclose(f, [[1., time], [0., 1.]], atol=2e-14, rtol=0)
    np.testing.assert_allclose(j, [[time, time**2 / 2], [0., time]], atol=2e-14, rtol=0)
    np.testing.assert_allclose(covariance,
        [[time**3 / 3, time**2 / 2], [time**2 / 2, time]], atol=2e-14, rtol=0)
    assert not np.allclose(covariance, affine_kernel(linear, noise, time, "euler_maruyama")[2])


def test_exact_kernel_semigroup_and_split_em_weak_convergence_without_particles():
    linear = np.array([[-.4, .2], [0., -.1]])
    noise = np.array([[1., .2], [.2, 2.]])
    f, j, covariance = affine_kernel(linear, noise, 1., "exact")
    half_f, half_j, half_cov = affine_kernel(linear, noise, .5, "exact")
    np.testing.assert_allclose(half_f @ half_f, f, atol=3e-14, rtol=0)
    np.testing.assert_allclose((np.eye(2) + half_f) @ half_j, j, atol=3e-14, rtol=0)
    np.testing.assert_allclose(half_f @ half_cov @ half_f.T + half_cov, covariance, atol=3e-14, rtol=0)
    for integrator, minimum_ratio in [("split", 3.5), ("euler_maruyama", 1.8)]:
        errors = []
        for count in (4, 8):
            step_f, _, step_cov = affine_kernel(linear, noise, 1 / count, integrator)
            propagated = np.zeros((2, 2))
            for _ in range(count):
                propagated = step_f @ propagated @ step_f.T + step_cov
            errors.append(np.linalg.norm(propagated - covariance))
        assert errors[0] / errors[1] > minimum_ratio


@pytest.mark.parametrize("kind", sorted(MODEL_KINDS))
def test_causal_feature_columns_match_legacy_training_columns(kind):
    positions = np.array([[1., 0.], [2., 1.], [4., 2.], [7., 4.]])
    time = np.array([1., 2., 4., 7.])
    segment = Segment("fixture", "fixture", "none", time, positions,
                      {"solar_elev": np.array([.1, .2, .3, .4])}, False)
    model = fitted(kind, names=("solar_elev",))
    velocity = (positions[2] - positions[1]) / (time[2] - time[1])
    actual = causal_features(model, positions[2:3], velocity[None], np.array([[.3]]))[0]
    np.testing.assert_array_equal(actual, feature_vector(segment, 2, model.condition_names, kind))


@pytest.mark.parametrize("integrator", sorted(INTEGRATORS))
@pytest.mark.parametrize("propagation", ["fp", "mc", "crn"])
def test_explicit_decomposition_forecast_cannot_read_hidden_future(integrator, propagation):
    states = np.column_stack((np.arange(8, dtype=float), np.zeros(8)))
    hidden_change = states.copy()
    hidden_change[4, 0] += 100.
    times = np.arange(8, dtype=float)
    original = causal_prefix(states[:4], times[:4])
    changed = causal_prefix(hidden_change[:4], times[:4])
    model = fitted("explicit_decomp", noise=.01, drift=(0., 0.))
    weights = model.weights[0].copy()
    weights[3, 0] = 1.
    dynamics = bound(replace(model, weights=(weights,)))
    settings = dict(integrator=integrator, propagation=propagation,
                    crn_pair_id="pair" if propagation == "crn" else None)
    a = forecast(dynamics, original, horizons=(1., 2., 3., 4.), **settings)
    b = forecast(dynamics, changed, horizons=(1., 2., 3., 4.), **settings)
    np.testing.assert_array_equal(a.forecast.positions_m, b.forecast.positions_m)
    np.testing.assert_array_equal(states[4], [4., 0.])
    assert a.forecast.positions_m.shape == (16, 4, 2)
    assert not a.forecast.positions_m.flags.writeable


@pytest.mark.parametrize("kind", sorted(MODEL_KINDS))
@pytest.mark.parametrize("propagation", ["fp", "mc", "crn"])
def test_all_model_routes_keep_physical_mode_and_history_clocks(kind, propagation):
    dynamics = bound(fitted(kind))
    settings = dict(propagation=propagation, crn_pair_id="pair" if propagation == "crn" else None,
                    horizons=(.4, 1.2, 2.2, 3.2), history_step_seconds=.8)
    coarse = forecast(dynamics, max_step_seconds=.5, **settings)
    fine = forecast(dynamics, max_step_seconds=.25, **settings)
    for result in (coarse, fine):
        np.testing.assert_allclose(result.diagnostics["history_ticks_seconds"], [.8, 1.6, 2.4, 3.2])
        switching = kind in {"pointwise_mixture", "gmm_kernel"}
        assert result.diagnostics["mode_resampling_times_seconds"] == ([1., 2., 3.] if switching else [])
        assert result.diagnostics["mode_change_opportunities"] == (48 if switching else 0)
    assert coarse.diagnostics["mode_changes"] == fine.diagnostics["mode_changes"]


def test_latent_model_routes_are_not_silently_collapsed_to_same_mode_process():
    persistent = fitted("seg_constant_mode", noise=0., drift=(0., 0.))
    weights = tuple(w.copy() for w in persistent.weights)
    for w, speed in zip(weights, [-1., 0., 1.]):
        w[0, 0] = speed
    persistent = replace(persistent, weights=weights)
    switching = replace(persistent, model_kind="pointwise_mixture")
    fixed = forecast(bound(persistent), horizons=(1., 2., 3., 4.))
    fresh = forecast(bound(switching), horizons=(1., 2., 3., 4.))
    assert fixed.diagnostics["mode_changes"] == 0
    assert fresh.diagnostics["mode_changes"] > 0
    assert not np.array_equal(fixed.forecast.positions_m, fresh.forecast.positions_m)


def test_crn_really_shares_noise_with_paired_fp_control_and_mc_remains_independent():
    control = bound(fitted(drift=(0., 0.)))
    candidate = bound(fitted(drift=(.2, -.1)))
    a = forecast(control, propagation="fp", crn_pair_id="paired-test", run_id="control")
    b = forecast(candidate, propagation="crn", crn_pair_id="paired-test", run_id="candidate")
    expected = b.forecast.elapsed_seconds[:, None] * np.array([.2, -.1])
    difference = b.forecast.positions_m - a.forecast.positions_m
    np.testing.assert_allclose(difference, np.broadcast_to(expected, difference.shape), atol=2e-14, rtol=0)
    independent = forecast(candidate, propagation="mc", run_id="independent-candidate")
    assert np.var(independent.forecast.positions_m - a.forecast.positions_m) > np.var(difference)
    assert a.diagnostics["random_stream_sha256"] == b.diagnostics["random_stream_sha256"]
    assert independent.diagnostics["random_stream_sha256"] != b.diagnostics["random_stream_sha256"]
    assert not b.diagnostics["variance_reduction_qualified"]  # Oracle is not empirical qualification.


def test_condition_provider_receives_only_readonly_predicted_positions_and_epoch():
    calls = []
    model = fitted(names=("solar_elev",), noise=0.)
    weights = model.weights[0].copy()
    weights[-1, 0] = .1
    dynamics = bound(replace(model, weights=(weights,)))
    def provider(positions, epoch):
        assert not positions.flags.writeable
        calls.append((positions.copy(), epoch))
        return np.full((len(positions), 1), .3)
    result = forecast(dynamics, condition_names=("solar_elev",), condition_at=provider)
    assert len(calls) == result.diagnostics["integration_steps"]
    np.testing.assert_array_equal(calls[0][0], np.broadcast_to(origin().position_m, (16, 2)))
    assert calls[0][1] == origin().epoch_seconds
    assert all(time < origin().epoch_seconds + 2. for _, time in calls)
    assert result.forecast.feature_query_rows == len(calls) * 16


@pytest.mark.parametrize("mode", ["known_velocity", "point_only"])
def test_other_origin_modes_are_explicit_and_use_no_hidden_history(mode):
    if mode == "known_velocity":
        start = known_velocity([2., 3.], 100., [1., 0.], source="software-observation")
    else:
        start = VelocityPrior([[1., 0.], [0., 1.]], "software-train-only").at([2., 3.], 100.)
    result = forecast(bound(fitted("explicit_decomp")), start)
    assert result.forecast.positions_m.shape == (16, 4, 2)
    assert result.diagnostics["origin_mode"] == mode
    assert result.diagnostics["velocity_source"] == start.velocity_source
    assert result.diagnostics["velocity_error_mps"] is None


def test_particle_prefix_and_origin_stream_separation_and_no_model_mutation():
    original = fitted()
    dynamics = bound(original)
    original.weights[0][:] = 99.  # The bound copy must remain unchanged.
    a, b = forecast(dynamics, particles=8), forecast(dynamics, particles=16)
    np.testing.assert_array_equal(a.forecast.positions_m, b.forecast.positions_m[:8])
    c = forecast(dynamics, particles=8, origin_id="different-origin")
    assert not np.array_equal(a.forecast.positions_m, c.forecast.positions_m)


@pytest.mark.parametrize("change", [
    {"particles": 0}, {"particles": True}, {"max_step_seconds": 0},
    {"history_step_seconds": float("nan")}, {"seed": -1}, {"seed": True},
    {"propagation": "unknown"}, {"integrator": "euler"},
    {"propagation": "crn"}, {"propagation": "mc", "crn_pair_id": "bad"},
    {"max_steps": 1}, {"max_particle_steps": 1}, {"origin_id": ""}, {"run_id": ""},
    {"condition_names": ("solar_elev",)}, {"crn_pair_id": ""},
])
def test_invalid_or_expanded_contracts_fail_without_retry(change):
    with pytest.raises(ValueError):
        forecast(**change)


def test_rejects_trajectory_container_and_bad_named_condition_values():
    with pytest.raises(ValueError, match="target-free Origin"):
        forecast(start=object())
    dynamics = bound(fitted(names=("solar_elev",)))
    for value in (np.zeros((16, 2)), np.full((16, 1), np.nan)):
        with pytest.raises(ValueError):
            forecast(dynamics, condition_names=("solar_elev",), condition_at=lambda state, time: value)


@pytest.mark.parametrize("covariance", [np.diag([-1., 1.]), [[1., 1.], [0., 1.]], np.full((2, 2), np.nan)])
def test_bad_noise_rejected_not_clipped_or_regularized(covariance):
    with pytest.raises(ValueError):
        affine_kernel(np.zeros((2, 2)), covariance, 1., "exact")


def test_model_binding_requires_units_provenance_and_frozen_family_shapes():
    model = fitted()
    for changes in ({"reference_interval_seconds": 0}, {"fit_identity": ""}, {"noise_binding_rationale": ""}):
        settings = dict(reference_interval_seconds=1., fit_identity="fixture", noise_binding_rationale="oracle")
        settings.update(changes)
        with pytest.raises(ValueError):
            MethodDynamics.bind(model, **settings)
    with pytest.raises(ValueError, match="mode count"):
        bound(replace(model, model_kind="pointwise_mixture"))
    with pytest.raises(ValueError, match="probabilities"):
        bound(replace(model, mode_probabilities=np.array([.5])))
    bound_model = bound(model)
    bound_model.model.weights[0].setflags(write=True)
    bound_model.model.weights[0][0, 0] += 1.
    with pytest.raises(ValueError, match="mutated"):
        forecast(bound_model)


REQUIRED_SLOTS = [row for row in method_inventory()["slots"] if row["disposition"] == "REQUIRED"]


@pytest.mark.parametrize("slot", REQUIRED_SLOTS, ids=lambda row: row["slot_id"])
def test_every_required_method_slot_has_a_causal_software_route(slot):
    """Dispatch coverage only: does not claim a trained slot or formal result."""
    config = slot["components"]
    assert config["bridge"] == "none" and config["state_dim"] == 2
    names = tuple(config["condition"])
    model = fitted(config["model"], names=names)
    dynamics = bound(model, tau=config["dt_seconds"])
    result = forecast(dynamics, propagation=config["poa"], integrator=config["integrator"],
        run_id=slot["slot_id"], condition_names=names,
        condition_at=(lambda positions, time: np.zeros((len(positions), len(names)))) if names else None,
        crn_pair_id="software-paired-control" if config["poa"] == "crn" else None)
    assert result.forecast.positions_m.shape == (16, 4, 2)
    assert result.diagnostics["run_id"] == slot["slot_id"]
    assert result.diagnostics["model_kind"] == config["model"]
    assert result.diagnostics["condition_names"] == config["condition"]
    assert result.diagnostics["dynamics"]["reference_interval_seconds"] == config["dt_seconds"]
    assert not result.diagnostics["scientific_claim_authorized"]


def test_manual_invalid_binding_cannot_bypass_frozen_family_validation():
    good = bound()
    invalid = replace(good, model=replace(good.model, model_kind="pointwise_mixture"))
    with pytest.raises(ValueError, match="mode count"):
        forecast(invalid)


def test_same_run_label_does_not_accidentally_pair_ordinary_mc_with_fp():
    fp = forecast(propagation="fp")
    mc = forecast(propagation="mc")
    assert fp.diagnostics["random_stream_sha256"] != mc.diagnostics["random_stream_sha256"]


@pytest.mark.parametrize("integrator", sorted(INTEGRATORS))
@pytest.mark.parametrize("propagation", ["fp", "mc", "crn"])
def test_returned_particles_obey_kernel_moments_not_just_reported_diagnostics(monkeypatch, integrator, propagation):
    # Deterministic algebraic cubature: zero mean and identity population second
    # moment, NOT claimed to be random empirical or Gaussian research samples.
    basis = np.sqrt(2.) * np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    original_rng = method_rollout._rng
    class Cubature:
        def standard_normal(self, shape):
            assert shape == (4, 2)
            return basis.copy()
    def factory(seed, key, purpose):
        return Cubature() if isinstance(purpose, list) and purpose[0] == "normal" else original_rng(seed, key, purpose)
    monkeypatch.setattr(method_rollout, "_rng", factory)
    linear = np.diag([-.2, -.6])
    dynamics = bound(fitted(linear=linear, noise=3.))
    result = forecast(dynamics, horizons=(1.,), particles=4, max_step_seconds=1.,
        propagation=propagation, integrator=integrator,
        crn_pair_id="moment-oracle" if propagation == "crn" else None)
    f, j, covariance = affine_kernel(linear, np.eye(2) * 3., 1., integrator)
    expected = f @ origin().position_m + j @ np.array([.3, -.2])
    samples = result.forecast.positions_m[:, 0]
    np.testing.assert_allclose(samples.mean(axis=0), expected, atol=2e-14, rtol=0)
    centered = samples - expected
    np.testing.assert_allclose(centered.T @ centered / 4, covariance, atol=2e-14, rtol=0)


def test_fp_mean_closure_is_distinct_from_mc_particle_feature_evaluation():
    model = fitted(names=("solar_elev",), noise=1., drift=(0., 0.))
    weights = model.weights[0].copy()
    weights[-1, 0] = .1
    dynamics = bound(replace(model, weights=(weights,)))
    observed = {}
    results = {}
    for route in ("fp", "crn"):
        calls = []
        def nonlinear(positions, time):
            calls.append(positions.copy())
            return positions[:, :1] ** 2
        results[route] = forecast(dynamics, propagation=route, crn_pair_id="same-noise",
            condition_names=("solar_elev",), condition_at=nonlinear)
        observed[route] = calls
    np.testing.assert_array_equal(observed["fp"][0], observed["crn"][0])
    assert np.max(np.std(observed["fp"][1], axis=0)) < 1e-12
    assert np.max(np.std(observed["crn"][1], axis=0)) > .1
    assert not np.array_equal(results["fp"].forecast.positions_m, results["crn"].forecast.positions_m)
    assert "assumed-density closure" in results["fp"].diagnostics["fp_semantics"]
    assert not results["fp"].diagnostics["nonlinear_exactness_claimed"]
