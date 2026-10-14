"""Synthetic, analytic and negative diagnostic oracles; not research data."""
import copy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from experiments.nex326.model import ModelState
from experiments.pirc17 import method_mechanisms as mm
from experiments.pirc17 import cached_method_mechanisms as cached
from experiments.pirc17.inference import SEEDS
from experiments.pirc17.method_rollout import MethodDynamics, forecast_method
from experiments.pirc17.origins import known_velocity
from experiments.pirc17.workload import method_inventory

CONTEXT = "a"*64


def dynamics(slot="arm-01/full", *, linear=-.02):
    c = next(r["components"] for r in method_inventory()["slots"] if r["slot_id"] == slot)
    n = 1 if c["model"] in {"single_gaussian", "explicit_decomp"} else 3
    width = (6 if c["model"] == "explicit_decomp" else 3)+len(c["condition"])
    weights = np.zeros((width, 2))
    weights[0] = [.3, -.2]
    weights[1:3] = linear*np.eye(2)
    weights[-1] = [.01, .02]
    model = ModelState(c["model"], tuple(c["condition"]), tuple(weights.copy() for _ in range(n)),
        tuple(np.eye(2) for _ in range(n)), np.full(n, 1/n), 24, 2., 1., 1.,
        meta_task_count=3, adaptation_parameter_delta=.1, estimator_method=c["estimator"],
        transfer_method=c["transfer"], finetune_method=c.get("finetune", "all"))
    return MethodDynamics.bind(model, reference_interval_seconds=c["dt_seconds"],
        fit_identity="fixture-fit-not-empirical", noise_binding_rationale="synthetic test only")


def forecast(*, integrator="split", propagation="fp", pair="fixture-pair", seed=SEEDS[0], origin_id="origin-0", **kwargs):
    d = dynamics()
    return forecast_method(d, known_velocity([0., 0.], 0., [1., 0.], source="synthetic-fixture"),
        [1., 2., 3., 4.], propagation=propagation, integrator=integrator, particles=8, seed=seed,
        max_step_seconds=1., history_step_seconds=60., max_steps=16, max_particle_steps=128,
        origin_id=origin_id, run_id=propagation, crn_pair_id=pair,
        condition_names=("solar_elev",), condition_at=lambda x, t: np.zeros((len(x), 1)), **kwargs)


def test_registry_preserves_all_slots_and_historical_thresholds():
    result = mm.mechanism_registry()
    assert result["counts"] == {"slots": 36, "required": 28, "excluded": 8}
    assert len(result["slots"]) == 36
    assert sum(r["disposition"] == "EXCLUDED" for r in result["slots"].values()) == 8
    for row in result["slots"].values():
        if row["disposition"] == "REQUIRED":
            assert row["operator"] == row["historical_gate"]["operator"]
            assert row["threshold"] == row["historical_gate"]["threshold"]
    assert result["score_draws_per_time"] == 256
    assert result["quadrature_order"] == 12
    assert result["gate_is_global_numerical_qualification"] is False
    assert result["final_eval_authorized"] is False
    workload = result["pre_seal_workload_supplement"]
    assert workload["additional_exact_kernel_forecasts"] == 290
    assert workload["additional_scoring_only_Gaussian_draws"] == 593920
    assert workload["total_stochastic_forecasts_including_audits"] == 11513
    assert workload["scientific_forecasts_unchanged"] == 11020
    assert sum(workload["formal_phase_caps_unchanged"].values()) == 48*3600


def test_registered_stream_pairs_are_real_and_mc_stays_independent():
    for names in mm.STREAM_GROUPS.values():
        pairs = {mm.forecast_stream_binding(n, "causal_prefix")["crn_pair_id"] for n in names}
        assert len(pairs) == 1 and None not in pairs
    assert mm.forecast_stream_binding("arm-21/mc", "causal_prefix")["crn_pair_id"] is None
    assert mm.forecast_stream_binding("arm-18/full", "causal_prefix")["crn_pair_id"] != mm.forecast_stream_binding("arm-18/full", "point_only")["crn_pair_id"]
    for slot in [*DIRECT_SLOTS, *mm.NUMERICAL_SLOTS, *mm.SCORE_SLOTS, *mm.VARIANCE_SLOTS]:
        assert mm.forecast_stream_binding(slot, "known_velocity")["forecast_reuse_authorized"] is False


DIRECT_SLOTS = [r["slot_id"] for r in method_inventory()["slots"] if r["disposition"] == "REQUIRED"
                and r["slot_id"] not in mm.NUMERICAL_SLOTS | mm.SCORE_SLOTS | mm.VARIANCE_SLOTS]


@pytest.mark.parametrize("slot", DIRECT_SLOTS)
def test_all_direct_parameter_and_fit_gates_recompute(slot):
    d = dynamics(slot)
    gate = mm.model_gate(slot, d)
    mm.validate_gate(gate)
    assert gate["status"] == "computed"
    assert gate["evidence"]["model_sha256"] == d.model_sha256
    assert gate["scientific_claim_authorized"] is False
    altered = copy.deepcopy(gate)
    altered["passed"] = not gate["passed"]
    altered["sha256"] = mm._digest({k: v for k, v in altered.items() if k != "sha256"})
    with pytest.raises(ValueError, match="gate differs"):
        mm.validate_gate(altered)


def test_direct_gate_exact_values_and_honest_names():
    assert mm.model_gate("arm-01/full", dynamics())["value"] == pytest.approx(np.log(3))
    assert mm.model_gate("arm-02/pointwise", dynamics("arm-02/pointwise"))["value"] == pytest.approx(2/3)
    assert mm.model_gate("arm-07/full", dynamics())["value"] == 1.
    assert mm.model_gate("arm-08/qmle", dynamics("arm-08/qmle"))["value"] == -1.
    assert mm.model_gate("arm-12/scratch", dynamics("arm-12/scratch"))["value"] == 24
    assert mm.model_gate("arm-14/reptile", dynamics("arm-14/reptile"))["value"] == 3
    assert mm.model_gate("arm-09/mixed", dynamics("arm-09/mixed"))["units"] == "dimensionless_normalized_objective"
    assert mm.model_gate("arm-09/pure_es", dynamics("arm-09/pure_es"))["units"] == "m/s"
    assert "not independent PDE" in mm.model_gate("arm-20/full", dynamics())["scope_limit"]


def test_cannot_substitute_another_fit_or_source():
    with pytest.raises(ValueError, match="model/training"):
        mm.model_gate("arm-08/qmle", dynamics())
    with pytest.raises(ValueError, match="model/training"):
        mm.model_gate("arm-06/dt30", dynamics())
    with pytest.raises(ValueError, match="evidence source"):
        mm.model_gate("arm-18/full", dynamics())
    excluded = next(k for k, v in mm.mechanism_registry()["slots"].items() if v["disposition"] == "EXCLUDED")
    with pytest.raises(ValueError, match="exclusions"):
        mm.model_gate(excluded, dynamics())


def test_cached_model_restore_rejects_identity_mutation():
    d = dynamics()
    training = {"training_identity_sha256": "b"*64}
    identity = mm._digest({**training, "model": d.model.to_dict()})
    d = replace(d, fit_identity=identity)
    row = {"model": d.model.to_dict(), "training": training, "dynamics": d.identity()}
    assert cached.restore_cached_fit(row).identity() == d.identity()
    row["model"]["validation_objective_after"] = 0.
    with pytest.raises(ValueError, match="does not reproduce"):
        cached.restore_cached_fit(row)


def test_cached_input_hash_and_output_exclusivity(tmp_path, monkeypatch):
    path = tmp_path/"fixture.json"
    raw = b'{"fixture": true}'
    path.write_bytes(raw)
    assert cached._read(path, hashlib.sha256(raw).hexdigest()) == {"fixture": True}
    with pytest.raises(ValueError, match="hash changed"):
        cached._read(path, "0"*64)
    monkeypatch.setattr(cached, "OUTPUT", path)
    with pytest.raises(FileExistsError):
        cached.run()  # refuses before reading any actual research artifact


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None])
def test_nonfinite_or_missing_cannot_be_a_passing_zero(value):
    with pytest.raises(ValueError):
        mm._receipt("arm-18/full", value, {})


def test_gaussian_quadrature_matches_isotropic_center_oracle():
    # Exactly unit sample covariance; independent closed expectation for d=2.
    x = np.array([[np.sqrt(1.5), 0], [-np.sqrt(1.5), 0], [0, np.sqrt(1.5)], [0, -np.sqrt(1.5)]])
    score, _, _ = mm._gaussian_score(x, np.zeros(2))
    expected = np.sqrt(np.pi/2)*(1-1/np.sqrt(2))
    assert score == pytest.approx(expected, rel=.025)  # finite quadrature, not exact closed form


def test_score_diagnostic_is_bounded_reproducible_and_not_primary():
    rng = np.random.default_rng(2)
    paths = rng.normal(size=(512, 4, 2))
    target = np.zeros((4, 2))
    inputs = dict(seed=SEEDS[0], origin_id="fixture", input_identity_sha256=CONTEXT)
    a = mm.score_diagnostic("arm-10/d2_mc", paths, target, **inputs)
    b = mm.score_diagnostic("arm-10/d2_closed", paths, target, **inputs)
    assert a["evidence"] == b["evidence"]  # identical shared standard draws, no extra forecast
    assert a == mm.score_diagnostic("arm-10/d2_mc", paths, target, **inputs)
    assert a["evidence"]["extra_forecasts"] == 0
    assert a["evidence"]["diagnostic_draws_per_time"] == 256
    assert len(a["evidence"]["per_time"]) == 4
    assert a["value"] == max(r["relative_error"] for r in a["evidence"]["per_time"])
    for t, row in enumerate(a["evidence"]["per_time"]):
        x = paths[:256, t]
        expected = np.linalg.norm(x-target[t], axis=1).mean()-.5*np.linalg.norm(x-np.roll(x, 1, axis=0), axis=1).mean()
        assert row["saved_forecast_cyclic_es_256_m"] == expected
    mm.validate_gate(a)


@pytest.mark.parametrize("bad", ["particles", "nonfinite", "times", "seed", "context"])
def test_score_rejects_unregistered_or_bad_inputs(bad):
    paths, target = np.zeros((512, 4, 2)), np.zeros((4, 2))
    inputs = dict(seed=SEEDS[0], origin_id="fixture", input_identity_sha256=CONTEXT)
    if bad == "particles": paths = paths[:256]
    if bad == "times": target = target[:1]
    if bad == "nonfinite": paths[0, 0, 0] = np.nan
    if bad == "seed": inputs["seed"] = 1
    if bad == "context": inputs["input_identity_sha256"] = "unbound"
    with pytest.raises(ValueError):
        mm.score_diagnostic("arm-10/d2_mc", paths, target, **inputs)


@pytest.fixture(scope="module")
def integrations():
    return {key: forecast(integrator=key) for key in ("split", "euler_maruyama", "exact")}


@pytest.mark.parametrize("slot,integrator", [("arm-18/full", "split"), ("arm-19/em", "euler_maruyama"), ("arm-19/euler", "euler_maruyama")])
def test_full_path_integration_mechanism_matches_isotropic_gaussian_formula(integrations, slot, integrator):
    a, r = integrations[integrator], integrations["exact"]
    gate = mm.integration_diagnostic(slot, a, r, approximate_context_sha256=CONTEXT, reference_context_sha256=CONTEXT)
    # These fixtures have isotropic covariance, so its root is sqrt(C[0,0])*I.
    mean_squared = np.sum((a.conditional_means_m-r.conditional_means_m)**2, axis=2)
    root_squared = 2*(np.sqrt(a.conditional_covariances_m2[:, :, 0, 0])-np.sqrt(r.conditional_covariances_m2[:, :, 0, 0]))**2
    expected = np.sqrt(np.mean(mean_squared+root_squared, axis=0)).max()
    assert gate["value"] == pytest.approx(expected, abs=1e-14, rel=1e-12)
    mm.validate_gate(gate)
    assert gate["evidence"]["global_exact_nonlinear_solution"] is False
    if integrator == "split":
        assert max(gate["evidence"]["per_time_rms_conditional_mean_error_m"]) < 1e-12
        assert gate["value"] > 0  # Correct mean alone cannot certify covariance accuracy.
        assert max(gate["evidence"]["per_time_rms_covariance_difference_m2"]) > 0
    else:
        assert gate["value"] > 0


@pytest.mark.parametrize("bad", ["grid", "stream", "missing", "context", "reference", "model", "covariance"])
def test_numerical_check_rejects_fake_pairing(integrations, bad):
    a, r = copy.deepcopy(integrations["split"]), copy.deepcopy(integrations["exact"])
    context = CONTEXT
    if bad == "grid": r.diagnostics["max_step_seconds"] = .5
    if bad == "stream": r.diagnostics["random_stream_sha256"] = "b"*64
    if bad == "missing": r.diagnostics.pop("history_ticks_seconds")
    if bad == "context": context = "b"*64
    if bad == "reference": r.diagnostics["integrator"] = "euler_maruyama"
    if bad == "model": r.diagnostics["model_kind"] = "gmm_kernel"
    if bad == "covariance":
        covariance = r.conditional_covariances_m2.copy()
        covariance[0, 0] = -np.eye(2)
        r = replace(r, conditional_covariances_m2=covariance)
    with pytest.raises(ValueError):
        mm.integration_diagnostic("arm-18/full", a, r, approximate_context_sha256=CONTEXT, reference_context_sha256=context)


def variance_rows(blocks=2, origins=2, factor=.5):
    # Fixed synthetic score differences with analytically known variance ratio.
    baseline = forecast().diagnostics
    rows, expected = [], {}
    for b in range(blocks):
        expected[f"b{b}"] = [f"o{b}-{o}" for o in range(origins)]
        for o in expected[f"b{b}"]:
            for i, seed in enumerate(SEEDS):
                for slot, kind, value in [("arm-20/full", "fp", 100.),
                                          ("arm-21/mc", "mc", 100.+(b+1)*i),
                                          ("arm-21/crn", "crn", 100.+(b+1)*i*factor)]:
                    diag = copy.deepcopy(baseline)
                    diag.update(origin_id=o, seed=seed, propagation=kind, run_id=slot,
                        crn_pair_id=None if kind == "mc" else "pair",
                        random_stream_sha256=("b" if kind == "mc" else "a")*64)
                    rows.append({"block_id": f"b{b}", "origin_id": o, "seed": seed,
                        "slot_id": slot, "status": "success", "score_m": value,
                        "context_sha256": CONTEXT, "elapsed_seconds": [1., 2., 3., 4.],
                        "split": "validation", "diagnostics": diag})
    return rows, expected


@pytest.mark.parametrize("factor,ratio", [(.5, 4.), (1., 1.), (2., .25)])
def test_real_score_variance_not_fabricated_shared_draws(factor, ratio):
    rows, expected = variance_rows(factor=factor)
    result = mm.variance_diagnostic(rows, expected_origins_by_block=expected)
    for gate in result.values():
        assert gate["value"] == pytest.approx(ratio)
        assert gate["passed"] == (ratio >= 1)
        assert gate["evidence"]["independent_block_count"] == 2
        assert gate["evidence"]["seed_count"] == 5
        assert gate["evidence"]["descriptive_block_bootstrap_interval"] == pytest.approx([ratio, ratio])
        mm.validate_gate(gate)
    assert result["arm-21/mc"]["evidence"] == result["arm-21/crn"]["evidence"]
    assert result == mm.variance_diagnostic(iter(rows), expected_origins_by_block=expected)


def test_zero_variance_is_unavailable_not_infinite_improvement():
    rows, expected = variance_rows(factor=0)
    result = mm.variance_diagnostic(rows, expected_origins_by_block=expected)["arm-21/crn"]
    assert result["value"] is None and result["passed"] is None
    assert result["status"] == "unavailable"
    assert result["evidence"]["undefined_bootstrap_replicates"] == 2000
    mm.validate_gate(result)


@pytest.mark.parametrize("bad", ["missing", "failure", "duplicate", "stream", "seed", "split", "model", "context", "nonfinite", "provenance"])
def test_variance_rejects_pseudopairing_and_selection(bad):
    rows, expected = variance_rows()
    if bad == "missing": rows.pop()
    if bad == "failure": rows[0]["status"] = "failure"
    if bad == "duplicate": rows.append(copy.deepcopy(rows[0]))
    if bad == "stream": rows[2]["diagnostics"]["random_stream_sha256"] = "b"*64
    if bad == "seed": rows[2]["diagnostics"]["seed"] += 1
    if bad == "split": rows[2]["split"] = "final_eval"
    if bad == "model": rows[2]["diagnostics"]["dynamics"]["fit_identity"] = "different-fit"
    if bad == "context": rows[2]["context_sha256"] = "b"*64
    if bad == "nonfinite": rows[2]["score_m"] = float("nan")
    if bad == "provenance": rows[2]["diagnostics"].pop("mode_step_seconds")
    with pytest.raises(ValueError):
        mm.variance_diagnostic(rows, expected_origins_by_block=expected)
