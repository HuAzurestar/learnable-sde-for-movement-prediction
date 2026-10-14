"""Joint-refinement software oracles; these are not empirical research samples."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.pirc17 import contrast_refinement as joint
from experiments.pirc17 import direct_linear_rollout as engine
from experiments.pirc17.metrics import energy_score
from experiments.pirc17.precision import energy_precision
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows


def brute_scores(arrays, targets):
    scores = [np.array([energy_score(x[:, t], target) for t, target in enumerate(targets)]) for x in arrays]
    return (scores[0]-scores[1])-(scores[2]-scores[3])


def brute_precision(arrays, targets, weights, kind):
    groups = len(arrays[2])
    replicates = []
    for i in range(groups):
        reduced = [np.delete(x, np.arange(i, len(x), groups) if kind == "particles" else i, axis=0)
                   for x in arrays]
        replicates.append(brute_scores(reduced, targets))
    replicates = np.asarray(replicates)
    centered = replicates-replicates.mean(axis=0)
    covariance = (groups-1)/groups*centered.T@centered
    return brute_scores(arrays, targets), covariance, np.sqrt(weights@covariance@weights)


@pytest.mark.parametrize("kind,multiple", [("integration", 1), ("particles", 2), ("particles", 3)])
def test_every_joint_replicate_matches_independent_brute_force_oracle(kind, multiple):
    rng = np.random.default_rng(42)
    candidate = rng.normal(size=(5*multiple, 3, 2))
    control = candidate*.3+rng.normal(size=candidate.shape)
    if kind == "integration":
        arrays = [candidate, control, candidate*.9+.2, control*1.1-.1]
    else:
        arrays = [candidate, control, candidate[:5], control[:5]]
    targets, weights = rng.normal(size=(3, 2)), np.array([.2, .3, .5])
    expected, covariance, se = brute_precision(arrays, targets, weights, kind)
    report = joint.joint_precision(*arrays, targets, weights, kind=kind)
    np.testing.assert_allclose(report["by_time_energy_score_m"], expected, atol=1e-12)
    np.testing.assert_allclose(report["time_covariance_m2"], covariance, atol=1e-12)
    assert report["time_weighted_energy_score_m"] == pytest.approx(weights@expected, abs=1e-12)
    assert report["time_weighted_standard_error_m"] == pytest.approx(se, abs=1e-12)
    assert report["jackknife_group_count"] == 5 and not report["certified"]
    assert report["refined_paths_per_group"] == multiple
    assert report["sign_convention"] == joint.SIGN


@pytest.mark.parametrize("kind", ["integration", "particles"])
def test_common_model_noise_cancels_exactly_instead_of_adding_marginal_variances(kind):
    x = np.random.default_rng(8).normal(size=(12, 2, 2))
    reference = x[:6] if kind == "particles" else 2*x
    target, weights = np.zeros((2, 2)), [.5, .5]
    assert energy_precision(x, target, weights)["time_weighted_standard_error_m"] > 0
    result = joint.joint_precision(x, x, reference, reference, target, weights, kind=kind)
    assert result["time_weighted_standard_error_m"] == result["time_weighted_energy_score_m"] == 0
    np.testing.assert_array_equal(result["time_covariance_m2"], np.zeros((2, 2)))


def test_swapping_models_reverses_difference_but_preserves_joint_covariance():
    rng = np.random.default_rng(91)
    a, b, c, d = rng.normal(size=(4, 7, 2, 2))
    first = joint.joint_precision(a, b, c, d, np.zeros((2, 2)), [.4, .6], kind="integration")
    reverse = joint.joint_precision(b, a, d, c, np.zeros((2, 2)), [.4, .6], kind="integration")
    np.testing.assert_array_equal(first["time_covariance_m2"], reverse["time_covariance_m2"])
    np.testing.assert_array_equal(first["by_time_energy_score_m"], -np.asarray(reverse["by_time_energy_score_m"]))


@pytest.mark.parametrize("fault", ["kind", "count", "shape", "same_budget", "ratio", "candidate_prefix",
    "control_prefix", "nonfinite", "target", "weights", "dimensions", "empty_time"])
def test_invalid_or_uncoupled_four_way_arrays_are_rejected(fault):
    x = np.random.default_rng(17).normal(size=(8, 2, 2))
    arrays = [x.copy(), 2*x, x[:4].copy(), 2*x[:4]]
    target, weights, kind = np.zeros((2, 2)), [.5, .5], "particles"
    if fault == "kind": kind = "other"
    if fault == "count": arrays[2:] = [a[:2] for a in arrays[2:]]
    if fault == "shape": arrays[1] = arrays[1][:-1]
    if fault == "same_budget": arrays[2:] = [a.copy() for a in arrays[:2]]
    if fault == "ratio": arrays[:2] = [a[:-1] for a in arrays[:2]]
    if fault == "candidate_prefix": arrays[2][0, 0, 0] += 1
    if fault == "control_prefix": arrays[3][0, 0, 0] += 1
    if fault == "nonfinite": arrays[0][-1, -1, -1] = np.nan
    if fault == "target": target[0, 0] = np.inf
    if fault == "weights": weights = [.5, .6]
    if fault == "dimensions": arrays[2] = arrays[2][:, :, 0]
    if fault == "empty_time": arrays = [a[:, :0] for a in arrays]
    with pytest.raises(ValueError):
        joint.joint_precision(*arrays, target, weights, kind=kind)


def test_integration_cannot_pair_different_particle_budgets():
    x = np.ones((8, 1, 2))
    with pytest.raises(ValueError, match="matching particle budgets"):
        joint.joint_precision(x, x, x[:4], x[:4], np.zeros((1, 2)), [1.], kind="integration")


@pytest.fixture
def diagnostic_inputs(inputs, evidence):
    args, tracker = inputs
    assert engine.run(**args)["status"] == "complete"
    ledger = args["output"]
    digest = joint._hash(ledger)
    # Deliberately tight SOFTWARE-only tolerance verifies failure retention.
    original = joint.audited(read_rows(ledger), ledger.parent, digest, 1e-12, evidence)
    audit = ledger.with_suffix(".audit.json")
    audit.write_text(json.dumps(original), encoding="utf-8")
    return dict(**evidence, ledger=ledger, ledger_sha256=digest, audit=audit,
                audit_sha256=joint._hash(audit), output=ledger.with_name("joint.json")), tracker, original


def test_bound_whole_workload_offline_report_preserves_all_failures_and_missing_family(diagnostic_inputs, monkeypatch):
    args, tracker, original = diagnostic_inputs
    before = args["ledger"].read_bytes()
    def forbidden(*a, **k):
        pytest.fail("offline diagnostic must not load new development data or start forecasts")
    monkeypatch.setattr(engine, "run", forbidden)
    monkeypatch.setattr(engine, "load_development", forbidden)
    result = joint.run(**args)
    assert len(tracker["calls"]) == 32 and args["ledger"].read_bytes() == before
    assert result["status"] == "complete" and not result["certified"] and not result["formal_training_accepted"]
    assert result["final_eval_label_prediction_metric_reads"] == 0
    assert len(result["comparisons"]) == 16 and len(result["aggregate_precision"]) == 4
    assert not result["full_primary_family_available"]
    assert {r["comparison"] for r in result["missing_primary_comparisons"]} == {"road", "river", "worldcover", "surface"}
    assert result["refinement_diagnostics_available"] and not result["unavailable_refinements"]
    assert result["original_numerical_diagnostic"]["out_of_tolerance"] == sum(
        not r["all_scoring_times_within_tolerance"] for r in original["numerical_audit"]["sensitivities"])
    assert result["original_numerical_diagnostic"]["out_of_tolerance"] > 0
    assert result["original_numerical_diagnostic"]["tolerance_m"] == 1e-12
    assert result["source_counts"]["expected_run_count"] == 32
    for aggregate in result["aggregate_precision"]:
        group = [r for r in result["comparisons"] if r["kind"] == aggregate["kind"] and
            r["candidate_workload"]["particles"] == aggregate["refined_particles"] and
            r["candidate_workload"]["max_step_seconds"] == aggregate["refined_step_seconds"]]
        assert len(group) == 4 and aggregate["independent_block_count"] == 2
        np.testing.assert_allclose(aggregate["slot_covariance_m2"], sum(
            np.asarray(r["precision"]["time_covariance_m2"])/16 for r in group), atol=1e-12)
        assert aggregate["time_weighted_difference_m"] == pytest.approx(sum(
            r["precision"]["time_weighted_energy_score_m"]/4 for r in group), abs=1e-12)
        assert aggregate["sign_convention"] == joint.SIGN
    assert json.loads(args["output"].read_text()) == result


def test_complete_primary_family_is_derived_without_hiding_missing_refinements(inputs, evidence):
    args, _ = inputs
    args.update(configurations=sorted({x for pair in joint.PRIMARY_FAMILY.values() for x in pair}),
                seeds=[engine.SEEDS[0]], particles=[4], steps=[5.], limit_origins=1)
    assert engine.run(**args)["status"] == "complete"
    ledger = args["output"]
    audit = ledger.with_suffix(".audit.json")
    audit.write_text(json.dumps(joint.audited(read_rows(ledger), ledger.parent, joint._hash(ledger), 1., evidence)))
    result = joint.run(**evidence, ledger=ledger, ledger_sha256=joint._hash(ledger), audit=audit,
        audit_sha256=joint._hash(audit), output=ledger.with_name("joint.json"))
    assert result["full_primary_family_available"] and not result["missing_primary_comparisons"]
    assert not result["refinement_diagnostics_available"] and len(result["unavailable_refinements"]) == 10
    assert not result["comparisons"] and not result["aggregate_precision"] and not result["certified"]


@pytest.mark.parametrize("field", ["ledger_sha256", "audit_sha256", "fit_sha256", "fit_ledger_sha256", "training_policy_sha256"])
def test_hash_or_policy_mismatch_produces_no_output(diagnostic_inputs, field):
    args, _, _ = diagnostic_inputs
    args[field] = "f"*64
    with pytest.raises(ValueError):
        joint.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("fault", ["incomplete", "duplicate", "brownian", "block", "time", "model", "final_read", "array"])
def test_mutated_or_partial_ledger_cannot_be_salvaged(diagnostic_inputs, fault):
    args, _, original = diagnostic_inputs
    rows = read_rows(args["ledger"])
    if fault == "incomplete": rows.pop(2)
    if fault == "duplicate": rows[3] = deepcopy(rows[2])
    if fault == "brownian": rows[3]["brownian_identity"]["path_sha256"] = "f"*64
    if fault == "block": rows[3]["independent_block_id"] = "different-block"
    if fault == "time": rows[3]["actual_horizons_seconds"][0] += 1
    if fault == "model": rows[3]["model_identity_sha256"] = "f"*64
    if fault == "final_read": rows[0]["final_eval_label_prediction_metric_reads"] = 1
    if fault == "array": rows[3]["particle_artifact"]["sha256"] = "f"*64
    args["ledger"].write_text("".join(json.dumps(row)+"\n" for row in rows))
    args["ledger_sha256"] = joint._hash(args["ledger"])
    original["ledger_sha256"] = args["ledger_sha256"]
    args["audit"].write_text(json.dumps(original))
    args["audit_sha256"] = joint._hash(args["audit"])
    with pytest.raises(ValueError):
        joint.run(**args)
    assert not args["output"].exists()


def test_false_rehashed_audit_is_not_treated_as_evidence(diagnostic_inputs):
    args, _, original = diagnostic_inputs
    original["particle_precision"]["certified"] = True
    args["audit"].write_text(json.dumps(original))
    args["audit_sha256"] = joint._hash(args["audit"])
    with pytest.raises(ValueError, match="exactly reproduce"):
        joint.run(**args)
    assert not args["output"].exists()


def test_output_is_exclusive_and_tolerance_or_workload_override_is_unavailable(diagnostic_inputs):
    args, _, _ = diagnostic_inputs
    with pytest.raises(TypeError):
        joint.run(**args, tolerance_m=100.)
    with pytest.raises(TypeError):
        joint.run(**args, particles=[4])
    args["output"].write_text("owned-marker")
    with pytest.raises(FileExistsError):
        joint.run(**args)
    assert args["output"].read_text() == "owned-marker"


def test_changed_source_during_analysis_produces_no_report(diagnostic_inputs, monkeypatch):
    args, _, _ = diagnostic_inputs
    original, calls = joint.source_hashes(), []
    def changed():
        calls.append(1)
        return original if len(calls) == 1 else {**original, "contrast_refinement.py": "f"*64}
    monkeypatch.setattr(joint, "source_hashes", changed)
    with pytest.raises(ValueError, match="changed during analysis"):
        joint.run(**args)
    assert not args["output"].exists()
