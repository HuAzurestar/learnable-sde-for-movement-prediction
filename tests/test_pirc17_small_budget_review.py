"""Offline review oracles; all generated evidence here is software-only."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pirc17 import small_budget_review as review
from experiments.pirc17.contrast_refinement import joint_precision
from experiments.pirc17.precision import paired_energy_precision
from tests.test_pirc17_candidate_power import build
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows


@pytest.mark.parametrize("bad", [[], [2, 4], [4], [8, 4], [4, 4], [4, 10], [True, 8], [4, 8, 16, 32]])
def test_invalid_prefix_budgets(bad):
    with pytest.raises(ValueError):
        review.check_budgets(bad)


def test_cached_contrasts_match_joint_four_ensemble_oracle():
    rng = np.random.default_rng(27)
    a = rng.normal(size=(16, 4, 2))
    b = a*.6+rng.normal(size=a.shape)*.2
    y = rng.normal(size=(4, 2))
    weights, budgets = [.1, .2, .3, .4], [4, 8, 16]
    actual = review.contrast_statistics(review.cached_statistics(a, y, budgets),
        review.cached_statistics(b, y, budgets), budgets, weights)
    for kind, large, small, row in actual:
        if kind == "small_budget_contrast":
            expected = paired_energy_precision(a[:small], b[:small], y, weights)
        else:
            expected = joint_precision(a, b, a[:small], b[:small], y, weights, kind="particles")
            assert "particle_count" not in row and row["jackknife_group_count"] == small
        for field in ("by_time_energy_score_m", "time_covariance_m2", "time_weighted_standard_error_m"):
            assert np.allclose(row[field], expected[field], rtol=1e-12, atol=1e-14)


def test_brute_force_prefix_refinement_deletions_and_cancellation():
    rng = np.random.default_rng(28)
    a, b, y = rng.normal(size=(8, 4, 2)), rng.normal(size=(8, 4, 2)), np.zeros((4, 2))
    def es(x):
        return np.array([sum(np.linalg.norm(v) for v in x[:, t])/len(x)
            -sum(np.linalg.norm(u-v) for u in x[:, t] for v in x[:, t])/(2*len(x)*(len(x)-1))
            for t in range(4)])
    ca, cb = review.cached_statistics(a, y, [4, 8]), review.cached_statistics(b, y, [4, 8])
    row = review.contrast_statistics(ca, cb, [4, 8], [.25]*4)[-1][-1]
    deleted = []
    for i in range(4):
        keep = [j for j in range(8) if j not in (i, i+4)]
        short = [j for j in range(4) if j != i]
        deleted.append((es(a[keep])-es(b[keep]))-(es(a[short])-es(b[short])))
    z = np.asarray(deleted)
    covariance = .75*(z-z.mean(axis=0)).T@(z-z.mean(axis=0))
    assert np.allclose(row["by_time_energy_score_m"], (es(a)-es(b))-(es(a[:4])-es(b[:4])))
    assert np.allclose(row["time_covariance_m2"], covariance)
    for _, _, _, zero in review.contrast_statistics(ca, ca, [4, 8], [.25]*4):
        assert zero["time_weighted_energy_score_m"] == zero["time_weighted_standard_error_m"] == 0


@pytest.fixture
def prepared(inputs, evidence):
    args, tracker = inputs
    power_args = build(args, evidence, overrides={0: {"particles": [8]}, 1: {"particles": [8]}}, stem="small")
    power_args["particles"] = 8
    review.parent.run(**power_args)
    plan = dict(schema_version=review.VERSION, new_rollouts=0, final_eval_authorized=False,
        particle_budgets=[4, 8], step_seconds=5., epsilon_m=1., alpha=.05,
        diagnostic_family_size=55, wall_seconds=600,
        reference_power=str(power_args["output"]), reference_power_sha256=review._hash(power_args["output"]),
        candidate_evidence={k: str(v) if isinstance(v, Path) else v for k, v in evidence.items()},
        inputs=[dict(ledger=str(p), ledger_sha256=s, audit=str(a), audit_sha256=h) for p, s, a, h in zip(
            power_args["ledgers"], power_args["ledger_sha256"], power_args["audits"], power_args["audit_sha256"])])
    path = args["output"].with_name("small-plan.json")
    path.write_text(json.dumps(plan), encoding="utf-8")
    return dict(plan_path=path, plan_sha256=review._hash(path), output=path.with_name("prefix-review-result.json")), plan, tracker


def test_complete_offline_run_preserves_sources_without_calling_forecast(prepared):
    args, plan, tracker = prepared
    calls, queries = len(tracker["calls"]), len(tracker["queries"])
    result = review.run(**args)
    assert len(tracker["calls"]) == calls and len(tracker["queries"]) == queries
    assert result["new_rollouts"] == 0 and result["selected_runs"] == 60
    assert not result["certified"] and not result["numerically_qualified"]
    assert len(result["paired_diagnostics"]) == 2*5*5*3
    assert len(result["aggregate_precision"]) == 15
    assert result["original_numerical_failures"] == review.read_bound(
        plan["reference_power"], plan["reference_power_sha256"])["source_numerical"]
    # Independent aggregation uses saved per-origin/seed covariance, not between-block SD.
    for aggregate in result["aggregate_precision"]:
        rows = [r for r in result["paired_diagnostics"] if r["kind"] == aggregate["kind"]
            and r["comparison"] == aggregate["comparison"]
            and r["candidate_workload"]["particles"] == aggregate["candidate_particles"]
            and r["control_workload"]["particles"] == aggregate["control_particles"]]
        assert len(rows) == 10
        mean = sum(np.array(r["precision"]["by_time_energy_score_m"]) for r in rows)/10
        cov = sum(np.array(r["precision"]["time_covariance_m2"]) for r in rows)/100
        assert np.allclose(aggregate["by_scoring_slot_difference_m"], mean)
        assert np.allclose(aggregate["slot_covariance_m2"], cov)
        assert aggregate["normal_quantile"] == review.Q
    with pytest.raises(FileExistsError):
        review.run(**args)


def test_plan_cannot_silently_expand_or_relax(prepared):
    args, original, _ = prepared
    for field, value in [("new_rollouts", 1), ("final_eval_authorized", True),
            ("diagnostic_family_size", 5), ("wall_seconds", 601), ("alpha", .5), ("particle_budgets", [3, 8])]:
        plan = deepcopy(original)
        plan[field] = value
        args["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
        args["plan_sha256"] = review._hash(args["plan_path"])
        with pytest.raises(ValueError): review.run(**args)
        assert not args["output"].exists()


def test_missing_or_mixed_parent_evidence_rejected(prepared):
    args, original, _ = prepared
    for fault in ["missing_source", "ledger_hash", "reference_hash", "epsilon", "step"]:
        plan = deepcopy(original)
        if fault == "missing_source": plan["inputs"].pop()
        elif fault == "ledger_hash": plan["inputs"][0]["ledger_sha256"] = "f"*64
        elif fault == "reference_hash": plan["reference_power_sha256"] = "f"*64
        elif fault == "epsilon": plan["epsilon_m"] = 2.
        else: plan["step_seconds"] = 2.5
        args["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
        args["plan_sha256"] = review._hash(args["plan_path"])
        with pytest.raises(ValueError): review.run(**args)
        assert not args["output"].exists()


def test_pairing_completeness_and_post_analysis_mutation_rejected(prepared, monkeypatch):
    args, plan, _ = prepared
    original = review.analyze
    def changed(entries, *pos, **kw):
        if fault == "changed_array_after_analysis":
            result = original(entries, *pos, **kw)
            path = Path(plan["inputs"][0]["ledger"])
            row = read_rows(path)[2]
            sidecar = path.parent/row["particle_artifact"]["path"]
            sidecar.write_bytes(sidecar.read_bytes()+b"changed")
            return result
        entries = deepcopy(entries)
        key = next(iter(entries))
        if fault == "changed_stream": entries[key][0]["brownian_identity"]["path_sha256"] = "f"*64
        elif fault == "changed_target": entries[key][1][1][0, 0] += 1
        else: entries.pop(key)
        return original(entries, *pos, **kw)
    for fault in ["changed_stream", "changed_target", "missing_grid", "changed_array_after_analysis"]:
        with monkeypatch.context() as patch:
            patch.setattr(review, "analyze", changed)
            with pytest.raises(ValueError): review.run(**args)
            assert not args["output"].exists()
