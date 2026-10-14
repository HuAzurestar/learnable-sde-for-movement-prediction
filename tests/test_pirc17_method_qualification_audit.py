"""Independent arithmetic oracles using synthetic arrays, never research data."""
import ast
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.pirc17 import method_qualification_audit as audit
from experiments.pirc17.metrics import EntropyGrid, score_path
from experiments.pirc17.method_mechanisms import score_diagnostic


def receipt(slot, value, evidence, *, statistic="split_exact_error", threshold=2.):
    result = {"slot_id": slot, "statistic": statistic, "operator": "le", "threshold": threshold,
        "value": value, "passed": value <= threshold, "scientific_claim_authorized": False, "evidence": evidence}
    result["sha256"] = audit.digest(result)
    definitions = {slot: {"statistic": statistic, "operator": "le", "threshold": threshold}}
    return result, definitions


def test_energy_unbiased_two_point_oracle():
    assert audit.energy(np.array([[0., 0.], [2., 0.]]), np.array([1., 0.])) == 0.
    assert audit.energy(np.array([[2., 3.], [2., 3.]]), np.zeros(2)) == math.sqrt(13)


def samples():
    x = np.random.default_rng(21).normal(size=(512, 4, 2))*100
    y = np.array([[3, 4], [13, 14], [23, 24], [33, 34]], dtype=float)
    times = np.array([55., 299., 906., 1796.])
    grid = EntropyGrid(tuple(np.arange(-10000, 10001, 250)), tuple(np.arange(-10000, 10001, 250)))
    saved = score_path(x, y, times, time_weights=[.25]*4, entropy_grid=grid)
    return x, y, times, saved


def test_all_common_scores_independently_match_on_software_arrays():
    x, y, times, saved = samples()
    assert audit.common_scores(x, y, times, saved) == pytest.approx(saved["time_weighted_energy_score_m"])


@pytest.mark.parametrize("kind", ["missing_time", "es", "crps", "coverage", "area", "entropy", "path", "nll"])
def test_saved_metric_tampering_is_not_trusted(kind):
    x, y, times, saved = samples()
    if kind == "missing_time": saved["by_time"].pop()
    elif kind == "es": saved["time_weighted_energy_score_m"] += 1
    elif kind == "crps": saved["by_time"][0]["marginal_crps_m"][0] += 1
    elif kind == "coverage": saved["by_time"][0]["region"]["levels"][0]["covered"] ^= True
    elif kind == "area": saved["by_time"][0]["region"]["levels"][0]["area_m2"] += 1
    elif kind == "entropy": saved["by_time"][0]["position_entropy"]["entropy_nats"] += 1
    elif kind == "path": saved["path_energy_score_m"] += 1
    else: saved["nll"]["status"] = "qualified"
    with pytest.raises(AssertionError): audit.common_scores(x, y, times, saved)


def test_rehashed_false_pass_is_rejected():
    gate, definitions = receipt("arm-18/full", 3., {})
    gate["passed"] = True
    gate["sha256"] = audit.digest({k: v for k, v in gate.items() if k != "sha256"})
    with pytest.raises(AssertionError): audit.check_gate(gate, 3., definitions)


def test_distributional_integration_oracle_includes_covariance():
    am = np.zeros((8, 4, 2))
    rm = np.broadcast_to([3., 4.], am.shape).copy()
    ac = np.broadcast_to(np.eye(2), (8, 4, 2, 2)).copy()
    rc = 4*ac
    evidence = {"per_time_gaussian_coupling_rms_m": [math.sqrt(27)]*4,
        "per_time_rms_conditional_mean_error_m": [5.]*4,
        "per_time_rms_covariance_difference_m2": [math.sqrt(18)]*4,
        "conditional_moment_sha256": [audit.array_digest(v) for v in (am, rm, ac, rc)]}
    gate, definitions = receipt("arm-18/full", math.sqrt(27), evidence)
    a = {"conditional_means_method_frame_m": am, "conditional_covariances_method_frame_m2": ac}
    r = {"conditional_means_method_frame_m": rm, "conditional_covariances_method_frame_m2": rc}
    audit.integration(a, r, gate, definitions)
    broken = copy.deepcopy(gate)
    broken["value"] = 5.
    with pytest.raises(AssertionError): audit.integration(a, r, broken, definitions)


def test_gaussian_diagnostic_recomputes_saved_draws_without_new_forecasts():
    x, y, _, _ = samples()
    g = score_diagnostic("arm-10/d2_mc", x, y, seed=20260814, origin_id="software-only", input_identity_sha256="a"*64)
    definitions = {g["slot_id"]: {"statistic": g["statistic"], "operator": g["operator"], "threshold": g["threshold"]}}
    audit.scoring_diagnostic(x, y, g, definitions)
    g["evidence"]["per_time"][0]["gaussian_mc_es_m"] += 1
    with pytest.raises(AssertionError): audit.scoring_diagnostic(x, y, g, definitions)


def test_auditor_has_no_scientific_producer_or_data_loader_imports():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported <= {"__future__", "pathlib", "statistics", "seed_resume_session"}
    assert {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} <= {
        "argparse", "ast", "hashlib", "itertools", "json", "math", "os", "sys", "time", "numpy"}
