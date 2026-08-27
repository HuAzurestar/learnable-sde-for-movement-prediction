import csv
import json
from pathlib import Path

import pytest

from experiments.capacity_preregistration.count_parameters import (
    FROZEN_STRUCTURE_CLAIM,
    NEURAL_DIFFUSION_CONTRACT,
    count_i1_parameters,
    count_module_trainable_parameters,
    count_neural_parameters,
    validate_matrix,
)
from models.segment_constant import SegmentConstantSDE


@pytest.mark.parametrize("n_modes, expected", [(1, 5), (3, 15), (10, 50), (30, 150)])
def test_i1_declared_and_runtime_counts_match(n_modes, expected):
    declared = count_i1_parameters(n_modes)
    runtime = count_module_trainable_parameters(SegmentConstantSDE(n_modes=n_modes))
    assert declared["stored_trainable_parameters"] == expected
    assert runtime == expected


@pytest.mark.parametrize(
    "hidden, expected",
    [((8, 8), 124), ((32, 32), 1252), ((64, 64), 4548), ((128, 128), 17284)],
)
def test_neural_contract_counts(hidden, expected):
    count = count_neural_parameters(hidden)
    assert count["stored_trainable_parameters"] == expected
    assert count["contract"]["diffusion"] == NEURAL_DIFFUSION_CONTRACT


def test_neural_contract_rejects_unregistered_depth():
    with pytest.raises(ValueError, match="exactly two"):
        count_neural_parameters((8,))


def test_registered_matrix_is_internally_consistent():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "capacity_preregistration"
        / "experiment_matrix.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert validate_matrix(document) == []


def test_v2_freezes_math_review_contracts():
    root = Path(__file__).parents[1] / "experiments" / "capacity_preregistration"
    document = json.loads((root / "experiment_matrix.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == "2.0"
    assert document["preregistration_id"] == "NEX-381-v2"
    assert len(document["arms"]) == 8

    i1_arms = [arm for arm in document["arms"] if arm["model"]["family"] == "I1"]
    assert [arm["registered_parameter_count"] for arm in i1_arms] == [5, 15, 50, 150]
    assert [arm["registered_effective_degrees_of_freedom"] for arm in i1_arms] == [4, 14, 49, 149]
    assert document["common_protocol"]["neural_diffusion_contract"] == NEURAL_DIFFUSION_CONTRACT

    contrast = document["predeclared_contrasts"][0]
    assert contrast["parameter_ratio_right_over_left"] == 2.48
    assert contrast["secondary_ratio_right_over_i1_effective_dof"] == pytest.approx(124 / 49)
    assert contrast["claim_boundary"] == FROZEN_STRUCTURE_CLAIM

    dual = document["common_protocol"]["evaluation"]["bridge_dual"]
    assert "same sampled paths" in dual["paths"]
    assert "1e-12" in dual["scalar_identity"]
    assert "underflow and overflow" in dual["reconstruction"]
    assert "undefined_stratum" in dual["strata"]
    assert "separately" in dual["fp_mc_l1"]

    delta = document["common_protocol"]["evaluation"]["delta_probe"]
    assert "DeltaX_i = X_T_i - X_0_i" in delta["displacement"]
    assert "compute exactly once" in delta["sigma_hist"]
    assert delta["delta"] == "0.3 * sigma_hist"
    assert "closed intervals" in delta["base_region"]
    assert "delete empty components" in delta["erosion"]
    assert "failure/NA" in delta["failure"]


def test_v2_result_template_has_frozen_audit_columns():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "capacity_preregistration"
        / "result_template.csv"
    )
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert len(rows) == 2
    assert len(rows[0]) == len(rows[1])
    assert rows[1][0] == "NEX-381-v2"
    for column in (
        "registered_effective_degrees_of_freedom",
        "dual_full_support_bins",
        "dual_undefined_stratum",
        "dual_algebra_pass",
        "fp_mc_l1",
        "sigma_hist",
        "delta_probe_status",
    ):
        assert column in rows[0]
