import copy
import csv
import json
from pathlib import Path

import pytest

from experiments.capacity_preregistration.count_parameters import (
    FROZEN_STRUCTURE_CLAIM,
    FROZEN_STATE_LAYOUT,
    FROZEN_STATE_LAYOUT_REF,
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


def test_v3_freezes_math_review_contracts_and_state_semantics():
    root = Path(__file__).parents[1] / "experiments" / "capacity_preregistration"
    document = json.loads((root / "experiment_matrix.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == "3.0"
    assert document["preregistration_id"] == "NEX-381-v3"
    assert document["approval_state"] == "draft_pending_v3_math_review"
    assert document["supersedes"]["preregistration_id"] == "NEX-381-v2"
    assert document["supersedes"]["git_sha"] == "5346377f4451c0c0e3774ba71a2b541a1c555f54"
    assert len(document["arms"]) == 8

    i1_arms = [arm for arm in document["arms"] if arm["model"]["family"] == "I1"]
    assert [arm["registered_parameter_count"] for arm in i1_arms] == [5, 15, 50, 150]
    assert [arm["registered_effective_degrees_of_freedom"] for arm in i1_arms] == [4, 14, 49, 149]
    common_protocol = document["common_protocol"]
    assert common_protocol["state_layout"] == list(FROZEN_STATE_LAYOUT)
    assert common_protocol["neural_diffusion_contract"] == NEURAL_DIFFUSION_CONTRACT
    neural_arms = [arm for arm in document["arms"] if arm["model"]["family"] == "neural"]
    assert len(neural_arms) == 4
    assert all(arm["model"]["state_layout"] == FROZEN_STATE_LAYOUT_REF for arm in neural_arms)

    contrast = document["predeclared_contrasts"][0]
    assert contrast["parameter_ratio_right_over_left"] == 2.48
    assert contrast["secondary_ratio_right_over_i1_effective_dof"] == pytest.approx(124 / 49)
    assert contrast["claim_boundary"] == FROZEN_STRUCTURE_CLAIM

    dual = common_protocol["evaluation"]["bridge_dual"]
    assert dual["state_layout"] == FROZEN_STATE_LAYOUT_REF
    assert dual["hit_event_coordinate"] == "state[0] = normalized_position_X only"
    assert "state[0]=X then state[1]=V" in dual["bridge_correction_coordinates"]
    assert "same sampled paths" in dual["paths"]
    assert "1e-12" in dual["scalar_identity"]
    assert "underflow and overflow" in dual["reconstruction"]
    assert "undefined_stratum" in dual["strata"]
    assert "separately" in dual["fp_mc_l1"]

    delta = common_protocol["evaluation"]["delta_probe"]
    assert delta["coordinate_index"] == 0
    assert delta["coordinate"] == "state[0] = normalized_position_X"
    assert "state_i[T,0] - state_i[0,0]" in delta["displacement"]
    assert "compute exactly once" in delta["sigma_hist"]
    assert delta["delta"] == "0.3 * sigma_hist"
    assert "closed intervals" in delta["base_region"]
    assert "delete empty components" in delta["erosion"]
    assert "failure/NA" in delta["failure"]


def test_v3_validator_rejects_state_layout_drift_and_missing_arm_reference():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "capacity_preregistration"
        / "experiment_matrix.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))

    wrong_layout = copy.deepcopy(document)
    wrong_layout["common_protocol"]["state_layout"] = [
        "normalized_position_X",
        "normalized_position_Y",
    ]
    assert any("common state layout" in error for error in validate_matrix(wrong_layout))

    missing_reference = copy.deepcopy(document)
    neural_arm = next(
        arm for arm in missing_reference["arms"] if arm["model"]["family"] == "neural"
    )
    del neural_arm["model"]["state_layout"]
    assert any(
        f"{neural_arm['arm_id']}: neural state layout" in error
        for error in validate_matrix(missing_reference)
    )


def test_v3_validator_rejects_bridge_or_delta_coordinate_drift():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "capacity_preregistration"
        / "experiment_matrix.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))

    wrong_bridge = copy.deepcopy(document)
    wrong_bridge["common_protocol"]["evaluation"]["bridge_dual"][
        "hit_event_coordinate"
    ] = "state[1]"
    assert any("bridge hit event" in error for error in validate_matrix(wrong_bridge))

    wrong_delta = copy.deepcopy(document)
    wrong_delta["common_protocol"]["evaluation"]["delta_probe"]["coordinate_index"] = 1
    assert any("coordinate_index must be 0" in error for error in validate_matrix(wrong_delta))


def test_v3_result_template_has_frozen_audit_columns():
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
    assert rows[1][0] == "NEX-381-v3"
    for column in (
        "state_layout",
        "registered_effective_degrees_of_freedom",
        "dual_full_support_bins",
        "dual_undefined_stratum",
        "dual_algebra_pass",
        "fp_mc_l1",
        "sigma_hist",
        "delta_probe_status",
    ):
        assert column in rows[0]
