import copy
import json
from pathlib import Path

import pytest

from experiments.capacity_preregistration.count_parameters import (
    FROZEN_MANIFEST_FIELDS,
    FROZEN_SEEDS,
    FROZEN_STATE_LAYOUT,
    FROZEN_STRUCTURE_CLAIM,
    NEURAL_DIFFUSION_CONTRACT,
    count_i1_parameters,
    count_module_trainable_parameters,
    count_neural_parameters,
    validate_matrix,
    validate_result_templates,
)
from models.neural import TimeVaryingNeuralSDE
from models.segment_constant import SegmentConstantSDE


ROOT = Path(__file__).parents[1] / "experiments" / "capacity_preregistration"


def load_matrix():
    return json.loads((ROOT / "experiment_matrix.json").read_text(encoding="utf-8"))


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


def test_v4_matrix_and_both_result_templates_are_executable_contracts():
    document = load_matrix()
    assert validate_matrix(document) == []
    assert validate_result_templates(document, ROOT) == []
    assert document["schema_version"] == "4.0"
    assert document["preregistration_id"] == "NEX-381-v4"
    assert document["approval_state"] == "draft_pending_v4_math_review"
    assert document["supersedes"]["preregistration_id"] == "NEX-381-v3"
    assert document["supersedes"]["git_sha"] == "7765dc002732349018dbf0d494f78046a88ddf30"
    assert TimeVaryingNeuralSDE().dt_scale == 60.0


def test_v4_freezes_manifest_adapter_environment_and_paired_seeds():
    common = load_matrix()["common_protocol"]
    assert common["data_lock"]["required_manifest_fields"] == list(FROZEN_MANIFEST_FIELDS)
    assert common["paired_seeds"] == list(FROZEN_SEEDS)
    assert common["state_layout"] == list(FROZEN_STATE_LAYOUT)
    assert common["state_adapter"]["call"].endswith("coord=0)")
    assert "column y is never read" in common["state_adapter"]["raw_coordinate"]
    assert common["environment_feature"]["dimension"] == 1
    assert "solar_elev only" in common["environment_feature"]["source"]
    assert "day_fraction" in common["environment_feature"]["aggregation"]


def test_v4_freezes_neural_optimizer_initialization_and_rollout():
    common = load_matrix()["common_protocol"]
    architecture = common["neural_architecture"]
    assert architecture["input_order"] == [
        "normalized_position_X", "normalized_velocity_V", "normalized_solar_elev"
    ]
    assert "Tanh" in architecture["layers"]
    assert "xavier_uniform_" in architecture["initialization"]
    assert architecture["dt_scale_seconds"] == 60.0
    adam = common["training"]["neural"]["adam"]
    assert adam == {
        "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 0.0, "amsgrad": False, "maximize": False,
        "foreach": False, "fused": False,
    }
    assert "Euler-Maruyama" in common["rollout"]["neural"]
    assert "no adaptive stepping" in common["rollout"]["neural"]


def test_v4_freezes_metric_event_bridge_and_delta_coordinates():
    evaluation = load_matrix()["common_protocol"]["evaluation"]
    assert evaluation["energy_half"]["state"] == "complete normalized state [X,V] at endpoint"
    assert "U-statistic" in evaluation["energy_half"]["finite_sample_estimator"]
    assert "2D Gaussian KDE" in evaluation["hdr90"]["density"]
    assert "ties inside" in evaluation["hdr90"]["membership"]
    event = evaluation["event"]
    assert event["event_id"] == "hit_X_origin_interval_v1"
    assert event["coordinate"] == "state[0] = normalized_position_X only"
    assert event["domain_D"] == [-8.0, 8.0]
    assert event["region_A"] == [[-0.5, 0.5]]
    assert event["full_support_bin_edges"][0] == "-inf"
    assert event["full_support_bin_edges"][-1] == "inf"
    assert "state[0]=X then state[1]=V" in evaluation["bridge_dual"]["bridge_correction_coordinates"]
    assert evaluation["delta_probe"]["coordinate_index"] == 0
    assert evaluation["delta_probe"]["directions"] == ["base", "dilation", "erosion"]


def test_v4_all_arms_resolve_frozen_protocol_and_contrasts_are_complete():
    document = load_matrix()
    assert len(document["arms"]) == 8
    for arm in document["arms"]:
        assert arm["data_lock"] == "common_protocol.data_lock"
        assert arm["state_adapter"] == "common_protocol.state_adapter"
        assert arm["seeds"] == "common_protocol.paired_seeds"
        assert arm["evaluation"] == "common_protocol.evaluation"
        if arm["model"]["family"] == "neural":
            assert arm["environment_feature"] == "common_protocol.environment_feature"
            assert arm["model"]["architecture"] == "common_protocol.neural_architecture"
    assert len(document["predeclared_contrasts"]) == 7
    structure = document["predeclared_contrasts"][-1]
    assert structure["claim"] == FROZEN_STRUCTURE_CLAIM
    assert structure["parameter_ratio_right_over_left"] == 2.48


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        (lambda d: d["common_protocol"]["data_lock"]["required_manifest_fields"].pop(), "data manifest fields"),
        (lambda d: d["common_protocol"].update(paired_seeds=[20260814]), "paired seeds"),
        (lambda d: d["common_protocol"]["state_adapter"].update(call="data.loader.transitions()"), "state adapter call"),
        (lambda d: d["common_protocol"]["environment_feature"].update(dimension=2), "environment dimension"),
        (lambda d: d["common_protocol"]["neural_architecture"].update(dt_scale_seconds=1.0), "neural dt_scale"),
        (lambda d: d["common_protocol"]["training"]["neural"]["adam"].update(eps=1e-7), "Adam parameters"),
        (lambda d: d["common_protocol"]["rollout"].update(neural="RK45"), "neural rollout"),
        (lambda d: d["common_protocol"]["evaluation"]["energy_half"].update(state="X only"), "Energy state"),
        (lambda d: d["common_protocol"]["evaluation"]["hdr90"].update(threshold="unspecified"), "HDR90 threshold"),
        (lambda d: d["common_protocol"]["evaluation"]["event"].update(region_A=None), "event region A"),
        (lambda d: d["common_protocol"]["evaluation"]["event"].update(full_support_bin_edges=[0, 1]), "event full-support bins"),
        (lambda d: d["common_protocol"]["evaluation"]["delta_probe"].update(coordinate_index=1), "delta coordinate index"),
        (lambda d: d["common_protocol"]["evaluation"]["inference"].update(bootstrap="segment only"), "paired hierarchical bootstrap"),
    ],
)
def test_v4_validator_negative_contract_mutations(mutation, expected_error):
    document = load_matrix()
    mutation(document)
    assert any(expected_error in error for error in validate_matrix(document))


def test_v4_validator_rejects_missing_arm_reference_or_contrast_identity():
    missing_ref = load_matrix()
    del missing_ref["arms"][4]["environment_feature"]
    assert any("missing executable keys" in error for error in validate_matrix(missing_ref))

    wrong_contrast = load_matrix()
    wrong_contrast["predeclared_contrasts"][0]["left"] = "CAP-I1-K30"
    assert any("predeclared contrast identities" in error for error in validate_matrix(wrong_contrast))


def test_v4_result_template_negative_header_and_direction(tmp_path):
    document = load_matrix()
    arm_source = (ROOT / "result_template.csv").read_text(encoding="utf-8")
    contrast_source = (ROOT / "contrast_result_template.csv").read_text(encoding="utf-8")
    (tmp_path / "result_template.csv").write_text(
        arm_source.replace("dilation", "expand"), encoding="utf-8"
    )
    (tmp_path / "contrast_result_template.csv").write_text(
        contrast_source.replace("raw_p_value", "p"), encoding="utf-8"
    )
    errors = validate_result_templates(document, tmp_path)
    assert any("base/dilation/erosion" in error for error in errors)
    assert any("CSV header" in error for error in errors)
