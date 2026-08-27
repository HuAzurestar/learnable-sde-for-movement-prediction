import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.capacity_preregistration.count_parameters import (
    EXPECTED_CONTRASTS,
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
VALIDATOR = ROOT / "count_parameters.py"


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


def test_v7_matrix_and_templates_are_executable_contracts():
    document = load_matrix()
    assert validate_matrix(document) == []
    assert validate_result_templates(document, ROOT) == []
    assert document["schema_version"] == "7.0"
    assert document["preregistration_id"] == "NEX-381-v7"
    assert document["approval_state"] == "draft_pending_v7_contrast_template_diff_review"
    assert document["supersedes"]["preregistration_id"] == "NEX-381-v6"
    assert document["supersedes"]["git_sha"] == "646e62ec27d586a5793a0a4759b595d4ce758d3b"
    assert TimeVaryingNeuralSDE().dt_scale == 60.0


def test_v7_inherits_i1_raw_kernel_and_affine_evaluation_contract():
    common = load_matrix()["common_protocol"]
    policy = common["state_adapter"]["model_coordinate_policy"]
    assert "only in raw" in policy["I1"]
    assert "never feed z" in policy["I1"]
    assert "at every registered grid point" in policy["evaluation"]
    assert "raw r=[X_raw,V_raw]" in common["rollout"]["I1"]
    dual = common["evaluation"]["bridge_dual"]
    assert "grad_r(log_h)=S^-T grad_z(log_h)" in dual["I1_raw_correction"]
    assert "S^-1 a_r S^-T" in dual["I1_raw_correction"]
    assert "A_raw=m_X+s_X*A" in dual["I1_event_transform"]


def test_v7_inherits_hashed_solar_time_and_file_maps():
    common = load_matrix()["common_protocol"]
    assert common["data_lock"]["required_manifest_fields"] == list(FROZEN_MANIFEST_FIELDS)
    environment = common["environment_feature"]
    assert "segment_id,file_id,absolute_start_epoch" in environment["start_map"]
    assert "hash must equal segment_start_map_sha256" in environment["start_map"]
    assert "file_id,relative_path,sha256" in environment["condition_file_map"]
    assert "runtime glob selection is forbidden" in environment["condition_file_map"]
    assert "every aligned solar_elev value finite" in environment["aggregation"]
    assert "filtering nonfinite rows is forbidden" in environment["aggregation"]


def test_v7_inherits_neural_contract_and_accepted_metrics():
    common = load_matrix()["common_protocol"]
    assert common["paired_seeds"] == list(FROZEN_SEEDS)
    assert common["state_layout"] == list(FROZEN_STATE_LAYOUT)
    architecture = common["neural_architecture"]
    assert architecture["input_order"] == [
        "normalized_position_X", "normalized_velocity_V", "normalized_solar_elev"
    ]
    assert "Tanh" in architecture["layers"]
    assert "xavier_uniform_" in architecture["initialization"]
    assert architecture["dt_scale_seconds"] == 60.0
    assert common["training"]["neural"]["optimizer"]["parameters"]["eps"] == 1e-8
    evaluation = common["evaluation"]
    assert "U-statistic" in evaluation["energy_half"]["finite_sample_estimator"]
    assert "2D Gaussian KDE" in evaluation["hdr90"]["density"]
    assert "901/1000" in evaluation["hdr90"]["membership"]
    assert "at least 90.1% forecast-sample mass" in evaluation["hdr90"]["report"]


def test_v7_inherits_crossed_bootstrap_and_failure_propagation():
    inference = load_matrix()["common_protocol"]["evaluation"]["inference"]
    assert inference["contrast_metric_ids"] == [
        "energy_half", "hdr90_abs_calibration_error"
    ]
    for token in ("Cartesian product", "all 8 arms", "all 7 contrasts", "never nested"):
        assert token in inference["bootstrap"]
    assert "same matrix-wide index vectors" in inference["family_shared_draws"]
    assert "Hyndman-Fan type 7" in inference["pointwise_ci"]
    assert "not an exact" in inference["raw_p"]
    assert "I1-capacity=3" in inference["holm"]
    assert "family sizes 3,3,1" in inference["simultaneous_ci"]
    assert "all 7 contrast rows" in inference["complete_cube_policy"]
    assert "no complete-case deletion" in inference["complete_cube_policy"]


def test_v7_result_contract_has_exact_14_rows_and_identity_keys():
    document = load_matrix()
    schemas = document["common_protocol"]["result_schemas"]
    assert schemas["arm_results"]["primary_key"] == [
        "preregistration_id", "execution_git_sha", "dataset_id",
        "global_splits_sha256", "arm_id", "seed", "delta_probe_direction",
    ]
    assert schemas["contrast_results"]["required_record_count"] == 14
    assert schemas["contrast_results"]["metric_enum"] == [
        "energy_half", "hdr90_abs_calibration_error"
    ]
    with (ROOT / "result_template.csv").open(newline="", encoding="utf-8") as source:
        arm_rows = list(csv.reader(source))
    assert "delta_probe_value" not in arm_rows[0]
    with (ROOT / "contrast_result_template.csv").open(newline="", encoding="utf-8") as source:
        contrast_rows = list(csv.DictReader(source))
    assert len(contrast_rows) == 14
    assert {
        (row["contrast_id"], row["metric_id"]) for row in contrast_rows
    } == {
        (contrast[0], metric)
        for contrast in EXPECTED_CONTRASTS
        for metric in ("energy_half", "hdr90_abs_calibration_error")
    }


def test_v7_all_arms_and_contrasts_remain_frozen():
    document = load_matrix()
    assert len(document["arms"]) == 8
    assert len(document["predeclared_contrasts"]) == 7
    for arm in document["arms"]:
        assert arm["state_adapter"] == "common_protocol.state_adapter"
        assert arm["seeds"] == "common_protocol.paired_seeds"
        assert arm["evaluation"] == "common_protocol.evaluation"
    structure = document["predeclared_contrasts"][-1]
    assert structure["claim"] == FROZEN_STRUCTURE_CLAIM
    assert structure["parameter_ratio_right_over_left"] == 2.48


@pytest.mark.parametrize(
    "case_id, mutation, expected_error",
    [
        ("leakage_unit", lambda d: d["common_protocol"]["splits"].update(leakage_unit="segment_id"), "split and leakage contract"),
        ("overlap_check", lambda d: d["common_protocol"]["splits"].update(segment_overlap_check=False), "split and leakage contract"),
        ("i1_max_iter", lambda d: d["common_protocol"]["training"]["i1"].update(maximum_iterations=999), "I1 training, stopping and forecast budget"),
        ("i1_stopping", lambda d: d["common_protocol"]["training"]["i1"]["stopping"].update(tolerance=0.5), "I1 training, stopping and forecast budget"),
        ("i1_forecast_samples", lambda d: d["common_protocol"]["training"]["i1"].update(forecast_samples_per_segment=17), "I1 training, stopping and forecast budget"),
        ("neural_objective", lambda d: d["common_protocol"]["training"]["neural"]["objective"].update(name="arbitrary_loss"), "neural objective, optimizer, stopping, checkpoint and forecast budget"),
        ("neural_stopping", lambda d: d["common_protocol"]["training"]["neural"]["stopping"].update(patience_completed_epochs=999), "neural objective, optimizer, stopping, checkpoint and forecast budget"),
        ("neural_checkpoint", lambda d: d["common_protocol"]["training"]["neural"]["checkpoint"].update(selection="last_epoch"), "neural objective, optimizer, stopping, checkpoint and forecast budget"),
        ("neural_forecast_samples", lambda d: d["common_protocol"]["training"]["neural"].update(forecast_samples_per_segment=17), "neural objective, optimizer, stopping, checkpoint and forecast budget"),
        ("solar_row_selection", lambda d: d["common_protocol"]["environment_feature"]["alignment"].update(row_selection="arbitrary first row"), "solar alignment"),
        ("rng_stream_reuse", lambda d: d["common_protocol"]["rng_contract"]["stream_derivation"].update(stream_order=["shared", "shared", "shared"]), "RNG contract"),
        ("rng_diagnostic_replacement", lambda d: d["common_protocol"]["rng_contract"]["determinism"].update(diagnostic_rerun_replaces_registered_run=True), "RNG contract"),
        ("event_time_grid", lambda d: d["common_protocol"]["evaluation"]["event"].update(time_grid_ref="arbitrary_grid"), "event time grid reference"),
        ("delta_reselection", lambda d: d["common_protocol"]["evaluation"]["delta_probe"]["failure"].update(post_failure_reselection_allowed=True), "delta failure policy"),
    ],
)
def test_v7_rejects_each_reviewer_counterexample(case_id, mutation, expected_error):
    document = load_matrix()
    mutation(document)
    errors = validate_matrix(document)
    assert errors, case_id
    assert any(expected_error in error for error in errors), (case_id, errors)
    assert any("canonical full-matrix sha256" in error for error in errors), case_id


def test_v7_canonical_digest_rejects_an_unlisted_field_mutation():
    document = load_matrix()
    document["code_baseline"]["execution_git_sha"] = "silently_drifted"
    errors = validate_matrix(document)
    assert any("canonical full-matrix sha256" in error for error in errors)


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        (lambda d: d["common_protocol"]["data_lock"]["required_manifest_fields"].pop(), "data manifest fields"),
        (lambda d: d["common_protocol"]["state_adapter"]["model_coordinate_policy"].update(I1="normalized"), "I1 must fit"),
        (lambda d: d["common_protocol"]["environment_feature"].update(start_map="unhashed"), "solar start map"),
        (lambda d: d["common_protocol"]["environment_feature"].update(condition_file_map="glob first"), "condition file manifest"),
        (lambda d: d["common_protocol"]["environment_feature"].update(aggregation="drop NaN then mean"), "nonfinite aligned solar"),
        (lambda d: d["common_protocol"]["rollout"].update(I1="exact normalized kernel"), "I1 rollout"),
        (lambda d: d["common_protocol"]["evaluation"]["bridge_dual"].update(I1_raw_correction="a grad h"), "I1 bridge affine"),
        (lambda d: d["common_protocol"]["evaluation"]["hdr90"].update(report="exactly 90%"), "finite-sample mass"),
        (lambda d: d["common_protocol"]["evaluation"]["inference"].update(bootstrap="nested"), "crossed bootstrap"),
        (lambda d: d["common_protocol"]["evaluation"]["inference"].update(pointwise_ci="percentile"), "pointwise CI"),
        (lambda d: d["common_protocol"]["evaluation"]["inference"].update(raw_p="exact"), "p-value approximation"),
        (lambda d: d["common_protocol"]["evaluation"]["inference"].update(complete_cube_policy="complete cases"), "failure propagation"),
        (lambda d: d["common_protocol"]["result_schemas"]["contrast_results"].update(required_record_count=7), "required record count"),
    ],
)
def test_v7_validator_rejects_inherited_mathematical_contract_drift(mutation, expected_error):
    document = load_matrix()
    mutation(document)
    assert any(expected_error in error for error in validate_matrix(document))


def test_v7_validator_rejects_missing_arm_reference_or_contrast_identity():
    missing_ref = load_matrix()
    del missing_ref["arms"][4]["environment_feature"]
    assert any("missing executable keys" in error for error in validate_matrix(missing_ref))
    wrong_contrast = load_matrix()
    wrong_contrast["predeclared_contrasts"][0]["left"] = "CAP-I1-K30"
    assert any("predeclared contrast identities" in error for error in validate_matrix(wrong_contrast))


def test_v7_result_template_rejects_direction_and_record_set_drift(tmp_path):
    document = load_matrix()
    arm_source = (ROOT / "result_template.csv").read_text(encoding="utf-8")
    contrast_source = (ROOT / "contrast_result_template.csv").read_text(encoding="utf-8")
    (tmp_path / "result_template.csv").write_text(
        arm_source.replace("dilation", "expand"), encoding="utf-8"
    )
    lines = contrast_source.splitlines()
    (tmp_path / "contrast_result_template.csv").write_text(
        "\n".join(lines[:-1]) + "\n", encoding="utf-8"
    )
    errors = validate_result_templates(document, tmp_path)
    assert any("base/dilation/erosion" in error for error in errors)
    assert any("7x2" in error for error in errors)


def _write_contrast_template_mutation(tmp_path, field, replacement):
    (tmp_path / "result_template.csv").write_text(
        (ROOT / "result_template.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with (ROOT / "contrast_result_template.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[0][field] = replacement
    with (tmp_path / "contrast_result_template.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    "field, replacement, expected_error",
    [
        ("left_arm_id", "CAP-I1-K30", "left_arm_id"),
        ("right_arm_id", "CAP-I1-K01", "right_arm_id"),
        ("effect_definition", "left_minus_right", "effect_definition"),
        ("n_paired_seeds", "1", "n_paired_seeds"),
        ("reject_alpha", "0.90", "reject_alpha"),
        ("bootstrap_B", "17", "bootstrap_B"),
    ],
)
def test_v7_cli_rejects_each_contrast_fixed_field_drift(
    tmp_path, field, replacement, expected_error
):
    _write_contrast_template_mutation(tmp_path, field, replacement)
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--templates-root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1, (field, completed.stdout, completed.stderr)
    errors = json.loads(completed.stdout)["matrix_errors"]
    assert any(expected_error in error for error in errors), (field, errors)
    assert any("canonical parsed-CSV sha256" in error for error in errors), (
        field,
        errors,
    )


def test_v7_contrast_template_digest_rejects_an_unlisted_cell_mutation(tmp_path):
    _write_contrast_template_mutation(tmp_path, "notes", "silent drift")
    errors = validate_result_templates(load_matrix(), tmp_path)
    assert any("canonical parsed-CSV sha256" in error for error in errors)
