"""Count capacity and exact-validate the frozen NEX-381-v7 contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


I1_PARAMETER_GROUPS = ("Gamma", "a", "c", "g", "prior_logits")
FROZEN_STATE_LAYOUT = ("normalized_position_X", "normalized_velocity_V")
FROZEN_STATE_LAYOUT_REF = "common_protocol.state_layout"
FROZEN_SEEDS = (20260814, 20260815, 20260816, 20260817, 20260818)
FROZEN_MATRIX_SHA256 = "e96bb1db4ca5b0414900e229f4fb79e8f3d507ef792aafae17471d3fcff2b135"
FROZEN_CONTRAST_TEMPLATE_SHA256 = "05db1811480bd629f93137630bcdcc7d7df3f542003602f0ae0aba4bd18044f5"
FROZEN_CONTRAST_EFFECT_DEFINITION = "right_minus_left"
FROZEN_REJECT_ALPHA = "0.05"
FROZEN_BOOTSTRAP_B = "2000"
FROZEN_SPLITS = {
    "source": "global_splits.json",
    "train": {"key": "train", "roles": ["fit_model", "fit_state_normalizer", "fit_environment_normalizer"]},
    "validation": {"key": "val", "roles": ["checkpoint_selection", "stopping"]},
    "evaluation": {"key": "eval", "roles": ["locked_final_metrics_only"], "unlock_condition": "every registered arm success/failure record frozen"},
    "leakage_unit": "file_id",
    "segment_overlap_check": True,
}
FROZEN_RNG_CONTRACT = {
    "paired_seeds_ref": "common_protocol.paired_seeds",
    "same_seed_scopes": ["all_8_arms", "initialization", "minibatch_order", "rollout", "bootstrap_pairing"],
    "stream_derivation": {"algorithm": "numpy.random.SeedSequence", "spawn_count": 3, "stream_order": ["train", "rollout", "bootstrap"]},
    "determinism": {"dtype": "float64", "torch_deterministic_algorithms": True, "diagnostic_rerun_replaces_registered_run": False},
}
FROZEN_SOLAR_ALIGNMENT = {
    "condition_time_column": "t",
    "condition_time_conversion": "datetime64[s] to signed integer Unix seconds",
    "start_source": "hashed segment_start_map.absolute_start_epoch",
    "duration_seconds": "segment.t[-1]-segment.t[0]",
    "lower_bound": "inclusive",
    "upper_bound": "inclusive",
    "row_selection": "all and only rows with start<=t_epoch<=start+duration",
    "ordering": "source row order does not affect arithmetic mean",
}
FROZEN_I1_TRAINING = {
    "estimator": {"name": "expectation_maximization", "objective": "raw_state_train_negative_log_likelihood"},
    "maximum_iterations": 50,
    "stopping": {"metric": "relative_train_nll_improvement", "comparison": "less_than", "tolerance": 1e-5, "check_after_each_completed_iteration": True, "otherwise": "stop_at_maximum_iterations"},
    "validation_hyperparameter_search": False,
    "checkpoint": {"selection": "final_accepted_iteration", "tie_break": "not_applicable"},
    "forecast_samples_per_segment": 1000,
}
FROZEN_NEURAL_TRAINING = {
    "objective": {"name": "one_step_gaussian_transition_nll", "state": "normalized_[X,V]", "mean": "z+(dt/60)*b_clipped(z,e)", "covariance": "(dt/60)*diag(exp(2ell_X),exp(2ell_V))"},
    "optimizer": {"name": "torch.optim.Adam", "parameters": {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False, "maximize": False, "foreach": False, "fused": False}},
    "batching": {"batch_size": 256, "shuffle": True, "drop_last": False},
    "gradient_clipping": "none",
    "maximum_epochs": 300,
    "stopping": {"metric": "validation_nll", "mode": "min", "patience_completed_epochs": 30, "min_delta_absolute": 1e-5},
    "checkpoint": {"selection": "minimum_validation_nll", "exact_tie_break": "earliest_epoch"},
    "forecast_samples_per_segment": 1000,
}
FROZEN_TIME_GRID = {
    "start_seconds": 0.0,
    "regular_step_seconds": 1.0,
    "construction": "t_k=min(k*regular_step_seconds,T)",
    "include_start": True,
    "include_exact_endpoint": True,
    "duplicate_endpoint_policy": "store endpoint once",
}
FROZEN_EVENT_HIT_RULE = {
    "quantifier": "exists",
    "grid_points": "all registered points including start and exact endpoint",
    "membership": "closed_interval",
    "coordinate_index": 0,
    "velocity_read": False,
}
FROZEN_DELTA_FAILURE = {
    "nonfinite_sigma_hist": "unavailable",
    "zero_sigma_hist": "unavailable",
    "empty_erosion": "unavailable",
    "replacement_delta_allowed": False,
    "post_failure_reselection_allowed": False,
    "result_literal": "NA",
}
FROZEN_MANIFEST_FIELDS = (
    "dataset_id",
    "schema_version",
    "unified_full_leg_sha256",
    "global_splits_sha256",
    "segment_start_map_path",
    "segment_start_map_sha256",
    "condition_file_manifest_path",
    "condition_file_manifest_sha256",
    "state_adapter_id",
    "state_normalization_sha256",
    "environment_feature_id",
    "environment_normalization_sha256",
    "preprocessing_git_sha",
    "created_at_utc",
)
NEURAL_DIFFUSION_CONTRACT = {
    "parameter": "ell = (ell_X, ell_V) in R^2",
    "parameter_order": ["ell_X", "ell_V"],
    "B": "diag(exp(ell_X), exp(ell_V))",
    "a": "B @ B.T = diag(exp(2*ell_X), exp(2*ell_V))",
    "noise_coordinates": list(FROZEN_STATE_LAYOUT),
    "state_dependent": False,
    "time_dependent": False,
    "transform_modifiers": None,
    "initial_value": ["log(0.1)", "log(0.1)"],
    "bridge_correction": (
        "a @ grad(log_h) in state order [X,V]; component 0 acts on X and "
        "component 1 acts on V"
    ),
}
FROZEN_STRUCTURE_CLAIM = "2.48×参数量、非参数匹配的 fitted-pipeline 比较"
EXPECTED_CONTRASTS = (
    ("I1-K03-vs-K01", "CAP-I1-K01", "CAP-I1-K03", "I1-capacity"),
    ("I1-K10-vs-K03", "CAP-I1-K03", "CAP-I1-K10", "I1-capacity"),
    ("I1-K30-vs-K10", "CAP-I1-K10", "CAP-I1-K30", "I1-capacity"),
    ("NN-H032-vs-H008", "CAP-NN-H008", "CAP-NN-H032", "neural-capacity"),
    ("NN-H064-vs-H032", "CAP-NN-H032", "CAP-NN-H064", "neural-capacity"),
    ("NN-H128-vs-H064", "CAP-NN-H064", "CAP-NN-H128", "neural-capacity"),
    ("STRUCT-I1-K10-vs-NN-H008", "CAP-I1-K10", "CAP-NN-H008", "structure"),
)
FROZEN_ARM_REFS = {
    "data_lock": "common_protocol.data_lock",
    "splits": "common_protocol.splits",
    "state_adapter": "common_protocol.state_adapter",
    "rng": "common_protocol.rng_contract",
    "seeds": "common_protocol.paired_seeds",
    "evaluation": "common_protocol.evaluation",
}
ARM_RESULT_HEADER = (
    "preregistration_id", "execution_git_sha", "dataset_id", "global_splits_sha256",
    "arm_id", "model_family", "state_layout", "state_adapter_id",
    "environment_feature_id", "registered_parameter_count",
    "registered_effective_degrees_of_freedom", "runtime_trainable_parameter_count",
    "seed", "run_status", "failure_stage", "failure_reason", "converged",
    "iterations_or_epochs", "selected_checkpoint", "train_objective_final",
    "val_objective_best", "energy_half_mean", "hdr90_coverage", "hdr90_abs_error",
    "event_id", "domain_D", "region_A", "full_support_bin_edges",
    "dual_pi_plus_surv_abs_error", "dual_reconstruction_l1", "dual_hit_count",
    "dual_non_hit_count", "dual_undefined_stratum", "dual_algebra_pass", "fp_mc_l1",
    "sigma_hist", "delta", "delta_probe_direction", "delta_probe_status",
    "wall_time_seconds", "peak_memory_mb", "notes",
)
CONTRAST_RESULT_HEADER = (
    "preregistration_id", "execution_git_sha", "dataset_id", "global_splits_sha256",
    "contrast_id", "left_arm_id", "right_arm_id", "metric_id", "effect_definition",
    "n_paired_seeds", "n_paired_segments", "estimate",
    "pointwise_ci_low", "pointwise_ci_high", "simultaneous_ci_low",
    "simultaneous_ci_high", "raw_p_value", "holm_family", "holm_rank",
    "holm_family_size", "holm_adjusted_p", "reject_alpha", "bootstrap_B", "status",
    "notes",
)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def count_i1_parameters(n_modes: int) -> dict[str, Any]:
    """Count stored parameters and separately report identifiable prior DoF."""

    n_modes = _positive_int(n_modes, "n_modes")
    breakdown = {name: n_modes for name in I1_PARAMETER_GROUPS}
    return {
        "family": "I1",
        "stored_trainable_parameters": sum(breakdown.values()),
        "effective_degrees_of_freedom": 5 * n_modes - 1,
        "breakdown": breakdown,
        "included": list(I1_PARAMETER_GROUPS),
        "excluded_fixed_values": ["kappa", "dt_ref"],
    }


def count_neural_parameters(
    hidden: Sequence[int], *, state_dim: int = 2, env_dim: int = 1,
    diffusion_dim: int = 2,
) -> dict[str, Any]:
    """Count ``[X,V,solar] -> H -> H -> [b_X,b_V]`` plus two log scales."""

    if len(hidden) != 2:
        raise ValueError(f"hidden must contain exactly two widths, got {hidden!r}")
    h1 = _positive_int(hidden[0], "hidden[0]")
    h2 = _positive_int(hidden[1], "hidden[1]")
    state_dim = _positive_int(state_dim, "state_dim")
    env_dim = _positive_int(env_dim, "env_dim")
    diffusion_dim = _positive_int(diffusion_dim, "diffusion_dim")
    input_dim = state_dim + env_dim
    breakdown = {
        "input_to_hidden_1.weight": input_dim * h1,
        "input_to_hidden_1.bias": h1,
        "hidden_1_to_hidden_2.weight": h1 * h2,
        "hidden_1_to_hidden_2.bias": h2,
        "hidden_2_to_drift.weight": h2 * state_dim,
        "hidden_2_to_drift.bias": state_dim,
        "diagonal_log_diffusion": diffusion_dim,
    }
    return {
        "family": "neural",
        "stored_trainable_parameters": sum(breakdown.values()),
        "breakdown": breakdown,
        "contract": {
            "input_features": ["state[2]", "environment[1]"],
            "input_dim": input_dim,
            "hidden": [h1, h2],
            "drift_output_dim": state_dim,
            "diffusion": NEURAL_DIFFUSION_CONTRACT,
            "time_is_input": False,
        },
        "excluded_buffers": ["train_split_feature_mean", "train_split_feature_std"],
        "excluded_fixed_values": ["dt_scale", "drift_clip"],
    }


def count_module_trainable_parameters(module: Any) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _count_from_arm(arm: Mapping[str, Any]) -> int:
    model = arm["model"]
    if model["family"] == "I1":
        return count_i1_parameters(model["n_modes"])["stored_trainable_parameters"]
    if model["family"] == "neural":
        return count_neural_parameters(
            model["hidden"], state_dim=model["state_dim"], env_dim=model["env_dim"],
            diffusion_dim=model["diffusion_dim"],
        )["stored_trainable_parameters"]
    raise ValueError(f"unsupported model family {model['family']!r}")


def _expect(errors: list[str], actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        errors.append(f"{label} must equal frozen v7 value {expected!r}; got {actual!r}")


def canonical_matrix_sha256(document: Mapping[str, Any]) -> str:
    """Hash the entire parsed matrix with deterministic JSON serialization."""

    payload = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_csv_sha256(rows: Sequence[Sequence[str]]) -> str:
    """Hash parsed CSV cells so newline style and quoting cannot affect the lock."""

    payload = json.dumps(
        rows, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_matrix(document: Mapping[str, Any]) -> list[str]:
    """Validate every field needed to resolve any arm without investigator choice."""

    errors: list[str] = []
    _expect(errors, document.get("schema_version"), "7.0", "schema_version")
    _expect(errors, document.get("preregistration_id"), "NEX-381-v7", "preregistration_id")
    _expect(errors, document.get("approval_state"), "draft_pending_v7_contrast_template_diff_review", "approval_state")
    supersedes = document.get("supersedes", {})
    _expect(errors, supersedes.get("preregistration_id"), "NEX-381-v6", "supersedes.preregistration_id")
    _expect(errors, supersedes.get("git_sha"), "646e62ec27d586a5793a0a4759b595d4ce758d3b", "supersedes.git_sha")

    common = document.get("common_protocol", {})
    lock = common.get("data_lock", {})
    _expect(errors, lock.get("required_manifest_fields"), list(FROZEN_MANIFEST_FIELDS), "data manifest fields")
    _expect(errors, lock.get("hash_algorithm"), "sha256 lowercase hex", "data manifest hash algorithm")
    _expect(errors, common.get("splits"), FROZEN_SPLITS, "split and leakage contract")
    _expect(errors, common.get("paired_seeds"), list(FROZEN_SEEDS), "paired seeds")
    _expect(errors, common.get("rng_contract"), FROZEN_RNG_CONTRACT, "RNG contract")
    _expect(errors, common.get("state_layout"), list(FROZEN_STATE_LAYOUT), "state layout")

    adapter = common.get("state_adapter", {})
    _expect(errors, adapter.get("adapter_id"), "phase_space_1d_coord0_v1", "state adapter id")
    _expect(errors, adapter.get("call"), "data.loader.to_phase_space_1d(segment, coord=0)", "state adapter call")
    if "population std (ddof=0)" not in adapter.get("normalization", ""):
        errors.append("state adapter must freeze train population normalization")
    coordinate_policy = adapter.get("model_coordinate_policy", {})
    if "only in raw" not in coordinate_policy.get("I1", "") or "never feed z" not in coordinate_policy.get("I1", ""):
        errors.append("I1 must fit and exact-rollout only in raw [X,V]")
    if "at every registered grid point" not in coordinate_policy.get("evaluation", ""):
        errors.append("I1 raw paths must transform to common normalized state at every evaluation point")
    environment = common.get("environment_feature", {})
    _expect(errors, environment.get("feature_id"), "aligned_solar_elev_mean_v1", "environment feature id")
    _expect(errors, environment.get("dimension"), 1, "environment dimension")
    _expect(errors, environment.get("alignment"), FROZEN_SOLAR_ALIGNMENT, "solar alignment")
    if "solar_elev only" not in environment.get("source", ""):
        errors.append("environment source must freeze the solar_elev column only")
    start_map = environment.get("start_map", "")
    for token in ("segment_id,file_id,absolute_start_epoch", "segment_id unique", "hash must equal segment_start_map_sha256"):
        if token not in start_map:
            errors.append(f"solar start map contract missing {token}")
    file_map = environment.get("condition_file_map", "")
    for token in ("file_id,relative_path,sha256", "file_id unique", "duplicate candidate files fail", "runtime glob selection is forbidden"):
        if token not in file_map:
            errors.append(f"condition file manifest contract missing {token}")
    if "every aligned solar_elev value finite" not in environment.get("aggregation", ""):
        errors.append("environment aggregation must fail on any nonfinite aligned solar value")
    if "filtering nonfinite rows is forbidden" not in environment.get("aggregation", ""):
        errors.append("environment aggregation must forbid finite-row filtering")
    if "day_fraction" not in environment.get("aggregation", ""):
        errors.append("environment aggregation must explicitly exclude day_fraction")
    if "population std (ddof=0)" not in environment.get("normalization", ""):
        errors.append("environment feature must freeze train population normalization")

    _expect(errors, common.get("neural_diffusion_contract"), NEURAL_DIFFUSION_CONTRACT, "neural diffusion contract")
    architecture = common.get("neural_architecture", {})
    _expect(errors, architecture.get("input_order"), ["normalized_position_X", "normalized_velocity_V", "normalized_solar_elev"], "neural input order")
    _expect(errors, architecture.get("layers"), "Linear(3,H)-Tanh-Linear(H,H)-Tanh-Linear(H,2)", "neural layers")
    _expect(errors, architecture.get("dt_scale_seconds"), 60.0, "neural dt_scale")
    if "[-10.0,10.0]" not in architecture.get("drift_clip", ""):
        errors.append("neural drift_clip must be elementwise [-10,10]")
    if "xavier_uniform_" not in architecture.get("initialization", ""):
        errors.append("neural initialization must be Xavier uniform with frozen biases")

    training = common.get("training", {})
    _expect(errors, training.get("i1"), FROZEN_I1_TRAINING, "I1 training, stopping and forecast budget")
    _expect(errors, training.get("neural"), FROZEN_NEURAL_TRAINING, "neural objective, optimizer, stopping, checkpoint and forecast budget")
    rollout = common.get("rollout", {})
    _expect(errors, rollout.get("time_grid"), FROZEN_TIME_GRID, "rollout and event time grid")
    if "raw r=[X_raw,V_raw]" not in rollout.get("I1", "") or "never propagate z" not in rollout.get("I1", ""):
        errors.append("I1 rollout must preserve raw dX=Vdt dynamics and transform only for evaluation")
    if "Euler-Maruyama" not in rollout.get("neural", "") or "no adaptive stepping" not in rollout.get("neural", ""):
        errors.append("neural rollout must freeze non-adaptive Euler-Maruyama")

    evaluation = common.get("evaluation", {})
    energy = evaluation.get("energy_half", {})
    _expect(errors, energy.get("state"), "complete normalized state [X,V] at endpoint", "Energy state")
    if "U-statistic" not in energy.get("finite_sample_estimator", "") or energy.get("M") != 1000:
        errors.append("Energy must freeze the M=1000 finite-sample U-statistic")
    hdr = evaluation.get("hdr90", {})
    for token in ("2D Gaussian KDE", "M=1000", "M-1", "M^(-1/6)", "1e-9*I"):
        if token not in hdr.get("density", ""):
            errors.append(f"HDR90 density contract missing {token}")
    if "ceil(0.10*M)=100" not in hdr.get("threshold", "") or "901/1000" not in hdr.get("membership", ""):
        errors.append("HDR90 threshold/tie boundary algorithm is not frozen")
    if "at least 90.1% forecast-sample mass" not in hdr.get("report", ""):
        errors.append("HDR90 must state its conservative finite-sample mass")
    event = evaluation.get("event", {})
    _expect(errors, event.get("event_id"), "hit_X_origin_interval_v1", "event id")
    _expect(errors, event.get("coordinate"), "state[0] = normalized_position_X only", "event coordinate")
    _expect(errors, event.get("domain_D"), [-8.0, 8.0], "event domain D")
    _expect(errors, event.get("region_A"), [[-0.5, 0.5]], "event region A")
    _expect(errors, event.get("time_grid_ref"), "common_protocol.rollout.time_grid", "event time grid reference")
    _expect(errors, event.get("hit_rule"), FROZEN_EVENT_HIT_RULE, "event hit rule")
    expected_edges = ["-inf", -8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, "inf"]
    _expect(errors, event.get("full_support_bin_edges"), expected_edges, "event full-support bins")
    dual = evaluation.get("bridge_dual", {})
    _expect(errors, dual.get("event_ref"), "common_protocol.evaluation.event", "bridge event reference")
    if "state[0]=X then state[1]=V" not in dual.get("neural_correction", ""):
        errors.append("neural bridge correction must act on X then V")
    for token in ("grad_r(log_h)=S^-T grad_z(log_h)", "a_r S^-T", "S^-1 a_r S^-T"):
        if token not in dual.get("I1_raw_correction", ""):
            errors.append(f"I1 bridge affine chain rule missing {token}")
    for token in ("z=S^-1(r-m)", "A_raw=m_X+s_X*A", "closed boundaries preserved"):
        if token not in dual.get("I1_event_transform", ""):
            errors.append(f"I1 event coordinate transform missing {token}")
    delta = evaluation.get("delta_probe", {})
    _expect(errors, delta.get("coordinate_index"), 0, "delta coordinate index")
    _expect(errors, delta.get("directions"), ["base", "dilation", "erosion"], "delta directions")
    _expect(errors, delta.get("failure"), FROZEN_DELTA_FAILURE, "delta failure policy")
    if "state_i[T,0]-state_i[0,0]" not in delta.get("displacement", ""):
        errors.append("delta must use normalized coordinate zero only")
    inference = evaluation.get("inference", {})
    _expect(errors, inference.get("contrast_metric_ids"), ["energy_half", "hdr90_abs_calibration_error"], "contrast metric enum")
    for token in ("B=2000", "length-5 seed index vector", "length-N_eval", "Cartesian product", "all 8 arms", "all 7 contrasts", "never nested"):
        if token not in inference.get("bootstrap", ""):
            errors.append(f"crossed bootstrap missing {token}")
    if "every contrast" not in inference.get("family_shared_draws", "") or "same matrix-wide index vectors" not in inference.get("family_shared_draws", ""):
        errors.append("all family contrasts must share replicate index vectors")
    for token in ("Q_0.025", "Q_0.975", "Hyndman-Fan type 7", "method='linear'"):
        if token not in inference.get("pointwise_ci", ""):
            errors.append(f"pointwise CI contract missing {token}")
    if "approximate centered-bootstrap" not in inference.get("raw_p", "") or "not an exact" not in inference.get("raw_p", ""):
        errors.append("raw p-value approximation status is not frozen")
    for token in ("I1-capacity=3", "neural-capacity=3", "structure=1", "lexical tie-break"):
        if token not in inference.get("holm", ""):
            errors.append(f"Holm family contract missing {token}")
    for token in ("Q_0.95", "max_c", "type-7", "family sizes 3,3,1"):
        if token not in inference.get("simultaneous_ci", ""):
            errors.append(f"simultaneous CI contract missing {token}")
    for token in ("8 arms x 5 seeds x N_eval", "all 7 contrast rows", "literal NA", "no complete-case deletion"):
        if token not in inference.get("complete_cube_policy", ""):
            errors.append(f"failure propagation contract missing {token}")

    required_arm_keys = {"arm_id", "question", "model", "registered_parameter_count", *FROZEN_ARM_REFS, "environment_feature", "training", "rollout", "failure_policy", "implementation_status"}
    arms = document.get("arms")
    if not isinstance(arms, list) or not arms:
        return errors + ["arms must be a non-empty list"]
    seen: set[str] = set()
    for index, arm in enumerate(arms):
        label = arm.get("arm_id", f"index:{index}")
        missing = sorted(required_arm_keys - set(arm))
        if missing:
            errors.append(f"{label}: missing executable keys {missing}")
            continue
        if label in seen:
            errors.append(f"{label}: duplicate arm_id")
        seen.add(label)
        for key, expected in FROZEN_ARM_REFS.items():
            _expect(errors, arm.get(key), expected, f"{label}.{key}")
        try:
            actual = _count_from_arm(arm)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid model contract: {exc}")
            continue
        _expect(errors, arm.get("registered_parameter_count"), actual, f"{label} registered count")
        if arm["model"]["family"] == "I1":
            _expect(errors, arm.get("registered_effective_degrees_of_freedom"), actual - 1, f"{label} effective DoF")
            _expect(errors, arm.get("environment_feature"), None, f"{label} environment")
            _expect(errors, arm.get("training"), "common_protocol.training.i1", f"{label} training")
            _expect(errors, arm.get("rollout"), "common_protocol.rollout.I1", f"{label} rollout")
        else:
            model = arm["model"]
            _expect(errors, model.get("state_layout"), FROZEN_STATE_LAYOUT_REF, f"{label} state layout ref")
            _expect(errors, model.get("diffusion_contract"), "common_protocol.neural_diffusion_contract", f"{label} diffusion ref")
            _expect(errors, model.get("architecture"), "common_protocol.neural_architecture", f"{label} architecture ref")
            _expect(errors, arm.get("environment_feature"), "common_protocol.environment_feature", f"{label} environment ref")
            _expect(errors, arm.get("training"), "common_protocol.training.neural", f"{label} training")
            _expect(errors, arm.get("rollout"), "common_protocol.rollout.neural", f"{label} rollout")
    if len(arms) != 8:
        errors.append("exactly eight registered arms are required")

    contrasts = document.get("predeclared_contrasts", [])
    actual_contrasts = [(c.get("contrast_id"), c.get("left"), c.get("right"), c.get("family")) for c in contrasts]
    _expect(errors, actual_contrasts, list(EXPECTED_CONTRASTS), "predeclared contrast identities")
    if contrasts:
        structure = contrasts[-1]
        _expect(errors, structure.get("claim"), FROZEN_STRUCTURE_CLAIM, "structure claim")
        _expect(errors, structure.get("parameter_ratio_right_over_left"), 2.48, "structure stored ratio")
        _expect(errors, structure.get("secondary_ratio_right_over_i1_effective_dof"), 124 / 49, "structure effective-DoF ratio")

    schemas = common.get("result_schemas", {})
    arm_schema = schemas.get("arm_results", {})
    _expect(errors, arm_schema.get("file"), "result_template.csv", "arm result filename")
    _expect(errors, arm_schema.get("primary_key"), ["preregistration_id", "execution_git_sha", "dataset_id", "global_splits_sha256", "arm_id", "seed", "delta_probe_direction"], "arm result primary key")
    _expect(errors, arm_schema.get("direction_enum"), ["base", "dilation", "erosion"], "arm result direction enum")
    if "delta_probe_value is forbidden" not in arm_schema.get("rule", ""):
        errors.append("arm result schema must remove ambiguous delta_probe_value")
    contrast_schema = schemas.get("contrast_results", {})
    _expect(errors, contrast_schema.get("file"), "contrast_result_template.csv", "contrast result filename")
    _expect(errors, contrast_schema.get("primary_key"), ["preregistration_id", "execution_git_sha", "dataset_id", "global_splits_sha256", "contrast_id", "metric_id"], "contrast result primary key")
    _expect(errors, contrast_schema.get("metric_enum"), ["energy_half", "hdr90_abs_calibration_error"], "contrast result metric enum")
    _expect(errors, contrast_schema.get("required_record_count"), 14, "contrast required record count")
    if "Cartesian product" not in contrast_schema.get("required_record_set", "") or "no extra or missing rows" not in contrast_schema.get("required_record_set", ""):
        errors.append("contrast result set must be the exact 7x2 Cartesian product")
    _expect(
        errors,
        canonical_matrix_sha256(document),
        FROZEN_MATRIX_SHA256,
        "canonical full-matrix sha256",
    )
    return errors


def validate_result_templates(document: Mapping[str, Any], root: Path) -> list[str]:
    """Lock CSV headers, delta records and every fixed contrast-template cell."""

    errors: list[str] = []
    schemas = document.get("common_protocol", {}).get("result_schemas", {})
    for schema_name, expected_header in (("arm_results", ARM_RESULT_HEADER), ("contrast_results", CONTRAST_RESULT_HEADER)):
        filename = schemas.get(schema_name, {}).get("file")
        if not filename:
            errors.append(f"{schema_name}: missing template filename")
            continue
        path = root / filename
        if not path.is_file():
            errors.append(f"{schema_name}: missing template {filename}")
            continue
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
        if not rows or tuple(rows[0]) != expected_header:
            errors.append(f"{schema_name}: CSV header does not match frozen schema")
            continue
        if any(len(row) != len(rows[0]) for row in rows[1:]):
            errors.append(f"{schema_name}: template row/header column count mismatch")
        if any(row[0] != "NEX-381-v7" for row in rows[1:]):
            errors.append(f"{schema_name}: template rows must identify NEX-381-v7")
        if schema_name == "arm_results":
            if "delta_probe_value" in rows[0]:
                errors.append("arm_results: delta_probe_value column is forbidden")
            direction_index = rows[0].index("delta_probe_direction")
            directions = [row[direction_index] for row in rows[1:]]
            if directions != ["base", "dilation", "erosion"]:
                errors.append("arm_results: require exactly one ordered base/dilation/erosion template row")
        else:
            contrast_index = rows[0].index("contrast_id")
            metric_index = rows[0].index("metric_id")
            left_index = rows[0].index("left_arm_id")
            right_index = rows[0].index("right_arm_id")
            effect_index = rows[0].index("effect_definition")
            seeds_index = rows[0].index("n_paired_seeds")
            family_index = rows[0].index("holm_family")
            size_index = rows[0].index("holm_family_size")
            alpha_index = rows[0].index("reject_alpha")
            bootstrap_index = rows[0].index("bootstrap_B")
            contrast_ids = [item[0] for item in EXPECTED_CONTRASTS]
            metric_ids = ["energy_half", "hdr90_abs_calibration_error"]
            expected_records = [(contrast_id, metric_id) for metric_id in metric_ids for contrast_id in contrast_ids]
            actual_records = [(row[contrast_index], row[metric_index]) for row in rows[1:]]
            if actual_records != expected_records:
                errors.append("contrast_results: rows must equal the ordered 7x2 contrast-metric Cartesian product")
            family_sizes = {"I1-capacity": "3", "neural-capacity": "3", "structure": "1"}
            expected_rows = [
                (contrast_id, left, right, family, metric_id)
                for metric_id in metric_ids
                for contrast_id, left, right, family in EXPECTED_CONTRASTS
            ]
            for row, expected in zip(rows[1:], expected_rows):
                contrast_id, left, right, family, metric_id = expected
                if row[left_index] != left:
                    errors.append(f"contrast_results: left_arm_id must equal {left!r} for {contrast_id}")
                if row[right_index] != right:
                    errors.append(f"contrast_results: right_arm_id must equal {right!r} for {contrast_id}")
                if row[effect_index] != FROZEN_CONTRAST_EFFECT_DEFINITION:
                    errors.append("contrast_results: effect_definition must equal 'right_minus_left'")
                if row[seeds_index] != str(len(FROZEN_SEEDS)):
                    errors.append("contrast_results: n_paired_seeds must equal 5")
                if family and row[family_index] != f"{metric_id}:{family}":
                    errors.append("contrast_results: Holm family id does not match metric and contrast family")
                if family and row[size_index] != family_sizes[family]:
                    errors.append("contrast_results: Holm family size does not match frozen 3/3/1")
                if row[alpha_index] != FROZEN_REJECT_ALPHA:
                    errors.append("contrast_results: reject_alpha must equal 0.05")
                if row[bootstrap_index] != FROZEN_BOOTSTRAP_B:
                    errors.append("contrast_results: bootstrap_B must equal 2000")
            if canonical_csv_sha256(rows) != FROZEN_CONTRAST_TEMPLATE_SHA256:
                errors.append("contrast_results: canonical parsed-CSV sha256 does not match frozen v7 template")
    return errors


def parameter_table() -> list[dict[str, Any]]:
    rows = []
    for n_modes in (1, 3, 10, 30):
        count = count_i1_parameters(n_modes)
        rows.append({"family": "I1", "capacity": {"n_modes": n_modes}, "parameters": count["stored_trainable_parameters"], "effective_dof": count["effective_degrees_of_freedom"]})
    for width in (8, 32, 64, 128):
        count = count_neural_parameters((width, width))
        rows.append({"family": "neural", "capacity": {"hidden": [width, width]}, "parameters": count["stored_trainable_parameters"], "effective_dof": None})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path(__file__).with_name("experiment_matrix.json"))
    parser.add_argument("--templates-root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args(argv)
    document = json.loads(args.matrix.read_text(encoding="utf-8"))
    errors = validate_matrix(document) + validate_result_templates(document, args.templates_root)
    print(json.dumps({"parameter_table": parameter_table(), "matrix_errors": errors}, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
