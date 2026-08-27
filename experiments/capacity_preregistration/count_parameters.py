"""Count and audit model capacity for the NEX-381 preregistration.

The declared neural count is intentionally independent of the current
``TimeVaryingNeuralSDE`` skeleton.  The experiment remains locked until the
implementation exposes exactly this parameter contract and a runtime count
matches the declaration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


I1_PARAMETER_GROUPS = ("Gamma", "a", "c", "g", "prior_logits")
FROZEN_STATE_LAYOUT = ("normalized_position_X", "normalized_velocity_V")
FROZEN_STATE_LAYOUT_REF = "common_protocol.state_layout"
NEURAL_DIFFUSION_CONTRACT = {
    "parameter": "ell = (ell_X, ell_V) in R^2",
    "parameter_order": ["ell_X", "ell_V"],
    "B": "diag(exp(ell_X), exp(ell_V))",
    "a": "B @ B.T = diag(exp(2*ell_X), exp(2*ell_V))",
    "noise_coordinates": list(FROZEN_STATE_LAYOUT),
    "state_dependent": False,
    "time_dependent": False,
    "transform_modifiers": None,
    "bridge_correction": (
        "a @ grad(log_h) in state order [X,V]; component 0 acts on X and "
        "component 1 acts on V"
    ),
}
FROZEN_STRUCTURE_CLAIM = "2.48×参数量、非参数匹配的 fitted-pipeline 比较"


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def count_i1_parameters(n_modes: int) -> dict[str, Any]:
    """Return the stored trainable-element count for ``SegmentConstantSDE``.

    The K prior logits are counted as stored/trainable parameters.  Because a
    softmax prior is invariant to a common logit shift, the effective statistical
    degrees of freedom are one smaller; that alternative number is reported but
    never substituted for the registered capacity.
    """

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
    hidden: Sequence[int],
    *,
    state_dim: int = 2,
    env_dim: int = 1,
    diffusion_dim: int = 2,
) -> dict[str, Any]:
    """Count the frozen EnvDriftNet contract used by the preregistration.

    Contract: ``[state, environment] -> hidden -> hidden -> drift`` with a bias
    in every linear layer and a separate trainable diagonal log-diffusion.
    Time is deliberately not an input.  Normalization statistics are fit on the
    training split and are buffers, not trainable parameters.
    """

    if len(hidden) != 2:
        raise ValueError(f"hidden must contain exactly two widths, got {hidden!r}")
    hidden_1 = _positive_int(hidden[0], "hidden[0]")
    hidden_2 = _positive_int(hidden[1], "hidden[1]")
    state_dim = _positive_int(state_dim, "state_dim")
    env_dim = _positive_int(env_dim, "env_dim")
    diffusion_dim = _positive_int(diffusion_dim, "diffusion_dim")
    input_dim = state_dim + env_dim

    breakdown = {
        "input_to_hidden_1.weight": input_dim * hidden_1,
        "input_to_hidden_1.bias": hidden_1,
        "hidden_1_to_hidden_2.weight": hidden_1 * hidden_2,
        "hidden_1_to_hidden_2.bias": hidden_2,
        "hidden_2_to_drift.weight": hidden_2 * state_dim,
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
            "hidden": [hidden_1, hidden_2],
            "drift_output_dim": state_dim,
            "diffusion": NEURAL_DIFFUSION_CONTRACT,
            "time_is_input": False,
        },
        "excluded_buffers": ["train_split_feature_mean", "train_split_feature_std"],
        "excluded_fixed_values": ["dt_scale", "drift_clip"],
    }


def count_module_trainable_parameters(module: Any) -> int:
    """Count trainable tensor elements in an instantiated PyTorch-like module."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _count_from_arm(arm: Mapping[str, Any]) -> int:
    family = arm["model"]["family"]
    if family == "I1":
        return count_i1_parameters(arm["model"]["n_modes"])["stored_trainable_parameters"]
    if family == "neural":
        return count_neural_parameters(
            arm["model"]["hidden"],
            state_dim=arm["model"]["state_dim"],
            env_dim=arm["model"]["env_dim"],
            diffusion_dim=arm["model"]["diffusion_dim"],
        )["stored_trainable_parameters"]
    raise ValueError(f"unsupported model family {family!r}")


def validate_matrix(document: Mapping[str, Any]) -> list[str]:
    """Validate capacity counts and the minimum reproducibility keys per arm."""

    required_arm_keys = {
        "arm_id",
        "question",
        "model",
        "registered_parameter_count",
        "data_lock",
        "splits",
        "seeds",
        "budget",
        "stopping",
        "failure_policy",
        "implementation_status",
    }
    errors: list[str] = []
    seen: set[str] = set()
    arms = document.get("arms")
    if not isinstance(arms, list) or not arms:
        return ["arms must be a non-empty list"]
    if document.get("preregistration_id") != "NEX-381-v3":
        errors.append("preregistration_id must be NEX-381-v3")
    common_protocol = document.get("common_protocol", {})
    if common_protocol.get("state_layout") != list(FROZEN_STATE_LAYOUT):
        errors.append("common state layout must be frozen to [normalized_position_X, normalized_velocity_V]")

    for index, arm in enumerate(arms):
        label = arm.get("arm_id", f"index:{index}")
        missing = sorted(required_arm_keys - set(arm))
        if missing:
            errors.append(f"{label}: missing keys {missing}")
            continue
        if label in seen:
            errors.append(f"{label}: duplicate arm_id")
        seen.add(label)
        try:
            actual = _count_from_arm(arm)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid model contract: {exc}")
            continue
        if actual != arm["registered_parameter_count"]:
            errors.append(
                f"{label}: registered count {arm['registered_parameter_count']} != computed {actual}"
            )
        if arm["model"]["family"] == "I1":
            expected_effective = actual - 1
            if arm.get("registered_effective_degrees_of_freedom") != expected_effective:
                errors.append(
                    f"{label}: effective DoF must be {expected_effective} beside stored count {actual}"
                )
        else:
            if arm["model"].get("state_layout") != FROZEN_STATE_LAYOUT_REF:
                errors.append(f"{label}: neural state layout does not explicitly reference common_protocol.state_layout")
            if (
                arm["model"].get("diffusion_contract")
                != "common_protocol.neural_diffusion_contract"
                or common_protocol.get("neural_diffusion_contract")
                != NEURAL_DIFFUSION_CONTRACT
            ):
                errors.append(f"{label}: neural diffusion contract is not frozen to v3 [X,V] semantics")
        if not arm["seeds"]:
            errors.append(f"{label}: seeds must not be empty")
    evaluation = common_protocol.get("evaluation", {})
    bridge_dual = evaluation.get("bridge_dual", {})
    if bridge_dual.get("state_layout") != FROZEN_STATE_LAYOUT_REF:
        errors.append("bridge dual check must explicitly reference the frozen state layout")
    if bridge_dual.get("hit_event_coordinate") != "state[0] = normalized_position_X only":
        errors.append("bridge hit event must read state[0] = X only")
    delta_probe = evaluation.get("delta_probe", {})
    if delta_probe.get("coordinate_index") != 0:
        errors.append("delta probe coordinate_index must be 0")
    if delta_probe.get("coordinate") != "state[0] = normalized_position_X":
        errors.append("delta probe must read normalized position X from state[0]")
    contrasts = document.get("predeclared_contrasts", [])
    if len(contrasts) != 1:
        errors.append("exactly one predeclared structure contrast is required")
    else:
        contrast = contrasts[0]
        if contrast.get("claim_boundary") != FROZEN_STRUCTURE_CLAIM:
            errors.append("structure contrast claim boundary is not the frozen wording")
        if contrast.get("parameter_ratio_right_over_left") != 2.48:
            errors.append("stored parameter ratio must be exactly 2.48")
        if contrast.get("secondary_ratio_right_over_i1_effective_dof") != 124 / 49:
            errors.append("secondary effective-DoF ratio must be exactly 124/49")
    return errors


def parameter_table() -> list[dict[str, Any]]:
    rows = []
    for n_modes in (1, 3, 10, 30):
        count = count_i1_parameters(n_modes)
        rows.append({
            "family": "I1",
            "capacity": {"n_modes": n_modes},
            "parameters": count["stored_trainable_parameters"],
            "effective_dof": count["effective_degrees_of_freedom"],
        })
    for width in (8, 32, 64, 128):
        count = count_neural_parameters((width, width))
        rows.append({
            "family": "neural",
            "capacity": {"hidden": [width, width]},
            "parameters": count["stored_trainable_parameters"],
            "effective_dof": None,
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("experiment_matrix.json"),
        help="preregistration matrix to validate",
    )
    args = parser.parse_args(argv)
    document = json.loads(args.matrix.read_text(encoding="utf-8"))
    errors = validate_matrix(document)
    print(json.dumps({"parameter_table": parameter_table(), "matrix_errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
