"""Build deterministic PIRC-21 variant and interaction ablation plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .pirc21_interactions import (
    INTERACTION_DEFINITIONS,
    interaction_output_columns,
    interaction_registry_fingerprint,
    validate_interaction_selection,
)
from .pirc21_pilot import PILOT_INTERACTION_IDS, PILOT_VARIANT_IDS


ABLATION_PLAN_SCHEMA_VERSION = "pirc21-ablation-plan-v1"


class PIRC21AblationError(ValueError):
    """A feature specification cannot produce an unambiguous ablation plan."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) != len(set(result)):
        raise PIRC21AblationError(f"{label} must be ordered and unique")
    return result


def _registries(
    spec: Mapping[str, object],
) -> tuple[
    tuple[str, ...],
    dict[str, Mapping[str, object]],
    dict[str, str],
    dict[str, tuple[str, ...]],
]:
    raw_factors = spec.get("factors")
    raw_variants = spec.get("variants")
    if not isinstance(raw_factors, list) or not isinstance(raw_variants, list):
        raise PIRC21AblationError("feature spec factors and variants must be arrays")
    factor_ids: list[str] = []
    for item in raw_factors:
        if not isinstance(item, Mapping) or not isinstance(item.get("factor_id"), str):
            raise PIRC21AblationError("feature spec contains an invalid factor")
        factor_ids.append(str(item["factor_id"]))
    if not factor_ids or len(factor_ids) != len(set(factor_ids)):
        raise PIRC21AblationError("feature spec factor IDs must be non-empty and unique")

    variants: dict[str, Mapping[str, object]] = {}
    output_owner: dict[str, str] = {}
    outputs_by_variant: dict[str, tuple[str, ...]] = {}
    for item in raw_variants:
        if not isinstance(item, Mapping):
            raise PIRC21AblationError("feature spec contains an invalid variant")
        variant_id = item.get("variant_id")
        factor_id = item.get("factor_id")
        output_columns = item.get("output_columns")
        if (
            not isinstance(variant_id, str)
            or not isinstance(factor_id, str)
            or factor_id not in factor_ids
            or not isinstance(output_columns, list)
        ):
            raise PIRC21AblationError("feature spec contains an invalid variant")
        if variant_id in variants:
            raise PIRC21AblationError(f"duplicate variant ID: {variant_id}")
        outputs: list[str] = []
        for column in output_columns:
            if not isinstance(column, Mapping) or not isinstance(column.get("name"), str):
                raise PIRC21AblationError(f"variant has an invalid output: {variant_id}")
            name = str(column["name"])
            if name in output_owner:
                raise PIRC21AblationError(
                    f"adapter output has multiple variant owners: {name}"
                )
            output_owner[name] = variant_id
            outputs.append(name)
        variants[variant_id] = item
        outputs_by_variant[variant_id] = tuple(outputs)
    if not variants:
        raise PIRC21AblationError("feature spec has no variants")
    return tuple(factor_ids), variants, output_owner, outputs_by_variant


def _interaction_dependencies(
    interaction_ids: Sequence[str], output_owner: Mapping[str, str]
) -> dict[str, tuple[str, ...]]:
    dependencies: dict[str, tuple[str, ...]] = {}
    for interaction_id in interaction_ids:
        definition = INTERACTION_DEFINITIONS.get(interaction_id)
        if definition is None:
            raise PIRC21AblationError(f"unknown interaction: {interaction_id}")
        missing = set(definition.required_columns) - set(output_owner)
        if missing:
            raise PIRC21AblationError(
                f"interaction {interaction_id} has unavailable outputs: {sorted(missing)}"
            )
        dependencies[interaction_id] = tuple(
            dict.fromkeys(output_owner[name] for name in definition.required_columns)
        )
    return dependencies


def _arm(
    *,
    arm_id: str,
    kind: str,
    variant_ids: Sequence[str],
    interaction_ids: Sequence[str],
    variants: Mapping[str, Mapping[str, object]],
    comparison_arm_id: str | None,
    include_validity_indicators: bool,
    factor_id: str | None = None,
    omitted_factor_id: str | None = None,
) -> dict[str, object]:
    selected_variants = tuple(variant_ids)
    selected_interactions = tuple(interaction_ids)
    static_dim = sum(int(variants[value]["output_dim"]) for value in selected_variants)
    interaction_dim = len(interaction_output_columns(selected_interactions))
    numeric_dim = static_dim + interaction_dim
    record: dict[str, object] = {
        "arm_id": arm_id,
        "kind": kind,
        "variant_ids": list(selected_variants),
        "composition_ids": [],
        "interaction_ids": list(selected_interactions),
        "static_numeric_dim": static_dim,
        "interaction_numeric_dim": interaction_dim,
        "numeric_dim": numeric_dim,
        "validity_indicator_dim": numeric_dim if include_validity_indicators else 0,
        "model_condition_dim": numeric_dim
        * (2 if include_validity_indicators else 1),
        "include_validity_indicators": include_validity_indicators,
        "comparison_arm_id": comparison_arm_id,
    }
    if factor_id is not None:
        record["factor_id"] = factor_id
    if omitted_factor_id is not None:
        record["omitted_factor_id"] = omitted_factor_id
    record["selection_sha256"] = _canonical_hash(
        {
            "variant_ids": selected_variants,
            "composition_ids": (),
            "interaction_ids": selected_interactions,
            "include_validity_indicators": include_validity_indicators,
        }
    )
    return record


def build_ablation_plan(
    feature_spec: Mapping[str, object],
    *,
    full_variant_ids: Sequence[str] = PILOT_VARIANT_IDS,
    full_interaction_ids: Sequence[str] = PILOT_INTERACTION_IDS,
    include_validity_indicators: bool = True,
) -> dict[str, object]:
    """Return an ordered, hash-bound ablation plan for one feature spec."""

    if not isinstance(include_validity_indicators, bool):
        raise PIRC21AblationError("include_validity_indicators must be boolean")
    factors, variants, output_owner, outputs_by_variant = _registries(feature_spec)
    selected_variants = _ordered_unique(full_variant_ids, "full_variant_ids")
    selected_interactions = _ordered_unique(
        full_interaction_ids, "full_interaction_ids"
    )
    unknown_variants = set(selected_variants) - set(variants)
    if unknown_variants:
        raise PIRC21AblationError(
            f"full selection has unknown variants: {sorted(unknown_variants)}"
        )
    dependencies = _interaction_dependencies(selected_interactions, output_owner)
    full_outputs = tuple(
        output
        for variant_id in selected_variants
        for output in outputs_by_variant[variant_id]
    )
    try:
        validate_interaction_selection(selected_interactions, full_outputs)
    except ValueError as error:
        raise PIRC21AblationError(str(error)) from error

    arms: list[dict[str, object]] = []
    baseline_id = "B00-no-pirc21"
    arms.append(
        _arm(
            arm_id=baseline_id,
            kind="baseline",
            variant_ids=(),
            interaction_ids=(),
            variants=variants,
            comparison_arm_id=None,
            include_validity_indicators=include_validity_indicators,
        )
    )

    for index, (variant_id, variant) in enumerate(variants.items(), start=1):
        arms.append(
            _arm(
                arm_id=f"V{index:03d}-{variant_id.replace('.', '_')}",
                kind="single_variant",
                variant_ids=(variant_id,),
                interaction_ids=(),
                variants=variants,
                comparison_arm_id=baseline_id,
                include_validity_indicators=include_validity_indicators,
                factor_id=str(variant["factor_id"]),
            )
        )

    reference_by_dependencies: dict[tuple[str, ...], str] = {(): baseline_id}
    nonempty_dependencies = sorted(
        {value for value in dependencies.values() if value},
        key=lambda value: tuple(selected_variants.index(item) for item in value),
    )
    for index, variant_ids in enumerate(nonempty_dependencies, start=1):
        reference_id = f"R{index:03d}-interaction-dependencies"
        reference_by_dependencies[variant_ids] = reference_id
        arms.append(
            _arm(
                arm_id=reference_id,
                kind="interaction_reference",
                variant_ids=variant_ids,
                interaction_ids=(),
                variants=variants,
                comparison_arm_id=baseline_id,
                include_validity_indicators=include_validity_indicators,
            )
        )

    for index, interaction_id in enumerate(selected_interactions, start=1):
        dependency_variants = dependencies[interaction_id]
        definition = INTERACTION_DEFINITIONS[interaction_id]
        arms.append(
            _arm(
                arm_id=f"I{index:03d}-{interaction_id.replace('.', '_')}",
                kind="single_interaction",
                variant_ids=dependency_variants,
                interaction_ids=(interaction_id,),
                variants=variants,
                comparison_arm_id=reference_by_dependencies[dependency_variants],
                include_validity_indicators=include_validity_indicators,
                factor_id=definition.factor_ids[0],
            )
        )

    full_id = "F00-full-mathematical-inputs"
    arms.append(
        _arm(
            arm_id=full_id,
            kind="full",
            variant_ids=selected_variants,
            interaction_ids=selected_interactions,
            variants=variants,
            comparison_arm_id=baseline_id,
            include_validity_indicators=include_validity_indicators,
        )
    )

    for index, factor_id in enumerate(factors, start=1):
        kept_variants = tuple(
            variant_id
            for variant_id in selected_variants
            if str(variants[variant_id]["factor_id"]) != factor_id
        )
        kept_outputs = {
            output
            for variant_id in kept_variants
            for output in outputs_by_variant[variant_id]
        }
        kept_interactions = tuple(
            interaction_id
            for interaction_id in selected_interactions
            if factor_id not in INTERACTION_DEFINITIONS[interaction_id].factor_ids
            and set(INTERACTION_DEFINITIONS[interaction_id].required_columns)
            <= kept_outputs
        )
        arms.append(
            _arm(
                arm_id=f"L{index:03d}-without-{factor_id.replace('_', '-')}",
                kind="leave_one_factor_out",
                variant_ids=kept_variants,
                interaction_ids=kept_interactions,
                variants=variants,
                comparison_arm_id=full_id,
                include_validity_indicators=include_validity_indicators,
                omitted_factor_id=factor_id,
            )
        )

    plan: dict[str, object] = {
        "schema_version": ABLATION_PLAN_SCHEMA_VERSION,
        "feature_spec_id": feature_spec.get("feature_spec_id"),
        "processing_version": feature_spec.get("processing_version"),
        "interaction_registry_sha256": interaction_registry_fingerprint(),
        "include_validity_indicators": include_validity_indicators,
        "counts": {
            "factors": len(factors),
            "registered_variants": len(variants),
            "selected_full_variants": len(selected_variants),
            "selected_full_interactions": len(selected_interactions),
            "arms": len(arms),
        },
        "arms": arms,
    }
    plan["plan_sha256"] = _canonical_hash(plan)
    return plan


def _read_spec(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PIRC21AblationError(f"cannot read feature spec: {path}") from error
    if not isinstance(value, dict):
        raise PIRC21AblationError("feature spec must be a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--without-validity-indicators",
        action="store_true",
        help="Record numeric inputs without paired masks; unsuitable for missing data.",
    )
    args = parser.parse_args(argv)
    plan = build_ablation_plan(
        _read_spec(args.feature_spec),
        include_validity_indicators=not args.without_validity_indicators,
    )
    _write_json(args.output, plan)
    print(json.dumps(plan["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ABLATION_PLAN_SCHEMA_VERSION",
    "PIRC21AblationError",
    "build_ablation_plan",
    "main",
]
