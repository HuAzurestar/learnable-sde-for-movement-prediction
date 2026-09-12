"""Strict loader for the frozen NEX326 process specification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SPEC_PATH = Path(__file__).with_name("experiment.json")
GROUP_COUNTS = {"model": 8, "algorithm": 4, "learning": 5, "inference": 5}
FULL_ANCHORS = {1, 7, 11, 16, 18, 20}
REQUIRED_ARM_FIELDS = {
    "arm_id",
    "group",
    "slot",
    "variant",
    "idea_ids",
    "sources",
    "lineage_role",
    "lineage_confidence",
    "control",
    "unique_change",
    "subconfigs",
    "data_requirements",
    "training_steps",
    "inference_steps",
    "metrics",
    "mechanism_gate",
    "implementation_status",
}


class SpecificationError(ValueError):
    """The checked-in experiment specification violates the frozen contract."""


@dataclass(frozen=True)
class ArmSpec:
    arm_id: int
    group: str
    slot: str
    variant: str
    idea_ids: tuple[str, ...]
    sources: tuple[str, ...]
    lineage_role: str
    lineage_confidence: str
    control: Mapping[str, Any]
    unique_change: str
    subconfigs: tuple[Mapping[str, Any], ...]
    data_requirements: tuple[str, ...]
    training_steps: tuple[str, ...]
    inference_steps: tuple[str, ...]
    metrics: tuple[str, ...]
    mechanism_gate: Mapping[str, Any]
    implementation_status: str
    source_correction: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArmSpec":
        missing = REQUIRED_ARM_FIELDS - payload.keys()
        if missing:
            raise SpecificationError(
                f"arm {payload.get('arm_id', '?')} missing fields: {sorted(missing)}"
            )
        subconfigs = tuple(payload["subconfigs"])
        if not subconfigs:
            raise SpecificationError(f"arm {payload['arm_id']} has no executable subconfig")
        ids = [item.get("subconfig_id") for item in subconfigs]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise SpecificationError(f"arm {payload['arm_id']} has invalid subconfig ids")
        return cls(
            arm_id=int(payload["arm_id"]),
            group=str(payload["group"]),
            slot=str(payload["slot"]),
            variant=str(payload["variant"]),
            idea_ids=tuple(payload["idea_ids"]),
            sources=tuple(payload["sources"]),
            lineage_role=str(payload["lineage_role"]),
            lineage_confidence=str(payload["lineage_confidence"]),
            control=dict(payload["control"]),
            unique_change=str(payload["unique_change"]),
            subconfigs=tuple(dict(item) for item in subconfigs),
            data_requirements=tuple(payload["data_requirements"]),
            training_steps=tuple(payload["training_steps"]),
            inference_steps=tuple(payload["inference_steps"]),
            metrics=tuple(payload["metrics"]),
            mechanism_gate=dict(payload["mechanism_gate"]),
            implementation_status=str(payload["implementation_status"]),
            source_correction=(
                dict(payload["source_correction"])
                if payload.get("source_correction") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: str
    experiment_id: str
    spec_version: str
    full_config_id: str
    full_components: Mapping[str, Any]
    seed: int
    arms: tuple[ArmSpec, ...]
    unnumbered_ideas: tuple[Mapping[str, Any], ...]
    historical_conflicts: tuple[Mapping[str, Any], ...]

    def validate(self) -> None:
        if self.schema_version != "nex326-experiment-spec-v2":
            raise SpecificationError("unsupported NEX326 experiment schema")
        if self.experiment_id != "NEX326" or self.spec_version != "nex326-process-v2":
            raise SpecificationError("unexpected experiment identity")
        ids = [arm.arm_id for arm in self.arms]
        if set(ids) != set(range(1, 23)) or len(ids) != 22:
            raise SpecificationError("numbered arms must be exactly 1..22")
        for group, expected in GROUP_COUNTS.items():
            actual = sum(arm.group == group for arm in self.arms)
            if actual != expected:
                raise SpecificationError(
                    f"group {group} must contain {expected} arms, got {actual}"
                )
        anchors = {arm.arm_id for arm in self.arms if arm.control.get("full_anchor")}
        if anchors != FULL_ANCHORS:
            raise SpecificationError(f"Full anchors mismatch: {sorted(anchors)}")
        for arm in self.arms:
            if arm.control.get("full_anchor") and arm.control.get("config_id") != self.full_config_id:
                raise SpecificationError(f"arm {arm.arm_id} does not reference shared Full")
            if arm.implementation_status != "implemented":
                raise SpecificationError(f"arm {arm.arm_id} is not implemented")
            if arm.lineage_confidence not in {"explicit", "conflict", "inferred"}:
                raise SpecificationError(f"arm {arm.arm_id} has invalid lineage confidence")
        external = {item.get("idea_id"): item for item in self.unnumbered_ideas}
        if external.get("C-4", {}).get("role") != "diagnostic_only":
            raise SpecificationError("C-4 must remain diagnostic_only")
        if external.get("C-8", {}).get("role") != "theory_only":
            raise SpecificationError("C-8 must remain theory_only")
        arm13 = next(arm for arm in self.arms if arm.arm_id == 13)
        correction = arm13.source_correction or {}
        if "C-3" not in arm13.idea_ids or correction.get("canonical_ref") != "C-3/NEX-95":
            raise SpecificationError("arm 13 source correction is missing")
        arm17 = next(arm for arm in self.arms if arm.arm_id == 17)
        terrain = [s for s in arm17.subconfigs if s["subconfig_id"] == "terrain"]
        if len(terrain) != 1:
            raise SpecificationError("17-terrain must be an arm 17 subconfig")

    @property
    def executions(self) -> tuple[tuple[ArmSpec, Mapping[str, Any]], ...]:
        return tuple((arm, subconfig) for arm in self.arms for subconfig in arm.subconfigs)


def load_experiment_spec(path: Path | str = SPEC_PATH) -> ExperimentSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults = payload.get("arm_defaults", {})
    expanded_arms = []
    for item in payload["arms"]:
        expanded = {**defaults, **item}
        expanded["control"] = {**defaults.get("control", {}), **item.get("control", {})}
        expanded["mechanism_gate"] = {
            **defaults.get("mechanism_gate", {}),
            **item.get("mechanism_gate", {}),
        }
        expanded_arms.append(expanded)
    spec = ExperimentSpec(
        schema_version=payload["schema_version"],
        experiment_id=payload["experiment_id"],
        spec_version=payload["spec_version"],
        full_config_id=payload["full_config"]["config_id"],
        full_components=dict(payload["full_config"]["components"]),
        seed=int(payload["protocol"]["seed"]),
        arms=tuple(ArmSpec.from_dict(item) for item in expanded_arms),
        unnumbered_ideas=tuple(payload["unnumbered_ideas"]),
        historical_conflicts=tuple(payload["historical_conflicts"]),
    )
    spec.validate()
    return spec


__all__ = [
    "ArmSpec",
    "ExperimentSpec",
    "FULL_ANCHORS",
    "GROUP_COUNTS",
    "SPEC_PATH",
    "SpecificationError",
    "load_experiment_spec",
]
