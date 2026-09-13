"""Validate a future PIRC-20 cohort before any confirmatory target is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from data.pirc20 import PIRC20Cohort, load_pirc20_cohort


class ConfirmationReadinessError(ValueError):
    """A proposed confirmation cohort is not independent or not policy-bound."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmationReadinessError(f"{label} cannot be read") from exc
    if not isinstance(payload, dict):
        raise ConfirmationReadinessError(f"{label} must contain a JSON object")
    return payload


def _identifier_sets(cohort: PIRC20Cohort) -> dict[str, set[str]]:
    result = {
        "sample_id": set(),
        "file_id": set(),
        "segment_id": set(),
        "independent_block_id": set(),
    }
    for sample in cohort.iter_samples():
        for name in result:
            result[name].add(str(getattr(sample, name)))
    return result


def _derive_seeds(policy_hash: str, cohort_hash: str, count: int) -> list[int]:
    seeds: list[int] = []
    used: set[int] = set()
    for index in range(count):
        digest = hashlib.sha256(
            f"{policy_hash}:{cohort_hash}:{index}".encode("ascii")
        ).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        seed = seed or 1
        while seed in used:
            seed = 1 if seed == 0x7FFFFFFF else seed + 1
        used.add(seed)
        seeds.append(seed)
    return seeds


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ConfirmationReadinessError(f"stale staging file exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def build_confirmation_readiness(
    policy_path: Path | str,
    candidate_selection_path: Path | str,
    discovery_cohort_path: Path | str,
    confirmation_cohort_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Prove zero identifier overlap without unlocking confirmation targets."""

    policy_file = Path(policy_path).resolve()
    selection_file = Path(candidate_selection_path).resolve()
    discovery_file = Path(discovery_cohort_path).resolve()
    confirmation_file = Path(confirmation_cohort_path).resolve()
    policy = _load(policy_file, "confirmation policy")
    selection = _load(selection_file, "candidate selection")
    if policy.get("schema_version") != "pirc20-confirmation-policy-v1":
        raise ConfirmationReadinessError("unsupported confirmation policy")
    if selection.get("schema_version") != "pirc20-candidate-selection-v1":
        raise ConfirmationReadinessError("unsupported candidate selection")
    selection_policy = selection.get("policy")
    candidates = selection.get("candidates")
    if (
        not isinstance(selection_policy, Mapping)
        or selection_policy.get("sha256") != _sha256(policy_file)
        or not isinstance(candidates, list)
        or len(candidates) != selection.get("candidate_count")
    ):
        raise ConfirmationReadinessError("candidate selection is not bound to policy")

    evidence = policy.get("discovery_evidence")
    data_rule = policy.get("confirmation_data")
    execution_rule = policy.get("confirmation_execution")
    decision_rule = policy.get("decision_rule")
    if not all(
        isinstance(item, Mapping)
        for item in (evidence, data_rule, execution_rule, decision_rule)
    ):
        raise ConfirmationReadinessError("confirmation policy is incomplete")

    discovery = load_pirc20_cohort(discovery_file)
    confirmation = load_pirc20_cohort(confirmation_file)
    discovery_hash = _sha256(discovery_file)
    confirmation_hash = _sha256(confirmation_file)
    if (
        discovery.cohort_id != evidence.get("cohort_id")
        or discovery_hash != evidence.get("cohort_sha256")
    ):
        raise ConfirmationReadinessError("discovery cohort differs from frozen evidence")
    if (
        confirmation.cohort_id == discovery.cohort_id
        or confirmation.data_version == discovery.data_version
        or confirmation_hash == discovery_hash
    ):
        raise ConfirmationReadinessError("confirmation cohort identity is not independent")
    if confirmation.final_eval_access != "sealed_identity_only":
        raise ConfirmationReadinessError("confirmation final_eval is not sealed")
    if any(confirmation.sample_counts_by_split.get(split, 0) <= 0 for split in data_rule["required_splits"]):
        raise ConfirmationReadinessError("confirmation cohort is missing a required split")

    discovery_ids = _identifier_sets(discovery)
    confirmation_ids = _identifier_sets(confirmation)
    overlap_counts: dict[str, int] = {}
    for name in data_rule["required_zero_overlap_identifiers"]:
        if name not in discovery_ids:
            raise ConfirmationReadinessError(f"unsupported overlap identifier: {name}")
        count = len(discovery_ids[name] & confirmation_ids[name])
        overlap_counts[name] = count
        if count:
            raise ConfirmationReadinessError(
                f"confirmation cohort overlaps discovery on {name}"
            )

    guardrails = decision_rule.get("guardrails", [])
    calibration_margin = next(
        (
            item.get("noninferiority_margin")
            for item in guardrails
            if isinstance(item, Mapping)
            and item.get("metric") == "hdr90_abs_error_from_90"
        ),
        None,
    )
    decision_ready = isinstance(calibration_margin, (int, float)) and calibration_margin >= 0
    policy_hash = _sha256(policy_file)
    result: dict[str, object] = {
        "schema_version": "pirc20-confirmation-readiness-v1",
        "task_id": "PIRC-20",
        "experiment_id": "NEX326",
        "status": (
            "ready_to_execute" if decision_ready else "waiting_for_decision_threshold"
        ),
        "policy": {"path": policy_file.name, "sha256": policy_hash},
        "candidate_selection": {
            "path": selection_file.name,
            "sha256": _sha256(selection_file),
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "arm_id": int(candidate["arm_id"]),
                    "subconfig_id": str(candidate["subconfig_id"]),
                    "reference_arm_id": int(candidate["reference_arm_id"]),
                    "reference_subconfig_id": str(
                        candidate["reference_subconfig_id"]
                    ),
                }
                for candidate in candidates
            ],
        },
        "discovery_cohort": {
            "cohort_id": discovery.cohort_id,
            "data_version": discovery.data_version,
            "sha256": discovery_hash,
            "sample_count": discovery.sample_count,
        },
        "confirmation_cohort": {
            "cohort_id": confirmation.cohort_id,
            "data_version": confirmation.data_version,
            "sha256": confirmation_hash,
            "sample_count": confirmation.sample_count,
            "sample_counts_by_split": dict(confirmation.sample_counts_by_split),
            "final_eval_access": confirmation.final_eval_access,
        },
        "independence": {
            "verified": True,
            "overlap_counts": overlap_counts,
        },
        "execution": {
            "replicate_seeds": _derive_seeds(
                policy_hash,
                confirmation_hash,
                int(execution_rule["replicate_seed_count"]),
            ),
            "execution_count_per_seed": int(
                execution_rule["required_execution_count_per_seed"]
            ),
            "approved_excluded_arm_ids": list(
                execution_rule["approved_excluded_arm_ids"]
            ),
            "final_eval_unlock": confirmation.cohort_id,
        },
        "decision_readiness": {
            "ready": decision_ready,
            "missing": (
                []
                if decision_ready
                else ["hdr90_abs_error_from_90 noninferiority margin"]
            ),
        },
        "privacy": {
            "contains_sample_ids": False,
            "contains_file_ids": False,
            "contains_coordinates": False,
            "contains_timestamps": False,
        },
    }
    _write_atomic(Path(output_path), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-selection", type=Path, required=True)
    parser.add_argument("--discovery-cohort", type=Path, required=True)
    parser.add_argument("--confirmation-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_confirmation_readiness(
        args.policy,
        args.candidate_selection,
        args.discovery_cohort,
        args.confirmation_cohort,
        args.output,
    )
    print(json.dumps({"status": result["status"], "execution": result["execution"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ConfirmationReadinessError", "build_confirmation_readiness"]
