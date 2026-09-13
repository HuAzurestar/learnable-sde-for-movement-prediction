"""Freeze exploratory PIRC-20 signals as candidates for a future independent test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence


class CandidateSelectionError(ValueError):
    """Discovery artifacts or the frozen candidate rule are inconsistent."""


METRICS = (
    "energy_score_d2",
    "hdr90_abs_error_from_90",
    "cep50_error",
)


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
        raise CandidateSelectionError(f"{label} cannot be read") from exc
    if not isinstance(payload, dict):
        raise CandidateSelectionError(f"{label} must contain a JSON object")
    return payload


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise CandidateSelectionError(f"stale staging file exists: {temporary}")
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


def build_candidate_selection(
    policy_path: Path | str,
    receipt_path: Path | str,
    replicate_summary_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Apply the frozen discovery rule and emit a hash-bound candidate receipt."""

    policy_file = Path(policy_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    summary_file = Path(replicate_summary_path).resolve()
    policy = _load(policy_file, "confirmation policy")
    receipt = _load(receipt_file, "multi-seed receipt")
    summary = _load(summary_file, "replicate summary")
    if policy.get("schema_version") != "pirc20-confirmation-policy-v1":
        raise CandidateSelectionError("unsupported confirmation policy")
    if receipt.get("schema_version") != "nex326-dsde-multi-seed-receipt-v1":
        raise CandidateSelectionError("unsupported multi-seed receipt")
    if summary.get("schema_version") != "nex326-tsde-replicate-aggregate-v1":
        raise CandidateSelectionError("unsupported replicate summary")

    evidence = policy.get("discovery_evidence")
    rule = policy.get("candidate_selection")
    artifacts = summary.get("artifacts")
    if not isinstance(evidence, Mapping) or not isinstance(rule, Mapping):
        raise CandidateSelectionError("policy discovery rule is incomplete")
    if not isinstance(artifacts, Mapping):
        raise CandidateSelectionError("replicate summary artifacts are missing")
    table_ref = artifacts.get("nex326_cross_replicate_comparisons.csv")
    if not isinstance(table_ref, Mapping):
        raise CandidateSelectionError("cross-replicate table reference is missing")
    table_file = (summary_file.parent / str(table_ref.get("path", ""))).resolve()
    try:
        table_file.relative_to(summary_file.parent.resolve())
    except ValueError as exc:
        raise CandidateSelectionError("cross-replicate table escapes summary root") from exc
    if not table_file.is_file():
        raise CandidateSelectionError("cross-replicate table is missing")

    receipt_integrity = receipt.get("integrity")
    receipt_cohort = receipt.get("cohort")
    if (
        evidence.get("use") != "candidate_selection_only_not_confirmation"
        or evidence.get("multi_seed_receipt_sha256") != _sha256(receipt_file)
        or evidence.get("replicate_aggregate_summary_sha256") != _sha256(summary_file)
        or evidence.get("cross_replicate_table_sha256") != _sha256(table_file)
        or table_ref.get("sha256") != _sha256(table_file)
        or not isinstance(receipt_integrity, Mapping)
        or receipt_integrity.get("replicate_aggregate_summary_sha256")
        != _sha256(summary_file)
        or not isinstance(receipt_cohort, Mapping)
        or receipt_cohort.get("sha256") != evidence.get("cohort_sha256")
        or receipt.get("scientific_status")
        != "exploratory_only_not_final_scientific_evidence"
        or receipt.get("assessment") != "not_assessed"
    ):
        raise CandidateSelectionError("discovery evidence hash chain differs from policy")

    expected_replicates = int(rule.get("required_replicate_count", 0))
    required_metrics = set(rule.get("required_improvement_metrics", []))
    if not required_metrics or not required_metrics.issubset(METRICS):
        raise CandidateSelectionError("policy improvement metrics are invalid")
    seeds = [int(seed) for seed in receipt.get("replicate_seeds", [])]
    selected: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    with table_file.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = (int(row["arm_id"]), row["subconfig_id"])
            if key in seen:
                raise CandidateSelectionError("cross-replicate table has duplicate rows")
            seen.add(key)
            if row["comparison_status"] != rule["required_comparison_status"]:
                continue
            if (
                row["replicate_status"] != rule["required_run_status"]
                or row["assessment"] != rule["required_assessment"]
                or int(row["replicate_count"]) != expected_replicates
                or [int(seed) for seed in row["replicate_seeds"].split(";")] != seeds
            ):
                raise CandidateSelectionError("candidate row identity is inconsistent")
            metrics: dict[str, object] = {}
            qualifies = True
            for metric in METRICS:
                mean = float(row[f"delta_{metric}_mean"])
                minimum = float(row[f"delta_{metric}_min"])
                maximum = float(row[f"delta_{metric}_max"])
                sign = row[f"delta_{metric}_sign_consistency"]
                if not all(math.isfinite(value) for value in (mean, minimum, maximum)):
                    raise CandidateSelectionError("candidate row contains a non-finite metric")
                if metric in required_metrics:
                    qualifies = (
                        qualifies
                        and mean < 0.0
                        and maximum < 0.0
                        and sign == "all_negative"
                    )
                metrics[metric] = {
                    "mean_delta": mean,
                    "min_delta": minimum,
                    "max_delta": maximum,
                    "sign_consistency": sign,
                }
            if qualifies:
                selected.append(
                    {
                        "arm_id": key[0],
                        "subconfig_id": key[1],
                        "slot": row["slot"],
                        "reference_arm_id": int(row["reference_arm_id"]),
                        "reference_subconfig_id": row["reference_subconfig_id"],
                        "discovery_metrics": metrics,
                    }
                )
    selected.sort(key=lambda item: (int(item["arm_id"]), str(item["subconfig_id"])))
    if len(selected) != int(rule.get("expected_candidate_count", -1)):
        raise CandidateSelectionError("frozen candidate count does not match discovery data")

    result: dict[str, object] = {
        "schema_version": "pirc20-candidate-selection-v1",
        "task_id": "PIRC-20",
        "experiment_id": "NEX326",
        "status": "frozen_candidates_for_future_independent_confirmation",
        "scientific_status": "exploratory_candidate_selection_not_a_verdict",
        "policy": {
            "path": policy_file.name,
            "sha256": _sha256(policy_file),
        },
        "discovery": {
            "cohort_id": evidence["cohort_id"],
            "cohort_sha256": evidence["cohort_sha256"],
            "replicate_seeds": seeds,
            "multi_seed_receipt_sha256": _sha256(receipt_file),
            "replicate_summary_sha256": _sha256(summary_file),
            "cross_replicate_table_sha256": _sha256(table_file),
        },
        "selection_rule": dict(rule),
        "candidate_count": len(selected),
        "candidates": selected,
        "limitations": [
            "Discovery outcomes selected these candidates and cannot confirm them.",
            "Only a zero-overlap future cohort evaluated under the frozen policy can issue a verdict.",
        ],
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
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--replicate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_candidate_selection(
        args.policy, args.receipt, args.replicate_summary, args.output
    )
    print(json.dumps({"candidates": result["candidates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CandidateSelectionError", "build_candidate_selection"]
