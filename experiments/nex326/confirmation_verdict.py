"""Issue PIRC-20 confirmation verdicts under the frozen independent-data policy."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from data.pirc20 import load_pirc20_cohort

from .dispersion_calibration import _predictions
from .runner import Prediction, _energy_score, validate_run_record


class ConfirmationVerdictError(ValueError):
    """Confirmation artifacts do not satisfy the frozen policy and readiness proof."""


@dataclass(frozen=True)
class BlockTotals:
    count: int
    energy_sum: float
    coverage_sum: float
    cep50_sum: float


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
        raise ConfirmationVerdictError(f"{label} cannot be read") from exc
    if not isinstance(payload, dict):
        raise ConfirmationVerdictError(f"{label} must contain a JSON object")
    return payload


def _verify_reference(root: Path, reference: Mapping[str, object], label: str) -> Path:
    path = (root / str(reference.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ConfirmationVerdictError(f"{label} escapes its run root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != reference.get("size_bytes")
        or _sha256(path) != reference.get("sha256")
    ):
        raise ConfirmationVerdictError(f"{label} fails integrity verification")
    return path


def _run_record(
    root: Path, arm_id: int, subconfig_id: str
) -> tuple[dict[str, object], Path]:
    manifest = _load(root / "manifest.json", "run manifest")
    if manifest.get("schema_version") != "nex326-scoped-run-manifest-v1":
        raise ConfirmationVerdictError("confirmation run is not a scoped run")
    for relative in manifest.get("run_records", []):
        record_path = (root / str(relative)).resolve()
        try:
            record_path.relative_to(root.resolve())
        except ValueError as exc:
            raise ConfirmationVerdictError("RunRecord escapes its run root") from exc
        record = _load(record_path, "RunRecord")
        if int(record.get("arm_id", -1)) != arm_id or record.get("subconfig_id") != subconfig_id:
            continue
        validate_run_record(record)
        prediction_ref = record["artifacts"]["predictions"]
        return record, _verify_reference(root, prediction_ref, "prediction artifact")
    raise ConfirmationVerdictError(f"run root lacks {arm_id}/{subconfig_id}")


def _mechanism_gates_passed(record: Mapping[str, object]) -> bool:
    gates = record.get("mechanism_gates")
    if not isinstance(gates, list) or not gates:
        raise ConfirmationVerdictError(
            "confirmation RunRecord has no registered mechanism gates"
        )
    if any(not isinstance(gate, Mapping) for gate in gates):
        raise ConfirmationVerdictError(
            "confirmation RunRecord mechanism gates are malformed"
        )
    return all(gate.get("passed") is True for gate in gates)


def block_totals(
    predictions: Sequence[Prediction], segment_to_block: Mapping[str, str]
) -> tuple[dict[str, BlockTotals], set[str]]:
    """Reduce row-level predictions to sufficient statistics per independent block."""

    accumulators: dict[str, list[float]] = {}
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.segment_id in seen or prediction.segment_id not in segment_to_block:
            raise ConfirmationVerdictError("prediction segment identity is duplicate or unknown")
        seen.add(prediction.segment_id)
        block = segment_to_block[prediction.segment_id]
        center = prediction.samples.mean(axis=0)
        radii = np.linalg.norm(prediction.samples - center, axis=1)
        hdr_radius = float(np.quantile(radii, 0.9))
        target_radius = float(np.linalg.norm(prediction.target - center))
        energy = float(_energy_score(prediction.samples, prediction.target))
        values = accumulators.setdefault(block, [0.0, 0.0, 0.0, 0.0])
        values[0] += 1.0
        values[1] += energy
        values[2] += float(target_radius <= hdr_radius)
        values[3] += target_radius
    return (
        {
            block: BlockTotals(int(values[0]), values[1], values[2], values[3])
            for block, values in accumulators.items()
        },
        seen,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""

    if not p_values or any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ConfirmationVerdictError("p-values must be a non-empty unit-interval list")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[index] = running
    return adjusted


def bootstrap_comparison(
    candidate: Mapping[int, Mapping[str, BlockTotals]],
    reference: Mapping[int, Mapping[str, BlockTotals]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap blocks while averaging algorithmic-seed deltas per draw."""

    seeds = sorted(candidate)
    if seeds != sorted(reference) or not seeds or iterations < 100:
        raise ConfirmationVerdictError("bootstrap seed matrices are invalid")
    blocks = sorted(candidate[seeds[0]])
    if not blocks:
        raise ConfirmationVerdictError("bootstrap has no independent blocks")
    if any(sorted(candidate[item]) != blocks or sorted(reference[item]) != blocks for item in seeds):
        raise ConfirmationVerdictError("candidate and reference block matrices differ")

    def matrix(source: Mapping[int, Mapping[str, BlockTotals]], field: str) -> np.ndarray:
        return np.asarray(
            [
                [float(getattr(source[item][block], field)) for block in blocks]
                for item in seeds
            ],
            dtype=float,
        )

    candidate_count = matrix(candidate, "count")
    reference_count = matrix(reference, "count")
    if not np.array_equal(candidate_count, reference_count):
        raise ConfirmationVerdictError("candidate and reference block counts differ")
    candidate_values = {
        "energy_score_d2": matrix(candidate, "energy_sum"),
        "hdr90_coverage": matrix(candidate, "coverage_sum"),
        "cep50_error": matrix(candidate, "cep50_sum"),
    }
    reference_values = {
        "energy_score_d2": matrix(reference, "energy_sum"),
        "hdr90_coverage": matrix(reference, "coverage_sum"),
        "cep50_error": matrix(reference, "cep50_sum"),
    }

    def deltas(weights: np.ndarray) -> np.ndarray:
        totals = candidate_count @ weights
        if np.any(totals <= 0.0):
            raise ConfirmationVerdictError("bootstrap draw contains no samples")
        values: list[float] = []
        for metric in ("energy_score_d2", "hdr90_coverage", "cep50_error"):
            candidate_metric = (candidate_values[metric] @ weights) / totals
            reference_metric = (reference_values[metric] @ weights) / totals
            if metric == "hdr90_coverage":
                delta = np.abs(candidate_metric - 0.9) - np.abs(reference_metric - 0.9)
            else:
                delta = candidate_metric - reference_metric
            values.append(float(np.mean(delta)))
        return np.asarray(values, dtype=float)

    observed = deltas(np.ones(len(blocks), dtype=float))
    rng = np.random.default_rng(seed)
    bootstraps = np.empty((iterations, 3), dtype=float)
    for index in range(iterations):
        weights = np.bincount(
            rng.integers(0, len(blocks), size=len(blocks)), minlength=len(blocks)
        ).astype(float)
        bootstraps[index] = deltas(weights)
    metrics = ("energy_score_d2", "hdr90_abs_error_from_90", "cep50_error")
    result_metrics = {
        metric: {
            "mean_delta": float(observed[offset]),
            "one_sided_95pct_lower": float(np.quantile(bootstraps[:, offset], 0.05)),
            "one_sided_95pct_upper": float(np.quantile(bootstraps[:, offset], 0.95)),
        }
        for offset, metric in enumerate(metrics)
    }
    return {
        "bootstrap_unit": "independent_block_id",
        "independent_block_count": len(blocks),
        "iterations": iterations,
        "metrics": result_metrics,
        "primary_one_sided_p": float(
            (1 + np.count_nonzero(bootstraps[:, 0] >= 0.0)) / (iterations + 1)
        ),
    }


def build_confirmation_verdict(
    policy_path: Path | str,
    readiness_path: Path | str,
    cohort_path: Path | str,
    run_roots: Sequence[Path | str],
    output_path: Path | str,
) -> dict[str, object]:
    """Verify confirmation runs, bootstrap frozen candidates, and issue verdicts."""

    policy_file = Path(policy_path).resolve()
    readiness_file = Path(readiness_path).resolve()
    cohort_file = Path(cohort_path).resolve()
    policy = _load(policy_file, "confirmation policy")
    readiness = _load(readiness_file, "confirmation readiness")
    if (
        policy.get("schema_version") != "pirc20-confirmation-policy-v1"
        or policy.get("status") != "frozen_for_future_independent_confirmation"
        or readiness.get("schema_version") != "pirc20-confirmation-readiness-v1"
        or readiness.get("status") != "ready_to_execute"
        or readiness.get("policy", {}).get("sha256") != _sha256(policy_file)
        or readiness.get("confirmation_cohort", {}).get("sha256") != _sha256(cohort_file)
        or not readiness.get("independence", {}).get("verified")
        or any(readiness.get("independence", {}).get("overlap_counts", {}).values())
    ):
        raise ConfirmationVerdictError("policy, readiness, and cohort are not executable")
    decision = policy["decision_rule"]
    if decision.get("status") != "frozen":
        raise ConfirmationVerdictError("confirmation decision rule is not frozen")
    margin = next(
        item["noninferiority_margin"]
        for item in decision["guardrails"]
        if item["metric"] == "hdr90_abs_error_from_90"
    )

    cohort = load_pirc20_cohort(cohort_file)
    segment_to_block = {
        sample.segment_id: sample.independent_block_id
        for sample in cohort.iter_samples("final_eval")
    }
    registered_seeds = [int(seed) for seed in readiness["execution"]["replicate_seeds"]]
    roots = tuple(Path(root).resolve() for root in run_roots)
    if len(roots) != len(registered_seeds):
        raise ConfirmationVerdictError("one run root is required per registered seed")
    candidates = readiness["candidate_selection"]["candidates"]
    configs = sorted(
        {
            (int(item["arm_id"]), str(item["subconfig_id"]))
            for item in candidates
        }
        | {
            (int(item["reference_arm_id"]), str(item["reference_subconfig_id"]))
            for item in candidates
        }
    )
    summaries: dict[tuple[int, int, str], dict[str, BlockTotals]] = {}
    mechanisms: dict[tuple[int, int, str], bool] = {}
    prediction_set: set[str] | None = None
    observed_seeds: set[int] = set()
    inputs: list[dict[str, object]] = []
    scope_hash = policy["confirmation_execution"]["scope_policy_sha256"]
    for root in roots:
        manifest = _load(root / "manifest.json", "run manifest")
        manifest_scope = manifest.get("scope")
        if (
            not isinstance(manifest_scope, Mapping)
            or manifest_scope.get("policy_sha256") != scope_hash
            or manifest.get("record_count") != 28
            or manifest.get("run_status") != {"succeeded": 28}
        ):
            raise ConfirmationVerdictError("confirmation run scope is invalid")
        root_seed = int(manifest["replicate_seed"])
        if root_seed in observed_seeds:
            raise ConfirmationVerdictError("confirmation run seeds are duplicated")
        observed_seeds.add(root_seed)
        for arm_id, subconfig_id in configs:
            record, prediction_file = _run_record(root, arm_id, subconfig_id)
            if (
                int(record["replicate_seed"]) != root_seed
                or record["dataset"]["dataset_id"] != cohort.cohort_id
                or record["run_status"] != "succeeded"
            ):
                raise ConfirmationVerdictError("confirmation RunRecord identity differs")
            predictions = _predictions(_load(prediction_file, "predictions"))
            totals, seen = block_totals(predictions, segment_to_block)
            if prediction_set is None:
                prediction_set = seen
            elif seen != prediction_set:
                raise ConfirmationVerdictError("confirmation prediction segment sets differ")
            summaries[(root_seed, arm_id, subconfig_id)] = totals
            mechanisms[(root_seed, arm_id, subconfig_id)] = _mechanism_gates_passed(
                record
            )
            inputs.append(
                {
                    "replicate_seed": root_seed,
                    "arm_id": arm_id,
                    "subconfig_id": subconfig_id,
                    "run_record_sha256": _sha256(root / f"{record['run_id']}/run_record.json"),
                    "predictions_sha256": _sha256(prediction_file),
                }
            )
    if observed_seeds != set(registered_seeds):
        raise ConfirmationVerdictError("confirmation run seeds differ from readiness")

    iterations = int(policy["confirmation_execution"]["bootstrap_replicates"])
    bootstrap_seed = int(
        hashlib.sha256(
            f"{_sha256(policy_file)}:{_sha256(readiness_file)}".encode("ascii")
        ).hexdigest()[:8],
        16,
    )
    comparisons: list[dict[str, object]] = []
    for offset, candidate in enumerate(candidates):
        arm_id = int(candidate["arm_id"])
        subconfig_id = str(candidate["subconfig_id"])
        reference_arm = int(candidate["reference_arm_id"])
        reference_subconfig = str(candidate["reference_subconfig_id"])
        comparison = bootstrap_comparison(
            {
                seed: summaries[(seed, arm_id, subconfig_id)]
                for seed in registered_seeds
            },
            {
                seed: summaries[(seed, reference_arm, reference_subconfig)]
                for seed in registered_seeds
            },
            iterations=iterations,
            seed=(bootstrap_seed + offset) % (2**32),
        )
        comparisons.append(
            {
                "arm_id": arm_id,
                "subconfig_id": subconfig_id,
                "reference_arm_id": reference_arm,
                "reference_subconfig_id": reference_subconfig,
                "mechanism_gates_passed": all(
                    mechanisms[(seed, arm_id, subconfig_id)]
                    and mechanisms[(seed, reference_arm, reference_subconfig)]
                    for seed in registered_seeds
                ),
                **comparison,
            }
        )
    adjusted = holm_adjust(
        [float(item["primary_one_sided_p"]) for item in comparisons]
    )
    for item, adjusted_p in zip(comparisons, adjusted):
        metrics = item["metrics"]
        primary_passed = (
            metrics["energy_score_d2"]["mean_delta"] < 0.0
            and adjusted_p <= float(decision["familywise_alpha"])
        )
        calibration_passed = (
            metrics["hdr90_abs_error_from_90"]["mean_delta"] <= margin
            and metrics["hdr90_abs_error_from_90"]["one_sided_95pct_upper"]
            <= margin
        )
        cep50_passed = (
            metrics["cep50_error"]["mean_delta"] <= 0.0
            and metrics["cep50_error"]["one_sided_95pct_upper"] <= 0.0
        )
        item["holm_adjusted_primary_p"] = adjusted_p
        item["gates"] = {
            "primary_energy": primary_passed,
            "hdr90_absolute_error": calibration_passed,
            "cep50_error": cep50_passed,
            "mechanism": item["mechanism_gates_passed"],
        }
        item["verdict"] = (
            "retain"
            if all(item["gates"].values())
            else "inconclusive"
        )

    result: dict[str, object] = {
        "schema_version": "pirc20-confirmation-verdict-v1",
        "task_id": "PIRC-20",
        "experiment_id": "NEX326",
        "scientific_status": "independent_confirmation_assessed",
        "policy_sha256": _sha256(policy_file),
        "readiness_sha256": _sha256(readiness_file),
        "cohort_sha256": _sha256(cohort_file),
        "replicate_seeds": registered_seeds,
        "bootstrap_seed": bootstrap_seed,
        "comparisons": comparisons,
        "verdict_counts": dict(Counter(str(item["verdict"]) for item in comparisons)),
        "integrity": {
            "implementation_sha256": _sha256(Path(__file__)),
            "inputs": sorted(
                inputs,
                key=lambda row: (
                    int(row["replicate_seed"]),
                    int(row["arm_id"]),
                    str(row["subconfig_id"]),
                ),
            ),
        },
        "privacy": {
            "contains_sample_ids": False,
            "contains_file_ids": False,
            "contains_coordinates": False,
            "contains_timestamps": False,
            "contains_row_level_predictions": False,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ConfirmationVerdictError(f"stale staging file exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_confirmation_verdict(
        args.policy, args.readiness, args.cohort, args.run_roots, args.output
    )
    print(json.dumps({"verdict_counts": result["verdict_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BlockTotals",
    "ConfirmationVerdictError",
    "block_totals",
    "bootstrap_comparison",
    "build_confirmation_verdict",
    "holm_adjust",
]
