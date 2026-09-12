"""Audit PIRC-19/NEX326 reconstruction completion from tracked evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .fidelity import build_fidelity_report
from .specification import load_experiment_spec


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "pirc19-nex326-completion-audit-v1"
DEFAULT_FIDELITY = ROOT / "implementation_fidelity_report.json"
DEFAULT_GAPS = ROOT / "implementation_gaps.json"
DEFAULT_PILOT = ROOT / "dsde_20pct_pilot_receipt.json"
DEFAULT_MULTI_SEED = ROOT / "dsde_20pct_multi_seed_receipt.json"
EXPECTED_UNAVAILABLE = {
    (13, "animal_pretrain"),
    (17, "weather"),
    (22, "doob"),
    (22, "sb"),
    (22, "soft_endpoint"),
}


class CompletionAuditError(ValueError):
    """Tracked completion evidence is missing or internally inconsistent."""


def _load(path: Path | str) -> dict[str, object]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompletionAuditError(f"cannot read completion evidence: {source}") from error
    if not isinstance(payload, dict):
        raise CompletionAuditError(f"completion evidence must be an object: {source}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    payload = _load(path)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _current_sources_match(implementation: object) -> bool:
    if not isinstance(implementation, Mapping):
        return False
    sources = implementation.get("files")
    if not isinstance(sources, list) or not sources:
        return False
    for source in sources:
        if not isinstance(source, Mapping):
            return False
        relative = source.get("path")
        expected = source.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            return False
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or _sha256(path) != expected:
            return False
    return True


def _fraction(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": numerator / denominator if denominator else 0.0,
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.name


def build_completion_report(
    *,
    fidelity_path: Path | str = DEFAULT_FIDELITY,
    gaps_path: Path | str = DEFAULT_GAPS,
    pilot_path: Path | str = DEFAULT_PILOT,
    multi_seed_path: Path | str = DEFAULT_MULTI_SEED,
) -> dict[str, object]:
    """Build a dimensioned completion report without inventing one overall score."""
    spec = load_experiment_spec()
    fidelity = _load(fidelity_path)
    gaps = _load(gaps_path)
    pilot = _load(pilot_path)
    multi_seed = _load(multi_seed_path)

    execution = pilot.get("execution", {})
    run_status = execution.get("run_status", {}) if isinstance(execution, Mapping) else {}
    verdict = execution.get("verdict", {}) if isinstance(execution, Mapping) else {}
    comparison = execution.get("comparison", {}) if isinstance(execution, Mapping) else {}
    uncertainty = comparison.get("uncertainty_status", {}) if isinstance(comparison, Mapping) else {}
    unavailable_rows = (
        execution.get("data_unavailable", []) if isinstance(execution, Mapping) else []
    )
    unavailable = {
        (int(item["arm_id"]), str(item["subconfig_id"]))
        for item in unavailable_rows
        if isinstance(item, Mapping)
        and "arm_id" in item
        and "subconfig_id" in item
    }
    succeeded = int(run_status.get("succeeded", 0)) if isinstance(run_status, Mapping) else 0
    unavailable_count = (
        int(run_status.get("data_unavailable", 0))
        if isinstance(run_status, Mapping)
        else 0
    )
    replicate_count = int(multi_seed.get("replicate_count", 0))
    total_replicate_executions = int(multi_seed.get("total_execution_count", 0))
    replicate_status = multi_seed.get("replicate_status", {})
    replicate_succeeded = (
        int(replicate_status.get("succeeded", 0)) * replicate_count
        if isinstance(replicate_status, Mapping)
        else 0
    )
    assessed = (
        sum(
            int(count)
            for status, count in verdict.items()
            if status not in {"not_assessed", "unavailable"}
        )
        if isinstance(verdict, Mapping)
        else 0
    )
    gap_items = gaps.get("items", [])
    arm22_gap = next(
        (
            item
            for item in gap_items
            if isinstance(item, Mapping) and item.get("arm") == 22
        ),
        {},
    )

    checks = [
        _check(
            "frozen_contract",
            len(spec.arms) == 22 and len(spec.executions) == 36,
            f"{len(spec.arms)} numbered arms and {len(spec.executions)} execution slots",
        ),
        _check(
            "implementation_routing",
            fidelity == build_fidelity_report() and fidelity.get("failed_route_count") == 0,
            "tracked fidelity report reproduces from the frozen spec with zero route failures",
        ),
        _check(
            "implementation_gap_policy",
            gaps.get("closure_rule")
            == "No implementation_missing, mock estimator, constant gate, or silent fallback is allowed."
            and all(
                "implementation_missing"
                not in str(item.get("implementation_status", ""))
                for item in gap_items
                if isinstance(item, Mapping)
            ),
            "tracked gaps retain the no-mock/no-silent-fallback closure rule",
        ),
        _check(
            "current_source_identity",
            _current_sources_match(pilot.get("implementation")),
            "pilot receipt implementation hashes match the current execution sources",
        ),
        _check(
            "receipt_identity",
            pilot.get("implementation") == multi_seed.get("implementation"),
            "single-seed and multi-seed receipts bind the same implementation",
        ),
        _check(
            "current_dsde_matrix",
            succeeded == 31
            and unavailable_count == 5
            and succeeded + unavailable_count == len(spec.executions),
            f"{succeeded} succeeded and {unavailable_count} data-unavailable executions",
        ),
        _check(
            "replicate_matrix",
            replicate_count == 3
            and total_replicate_executions == 108
            and replicate_succeeded == 93,
            f"{replicate_succeeded} succeeded of {total_replicate_executions} across {replicate_count} seeds",
        ),
        _check(
            "declared_unavailable_scope",
            unavailable == EXPECTED_UNAVAILABLE,
            "only animal, weather, and three expert-prior extension executions are unavailable",
        ),
        _check(
            "terrain_evidence",
            isinstance(comparison, Mapping)
            and comparison.get("status", {}).get("exploratory_point_estimate") == 23
            and isinstance(uncertainty, Mapping)
            and uncertainty.get("complete") == 22,
            "Arm 17 terrain has a coverage-matched point comparison and paired bootstrap",
        ),
        _check(
            "arm22_extension_scope",
            isinstance(arm22_gap, Mapping)
            and str(arm22_gap.get("fidelity_status", "")).endswith("_extension"),
            "Arm 22 remains implemented but excluded from current core evidence",
        ),
        _check(
            "scientific_claim_boundary",
            assessed == 0
            and multi_seed.get("assessment") == "not_assessed"
            and multi_seed.get("scientific_status")
            == "exploratory_only_not_final_scientific_evidence",
            "no pilot execution has been promoted to a scientific verdict",
        ),
    ]
    checks_passed = sum(bool(item["passed"]) for item in checks)
    all_checks_passed = checks_passed == len(checks)
    evidence_paths = [
        Path(fidelity_path),
        Path(gaps_path),
        Path(pilot_path),
        Path(multi_seed_path),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "PIRC-19",
        "experiment_id": spec.experiment_id,
        "spec_version": spec.spec_version,
        "overall_status": (
            "engineering_reconstruction_complete_scientific_reproduction_incomplete"
            if all_checks_passed
            else "completion_audit_failed"
        ),
        "paper_equivalent": False,
        "dimensions": {
            "frozen_contract_implementation": {
                "status": "complete" if all_checks_passed else "audit_failed",
                **_fraction(len(spec.executions), 36),
            },
            "current_dsde_execution": {
                "status": "complete_for_available_inputs",
                **_fraction(succeeded, len(spec.executions)),
            },
            "replicated_current_dsde_execution": {
                "status": "complete_for_available_inputs",
                **_fraction(replicate_succeeded, total_replicate_executions),
            },
            "scientifically_assessed_succeeded_executions": {
                "status": "pending",
                **_fraction(assessed, succeeded),
            },
        },
        "checks": checks,
        "check_summary": {
            "passed": checks_passed,
            "total": len(checks),
            "all_passed": all_checks_passed,
        },
        "current_data_blockers": [
            {
                "arm_id": 13,
                "subconfig_id": "animal_pretrain",
                "required_input": "licensed versioned animal pretraining cohort",
            },
            {
                "arm_id": 17,
                "subconfig_id": "weather",
                "required_input": "time-aligned temperature and precipitation",
            },
        ],
        "deferred_extensions": [
            {
                "arm_id": 22,
                "subconfig_ids": ["doob", "sb", "soft_endpoint"],
                "required_input": "independently attested expert endpoint-prior feed",
            }
        ],
        "scientific_blockers": [
            "the tracked DSDE cohort is a deterministic 20% pilot, not the full registered cohort",
            "three replicate seeds over one cohort are descriptive, not independent scientific replications",
            "all succeeded executions remain not_assessed",
        ],
        "evidence": [
            {
                "path": _display_path(path),
                "canonical_json_sha256": _canonical_json_sha256(path),
            }
            for path in evidence_paths
        ],
        "next_core_step_requires_external_input": True,
    }


def write_completion_report(
    destination: Path | str,
    **kwargs: object,
) -> dict[str, object]:
    report = build_completion_report(**kwargs)
    if not report["check_summary"]["all_passed"]:
        failed = [
            item["name"] for item in report["checks"] if not item["passed"]
        ]
        raise CompletionAuditError(f"completion audit failed: {failed}")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = write_completion_report(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompletionAuditError",
    "build_completion_report",
    "main",
    "write_completion_report",
]
