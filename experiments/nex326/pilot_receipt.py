"""Build a portable, hash-bound receipt for an NEX326 DSDE pilot run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


class PilotReceiptError(ValueError):
    """Pilot inputs disagree or do not describe a complete NEX326 execution."""


def _load_mapping(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PilotReceiptError(f"{label} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_set_sha256(entries: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps(list(entries), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_pilot_receipt(
    cohort_path: Path | str,
    records_root: Path | str,
    aggregate_summary_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Cross-check cohort, RunRecords, and TSDE output and write one small receipt."""
    cohort_file = Path(cohort_path)
    records_dir = Path(records_root)
    manifest_file = records_dir / "manifest.json"
    summary_file = Path(aggregate_summary_path)
    cohort = _load_mapping(cohort_file, "cohort")
    manifest = _load_mapping(manifest_file, "run manifest")
    summary = _load_mapping(summary_file, "aggregate summary")

    if cohort.get("schema_version") != "nex326-cohort-v1":
        raise PilotReceiptError("incompatible cohort schema")
    if manifest.get("experiment_id") != "NEX326" or manifest.get("record_count") != 36:
        raise PilotReceiptError("manifest must describe all 36 NEX326 executions")
    if summary.get("spec_version") != manifest.get("spec_version"):
        raise PilotReceiptError("manifest and aggregate spec versions differ")
    if summary.get("arm_count") != 22 or summary.get("execution_count") != 36:
        raise PilotReceiptError("aggregate must describe 22 arms and 36 executions")
    comparison = summary.get("comparison")
    if (
        not isinstance(comparison, Mapping)
        or comparison.get("scientific_status") != "exploratory_only"
        or comparison.get("assessment") != "not_assessed"
    ):
        raise PilotReceiptError("aggregate comparison must remain exploratory and not assessed")
    aggregate_artifacts = summary.get("artifacts")
    if not isinstance(aggregate_artifacts, Mapping) or not aggregate_artifacts:
        raise PilotReceiptError("aggregate summary lacks hash-bound artifacts")
    resolved_aggregate_root = summary_file.parent.resolve()
    for name, reference in aggregate_artifacts.items():
        if not isinstance(reference, Mapping):
            raise PilotReceiptError(f"aggregate artifact {name} has an invalid reference")
        relative = reference.get("path")
        if not isinstance(relative, str) or not relative:
            raise PilotReceiptError(f"aggregate artifact {name} has no path")
        artifact_path = (summary_file.parent / relative).resolve()
        try:
            artifact_path.relative_to(resolved_aggregate_root)
        except ValueError as exc:
            raise PilotReceiptError(f"aggregate artifact {name} escapes output root") from exc
        if not artifact_path.is_file():
            raise PilotReceiptError(f"aggregate artifact {name} is missing")
        if artifact_path.stat().st_size != reference.get("size_bytes"):
            raise PilotReceiptError(f"aggregate artifact {name} size mismatch")
        if _sha256(artifact_path) != reference.get("sha256"):
            raise PilotReceiptError(f"aggregate artifact {name} SHA-256 mismatch")

    record_entries: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    listed_records = manifest.get("run_records")
    if not isinstance(listed_records, list) or len(listed_records) != 36:
        raise PilotReceiptError("manifest must list exactly 36 RunRecords")
    resolved_root = records_dir.resolve()
    for relative in listed_records:
        if not isinstance(relative, str):
            raise PilotReceiptError("manifest RunRecord paths must be strings")
        record_path = (records_dir / relative).resolve()
        try:
            record_path.relative_to(resolved_root)
        except ValueError as exc:
            raise PilotReceiptError("manifest RunRecord path escapes records root") from exc
        if not record_path.is_file():
            raise PilotReceiptError(f"missing RunRecord: {relative}")
        records.append(_load_mapping(record_path, "RunRecord"))
        record_entries.append({"path": relative, "sha256": _sha256(record_path)})

    identities = {
        (record.get("experiment_id"), record.get("spec_version")) for record in records
    }
    if identities != {("NEX326", manifest.get("spec_version"))}:
        raise PilotReceiptError("RunRecord experiment identities disagree with manifest")
    dataset_ids = {record.get("dataset", {}).get("dataset_id") for record in records}
    if dataset_ids != {cohort.get("dataset_id")}:
        raise PilotReceiptError("RunRecord dataset identity disagrees with cohort")
    requested_samples = {
        record.get("runtime", {}).get("requested_prediction_samples") for record in records
    }
    if requested_samples != {manifest.get("requested_prediction_samples")}:
        raise PilotReceiptError("RunRecord sample counts disagree with manifest")
    protocol_seeds = {record.get("seed") for record in records}
    replicate_seeds = {record.get("replicate_seed") for record in records}
    if protocol_seeds != {manifest.get("protocol_seed")} or protocol_seeds != {
        summary.get("protocol_seed")
    }:
        raise PilotReceiptError("RunRecord, manifest, and aggregate protocol seeds differ")
    if replicate_seeds != {manifest.get("replicate_seed")} or replicate_seeds != {
        summary.get("replicate_seed")
    }:
        raise PilotReceiptError("RunRecord, manifest, and aggregate replicate seeds differ")
    manifest_implementation = manifest.get("implementation")
    manifest_source_bundle = (
        manifest_implementation.get("source_bundle_sha256")
        if isinstance(manifest_implementation, Mapping)
        else None
    )
    manifest_execution_identity = (
        manifest_implementation.get("execution_identity_sha256")
        if isinstance(manifest_implementation, Mapping)
        else None
    )
    record_source_bundles = {
        record.get("implementation", {}).get("source_bundle_sha256")
        for record in records
        if isinstance(record.get("implementation"), Mapping)
    }
    record_execution_identities = {
        record.get("implementation", {}).get("execution_identity_sha256")
        for record in records
        if isinstance(record.get("implementation"), Mapping)
    }
    if manifest_implementation is not None and (
        not isinstance(manifest_source_bundle, str)
        or not isinstance(manifest_execution_identity, str)
        or record_source_bundles != {manifest_source_bundle}
        or record_execution_identities != {manifest_execution_identity}
        or summary.get("implementation_source_bundle_sha256") != manifest_source_bundle
        or summary.get("implementation_execution_identity_sha256")
        != manifest_execution_identity
    ):
        raise PilotReceiptError("RunRecord, manifest, and aggregate implementations differ")

    run_status = dict(Counter(str(record.get("run_status")) for record in records))
    verdict = dict(Counter(str(record.get("verdict")) for record in records))
    if run_status != manifest.get("run_status") or run_status != summary.get("run_status"):
        raise PilotReceiptError("RunRecord, manifest, and aggregate status counts differ")
    if verdict != summary.get("verdict"):
        raise PilotReceiptError("RunRecord and aggregate verdict counts differ")

    unavailable = [
        {
            "arm_id": record["arm_id"],
            "subconfig_id": record["subconfig_id"],
            "stage": record["failure"]["stage"],
            "reason": record["failure"]["reason"],
        }
        for record in records
        if record.get("run_status") == "data_unavailable"
    ]
    unavailable.sort(key=lambda item: (int(item["arm_id"]), str(item["subconfig_id"])))
    receipt: dict[str, object] = {
        "schema_version": "nex326-dsde-pilot-receipt-v1",
        "experiment_id": "NEX326",
        "spec_version": manifest["spec_version"],
        "scientific_status": "pilot_not_final_scientific_evidence",
        "implementation": manifest_implementation,
        "dataset": {
            "dataset_id": cohort["dataset_id"],
            "data_version": cohort["data_version"],
            "purpose": cohort["purpose"],
            "source": cohort["source"],
            "selection": cohort["selection"],
        },
        "execution": {
            "arm_count": 22,
            "execution_count": 36,
            "requested_prediction_samples": manifest["requested_prediction_samples"],
            "protocol_seed": manifest["protocol_seed"],
            "replicate_seed": manifest["replicate_seed"],
            "effective_prediction_samples": summary["effective_prediction_samples"],
            "run_status": run_status,
            "verdict": verdict,
            "mechanism_status": summary["mechanism_status"],
            "comparison": comparison,
            "data_unavailable": unavailable,
        },
        "integrity": {
            "cohort_sha256": _sha256(cohort_file),
            "run_manifest_sha256": _sha256(manifest_file),
            "run_record_set_sha256": _record_set_sha256(record_entries),
            "tsde_summary_sha256": _sha256(summary_file),
            "tsde_artifacts": aggregate_artifacts,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_pilot_receipt(
        args.cohort, args.records, args.aggregate_summary, args.output
    )
    print(json.dumps(receipt["execution"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PilotReceiptError", "build_pilot_receipt"]
