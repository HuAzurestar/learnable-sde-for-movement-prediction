"""Build a compact receipt for a hash-verified NEX326 multi-seed pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


class MultiSeedReceiptError(ValueError):
    """Multi-seed artifacts are incomplete, inconsistent, or tampered."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MultiSeedReceiptError(f"{label} must contain a JSON object")
    return payload


def _inside(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise MultiSeedReceiptError(f"{label} has no path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise MultiSeedReceiptError(f"{label} escapes its declared root") from exc
    if not path.is_file():
        raise MultiSeedReceiptError(f"{label} is missing")
    return path


def _verify_reference(root: Path, reference: Mapping[str, object], label: str) -> Path:
    path = _inside(root, reference.get("path"), label)
    if path.stat().st_size != reference.get("size_bytes"):
        raise MultiSeedReceiptError(f"{label} size mismatch")
    if _sha256(path) != reference.get("sha256"):
        raise MultiSeedReceiptError(f"{label} SHA-256 mismatch")
    return path


def build_multi_seed_receipt(
    batch_manifest_path: Path | str,
    replicate_summary_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    batch_file = Path(batch_manifest_path).resolve()
    batch_root = batch_file.parent
    summary_file = Path(replicate_summary_path).resolve()
    try:
        summary_file.relative_to(batch_root)
    except ValueError as exc:
        raise MultiSeedReceiptError("replicate summary escapes the batch root") from exc
    batch = _load(batch_file, "multi-seed manifest")
    summary = _load(summary_file, "replicate aggregate summary")
    if batch.get("schema_version") != "nex326-multi-seed-manifest-v1":
        raise MultiSeedReceiptError("incompatible multi-seed manifest")
    if summary.get("schema_version") != "nex326-tsde-replicate-aggregate-v1":
        raise MultiSeedReceiptError("incompatible replicate aggregate summary")
    seeds = tuple(int(seed) for seed in batch.get("replicate_seeds", []))
    if (
        len(seeds) < 2
        or len(seeds) != len(set(seeds))
        or summary.get("replicate_seeds") != list(seeds)
        or summary.get("replicate_count") != len(seeds)
    ):
        raise MultiSeedReceiptError("batch and aggregate replicate seeds differ")
    if (
        summary.get("batch_manifest_sha256") != _sha256(batch_file)
        or summary.get("spec_version") != batch.get("spec_version")
        or summary.get("protocol_seed") != batch.get("protocol_seed")
        or summary.get("dataset_fingerprint") != batch.get("cohort", {}).get("fingerprint")
        or summary.get("scientific_status") != "exploratory_only"
        or summary.get("assessment") != "not_assessed"
    ):
        raise MultiSeedReceiptError("batch and replicate aggregate identities differ")
    batch_implementation = batch.get("implementation")
    batch_source_bundle = (
        batch_implementation.get("source_bundle_sha256")
        if isinstance(batch_implementation, Mapping)
        else None
    )
    batch_execution_identity = (
        batch_implementation.get("execution_identity_sha256")
        if isinstance(batch_implementation, Mapping)
        else None
    )
    if batch_implementation is not None and (
        not isinstance(batch_source_bundle, str)
        or len(batch_source_bundle) != 64
        or not isinstance(batch_execution_identity, str)
        or len(batch_execution_identity) != 64
        or summary.get("implementation_source_bundle_sha256") != batch_source_bundle
        or summary.get("implementation_execution_identity_sha256")
        != batch_execution_identity
    ):
        raise MultiSeedReceiptError("implementation source bundle identity differs")

    replicate_entries = batch.get("replicates")
    if not isinstance(replicate_entries, list) or len(replicate_entries) != len(seeds):
        raise MultiSeedReceiptError("batch replicate entries are incomplete")
    per_seed_manifests: list[dict[str, object]] = []
    for entry in replicate_entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("manifest"), Mapping):
            raise MultiSeedReceiptError("batch contains an invalid replicate entry")
        manifest_path = _verify_reference(batch_root, entry["manifest"], "per-seed manifest")
        manifest_payload = _load(manifest_path, "per-seed manifest")
        manifest_implementation = manifest_payload.get("implementation")
        if batch_source_bundle is not None and (
            not isinstance(manifest_implementation, Mapping)
            or manifest_implementation.get("source_bundle_sha256") != batch_source_bundle
            or manifest_implementation.get("execution_identity_sha256")
            != batch_execution_identity
        ):
            raise MultiSeedReceiptError("per-seed implementation source bundle differs")
        per_seed_manifests.append(
            {
                "replicate_seed": entry["replicate_seed"],
                "path": manifest_path.relative_to(batch_root).as_posix(),
                "sha256": entry["manifest"]["sha256"],
                "size_bytes": entry["manifest"]["size_bytes"],
            }
        )

    aggregate_inputs = summary.get("inputs")
    if not isinstance(aggregate_inputs, list) or len(aggregate_inputs) != len(seeds):
        raise MultiSeedReceiptError("replicate aggregate inputs are incomplete")
    observed_input_seeds: set[int] = set()
    for entry in aggregate_inputs:
        if not isinstance(entry, Mapping):
            raise MultiSeedReceiptError("replicate aggregate contains an invalid input")
        seed = int(entry.get("replicate_seed", -1))
        observed_input_seeds.add(seed)
        seed_summary = _inside(batch_root, entry.get("summary_path"), f"seed {seed} summary")
        comparisons = _inside(
            batch_root, entry.get("comparisons_path"), f"seed {seed} comparisons"
        )
        if _sha256(seed_summary) != entry.get("summary_sha256"):
            raise MultiSeedReceiptError(f"seed {seed} summary SHA-256 mismatch")
        if _sha256(comparisons) != entry.get("comparisons_sha256"):
            raise MultiSeedReceiptError(f"seed {seed} comparisons SHA-256 mismatch")
        seed_summary_payload = _load(seed_summary, f"seed {seed} summary")
        if seed_summary_payload.get("replicate_seed") != seed:
            raise MultiSeedReceiptError(f"seed {seed} summary identity mismatch")
        artifacts = seed_summary_payload.get("artifacts")
        comparison_reference = (
            artifacts.get("nex326_pilot_comparisons.csv")
            if isinstance(artifacts, Mapping)
            else None
        )
        if (
            not isinstance(comparison_reference, Mapping)
            or comparison_reference.get("sha256") != entry.get("comparisons_sha256")
        ):
            raise MultiSeedReceiptError(f"seed {seed} comparison reference mismatch")
    if observed_input_seeds != set(seeds):
        raise MultiSeedReceiptError("replicate aggregate inputs do not cover all seeds")

    aggregate_artifacts = summary.get("artifacts")
    if not isinstance(aggregate_artifacts, Mapping) or not aggregate_artifacts:
        raise MultiSeedReceiptError("replicate aggregate has no artifacts")
    for name, reference in aggregate_artifacts.items():
        if not isinstance(reference, Mapping):
            raise MultiSeedReceiptError(f"aggregate artifact {name} has an invalid reference")
        _verify_reference(summary_file.parent, reference, f"aggregate artifact {name}")

    receipt: dict[str, object] = {
        "schema_version": "nex326-dsde-multi-seed-receipt-v1",
        "experiment_id": "NEX326",
        "spec_version": batch["spec_version"],
        "scientific_status": "exploratory_only_not_final_scientific_evidence",
        "assessment": "not_assessed",
        "protocol_seed": batch["protocol_seed"],
        "replicate_seeds": list(seeds),
        "replicate_count": len(seeds),
        "execution_count_per_replicate": summary["execution_count_per_replicate"],
        "total_execution_count": len(seeds) * int(summary["execution_count_per_replicate"]),
        "cohort": batch["cohort"],
        "implementation": batch.get("implementation"),
        "comparison_status": summary["comparison_status"],
        "replicate_status": summary["replicate_status"],
        "method": summary["method"],
        "integrity": {
            "batch_manifest_sha256": _sha256(batch_file),
            "per_seed_manifests": per_seed_manifests,
            "replicate_aggregate_summary_sha256": _sha256(summary_file),
            "replicate_aggregate_inputs": aggregate_inputs,
            "replicate_aggregate_artifacts": aggregate_artifacts,
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
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--replicate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_multi_seed_receipt(
        args.batch_manifest, args.replicate_summary, args.output
    )
    print(
        json.dumps(
            {
                "replicate_seeds": receipt["replicate_seeds"],
                "total_execution_count": receipt["total_execution_count"],
                "scientific_status": receipt["scientific_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MultiSeedReceiptError", "build_multi_seed_receipt"]
