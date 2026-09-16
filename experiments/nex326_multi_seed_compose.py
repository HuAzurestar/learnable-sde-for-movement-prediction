"""Compose verified standalone NEX326 run roots into one multi-seed manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.nex326.runner import validate_run_record
from experiments.nex326.specification import load_experiment_spec


class ComposeError(ValueError):
    """Standalone replicate roots cannot form one coherent batch."""


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
        raise ComposeError(f"{label} cannot be read") from exc
    if not isinstance(payload, dict):
        raise ComposeError(f"{label} must be a JSON object")
    return payload


def _relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ComposeError(f"{label} escapes the combined manifest root") from exc


def _verified_run(root: Path) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    manifest_path = root / "manifest.json"
    manifest = _load(manifest_path, "replicate manifest")
    if manifest.get("schema_version") != "nex326-scoped-run-manifest-v1":
        raise ComposeError("replicate is not an approved scoped run")
    references = manifest.get("run_records")
    if not isinstance(references, list) or len(references) != 28:
        raise ComposeError("replicate manifest must reference exactly 28 records")
    resolved_root = root.resolve()
    records: list[dict[str, object]] = []
    for relative in references:
        record_path = (root / str(relative)).resolve()
        try:
            record_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ComposeError("RunRecord path escapes its replicate root") from exc
        record = _load(record_path, "RunRecord")
        validate_run_record(record)
        if f"{record['run_id']}/run_record.json" != str(relative).replace("\\", "/"):
            raise ComposeError("RunRecord identity does not match its manifest path")
        artifacts = record["artifacts"]
        for name, reference in artifacts.items():  # type: ignore[union-attr]
            artifact_path = (root / str(reference["path"])).resolve()
            try:
                artifact_path.relative_to(resolved_root)
            except ValueError as exc:
                raise ComposeError(f"artifact {name} escapes its replicate root") from exc
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != int(reference["size_bytes"])
                or _sha256(artifact_path) != reference["sha256"]
            ):
                raise ComposeError(f"artifact {name} fails integrity verification")
        records.append(record)
    if len({str(record["run_id"]) for record in records}) != len(records):
        raise ComposeError("replicate contains duplicate RunRecord identities")
    expected_keys = {
        (arm.arm_id, str(subconfig["subconfig_id"]))
        for arm, subconfig in load_experiment_spec().executions
        if arm.arm_id not in {13, 17, 22}
    }
    observed_keys = {
        (int(record["arm_id"]), str(record["subconfig_id"])) for record in records
    }
    if observed_keys != expected_keys:
        raise ComposeError("replicate does not contain the exact approved 28-slot matrix")
    if any(record["implementation"] != manifest.get("implementation") for record in records):
        raise ComposeError("replicate manifest and RunRecords bind different implementations")
    return manifest, tuple(records)


def compose_replicates(
    cohort_path: Path | str,
    run_roots: Sequence[Path | str],
    output: Path | str,
    *,
    scope_policy: Path | str,
) -> dict[str, object]:
    """Verify and compose at least two standalone scoped replicate roots."""

    roots = tuple(Path(root).resolve() for root in run_roots)
    if len(roots) < 2 or len(roots) != len(set(roots)):
        raise ComposeError("at least two distinct replicate roots are required")
    destination = Path(output).resolve()
    if destination.exists():
        raise ComposeError(f"combined manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cohort_file = Path(cohort_path).resolve()
    cohort = _load(cohort_file, "PIRC-20 cohort")
    if cohort.get("schema_version") != "pirc20-cohort-v1":
        raise ComposeError("combined batch replicate composition requires a PIRC-20 cohort")
    policy_file = Path(scope_policy).resolve()
    policy = _load(policy_file, "scope policy")
    policy_hash = _sha256(policy_file)
    if (
        policy.get("schema_version") != "pirc19-reproduction-scope-v1"
        or policy.get("experiment_id") != "NEX326"
        or {
            int(item["arm_id"])
            for item in policy.get("approved_excluded_arms", [])
        }
        != {13, 17, 22}
        or policy.get("required_execution_count") != 28
    ):
        raise ComposeError("scope policy does not declare the approved 28-slot scope")

    entries: list[dict[str, object]] = []
    seeds: list[int] = []
    implementations: list[object] = []
    fingerprints: set[str] = set()
    protocol_seeds: set[int] = set()
    requested_sample_counts: set[int] = set()
    spec_versions: set[str] = set()
    dataset_ids: set[str] = set()
    for root in roots:
        relative_root = _relative(root, destination.parent, "replicate root")
        manifest, records = _verified_run(root)
        manifest_scope = manifest.get("scope")
        if (
            not isinstance(manifest_scope, Mapping)
            or manifest_scope.get("policy_sha256") != policy_hash
            or manifest.get("record_count") != 28
            or manifest.get("run_status") != {"succeeded": 28}
        ):
            raise ComposeError("replicate manifest does not match the approved scope")
        seed = int(manifest["replicate_seed"])
        if any(int(record["replicate_seed"]) != seed for record in records):
            raise ComposeError("RunRecord replicate seed disagrees with its manifest")
        seeds.append(seed)
        implementations.append(manifest.get("implementation"))
        protocol_seeds.add(int(manifest["protocol_seed"]))
        requested_sample_counts.add(int(manifest["requested_prediction_samples"]))
        spec_versions.add(str(manifest["spec_version"]))
        fingerprints.update(str(record["dataset"]["fingerprint"]) for record in records)
        dataset_ids.update(str(record["dataset"]["dataset_id"]) for record in records)
        entries.append(
            {
                "replicate_seed": seed,
                "records_root": relative_root,
                "record_count": 28,
                "run_status": dict(Counter(str(record["run_status"]) for record in records)),
                "manifest": {
                    "path": f"{relative_root}/manifest.json",
                    "sha256": _sha256(root / "manifest.json"),
                    "size_bytes": (root / "manifest.json").stat().st_size,
                },
            }
        )
    if len(seeds) != len(set(seeds)):
        raise ComposeError("replicate seeds must be unique")
    canonical_implementations = {
        json.dumps(item, sort_keys=True, separators=(",", ":"))
        for item in implementations
    }
    if (
        len(canonical_implementations) != 1
        or len(fingerprints) != 1
        or len(protocol_seeds) != 1
        or len(requested_sample_counts) != 1
        or len(spec_versions) != 1
        or dataset_ids != {str(cohort.get("cohort_id"))}
    ):
        raise ComposeError("replicates do not share one implementation and cohort")
    ordered = sorted(zip(seeds, entries), key=lambda item: item[0])
    payload: dict[str, object] = {
        "schema_version": "nex326-multi-seed-manifest-v1",
        "experiment_id": "NEX326",
        "spec_version": next(iter(spec_versions)),
        "scientific_status": "replicate_runs_not_combined_verdict",
        "protocol_seed": next(iter(protocol_seeds)),
        "replicate_seeds": [seed for seed, _ in ordered],
        "replicate_count": len(ordered),
        "requested_prediction_samples": next(iter(requested_sample_counts)),
        "implementation": implementations[0],
        "executions_per_replicate": 28,
        "total_executions": 28 * len(ordered),
        "cohort": {
            "source_file": cohort_file.name,
            "sha256": _sha256(cohort_file),
            "fingerprint": next(iter(fingerprints)),
        },
        "scope_policy": {
            "source_file": policy_file.name,
            "sha256": policy_hash,
        },
        "replicates": [entry for _, entry in ordered],
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ComposeError(f"stale manifest staging file exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--scope-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = compose_replicates(
        args.cohort,
        args.run_roots,
        args.output,
        scope_policy=args.scope_policy,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ComposeError", "compose_replicates"]
