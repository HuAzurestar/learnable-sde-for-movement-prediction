"""Execute exactly the non-excluded portion of a frozen NEX326 specification."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from .runner import NEX326Runner


DEFAULT_SCOPE_POLICY = Path(__file__).with_name("pirc19_scope_policy.json")
SCOPE_SCHEMA_VERSION = "nex326-scoped-run-manifest-v1"


class ScopeRunError(ValueError):
    """A scope policy is invalid or does not match the frozen experiment matrix."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(path: Path, runner: NEX326Runner) -> tuple[dict[str, object], set[int]]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScopeRunError("scope policy cannot be read") from error
    exclusions = policy.get("approved_excluded_arms")
    if (
        policy.get("schema_version") != "pirc19-reproduction-scope-v1"
        or policy.get("experiment_id") != runner.spec.experiment_id
        or not isinstance(exclusions, list)
    ):
        raise ScopeRunError("scope policy does not match the NEX326 contract")
    excluded_ids = {
        int(item["arm_id"])
        for item in exclusions
        if isinstance(item, Mapping) and "arm_id" in item
    }
    if excluded_ids != {13, 17, 22}:
        raise ScopeRunError("approved excluded arms must remain exactly 13, 17, and 22")
    required = [
        (arm, subconfig)
        for arm, subconfig in runner.spec.executions
        if arm.arm_id not in excluded_ids
    ]
    if len(required) != int(policy.get("required_execution_count", -1)):
        raise ScopeRunError("scope policy required execution count is stale")
    return policy, excluded_ids


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ScopeRunError(f"stale manifest staging file exists: {temporary}")
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


def run_approved_scope(
    runner: NEX326Runner,
    policy_path: Path | str = DEFAULT_SCOPE_POLICY,
) -> tuple[dict[str, object], ...]:
    """Run the 28 required slots while recording all approved exclusions."""

    source = Path(policy_path).resolve()
    policy, excluded_ids = _load_policy(source, runner)
    manifest_path = runner.output_root / "manifest.json"
    if manifest_path.exists():
        raise ScopeRunError(f"run manifest already exists: {manifest_path}")
    executions = tuple(
        (arm, subconfig)
        for arm, subconfig in runner.spec.executions
        if arm.arm_id not in excluded_ids
    )
    records = tuple(runner.run_one(arm, subconfig) for arm, subconfig in executions)
    status = Counter(str(record["run_status"]) for record in records)
    excluded = [
        {
            "arm_id": int(item["arm_id"]),
            "reason": str(item["reason"]),
            "execution_count": sum(
                arm.arm_id == int(item["arm_id"])
                for arm, _ in runner.spec.executions
            ),
        }
        for item in policy["approved_excluded_arms"]
    ]
    manifest = {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "experiment_id": runner.spec.experiment_id,
        "spec_version": runner.spec.spec_version,
        "record_count": len(records),
        "protocol_seed": runner.spec.seed,
        "replicate_seed": runner.replicate_seed,
        "requested_prediction_samples": runner.n_samples,
        "implementation": runner.implementation,
        "arm_ids": sorted({int(record["arm_id"]) for record in records}),
        "run_status": dict(sorted(status.items())),
        "scope": {
            "policy_file": source.name,
            "policy_sha256": _sha256(source),
            "approved_excluded_arms": excluded,
            "required_execution_count": len(executions),
        },
        "run_records": [
            f"{record['run_id']}/run_record.json" for record in records
        ],
    }
    _write_json_atomic(manifest_path, manifest)
    return records


__all__ = [
    "DEFAULT_SCOPE_POLICY",
    "SCOPE_SCHEMA_VERSION",
    "ScopeRunError",
    "run_approved_scope",
]
