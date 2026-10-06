"""Read-only, versioned admission of existing public contracts.

An audit receipt is not a data authorization. Optional inputs are checked only
when requested, and frozen selections use their original validation routine.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from infrastructure.research_files import opened_regular_file
from infrastructure.research_json import read_json
from infrastructure.research_store import ResearchError
from .snapshot import UpstreamSnapshot, resolve_snapshot


class AdmissionError(ValueError):
    """An upstream input cannot satisfy its registered contract."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Binding:
    object_id: str
    path: str
    schema_version: str
    sha256: str
    status: str
    role: str = "metadata-only"
    units: tuple[str, ...] = ()
    compatibility: str = "canonical-json-sha256-v1"


PUBLIC_BINDINGS = (
    Binding("upstream-completion", "experiments/nex326/pirc19_completion_report.json",
            "pirc19-nex326-completion-audit-v2",
            "2fbc9ec36df35521a01ac67f541bb6d0ef8f860242c37002159f887c88547ae1",
            "pirc19_complete_with_approved_arm_exclusions"),
    Binding("cohort-audit", "experiments/nex326/pirc20_runtime_audit.json",
            "pirc20-nex326-runtime-audit-v1",
            "e45d1ab0c8f68717333b28725ca96161c5dbe026c633ad8da8671aaa45a54f22",
            "historical-audit-not-new-authorization"),
    Binding("affine-4d", "experiments/nex326/phase_space_benchmark.json",
            "nex326-phase-space-benchmark-spec-v1",
            "a1fc0d0964f7bc2b6cc2315479d7223af2f77ab2532fd77ac2850882bce41c7a",
            "supplemental_benchmark_not_a_frozen_arm", units=("m", "m", "m/s", "m/s")),
    Binding("terrain-selection", "experiments/pirc22/benchmark_selection.consumer.json",
            "pirc22-benchmark-selection-consumer-v1",
            "a9a8feba6d53353ab30a304c0d0ef176bda92e5b889ef5abc63f30c9b4c9ed9f",
            "selected"),
)


@contextmanager
def _metadata_scope():
    # Keep actual descriptors until every frozen validator and the final
    # uncached identity checks finish. This is not a trajectory authorization.
    try:
        with ExitStack() as stack:
            yield stack
    except ResearchError as exc:
        raise AdmissionError(str(exc)) from exc
    except OSError as exc:
        raise AdmissionError("MISSING_INPUT: upstream metadata at final verification") from exc


def _read_metadata(root: Path, relative, object_id, stack) -> Any:
    # Normalize .. without blessing a redirected root or ancestor.
    root = Path(os.path.abspath(root))
    path = Path(os.path.abspath(root / relative))
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise AdmissionError("UNAUTHORIZED_DATA: input path escapes root")
    try:
        stream, size, verify = stack.enter_context(opened_regular_file(root, path))
        # Preserve the historical UTF-8 and canonical-value JSON identity.
        # No arbitrary metadata quota: legitimate large JSON remains valid.
        return read_json(stream, size, encoding="utf-8", verify_identity=verify)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"MISSING_INPUT: {object_id}") from exc
    except UnicodeDecodeError as exc:
        raise AdmissionError(f"CORRUPT_ARTIFACT: upstream UTF-8: {object_id}") from exc


def _validate_binding(root, binding, stack):
    if binding.role != "metadata-only":
        raise AdmissionError("UNAUTHORIZED_DATA: data needs the exposure ledger")
    payload = _read_metadata(root, binding.path, binding.object_id, stack)
    if not isinstance(payload, dict) or canonical_hash(payload) != binding.sha256:
        raise AdmissionError("CORRUPT_ARTIFACT: upstream hash mismatch")
    if payload.get("schema_version") != binding.schema_version:
        raise AdmissionError("CONTRACT_MISMATCH: upstream schema")
    status = payload.get("status", payload.get("overall_status", payload.get("scientific_role")))
    if status is not None and status != binding.status:
        raise AdmissionError("CONTRACT_MISMATCH: upstream status")
    if binding.object_id == "affine-4d":
        if (binding.units != ("m", "m", "m/s", "m/s")
                or payload["state_contract"]["layout"] != ["x", "y", "vx", "vy"]):
            raise AdmissionError("CONTRACT_MISMATCH: affine state/units")
    return payload


def validate_binding(root: Path, binding: Binding) -> dict:
    with _metadata_scope() as stack:
        return _validate_binding(root, binding, stack)


def audit_inputs(root: Path, required: tuple[str, ...]) -> dict:
    """Return a hash-bound admission manifest without reading any trajectory."""
    from dataclasses import asdict

    registry = {binding.object_id: binding for binding in PUBLIC_BINDINGS}
    if len(set(required)) != len(required) or set(required) - registry.keys():
        raise AdmissionError("CONTRACT_MISMATCH: unknown or duplicate upstream ID")
    admitted = []
    with _metadata_scope() as stack:
        for object_id in required:
            binding = registry[object_id]
            payload = _validate_binding(root, binding, stack)
            if object_id == "terrain-selection":
                from experiments.pirc22.consumer import validate_benchmark_selection_binding
                from experiments.pirc22.representations import validate_representation_matrix
                matrix_path = Path("experiments/pirc22/representation_matrix.json")
                matrix = validate_representation_matrix(
                    _read_metadata(root, matrix_path, "terrain-matrix", stack),
                    _read_metadata(root, matrix_path.with_suffix(".lock.json"), "terrain-lock", stack),
                )
                validate_benchmark_selection_binding(payload, matrix=matrix)
            admitted.append(asdict(binding))
    manifest = {"schema_version": "pirc25-upstream-binding-v1", "objects": admitted,
                "data_authorization": "none", "fit_scope": "train-only",
                "feature_adapter": "pirc21-psde-adapter-v2",
                "legacy_adapt_policy": "explicit-role-mapping-required"}
    return {**manifest, "manifest_hash": canonical_hash(manifest)}


def affine_state(segment, *, units=("m", "m", "m/s", "m/s")):
    """Delegate the four-dimensional conversion, preserving its causal policy."""
    from experiments.nex326.phase_space import phase_space_state

    if tuple(units) != ("m", "m", "m/s", "m/s"):
        raise AdmissionError("CONTRACT_MISMATCH: state units")
    segment.validate()
    return phase_space_state(segment)
