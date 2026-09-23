"""Leakage-closed data boundary for the PIRC-22 terrain benchmark.

The benchmark is intentionally narrower than the general NEX326 runtime: it can
materialize only the frozen train, adapt, and validation roles.  Final evaluation
is not retained by this object and there is no unlock parameter on its production
factory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .nex326.cohort import Cohort, Segment
from .nex326.pirc20_adapter import load_pirc20_nex326_cohort


BENCHMARK_DATA_SCHEMA_VERSION = "pirc22-benchmark-data-v1"
DEFAULT_FOLD_SEED = 20260922
BENCHMARK_ROLES = ("train", "adapt", "validation")
_FINAL_EVAL_NAMES = {"evaluation", "eval", "final_eval", "final-eval"}


class BenchmarkIsolationError(ValueError):
    """The requested operation could expose data outside the benchmark boundary."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _rank(seed: int, block_id: str) -> str:
    return hashlib.sha256(f"{seed}:{block_id}".encode("utf-8")).hexdigest()


def _segment_identity(segment: Segment) -> dict[str, str]:
    independent_block_id = getattr(segment, "independent_block_id", None)
    if independent_block_id is None:
        raise BenchmarkIsolationError(
            f"segment lacks independent_block_id: {segment.segment_id}"
        )
    return {
        "segment_id": segment.segment_id,
        "independent_block_id": str(independent_block_id),
    }


@dataclass(frozen=True)
class BenchmarkFold:
    """One deterministic partition of the frozen validation blocks."""

    fold_id: str
    validation_block_ids: tuple[str, ...]
    validation_segment_ids: tuple[str, ...]
    identity_sha256: str

    @property
    def independent_block_count(self) -> int:
        return len(self.validation_block_ids)

    @property
    def segment_count(self) -> int:
        return len(self.validation_segment_ids)


class BenchmarkDataBoundary:
    """A hash-addressed train/adapt/validation view with grouped folds."""

    def __init__(
        self,
        cohort: Cohort,
        *,
        fold_count: int = 5,
        fold_seed: int = DEFAULT_FOLD_SEED,
    ) -> None:
        if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 1:
            raise BenchmarkIsolationError("fold_count must be a positive integer")
        if isinstance(fold_seed, bool) or not isinstance(fold_seed, int):
            raise BenchmarkIsolationError("fold_seed must be an integer")
        try:
            cohort.validate()
        except ValueError as error:
            raise BenchmarkIsolationError(str(error)) from error
        if cohort.splits.get("evaluation"):
            raise BenchmarkIsolationError(
                "benchmark input already materialized final evaluation"
            )
        evaluation_reason = cohort.unavailable_reasons.get("evaluation", "")
        if "sealed" not in evaluation_reason.casefold():
            raise BenchmarkIsolationError(
                "benchmark input must prove that final evaluation remained sealed"
            )

        segments = {
            role: tuple(cohort.splits.get(role, ())) for role in BENCHMARK_ROLES
        }
        if not segments["train"]:
            raise BenchmarkIsolationError("benchmark train role is empty")
        if not segments["validation"]:
            raise BenchmarkIsolationError("benchmark validation role is empty")

        block_roles: dict[str, str] = {}
        for role, values in segments.items():
            for segment in values:
                identity = _segment_identity(segment)
                block_id = identity["independent_block_id"]
                previous = block_roles.setdefault(block_id, role)
                if previous != role:
                    raise BenchmarkIsolationError(
                        f"independent block crosses benchmark roles: {block_id}"
                    )

        validation_by_block: dict[str, list[Segment]] = defaultdict(list)
        for segment in segments["validation"]:
            validation_by_block[
                str(getattr(segment, "independent_block_id", ""))
            ].append(segment)
        effective_fold_count = min(fold_count, len(validation_by_block))
        if effective_fold_count < 1:
            raise BenchmarkIsolationError("validation has no independent blocks")

        assignments: list[list[str]] = [[] for _ in range(effective_fold_count)]
        assigned_segments = [0] * effective_fold_count
        ordered_blocks = sorted(
            validation_by_block,
            key=lambda block_id: (
                -len(validation_by_block[block_id]),
                _rank(fold_seed, block_id),
                block_id,
            ),
        )
        for block_id in ordered_blocks:
            target = min(
                range(effective_fold_count),
                key=lambda index: (
                    assigned_segments[index], len(assignments[index]), index
                ),
            )
            assignments[target].append(block_id)
            assigned_segments[target] += len(validation_by_block[block_id])

        folds: list[BenchmarkFold] = []
        for index, block_ids in enumerate(assignments):
            canonical_blocks = tuple(sorted(block_ids))
            segment_ids = tuple(
                sorted(
                    segment.segment_id
                    for block_id in canonical_blocks
                    for segment in validation_by_block[block_id]
                )
            )
            fold_id = f"fold-{index + 1:02d}"
            folds.append(
                BenchmarkFold(
                    fold_id=fold_id,
                    validation_block_ids=canonical_blocks,
                    validation_segment_ids=segment_ids,
                    identity_sha256=_canonical_hash(
                        {
                            "schema_version": BENCHMARK_DATA_SCHEMA_VERSION,
                            "fold_id": fold_id,
                            "validation_block_ids": canonical_blocks,
                            "validation_segment_ids": segment_ids,
                        }
                    ),
                )
            )

        self._segments = segments
        self._folds = tuple(folds)
        self._fold_by_id = {fold.fold_id: fold for fold in folds}
        self._validation_by_segment = {
            segment.segment_id: segment for segment in segments["validation"]
        }
        self._allowed_reads: Counter[str] = Counter()
        self._denied_attempts: list[dict[str, object]] = []
        self._identity = self._build_identity(
            cohort,
            requested_fold_count=fold_count,
            fold_seed=fold_seed,
        )

    @classmethod
    def from_pirc20_release(
        cls,
        cohort_path: str | Path,
        trajectory_path: str | Path,
        condition_root: str | Path,
        *,
        fold_count: int = 5,
        fold_seed: int = DEFAULT_FOLD_SEED,
        maximum_segments_per_role: Mapping[str, int] | None = None,
    ) -> "BenchmarkDataBoundary":
        """Create the only production benchmark view; final-eval unlock is absent."""

        cohort = load_pirc20_nex326_cohort(
            cohort_path,
            trajectory_path,
            condition_root,
            maximum_segments_per_role=maximum_segments_per_role,
        )
        return cls(cohort, fold_count=fold_count, fold_seed=fold_seed)

    def _build_identity(
        self, cohort: Cohort, *, requested_fold_count: int, fold_seed: int
    ) -> dict[str, object]:
        role_records: dict[str, object] = {}
        for role, segments in self._segments.items():
            identities = tuple(
                sorted(
                    (_segment_identity(segment) for segment in segments),
                    key=lambda value: value["segment_id"],
                )
            )
            role_records[role] = {
                "segment_count": len(segments),
                "independent_block_count": len(
                    {value["independent_block_id"] for value in identities}
                ),
                "identity_sha256": _canonical_hash(identities),
            }
        fold_counts = [fold.segment_count for fold in self._folds]
        payload: dict[str, object] = {
            "schema_version": BENCHMARK_DATA_SCHEMA_VERSION,
            "dataset_id": cohort.dataset_id,
            "data_version": cohort.data_version,
            "source_cohort_fingerprint": cohort.fingerprint,
            "allowed_roles": list(BENCHMARK_ROLES),
            "fit_role": "train",
            "final_eval_policy": "sealed_identity_only_no_materialization",
            "requested_fold_count": requested_fold_count,
            "effective_fold_count": len(self._folds),
            "fold_seed": fold_seed,
            "fold_balance": {
                "validation_segments_per_fold": fold_counts,
                "minimum": min(fold_counts),
                "maximum": max(fold_counts),
            },
            "roles": role_records,
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "validation_block_ids": list(fold.validation_block_ids),
                    "validation_segment_ids": list(fold.validation_segment_ids),
                    "independent_block_count": fold.independent_block_count,
                    "segment_count": fold.segment_count,
                    "identity_sha256": fold.identity_sha256,
                }
                for fold in self._folds
            ],
        }
        payload["benchmark_data_identity_sha256"] = _canonical_hash(payload)
        return payload

    @property
    def folds(self) -> tuple[BenchmarkFold, ...]:
        return self._folds

    @property
    def identity_record(self) -> dict[str, object]:
        return json.loads(json.dumps(self._identity))

    @property
    def benchmark_data_identity_sha256(self) -> str:
        return str(self._identity["benchmark_data_identity_sha256"])

    def _deny(self, role: str, reason: str) -> None:
        self._denied_attempts.append(
            {"sequence": len(self._denied_attempts) + 1, "role": role, "reason": reason}
        )
        raise BenchmarkIsolationError(reason)

    def segments(
        self, role: str, *, fold_id: str | None = None
    ) -> tuple[Segment, ...]:
        """Return an allowed role, requiring an explicit fold for validation."""

        normalized = role.casefold().replace("-", "_")
        if normalized in {value.replace("-", "_") for value in _FINAL_EVAL_NAMES}:
            self._deny(role, "final evaluation is not readable in benchmark mode")
        if normalized not in BENCHMARK_ROLES:
            self._deny(role, f"role is outside benchmark scope: {role}")
        if normalized == "validation":
            if fold_id is None:
                self._deny(role, "validation access requires an explicit fold_id")
            fold = self._fold_by_id.get(fold_id)
            if fold is None:
                self._deny(role, f"unknown validation fold: {fold_id}")
            result = tuple(
                self._validation_by_segment[segment_id]
                for segment_id in fold.validation_segment_ids
            )
        else:
            if fold_id is not None and fold_id not in self._fold_by_id:
                self._deny(role, f"unknown benchmark fold: {fold_id}")
            result = self._segments[normalized]
        self._allowed_reads[normalized] += 1
        return result

    def fit_segments(self, *, fold_id: str | None = None) -> tuple[Segment, ...]:
        """Expose the only legal preprocessing-fit population: outer train."""

        return self.segments("train", fold_id=fold_id)

    @property
    def audit_record(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "pirc22-benchmark-access-audit-v1",
            "benchmark_data_identity_sha256": self.benchmark_data_identity_sha256,
            "allowed_read_counts": {
                role: self._allowed_reads[role] for role in BENCHMARK_ROLES
            },
            "denied_attempts": list(self._denied_attempts),
            "final_eval_materialized": False,
            "final_eval_read_count": 0,
        }
        payload["audit_identity_sha256"] = _canonical_hash(payload)
        return payload


__all__ = [
    "BENCHMARK_DATA_SCHEMA_VERSION",
    "BENCHMARK_ROLES",
    "BenchmarkDataBoundary",
    "BenchmarkFold",
    "BenchmarkIsolationError",
    "DEFAULT_FOLD_SEED",
]
