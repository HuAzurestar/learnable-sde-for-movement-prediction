"""Public reader for a frozen PIRC-20 cohort manifest."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping


COHORT_SCHEMA_VERSION = "pirc20-cohort-v1"
SPLITS = ("train", "validation", "final_eval")


class PIRC20CohortError(ValueError):
    """A frozen DSDE cohort is missing, changed, or internally inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(payload: Mapping[str, object]) -> str:
    identity = {
        name: payload[name]
        for name in (
            "data_version",
            "file_id",
            "segment_id",
            "history_start",
            "history_end",
            "target_start",
            "target_end",
        )
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PIRC20Sample:
    sample_id: str
    data_version: str
    file_id: str
    segment_id: str
    split: str
    independent_block_id: str
    history_start: int
    history_end: int
    target_start: int
    target_end: int
    factor_availability: Mapping[str, object]


@dataclass(frozen=True)
class PIRC20Cohort:
    path: Path
    cohort_id: str
    data_version: str
    sample_manifest: Path
    sample_count: int
    sample_counts_by_split: Mapping[str, int]
    window: Mapping[str, object]
    final_eval_access: str

    def _payloads(self) -> Iterator[dict[str, object]]:
        with self.sample_manifest.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise PIRC20CohortError(
                        f"invalid sample JSON at line {line_number}"
                    ) from error
                if not isinstance(payload, dict):
                    raise PIRC20CohortError(
                        f"sample line {line_number} is not an object"
                    )
                yield payload

    def iter_samples(self, split: str | None = None) -> Iterator[PIRC20Sample]:
        """Iterate immutable references; this never opens trajectory or feature data."""

        if split is not None and split not in SPLITS:
            raise PIRC20CohortError(f"unsupported split: {split}")
        for payload in self._payloads():
            if split is None or payload["split"] == split:
                yield _parse_sample(payload)

    def samples_for_model(
        self, split: str, *, final_eval_unlocked: bool = False
    ) -> Iterator[PIRC20Sample]:
        """Return model inputs while keeping final eval sealed by default."""

        if split == "final_eval" and not final_eval_unlocked:
            raise PIRC20CohortError("final_eval is sealed; explicit unlock is required")
        return self.iter_samples(split)


def _parse_sample(payload: Mapping[str, object]) -> PIRC20Sample:
    required = {
        "sample_id",
        "data_version",
        "file_id",
        "segment_id",
        "split",
        "independent_block_id",
        "history_start",
        "history_end",
        "target_start",
        "target_end",
        "factor_availability",
    }
    missing = required - set(payload)
    if missing:
        raise PIRC20CohortError(f"sample is missing fields: {sorted(missing)}")
    sample = PIRC20Sample(
        sample_id=str(payload["sample_id"]),
        data_version=str(payload["data_version"]),
        file_id=str(payload["file_id"]),
        segment_id=str(payload["segment_id"]),
        split=str(payload["split"]),
        independent_block_id=str(payload["independent_block_id"]),
        history_start=int(payload["history_start"]),
        history_end=int(payload["history_end"]),
        target_start=int(payload["target_start"]),
        target_end=int(payload["target_end"]),
        factor_availability=dict(payload["factor_availability"]),
    )
    if sample.split not in SPLITS:
        raise PIRC20CohortError(f"unsupported sample split: {sample.split}")
    if not sample.file_id or not sample.segment_id or not sample.independent_block_id:
        raise PIRC20CohortError("sample identity fields must be non-empty")
    if not (
        0 <= sample.history_start <= sample.history_end
        < sample.target_start <= sample.target_end
    ):
        raise PIRC20CohortError(f"invalid history/target bounds: {sample.sample_id}")
    if sample.target_start != sample.history_end + 1:
        raise PIRC20CohortError(f"history and target are not adjacent: {sample.sample_id}")
    if sample.sample_id != _sample_id(payload):
        raise PIRC20CohortError(f"sample identity hash mismatch: {sample.sample_id}")
    return sample


def load_pirc20_cohort(path: str | Path) -> PIRC20Cohort:
    """Load and fully validate one explicit cohort file and its bound sample list."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != COHORT_SCHEMA_VERSION:
        raise PIRC20CohortError("unsupported PIRC-20 cohort schema")
    manifest_name = str(payload.get("sample_manifest", ""))
    if not manifest_name or Path(manifest_name).name != manifest_name:
        raise PIRC20CohortError("sample_manifest must be one explicit sibling file")
    sample_manifest = source.parent / manifest_name
    if not sample_manifest.is_file() or sample_manifest.is_symlink():
        raise PIRC20CohortError("declared sample manifest is missing or is a symlink")
    if _sha256(sample_manifest) != payload.get("sample_manifest_sha256"):
        raise PIRC20CohortError("sample manifest hash mismatch")
    if payload.get("ordered_sample_ids_hash_encoding") != (
        "utf8_sample_id_newline_in_manifest_order"
    ):
        raise PIRC20CohortError("unsupported ordered sample ID hash encoding")

    cohort = PIRC20Cohort(
        path=source,
        cohort_id=str(payload.get("cohort_id", "")),
        data_version=str(payload.get("data_version", "")),
        sample_manifest=sample_manifest,
        sample_count=int(payload.get("sample_count", -1)),
        sample_counts_by_split={
            name: int(value)
            for name, value in payload.get("sample_counts_by_split", {}).items()
        },
        window=dict(payload.get("window", {})),
        final_eval_access=str(payload.get("final_eval_access", "")),
    )
    if not cohort.cohort_id or not cohort.data_version:
        raise PIRC20CohortError("cohort identity is missing")
    if set(cohort.sample_counts_by_split) != set(SPLITS):
        raise PIRC20CohortError("cohort split counts are incomplete")
    if cohort.final_eval_access != "sealed_identity_only":
        raise PIRC20CohortError("final_eval must be sealed at publication")

    counts: Counter[str] = Counter()
    ordered_digest = hashlib.sha256()
    segment_split: dict[str, str] = {}
    block_splits: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    window_mode = str(cohort.window.get("mode", "fixed_points"))
    if window_mode == "fixed_points":
        history_points = int(cohort.window.get("history_points", -1))
        target_points = int(cohort.window.get("target_points", -1))
    elif window_mode == "nex326_midpoint":
        history_points = target_points = None
    else:
        raise PIRC20CohortError(f"unsupported cohort window mode: {window_mode}")
    for sample in cohort.iter_samples():
        if sample.data_version != cohort.data_version:
            raise PIRC20CohortError(f"sample data version mismatch: {sample.sample_id}")
        if sample.sample_id in seen_ids:
            raise PIRC20CohortError(f"duplicate sample ID: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        if window_mode == "fixed_points":
            if sample.history_end - sample.history_start + 1 != history_points:
                raise PIRC20CohortError(f"history width mismatch: {sample.sample_id}")
            if sample.target_end - sample.target_start + 1 != target_points:
                raise PIRC20CohortError(f"target width mismatch: {sample.sample_id}")
        else:
            point_count = sample.target_end + 1
            expected_history_end = max(1, point_count // 2 - 1)
            if sample.history_start != 0 or sample.history_end != expected_history_end:
                raise PIRC20CohortError(
                    f"NEX326 midpoint boundary mismatch: {sample.sample_id}"
                )
        prior_split = segment_split.setdefault(sample.segment_id, sample.split)
        if prior_split != sample.split:
            raise PIRC20CohortError(f"segment crosses splits: {sample.segment_id}")
        block_splits[sample.independent_block_id].add(sample.split)
        counts[sample.split] += 1
        ordered_digest.update(sample.sample_id.encode("utf-8"))
        ordered_digest.update(b"\n")

    if len(seen_ids) != cohort.sample_count:
        raise PIRC20CohortError("sample count mismatch")
    if {name: counts[name] for name in SPLITS} != dict(cohort.sample_counts_by_split):
        raise PIRC20CohortError("sample split count mismatch")
    if any(len(values) != 1 for values in block_splits.values()):
        raise PIRC20CohortError("independent block crosses splits")
    if ordered_digest.hexdigest() != payload.get("ordered_sample_ids_sha256"):
        raise PIRC20CohortError("ordered sample ID hash mismatch")
    return cohort


__all__ = [
    "COHORT_SCHEMA_VERSION",
    "PIRC20Cohort",
    "PIRC20CohortError",
    "PIRC20Sample",
    "load_pirc20_cohort",
]
