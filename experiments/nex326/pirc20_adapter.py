"""Bind a frozen PIRC-20 release to the NEX326 22-arm runtime.

The adapter does not rewrite the published cohort.  It verifies the release and
source hashes, derives the registered adaptation role only inside the outer training
split, and leaves final evaluation unread unless the caller supplies the exact cohort
ID as an unlock acknowledgement.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import duckdb
import numpy as np
import pyarrow.parquet as pq

from data.pirc20 import PIRC20Cohort, PIRC20Sample, load_pirc20_cohort

from .cohort import Cohort, Segment


ADAPTER_VERSION = "pirc20-nex326-adapter-v1"
ADAPT_SEED = 20260912
ADAPT_FRACTION = 0.20


class PIRC20AdapterError(ValueError):
    """The frozen PIRC-20 release cannot be bound to the NEX326 runtime."""


class _UnusableSample(PIRC20AdapterError):
    """A sample has too few distinct exact timestamps for SDE execution."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PIRC20AdapterError(f"cannot read JSON artifact: {path.name}") from error
    if not isinstance(payload, dict):
        raise PIRC20AdapterError(f"JSON artifact is not an object: {path.name}")
    return payload


def _artifact_path(release_root: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise PIRC20AdapterError(f"release artifact name is unsafe: {name!r}")
    path = release_root / name
    if not path.is_file() or path.is_symlink():
        raise PIRC20AdapterError(f"release artifact is missing or symlinked: {name}")
    return path


def _verify_release(
    cohort: PIRC20Cohort, trajectory_path: Path
) -> tuple[dict[str, object], Path, Path]:
    release_root = cohort.path.parent
    dataset_path = _artifact_path(release_root, "dataset.json")
    dataset = _json(dataset_path)
    if dataset.get("schema_version") != "pirc20-release-v1":
        raise PIRC20AdapterError("unsupported PIRC-20 release schema")
    if dataset.get("dataset_id") != cohort.cohort_id:
        raise PIRC20AdapterError("dataset/cohort identity mismatch")

    artifacts = dataset.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PIRC20AdapterError("dataset artifact registry is missing")
    required = ("cohort.json", "alignment.jsonl", "condition_file_manifest.jsonl")
    paths: dict[str, Path] = {}
    for name in required:
        entry = artifacts.get(name)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
            raise PIRC20AdapterError(f"dataset artifact hash is missing: {name}")
        path = _artifact_path(release_root, name)
        if _sha256(path) != entry["sha256"]:
            raise PIRC20AdapterError(f"dataset artifact hash mismatch: {name}")
        paths[name] = path

    source = dataset.get("source")
    trajectory = source.get("trajectory") if isinstance(source, Mapping) else None
    expected = trajectory.get("sha256") if isinstance(trajectory, Mapping) else None
    if not isinstance(expected, str) or _sha256(trajectory_path) != expected:
        raise PIRC20AdapterError("trajectory source hash mismatch")
    return dataset, paths["alignment.jsonl"], paths["condition_file_manifest.jsonl"]


def _rank(block_id: str) -> str:
    return hashlib.sha256(f"{ADAPT_SEED}:{block_id}".encode("utf-8")).hexdigest()


def _roles(
    samples: Sequence[PIRC20Sample], final_eval_unlocked: bool
) -> dict[str, str]:
    train_blocks = sorted(
        {sample.independent_block_id for sample in samples if sample.split == "train"},
        key=_rank,
    )
    if len(train_blocks) < 2:
        raise PIRC20AdapterError("outer train needs at least two independent blocks")
    train_count = max(
        1, min(len(train_blocks) - 1, round((1.0 - ADAPT_FRACTION) * len(train_blocks)))
    )
    fit_blocks = set(train_blocks[:train_count])
    roles: dict[str, str] = {}
    for sample in samples:
        if sample.split == "train":
            roles[sample.segment_id] = (
                "train" if sample.independent_block_id in fit_blocks else "adapt"
            )
        elif sample.split == "validation":
            roles[sample.segment_id] = "validation"
        elif final_eval_unlocked:
            roles[sample.segment_id] = "evaluation"
    return roles


class _ConditionStore:
    def __init__(self, root: Path, manifest_path: Path, *, cache_size: int = 8) -> None:
        self.root = root.resolve()
        self.cache_size = cache_size
        self.entries: dict[str, tuple[Path, str]] = {}
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        with manifest_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                    file_id = str(row["file_id"])
                    relative = Path(str(row["relative_path"]))
                    digest = str(row["sha256"])
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise PIRC20AdapterError(
                        f"invalid condition manifest row: {line_number}"
                    ) from error
                path = (self.root / relative).resolve()
                try:
                    path.relative_to(self.root)
                except ValueError as error:
                    raise PIRC20AdapterError(
                        "condition path escapes its declared root"
                    ) from error
                if file_id in self.entries:
                    raise PIRC20AdapterError(f"duplicate condition identity: {file_id}")
                self.entries[file_id] = (path, digest)

    def _load(self, file_id: str) -> dict[str, np.ndarray]:
        cached = self.cache.pop(file_id, None)
        if cached is not None:
            self.cache[file_id] = cached
            return cached
        if file_id not in self.entries:
            raise PIRC20AdapterError(f"condition file is not release-bound: {file_id}")
        path, expected = self.entries[file_id]
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise PIRC20AdapterError(f"condition source hash mismatch: {file_id}")
        names = (
            "file_id",
            "solar_elev",
            "is_day",
            "dem_elev",
            "dem_slope",
        )
        table = pq.read_table(path, columns=list(names)).combine_chunks()
        identities = table["file_id"].to_pylist()
        if not identities or any(str(value) != file_id for value in identities):
            raise PIRC20AdapterError(f"condition source identity mismatch: {file_id}")
        arrays = {
            name: np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=float)
            for name in names[1:]
        }
        self.cache[file_id] = arrays
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return arrays

    def select(
        self, file_id: str, indexes: np.ndarray
    ) -> tuple[dict[str, np.ndarray], bool]:
        arrays = self._load(file_id)
        if indexes.size == 0 or indexes.min() < 0:
            raise PIRC20AdapterError(f"invalid condition indexes: {file_id}")
        if indexes.max() >= len(arrays["solar_elev"]):
            raise PIRC20AdapterError(f"condition index is out of range: {file_id}")
        selected = {name: values[indexes] for name, values in arrays.items()}
        conditions: dict[str, np.ndarray] = {}
        for source_name, runtime_name in (
            ("solar_elev", "solar_elev"),
            ("is_day", "is_day"),
        ):
            if np.isfinite(selected[source_name]).all():
                conditions[runtime_name] = selected[source_name]
        has_terrain = bool(
            np.isfinite(selected["dem_elev"]).all()
            and np.isfinite(selected["dem_slope"]).all()
        )
        if has_terrain:
            conditions["terrain_elevation"] = selected["dem_elev"]
            conditions["terrain_slope"] = selected["dem_slope"]
        return conditions, has_terrain


def _joined_rows(
    alignment_path: Path,
    trajectory_path: Path,
    wanted: Sequence[tuple[str, int]],
) -> Iterator[list[dict[str, object]]]:
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TEMP TABLE wanted(segment_id VARCHAR, sample_ordinal BIGINT)"
        )
        connection.executemany("INSERT INTO wanted VALUES (?, ?)", wanted)
        query = """
            WITH alignment_ranked AS (
                SELECT
                    a.*,
                    row_number() OVER (
                        PARTITION BY source_segment_id ORDER BY source_point_index
                    ) - 1 AS source_segment_point_index,
                    count(*) OVER (PARTITION BY source_segment_id) AS alignment_source_count
                FROM read_json_auto(?) a
            ),
            selected_alignment AS (
                SELECT a.*, w.sample_ordinal
                FROM alignment_ranked a
                JOIN wanted w USING (segment_id)
            ),
            source_ids AS (
                SELECT DISTINCT source_segment_id FROM selected_alignment
            ),
            trajectory_ranked AS (
                SELECT
                    cast(t.file_id AS VARCHAR) AS trajectory_file_id,
                    cast(t.segment_id AS VARCHAR) AS source_segment_id,
                    cast(t.t AS DOUBLE) AS source_time,
                    cast(t.x AS DOUBLE) AS x,
                    cast(t.y AS DOUBLE) AS y,
                    cast(t.region AS VARCHAR) AS region,
                    cast(t.city AS VARCHAR) AS city,
                    row_number() OVER (
                        PARTITION BY cast(t.segment_id AS VARCHAR) ORDER BY t.t
                    ) - 1 AS source_segment_point_index,
                    count(*) OVER (
                        PARTITION BY cast(t.segment_id AS VARCHAR)
                    ) AS trajectory_source_count
                FROM read_parquet(?) t
                JOIN source_ids s
                  ON cast(t.segment_id AS VARCHAR) = s.source_segment_id
            )
            SELECT
                a.sample_ordinal,
                a.segment_id,
                a.file_id,
                a.source_segment_id,
                a.segment_point_index,
                a.source_point_index,
                a.relative_time_s,
                a.absolute_epoch_ns,
                a.alignment_source_count,
                t.trajectory_source_count,
                t.trajectory_file_id,
                t.source_time,
                t.x,
                t.y,
                t.region,
                t.city
            FROM selected_alignment a
            JOIN trajectory_ranked t USING (
                source_segment_id, source_segment_point_index
            )
            ORDER BY a.sample_ordinal, a.segment_point_index
        """
        reader = connection.execute(
            query, [str(alignment_path), str(trajectory_path)]
        ).to_arrow_reader(batch_size=65536)
        current_ordinal: int | None = None
        group: list[dict[str, object]] = []
        for batch in reader:
            for row in batch.to_pylist():
                ordinal = int(row["sample_ordinal"])
                if current_ordinal is not None and ordinal != current_ordinal:
                    yield group
                    group = []
                current_ordinal = ordinal
                group.append(row)
        if group:
            yield group
    except duckdb.Error as error:
        raise PIRC20AdapterError(
            f"cannot join PIRC-20 point identities: {error}"
        ) from error
    finally:
        connection.close()


def _segment(
    sample: PIRC20Sample,
    rows: Sequence[Mapping[str, object]],
    conditions: _ConditionStore,
) -> Segment:
    expected_points = sample.target_end + 1
    if len(rows) != expected_points:
        raise PIRC20AdapterError(f"point join count mismatch: {sample.segment_id}")
    indexes = np.asarray([int(row["segment_point_index"]) for row in rows], dtype=int)
    if not np.array_equal(indexes, np.arange(expected_points)):
        raise PIRC20AdapterError(f"point order mismatch: {sample.segment_id}")
    if any(str(row["segment_id"]) != sample.segment_id for row in rows):
        raise PIRC20AdapterError(f"segment identity mismatch: {sample.segment_id}")
    if any(str(row["file_id"]) != sample.file_id for row in rows):
        raise PIRC20AdapterError(
            f"alignment file identity mismatch: {sample.segment_id}"
        )
    if any(str(row["trajectory_file_id"]) != sample.file_id for row in rows):
        raise PIRC20AdapterError(
            f"trajectory file identity mismatch: {sample.segment_id}"
        )
    if any(
        int(row["alignment_source_count"]) != int(row["trajectory_source_count"])
        for row in rows
    ):
        raise PIRC20AdapterError(f"source segment count mismatch: {sample.segment_id}")

    epoch_ns = np.asarray(
        [int(row["absolute_epoch_ns"]) for row in rows], dtype=np.int64
    )
    if np.any(np.diff(epoch_ns) < 0):
        raise PIRC20AdapterError(
            f"backward time survived refinement: {sample.segment_id}"
        )
    # Exact duplicate timestamps cannot form a positive SDE step.  Keep the first
    # observation at each timestamp so a duplicate spanning the forecast boundary
    # can never import a target state into observed history.
    keep = np.concatenate(([True], np.diff(epoch_ns) > 0))
    history = np.flatnonzero(keep & (indexes <= sample.history_end))
    target = np.flatnonzero(keep & (indexes >= sample.target_start))
    balanced_history = min(len(history), len(target))
    if balanced_history < 2:
        raise _UnusableSample(
            f"exact-time de-duplication leaves an unusable sample: {sample.segment_id}"
        )

    # The legacy NEX326 runner owns a midpoint boundary.  Preserve the published
    # history/target boundary by retaining h history points and h or h+1 target
    # points.  When duplicate removal made a side longer, downsample by position
    # while retaining both endpoints; no timestamp or state is invented.
    def spaced(values: np.ndarray, count: int) -> np.ndarray:
        if len(values) == count:
            return values
        offsets = np.floor(
            np.arange(count, dtype=float) * (len(values) - 1) / (count - 1)
        ).astype(int)
        return values[offsets]

    history = spaced(history, balanced_history)
    target = spaced(target, min(len(target), balanced_history + 1))
    selected_positions = np.concatenate((history, target))
    kept_epoch_ns = epoch_ns[selected_positions]
    time = (kept_epoch_ns - kept_epoch_ns[0]).astype(float) / 1_000_000_000.0
    state = np.column_stack(
        (
            np.asarray([float(row["x"]) for row in rows], dtype=float)[
                selected_positions
            ],
            np.asarray([float(row["y"]) for row in rows], dtype=float)[
                selected_positions
            ],
        )
    )
    condition_indexes = np.asarray(
        [int(row["source_point_index"]) for row in rows], dtype=int
    )[selected_positions]
    feature_values, has_terrain = conditions.select(sample.file_id, condition_indexes)
    source_relative = np.asarray(
        [float(row["source_time"]) for row in rows], dtype=float
    )
    release_relative = np.asarray(
        [float(row["relative_time_s"]) for row in rows], dtype=float
    )
    source_relative -= source_relative[0]
    if not np.allclose(source_relative, release_relative, rtol=1e-9, atol=1e-6):
        raise PIRC20AdapterError(
            f"trajectory/release time mismatch: {sample.segment_id}"
        )
    city = str(rows[0]["city"] or "").strip()
    region = (
        city if city and city.lower() != "nan" else str(rows[0]["region"] or "unknown")
    )
    segment = Segment(
        segment_id=sample.segment_id,
        source_domain="human",
        region=region,
        time=time,
        state=state,
        conditions=feature_values,
        has_terrain=has_terrain,
    )
    segment.validate()
    return segment


def _fingerprint(
    cohort: PIRC20Cohort,
    dataset: Mapping[str, object],
    roles: Mapping[str, str],
    excluded_sample_ids: Sequence[str],
    *,
    final_eval_unlocked: bool,
    maximum_segments_per_role: Mapping[str, int] | None,
    selected_segment_ids: Sequence[str],
) -> str:
    role_digest = hashlib.sha256()
    for segment_id, role in sorted(roles.items()):
        role_digest.update(f"{segment_id}\0{role}\n".encode("utf-8"))
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "adapter_source_sha256": _sha256(Path(__file__)),
        "duckdb_version": duckdb.__version__,
        "cohort_id": cohort.cohort_id,
        "cohort_sha256": _sha256(cohort.path),
        "release_artifacts": dataset["artifacts"],
        "source": dataset["source"],
        "adapt_seed": ADAPT_SEED,
        "adapt_fraction": ADAPT_FRACTION,
        "role_assignment_sha256": role_digest.hexdigest(),
        "exact_time_excluded_sample_ids_sha256": hashlib.sha256(
            "".join(
                f"{sample_id}\n" for sample_id in sorted(excluded_sample_ids)
            ).encode("utf-8")
        ).hexdigest(),
        "exact_time_excluded_sample_count": len(excluded_sample_ids),
        "final_eval_unlocked": final_eval_unlocked,
        "maximum_segments_per_role": dict(
            sorted((maximum_segments_per_role or {}).items())
        ),
        "selected_segment_ids_sha256": hashlib.sha256(
            "".join(
                f"{segment_id}\n" for segment_id in sorted(selected_segment_ids)
            ).encode("utf-8")
        ).hexdigest(),
        "selected_segment_count": len(selected_segment_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_pirc20_nex326_cohort(
    cohort_path: str | Path,
    trajectory_path: str | Path,
    condition_root: str | Path,
    *,
    final_eval_unlock: str | None = None,
    maximum_segments_per_role: Mapping[str, int] | None = None,
) -> Cohort:
    """Load one hash-bound PIRC-20 release as the common NEX326 cohort.

    ``final_eval_unlock`` must equal the frozen cohort ID.  Omitting it validates and
    materializes only train/validation/adapt data, keeping evaluation identities and
    source rows sealed.
    """

    pirc = load_pirc20_cohort(cohort_path)
    if pirc.window.get("mode") != "nex326_midpoint":
        raise PIRC20AdapterError("NEX326 requires the frozen nex326_midpoint window")
    if final_eval_unlock is not None and final_eval_unlock != pirc.cohort_id:
        raise PIRC20AdapterError(
            "final-eval unlock acknowledgement does not match cohort ID"
        )
    final_eval_unlocked = final_eval_unlock == pirc.cohort_id
    trajectory = Path(trajectory_path).resolve()
    condition_source = Path(condition_root).resolve()
    if not trajectory.is_file() or trajectory.is_symlink():
        raise PIRC20AdapterError("trajectory source is missing or symlinked")
    if not condition_source.is_dir():
        raise PIRC20AdapterError("condition source root is missing")
    dataset, alignment_path, condition_manifest = _verify_release(pirc, trajectory)

    samples = tuple(pirc.iter_samples())
    if len({sample.segment_id for sample in samples}) != len(samples):
        raise PIRC20AdapterError("nex326_midpoint must contain one sample per segment")
    roles = _roles(samples, final_eval_unlocked)
    limits = dict(maximum_segments_per_role or {})
    unknown_roles = set(limits) - {"train", "validation", "adapt", "evaluation"}
    if unknown_roles or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in limits.values()
    ):
        raise PIRC20AdapterError(
            "maximum_segments_per_role requires positive integers for known roles"
        )
    eligible = [sample for sample in samples if sample.segment_id in roles]
    if limits:
        selected_ids: set[str] = set()
        for role in ("train", "validation", "adapt", "evaluation"):
            candidates = [
                sample for sample in eligible if roles[sample.segment_id] == role
            ]
            limit = limits.get(role)
            if limit is not None:
                by_block: dict[str, list[PIRC20Sample]] = {}
                for sample in candidates:
                    by_block.setdefault(sample.independent_block_id, []).append(sample)
                for values in by_block.values():
                    values.sort(key=lambda sample: sample.segment_id)
                ordered_blocks = sorted(
                    by_block, key=lambda block: (_rank(block), block)
                )
                candidates = [
                    sample
                    for offset in range(max(map(len, by_block.values()), default=0))
                    for block in ordered_blocks
                    for sample in by_block[block][offset : offset + 1]
                ][:limit]
            selected_ids.update(sample.segment_id for sample in candidates)
        selected = [sample for sample in eligible if sample.segment_id in selected_ids]
    else:
        selected = eligible
    wanted = [(sample.segment_id, ordinal) for ordinal, sample in enumerate(selected)]
    condition_store = _ConditionStore(condition_source, condition_manifest)
    splits: dict[str, list[Segment]] = {
        "train": [],
        "validation": [],
        "adapt": [],
        "evaluation": [],
        "animal_pretrain": [],
    }
    groups = _joined_rows(alignment_path, trajectory, wanted)
    processed = 0
    excluded_sample_ids: list[str] = []
    for sample, rows in zip(selected, groups):
        role = roles[sample.segment_id]
        try:
            splits[role].append(_segment(sample, rows, condition_store))
        except _UnusableSample:
            excluded_sample_ids.append(sample.sample_id)
        processed += 1
    if processed != len(selected):
        raise PIRC20AdapterError("not every selected sample identity was joined")

    unavailable = {
        "animal_pretrain": "PIRC-20 contains no licensed animal pretraining cohort",
        "condition:temperature": "PIRC-20 release has no aligned temperature",
        "condition:precipitation": "PIRC-20 release has no aligned precipitation",
        "endpoint_prior": "PIRC-20 release has no independent endpoint-prior feed",
    }
    if not final_eval_unlocked:
        unavailable["evaluation"] = (
            "PIRC-20 final_eval remains sealed; provide the exact cohort ID acknowledgement"
        )
    if excluded_sample_ids:
        unavailable["adapter:exact_time_exclusions"] = (
            f"{len(excluded_sample_ids)} samples have fewer than two distinct exact "
            "timestamps on one side of the registered midpoint"
        )
    result = Cohort(
        schema_version="nex326-cohort-v1",
        dataset_id=pirc.cohort_id,
        data_version=pirc.data_version,
        purpose="pirc20_frozen_common_22_arm_cohort",
        splits={name: tuple(values) for name, values in splits.items()},
        unavailable_reasons=unavailable,
        fingerprint=_fingerprint(
            pirc,
            dataset,
            roles,
            excluded_sample_ids,
            final_eval_unlocked=final_eval_unlocked,
            maximum_segments_per_role=limits,
            selected_segment_ids=[sample.segment_id for sample in selected],
        ),
    )
    result.validate()
    return result


__all__ = [
    "ADAPTER_VERSION",
    "ADAPT_FRACTION",
    "ADAPT_SEED",
    "PIRC20AdapterError",
    "load_pirc20_nex326_cohort",
]
