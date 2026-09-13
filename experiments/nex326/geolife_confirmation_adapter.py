"""Bind a frozen user-partitioned GeoLife release to the NEX326 runtime."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import duckdb
import numpy as np

from data.pirc20 import PIRC20Cohort, PIRC20Sample, load_pirc20_cohort

from .cohort import Cohort, Segment


ADAPTER_VERSION = "pirc20-geolife-nex326-adapter-v1"
ADAPT_SEED = 20260912
ADAPT_FRACTION = 0.20


class GeoLifeConfirmationAdapterError(ValueError):
    """The GeoLife release cannot satisfy the NEX326 runtime contract."""


class _UnusableSample(GeoLifeConfirmationAdapterError):
    """A sample has too few distinct timestamps around its frozen midpoint."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeoLifeConfirmationAdapterError(
            f"cannot read release artifact: {path.name}"
        ) from error
    if not isinstance(payload, dict):
        raise GeoLifeConfirmationAdapterError(
            f"release artifact is not an object: {path.name}"
        )
    return payload


def _rank(block_id: str) -> str:
    return hashlib.sha256(f"{ADAPT_SEED}:{block_id}".encode("ascii")).hexdigest()


def _roles(samples: Sequence[PIRC20Sample], unlocked: bool) -> dict[str, str]:
    train_blocks = sorted(
        {sample.independent_block_id for sample in samples if sample.split == "train"},
        key=_rank,
    )
    if len(train_blocks) < 2:
        raise GeoLifeConfirmationAdapterError(
            "outer train needs at least two independent users"
        )
    fit_count = max(
        1,
        min(
            len(train_blocks) - 1,
            round((1.0 - ADAPT_FRACTION) * len(train_blocks)),
        ),
    )
    fit_blocks = set(train_blocks[:fit_count])
    result: dict[str, str] = {}
    for sample in samples:
        if sample.split == "train":
            result[sample.segment_id] = (
                "train" if sample.independent_block_id in fit_blocks else "adapt"
            )
        elif sample.split == "validation":
            result[sample.segment_id] = "validation"
        elif unlocked:
            result[sample.segment_id] = "evaluation"
    return result


def _verify_release(
    cohort: PIRC20Cohort, trajectory: Path
) -> dict[str, object]:
    root = cohort.path.parent
    dataset_path = root / "dataset.json"
    if not dataset_path.is_file() or dataset_path.is_symlink():
        raise GeoLifeConfirmationAdapterError("dataset.json is missing or symlinked")
    dataset = _load(dataset_path)
    if (
        dataset.get("schema_version") != "pirc20-geolife-release-v1"
        or dataset.get("dataset_id") != cohort.cohort_id
    ):
        raise GeoLifeConfirmationAdapterError("unsupported GeoLife release identity")
    artifacts = dataset.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise GeoLifeConfirmationAdapterError("release artifact registry is missing")
    for name in (
        "cohort.json",
        "samples.jsonl",
        "split.json",
        "coverage_report.json",
        "leakage_report.json",
    ):
        entry = artifacts.get(name)
        path = root / name
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("sha256"), str)
            or not path.is_file()
            or path.is_symlink()
            or _sha256(path) != entry["sha256"]
        ):
            raise GeoLifeConfirmationAdapterError(
                f"release artifact hash mismatch: {name}"
            )
    source = dataset.get("source")
    source_trajectory = source.get("trajectory") if isinstance(source, Mapping) else None
    expected = (
        source_trajectory.get("sha256")
        if isinstance(source_trajectory, Mapping)
        else None
    )
    if not isinstance(expected, str) or _sha256(trajectory) != expected:
        raise GeoLifeConfirmationAdapterError("GeoLife trajectory hash mismatch")
    return dataset


def _verify_conditions(
    condition_path: Path, trajectory: Path
) -> tuple[dict[str, object], Path]:
    receipt_path = condition_path.resolve()
    if (
        receipt_path.name != "receipt.json"
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        raise GeoLifeConfirmationAdapterError(
            "GeoLife conditions must name an explicit receipt.json"
        )
    receipt = _load(receipt_path)
    if receipt.get("schema_version") != "pirc20-geolife-solar-conditions-v1":
        raise GeoLifeConfirmationAdapterError("unsupported GeoLife condition receipt")
    source = receipt.get("source")
    artifact = receipt.get("artifact")
    if (
        not isinstance(source, Mapping)
        or source.get("cleaned_trajectory_sha256") != _sha256(trajectory)
        or not isinstance(artifact, Mapping)
    ):
        raise GeoLifeConfirmationAdapterError(
            "GeoLife conditions are not bound to the trajectory"
        )
    artifact_name = str(artifact.get("path", ""))
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise GeoLifeConfirmationAdapterError(
            "GeoLife condition artifact name is unsafe"
        )
    sidecar = receipt_path.parent / artifact_name
    if (
        not sidecar.is_file()
        or sidecar.is_symlink()
        or _sha256(sidecar) != artifact.get("sha256")
    ):
        raise GeoLifeConfirmationAdapterError(
            "GeoLife condition artifact hash mismatch"
        )
    return receipt, sidecar


def _joined_rows(
    trajectory: Path,
    condition_path: Path,
    wanted: Sequence[tuple[str, int]],
) -> Iterator[list[dict[str, object]]]:
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TEMP TABLE wanted(source_segment_id VARCHAR, sample_ordinal BIGINT)"
        )
        connection.executemany("INSERT INTO wanted VALUES (?, ?)", wanted)
        reader = connection.execute(
            """
            WITH trajectory_ranked AS (
                SELECT
                    cast(t.file_id AS VARCHAR) AS file_id,
                    cast(t.segment_id AS VARCHAR) AS source_segment_id,
                    cast(t.t AS DOUBLE) AS time,
                    cast(t.x AS DOUBLE) AS x,
                    cast(t.y AS DOUBLE) AS y,
                    cast(t.region AS VARCHAR) AS region,
                    cast(t.city AS VARCHAR) AS city,
                    row_number() OVER (
                        PARTITION BY cast(t.segment_id AS VARCHAR) ORDER BY t.t
                    ) - 1 AS point_index
                FROM read_parquet(?) t
            )
            SELECT
                w.sample_ordinal,
                t.file_id,
                t.source_segment_id,
                t.time,
                t.x,
                t.y,
                t.region,
                t.city,
                t.point_index,
                cast(c.solar_elev AS DOUBLE) AS solar_elev,
                cast(c.is_day AS DOUBLE) AS is_day
            FROM trajectory_ranked t
            JOIN wanted w
              ON t.source_segment_id = w.source_segment_id
            JOIN read_parquet(?) c
              ON cast(c.segment_id AS VARCHAR) = t.source_segment_id
             AND cast(c.file_id AS VARCHAR) = t.file_id
             AND cast(c.point_index AS BIGINT) = t.point_index
            ORDER BY w.sample_ordinal, point_index
            """,
            [str(trajectory), str(condition_path)],
        ).to_arrow_reader(batch_size=65536)
        ordinal: int | None = None
        group: list[dict[str, object]] = []
        for batch in reader:
            for row in batch.to_pylist():
                next_ordinal = int(row["sample_ordinal"])
                if ordinal is not None and ordinal != next_ordinal:
                    yield group
                    group = []
                ordinal = next_ordinal
                group.append(row)
        if group:
            yield group
    except duckdb.Error as error:
        raise GeoLifeConfirmationAdapterError(
            f"cannot join GeoLife segment identities: {error}"
        ) from error
    finally:
        connection.close()


def _spaced(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) == count:
        return values
    offsets = np.floor(
        np.arange(count, dtype=float) * (len(values) - 1) / (count - 1)
    ).astype(int)
    return values[offsets]


def _segment(
    sample: PIRC20Sample, rows: Sequence[Mapping[str, object]]
) -> Segment:
    expected_points = sample.target_end + 1
    if len(rows) != expected_points:
        raise GeoLifeConfirmationAdapterError(
            f"source point count mismatch: {sample.segment_id}"
        )
    if any(str(row["file_id"]) != sample.file_id for row in rows):
        raise GeoLifeConfirmationAdapterError(
            f"source file identity mismatch: {sample.segment_id}"
        )
    indexes = np.asarray([int(row["point_index"]) for row in rows], dtype=int)
    if not np.array_equal(indexes, np.arange(expected_points)):
        raise GeoLifeConfirmationAdapterError(
            f"source point order mismatch: {sample.segment_id}"
        )
    times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    keep = np.concatenate(([True], np.diff(times) > 0))
    history = np.flatnonzero(keep & (indexes <= sample.history_end))
    target = np.flatnonzero(keep & (indexes >= sample.target_start))
    balanced = min(len(history), len(target))
    if balanced < 2:
        raise _UnusableSample(
            f"timestamp de-duplication leaves unusable midpoint: {sample.segment_id}"
        )
    history = _spaced(history, balanced)
    target = _spaced(target, min(len(target), balanced + 1))
    selected = np.concatenate((history, target))
    selected_time = times[selected] - times[selected][0]
    state = np.column_stack(
        (
            np.asarray([float(row["x"]) for row in rows], dtype=float)[selected],
            np.asarray([float(row["y"]) for row in rows], dtype=float)[selected],
        )
    )
    city = str(rows[0]["city"] or "").strip()
    region = city if city and city.lower() != "nan" else str(rows[0]["region"] or "unknown")
    solar_elevation = np.asarray(
        [float(row["solar_elev"]) for row in rows], dtype=float
    )[selected]
    is_day = np.asarray([float(row["is_day"]) for row in rows], dtype=float)[
        selected
    ]
    if not np.isfinite(solar_elevation).all() or not np.isfinite(is_day).all():
        raise GeoLifeConfirmationAdapterError(
            f"non-finite solar conditions: {sample.segment_id}"
        )
    result = Segment(
        segment_id=sample.segment_id,
        source_domain="human",
        region=region,
        time=selected_time,
        state=state,
        conditions={"solar_elev": solar_elevation, "is_day": is_day},
        has_terrain=False,
    )
    result.validate()
    return result


def _fingerprint(
    cohort: PIRC20Cohort,
    dataset: Mapping[str, object],
    condition_receipt: Mapping[str, object],
    roles: Mapping[str, str],
    excluded: Sequence[str],
    *,
    unlocked: bool,
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
        "condition_receipt": condition_receipt,
        "adapt_seed": ADAPT_SEED,
        "adapt_fraction": ADAPT_FRACTION,
        "role_assignment_sha256": role_digest.hexdigest(),
        "timestamp_excluded_sample_ids_sha256": hashlib.sha256(
            "".join(f"{item}\n" for item in sorted(excluded)).encode("ascii")
        ).hexdigest(),
        "timestamp_excluded_sample_count": len(excluded),
        "final_eval_unlocked": unlocked,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_geolife_confirmation_cohort(
    cohort_path: str | Path,
    trajectory_path: str | Path,
    condition_receipt_path: str | Path,
    *,
    final_eval_unlock: str | None = None,
) -> Cohort:
    """Load GeoLife while preserving its frozen user and midpoint boundaries."""

    pirc = load_pirc20_cohort(cohort_path)
    if pirc.window.get("mode") != "nex326_midpoint":
        raise GeoLifeConfirmationAdapterError("GeoLife confirmation requires midpoint windows")
    if final_eval_unlock is not None and final_eval_unlock != pirc.cohort_id:
        raise GeoLifeConfirmationAdapterError("final-eval acknowledgement does not match cohort ID")
    unlocked = final_eval_unlock == pirc.cohort_id
    trajectory = Path(trajectory_path).resolve()
    if not trajectory.is_file() or trajectory.is_symlink():
        raise GeoLifeConfirmationAdapterError("GeoLife trajectory is missing or symlinked")
    dataset = _verify_release(pirc, trajectory)
    condition_receipt, condition_path = _verify_conditions(
        Path(condition_receipt_path), trajectory
    )
    samples = tuple(pirc.iter_samples())
    if len({sample.segment_id for sample in samples}) != len(samples):
        raise GeoLifeConfirmationAdapterError("midpoint cohort must contain one sample per segment")
    roles = _roles(samples, unlocked)
    selected = [sample for sample in samples if sample.segment_id in roles]
    wanted = [
        (sample.segment_id.removeprefix("geolife:"), ordinal)
        for ordinal, sample in enumerate(selected)
    ]
    if any(not sample.segment_id.startswith("geolife:") for sample in selected):
        raise GeoLifeConfirmationAdapterError("GeoLife segment namespace is invalid")
    splits: dict[str, list[Segment]] = defaultdict(list)
    groups = _joined_rows(trajectory, condition_path, wanted)
    processed = 0
    excluded: list[str] = []
    for sample, rows in zip(selected, groups):
        try:
            splits[roles[sample.segment_id]].append(_segment(sample, rows))
        except _UnusableSample:
            excluded.append(sample.sample_id)
        processed += 1
    if processed != len(selected):
        raise GeoLifeConfirmationAdapterError("not every selected GeoLife sample was joined")
    unavailable = {
        "animal_pretrain": "GeoLife contains no licensed animal pretraining cohort",
        "condition:temperature": "GeoLife release has no aligned temperature",
        "condition:precipitation": "GeoLife release has no aligned precipitation",
        "condition:terrain_elevation": "GeoLife release has no aligned DEM elevation",
        "condition:terrain_slope": "GeoLife release has no aligned DEM slope",
        "endpoint_prior": "GeoLife release has no independent endpoint-prior feed",
    }
    if not unlocked:
        unavailable["evaluation"] = (
            "GeoLife final_eval remains sealed; provide the exact cohort ID acknowledgement"
        )
    if excluded:
        unavailable["adapter:timestamp_exclusions"] = (
            f"{len(excluded)} samples have fewer than two distinct timestamps on one midpoint side"
        )
    result = Cohort(
        schema_version="nex326-cohort-v1",
        dataset_id=pirc.cohort_id,
        data_version=pirc.data_version,
        purpose="pirc20_external_domain_independent_confirmation",
        splits={name: tuple(splits[name]) for name in ("train", "validation", "adapt", "evaluation", "animal_pretrain")},
        unavailable_reasons=unavailable,
        fingerprint=_fingerprint(
            pirc,
            dataset,
            condition_receipt,
            roles,
            excluded,
            unlocked=unlocked,
        ),
    )
    result.validate()
    return result


__all__ = [
    "ADAPTER_VERSION",
    "GeoLifeConfirmationAdapterError",
    "load_geolife_confirmation_cohort",
]
