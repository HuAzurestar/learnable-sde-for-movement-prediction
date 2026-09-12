"""Materialize a bounded NEX326 pilot cohort from the registered DSDE Zhejiang leg."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PILOT_SEED = 20260814
ZHEJIANG_LATITUDE_DEGREES = 30.1
TIMESTAMP_PATTERN = re.compile(
    r"(20\d{2})\D?(\d{2})\D?(\d{2})\D?(\d{2})\D?(\d{2})\D?(\d{2})"
)


class DSDEPilotError(ValueError):
    """The registered DSDE pilot source cannot satisfy its declared contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ids(payload: Mapping[str, object], name: str) -> tuple[str, ...]:
    splits = payload.get("splits", payload)
    entry = splits.get(name)
    if isinstance(entry, Mapping):
        entry = entry.get("files")
    if not isinstance(entry, list) or not entry:
        raise DSDEPilotError(f"DSDE split {name} has no registered file ids")
    return tuple(str(value) for value in entry)


def _rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _parse_local_start(file_id: str) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(file_id)
    if match is None:
        return None
    try:
        return datetime(*(int(value) for value in match.groups()))
    except ValueError:
        return None


def _solar_elevation(start: datetime, seconds: np.ndarray) -> np.ndarray:
    latitude = math.radians(ZHEJIANG_LATITUDE_DEGREES)
    elevations: list[float] = []
    for offset in seconds:
        local = start + timedelta(seconds=float(offset))
        day = local.timetuple().tm_yday
        hour = local.hour + local.minute / 60.0 + local.second / 3600.0
        declination = math.radians(23.44) * math.sin(
            2.0 * math.pi * (284.0 + day) / 365.0
        )
        hour_angle = math.radians(15.0 * (hour - 12.0))
        sine = (
            math.sin(latitude) * math.sin(declination)
            + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
        )
        elevations.append(math.degrees(math.asin(max(-1.0, min(1.0, sine)))))
    return np.asarray(elevations, dtype=float)


def _segment_payload(frame: pd.DataFrame, split: str) -> dict[str, object] | None:
    ordered = frame.sort_values("t").drop_duplicates("t")
    if len(ordered) < 4:
        return None
    time = ordered["t"].to_numpy(dtype=float)
    state = ordered[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(time).all() or not np.isfinite(state).all() or not np.all(np.diff(time) > 0):
        return None
    file_id = str(ordered["file_id"].iloc[0])
    start = _parse_local_start(file_id)
    if start is None:
        return None
    solar = _solar_elevation(start, time - time[0])
    source_segment_id = str(ordered["segment_id"].iloc[0])
    city = str(ordered["city"].iloc[0]).strip()
    region = city if city and city.lower() != "nan" else str(ordered["region"].iloc[0])
    return {
        "segment_id": f"{split}:{file_id}:{source_segment_id}",
        "source_domain": "human",
        "region": region,
        "time": (time - time[0]).tolist(),
        "state": state.tolist(),
        "conditions": {
            "solar_elev": solar.tolist(),
            "is_day": (solar >= 0.0).astype(float).tolist(),
        },
        "has_terrain": False,
    }


def materialize_dsde_zhejiang_pilot(
    trajectory_path: Path | str,
    splits_path: Path | str,
    output_path: Path | str,
    *,
    sample_fraction: float = 0.2,
    seed: int = PILOT_SEED,
) -> dict[str, object]:
    """Create a deterministic 20%-by-segment pilot without copying source parquet."""
    if not 0.0 < sample_fraction <= 1.0:
        raise DSDEPilotError("sample_fraction must be in (0, 1]")
    trajectory = Path(trajectory_path)
    split_source = Path(splits_path)
    if not trajectory.is_file() or not split_source.is_file():
        raise DSDEPilotError("DSDE trajectory parquet and split registry must both exist")

    split_payload = json.loads(split_source.read_text(encoding="utf-8"))
    finetune = _file_ids(split_payload, "finetune")
    validation = _file_ids(split_payload, "val")
    evaluation = _file_ids(split_payload, "eval")
    if set(finetune) & set(validation) or set(finetune) & set(evaluation) or set(validation) & set(evaluation):
        raise DSDEPilotError("DSDE source splits overlap")

    ranked_finetune = sorted(finetune, key=lambda value: _rank(value, seed))
    cut = max(1, min(len(ranked_finetune) - 1, round(0.8 * len(ranked_finetune))))
    file_splits = {
        "train": set(ranked_finetune[:cut]),
        "adapt": set(ranked_finetune[cut:]),
        "validation": set(validation),
        "evaluation": set(evaluation),
    }
    if any(not values for values in file_splits.values()):
        raise DSDEPilotError("DSDE pilot requires non-empty train/validation/adapt/evaluation files")

    columns = ["file_id", "segment_id", "t", "x", "y", "region", "city"]
    frame = pd.read_parquet(trajectory, columns=columns)
    registered_files = set().union(*file_splits.values())
    frame = frame[frame["file_id"].isin(registered_files)]
    observed_files = set(str(value) for value in frame["file_id"].unique())
    absent_files = registered_files - observed_files
    if absent_files:
        raise DSDEPilotError(f"registered DSDE files are absent from parquet: {len(absent_files)}")

    materialized: dict[str, list[dict[str, object]]] = {}
    source_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    for split, file_ids in file_splits.items():
        candidates: list[dict[str, object]] = []
        subset = frame[frame["file_id"].isin(file_ids)]
        grouped = subset.groupby(["file_id", "segment_id"], sort=False)
        for _, segment_frame in grouped:
            segment = _segment_payload(segment_frame, split)
            if segment is not None:
                candidates.append(segment)
        source_counts[split] = len(candidates)
        count = max(1, math.ceil(len(candidates) * sample_fraction))
        candidates.sort(key=lambda item: _rank(str(item["segment_id"]), seed))
        materialized[split] = candidates[:count]
        skipped_counts[split] = grouped.ngroups - len(candidates)
        if not materialized[split]:
            raise DSDEPilotError(f"no usable DSDE segments remain in split {split}")

    materialized["animal_pretrain"] = []
    trajectory_sha256 = _sha256(trajectory)
    split_sha256 = _sha256(split_source)
    fraction_percent = round(sample_fraction * 100)
    payload: dict[str, object] = {
        "schema_version": "nex326-cohort-v1",
        "dataset_id": f"NEX326-DSDE-ZHEJIANG-{fraction_percent}P-PILOT-001",
        "data_version": (
            f"nex313-zhejiang-{trajectory_sha256[:12]}-{fraction_percent}pct-pilot-v1"
        ),
        "purpose": (
            f"real_trajectory_{fraction_percent}pct_pilot_not_final_scientific_evidence"
        ),
        "source": {
            "registry": "DSDE NEX-313 Zhejiang holdout",
            "trajectory_file": trajectory.name,
            "trajectory_sha256": trajectory_sha256,
            "splits_file": split_source.name,
            "splits_sha256": split_sha256,
            "license": "ODbL-derived research data; not embedded in the repository",
        },
        "selection": {
            "seed": seed,
            "sample_fraction": sample_fraction,
            "unit": "segment_within_file_disjoint_split",
            "meta_task_unit": "city_with_region_fallback",
            "finetune_partition": "80_percent_train_20_percent_adapt_by_seeded_file_hash",
            "source_usable_segments": source_counts,
            "materialized_segments": {
                split: len(segments) for split, segments in materialized.items()
            },
            "skipped_unusable_segments": skipped_counts,
        },
        "feature_provenance": {
            "solar_elev": "derived from file-local timestamp using Zhejiang latitude 30.1 degrees; pilot approximation",
            "is_day": "solar_elev >= 0",
        },
        "feature_status": {
            "temperature": {"status": "unavailable", "reason": "DSDE Zhejiang pilot has no aligned weather"},
            "precipitation": {"status": "unavailable", "reason": "DSDE Zhejiang pilot has no aligned weather"},
            "terrain_elevation": {"status": "unavailable", "reason": "DSDE Zhejiang pilot has no aligned terrain"},
            "terrain_slope": {"status": "unavailable", "reason": "DSDE Zhejiang pilot has no aligned terrain"},
        },
        "endpoint_prior_status": {
            "status": "unavailable",
            "reason": "DSDE Zhejiang pilot has no independent endpoint-prior feed",
        },
        "split_status": {
            "animal_pretrain": {
                "status": "unavailable",
                "reason": "DSDE Zhejiang holdout contains human trajectories only",
            }
        },
        "splits": materialized,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    args = parser.parse_args(argv)
    payload = materialize_dsde_zhejiang_pilot(
        args.trajectory,
        args.splits,
        args.output,
        sample_fraction=args.fraction,
        seed=args.seed,
    )
    print(json.dumps({"dataset_id": payload["dataset_id"], **payload["selection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DSDEPilotError", "materialize_dsde_zhejiang_pilot"]
