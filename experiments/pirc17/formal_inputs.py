"""Guarded final position loading with disjoint causal-input and truth objects.

No final sample is relabelled as development. Predictors receive FinalPrefix,
never the loader's ScoringTargets or a full future route. Training, secondary
origin construction, feature/map loading and domain execution are separate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import final_eval_guard as guard
from .features import LocalFrame
from .formal_eligibility import (NS, POINT_FIELDS, RESULT_VERSION, SAMPLE_FIELDS, SCORE_NS,
                                 WINDOW_VERSION, _bound_file, _release_sources)
from .method_inputs import SolarConditionField, _elapsed, _epochs, _lonlat
from .origins import Origin, causal_prefix, frozen_array
from .protocol_core import decode, digest, read_json, sha256, under, unpack


@dataclass(frozen=True)
class FinalPrefix:
    sample_id: str
    independent_block_id: str
    split: str
    window_sha256: str
    population_sha256: str
    condition_sha256: str
    origin_epoch_ns: int
    terrain_origin: Origin
    method_origin: Origin
    scoring_frame: LocalFrame
    condition_at: SolarConditionField
    score_seconds: np.ndarray

    def identity(self):
        return {"schema_version": "pirc17-final-causal-prefix-v1", "sample_id": self.sample_id,
            "independent_block_id": self.independent_block_id, "split": self.split,
            "window_sha256": self.window_sha256, "population_sha256": self.population_sha256,
            "condition_sha256": self.condition_sha256, "origin_epoch_ns": self.origin_epoch_ns,
            "score_seconds": self.score_seconds.tolist(), "solar": self.condition_at.identity(),
            "scoring_frame": [self.scoring_frame.longitude, self.scoring_frame.latitude],
            "terrain_prefix_positions_m": self.terrain_origin.history_positions_m.tolist(),
            "method_prefix_positions_m": self.method_origin.history_positions_m.tolist(),
            "prefix_times_seconds": self.terrain_origin.history_times_seconds.tolist()}

    def to_scoring_frame(self, method_positions_m):
        return frozen_array(self.scoring_frame.from_lonlat(self.condition_at.frame.to_lonlat(method_positions_m)))


@dataclass(frozen=True)
class ScoringTargets:
    sample_id: str
    window_sha256: str
    elapsed_seconds: np.ndarray
    positions_m: np.ndarray


@dataclass(frozen=True)
class FinalPositionInputs:
    """Owner-side bundle; never pass this bundle to a prediction function."""
    prefixes: tuple[FinalPrefix, ...]
    targets: tuple[ScoringTargets, ...]
    population_sha256: str
    access_started_sha256: str


def validate_window(window, *, expected_sha256=None):
    p = unpack(window, expected_sha256=expected_sha256)
    if (set(p) != {"schema_version", "input_binding_sha256", "eligibility_rule_sha256", "sample",
                   "alignment_validity_sha256", "prefix", "targets", "origin_epoch_ns", "score_elapsed_ns"}
            or p["schema_version"] != WINDOW_VERSION or set(p["sample"]) != set(SAMPLE_FIELDS)
            or p["sample"]["split"] != "final_eval"):
        raise ValueError("exact qualified final window required")
    for name in ("input_binding_sha256", "eligibility_rule_sha256", "alignment_validity_sha256"):
        sha256(p[name])
    s = p["sample"]
    sha256(s["sample_id"])
    bounds = [s[k] for k in ("history_start", "history_end", "target_start", "target_end")]
    if (any(type(x) is not int for x in bounds) or not 0 <= bounds[0] < bounds[1] < bounds[2] <= bounds[3]
            or bounds[2] != bounds[1]+1 or not isinstance(p["prefix"], list) or not isinstance(p["targets"], list)):
        raise ValueError("qualified history/target bounds required")
    points = p["prefix"]+p["targets"]
    if (any(set(r) != set(POINT_FIELDS) or any(type(r[k]) is not int for k in POINT_FIELDS)
            or r["source_point_index"] < 0 for r in points)
            or len({r["source_point_index"] for r in points}) != len(points)):
        raise ValueError("unique original point identities required")
    if ([r["segment_point_index"] for r in p["prefix"]] != list(range(bounds[0], bounds[1]+1))
            or len(p["targets"]) != 4 or len(p["score_elapsed_ns"]) != 4):
        raise ValueError("complete visible prefix and four original targets required")
    epochs = _epochs([r["absolute_epoch_ns"] for r in points])
    prefix_epochs = epochs[:len(p["prefix"])]
    if type(p["origin_epoch_ns"]) is not int or p["origin_epoch_ns"] != int(prefix_epochs[-1]):
        raise ValueError("explicit original UTC origin required")
    gaps = [int(b)-int(a) for a, b in zip(prefix_epochs, prefix_epochs[1:])]
    if any(not 0 < gap <= 60*NS for gap in gaps):
        raise ValueError("invalid visible observation clock")
    indexes = [r["segment_point_index"] for r in p["targets"]]
    elapsed = [int(t)-p["origin_epoch_ns"] for t in epochs[-4:]]
    if (any(not bounds[2] <= i <= bounds[3] for i in indexes) or any(a >= b for a, b in zip(indexes, indexes[1:]))
            or any(type(t) is not int for t in p["score_elapsed_ns"]) or elapsed != p["score_elapsed_ns"]
            or any(t <= 0 or abs(t-nominal) > 30*NS for t, nominal in zip(elapsed, SCORE_NS))
            or any(a >= b for a, b in zip(elapsed, elapsed[1:]))):
        raise ValueError("original observed scoring clock differs from qualification")
    return p


def bind_final_prefix(window, visible_lonlat, *, population_sha256, condition_sha256):
    """Pure constructor accepting only visible coordinates, never target positions."""
    p = validate_window(window)
    sha256(population_sha256); sha256(condition_sha256)
    epochs = _epochs([r["absolute_epoch_ns"] for r in p["prefix"]])
    lonlat = _lonlat(visible_lonlat, len(epochs))
    scoring_frame, method_frame = LocalFrame(*lonlat[-1]), LocalFrame(*lonlat[0])
    times = _elapsed(epochs[-3:], p["origin_epoch_ns"])
    terrain_origin = causal_prefix(scoring_frame.from_lonlat(lonlat[-3:]), times)
    method_origin = causal_prefix(method_frame.from_lonlat(lonlat[-3:]), times)
    return FinalPrefix(p["sample"]["sample_id"], p["sample"]["independent_block_id"], "final_eval",
        window["sha256"], population_sha256, condition_sha256, p["origin_epoch_ns"], terrain_origin,
        method_origin, scoring_frame, SolarConditionField(method_frame, p["origin_epoch_ns"]),
        frozen_array(np.array(p["score_elapsed_ns"], dtype=np.float64)/NS))


def _positions(access, *, protocol, execution, population_path, population_sha256, eligibility_path,
               qualification_path, qualification_sha256, release, data_root):
    import pyarrow.parquet as pq

    sealed = unpack(protocol)
    binding = sealed["dataset_inputs"]
    if access.access_kind != "final_eval_positions" or access.population_sha256 != population_sha256:
        raise ValueError("verified population-bound position access required")
    # Read and validate artifacts again in the consumer; never trust a caller's
    # selected IDs, window directory or arbitrary ordering.
    report, population = read_json(eligibility_path), read_json(population_path)
    pp = guard.validate_population(population, report, expected_sha256=population_sha256,
                                    protocol=protocol, execution=execution)
    q = unpack(read_json(qualification_path), expected_sha256=qualification_sha256)
    root = Path(qualification_path).parent
    if (q["schema_version"] != RESULT_VERSION or q["protocol_sha256"] != protocol["sha256"]
            or q["execution_sha256"] != execution["sha256"] or q["approval_sha256"] != access.approval_sha256
            or q["population_sha256"] != population_sha256 or q["eligibility_sha256"] != report["sha256"]
            or under(root, q["eligibility_path"]).resolve() != Path(eligibility_path).resolve()
            or under(root, q["population_path"]).resolve() != Path(population_path).resolve()):
        raise ValueError("qualification artifacts do not bind this approved population")
    dispositions = {r["sample_id"]: r for r in unpack(report)["rows"]}
    expected_windows = {k: "windows/"+r["window_sha256"]+".json" for k, r in dispositions.items() if r["eligible"]}
    if q["windows"] != expected_windows:
        raise ValueError("qualification window inventory differs from full report")
    _, samples = _release_sources(release, binding)
    by_id = {s["sample_id"]: s for s in samples}
    selected = pp["selection"]["selected"]
    windows, by_file = {}, defaultdict(list)
    for row in selected:
        sid = row["sample_id"]
        window = read_json(under(root, q["windows"][sid]))
        w = validate_window(window, expected_sha256=dispositions[sid]["window_sha256"])
        if (w["sample"] != by_id[sid] or w["input_binding_sha256"] != binding["sha256"]
                or w["eligibility_rule_sha256"] != digest(sealed["eligibility_contract"])):
            raise ValueError("qualified window no longer matches original sample/input identity")
        windows[sid] = window
        by_file[w["sample"]["file_id"]].append(sid)
    manifest_path = _bound_file(release, "condition_file_manifest.jsonl",
                                binding["release_artifact_sha256"]["condition_file_manifest.jsonl"])
    conditions = {}
    with manifest_path.open("rb") as stream:
        while raw := stream.readline(16385):
            if len(raw) > 16384:
                raise ValueError("condition manifest row exceeds bound")
            row = decode(raw)
            file_id = row["file_id"]
            if not isinstance(file_id, str) or not file_id or file_id in conditions:
                raise ValueError("duplicate or invalid condition identity")
            sha256(row["sha256"])
            conditions[file_id] = row
    prefixes, targets = {}, {}
    for file_id, sample_ids in by_file.items():
        if file_id not in conditions:
            raise ValueError("selected condition source absent; no replacement")
        source = conditions[file_id]
        path = _bound_file(Path(data_root)/"cond_slices", source["relative_path"], source["sha256"])
        # Numeric terrain/solar columns and the historical future-centered
        # trajectory representation are deliberately not prediction inputs.
        table = pq.read_table(path, columns=["file_id", "t", "lon", "lat"])
        if any(str(x) != file_id for x in table["file_id"].to_pylist()):
            raise ValueError("condition rows have another source identity")
        source_times = table["t"].to_numpy()
        if source_times.dtype.kind != "M":
            raise ValueError("condition source must carry exact datetime UTC timestamps")
        source_epochs = source_times.astype("datetime64[ns]").astype(np.int64)
        lonlat = np.column_stack((table["lon"].to_numpy(), table["lat"].to_numpy()))
        for sid in sample_ids:
            w = unpack(windows[sid])
            prefix_indexes = [r["source_point_index"] for r in w["prefix"]]
            target_indexes = [r["source_point_index"] for r in w["targets"]]
            if max(prefix_indexes+target_indexes) >= len(lonlat):
                raise ValueError("original source point out of range; no interpolation or replacement")
            indexes = prefix_indexes+target_indexes
            expected_epochs = _epochs([r["absolute_epoch_ns"] for r in w["prefix"]+w["targets"]])
            if not np.array_equal(_epochs(source_epochs[indexes]), expected_epochs):
                raise ValueError("exact source UTC timestamp join differs from qualified window")
            prefix = bind_final_prefix(windows[sid], lonlat[prefix_indexes],
                                       population_sha256=population_sha256, condition_sha256=source["sha256"])
            truth = _lonlat(lonlat[target_indexes], 4)
            prefixes[sid] = prefix
            targets[sid] = ScoringTargets(sid, windows[sid]["sha256"], frozen_array(prefix.score_seconds),
                                           frozen_array(prefix.scoring_frame.from_lonlat(truth)))
    return FinalPositionInputs(tuple(prefixes[r["sample_id"]] for r in selected),
        tuple(targets[r["sample_id"]] for r in selected), population_sha256, access.access_started_sha256)


def load_final_positions(*, protocol, execution, approval_path, approval_sha256, test_path, review_path,
                         journal_directory, population_path, population_sha256, eligibility_path,
                         qualification_path, qualification_sha256, release, data_root):
    return guard.guarded_call(access_kind="final_eval_positions", protocol=protocol, execution=execution,
        approval_path=approval_path, approval_sha256=approval_sha256, test_path=test_path, review_path=review_path,
        journal_directory=journal_directory, population_path=population_path,
        population_sha256=population_sha256, eligibility_path=eligibility_path,
        operation=lambda access: _positions(access, protocol=protocol, execution=execution,
            population_path=population_path, population_sha256=population_sha256, eligibility_path=eligibility_path,
            qualification_path=qualification_path, qualification_sha256=qualification_sha256,
            release=release, data_root=data_root))
