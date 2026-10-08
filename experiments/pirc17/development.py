"""Hash-bound existing train/validation origins for a bounded model pilot.

One first-future transition per qualified origin is used to fit/check dynamics.
Long-horizon targets are kept separate and never passed to rollout callbacks.
This pilot population is not implicitly the final production training protocol.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq

from .calibration import observed_indices
from .configurations import subset_matrix
from .dynamics import TransitionRows
from .features import LocalFrame
from .origins import Origin, causal_prefix, frozen_array
from .qualification import _hash
from .rollout import PredictedState


@dataclass(frozen=True)
class DevelopmentWindow:
    sample_id: str
    block_id: str
    role: str
    origin: Origin
    frame: LocalFrame
    horizon_seconds: np.ndarray
    target_positions_m: np.ndarray
    transition_displacement_m: np.ndarray
    transition_seconds: float
    canonical_origin: dict


def window_from_rows(sample, metadata, source_lonlat):
    """Pure conversion from already verified rows; source indices are not segment offsets."""
    if sample["split"] not in {"train","validation"}:
        raise ValueError("development loader forbids final-eval")
    rows=sorted(metadata,key=lambda r:r["point_index"])
    if len(rows)!=sample["target_end"]+1 or sample["target_start"]!=sample["history_end"]+1:
        raise ValueError("registered segment/sample boundaries differ")
    if any(r["split"]!=sample["split"] or r["independent_block_id"]!=sample["independent_block_id"] for r in rows):
        raise ValueError("row role/block identity differs")
    epoch=rows[sample["history_end"]]["absolute_epoch_ns"]
    prefix=rows[max(sample["history_start"],sample["history_end"]-2):sample["history_end"]+1]
    future=rows[sample["target_start"]:sample["target_end"]+1]
    actual=np.array([(r["absolute_epoch_ns"]-epoch)/1e9 for r in future])
    indexes=observed_indices(actual)
    prefix_times=np.array([(r["absolute_epoch_ns"]-epoch)/1e9 for r in prefix])
    if len(prefix)<2 or actual[0]>60 or np.any(np.diff(actual)>60):
        raise ValueError("unsupported observed time gap")
    chosen=prefix+[future[0]]+[future[i] for i in indexes]
    positions=np.asarray(source_lonlat)[[r["point_index"] for r in chosen]]
    frame=LocalFrame(*positions[len(prefix)-1])
    xy=frame.from_lonlat(positions)
    origin=causal_prefix(xy[:len(prefix)],prefix_times)
    return DevelopmentWindow(sample["sample_id"],sample["independent_block_id"],sample["split"],origin,frame,
        frozen_array(actual[indexes]),frozen_array(xy[len(prefix)+1:]),
        frozen_array(xy[len(prefix)]-origin.position_m),float(actual[0]),dict(prefix[-1]))


def load_development(eligibility_path, eligibility_sha256, release, data_root, encoder):
    if _hash(eligibility_path)!=eligibility_sha256:
        raise ValueError("eligibility artifact hash mismatch")
    audit=json.loads(eligibility_path.read_text(encoding="utf-8"))
    dataset=json.loads((release/"dataset.json").read_text(encoding="utf-8"))
    if not audit["dataset_id"]==dataset["dataset_id"]==encoder.adapter.dataset_id:
        raise ValueError("dataset identity mismatch")
    if audit["snapshot"]["content_inventory_sha256"]!=encoder.adapter.manifest["content_inventory_sha256"]:
        raise ValueError("snapshot differs from eligibility")
    for name in ("cohort.json","condition_file_manifest.jsonl"):
        if _hash(release/name)!=dataset["artifacts"][name]["sha256"]:
            raise ValueError("release artifact changed")
    if _hash(release/"samples.jsonl")!=audit["inputs_sha256"]["samples.jsonl"]:
        raise ValueError("sample manifest changed")
    if audit["eligibility"]["horizon_minutes"]!=30:
        raise ValueError("pilot requires the qualified 30-minute cohort")
    with (release/"samples.jsonl").open(encoding="utf-8") as source:
        samples={s["sample_id"]:s for s in map(json.loads,source)}
    with (release/"condition_file_manifest.jsonl").open(encoding="utf-8") as source:
        conditions={r["file_id"]:r for r in map(json.loads,source)}
    selected=defaultdict(list)
    seen=set()
    for eligible in audit["eligibility"]["rows"]:
        sample=samples[eligible["sample_id"]]
        if sample["split"] not in {"train","validation"} or sample["sample_id"] in seen:
            raise ValueError("pilot admits unique development samples only")
        for key in ("split","file_id","segment_id","independent_block_id","history_start","history_end","target_start","target_end"):
            if eligible[key]!=sample[key]:
                raise ValueError("eligibility/sample mismatch")
        seen.add(sample["sample_id"])
        selected[sample["split"],sample["file_id"]].append(sample)
    windows={"train":[],"validation":[]}
    parents=set()
    sources={}
    for (role,file_id),group in sorted(selected.items()):
        table=encoder.adapter._load(role,final_eval_unlock=None,file_ids=[file_id],segment_ids=[s["segment_id"] for s in group])
        by_segment=defaultdict(list)
        for row in table.to_pylist():
            by_segment[row["segment_id"]].append(row)
        entry=conditions[file_id]
        path=(data_root/"cond_slices"/entry["relative_path"]).resolve()
        if not path.is_relative_to((data_root/"cond_slices").resolve()) or _hash(path)!=entry["sha256"]:
            raise ValueError("condition locator/hash mismatch")
        source=pq.read_table(path,columns=["lon","lat"])
        lonlat=np.column_stack((source["lon"].to_numpy(),source["lat"].to_numpy()))
        for sample in group:
            windows[role].append(window_from_rows(sample,by_segment[sample["segment_id"]],lonlat))
        feature_entry=next(f for f in encoder.adapter.manifest["files"] if f["file_id"]==file_id and f["split"]==role)
        provenance=pq.read_table(encoder.adapter.root/feature_entry["path"],columns=["dem_surface_parent_asset_id","worldcover_parent_asset_id"])
        for column in provenance.columns:
            parents.update(p for p in column.to_pylist() if p and p.startswith("registered-file:"))
        sources[role+":"+file_id]={"condition_sha256":entry["sha256"],"feature_sha256":feature_entry["sha256"]}
    if any(not rows for rows in windows.values()) or ({w.block_id for w in windows["train"]}&{w.block_id for w in windows["validation"]}):
        raise ValueError("development roles are empty or independent blocks overlap")
    for role in windows:
        windows[role].sort(key=lambda w:w.sample_id)
    identity={"version":"pirc17-development-windows-v1","eligibility_sha256":eligibility_sha256,
        "fit_population":"one first-future transition per qualified origin; pilot only",
        "sources":sources,"sample_ids":{role:[w.sample_id for w in rows] for role,rows in windows.items()}}
    identity["sha256"]=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return windows,sorted(parents),identity


def transition_rows(windows, full_encoder, selected_encoder):
    roles={w.role for w in windows}
    if len(roles)!=1:
        raise ValueError("transition batch must contain exactly one role")
    matrices=full_encoder.encode([w.canonical_origin for w in windows]).model_matrix(len(windows))
    directions=[]
    validity=[]
    for window in windows:
        origin=window.origin
        state=PredictedState(0.,origin.position_m[None],origin.velocity_mps[None],
            origin.history_positions_m[None],origin.history_times_seconds)
        direction,valid=state.history_direction
        directions.append(direction[0])
        validity.append(valid[0])
    return TransitionRows(next(iter(roles)),tuple(w.block_id for w in windows),
        np.stack([w.origin.velocity_mps for w in windows]),np.array(directions),np.array(validity,dtype=bool),
        np.stack([w.transition_displacement_m for w in windows]),np.array([w.transition_seconds for w in windows]),
        subset_matrix(matrices,full_encoder.columns,selected_encoder.columns))
