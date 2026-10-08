"""Derive candidate practical-scale parameters from existing validation data.

This is an inertial reference-scale diagnostic, not fitted-model performance or
paired-effect power certification. Final choices still require protocol sealing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .features import LocalFrame
from .origins import causal_prefix
from .qualification import _hash
from .inference import planning_power

NOMINAL_SECONDS = (60.,300.,900.,1800.)
TIME_WEIGHTS = (.25,.25,.25,.25)
TARGET_TOLERANCE_SECONDS = 30.


def observed_indices(actual_seconds, nominal_seconds=NOMINAL_SECONDS, tolerance_seconds=TARGET_TOLERANCE_SECONDS):
    actual=np.asarray(actual_seconds,dtype=float)
    nominal=np.asarray(nominal_seconds,dtype=float)
    if actual.ndim!=1 or nominal.ndim!=1 or not len(actual) or not len(nominal):
        raise ValueError("nonempty one-dimensional timestamp grids required")
    if not np.isfinite(actual).all() or not np.isfinite(nominal).all() or actual[0]<=0 or nominal[0]<=0:
        raise ValueError("finite positive elapsed times required")
    if np.any(np.diff(actual)<=0) or np.any(np.diff(nominal)<=0) or not np.isfinite(tolerance_seconds) or tolerance_seconds<0:
        raise ValueError("strict time order and nonnegative finite tolerance required")
    # argmin breaks exact ties towards the earlier original observation.
    indexes=np.array([np.argmin(np.abs(actual-t)) for t in nominal])
    if np.any(np.abs(actual[indexes]-nominal)>tolerance_seconds) or np.any(np.diff(indexes)<=0):
        raise ValueError("registered time grid lacks distinct observed targets within tolerance")
    return indexes


def calibrate(eligibility_path, eligibility_sha256, release, snapshot, data_root):
    if _hash(eligibility_path)!=eligibility_sha256:
        raise ValueError("eligibility artifact hash mismatch")
    audit=json.loads(eligibility_path.read_text(encoding="utf-8"))
    if audit["eligibility"]["horizon_minutes"]!=30:
        raise ValueError("this calibration requires the registered 30-minute candidate")
    manifest=json.loads((snapshot/"manifest.json").read_text(encoding="utf-8"))
    cohort=json.loads((release/"cohort.json").read_text(encoding="utf-8"))
    dataset=json.loads((release/"dataset.json").read_text(encoding="utf-8"))
    if not (audit["dataset_id"]==manifest["dataset_id"]==cohort["cohort_id"]==dataset["dataset_id"]):
        raise ValueError("dataset identity mismatch")
    inventory=hashlib.sha256()
    for entry in sorted(manifest["files"],key=lambda f:f["path"]):
        inventory.update(f"{entry['path']}\0{entry['sha256']}\0{entry['row_count']}\n".encode())
    if inventory.hexdigest()!=audit["snapshot"]["content_inventory_sha256"]:
        raise ValueError("snapshot inventory differs from qualification")
    if _hash(release/"cohort.json")!=dataset["artifacts"]["cohort.json"]["sha256"]:
        raise ValueError("cohort manifest changed")
    if _hash(release/"condition_file_manifest.jsonl")!=dataset["artifacts"]["condition_file_manifest.jsonl"]["sha256"]:
        raise ValueError("condition manifest changed")
    if _hash(release/"samples.jsonl")!=audit["inputs_sha256"]["samples.jsonl"]:
        raise ValueError("registered samples changed")
    with (release/"samples.jsonl").open(encoding="utf-8") as source:
        samples={s["sample_id"]:s for s in map(json.loads,source)}
    with (release/"condition_file_manifest.jsonl").open(encoding="utf-8") as source:
        conditions={r["file_id"]:r for r in map(json.loads,source)}
    selected=defaultdict(list)
    for eligible in audit["eligibility"]["rows"]:
        if eligible["split"]!="validation":
            continue
        sample=samples[eligible["sample_id"]]
        for key in ("split","file_id","segment_id","independent_block_id","history_start","history_end","target_start","target_end"):
            if eligible[key]!=sample[key]:
                raise ValueError("eligibility/sample identity mismatch")
        selected[sample["file_id"]].append(sample)
    if not selected:
        raise ValueError("no eligible real validation data")
    feature_files={f["file_id"]:f for f in manifest["files"] if f["split"]=="validation"}
    records=[]
    source_hashes={}
    for file_id,group in sorted(selected.items()):
        entry=feature_files[file_id]
        feature_path=(snapshot/entry["path"]).resolve()
        if not feature_path.is_relative_to(snapshot.resolve()) or _hash(feature_path)!=entry["sha256"]:
            raise ValueError("qualified feature file changed")
        table=pq.read_table(feature_path,columns=["segment_id","point_index","absolute_epoch_ns","independent_block_id","split"])
        by_segment=defaultdict(list)
        for row in table.to_pylist():
            if row["split"]!="validation":
                raise ValueError("feature file crossed semantic role")
            by_segment[row["segment_id"]].append(row)
        condition=conditions[file_id]
        condition_path=(data_root/"cond_slices"/condition["relative_path"]).resolve()
        if not condition_path.is_relative_to((data_root/"cond_slices").resolve()) or _hash(condition_path)!=condition["sha256"]:
            raise ValueError("condition source changed")
        source_hashes[file_id]={"feature_sha256":entry["sha256"],"condition_sha256":condition["sha256"]}
        source=pq.read_table(condition_path,columns=["lon","lat"])
        for sample in group:
            metadata=sorted(by_segment[sample["segment_id"]],key=lambda r:r["point_index"])
            if len(metadata)!=sample["target_end"]+1 or any(r["independent_block_id"]!=sample["independent_block_id"] for r in metadata):
                raise ValueError("segment length/block differs from registered sample")
            origin_index=sample["history_end"]
            epoch=metadata[origin_index]["absolute_epoch_ns"]
            future=metadata[sample["target_start"]:sample["target_end"]+1]
            actual=np.array([(r["absolute_epoch_ns"]-epoch)/1e9 for r in future])
            indexes=observed_indices(actual)
            prefix=metadata[max(sample["history_start"],origin_index-2):origin_index+1]
            targets=[future[i] for i in indexes]
            points=source.take([r["point_index"] for r in prefix+targets]).to_pylist()
            lonlat=np.array([[r["lon"],r["lat"]] for r in points])
            frame=LocalFrame(*lonlat[len(prefix)-1])
            xy=frame.from_lonlat(lonlat)
            times=[(r["absolute_epoch_ns"]-epoch)/1e9 for r in prefix]
            origin=causal_prefix(xy[:len(prefix)],times)
            forecast=origin.position_m+actual[indexes,None]*origin.velocity_mps
            errors=np.linalg.norm(forecast-xy[len(prefix):],axis=1)
            records.append({"sample_id":sample["sample_id"],"independent_block_id":sample["independent_block_id"],
                "actual_target_seconds":actual[indexes].tolist(),"target_offset_seconds":(actual[indexes]-NOMINAL_SECONDS).tolist(),
                "inertial_energy_score_m":float(np.dot(TIME_WEIGHTS,errors)),"by_time_error_m":errors.tolist()})
    grouped=defaultdict(list)
    for row in records:
        grouped[row["independent_block_id"]].append(row["inertial_energy_score_m"])
    block_scores={block:float(np.mean(scores)) for block,scores in sorted(grouped.items())}
    mean=float(np.mean(list(block_scores.values())))
    margin=round(.05*mean,6)
    if margin<=0 or len(block_scores)<2:
        raise ValueError("reference scale cannot define positive practical margin/power inputs")
    proxy=2*float(np.std(list(block_scores.values()),ddof=1))
    return {"schema_version":"pirc17-validation-scale-calibration-v1","purpose":"candidate_margin_scale_not_model_comparison",
        "dataset_id":audit["dataset_id"],"eligibility_sha256":eligibility_sha256,"validation_samples":len(records),
        "validation_independent_blocks":len(block_scores),"nominal_seconds":list(NOMINAL_SECONDS),"time_weights":list(TIME_WEIGHTS),
        "target_policy":"nearest actual original observation; earlier tie; predict at actual time; no interpolation",
        "target_tolerance_seconds":TARGET_TOLERANCE_SECONDS,"inertial_equal_block_mean_ES_m":mean,
        "candidate_delta_m":margin,"margin_rule":"5 percent of validation equal-block inertial ES; rounded to 1e-6 m; not a business-validated utility threshold",
        "planning_sd_proxy_m":proxy,"planning_sd_status":"twice SD of inertial block scores; NOT an estimated paired model-difference SD",
        "planning_sensitivity":[planning_power(paired_sd_m=proxy,blocks=len(block_scores),true_improvement_m=k*margin,delta_m=margin) for k in (2,5,10,20)],
        "power_certified":False,"power_reason":"paired fitted-model validation differences are still required; the reference proxy alone cannot certify power",
        "source_hashes":source_hashes,"block_scores_m":block_scores,"records":records,
        "final_eval_label_prediction_metric_reads":0,"requires_exact_protocol_acceptance":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("eligibility","release","snapshot","data-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--eligibility-sha256",required=True)
    args=parser.parse_args()
    result=calibrate(args.eligibility,args.eligibility_sha256,args.release,args.snapshot,args.data_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as target:
        json.dump(result,target,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in {"source_hashes","block_scores_m","records"}},indent=2))


if __name__=="__main__":
    main()
