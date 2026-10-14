"""Audit one bounded development ledger, not scientific or numerical certification.

No source dataset is opened. All attempted settings, including failures, must
match the registered Cartesian workload before any sensitivity is reported.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .qualification import _hash


def audit(rows, *, tolerance_m):
    if not np.isfinite(tolerance_m) or tolerance_m<=0:
        raise ValueError("positive finite sensitivity tolerance required")
    if len(rows)<3 or rows[0].get("type")!="header" or rows[-1].get("type")!="completion":
        raise ValueError("complete header/run/completion ledger required")
    header,completion=rows[0],rows[-1]
    if (header.get("schema_version")!="pirc17-development-rollout-v1" or
        header.get("purpose")!="bounded_validation_engineering_pilot" or
        header.get("final_eval_label_prediction_metric_reads")!=0):
        raise ValueError("only the explicit development pilot schema is permitted")
    axes=[header[name] for name in ("sample_ids","configurations","seeds","particle_counts","max_steps_seconds")]
    if any(not axis or len(axis)!=len(set(axis)) for axis in axes):
        raise ValueError("nonempty unique registered workload axes required")
    expected=set(itertools.product(*axes))
    runs=rows[1:-1]
    if header["expected_run_count"]!=len(expected) or completion["attempted_run_count"]!=len(expected):
        raise ValueError("registered/completed workload counts differ")
    ledger={}
    origin_identity={}
    streams={}
    failures=[]
    for row in runs:
        if row.get("type")!="run":
            raise ValueError("unexpected interior ledger row")
        key=tuple(row[name] for name in ("sample_id","configuration","seed","particles","max_step_seconds"))
        if key not in expected or key in ledger:
            raise ValueError("duplicate or unregistered attempted workload")
        ledger[key]=row
        origin=row["sample_id"]
        identity=(row["independent_block_id"],tuple(row["actual_horizons_seconds"]))
        if not identity[0] or (origin in origin_identity and origin_identity[origin]!=identity):
            raise ValueError("origin changes block or scoring times")
        origin_identity[origin]=identity
        stream=(origin,row["seed"])
        driver=row["brownian_identity"]
        if driver["version"]!=header["brownian_driver"] or driver["seed"]!=row["seed"]:
            raise ValueError("Brownian identity differs from workload")
        if driver["max_particles"]<row["particles"]:
            raise ValueError("requested particles exceed registered driver")
        if driver["version"]=="pirc17-coupled-brownian-v2" and driver["stream_id"]!=origin:
            raise ValueError("Brownian stream is not bound to this origin")
        if stream in streams and streams[stream]!=driver:
            raise ValueError("settings do not share the same registered Brownian path")
        streams[stream]=driver
        if row["status"]=="failure":
            failures.append({"workload":list(key),"error_type":row.get("error_type"),"error_message":row.get("error_message")})
        elif row["status"]=="success":
            scores=row["scores"]
            vector=[scores["time_weighted_energy_score_m"],*[v["energy_score_m"] for v in scores["by_time"]]]
            if len(vector)!=1+len(identity[1]) or not np.isfinite(vector).all():
                raise ValueError("invalid successful score vector")
            if tuple(v["elapsed_seconds"] for v in scores["by_time"])!=identity[1]:
                raise ValueError("scores differ from registered scoring times")
        else:
            raise ValueError("unknown run status")
    if set(ledger)!=expected:
        raise ValueError("missing required attempted workloads")
    if completion["failure_count"]!=len(failures) or completion["success_count"]!=len(runs)-len(failures):
        raise ValueError("completion status counts differ from attempted workloads")
    report={"schema_version":"pirc17-numerical-ledger-audit-v1","attempted_runs":len(runs),
        "origin_count":len(origin_identity),"independent_block_count":len({v[0] for v in origin_identity.values()}),
        "tolerance_m":float(tolerance_m),"failures":failures,"sensitivities":[],
        "certified":False,"scope":"engineering sensitivity only; not representative convergence, Monte Carlo accuracy, power or terrain efficacy"}
    if failures:
        report["status"]="failed_workloads_no_sensitivity_conclusion"
        return report
    def es(key):
        score=ledger[key]["scores"]
        return np.array([score["time_weighted_energy_score_m"],*[v["energy_score_m"] for v in score["by_time"]]])
    # Adjacent budgets only; signs and all underlying values remain visible.
    for axis,name in ((4,"integration_step_seconds"),(3,"particle_count")):
        settings=sorted(axes[axis])
        other_axes=[i for i in range(5) if i!=axis]
        for values in itertools.product(*(axes[i] for i in other_axes)):
            for first,second in zip(settings,settings[1:]):
                key=[None]*5
                for i,value in zip(other_axes,values):
                    key[i]=value
                key[axis]=first
                a=es(tuple(key))
                key[axis]=second
                b=es(tuple(key))
                difference=b-a
                report["sensitivities"].append({"axis":name,"sample_id":key[0],"configuration":key[1],
                    "seed":key[2],"fixed_setting":{("particles" if axis==4 else "step_seconds"):key[3 if axis==4 else 4]},
                    "settings":[first,second],"weighted_ES_m":[float(a[0]),float(b[0])],
                    "second_minus_first_ES_m":float(difference[0]),
                    "second_minus_first_by_time_ES_m":difference[1:].tolist(),
                    "all_scoring_times_within_tolerance":bool(np.max(np.abs(difference))<=tolerance_m)})
    report["status"]="complete_sensitivity_only"
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--ledger-sha256",required=True)
    parser.add_argument("--tolerance-m",type=float,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an audit")
    if _hash(args.ledger)!=args.ledger_sha256:
        raise ValueError("ledger hash mismatch")
    report=audit([json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines()],tolerance_m=args.tolerance_m)
    report["ledger_sha256"]=args.ledger_sha256
    report["audit_source_sha256"]=_hash(Path(__file__))
    with args.output.open("x",encoding="utf-8") as target:
        json.dump(report,target,indent=2,allow_nan=False)
    print(json.dumps(report,allow_nan=False))
    if report["failures"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
