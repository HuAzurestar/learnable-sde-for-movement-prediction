"""Synthetic null/tail qualification only; not invented research data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .inference import INFERENCE_VERSION, InferenceConfig, PRIMARY_FAMILY, SEEDS, infer


def _wilson(successes, total):
    z=1.959963984540054
    p=successes/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [float(center-half),float(center+half)]


def check(repetitions=200):
    if repetitions<100:
        raise ValueError("at least 100 null fixtures required for this qualification")
    rng=np.random.default_rng(20260928)
    implementation_sha256=hashlib.sha256(Path(__file__).with_name("inference.py").read_bytes()).hexdigest()
    config=InferenceConfig(delta_m=1.)
    names=list(PRIMARY_FAMILY)
    rejected=missed=unstable=0
    for iteration in range(repetitions):
        # Correlated, independent block contrasts, all with exactly zero true mean.
        common=rng.normal(size=(62,1))
        differences=.5*common+np.sqrt(.75)*rng.normal(size=(62,5))
        rows=[]
        for block in range(62):
            scores={"all-terrain":100.}
            for c,name in enumerate(names):
                scores[PRIMARY_FAMILY[name][1]]=100.-differences[block,c]
            for configuration,score in scores.items():
                for seed in SEEDS:
                    rows.append({"configuration":configuration,"origin_id":f"fixture-{block}",
                        "independent_block_id":f"fixture-{block}","seed":seed,"score_m":score,"status":"success"})
        report=infer(rows,config=config,mechanism_passed={name:True for name in names})
        effects=list(report["results"].values())
        rejected+=any(e["holm_adjusted_p_zero"]<.05 for e in effects)
        missed+=any(not e["simultaneous_interval_m"][0]<=0<=e["simultaneous_interval_m"][1] for e in effects)
        unstable+=any(not e["tail_check_passed"] for e in effects)
    p_interval=_wilson(rejected,repetitions)
    ci_interval=_wilson(missed,repetitions)
    return {"schema_version":"pirc17-inference-fixture-qualification-v1","purpose":"synthetic_software_null_calibration_not_research_evidence",
        "inference_version":INFERENCE_VERSION,"implementation_sha256":implementation_sha256,
        "fixture_seed":20260928,"blocks_per_fixture":62,"family_size":5,"registered_seeds":list(SEEDS),
        "config":vars(config),"fixture_repetitions":repetitions,
        "holm_family_rejections":rejected,"holm_family_rejection_rate":rejected/repetitions,
        "holm_rate_wilson95":p_interval,"simultaneous_interval_family_misses":missed,
        "interval_family_miss_rate":missed/repetitions,"interval_miss_rate_wilson95":ci_interval,
        "families_with_tail_instability":unstable,
        "criterion":"flag gross inflation if lower Wilson95 bound exceeds nominal .05; this is not proof of finite-sample control",
        "passed":p_interval[0]<=.05 and ci_interval[0]<=.05,
        "limitations":["normal fixture only; not a guarantee for heavy-tailed real scores",
                       "conditional five-seed inference does not estimate general training-seed uncertainty"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=200)
    args=parser.parse_args()
    result=check(args.repetitions)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as target:
        json.dump(result,target,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
