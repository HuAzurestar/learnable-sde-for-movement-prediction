"""Fit pilot models on existing training origins, never on final-eval data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .configurations import configuration_encoder, terrain_configurations
from .development import load_development, transition_rows
from .dynamics import fit_dynamics
from .features import CanonicalEncoder
from .inference import SEEDS


def fit(eligibility, eligibility_sha256, release, snapshot, data_root):
    start=time.perf_counter()
    encoder=CanonicalEncoder.frozen_pirc22(snapshot)
    print("Verified frozen encoder",flush=True)
    windows,parents,identity=load_development(eligibility,eligibility_sha256,release,data_root,encoder)
    print(json.dumps({"loaded_origins":{r:len(w) for r,w in windows.items()}}),flush=True)
    configs=terrain_configurations()
    models={}
    for name,config in configs.items():
        selected=configuration_encoder(encoder,name)
        train=transition_rows(windows["train"],encoder,selected)
        validation=transition_rows(windows["validation"],encoder,selected)
        # Every configuration must receive exactly the same baseline and origins.
        if name=="base":
            reference_train=train.design()
            reference_validation=validation.design()
        else:
            np.testing.assert_array_equal(train.design(),reference_train)
            np.testing.assert_array_equal(validation.design(),reference_validation)
        models[name]={}
        for seed in SEEDS:
            fitted=fit_dynamics(train,validation,seed=seed,training_identity=identity["sha256"],
                configuration_identity=config["sha256"])
            models[name][str(seed)]=fitted.identity
        print(json.dumps({"fitted_configuration":name,"seeds":len(SEEDS),"conditioner_inputs":len(selected.columns)*2}),flush=True)
    return {"schema_version":"pirc17-development-fit-v1","purpose":"paired_pilot_checkpoints_not_final_model_acceptance",
        "development_identity":identity,"configurations":configs,"models":models,"snapshot_parent_assets":parents,
        "data_roles":{"gradient":"train","checkpoint_selection":"validation","final_eval_reads":0},
        "pilot_population":{"samples":{r:len(w) for r,w in windows.items()},
            "independent_blocks":{r:len({x.block_id for x in w}) for r,w in windows.items()},
            "transition_rule":"one first-future transition per qualified origin; no interpolated observations"},
        "elapsed_seconds":time.perf_counter()-start,
        "limitations":["one-step fitting diagnostics are not long-horizon forecast evidence",
            "paired rollout, step/particle sensitivity, map failures and power remain to be qualified",
            "pilot fitting population is not implicitly the sealed production training policy"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("eligibility","release","snapshot","data-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--eligibility-sha256",required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing fit artifact")
    report=fit(args.eligibility,args.eligibility_sha256,args.release,args.snapshot,args.data_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as target:
        json.dump(report,target,indent=2,allow_nan=False)
    print(json.dumps({"output":str(args.output),"population":report["pilot_population"],
        "elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
