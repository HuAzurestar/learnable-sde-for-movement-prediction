"""Pure stage worker: no store, authority, grants or runtime paths."""
import json
from pathlib import Path
import sys
import time
from infrastructure.research_store import ResearchError, atomic_write, encode, digest
from .walking_stage import read_package, result_binding

ARMS=("M0-UO2","M2-UO2","A-S","A-N","B-S","B-N","C-T-S","C-T-N","C-R-S","C-R-N")

def pack(value):
    import torch
    if isinstance(value,torch.Tensor):
        return {"tensor":value.detach().tolist()}
    if isinstance(value,dict):
        return {k:pack(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [pack(v) for v in value]
    return value


def unpack(value):
    import torch
    if isinstance(value,dict):
        if set(value)=={"tensor"}:
            return torch.tensor(value["tensor"],dtype=torch.float64)
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list):
        return [unpack(v) for v in value]
    return value


def compute(package_path, output):
    package, values, package_hash = read_package(package_path)
    if time.monotonic() >= package["deadline_monotonic"]:
        raise ResearchError("BUDGET_EXHAUSTED", "stage deadline passed")
    binding=result_binding(package,package_hash)
    def write_result(path, content):
        value=json.loads(content); value["stage_binding"]=binding
        atomic_write(path,encode(value))
    import torch
    torch.set_num_threads(1)
    from models.walking_pilot import PilotError,PilotModel
    from estimation.walking_pilot import prefix_state,prepare_training,calibrated_model,fit_pilot
    from evaluation.walking_pilot import evaluate
    stage=package["stage"]; arm=package["cell"]["pilot_arm"]
    config=package["numeric_config"]; inputs=package["input_refs"]
    if stage=="source":
        from .walking_population import parse_population
        source=next(s for s in config["sources"] if s["source_id"]==package["cell"]["source_id"])
        result=parse_population(values.get("population"),source,package["lineage"])
        write_result(Path(output),encode(result))
        return
    track_bytes=values
    prepared_bytes=values.get("prepared")
    trained_bytes=values.get("trained")
    if stage=="prepare":
        tracks={k:json.loads(v) for k,v in track_bytes.items()}
        train=[tracks[s["source_id"]] for s in config["sources"] if s["role"]=="train" and s["source_id"] in tracks]
        if not train:
            raise PilotError("NO_AVAILABLE_ORIGINAL_TRAIN_SOURCE")
        prepared=prepare_training(train); dev=[]; models={}
        present={t["source_id"] for t in train}
        prepared["sources"].extend({"source_id":s["source_id"],"training_origin":"SOURCE_UNAVAILABLE","calibration_increments":0}
            for s in config["sources"] if s["role"]=="train" and s["source_id"] not in present)
        prepared["original_train_denominator"]=4
        for source in config["sources"]:
            if source["role"]!="selection":
                continue
            track=tracks.get(source["source_id"])
            if track is None or track.get("status")=="CONTRACT_REJECTED":
                dev.append({"source_id":source["source_id"],"status":"CONTRACT_REJECTED" if track else "SOURCE_UNAVAILABLE"}); continue
            try:
                prefix=None
                for segment in track.get("segments",[track]):
                    for origin in segment["times"]:
                        if origin<segment["times"][0]+5-1e-8:
                            continue
                        try:
                            prefix=prefix_state(segment["times"],segment["positions"],origin)
                            break
                        except PilotError:
                            pass
                    if prefix is not None:
                        track=segment
                        break
                if prefix is None:
                    raise PilotError("MISSING_CAUSAL_FIVE_SECOND_PREFIX")
                native={}; targets={}
                for t,position in zip(track["times"],track["positions"]):
                    dt=round(t-origin,10)
                    if 0<dt<=60+1e-8:
                        local=(torch.tensor(position,dtype=torch.float64)-prefix["origin_position"]).tolist()
                        native[str(float(dt))]=local
                        for horizon in (10,30,60):
                            if abs(dt-horizon)<1e-8:
                                targets[str(horizon)]=local
                dev.append({"source_id":source["source_id"],"status":"READY","particles":prefix["particles"],
                            "targets":targets,"native_times":[float(t) for t in native],"native_targets":native})
            except (PilotError,StopIteration) as exc:
                dev.append({"source_id":source["source_id"],"status":str(exc) or "MISSING_PREFIX"})
        for candidate in ARMS:
            try:
                m=calibrated_model(candidate,prepared,config["speed_limit"])
                models[candidate]={"status":"READY","checkpoint":m.checkpoint(),"card":m.model_card()}
            except PilotError as exc:
                models[candidate]={"status":str(exc)}
        result=pack({"prepared":prepared,"development":dev,"models":models})
    elif stage=="train":
        prepared=unpack(json.loads(prepared_bytes))
        initial=prepared["models"][arm]
        if initial["status"]!="READY":
            raise PilotError(initial["status"])
        model=PilotModel.from_checkpoint(initial["checkpoint"])
        def progress(value):
            write_result(Path(output).with_name("progress.json"),encode({"arm":arm,"training":value,"prepared":inputs["prepared"]}))
        train_result=fit_pilot(model,prepared["prepared"]["examples"],prepared["prepared"]["Rp"],
                              max_steps=config.get("train_steps",100),progress=progress)
        result={"arm":arm,"training":train_result,"prepared":inputs["prepared"]}
    elif stage=="predict":
        trained=json.loads(trained_bytes)
        prepared=unpack(json.loads(prepared_bytes))
        if trained["training"]["monitor_score"] is None:
            raise PilotError("TRAINING_NO_FINITE_MONITORED_CHECKPOINT")
        model=PilotModel.from_checkpoint(trained["training"]["checkpoint"])
        rows=[]
        result={"arm":arm,"sources":rows,"checkpoint_hash":digest(model.checkpoint()),"prepared":trained["prepared"]}
        for example in prepared["development"]:
            row={"source_id":example["source_id"],"status":example["status"],"horizons":[]}
            if example["status"]=="READY":
                try:
                    row["horizons"]=evaluate(model,example,prepared["prepared"]["Rp"])
                    row["status"]="COMPLETE"
                except PilotError as exc:
                    row["status"]=str(exc)
            rows.append(row)
            write_result(Path(output).with_name("progress.json"),encode(result))
    else:
        raise ResearchError("CONTRACT_MISMATCH","unknown pilot stage")
    write_result(Path(output),encode(result))


def main():
    package_path,output=sys.argv[1:]
    package=None
    try:
        package,_,_=read_package(package_path)
        compute(package_path,output)
        return 0
    except ResearchError as exc:
        if package is not None:
            # Same bounded error ABI, without importing the store supervisor.
            atomic_write(Path(output).with_name("worker-error.json"),encode({
                "schema":"walking-worker-error-v1","attempt_id":package["attempt_id"],
                "stage":package["stage"],"code":exc.code}))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
