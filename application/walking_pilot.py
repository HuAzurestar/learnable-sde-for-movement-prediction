"""Registered walking pilot controller and supervised stage adapter.

Plan: python -m application.walking_pilot plan --config frozen.json
Run:  python -m application.walking_pilot run --config frozen.json --report status.json
The controller registers attempts only, never grants, studies or budgets.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from infrastructure.research_store import ResearchError,ResearchStore,atomic_write,digest,encode,file_lock,identifier
from .research_budget import BudgetLedger,BudgetSpec
from .walking_pilot_budget import policy,remaining,usage,charge_owner,OwnerMeter,owner_session

ARMS=("M0-UO2","M2-UO2","A-S","A-N","B-S","B-N","C-T-S","C-T-N","C-R-S","C-R-N")
CODE_FILES=("models/walking_pilot.py","estimation/walking_pilot.py","evaluation/walking_pilot.py",
            "application/walking_pilot.py","application/walking_pilot_budget.py",
            "application/walking_pilot_sources.py", "application/walking_execution.py",
            "application/walking_stage.py", "application/walking_compute.py", "application/walking_population.py",
            "application/walking_watch.py",
            "application/research_budget.py","application/research_supervisor.py",
            "application/research_data.py","application/research_continuation.py","infrastructure/process_tree.py",
            "infrastructure/research_store.py")


def code_identity():
    root=Path(__file__).resolve().parents[1]
    return digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CODE_FILES})


def cells(config):
    """Exact preregistration matrix; all payload lineage remains restricted."""
    blocks=[s["block_id"] for s in config["sources"]]
    anchor=blocks[0]
    common={**({"execution_id":config["execution_id"]} if config.get("execution_id") else {}),"seed":2601,"resource_class":"cpu","pilot_id":config["pilot_id"]}
    prep=config.get("preparation_arm","M0-UO2")
    rows=[{**common,"arm_id":config["arms"][prep],"pilot_arm":prep,
           "pilot_stage":"source","source_id":s["source_id"],"block_id":s["block_id"]}
          for s in config["sources"]]
    rows.append({**common,"arm_id":config["arms"][prep],"pilot_arm":prep,
                 "pilot_stage":"prepare","block_id":anchor,"lineage_block_ids":blocks})
    for arm in ARMS:
        for stage in ("train","predict"):
            rows.append({**common,"arm_id":config["arms"][arm],"pilot_arm":arm,
                         "pilot_stage":stage,"block_id":anchor,"lineage_block_ids":blocks})
    return rows


def validate_config(config):
    from .walking_pilot_budget import GRID_FIELDS,GRID_BASIS,LIMIT_BASIS
    grid=config.get("schema_version")=="pirc26-walking-config-grid-v1"
    if grid:
        if config.get("billing_basis")!=GRID_BASIS or config.get("limit_basis")!=LIMIT_BASIS or not isinstance(config.get("billing_decision_ref"),dict):
            raise ResearchError("CONTRACT_MISMATCH","explicit grid billing binding required")
    elif any(k in config for k in GRID_FIELDS):
        raise ResearchError("CONTRACT_MISMATCH","legacy config cannot carry grid billing")
    if (config.get("schema_version") not in {"pirc26-walking-config-v1","pirc26-walking-config-grid-v1"} or config.get("motion")!="walking"
            or config.get("final_test") is not False or set(config.get("arms",{}))!=set(ARMS)
            or len(set(config["arms"].values()))!=10
            or config["arms"]["M0-UO2"]!="pirc26-m0-o1" or config["arms"]["M2-UO2"]!="pirc26-m2-o1"
            or type(config.get("speed_limit")) not in (int,float) or not math.isfinite(config["speed_limit"]) or config["speed_limit"]<=0
            or not config.get("speed_basis") or not config.get("speed_frozen_at")
            or config.get("train_steps",100) not in range(101) or config.get("preparation_arm","M0-UO2") not in ARMS):
        raise ResearchError("CONTRACT_MISMATCH","explicit frozen walking configuration required")
    source=config.get("sources",[])
    if (len(source)!=11 or len({s["source_id"] for s in source})!=11
            or len({s["block_id"] for s in source})!=11
            or sum(s["role"]=="train" for s in source)!=4
            or sum(s["role"]=="selection" for s in source)!=7):
        raise ResearchError("UNAUTHORIZED_DATA","original four train and seven development identities required")
    for s in source:
        if config.get("source_mode")=="registered-artifacts":
            if s.get("purpose")!=("resume" if s["role"]=="train" else "evaluate"):
                raise ResearchError("UNAUTHORIZED_DATA","original artifact purpose differs")
        elif (s.get("units")!={"position":"m","time":"s"} or set(s.get("columns",{}))!={"time","x","y"}
                or len(s.get("sha256",""))!=64 or not s.get("dataset_id") or not s.get("release_id")):
            raise ResearchError("CONTRACT_MISMATCH","source requires frozen identity, metre/second CSV column map")
        identifier(s["source_id"]); identifier(s["block_id"])
    frozen=datetime.fromisoformat(config["speed_frozen_at"])
    if frozen.tzinfo is None or frozen>datetime.now(timezone.utc):
        raise ResearchError("CONTRACT_MISMATCH","speed assumption requires an observed pre-development freeze time")
    if config.get("source_mode")=="registered-artifacts":
        from .walking_pilot_sources import validate_source_metadata
        validate_source_metadata(config)


def preflight(store,config):
    with store._read_transaction():
        expiries=[]
        ledger=_preflight_in_scope(store,config,expiries)
        def current():
            expiries.clear()
            _preflight_in_scope(store,config,expiries)
        store._read_completion(current,lambda: _check_expiries(expiries))
        return ledger


def _check_expiries(expiries):
    now=datetime.now(timezone.utc)
    if any(datetime.fromisoformat(value)<=now for value in expiries):
        raise ResearchError("UNAUTHORIZED_DATA","pilot or source authorization expired")


def _preflight_in_scope(store,config,expiries):
    # Called both before and after the physical final chain verification. The
    # completion callback must not register another completion recursively.
    from .walking_pilot_budget import _policy_in_scope
    validate_config(config)
    ledger=BudgetLedger(store); p=_policy_in_scope(ledger,config["pilot_id"])
    from .walking_execution import resolve_execution
    p, registered_config=resolve_execution(store,p,config.get("execution_id"))
    expiries.append(p["expires_at"])
    if (p["arms"]!=config["arms"] or p["study_id"]!=config["study_id"]
            or p["speed_limit"]!=config["speed_limit"] or p.get("config_hash")!=digest(config)
            or registered_config!=config):
        raise ResearchError("CONTRACT_MISMATCH","configuration differs from immutable pilot policy")
    spec=store.manifest("study-"+config["study_id"])["spec"]
    if (spec["code_hash"]!=code_identity() or spec["data_hash"]!=digest(config["sources"])
            or spec["cells"]!=cells(config) or {a["arm_id"] for a in spec["arms"]}!=set(config["arms"].values())):
        raise ResearchError("CONTRACT_MISMATCH","registered code, sources or execution matrix differs")
    qualification=store.manifest(config["qualification"]["manifest_id"])
    if (digest(qualification)!=config["qualification"]["sha256"] or qualification.get("code_hash")!=code_identity()
            or qualification.get("scope")!="independent-walking-pilot" or qualification.get("status")!="passed"):
        raise ResearchError("UNAUTHORIZED_DATA","independent frozen-code qualification required")
    artifact_mode=config.get("source_mode")=="registered-artifacts"
    protocol=None if artifact_mode else store.manifest("protocol-"+config["protocol_id"])
    if artifact_mode:
        expected=digest({"inventory":config["source_inventory"]["sha256"],"geometry":config["source_geometry"]["sha256"]})
        if spec["protocol_hash"]!=expected:
            raise ResearchError("CONTRACT_MISMATCH","original artifact lineage protocol hash differs")
    elif protocol["study_id"]!=config["study_id"] or spec["protocol_hash"]!=digest(protocol):
        raise ResearchError("UNAUTHORIZED_DATA","data protocol differs from study")
    for source in config["sources"]:
        if artifact_mode:
            from .walking_pilot_sources import _source_authority_in_scope
            grant=_source_authority_in_scope(store,config,source)
            if grant is not None:
                expiries.append(grant["expires_at"])
            continue
        matches=[b for b in protocol["blocks"] if b["block_id"]==source["block_id"]]
        if len(matches)!=1:
            raise ResearchError("UNAUTHORIZED_DATA","original source absent from protocol")
        block=matches[0]
        if any(block.get(k)!=source[k] for k in ("sha256","dataset_id","release_id")) or block["split_role"]!=source["role"]:
            raise ResearchError("UNAUTHORIZED_DATA","source identity/role changed")
        grant=store.authorization(source["authorization_id"],version=source.get("authorization_version"))
        expiries.append(grant["expires_at"])
        purpose="fit" if source["role"]=="train" else "select"
        if (grant["study_id"]!=config["study_id"] or grant.get("protocol_hash")!=digest(protocol)
                or source["block_id"] not in grant["block_ids"] or purpose not in grant["purposes"]
                or grant.get("test_authorization") is not False
                or datetime.fromisoformat(grant["expires_at"])<=datetime.now(timezone.utc)):
            raise ResearchError("UNAUTHORIZED_DATA","source grant missing, expired or final-test enabled")
    artifact_grant=store.authorization(config["artifact_grant"]["authorization_id"],version=config["artifact_grant"].get("version"))
    expiries.append(artifact_grant["expires_at"])
    if (artifact_grant["study_id"]!=config["study_id"] or artifact_grant.get("test_authorization") is not False
            or not {"evaluate","resume"}<=set(artifact_grant["purposes"])
            or not {s["block_id"] for s in config["sources"]}<=set(artifact_grant["block_ids"])
            or datetime.fromisoformat(artifact_grant["expires_at"])<=datetime.now(timezone.utc)):
        raise ResearchError("UNAUTHORIZED_DATA","derived internal artifact grant required")
    return ledger


def original_account_preflight(store,config,arm_id):
    if arm_id not in (config["arms"]["M0-UO2"],config["arms"]["M2-UO2"]):
        return
    # An unavailable original account is an arm row, not permission to spend it
    # or a global block on other accounts. The frozen preparation account can
    # be a new pilot arm, still charged to this same global ledger allowance.
    ledger=BudgetLedger(store)
    try:
        with store._read_transaction():
            references=[]
            for event in store._events():
                if event["event_kind"]=="ARM_CONTINUATION":
                    receipt=store._manifest(event["payload"]["manifest_id"])
                    if receipt["arm_id"]==arm_id:
                        references.append(receipt["approval"])
            expected=config.get("continuation_approval")
            if references:
                if references[-1]!=expected or digest(ledger._continuation_policy(arm_id))!=expected["sha256"]:
                    raise ResearchError("ACCOUNT_NOT_READY","exact live original-arm continuation reference required")
            elif expected is not None:
                raise ResearchError("ACCOUNT_NOT_READY","original-arm continuation has not been registered")
    except ResearchError as exc:
        if exc.code=="UNAUTHORIZED_DATA":
            raise ResearchError("ACCOUNT_NOT_READY","original-arm continuation expired") from exc
        raise


def read_internal(store,config,artifact_id,consumer):
    return json.loads(_read_internal_bytes(store,config,artifact_id,consumer))


def _read_internal_bytes(store,config,artifact_id,consumer):
    with store._read_transaction():
        metadata=store.manifest("artifact-"+artifact_id)
        grant=store.authorization(config["artifact_grant"]["authorization_id"],version=config["artifact_grant"].get("version"))
        purpose="resume" if metadata["role"]=="checkpoint" else "evaluate"
        raw=store.read_artifact(artifact_id,purpose=purpose,authorization=grant,consumer=consumer)
        store.verify_artifact_read(artifact_id,purpose=purpose,authorization=grant,consumer=consumer)
    return raw


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


def worker(store,config,attempt_id,inputs,output):
    """Called only behind ResearchSupervisor's GO barrier and native tree fuse."""
    if config.get("execution_id"):
        raise ResearchError("UNAUTHORIZED_DATA","B1 requires the pure computation stage entrypoint")
    import torch
    torch.set_num_threads(1)
    from .research_data import EvaluationExposureLedger
    from models.walking_pilot import PilotError,PilotModel
    from estimation.walking_pilot import prefix_state,prepare_training,calibrated_model,fit_pilot
    from evaluation.walking_pilot import evaluate
    with store._read_transaction():
        preflight(store,config)
        attempt=store.attempts()[attempt_id]; run=store.manifest("run-"+attempt["run_id"]); cell=run["cell"]
        if attempt["state"]!="RUNNING" or cell not in cells(config):
            raise ResearchError("UNAUTHORIZED_DATA","worker has no active registered pilot cell")
        original_account_preflight(store,config,cell["arm_id"])
        stage=cell["pilot_stage"]; arm=cell["pilot_arm"]
        consumer={"attempt_id":attempt_id,"run_id":run["run_id"],"entrypoint":"walking-pilot-v1"}
        def current_admission():
            current=store._attempts().get(attempt_id)
            if not current or current["state"]!="RUNNING" or current["run_id"]!=run["run_id"]:
                raise ResearchError("UNAUTHORIZED_DATA","walking attempt stopped during admission")
            if (store.path/"recovery-hold.json").exists():
                raise ResearchError("RECOVERY_REQUIRED","walking input admission held for recovery")
            if BudgetLedger(store)._balance(cell["arm_id"])["closed"]:
                raise ResearchError("BUDGET_EXHAUSTED","walking arm closed before input disclosure")
        current_admission()
        if stage=="source":
            source=next(s for s in config["sources"] if s["source_id"]==cell["source_id"])
            if config.get("source_mode")=="registered-artifacts":
                from .walking_pilot_sources import _read_population_bytes
                disclosure=_read_population_bytes(store,config,source,consumer)
            else:
                raw=EvaluationExposureLedger(store).read(config["protocol_id"],source["block_id"],
                    purpose="fit" if source["role"]=="train" else "select",authorization_id=source["authorization_id"],
                    authorization_version=source.get("authorization_version"),data_root=Path(config["data_root"]),
                    consumer=consumer,maximum_bytes=16*1024**2)
        elif stage=="prepare":
            track_bytes={k:_read_internal_bytes(store,config,v,consumer) for k,v in inputs["sources"].items()}
        elif stage=="train":
            prepared_bytes=_read_internal_bytes(store,config,inputs["prepared"],consumer)
        elif stage=="predict":
            trained_bytes=_read_internal_bytes(store,config,inputs["trained"],consumer)
            # Only decode the bounded lineage pointer here; tensors and all
            # numerical operations are constructed after releasing the scope.
            prepared_bytes=_read_internal_bytes(store,config,json.loads(trained_bytes)["prepared"],consumer)
        # Follow all input guards, then re-evaluate active state/hold/closure on
        # the final verified prefix before any parsing or numerical computation.
        store._read_completion(current_admission,lambda: None)
    if stage=="source":
        if config.get("source_mode")=="registered-artifacts":
            from .walking_pilot_sources import _finish_population
            result=_finish_population(store,config,source,consumer,disclosure)
            atomic_write(Path(output),encode(result))
            return
        reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        times=[]; positions=[]
        for record in reader:
            if len(times)>=20000:
                raise ResearchError("RESOURCE_PLAN_REJECTED","bounded source row quota exceeded")
            times.append(float(record[source["columns"]["time"]]))
            positions.append([float(record[source["columns"][k]]) for k in ("x","y")])
        # The canonical, registered block is already the frozen range. No source
        # substitution, future smoothing, GPS guess or unit conversion occurs.
        result={"source_id":source["source_id"],"times":times,"positions":positions,"origin":source.get("origin")}
        prefix_state(times,positions,source.get("origin") if source.get("origin") is not None else times[0]+5)
    elif stage=="prepare":
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
            atomic_write(Path(output).with_name("progress.json"),encode({"arm":arm,"training":value,"prepared":inputs["prepared"]}))
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
            atomic_write(Path(output).with_name("progress.json"),encode(result))
    else:
        raise ResearchError("CONTRACT_MISMATCH","unknown pilot stage")
    atomic_write(Path(output),encode(result))


def run_pilot(store,config,report):
    from .walking_pilot_budget import GridClock,GridOwnerMeter
    origin=time.monotonic_ns() if config.get("schema_version")=="pirc26-walking-config-grid-v1" else None
    start=origin/1000000000 if origin is not None else time.monotonic()
    ledger=BudgetLedger(store)
    invocation=uuid.uuid4().hex
    grid=GridClock(origin,time.monotonic_ns,invocation) if config.get("schema_version")=="pirc26-walking-config-grid-v1" else None
    meter=(GridOwnerMeter(ledger,config,start,time.monotonic,invocation,grid=grid) if grid else OwnerMeter(ledger,config,start,time.monotonic,invocation))
    from contextlib import nullcontext
    owner_primary=None
    try:
        with (nullcontext() if grid else file_lock(store.path/(".walking-pilot-"+identifier(config["pilot_id"])+".lock"),timeout=0)):
            with owner_session(ledger,config["pilot_id"],config.get("execution_id"),invocation,config=config,grid=grid):
                try:
                    report["settlement_pending"]=True
                    primary=None
                    try:
                        _run_pilot(store,config,report,meter)
                        report["status"]="LIMITED_RESULT"
                    except BaseException as exc:
                        primary=exc
                        report["status"]=getattr(exc,"code","CONFIG_OR_IO_ERROR")
                        report["error_type"]=type(exc).__name__
                        if hasattr(exc,"settlement_error"): report["stage_settlement_error"]=exc.settlement_error
                    try:
                        def final_save():
                            report["elapsed_s"]=grid.sample()/1000000000 if grid else time.monotonic()-start
                            raw=encode(report)
                            atomic_write(Path(report["report_path"]),raw)
                            return raw
                        meter.terminal(report,final_save)
                    except BaseException as exc:
                        report["settlement_error"]=getattr(exc,"code",type(exc).__name__)
                        if primary is None:
                            raise
                    if primary is not None:
                        raise primary
                except BaseException as exc:
                    owner_primary=exc
                    raise
    except BaseException:
        if grid is not None and owner_primary is not None:
            raise owner_primary
        raise
    return report


def _run_pilot(store,config,report,meter):
    """Foreground controller; all source/numeric work uses the shared supervisor."""
    from .research_supervisor import ResearchSupervisor
    ledger=BudgetLedger(store); start=meter.start
    report.update(pilot_id=config.get("pilot_id"),code_hash=code_identity(),source_count=11,
                  rows=[{"arm":a,"status":"NOT_EXECUTED","training":None,"prediction":None} for a in ARMS])
    rows={r["arm"]:r for r in report["rows"]}
    development_ids=[s["source_id"] for s in config["sources"] if s["role"]=="selection"]
    for row in rows.values():
        row["development"]=[{"source_id":sid,"horizons":[{"horizon_s":h,"status":"NOT_EXECUTED"} for h in (10,30,60)]} for sid in development_ids]
    metric_rows={a:{} for a in ARMS}
    consumer={"controller":True,"pilot_id":config["pilot_id"],"entrypoint":"walking-pilot-v1"}
    def save():
        report["elapsed_s"]=grid.sample()/1000000000 if grid else time.monotonic()-start
        atomic_write(Path(report["report_path"]),encode(report))
    def owner_cost():
        meter.checkpoint()
    def partial_validator(value):
        if type(value) is not dict or value.get("arm") not in ARMS:
            raise ResearchError("CONTRACT_MISMATCH","invalid pilot partial checkpoint")
        if "training" in value:
            from models.walking_pilot import PilotModel
            PilotModel.from_checkpoint(value["training"]["checkpoint"])
    ledger=preflight(store,config)
    from .walking_pilot_budget import remaining_ns,deadline_float
    grid=getattr(meter,"grid",None)
    wall_deadline=(deadline_float(grid.origin_ns+remaining_ns(ledger,config["pilot_id"],config["arms"][config.get("preparation_arm","M0-UO2")],"prepare",grid=grid,stage_cap=False))
        if grid else start+remaining(ledger,config["pilot_id"],config["arms"][config.get("preparation_arm","M0-UO2")],"prepare"))
    if config.get("execution_id"):
        # All relevant source/internal expiry times also cap the immutable
        # absolute fuse. Parent reads and lock waits cannot renew permission.
        with store._read_transaction():
            expiries=[policy(ledger,config["pilot_id"])["expires_at"]]
            for ref in [config["artifact_grant"], *({"authorization_id":s["authorization_id"],"version":s.get("authorization_version")} for s in config["sources"])]:
                expiries.append(store.authorization(ref["authorization_id"],version=ref.get("version"))["expires_at"])
            permission_end=min(map(datetime.fromisoformat,expiries))
        wall_deadline=min(wall_deadline,time.monotonic()+max(0,(permission_end-datetime.now(timezone.utc)).total_seconds()))
    def worker_result(value):
        from .research_supervisor import WALKING_FATAL_CODES
        if value.get("error_code") in WALKING_FATAL_CODES:
            raise ResearchError(value["error_code"],"walking worker failed structurally after confirmed stop")
        return value
    def execute(cell,inputs):
        original_account_preflight(store,config,cell["arm_id"])
        owner_cost()
        stage=cell["pilot_stage"]; phase="train" if stage=="train" else "predict" if stage=="predict" else "prepare"
        seconds=0 if grid else min(120. if config.get("execution_id") else float("inf"),remaining(ledger,config["pilot_id"],cell["arm_id"],phase),max(0,wall_deadline-time.monotonic()))
        allocation=None
        if grid:
            allocation=min(remaining_ns(ledger,config["pilot_id"],cell["arm_id"],phase,grid=grid),
                max(0,int(wall_deadline*1000000000)-grid.origin_ns-grid.sample()))//1000000
            seconds=allocation/1000
        if seconds<=0:
            raise ResearchError("BUDGET_EXHAUSTED","global or cumulative training allowance consumed")
        with store._read_transaction():
            run_id=store.register_run(config["study_id"],cell)
            old=[a for a in store.attempts().values() if a["run_id"]==run_id]
            if old:
                last=old[-1]
                if last["state"]=="SUCCEEDED":
                    return {"state":"SUCCEEDED","artifact_id":last["artifact_id"],"reused":True}
                partials=[e["payload"]["artifact_id"] for e in store.events() if e["event_kind"]=="PARTIAL_CHECKPOINT_SAVED" and e["payload"]["attempt_id"]==last["attempt_id"]]
                if last["state"] not in store.TERMINAL:
                    raise ResearchError("RECOVERY_REQUIRED","previous attempt has unresolved costs")
                return worker_result({"state":last["state"],"error_code":last.get("error_code"),
                    "artifact_id":None,"partial_checkpoint_artifact_id":partials[-1] if partials else None,"reused":True})
            attempt=store.new_attempt(run_id)
        budget=BudgetSpec(seconds,category="pilot",pilot_id=config["pilot_id"],pilot_phase=phase)
        def command(output):
            return [sys.executable,"-m","application.walking_pilot","worker","--runtime-root",str(store.path.parent),
                    "--store-id",store.store_id,"--pilot-id",config["pilot_id"],"--attempt",attempt,
                    "--worker-stage",stage,
                    "--inputs",json.dumps(inputs,separators=(",",":")),"--output",str(output)]
        before=grid.sample()/1000000000 if grid else time.monotonic()
        prepare=None
        if config.get("execution_id"):
            from .walking_stage import prepare_package
            prepare=lambda output,deadline: prepare_package(store,config,attempt,inputs,output,deadline)
        try:
            value=ResearchSupervisor(store).run(attempt,command,budget,
                resource_plan={"maximum_result_bytes":4*1024**2,"tree_memory_bytes":2*1024**3},
                partial_result_validator=partial_validator if stage in {"train","predict"} else None,
                execution_deadline_monotonic=wall_deadline, walking_prepare=prepare,
                walking_clock=grid,walking_allocation_ms=allocation,
                execution_deadline_ns=int(wall_deadline*1000000000) if grid else None)
        finally:
            # Settlement can succeed before a later partial-publication error.
            # Read its real charge even on that error path; never double-bill
            # worker time as owner preparation.
            stage_primary=sys.exc_info()[1]
            stage_recorded=False
            try:
                meter.stage(attempt)
                stage_recorded=True
                with store._read_transaction():
                    paid=[r for r in ledger._state()[0].values() if r["attempt_id"]==attempt]
                if config.get("execution_id") and paid and all(r["settled"] for r in paid):
                    # Only after confirmed native stop and settlement. The durable
                    # artifact remains under the original controlled store rules.
                    work=store.path/"artifacts"/(".attempt-"+attempt)
                    if work.resolve()==work.absolute():
                        for name in ["stage-input.json",*("input-"+str(i)+".json" for i in range(16))]:
                            (work/name).unlink(missing_ok=True)
            except BaseException:
                # Preserve grid primary/cause without replacing stage diagnostics.
                # Legacy preserves the primary only when meter.stage fails.
                if stage_primary is not None and (grid is not None or not stage_recorded):
                    raise stage_primary
                raise
        value["controller_phase_wall_s"]=(grid.sample()/1000000000 if grid else time.monotonic())-before
        worker_result(value)
        owner_cost(); save()
        return value
    source_artifacts={}; report["sources"]=[]; prepared=None
    try:
        if os.name!="nt":
            raise ResearchError("RESOURCE_PLAN_REJECTED","2GiB whole-tree memory containment currently requires Windows Job Objects")
        for cell in cells(config):
            stage=cell["pilot_stage"]; arm=cell["pilot_arm"]
            try:
                if stage=="source":
                    result=execute(cell,{})
                    report["sources"].append({"source_id":cell["source_id"],**result})
                    if result["state"]=="SUCCEEDED":
                        source_artifacts[cell["source_id"]]=result["artifact_id"]
                elif stage=="prepare":
                    result=execute(cell,{"sources":source_artifacts}); report["preparation"]=result
                    prepared=result.get("artifact_id") if result["state"]=="SUCCEEDED" else None
                    if prepared:
                        details=unpack(read_internal(store,config,prepared,consumer))
                        report["train_sources"]=details["prepared"]["sources"]
                        for name in ARMS:
                            rows[name]["train_support"]=details["models"][name]["status"]
                            rows[name]["parameter_count"]=(details["models"][name].get("card") or {}).get("parameters")
                elif prepared is None:
                    rows[arm]["status"]="PREPARATION_UNAVAILABLE"
                elif stage=="train":
                    result=execute(cell,{"prepared":prepared}); rows[arm]["training"]=result
                    rows[arm]["status"]="TRAIN_"+result["state"]
                    ref=result.get("artifact_id") or result.get("partial_checkpoint_artifact_id")
                    if ref:
                        training=read_internal(store,config,ref,consumer)["training"]
                        rows[arm]["training_progress"]={k:training[k] for k in ("updates","best_update","termination","elapsed_s","monitor_score")}
                        rows[arm]["selected_checkpoint_hash"]=digest(training["checkpoint"])
                else:
                    trained=rows[arm]["training"] or {}
                    ref=trained.get("artifact_id") or trained.get("partial_checkpoint_artifact_id")
                    if ref is None:
                        rows[arm]["status"]="NO_TRAIN_CHECKPOINT"; continue
                    result=execute(cell,{"trained":ref}); rows[arm]["prediction"]=result
                    rows[arm]["status"]="PREDICT_"+result["state"]
                    reference=result.get("artifact_id") or result.get("partial_checkpoint_artifact_id")
                    if reference:
                        metrics=read_internal(store,config,reference,consumer)
                        by_source={s["source_id"]:s for s in metrics["sources"]}
                        for item in rows[arm]["development"]:
                            source=by_source.get(item["source_id"])
                            if source:
                                hs={h["horizon_s"]:h for h in source["horizons"]}
                                for h in item["horizons"]:
                                    found=hs.get(h["horizon_s"])
                                    h["status"]=found["status"] if found else source["status"]
                                    if found:
                                        metric_rows[arm][item["source_id"]+":"+str(h["horizon_s"])]=found
            except ResearchError as exc:
                if stage=="source":
                    report["sources"].append({"source_id":cell["source_id"],"status":exc.code})
                else:
                    rows[arm]["status"]=exc.code
                from .research_supervisor import WALKING_FATAL_CODES
                if exc.code in WALKING_FATAL_CODES:
                    report["status"]=exc.code
                    for row in rows.values():
                        if row["status"]=="NOT_EXECUTED":
                            row["status"]=exc.code
                        for source in row["development"]:
                            for horizon in source["horizons"]:
                                if horizon["status"]=="NOT_EXECUTED":
                                    horizon["status"]=exc.code
                    raise
                # Global exhaustion still leaves every arm/source/horizon in
                # the report; exhausted training never blocks a prediction.
            save()
    finally:
        primary=sys.exc_info()[1]
        try:
            owner_cost()
            with store._read_transaction():
                report["budget"]=usage(ledger,config["pilot_id"])
            for row in rows.values():
                for source in row["development"]:
                    for h in source["horizons"]:
                        if h["status"]=="NOT_EXECUTED":
                            h["status"]=row["status"]
            common={}
            for horizon in (10,30,60):
                common[str(horizon)]=[sid for sid in development_ids if all(metric_rows[a].get(sid+":"+str(horizon),{}).get("status")=="OK" for a in ARMS)]
            report["common_sources"]=common
            report["three_horizon_intersection"]=sorted(set.intersection(*(set(v) for v in common.values())))
            # Metrics stay inside ResearchStore; the local report contains only
            # statuses, costs and pointers, never trajectories or weights.
            summary={"schema_version":"pirc26-walking-summary-v1","metrics":metric_rows,
                     "common_sources":common,"three_horizon_intersection":report["three_horizon_intersection"],
                     "scope":"exploratory; source-equal only; no confirmatory superiority claim"}
            report["summary_artifact_id"]=store.artifact(encode(summary),role="result",visibility="restricted",
                block_ids=[s["block_id"] for s in config["sources"]],study_id=config["study_id"])["artifact_id"]
            save()
        except BaseException as exc:
            report["summary_error"]=getattr(exc,"code",type(exc).__name__)
            if primary is not None:
                raise primary
            raise
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","run","worker"))
    parser.add_argument("--config"); parser.add_argument("--report")
    for name in ("runtime-root","store-id","pilot-id","attempt","inputs","output","worker-stage"):
        parser.add_argument("--"+name)
    args=parser.parse_args(argv)
    os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    if args.command=="worker":
        stage=args.worker_stage
        try:
            store=ResearchStore(Path(args.runtime_root),args.store_id)
            config=store.manifest("walking-config-"+identifier(args.pilot_id))
            # Registered attempt/run/cell and every input are checked together
            # by worker admission. The CLI stage labels diagnostics only; the
            # parent matches it against the registered stage after native stop.
            worker(store,config,args.attempt,json.loads(args.inputs),args.output)
        except ResearchError as exc:
            from .research_supervisor import write_walking_error,WALKING_STAGES
            if stage in WALKING_STAGES:
                write_walking_error(args.output,args.attempt,stage,exc.code)
            return 2
        return 0
    if args.command=="plan":
        config=json.loads(Path(args.config).read_text(encoding="utf-8-sig")); validate_config(config)
        print(json.dumps({"code_hash":code_identity(),"config_hash":digest(config),"data_hash":digest(config["sources"]),"cells":cells(config)},indent=2))
        return 0
    if not args.report:
        parser.error("run requires --report for fail-closed status delivery")
    report={"schema_version":"pirc26-walking-report-v1","report_path":str(Path(args.report).absolute()),
            "status":"STARTING","rows":[{"arm":a,"status":"NOT_EXECUTED"} for a in ARMS]}
    try:
        config=json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        store=ResearchStore(Path(config["runtime_root"]),config["store_id"])
        run_pilot(store,config,report)
    except (ResearchError,ValueError,KeyError,OSError) as exc:
        report["status"]=getattr(exc,"code","CONFIG_OR_IO_ERROR")
        report["error_type"]=type(exc).__name__  # Never echo payload-bearing parser errors.
        for row in report["rows"]:
            if row["status"]=="NOT_EXECUTED":
                row["status"]=report["status"]
    if "terminal_receipt_id" not in report:
        atomic_write(Path(args.report).absolute(),encode(report))
    if report.get("settlement_error"):
        print(json.dumps({"status":report["status"],"settlement_error":report["settlement_error"],"settlement_pending":True}))
    else:
        print(report.get("cli_output") or json.dumps({"status":report["status"],"report":report["report_path"]}))
    return 0 if report["status"]=="LIMITED_RESULT" and not report.get("settlement_error") else 2


if __name__=="__main__":
    raise SystemExit(main())
