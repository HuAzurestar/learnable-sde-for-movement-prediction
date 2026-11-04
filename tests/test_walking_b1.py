"""DEV-only artificial B1 contracts and native whole-tree execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest

from infrastructure.research_store import ResearchError, digest, encode
from application.walking_execution import activate_execution, execution_config, legacy_ref, resolve_execution
from application.walking_pilot import cells, code_identity, preflight, read_internal
from application.walking_pilot_budget import policy, charge_owner, usage
from application.walking_pilot_sources import checked_metadata
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_supervisor import ResearchSupervisor
from application.walking_stage import prepare_package, read_package
from application.walking_watch import WalkingWatch
from tests.test_walking_pilot_sources import setup


def execution(tmp_path,continued=False,grant_changes=None):
    store, old = setup(tmp_path)
    cfg = copy.deepcopy(old); cfg.update(execution_id="b1-fixture", study_id="synthetic-b1")
    if continued:
        from tests.test_research_continuation import approved
        ledger=BudgetLedger(store); cell=next(c for c in cells(old) if c["pilot_stage"]=="train" and c["pilot_arm"]=="M0-UO2")
        attempt=store.new_attempt(store.register_run(old["study_id"],cell))
        reservation=ledger.reserve(attempt,BudgetSpec(120,category="pilot",pilot_id=old["pilot_id"],pilot_phase="train"))
        ledger.settle(reservation["reservation_id"],119000,outcome="TIMEOUT"); store.transition(attempt,"TIMEOUT",error_code="TIMEOUT")
        ref=approved(store,arm_ids=[cell["arm_id"]],pilot_study_id=old["study_id"])
        ledger.continue_arm(cell["arm_id"],approval_ref=ref)
        cfg["continuation_approval"]=ref
    cfg["artifact_grant"] = {"authorization_id":"internal-grant", "version":"v2"}
    grants = checked_metadata(old["source_grants"])
    for grant in grants["grants"]:
        original = grant["authorization_id"]
        grant.update(version="v2",pilot_study_id=cfg["study_id"])
        if grant_changes: grant.update(grant_changes)
        store.authorize(grant)
        for row in cfg["sources"]:
            if row["authorization_id"] == original:
                row["authorization_version"] = grant["version"]
    grants["pilot_study_id"] = cfg["study_id"]
    path = tmp_path/"b1-grants.json"; path.write_bytes(encode(grants))
    cfg["source_grants"] = {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    internal = dict(store.authorization("internal-grant",version="v1"))
    internal.update(version="v2",study_id=cfg["study_id"],pilot_study_id=cfg["study_id"])
    store.authorize(internal)
    study = copy.deepcopy(store.manifest("study-"+old["study_id"])["spec"])
    study.update(study_id=cfg["study_id"],code_hash=code_identity(),data_hash=digest(cfg["sources"]),cells=cells(cfg))
    store.register(study,digest(study))
    store.publish("b1-config",cfg)
    decision={"user":"azurestarhu","decision_record":"PIRC-26/DECISION-03@v1","solution":"SOL-014@v2/B1","implementation":True}
    store.publish("b1-decision",decision)
    p=policy(BudgetLedger(store),cfg["pilot_id"])
    binding={"schema":"walking-execution-v1","execution_id":cfg["execution_id"],"budget_pilot_id":cfg["pilot_id"],
             "previous_execution_ref":legacy_ref(store,p),"study_id":cfg["study_id"],
             "config_ref":{"manifest_id":"b1-config","sha256":digest(cfg)},"code_hash":code_identity(),
             "qualification_ref":cfg["qualification"],"source_inventory_hash":cfg["source_inventory"]["sha256"],
             "source_geometry_hash":cfg["source_geometry"]["sha256"],"arms":cfg["arms"],"expires_at":p["expires_at"],
             "decision_ref":{"manifest_id":"b1-decision","sha256":digest(decision)}}
    return store,old,cfg,binding


def test_execution_is_explicit_replay_does_not_replenish(tmp_path):
    store,old,cfg,b=execution(tmp_path); ledger=BudgetLedger(store)
    charge_owner(ledger,cfg["pilot_id"],1181511,"artificial-old-spend")
    with pytest.raises(ResearchError): preflight(store,cfg)
    before=usage(ledger,cfg["pilot_id"])
    ref=activate_execution(store,b)
    preflight(store,cfg)
    assert activate_execution(store,b)==ref
    assert usage(ledger,cfg["pilot_id"])==before
    assert execution_config(store,cfg["pilot_id"],cfg["execution_id"])==cfg
    assert store.manifest("walking-config-"+cfg["pilot_id"])==old
    with pytest.raises(ResearchError): preflight(store,old)
    with pytest.raises(ResearchError): execution_config(store,cfg["pilot_id"],"unregistered")
    assert before["global_ms"]==1181511


@pytest.mark.parametrize("field,value",[("arms",{}),("expires_at","2026-11-04T23:59:00+08:00"),
                                      ("code_hash","0"*64),("source_inventory_hash","0"*64),
                                      ("previous_execution_ref",{})])
def test_execution_cannot_change_original_authority(tmp_path,field,value):
    store,old,cfg,b=execution(tmp_path); b[field]=value
    with pytest.raises(ResearchError): activate_execution(store,b)
    assert not any(e["event_kind"]=="WALKING_EXECUTION_ACTIVATED" for e in store.events())


def test_pending_prevents_activation(tmp_path):
    store,old,cfg,b=execution(tmp_path)
    run=store.register_run(old["study_id"],cells(old)[0]); store.new_attempt(run)
    with pytest.raises(ResearchError,match="unresolved"): activate_execution(store,b)


@pytest.mark.parametrize("changes",[{"evidence_hash":"f"*64},{"expires_at":"2027-01-01T00:00:00+00:00"},
                                    {"public_export":True},{"consumer_arm_ids":["M0-UO2"]}])
def test_execution_grants_must_preserve_exact_old_authority(tmp_path,changes):
    store,old,cfg,b=execution(tmp_path,grant_changes=changes)
    with pytest.raises(ResearchError): activate_execution(store,b)


def test_original_arm_continuation_maps_only_activated_study_with_same_training_spend(tmp_path):
    store,old,cfg,b=execution(tmp_path,continued=True); ledger=BudgetLedger(store)
    activate_execution(store,b); preflight(store,cfg)
    cell=next(c for c in cells(cfg) if c["pilot_stage"]=="train" and c["pilot_arm"]=="M0-UO2")
    run=store.register_run(cfg["study_id"],cell); attempt=store.new_attempt(run)
    spec=BudgetSpec(1,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="train")
    reservation=ledger.reserve(attempt,spec)
    assert reservation["study_id"]==cfg["study_id"] and reservation["original_study_id"]==old["study_id"]
    ledger.settle(reservation["reservation_id"],800,outcome="TIMEOUT"); store.transition(attempt,"TIMEOUT",error_code="TIMEOUT")
    assert ledger.reserve(attempt,spec)["settled"]
    assert usage(ledger,cfg["pilot_id"])["training_ms"][cell["arm_id"]]==119800
    retry=store.new_attempt(run,parent_attempt_id=attempt,reason="artificial-retry")
    with pytest.raises(ResearchError,match="training limit"):
        ledger.reserve(retry,BudgetSpec(.201,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="train"))


@pytest.mark.parametrize("change",["head","event-name","hold","stall","read-failure"])
def test_light_detector_fails_closed_without_chain_read(tmp_path,change):
    store,old=setup(tmp_path); now=[1.]
    with store._read_transaction(): watch=WalkingWatch(store.path,lambda:now[0])
    if change=="head": store.append("ARTIFICIAL_CHANGE",{})
    elif change=="event-name": (store.path/"events"/"extra.json").write_bytes(b'{}')
    elif change=="hold": (store.path/"recovery-hold.json").write_bytes(b'{}')
    elif change=="stall": now[0]+=.251
    else: (store.path/"head.json").unlink()
    with pytest.raises((ResearchError,OSError)): watch.check()


def test_native_parent_read_pure_compute_chain(tmp_path):
    if os.name!="nt": pytest.skip("formal Job containment is Windows-only")
    store,old,cfg,b=execution(tmp_path); activate_execution(store,b)
    artifacts={}; parent=os.getpid(); read_pids=[]
    original=store.read_artifact
    def observed(*args,**kwargs):
        read_pids.append(os.getpid()); return original(*args,**kwargs)
    store.read_artifact=observed
    def execute(cell,inputs):
        run=store.register_run(cfg["study_id"],cell); attempt=store.new_attempt(run)
        result=ResearchSupervisor(store).run(attempt,lambda output:[sys.executable,"-c","raise Exception('old worker used')"],
            BudgetSpec(30,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="train" if cell["pilot_stage"]=="train" else "predict" if cell["pilot_stage"]=="predict" else "prepare"),
            resource_plan={"maximum_result_bytes":4*1024**2,"tree_memory_bytes":2*1024**3},
            walking_prepare=lambda output,deadline:prepare_package(store,cfg,attempt,inputs,output,deadline))
        assert result["state"]=="SUCCEEDED",result
        return result["artifact_id"]
    rows=cells(cfg)
    source=rows[0]; artifacts[source["source_id"]]=execute(source,{})
    prepared=execute(next(c for c in rows if c["pilot_stage"]=="prepare"),{"sources":artifacts})
    trained=execute(next(c for c in rows if c["pilot_stage"]=="train" and c["pilot_arm"]=="C-R-N"),{"prepared":prepared})
    predicted=execute(next(c for c in rows if c["pilot_stage"]=="predict" and c["pilot_arm"]=="C-R-N"),{"trained":trained})
    assert predicted and read_pids and set(read_pids)=={parent}
    events=store.events(); assert not any(e["event_kind"]=="HEARTBEAT" for e in events)
    assert sum(e["event_kind"]=="WORKER_TREE_STOPPED" for e in events)==4
    assert not usage(BudgetLedger(store),cfg["pilot_id"])["pending"]


def test_unchanged_head_historical_tamper_detected_at_boundary(tmp_path):
    store,old=setup(tmp_path)
    with store._read_transaction(): watch=WalkingWatch(store.path)
    event=store.path/"events/0000000000000001.json"
    before=event.stat(); content=event.read_bytes()
    event.write_bytes(content.replace(b'"sequence":1',b'"sequence":0'))
    if event.read_bytes()==content:
        value=json.loads(content); value["hash"]="0"*64; event.write_bytes(encode(value))
    os.utime(event,ns=(before.st_atime_ns,before.st_mtime_ns))
    # Directory membership/head do not prove contents unchanged. At most one
    # bounded stage can compute; its post-stop full validation must still fail.
    watch.check()
    with pytest.raises(ResearchError): store.events()


@pytest.mark.parametrize("mutation",["escape","hash","oversize","duplicate"])
def test_private_input_rejects_paths_hashes_and_duplicate_names(tmp_path,mutation):
    p=tmp_path/"stage-input.json"; raw=b'{}'; (tmp_path/"i.json").write_bytes(raw)
    item={"name":"one","relative_file":"i.json","sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":2}
    package={"schema":"walking-stage-input-v2","deadline_monotonic":time.monotonic()+10,"inputs":[item]}
    if mutation=="escape": item["relative_file"]="../i.json"
    elif mutation=="hash": item["sha256"]="0"*64
    elif mutation=="oversize": (tmp_path/"i.json").write_bytes(b' '* (4*1024**2+1))
    else: package["inputs"].append(dict(item,name="two"))
    p.write_bytes(encode(package))
    with pytest.raises(ResearchError): read_package(p)
