"""Synthetic old-study artifact compatibility and the actual stage pipeline."""
import hashlib
import json
from pathlib import Path

import pytest

from application.walking_pilot import ARMS,cells,code_identity,preflight,worker,read_internal
from application.walking_pilot_sources import sources_from_metadata,source_authority,read_population,consumer_scope
from infrastructure.research_store import ResearchStore,ResearchError,digest,encode
from tests.test_research_store import spec


def setup(tmp_path, *, train_steps=0, include_runtime=False):
    from tests.test_walking_pilot import train_tracks
    store=ResearchStore(tmp_path/"runtime","artifact-pilot-fixture",initialize=True)
    train_ids=["train-"+str(i) for i in range(4)]; dev_ids=["dev-"+str(i) for i in range(7)]
    base=train_tracks()[0]
    def document(block,ids):
        return {"schema_version":"pirc26-observed-block-v1","block_id":block,"coordinate_frame":"synthetic-metres",
                "state_units":["m","m","m/s","m/s"],"time_unit":"s","velocity_source":"backward-difference-v1",
                "train_binding_hash":"ignored-old-normalizer","normalizer_hash":"ignored-old-normalizer","context_hash":"ignored-context",
                "segments":[{"segment_id":sid,"time":base["times"],"position":base["positions"],
                             "condition":[],"condition_available_at":[]} for sid in ids]}
    training=store.artifact(encode(document("pooled",["pooled-"+i for i in train_ids])),role="training-population",
                            visibility="restricted",block_ids=train_ids,study_id="old-train")
    development=[]
    for i,sid in enumerate(dev_ids):
        artifact=None if i==6 else store.artifact(encode(document(sid,["seg-"+sid])),role="selection-population",
                        visibility="restricted",block_ids=[sid],study_id="old-dev")
        development.append({"source_block_id":sid,"artifact":artifact,"historical_source_status":"CONTRACT_REJECTED" if artifact is None else "CONVERTIBLE"})
    inventory={"schema_version":"pirc26-walking-pilot-source-inventory-v1","study_id":"synthetic-new-pilot",
               "training_source_ids":train_ids,"training_artifact":training,"development_candidates":development,
               "walking_speed_assumption":{"U_m_s":2.5,"V_m_s":2.5}}
    def metadata(name,value):
        p=tmp_path/name; p.write_bytes(encode(value)); return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    inv=metadata("inventory.json",inventory)
    geometry={"schema_version":"pirc26-pilot-original-segment-mapping-v1","source_inventory_sha256":inv["sha256"],
              "training_artifact_id":training["artifact_id"],"training_segment_source_map":[{"pooled_segment_id":"pooled-"+i,"independent_block_id":i} for i in train_ids]}
    grants=[]
    for name,oldstudy,purpose,blocks,artifacts in (("train-grant","old-train","resume",train_ids,[training["artifact_id"]]),
            ("dev-grant","old-dev","evaluate",dev_ids[:6],[c["artifact"]["artifact_id"] for c in development[:6]])):
        grant={"authorization_id":name,"version":"v1","study_id":oldstudy,"purposes":[purpose],"visibilities":["restricted"],
               "expires_at":"2026-11-03T23:59:00+08:00","evidence_hash":digest("synthetic only"),"pilot_study_id":"synthetic-new-pilot",
               "block_ids":blocks,"artifact_ids":artifacts,"test_authorization":False,"public_export":False,"movement_scope":"walking",
               "source_inventory_sha256":inv["sha256"],"consumer_arm_ids":list(ARMS),
               "consumer_scope":consumer_scope("fixture-pilot")}
        store.authorize(grant); grants.append(grant)
    grant_doc={"source_inventory_sha256":inv["sha256"],"pilot_study_id":"synthetic-new-pilot","grants":grants}
    cfg={"schema_version":"pirc26-walking-config-v1","source_mode":"registered-artifacts","pilot_id":"fixture-pilot","study_id":"synthetic-new-pilot",
         "motion":"walking","final_test":False,"speed_limit":2.5,"speed_basis":"synthetic fixture","speed_frozen_at":"2026-10-27T00:00:00+08:00",
         "arms":{a:"fixture-"+a for a in ARMS},"source_inventory":inv,"source_grants":metadata("grants.json",grant_doc),
         "source_geometry":metadata("geometry.json",geometry),"sources":sources_from_metadata(inventory,grant_doc,geometry),"train_steps":train_steps,
         "artifact_grant":{"authorization_id":"internal-grant","version":"v1"}}
    if include_runtime:
        cfg.update(runtime_root=str(store.path.parent),store_id=store.store_id)
    cfg["arms"].update({"M0-UO2":"pirc26-m0-o1","M2-UO2":"pirc26-m2-o1"})
    qualification={"scope":"independent-walking-pilot","status":"passed","code_hash":code_identity(),"note":"TEST FIXTURE ONLY; never a real qualification"}
    store.publish("fixture-qualification",qualification)
    cfg["qualification"]={"manifest_id":"fixture-qualification","sha256":digest(qualification)}
    approval={"schema_version":"pirc26-user-scoped-pilot-permission-v1","user":"azurestarhu","decision_record":"PIRC-26/DECISION-01@v1",
              "pilot_study_id":cfg["study_id"],"final_test_authorization":False,"public_export":False,"movement_scope":"walking",
              "walking_speed_assumption":{"U_m_s":2.5,"V_m_s":2.5},
              "budget":{"deadline_reset":False,"global_scientific_seconds":1800,"per_arm_training_seconds":120,"timeout_policy":"partial-results"}}
    store.publish("fixture-approval",approval)
    store.publish("walking-pilot-fixture-pilot",{"schema_version":"pirc26-walking-budget-v1","pilot_id":"fixture-pilot","study_id":cfg["study_id"],
                  "global_ms":1800000,"training_ms":120000,"expires_at":"2026-11-03T23:59:00+08:00","motion":"walking","final_test":False,
                  "speed_limit":2.5,"arms":cfg["arms"],"config_hash":digest(cfg),"approval":{"manifest_id":"fixture-approval","sha256":digest(approval)}})
    store.publish("walking-config-fixture-pilot",cfg)
    study=spec(); study.update(study_id=cfg["study_id"],code_hash=code_identity(),data_hash=digest(cfg["sources"]),
        protocol_hash=digest({"inventory":inv["sha256"],"geometry":cfg["source_geometry"]["sha256"]}),cells=cells(cfg))
    study["arms"]=[{"arm_id":v,"model_family_id":a,"method_family_id":"synthetic","objective_id":"uo2","budget_seconds":86400} for a,v in cfg["arms"].items()]
    store.register(study,digest(study))
    store.authorize({"authorization_id":"internal-grant","version":"v1","study_id":cfg["study_id"],"purposes":["evaluate","resume"],
        "visibilities":["restricted"],"block_ids":train_ids+dev_ids,"test_authorization":False,"evidence_hash":digest("synthetic"),"expires_at":"2026-11-03T23:59:00+08:00",
        "pilot_study_id":cfg["study_id"],"source_inventory_sha256":inv["sha256"],"movement_scope":"walking","public_export":False,
        "consumer_arm_ids":list(ARMS),"consumer_scope":consumer_scope(cfg["pilot_id"],internal=True)})
    return store,cfg


def test_original_study_artifact_to_four_independent_sources_and_missing_seventh(tmp_path):
    store,cfg=setup(tmp_path); preflight(store,cfg)
    assert len(cells(cfg))==32
    source=cfg["sources"][0]
    cell=next(c for c in cells(cfg) if c.get("source_id")==source["source_id"])
    run=store.register_run(cfg["study_id"],cell); attempt=store.new_attempt(run); store.transition(attempt,"RUNNING")
    read=read_population(store,cfg,source,{"entrypoint":"walking-pilot-v1","attempt_id":attempt,"run_id":run})
    assert len(read["segments"])==1 and read["segments"][0]["segment_id"]=="pooled-train-0"
    assert read["lineage"]["original_study_id"]=="old-train"
    missing=read_population(store,cfg,cfg["sources"][-1],{})
    assert missing["status"]=="CONTRACT_REJECTED"
    assert any(e["event_kind"]=="READ_COMPLETED" for e in store.events())
    bad={**source,"purpose":"evaluate"}
    with pytest.raises(ResearchError,match="grant"):
        source_authority(store,cfg,bad)


def test_actual_source_prepare_train_predict_stage_chain_with_restricted_lineage(tmp_path):
    store,cfg=setup(tmp_path); artifacts={}
    def execute(cell,inputs):
        run=store.register_run(cfg["study_id"],cell); attempt=store.new_attempt(run); store.transition(attempt,"RUNNING")
        path=tmp_path/("result-"+attempt+".json")
        worker(store,cfg,attempt,inputs,path)
        value=json.loads(path.read_bytes())
        artifact=store.artifact(path.read_bytes(),role="result",visibility="restricted",study_id=cfg["study_id"],block_ids=cell.get("lineage_block_ids",[cell["block_id"]]))
        store.transition(attempt,"SUCCEEDED",artifact_id=artifact["artifact_id"])
        return artifact["artifact_id"],value
    for cell in cells(cfg):
        if cell["pilot_stage"]=="source":
            ref,_=execute(cell,{}); artifacts[cell["source_id"]]=ref
    prepared,details=execute(next(c for c in cells(cfg) if c["pilot_stage"]=="prepare"),{"sources":artifacts})
    assert len(details["prepared"]["sources"])==4 and len(details["development"])==7
    assert details["development"][-1]["status"]=="CONTRACT_REJECTED"
    trained,fit=execute(next(c for c in cells(cfg) if c["pilot_stage"]=="train" and c["pilot_arm"]=="C-R-N"),{"prepared":prepared})
    assert fit["training"]["updates"]==0 and fit["training"]["monitor_score"] is not None
    predicted,result=execute(next(c for c in cells(cfg) if c["pilot_stage"]=="predict" and c["pilot_arm"]=="C-R-N"),{"trained":trained})
    assert len(result["sources"])==7
    assert result["sources"][0]["horizons"][0]["paths"]==64
    assert result["sources"][-1]["status"]=="CONTRACT_REJECTED"
    assert store.manifest("artifact-"+predicted)["visibility"]=="restricted"


def test_global_exhaustion_delivers_complete_matrix_without_starting_workers(tmp_path):
    from application.walking_pilot import run_pilot
    from application.research_budget import BudgetLedger
    from application.walking_pilot_budget import charge_owner
    store,cfg=setup(tmp_path)
    charge_owner(BudgetLedger(store),cfg["pilot_id"],1799999,"prior-science-fixture")
    report={"report_path":str(tmp_path/"status.json"),"status":"RUNNING"}
    run_pilot(store,cfg,report)
    assert len(report["rows"])==10
    assert sum(len(s["horizons"]) for r in report["rows"] for s in r["development"])==210
    assert all(h["status"]!="NOT_EXECUTED" for r in report["rows"] for s in r["development"] for h in s["horizons"])
    assert not any(e["event_kind"]=="WORKER_STARTED" for e in store.events())
    assert report["summary_artifact_id"] and report["budget"]["global_ms"]>=1800000


def test_source_grants_do_not_substitute_for_exact_original_account_continuation(tmp_path):
    from application.walking_pilot import original_account_preflight
    store,cfg=setup(tmp_path)
    store.publish("synthetic-expired-continuation-receipt",{"arm_id":"pirc26-m0-o1",
                  "approval":{"manifest_id":"old-approval","sha256":digest("old")}})
    store.append("ARM_CONTINUATION",{"manifest_id":"synthetic-expired-continuation-receipt"})
    preflight(store,cfg)  # Source readiness is separate from one old account.
    with pytest.raises(ResearchError,match="exact live original-arm continuation"):
        original_account_preflight(store,cfg,"pirc26-m0-o1")
    original_account_preflight(store,cfg,cfg["arms"]["A-S"])
    moved={**cfg,"preparation_arm":"A-S"}
    assert all(c["arm_id"]==cfg["arms"]["A-S"] for c in cells(moved) if c["pilot_stage"] in {"source","prepare"})
    assert not any(e["event_kind"]=="READ_STARTED" for e in store.events())
