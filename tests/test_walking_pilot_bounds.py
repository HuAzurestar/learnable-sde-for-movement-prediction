"""DEV-14 artificial counterexamples and native supervised integration only."""
from copy import deepcopy
import json
import os
import sys

import pytest

from application.research_budget import BudgetLedger,BudgetSpec
from application.research_supervisor import ResearchSupervisor
from application.walking_pilot import ARMS,CODE_FILES,cells,read_internal
from application.walking_pilot_budget import usage
from application.walking_pilot_sources import read_population
from infrastructure.research_store import ResearchError,digest,encode
from tests.test_walking_pilot_sources import setup


def active(store,cfg,stage="source",arm=None):
    cell=next(c for c in cells(cfg) if c["pilot_stage"]==stage and (arm is None or c["pilot_arm"]==arm))
    run=store.register_run(cfg["study_id"],cell); attempt=store.new_attempt(run)
    store.transition(attempt,"RUNNING")
    return {"attempt_id":attempt,"run_id":run,"entrypoint":"walking-pilot-v1"}


def test_review_unlisted_artifact_and_unbound_consumer_refused_before_payload(tmp_path,monkeypatch):
    store,cfg=setup(tmp_path); source=cfg["sources"][0]
    grant=store.authorization(source["authorization_id"],version="v1")
    extra=store.artifact(encode({"synthetic_unlisted":True}),role="result",visibility="restricted",
                         block_ids=source["artifact"]["block_ids"],study_id=source["artifact"]["study_id"])
    consumer=active(store,cfg)
    reads=[]
    original=store._verified_artifact_content
    def read(metadata):
        reads.append(metadata["artifact_id"]); return original(metadata)
    monkeypatch.setattr(store,"_verified_artifact_content",read)
    for method in (store.read_artifact,store.verify_artifact_read):
        with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
            method(extra["artifact_id"],purpose="resume",authorization=grant,consumer=consumer)
    for bad in ({"attempt_id":"not-an-attempt","run_id":"not-a-run","entrypoint":"unrelated-consumer"},
                {**consumer,"run_id":"wrong"},{**consumer,"entrypoint":"unrelated-consumer"},{}):
        with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
            read_population(store,cfg,source,bad)
    assert not reads
    assert not any(e["event_kind"]=="READ_STARTED" for e in store.events())
    assert "infrastructure/research_store.py" in CODE_FILES


def test_actual_consumer_arm_stage_source_and_scope_rechecked_after_io(tmp_path,monkeypatch):
    store,cfg=setup(tmp_path); source=cfg["sources"][0]
    consumer=active(store,cfg)
    grant=store.authorization(source["authorization_id"],version="v1")
    # Narrower registered owner grants stay narrower even with a valid pilot.
    for changes in ({"consumer_arm_ids":["C-T-S"]},
                    {"consumer_scope":{**grant["consumer_scope"],"worker_stages":["train"]}},
                    {"consumer_scope":["human prose cannot authorize arbitrary workers"]},
                    {"pilot_study_id":"another-study"}):
        narrowed={**grant,**changes,"authorization_id":"narrow-"+digest(changes)}; store.authorize(narrowed)
        with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
            store.read_artifact(source["artifact"]["artifact_id"],purpose="resume",authorization=narrowed,consumer=consumer)
    with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
        read_population(store,cfg,cfg["sources"][1],consumer)  # same pooled bytes, wrong source cell
    train_consumer=active(store,cfg,"train","C-T-S")
    with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
        store.read_artifact(source["artifact"]["artifact_id"],purpose="resume",authorization=grant,consumer=train_consumer)
    original=store._append; touched=[]
    def append(kind,*args,**kwargs):
        value=original(kind,*args,**kwargs)
        if kind=="READ_COMPLETED" and not touched:
            touched.append(True); store.transition(consumer["attempt_id"],"FAILED",error_code="fixture-stop")
        return value
    monkeypatch.setattr(store,"_append",append)
    with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
        store.read_artifact(source["artifact"]["artifact_id"],purpose="resume",authorization=grant,consumer=consumer)
    assert touched and store.events()[-1]["event_kind"]=="EXPOSURE_DENIED"


def test_review_phase_omission_and_wrong_pilot_rejected_for_all_ten_arms(tmp_path):
    store,cfg=setup(tmp_path); ledger=BudgetLedger(store)
    for arm in ARMS:
        before=sum(e["event_kind"]=="RESERVE" for e in store.events())
        cell=next(c for c in cells(cfg) if c["pilot_stage"]=="train" and c["pilot_arm"]==arm)
        attempt=store.new_attempt(store.register_run(cfg["study_id"],cell))
        for spec in (BudgetSpec(121,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="predict"),
                     BudgetSpec(1801),BudgetSpec(1),
                     BudgetSpec(1,category="pilot",pilot_id="another-pilot",pilot_phase="train"),
                     BudgetSpec(120.001,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="train")):
            with pytest.raises(ResearchError): ledger.reserve(attempt,spec)
        assert sum(e["event_kind"]=="RESERVE" for e in store.events())==before
        # Actual monitoring is part of this training worker; retry shares its cap.
        first=ledger.reserve(attempt,BudgetSpec(120,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="train"))
        ledger.settle(first["reservation_id"],119000,outcome="TIMEOUT")
        store.transition(attempt,"TIMEOUT",error_code="TIMEOUT")
        retry=store.new_attempt(first["run_id"],parent_attempt_id=attempt,reason="synthetic train monitor retry")
        with pytest.raises(ResearchError,match="training limit"):
            ledger.reserve(retry,BudgetSpec(1.001,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="train"))
        last=ledger.reserve(retry,BudgetSpec(1,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="train"))
        ledger.settle(last["reservation_id"],1000,outcome="FAILED")
        store.transition(retry,"FAILED",error_code="fixture")
        with store._read_transaction(): assert usage(ledger,cfg["pilot_id"])["training_ms"][cfg["arms"][arm]]==120000
        prediction=next(c for c in cells(cfg) if c["pilot_stage"]=="predict" and c["pilot_arm"]==arm)
        pred=store.new_attempt(store.register_run(cfg["study_id"],prediction))
        r=ledger.reserve(pred,BudgetSpec(1,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="predict"))
        ledger.settle(r["reservation_id"],1000,outcome="SUCCEEDED")
        # Only reservations belonging to this iteration are added.
        assert ledger.balance(cfg["arms"][arm])["committed_ms"]==121000


def test_policy_study_and_new_arm_cannot_be_detached_from_cell_binding(tmp_path):
    store,cfg=setup(tmp_path); ledger=BudgetLedger(store)
    original=store.manifest("study-"+cfg["study_id"])["spec"]
    for label,arm,study_id,changes in (
        ("missing", "M0-UO2",cfg["study_id"],{"pilot_id":None}),
        ("wrong-arm","M2-UO2",cfg["study_id"],{"pilot_arm":"A-S"}),
        ("old-study","A-S","synthetic-foreign",{})):
        # Fresh registered studies give valid run manifests with bad pilot scope.
        spec=deepcopy(original); spec["study_id"]="synthetic-foreign" if label=="old-study" else "synthetic-"+label
        cell=next(c for c in cells(cfg) if c["pilot_stage"]=="train" and c["pilot_arm"]==arm)
        cell={**cell,**changes}; spec["cells"]=[cell]
        store.register(spec,digest(spec))
        attempt=store.new_attempt(store.register_run(spec["study_id"],cell))
        with pytest.raises(ResearchError): ledger.reserve(attempt,BudgetSpec(1))
        with pytest.raises(ResearchError): ledger.reserve(attempt,BudgetSpec(1,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="train"))
    assert not any(e["event_kind"]=="RESERVE" for e in store.events())


def test_pilot_owner_continuation_cannot_fund_old_study(tmp_path):
    from tests.test_research_budget import registered
    from tests.test_research_continuation import approved,closed
    store,attempts=registered(tmp_path,2); ledger=closed(store,attempts[0])
    ref=approved(store,pilot_study_id="only-the-new-pilot")
    ledger.continue_arm("affine",approval_ref=ref)
    count=sum(e["event_kind"]=="RESERVE" for e in store.events())
    with pytest.raises(ResearchError,match="restricted"):
        ledger.reserve(attempts[1],BudgetSpec(1))
    assert sum(e["event_kind"]=="RESERVE" for e in store.events())==count
    assert ledger.balance("affine")["committed_ms"]==500


def test_v2_templates_preserve_source_whitelists_and_owner_limits():
    from scripts.build_walking_pilot_grants import templates
    grant={"authorization_id":"source","version":"v1","study_id":"old-study","pilot_study_id":"new-study",
           "artifact_ids":["exact-source"],"consumer_arm_ids":list(ARMS),"consumer_scope":["original owner words"],
           "expires_at":"2026-11-03T23:59:00+08:00","purposes":["resume"],"test_authorization":False}
    doc={"pilot_study_id":"new-study","grants":[grant],"registered_at":"old-time","head":{"sequence":1}}
    internal={"grant":{**grant,"authorization_id":"internal","study_id":"new-study"}}
    before=deepcopy((doc,internal)); sources,new_internal=templates(doc,internal,"pilot")
    assert (doc,internal)==before
    for old,new in ((grant,sources["grants"][0]),(internal["grant"],new_internal)):
        assert all(new[k]==v for k,v in old.items() if k not in {"version","consumer_scope"})
        assert new["version"]=="v2"
    assert sources["grants"][0]["consumer_scope"]["worker_stages"]==["source"]
    assert new_internal["consumer_scope"]["controller"] is True
    assert sources["registration_status"]=="TEMPLATE_ONLY_NOT_REGISTERED"


@pytest.mark.skipif(os.name!="nt",reason="real native Windows Job Object integration")
def test_native_supervised_source_prepare_C_train_predict(tmp_path,monkeypatch):
    import application.research_supervisor as supervisor_module
    native_tree=supervisor_module.ProcessTree; trees=[]
    def observe_tree(process,**kwargs):
        trees.append({"pid":process.pid,**kwargs})
        return native_tree(process,**kwargs)
    monkeypatch.setattr(supervisor_module,"ProcessTree",observe_tree)
    store,cfg=setup(tmp_path,train_steps=1)
    supervisor=ResearchSupervisor(store); results=[]
    controller={"controller":True,"pilot_id":cfg["pilot_id"],"entrypoint":"walking-pilot-v1"}
    def execute(cell,inputs):
        run=store.register_run(cfg["study_id"],cell); attempt=store.new_attempt(run)
        phase="prepare" if cell["pilot_stage"]=="source" else cell["pilot_stage"]
        def command(output):
            return [sys.executable,"-m","application.walking_pilot","worker","--runtime-root",str(store.path.parent),
                    "--store-id",store.store_id,"--pilot-id",cfg["pilot_id"],"--attempt",attempt,
                    "--inputs",json.dumps(inputs),"--output",str(output)]
        result=supervisor.run(attempt,command,BudgetSpec(100 if phase=="train" else 90,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase=phase),
                              resource_plan={"maximum_result_bytes":4*1024**2,"tree_memory_bytes":2*1024**3})
        results.append({"stage":cell["pilot_stage"],"attempt":attempt,**result})
        assert result["state"]=="SUCCEEDED",result
        return result["artifact_id"],read_internal(store,cfg,result["artifact_id"],controller)
    source_refs={}
    for source in (cfg["sources"][0],cfg["sources"][4]):
        cell=next(c for c in cells(cfg) if c.get("source_id")==source["source_id"])
        artifact,_=execute(cell,{})
        source_refs[source["source_id"]]=artifact
    prepared,_=execute(next(c for c in cells(cfg) if c["pilot_stage"]=="prepare"),{"sources":source_refs})
    trained,fit=execute(next(c for c in cells(cfg) if c["pilot_stage"]=="train" and c["pilot_arm"]=="C-T-S"),{"prepared":prepared})
    assert fit["training"]["updates"]==1 and fit["training"]["monitor_score"] is not None
    _,forecast=execute(next(c for c in cells(cfg) if c["pilot_stage"]=="predict" and c["pilot_arm"]=="C-T-S"),{"trained":trained})
    assert forecast["sources"][0]["horizons"][0]["paths"]==64
    events=store.events()
    assert sum(e["event_kind"]=="WORKER_STARTED" for e in events)==5
    assert sum(e["event_kind"]=="WORKER_TREE_STOPPED" for e in events)==5
    assert len(trees)==5 and all(t["memory_bytes"]==2*1024**3 for t in trees)
    (tmp_path/"native-evidence.json").write_bytes(encode({"scope":"synthetic DEV only","results":results,
        "torch_imported_by":"application.walking_pilot.worker","tree_memory_bytes":2*1024**3,
        "native_trees":trees,"budget":usage(BudgetLedger(store),cfg["pilot_id"])}))
