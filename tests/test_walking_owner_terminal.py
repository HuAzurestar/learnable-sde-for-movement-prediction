"""DEV-only deterministic billing/recovery regressions; synthetic stores."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import pytest
from application.research_budget import BudgetLedger
from application.walking_pilot_budget import OwnerMeter,owner_session,usage,remaining,verified_report
from infrastructure.research_store import ResearchError,encode,ResearchStore
from tests.test_walking_pilot_budget import fixture,budget


def meter_fixture(tmp_path):
    store,arms,attempts=fixture(tmp_path)
    clock=[0.]
    cfg={"pilot_id":"synthetic-pilot"}
    meter=OwnerMeter(BudgetLedger(store),cfg,0.,lambda:clock[0],"invocation-one")
    report={"report_path":str(tmp_path/"report.json"),"rows":[],"status":"LIMITED_RESULT"}
    def save():
        clock[0]+=1
        raw=encode(report); Path(report["report_path"]).write_bytes(raw)
        return raw
    return store,arms,attempts,clock,meter,report,save


def test_clock_covers_checkpoints_usage_report_and_discloses_tail(tmp_path,monkeypatch):
    store,arms,attempts,c,m,r,save=meter_fixture(tmp_path)
    original=store._append
    def append(kind,*a,**kw):
        result=original(kind,*a,**kw)
        c[0]+=2 if kind=="PILOT_OWNER_TERMINAL" else 1
        return result
    monkeypatch.setattr(store,"_append",append)
    with owner_session(m.ledger,"synthetic-pilot",None,m.invocation_id):
        c[0]+=1
        m.checkpoint()  # open cost + ordinary work
        c[0]+=1
        receipt=m.terminal(r,save)
    u=usage(m.ledger,"synthetic-pilot")
    assert u["owner_ms"]==5000 and not u["pending"]
    assert c[0]==7 and receipt["payload"]["global_ms"]==5000
    assert verified_report(m.ledger,"synthetic-pilot",r["report_path"])["receipt"]==receipt
    disk=json.loads(Path(r["report_path"]).read_bytes())
    assert disk["settlement_pending"] and disk["settlement_tail"]["included_in_qualification_wall"]
    assert disk["budget"]["global_ms"]<receipt["payload"]["global_ms"]


def test_checkpoint_uncertain_append_adopts_exact_id_once(tmp_path,monkeypatch):
    store,arms,attempts,c,m,r,save=meter_fixture(tmp_path)
    original=store._append; failed=[]
    def append(kind,*a,**kw):
        result=original(kind,*a,**kw)
        if kind=="PILOT_OWNER_CHECKPOINT" and not failed:
            failed.append(True); raise OSError("after durable append")
        return result
    with owner_session(m.ledger,"synthetic-pilot",None,m.invocation_id):
        monkeypatch.setattr(store,"_append",append); c[0]=1
        with pytest.raises(OSError): m.checkpoint()
        assert m.billed==0
        c[0]=2; m.checkpoint()
        assert m.billed==1000
        m.terminal(r,save)
    assert usage(m.ledger,"synthetic-pilot")["owner_ms"]==3000
    assert len([e for e in store.events() if e["event_kind"]=="PILOT_OWNER_CHECKPOINT"])==1


@pytest.mark.parametrize("where",["open","report"])
def test_crash_without_receipt_retains_unknown_and_blocks_new_owner_reserve_activation(tmp_path,where):
    store,arms,attempts,c,m,r,save=meter_fixture(tmp_path)
    with pytest.raises(RuntimeError):
        with owner_session(m.ledger,"synthetic-pilot",None,m.invocation_id):
            if where=="report": save()
            raise RuntimeError("simulated abrupt controller loss")
    assert usage(m.ledger,"synthetic-pilot")["pending"]
    with pytest.raises(ResearchError,match="RECOVERY_REQUIRED"):
        remaining(m.ledger,"synthetic-pilot",arms["A-S"],"train")
    with pytest.raises(ResearchError,match="RECOVERY_REQUIRED"):
        m.ledger.reserve(attempts[arms["A-S"]][0],budget(1))
    with pytest.raises(ResearchError,match="RECOVERY_REQUIRED"):
        with owner_session(m.ledger,"synthetic-pilot",None,"other"): pass
    from application.walking_execution import activate_execution
    with pytest.raises(ResearchError,match="RECOVERY_REQUIRED"):
        activate_execution(store,{"budget_pilot_id":"synthetic-pilot"})


def test_append_then_exit_failure_reopens_and_adopts_terminal_without_recharge(tmp_path,monkeypatch):
    store,arms,attempts,c,m,r,save=meter_fixture(tmp_path)
    original=store._read_transaction
    @contextmanager
    def transaction():
        with original(): yield
        if any(e["event_kind"]=="PILOT_OWNER_TERMINAL" for e in store._events()):
            raise OSError("exit failed after durable append")
    with owner_session(m.ledger,"synthetic-pilot",None,m.invocation_id):
        monkeypatch.setattr(store,"_read_transaction",transaction)
        with pytest.raises(OSError): m.terminal(r,save)
    monkeypatch.setattr(store,"_read_transaction",original)
    reopened=BudgetLedger(ResearchStore(store.path.parent,store.store_id))
    first=verified_report(reopened,"synthetic-pilot",r["report_path"])
    second=verified_report(reopened,"synthetic-pilot",r["report_path"])
    assert first==second and usage(reopened,"synthetic-pilot")["owner_ms"]==1000
    Path(r["report_path"]).write_bytes(b'{}')
    with pytest.raises(ResearchError): verified_report(reopened,"synthetic-pilot",r["report_path"])


def test_stage_deduction_once_and_unknown_never_zero(tmp_path):
    store,arms,attempts,c,m,r,save=meter_fixture(tmp_path)
    with owner_session(m.ledger,"synthetic-pilot",None,m.invocation_id):
        reserved=m.ledger.reserve(attempts[arms["A-S"]][0],budget(1))
        m.ledger.settle(reserved["reservation_id"],500,outcome="SUCCEEDED")
        m.stage(attempts[arms["A-S"]][0]); m.stage(attempts[arms["A-S"]][0])
        c[0]=1; m.terminal(r,save)
    assert sum(m.stages.values())==500
    assert usage(m.ledger,"synthetic-pilot")["owner_ms"]==1500
    assert usage(m.ledger,"synthetic-pilot")["global_ms"]==2000


@pytest.mark.parametrize("failure",[None,"UNAUTHORIZED_DATA","CORRUPT_ARTIFACT"])
def test_controller_preflight_normal_fatal_natural_cli(tmp_path,monkeypatch,capsys,failure):
    from tests.test_walking_pilot_sources import setup
    import application.walking_pilot as pilot
    store,cfg=setup(tmp_path,include_runtime=True)
    monkeypatch.setattr(pilot,"cells",lambda c:[])
    if failure:
        def fail(*a): raise ResearchError(failure,"injected preflight failure")
        monkeypatch.setattr(pilot,"preflight",fail)
    cfg["runtime_root"]=str(store.path.parent); cfg["store_id"]=store.store_id
    # Empty dispatch matrix exercises only owner/CLI termination; full native
    # preflight and one-source CLI are covered separately below.
    if failure is None:
        monkeypatch.setattr(pilot,"preflight",lambda s,c:BudgetLedger(s))
    config_path=tmp_path/"cli.json"; config_path.write_bytes(encode(cfg))
    path=tmp_path/"status.json"
    code=pilot.main(["run","--config",str(config_path),"--report",str(path)])
    assert code==(2 if failure else 0)
    result=verified_report(BudgetLedger(store),cfg["pilot_id"],path)
    assert result["report"]["status"]==(failure or "LIMITED_RESULT")
    assert not usage(BudgetLedger(store),cfg["pilot_id"])["pending"]
    assert json.loads(capsys.readouterr().out)["terminal_receipt_id"]



def test_unknown_settle_cannot_close_owner(tmp_path):
    store,arms,attempts,c,m,r,save=meter_fixture(tmp_path)
    with owner_session(m.ledger,"synthetic-pilot",None,m.invocation_id):
        res=m.ledger.reserve(attempts[arms["A-S"]][0],budget(1))
        m.ledger.settle(res["reservation_id"],None,outcome="INTERRUPTED")
        with pytest.raises(ResearchError): m.stage(attempts[arms["A-S"]][0])
        with pytest.raises(ResearchError): m.terminal(r,save)
    assert usage(m.ledger,"synthetic-pilot")["pending"]


def test_historical_success_reuse_does_not_subtract_old_stage(tmp_path,monkeypatch):
    from tests.test_walking_pilot_sources import setup
    import application.walking_pilot as pilot
    store,cfg=setup(tmp_path); cell=pilot.cells(cfg)[0]
    run=store.register_run(cfg["study_id"],cell); attempt=store.new_attempt(run)
    ledger=BudgetLedger(store)
    res=ledger.reserve(attempt,__import__("application.research_budget",fromlist=["BudgetSpec"]).BudgetSpec(
        1,category="pilot",pilot_id=cfg["pilot_id"],pilot_phase="prepare"))
    ledger.settle(res["reservation_id"],500,outcome="SUCCEEDED")
    store.transition(attempt,"RUNNING")
    artifact=store.artifact(encode({"synthetic":True}),role="result",visibility="restricted",study_id=cfg["study_id"],block_ids=[cell["block_id"]])
    store.transition(attempt,"SUCCEEDED",artifact_id=artifact["artifact_id"])
    monkeypatch.setattr(pilot,"cells",lambda c:[cell])
    monkeypatch.setattr(pilot,"preflight",lambda s,c:BudgetLedger(s))
    report={"report_path":str(tmp_path/"reuse.json"),"status":"RUNNING"}
    pilot.run_pilot(store,cfg,report)
    verified=verified_report(ledger,cfg["pilot_id"],report["report_path"])
    assert verified["receipt"]["payload"]["stage_deductions_ms"]=={}
    assert verified["report"]["sources"][0]["reused"]
    assert len(store.attempts())==1


def test_foreground_native_source_controller_cli_returns_and_receipt_verifies(tmp_path):
    import subprocess,sys,time
    from tests.test_walking_pilot_sources import setup
    store,cfg=setup(tmp_path,include_runtime=True)
    config=tmp_path/"config.json"; config.write_bytes(encode(cfg))
    report=tmp_path/"natural.json"
    # Artificial one-source matrix selection only; the real preflight still
    # validates the complete immutable 32-cell study and all eleven sources.
    script="""import sys,time,json,pathlib
import application.walking_pilot as p
from application.walking_pilot_budget import OwnerMeter
samples=[]
original_sample=OwnerMeter.sample
def sampled(self):
    value=original_sample(self)
    samples.append(time.monotonic())
    return value
OwnerMeter.sample=sampled
original_cells=p.cells
original_preflight=p.preflight
def checked(s,c):
    selected=p.cells
    p.cells=original_cells
    try: return original_preflight(s,c)
    finally: p.cells=selected
p.preflight=checked
p.cells=lambda c:original_cells(c)[:1]
code=p.main(['run','--config',sys.argv[1],'--report',sys.argv[2]])
returned=time.monotonic()
# External DEV harness writes measurement AFTER the CLI function returns;
# this is outside the product's t* business-I/O promise and separately timed.
pathlib.Path(sys.argv[2]+'.tail.json').write_text(json.dumps({'sample_to_main_return_s':returned-samples[-1]}))
raise SystemExit(code)
"""
    before=time.monotonic()
    result=subprocess.run([sys.executable,"-B","-c",script,str(config),str(report)],capture_output=True,text=True,timeout=180)
    wall=time.monotonic()-before
    assert result.returncode==0,(result.stdout,result.stderr)
    verified=verified_report(BudgetLedger(store),cfg["pilot_id"],report)
    assert verified["report"]["sources"][0]["state"]=="SUCCEEDED"
    assert verified["receipt"]["payload"]["stage_deductions_ms"]
    events=store.events()
    assert any(e["event_kind"]=="WORKER_TREE_STOPPED" for e in events)
    assert not usage(BudgetLedger(store),cfg["pilot_id"])["pending"]
    assert wall*1000>=verified["receipt"]["payload"]["global_ms"]
    (tmp_path/"natural-cli-measurement.json").write_bytes(encode({"wall_s":wall,"tail":json.loads(Path(str(report)+".tail.json").read_text()),"receipt":verified["receipt"],"cli":json.loads(result.stdout)}))
