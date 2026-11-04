"""DEV P1: execution limits must not erase a validated fatal stop signal."""
import json
from pathlib import Path
import sys
import time

import pytest

import application.research_supervisor as supervisor
from application.research_budget import BudgetLedger
from application.walking_pilot import cells,run_pilot
from application.walking_pilot_budget import usage
from infrastructure.research_store import ResearchError,encode
from tests.test_walking_pilot_budget import fixture,budget
from tests.test_walking_pilot_sources import setup


@pytest.mark.parametrize("limit,diagnosis",[
    ("deadline","CORRUPT_ARTIFACT"),("deadline","WRITER_BUSY"),("deadline","WORKER_FAILED"),
    ("budget","CORRUPT_ARTIFACT"),("budget",None)])
def test_native_limit_preserves_execution_fact_and_validated_fatal(tmp_path,monkeypatch,limit,diagnosis):
    store,arms,attempts=fixture(tmp_path); attempt=attempts[arms["A-S"]][0]
    offset=[0.]; observed=[]; requested=[]; original_read=supervisor.read_walking_error
    def read(*args):
        code=original_read(*args)
        assert any(e["event_kind"]=="WORKER_TREE_STOPPED" for e in store.events())
        observed.append(code)
        if limit=="deadline":
            # Real native stop/charge precede this emulated owner processing
            # delay. Do not wait a long scientific allocation to cross it.
            offset[0]+=30.
        return code
    monkeypatch.setattr(supervisor,"read_walking_error",read)
    if limit=="budget":
        original_append=store._append
        def append(kind,*args,**kwargs):
            event=original_append(kind,*args,**kwargs)
            if kind=="CHECKPOINT_REQUESTED" and not requested:
                requested.append(True)
                # The real checkpoint collector observes a fresh closed budget
                # and sets budget_stop_requested before reading the fatal frame.
                store.append("ARM_CLOSED",{"arm_id":arms["A-S"],"reason":"DEV budget/fatal collision"})
            return event
        monkeypatch.setattr(store,"_append",append)
    code="import sys,time; from application.research_supervisor import write_walking_error; "
    if diagnosis is not None:
        code+=f"write_walking_error(sys.argv[1],sys.argv[2],'train',{diagnosis!r}); "
    code+="time.sleep(30)" if limit=="budget" else "sys.exit(2)"
    before=time.perf_counter()
    result=supervisor.ResearchSupervisor(store,monotonic=lambda:time.monotonic()+offset[0]).run(
        attempt,lambda output:[sys.executable,"-c",code,str(output),attempt],budget(10),
        resource_plan={"maximum_result_bytes":4*1024**2,"tree_memory_bytes":2*1024**3},
        checkpoint_handler=(lambda *args:pytest.fail("closed budget must veto checkpoint handling")) if limit=="budget" else None)
    state="TIMEOUT" if limit=="deadline" else "BUDGET_EXHAUSTED"
    expected_code=diagnosis if diagnosis in supervisor.WALKING_FATAL_CODES else state
    persisted=store.attempts()[attempt]; events=store.events()
    reservations=BudgetLedger(store)._state()[0]
    settled=next(r for r in reservations.values() if r["attempt_id"]==attempt)
    stopped=next(e["payload"] for e in events if e["event_kind"]=="WORKER_TREE_STOPPED")
    assert observed==[diagnosis]
    assert result["state"]==persisted["state"]==settled["outcome"]==state
    assert result["error_code"]==persisted["error_code"]==expected_code
    assert result["artifact_id"] is None
    assert result["elapsed_ms"]==settled["charged_ms"]==stopped["observed_elapsed_ms"]
    assert settled["settled"] and not usage(BudgetLedger(store),"synthetic-pilot")["pending"]
    assert bool(requested)==(limit=="budget")
    (tmp_path/"fatal-limit-evidence.json").write_bytes(encode({
        "scope":"artificial DEV native worker; injected post-stop clock only for deadline case",
        "limit":limit,"diagnosis":diagnosis,"result":result,"attempt_state":persisted,
        "settlement":settled,"tree_stopped":stopped,"budget_stop_requested_path":bool(requested),
        "wall_s":time.perf_counter()-before,"pending":False}))


@pytest.mark.parametrize("state",["TIMEOUT","BUDGET_EXHAUSTED"])
@pytest.mark.parametrize("history",[False,True])
def test_controller_stops_current_and_reused_limit_with_fatal(tmp_path,monkeypatch,state,history):
    store,cfg=setup(tmp_path); calls=[]
    if history:
        run=store.register_run(cfg["study_id"],cells(cfg)[0]); attempt=store.new_attempt(run)
        store.transition(attempt,state,error_code="CORRUPT_ARTIFACT")
    def returned(self,attempt,*args,**kwargs):
        if history:
            pytest.fail("historical fatal must prevent any new Supervisor dispatch")
        calls.append(attempt)
        store.transition(attempt,state,error_code="CORRUPT_ARTIFACT")
        return {"state":state,"error_code":"CORRUPT_ARTIFACT","artifact_id":None}
    monkeypatch.setattr(supervisor.ResearchSupervisor,"run",returned)
    report={"report_path":str(tmp_path/"status.json"),"status":"RUNNING"}
    with pytest.raises(ResearchError,match="CORRUPT_ARTIFACT"):
        run_pilot(store,cfg,report)
    assert len(calls)==(0 if history else 1)
    assert len(store.attempts())==1
    assert next(iter(store.attempts().values()))["state"]==state
    assert report["status"]=="CORRUPT_ARTIFACT"
    horizons=[h for r in report["rows"] for s in r["development"] for h in s["horizons"]]
    assert len(horizons)==210 and all(h["status"]=="CORRUPT_ARTIFACT" for h in horizons)
