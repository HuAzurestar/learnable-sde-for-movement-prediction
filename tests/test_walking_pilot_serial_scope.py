"""DEV regressions for the diagnosed writer/read interleaving and fatal frames."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

import pytest

from application.research_budget import BudgetLedger
from application.research_supervisor import read_walking_error,write_walking_error
from application.walking_pilot import preflight
from application.walking_pilot_budget import policy
from infrastructure.research_store import ResearchError,encode
from tests.test_walking_pilot_budget import fixture
from tests.test_walking_pilot_sources import setup


@pytest.mark.parametrize("fixed",[False,True])
def test_cross_process_writer_at_old_list_new_head_boundary(tmp_path,monkeypatch,fixed):
    store,_,_=fixture(tmp_path)
    # Load the exact baseline public policy for the negative control.
    source=subprocess.check_output(["git","show","2f3b15c:application/walking_pilot_budget.py"],text=True)
    baseline={}; exec(compile(source,"baseline-walking-budget","exec"),baseline)
    code=("from pathlib import Path; import sys; "
          "from infrastructure.research_store import ResearchStore; "
          "s=ResearchStore(Path(sys.argv[1]),'walking-fixture',lock_wait_seconds=5); "
          "print('READY',flush=True); sys.stdin.readline(); print('TRYING',flush=True); "
          "s.append('HEARTBEAT',{'attempt_id':'artificial-only','monotonic_elapsed_ms':1}); "
          "print('DONE',flush=True)")
    child=subprocess.Popen([sys.executable,"-c",code,str(tmp_path)],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    original=store._json; fired=[]
    try:
        assert child.stdout.readline().strip()=="READY"
        def at_head(path):
            if Path(path)==store.path/"head.json" and not fired:
                fired.append(True); child.stdin.write("GO\n"); child.stdin.flush()
                assert child.stdout.readline().strip()=="TRYING"
                if fixed:
                    time.sleep(.08)
                    assert child.poll() is None
                else:
                    out,err=child.communicate(timeout=10)
                    assert child.returncode==0 and "DONE" in out,err
            return original(path)
        monkeypatch.setattr(store,"_json",at_head)
        if fixed:
            assert policy(BudgetLedger(store),"synthetic-pilot")["pilot_id"]=="synthetic-pilot"
            out,err=child.communicate(timeout=10)
            assert child.returncode==0 and "DONE" in out,err
        else:
            with pytest.raises(ResearchError,match="authoritative tail lost or changed"):
                baseline["policy"](BudgetLedger(store),"synthetic-pilot")
        assert fired and store.events()[-1]["event_kind"]=="HEARTBEAT"
    finally:
        if child.poll() is None:
            child.kill(); child.communicate(timeout=5)


@pytest.mark.parametrize("damage",["head","gap","manifest"])
def test_preflight_still_rejects_real_corruption(tmp_path,damage):
    store,cfg=setup(tmp_path)
    if damage=="head":
        head=json.loads((store.path/"head.json").read_bytes()); head["hash"]="f"*64
        (store.path/"head.json").write_bytes(encode(head))
    elif damage=="gap":
        (store.path/"events/0000000000000001.json").unlink()
    else:
        (store.path/"manifests/walking-config-fixture-pilot.json").write_bytes(b'{}')
    with pytest.raises(ResearchError,match="CORRUPT_ARTIFACT"):
        preflight(store,cfg)


@pytest.mark.parametrize("change",["attempt","stage","large","broken","duplicate","extra"])
def test_error_frame_is_bounded_and_bound_to_registered_attempt(tmp_path,change):
    result=tmp_path/"result.json"
    write_walking_error(result,"attempt-one","source","CORRUPT_ARTIFACT")
    path=tmp_path/"worker-error.json"; frame=json.loads(path.read_bytes())
    if change in {"attempt","stage"}:
        frame["attempt_id" if change=="attempt" else "stage"]="wrong"
        path.write_bytes(encode(frame))
    elif change=="large": path.write_bytes(b" "*1025)
    elif change=="broken": path.write_bytes(b'{')
    elif change=="duplicate": path.write_bytes(path.read_bytes()[:-1]+b',"code":"CORRUPT_ARTIFACT"}')
    else:
        frame["payload"]="must never be accepted"; path.write_bytes(encode(frame))
    assert read_walking_error(result,"attempt-one","source")=="CHILD_ERROR_INVALID"


def test_worker_initialization_error_has_safe_structured_frame(tmp_path):
    root=tmp_path/"absent-runtime"; result=tmp_path/"result.json"
    run=subprocess.run([sys.executable,"-m","application.walking_pilot","worker",
        "--runtime-root",str(root),"--store-id","missing","--pilot-id","missing",
        "--attempt","attempt-one","--worker-stage","source","--inputs","{}","--output",str(result)],
        capture_output=True,timeout=20)
    assert run.returncode==2
    assert read_walking_error(result,"attempt-one","source")=="STORE_MISSING"
    assert not run.stderr and not result.exists()


def test_preflight_does_not_admit_after_final_scan_crosses_expiry(tmp_path,monkeypatch):
    import application.walking_pilot_budget as budget
    store,cfg=setup(tmp_path); original=store._json; heads=[]
    class Expired(datetime):
        @classmethod
        def now(cls,tz=None):
            return datetime(2026,11,4,tzinfo=timezone.utc)
    def at_head(path):
        value=original(path)
        if Path(path)==store.path/"head.json":
            heads.append(True)
            if len(heads)==2:
                monkeypatch.setattr(budget,"datetime",Expired)
        return value
    monkeypatch.setattr(store,"_json",at_head)
    with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
        preflight(store,cfg)
    assert len(heads)==2


def test_native_error_is_read_only_after_stop_and_settled(tmp_path):
    from application.research_supervisor import ResearchSupervisor
    from tests.test_walking_pilot_budget import budget
    store,arms,attempts=fixture(tmp_path)
    attempt=attempts[arms["A-S"]][0]
    code=("import sys; from application.research_supervisor import write_walking_error; "
          "write_walking_error(sys.argv[1],sys.argv[2],'train','CORRUPT_ARTIFACT'); sys.exit(2)")
    result=ResearchSupervisor(store).run(attempt,
        lambda output:[sys.executable,"-c",code,str(output),attempt],budget(10),
        resource_plan={"maximum_result_bytes":4*1024**2,"tree_memory_bytes":2*1024**3})
    assert result["state"]=="FAILED" and result["error_code"]=="CORRUPT_ARTIFACT"
    kinds=[e["event_kind"] for e in store.events()]
    assert kinds.index("WORKER_TREE_STOPPED")<kinds.index("SETTLE")
    assert not usage_pending(store)


def usage_pending(store):
    from application.walking_pilot_budget import usage
    return usage(BudgetLedger(store),"synthetic-pilot")["pending"]


def test_fatal_child_stops_controller_dispatch_and_preserves_missing_matrix(tmp_path,monkeypatch):
    import application.research_supervisor as supervisor
    from application.walking_pilot import run_pilot
    store,cfg=setup(tmp_path); dispatched=[]
    def fatal(self,attempt,*args,**kwargs):
        dispatched.append(attempt)
        self.store.transition(attempt,"FAILED",error_code="CORRUPT_ARTIFACT")
        return {"state":"FAILED","artifact_id":None,"error_code":"CORRUPT_ARTIFACT"}
    monkeypatch.setattr(supervisor.ResearchSupervisor,"run",fatal)
    report={"report_path":str(tmp_path/"report.json"),"status":"RUNNING"}
    with pytest.raises(ResearchError,match="CORRUPT_ARTIFACT"):
        run_pilot(store,cfg,report)
    assert len(dispatched)==1 and report["status"]=="CORRUPT_ARTIFACT"
    horizons=[h for row in report["rows"] for source in row["development"] for h in source["horizons"]]
    assert len(horizons)==210 and all(h["status"]=="CORRUPT_ARTIFACT" for h in horizons)


@pytest.mark.parametrize("code",["CORRUPT_ARTIFACT","WRITER_BUSY"])
def test_restarted_controller_stops_on_reused_fatal_attempt(tmp_path,monkeypatch,code):
    import application.research_supervisor as supervisor
    from application.walking_pilot import cells,run_pilot
    store,cfg=setup(tmp_path)
    run=store.register_run(cfg["study_id"],cells(cfg)[0])
    attempt=store.new_attempt(run)
    store.transition(attempt,"FAILED",error_code=code)
    def forbidden(*args,**kwargs):
        pytest.fail("historical fatal must stop before any worker dispatch")
    monkeypatch.setattr(supervisor.ResearchSupervisor,"run",forbidden)
    report={"report_path":str(tmp_path/"restarted-report.json"),"status":"RUNNING"}
    with pytest.raises(ResearchError,match=code):
        run_pilot(store,cfg,report)
    assert len(store.attempts())==1 and not any(e["event_kind"]=="WORKER_STARTED" for e in store.events())
    assert report["status"]==code and report["sources"][0]["status"]==code
    horizons=[h for row in report["rows"] for source in row["development"] for h in source["horizons"]]
    assert len(horizons)==210 and all(h["status"]==code for h in horizons)


def test_busy_claim_releases_admission_scope_and_rechecks_current_closure(tmp_path,monkeypatch):
    import application.research_supervisor as supervisor
    from application.research_queue import ResearchQueue
    from tests.test_walking_pilot_budget import budget
    store,arms,attempts=fixture(tmp_path); calls=[]; original=ResearchQueue.claim
    def claim(queue,reservation,**kwargs):
        calls.append(queue.store._read_snapshot() is not None)
        if len(calls)==1:
            return None
        return original(queue,reservation,**kwargs)
    def wait(seconds):
        assert store._read_snapshot() is None,"queue wait borrowed admission authority"
        store.append("ARM_CLOSED",{"arm_id":arms["A-S"],"reason":"artificial-closure-while-queued"})
    monkeypatch.setattr(ResearchQueue,"claim",claim)
    monkeypatch.setattr(supervisor.time,"sleep",wait)
    with pytest.raises(ResearchError,match="BUDGET_EXHAUSTED"):
        supervisor.ResearchSupervisor(store).run(attempts[arms["A-S"]][0],lambda output:[],budget(10))
    assert calls==[True,False]
    assert not any(e["event_kind"]=="WORKER_STARTED" for e in store.events())
    assert not usage_pending(store)


def test_controller_register_and_new_attempt_share_short_scope(tmp_path,monkeypatch):
    import application.research_supervisor as supervisor
    from application.walking_pilot import run_pilot
    store,cfg=setup(tmp_path); scopes=[]
    original_register=store.register_run; original_attempt=store.new_attempt
    def register(*args,**kwargs):
        scopes.append(store._read_snapshot()); return original_register(*args,**kwargs)
    def new_attempt(*args,**kwargs):
        scopes.append(store._read_snapshot()); return original_attempt(*args,**kwargs)
    def fatal(*args,**kwargs):
        assert store._read_snapshot() is None,"controller held registration lock across supervisor"
        return {"state":"FAILED","artifact_id":None,"error_code":"CORRUPT_ARTIFACT"}
    monkeypatch.setattr(store,"register_run",register)
    monkeypatch.setattr(store,"new_attempt",new_attempt)
    monkeypatch.setattr(supervisor.ResearchSupervisor,"run",fatal)
    with pytest.raises(ResearchError,match="CORRUPT_ARTIFACT"):
        run_pilot(store,cfg,{"report_path":str(tmp_path/"scope-report.json"),"status":"RUNNING"})
    assert len(scopes)==2 and scopes[0] is not None and scopes[0] is scopes[1]


@pytest.mark.parametrize("fault,expected",[("hold","RECOVERY_REQUIRED"),("closure","BUDGET_EXHAUSTED")])
def test_current_hold_or_closure_after_claim_vetoes_go(tmp_path,monkeypatch,fault,expected):
    import application.research_supervisor as supervisor
    from tests.test_walking_pilot_budget import budget
    store,arms,attempts=fixture(tmp_path); native=supervisor.ProcessTree; created=[]
    marker=tmp_path/"must-not-run"
    def tree(process,**kwargs):
        value=native(process,**kwargs); created.append(process.pid)
        if fault=="hold":
            (store.path/"recovery-hold.json").write_bytes(encode({"reason":"artificial-hold-after-claim"}))
        else:
            store.append("ARM_CLOSED",{"arm_id":arms["A-S"],"reason":"artificial-close-after-claim"})
        return value
    monkeypatch.setattr(supervisor,"ProcessTree",tree)
    with pytest.raises(ResearchError,match=expected):
        supervisor.ResearchSupervisor(store).run(attempts[arms["A-S"]][0],
            lambda output:[sys.executable,"-c","import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",str(marker)],
            budget(30),resource_plan={"maximum_result_bytes":4*1024**2,"tree_memory_bytes":2*1024**3})
    assert created and not marker.exists()
    events=store.events()
    assert not any(e["event_kind"]=="WORKER_STARTED" for e in events)
    assert any(e["event_kind"]=="WORKER_TREE_STOPPED" for e in events)
    assert not usage_pending(store)
