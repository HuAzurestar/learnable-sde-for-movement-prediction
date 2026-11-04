"""Artificial native detector/stop/accounting probes; no research payload."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import pytest

from application.research_budget import BudgetLedger
from application.research_supervisor import ResearchSupervisor
from application.walking_pilot_budget import usage
from infrastructure.research_store import file_lock
from tests.test_walking_pilot_budget import fixture, budget
from tests.walking_harness import ExactWindowsProcess, creation_receipt
from tests.walking_creation_proof import CreationProof


@pytest.mark.parametrize("mutation",["hold","append","writer","stall","deadline","grandchild","fatal-timeout",
                                    "duplicate-result","large-result","hardlink-result","heartbeat-loss","closed","fatal-closed"])
def test_native_b1_stops_before_authority_review_and_settles(tmp_path,monkeypatch,mutation):
    store,arms,attempts=fixture(tmp_path)
    arm=arms["A-S"]; attempt=attempts[arm][2]
    paths={}; observed=[]; timings={}; changed=threading.Event(); cancel=threading.Event()
    owned=[]; identity_errors=[]; creation=CreationProof()
    nonce=uuid.uuid4().hex
    real_popen=subprocess.Popen
    def substitute(args,*a,**kw):
        if len(args)>6 and args[5:7]==["-m","application.walking_compute"]:
            code="import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.with_name('active').write_text('ready'); "
            if mutation=="grandchild":
                # Publish a stable identity and wait for the parent to hold its
                # exact live handle before publishing active. No deadline reset.
                code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\n"
                code += "import subprocess,json; from tests.walking_harness import ExactWindowsProcess, creation_receipt\n"
                code += "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                code += "identity=ExactWindowsProcess(c.pid)\n"
                code += "p.with_name('grandchild.pending').write_text(json.dumps({'pid':c.pid,'birth':identity.birth,'nonce':"+repr(nonce)+"})); p.with_name('grandchild.pending').replace(p.with_name('grandchild.ready'))\n"
                code += "deadline="+repr(paths['deadline'])+"\n"
                code += "while not p.with_name('grandchild.ack').exists():\n    if time.monotonic()>=deadline: raise RuntimeError('creation acknowledgement deadline')\n    time.sleep(.005)\n"
                code += "identity.close(); p.with_name('active').write_text('ready'); "

            if mutation in {"fatal-timeout","fatal-closed"}:
                code+="import json; p.with_name('worker-error.json').write_text(json.dumps("+repr({"schema":"walking-worker-error-v1","attempt_id":attempt,"stage":"predict","code":"CORRUPT_ARTIFACT"})+")); "
            if mutation=="duplicate-result": code+="p.write_text('{\"ok\":1,\"ok\":2}')"
            elif mutation=="large-result": code+="p.write_bytes(b' '* (4*1024**2+1))"
            elif mutation=="hardlink-result": code+="import os; q=p.with_name('linked-target.json'); q.write_text('{}'); os.link(q,p)"
            else: code+="time.sleep(60)"
            args=args[:4]+[sys.executable,"-c",code,str(paths["output"])]
        return real_popen(args,*a,**kw)
    monkeypatch.setattr(subprocess,"Popen",substitute)
    def prepare(output,deadline):
        paths["output"]=output
        paths["deadline"]=deadline
        # Parent read cost must be part of this funded stage, exactly once.
        time.sleep(.075)
        def verify(result=None,terminal=False):
            # An active marker alone cannot prove stop. Observe actual tree
            # stop via ProcessTree before allowing any authoritative checks.
            assert observed and observed[-1]=="stopped"
            store.events()
        return output.with_name("unused-input.json"),verify
    from infrastructure.process_tree import ProcessTree
    original_terminate=ProcessTree.terminate
    def terminate(self):
        if mutation=="grandchild":
            # Parent timer, ordinary stop and exception/finally routes all
            # enter here. The separate wrapper timer/poll/EOF/kill-on-close
            # cannot be monkeypatched here and may already have stopped it.
            # Never sample liveness here or wait before the actual stop.
            creation.parent_stopping()
        return original_terminate(self)
    monkeypatch.setattr(ProcessTree,"terminate",terminate)
    original_wait=ProcessTree.wait_stopped
    def stopped(self,*a,**kw):
        value=original_wait(self,*a,**kw)
        if value:
            observed.append("stopped"); timings.setdefault("stopped",time.monotonic())
        return value
    monkeypatch.setattr(ProcessTree,"wait_stopped",stopped)
    if mutation=="stall":
        from application.walking_watch import WalkingWatch
        original_check=WalkingWatch.check
        once=[]
        def stall(self):
            if paths.get("output") and paths["output"].with_name("active").exists() and not once:
                once.append(True); time.sleep(.3)
            return original_check(self)
        monkeypatch.setattr(WalkingWatch,"check",stall)
    def external():
        while not cancel.wait(.005):
            if mutation=="grandchild" and paths.get("output") and not owned:
                receipt=paths["output"].with_name("grandchild.ready")
                if receipt.exists():
                    try:
                        assert time.monotonic()<paths["deadline"], "late creation receipt"
                        with receipt.open('rb') as stream: raw=stream.read(513)
                        row=creation_receipt(raw,nonce)
                        identity=ExactWindowsProcess(row["pid"],row["birth"])
                        owned.append(identity)
                        # Freeze verified live READY/handle/birth/nonce/ACK
                        # before the original deadline or any parent stop.
                        creation.confirm(identity,row,nonce,paths["deadline"],
                            lambda: paths["output"].with_name("grandchild.ack").write_text(nonce))
                        timings["creation_ready"]=time.monotonic()
                    except BaseException as exc:
                        identity_errors.append(str(exc)); return
            if paths.get("output") and paths["output"].with_name("active").exists(): break
            if paths.get("deadline") and time.monotonic()>=paths["deadline"]:
                if mutation=="grandchild": identity_errors.append("missing bounded creation-ready receipt")
                return
        if cancel.is_set(): return
        timings["external_observed_ready"]=time.monotonic()
        if mutation=="hold": (store.path/"recovery-hold.json").write_bytes(b'{}')
        elif mutation=="append": store.append("ARTIFICIAL_EXTERNAL_CHANGE",{})
        elif mutation in {"closed","fatal-closed"}: store.append("ARM_CLOSED",{"arm_id":arm,"reason":"artificial-external-close"})
        elif mutation=="writer":
            with file_lock(store.path/".writer.lock",timeout=2):
                changed.set(); cancel.wait(.4)
        elif mutation=="heartbeat-loss":
            heartbeat=paths["output"].with_name("heartbeat.json")
            cancel.wait(.15)
            until=time.monotonic()+.5
            while not cancel.is_set() and time.monotonic()<until:
                heartbeat.unlink(missing_ok=True); changed.set(); cancel.wait(.005)
        changed.set()
    thread=threading.Thread(target=external); thread.start()
    try:
        result=ResearchSupervisor(store).run(attempt,lambda output:[sys.executable,"-c","pass"],
            budget(1.5 if mutation in {"deadline","grandchild","fatal-timeout"} else 10,"predict"),
            resource_plan={"maximum_result_bytes":4*1024**2,"tree_memory_bytes":2*1024**3},walking_prepare=prepare)
    finally:
        cancel.set(); thread.join(timeout=2)
        if mutation=="grandchild":
            try:
                # The kernel handle still identifies the originally observed
                # descendant even if a numeric PID has since been reused.
                assert not thread.is_alive(), "creation observer still active"
                assert not identity_errors, identity_errors
                assert owned, "creation not proven before stop"
                # Both parent-first and wrapper-first stops consume the same
                # earlier live creation proof and mandatory pre-cleanup exit.
                creation.assert_exited(owned[0],bool(observed))
            finally:
                for identity in owned: identity.cleanup()
    assert not thread.is_alive() and observed
    assert paths["output"].with_name("active").exists()
    if mutation in {"hold","append","writer"}: assert changed.is_set()
    (tmp_path/"native-observation.json").write_text(json.dumps({"mutation":mutation,"state":result["state"],
        "error_code":result["error_code"],"charged_ms":result["elapsed_ms"],
        "observed_ready_to_stop_ms":1000*(timings["stopped"]-timings["external_observed_ready"]) if "external_observed_ready" in timings else None}))
    if mutation in {"hold","append","writer","stall"}:
        assert result["error_code"] in {"RECOVERY_REQUIRED","WRITER_BUSY"},result
    if mutation.endswith("-result"):
        assert result["artifact_id"] is None
        assert result["error_code"] in {"CONTRACT_MISMATCH","RESOURCE_PLAN_REJECTED","UNAUTHORIZED_DATA"},result
    if mutation=="heartbeat-loss": assert result["error_code"]=="HEARTBEAT_LOST",result
    assert result["state"] in {"INTERRUPTED","TIMEOUT","BUDGET_EXHAUSTED"},result
    assert result["elapsed_ms"]>=75
    assert not usage(BudgetLedger(store),"synthetic-pilot")["pending"]
    assert not any(e["event_kind"]=="HEARTBEAT" for e in store.events())
    if mutation in {"deadline","grandchild","fatal-timeout"}: assert result["state"]=="TIMEOUT"
    if mutation=="fatal-timeout": assert result["error_code"]=="CORRUPT_ARTIFACT"
    if mutation in {"closed","fatal-closed"}: assert result["state"]=="BUDGET_EXHAUSTED",result
    if mutation=="fatal-closed": assert result["error_code"]=="CORRUPT_ARTIFACT",result
