"""Protected-PID fallback safety, using artificial observations only."""
from datetime import datetime,timedelta,timezone
import io
import json
import os
import subprocess
from types import SimpleNamespace

import pytest
import application.research_continuation as module

pytestmark=pytest.mark.skipif(os.name!="nt",reason="Windows-specific read-only identity fallback")


def row(pid=9344,age=1,name="SyntheticProtected.exe"):
    return {"pid":pid,"name":name,"created_at_utc":(datetime.now(timezone.utc)-timedelta(days=age)).isoformat()}


def test_two_fresh_equal_births_prove_reuse_without_stopping_target(monkeypatch):
    value=row(); calls=[]
    def query(pid):
        calls.append(pid); return value.copy()
    monkeypatch.setattr(module,"_cim_process_birth",query)
    old=(datetime.now(timezone.utc)-timedelta(days=18)).isoformat()
    proof=module.cim_pid_reuse_observation(9344,old)
    assert proof["pid"]==9344 and proof["cim_observations"]==[value,value] and calls==[9344,9344]


@pytest.mark.parametrize("fault",["wrong-pid","old","different","denied","naive","future","bad-pid"])
def test_uncertainty_is_not_a_reuse_proof(monkeypatch,fault):
    first=row(); second=first.copy(); old=(datetime.now(timezone.utc)-timedelta(days=18)).isoformat(); pid=9344
    if fault=="wrong-pid": first["pid"]=second["pid"]=9345
    if fault=="old": first["created_at_utc"]=second["created_at_utc"]=old
    if fault=="different": second["name"]="AnotherInstance.exe"
    if fault=="denied": first=None
    if fault=="naive": first["created_at_utc"]=second["created_at_utc"]="2026-10-20T00:00:00"
    if fault=="future": first["created_at_utc"]=second["created_at_utc"]=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
    if fault=="bad-pid": pid=True
    values=iter([first,second]); monkeypatch.setattr(module,"_cim_process_birth",lambda pid:next(values))
    assert module.cim_pid_reuse_observation(pid,old) is None


@pytest.mark.parametrize("fault",[None,"wrong-pid","multiple","oversized","denied","timeout"])
def test_query_command_is_fixed_and_bounded_and_only_own_helper_can_be_killed(monkeypatch,fault):
    value={"ProcessId":9344,"Name":"SyntheticProtected.exe","CreationDateUtc":"2026-10-20T00:00:00.1234567Z"}
    if fault=="wrong-pid": value["ProcessId"]=9345
    if fault=="multiple": value=[value,value]
    content=json.dumps(value).encode() if fault!="oversized" else b"x"*4096
    child=SimpleNamespace(stdout=io.BytesIO(content),returncode=4 if fault=="denied" else 0)
    killed=[]; waits=[]
    def wait(timeout):
        waits.append(timeout)
        if fault=="timeout" and len(waits)==1:
            child.returncode=None; raise subprocess.TimeoutExpired("own-query",timeout)
        child.returncode=0 if fault!="denied" else 4
    child.wait=wait; child.poll=lambda:child.returncode
    def kill():
        killed.append("owned-query-helper"); child.returncode=1
    child.kill=kill
    commands=[]
    def popen(command,**kwargs):
        commands.append((command,kwargs)); return child
    monkeypatch.setattr(subprocess,"Popen",popen)
    result=module._cim_process_birth(9344)
    assert (result is not None)==(fault is None)
    assert commands[0][0][-1]==module._CIM_BIRTH_QUERY and "9344" not in commands[0][0][-1]
    assert commands[0][1]["env"]["PSDE_QUERY_PROCESS_ID"]=="9344"
    assert waits[0]==5 and killed==(["owned-query-helper"] if fault=="timeout" else [])


def test_native_access_denial_uses_fallback_but_keeps_prior_absence_semantics(monkeypatch):
    import ctypes
    def zero(*args):return 0
    kernel=SimpleNamespace(OpenProcess=zero,GetProcessTimes=lambda *a:0,CloseHandle=lambda *a:1)
    monkeypatch.setattr(ctypes,"WinDLL",lambda *a,**k:kernel)
    monkeypatch.setattr(ctypes,"get_last_error",lambda:5)
    monkeypatch.setattr(module,"pid_absence_observation",lambda *a,**k:None)
    expected={"pid":9344,"source":"synthetic-two-birth-proof"}
    monkeypatch.setattr(module,"cim_pid_reuse_observation",lambda *a:expected)
    assert module.pid_reuse_observation(9344,"2026-10-08T00:00:00+00:00")==expected
