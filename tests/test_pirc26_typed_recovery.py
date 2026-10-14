"""Real state codec and pre-read refusals; manual saves are not ACK proof."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from application.pirc26_components import preflight_model
from application.pirc26_runtime import validate_checkpoint_progress, validate_restored_state
from estimation.phase_space import fit_o1,O1Plan
from estimation.phase_space_checkpoint import decode_state,encode_state,managed_envelope,rng_state
from infrastructure.pirc26_checkpoint_contract import CheckpointContractError
from infrastructure.pirc26_training_state_contract import inspect_encoded,inspect_rng,inspect_training_envelope,TypedArray
from infrastructure.research_store import digest,ResearchError
from tests.test_pirc26_dynamics import model
from tests.test_pirc26_training_forecast import batch
from tests.test_pirc26_components_data import declarations


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def saved_training(family="M2", objective="O1",dtype=torch.float64,diffusion=True):
    m = model(family).to(dtype=dtype)
    initial = m.checkpoint()
    cfg,inputs,*_ = declarations(m)
    inputs["dtype"] = str(dtype).split(".")[-1]
    if objective == "O1":
        plan = O1Plan(max_steps=3,patience=3,tolerance=0.,fit_diffusion=diffusion)
        b = batch(m,8)
        b = replace(b,time=b.time.to(dtype),state=b.state.to(dtype),next_state=b.next_state.to(dtype),dt=b.dt.to(dtype))
        r = fit_o1(m,[b],plan,checkpoint_requested=lambda:True,checkpoint_handler=lambda *_:None)
    else:
        from estimation.phase_space_o2 import fit_o2,O2Plan
        from tests.test_pirc26_training_resume import examples
        o1 = fit_o1(m,[batch(m,8)],O1Plan(max_steps=2,patience=2,fit_diffusion=False))
        initial = m.checkpoint()
        cfg,inputs,*_ = declarations(m,"O2",o1)
        plan = O2Plan(max_steps=3,patience=3,curriculum_steps=1,tolerance=0.)
        r = fit_o2(m,examples(m),plan,o1,checkpoint_requested=lambda:True,checkpoint_handler=lambda *_:None)
    assert r["status"] == "CHECKPOINTED"
    cfg["plan"] = json.loads(json.dumps(asdict(plan)))
    job = {"operation":"fit-and-forecast","initial_checkpoint":initial}
    method = r["training_state"]
    decoded = decode_state(method["state"])
    state = managed_envelope(method,decoded["rng"],1)
    receipt = {"cell":{"execution":{"config":cfg,"inputs":inputs}},"documents":{"package":{"payload":{"pirc26_job":job}}}}
    progress = {"completed_steps":1,"total_steps":3,"throughput_per_second":1.,"eta_seconds":1.}
    return state,receipt,progress


@pytest.mark.parametrize("family,dtype,diffusion", [("M0",torch.float32,False),("M0",torch.float64,True),
                                                    ("M2",torch.float32,True),("M2",torch.float64,False)])
def test_real_o1_complete_state_preflight_preserves_values_and_rng_without_allocation(monkeypatch,family,dtype,diffusion):
    state,receipt,progress = saved_training(family,dtype=dtype,diffusion=diffusion)
    before = encode_state(rng_state())
    for engine,name in ((torch,"tensor"),(torch,"empty"),(np,"asarray")):
        monkeypatch.setattr(engine,name,lambda *_a,**_k:pytest.fail("pure recovery allocated an array"))
    validate_checkpoint_progress(receipt,state,progress)
    validate_restored_state(receipt,state)
    assert encode_state(rng_state()) == before


def test_real_o2_state_uses_actual_frozen_diffusion_and_adam_recipe_without_engines(monkeypatch):
    state,receipt,progress = saved_training(objective="O2")
    monkeypatch.setattr(torch,"tensor",lambda *_a,**_k:pytest.fail("O2 preflight allocated tensors"))
    validate_checkpoint_progress(receipt,state,progress)


def change_state(state,fault):
    value = decode_state(state["method_state"]["state"])
    if fault == "extra":
        value["restored_budget"] = 99
    elif fault == "history":
        value["history"]["gradient_norm"].append(1.)
    elif fault == "negative-gradient":
        value["history"]["gradient_norm"][0] = -1.
    elif fault == "stale":
        value["stale"] = True
    elif fault == "model":
        value["model"]["sha256"] = "f"*64
    elif fault == "best-model":
        value["best_checkpoint"]["sha256"] = "f"*64
    elif fault == "group":
        value["optimizer"]["param_groups"][0]["lr"] *= 2
    elif fault == "parameter":
        value["optimizer"]["state"][999] = value["optimizer"]["state"][0]
    elif fault == "shape":
        value["optimizer"]["state"][0]["exp_avg"] = torch.ones(4,dtype=torch.float64)
    elif fault == "second-moment":
        value["optimizer"]["state"][0]["exp_avg_sq"][0,0] = -1.
    elif fault == "update":
        value["optimizer"]["state"][0]["step"] = torch.tensor(2.)
    elif fault == "auxiliary":
        value["auxiliary"]["raw_diagonal"] = torch.ones(3,dtype=torch.float64)
    elif fault == "rng-python":
        value["rng"]["python"] = (3,(1,2),None)
    elif fault == "rng-numpy":
        value["rng"]["numpy"] = ("MT19937",np.array([1],dtype="uint32"),0,0,0.)
    elif fault == "rng-torch":
        value["rng"]["torch_cpu"] = torch.ones(3,dtype=torch.int64)
    elif fault == "rng-duplicate":
        state["rng_state"]["map"][0][1]["tuple"][1]["tuple"][0] ^= 1
    state["method_state"]["state"] = encode_state(value)
    if fault == "late-array":
        # Malformed last field AFTER legitimate model/optimizer arrays.
        state["method_state"]["state"]["map"].append(["last",{"tensor":{"dtype":"float64","shape":[1],"data":[True]}}])
    state["method_state"]["sha256"] = digest({k:v for k,v in state["method_state"].items() if k != "sha256"})


@pytest.mark.parametrize("fault",["extra","history","negative-gradient","stale","model","best-model","group","parameter","shape",
                                  "second-moment","update","auxiliary","rng-python","rng-numpy","rng-torch","rng-duplicate","late-array"])
def test_resigned_training_semantic_faults_refuse_before_any_tensor(fault,monkeypatch):
    state,receipt,progress = saved_training()
    change_state(state,fault)
    for engine,name in ((torch,"tensor"),(torch,"empty"),(np,"asarray")):
        monkeypatch.setattr(engine,name,lambda *_a,**_k:pytest.fail("late state fault allocated an earlier array"))
    with pytest.raises(ResearchError):
        validate_checkpoint_progress(receipt,state,progress)


@pytest.mark.parametrize("fault",[None,"shape","rng-duplicate","late-array"])
def test_actual_owner_prepare_inspects_real_training_payload_before_retry_and_input_reads(tmp_path,monkeypatch,fault):
    from tests.test_pirc26_runtime import prepare,owner_admit
    from application.research_recovery import SharedRecovery
    from application.pirc26_data import decode_block
    from tests.test_pirc26_components_data import admitted,document
    store,spec,plugin,job,registry,recovery,grant = prepare(tmp_path)
    _,receipt = owner_admit(store,spec,plugin)
    cfg = receipt["cell"]["execution"]["config"]
    m = model("M0")
    b = decode_block(admitted(document(m)),m).transitions(batch_size=job["batch_size"])
    fit = fit_o1(m,b,O1Plan(**cfg["plan"]),checkpoint_requested=lambda:True,checkpoint_handler=lambda *_:None)
    method = fit["training_state"]
    state = managed_envelope(method,decode_state(method["state"])["rng"],1)
    progress = {"completed_steps":1,"total_steps":cfg["plan"]["max_steps"],"throughput_per_second":1.,"eta_seconds":1.}
    validate_checkpoint_progress(receipt,state,progress)
    if fault is not None:
        change_state(state,fault)
    owned = SharedRecovery(store,registry,recovery)
    # Explicit manual save is ONLY a negative pre-read fixture, never ACK proof.
    artifact = owned.checkpoint(receipt["attempt_id"],state,admission_hash=receipt["admission_hash"],progress=progress)
    store.transition(receipt["attempt_id"],"FAILED",error_code="TRANSIENT")
    previous = len(store.events())
    for engine,name in ((torch,"tensor"),(torch,"empty"),(np,"asarray")):
        monkeypatch.setattr(engine,name,lambda *_a,**_k:pytest.fail("owner prepare allocated an array"))
    if fault is None:
        assert owned.prepare(receipt["attempt_id"],artifact,authorization=grant)["state"] == state
    else:
        with pytest.raises(ResearchError):
            owned.prepare(receipt["attempt_id"],artifact,authorization=grant)
    assert len(store.attempts()) == 1
    assert not any(e["event_kind"] == "READ_STARTED" and e["payload"].get("purpose") != "resume" for e in store.events()[previous:])


def test_actual_state_validation_runs_in_cold_interpreter_with_all_numerical_imports_blocked(tmp_path):
    state,receipt,progress = saved_training()
    payload = json.dumps([state,receipt,progress])
    script = """
import importlib.abc,json,sys
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy'}:
            raise AssertionError('numerical import: '+fullname)
sys.meta_path.insert(0,NoEngine())
from application.pirc26_runtime import validate_checkpoint_progress
state,receipt,progress=json.load(sys.stdin)
validate_checkpoint_progress(receipt,state,progress)
assert not {'torch','numpy','scipy'}.intersection(sys.modules)
print('cold-real-state-ok')
"""
    result = subprocess.run([sys.executable,"-B","-c",script],input=payload,text=True,capture_output=True,
                            cwd=Path(__file__).resolve().parents[1],timeout=40)
    assert result.returncode == 0,result.stdout+result.stderr
    assert result.stdout.strip() == "cold-real-state-ok"


@pytest.mark.parametrize("fault",["shape","rng-duplicate","late-array"])
def test_actual_recovery_dispatch_refuses_before_provider_transport_or_handoff(tmp_path,monkeypatch,fault):
    from tests.test_pirc26_runtime import prepare
    from application.research_admission import AdmissionGate
    from application.pirc26_runtime import resume_command
    from application.pirc26_data import decode_block
    from tests.test_pirc26_components_data import admitted,document
    store,spec,plugin,job,*_ = prepare(tmp_path)
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(spec["study_id"],cell))
    output = store.path/"artifacts"/(".attempt-"+attempt)/"result.json"
    output.parent.mkdir()
    receipt = AdmissionGate(store).prepare(spec,cell,plugin,attempt,recovery_builder=resume_command)
    m = model("M0")
    b = decode_block(admitted(document(m)),m).transitions(batch_size=job["batch_size"])
    fit = fit_o1(m,b,O1Plan(**receipt["cell"]["execution"]["config"]["plan"]),
                 checkpoint_requested=lambda:True,checkpoint_handler=lambda *_:None)
    method = fit["training_state"]
    state = managed_envelope(method,decode_state(method["state"])["rng"],1)
    change_state(state,fault)
    previous = len(store.events())
    monkeypatch.setattr("application.pirc26_runtime.read_block",lambda *_a,**_k:pytest.fail("invalid state reached physical provider"))
    with pytest.raises(ResearchError):
        resume_command(output,spec,cell,state)
    assert not any(e["event_kind"] == "READ_STARTED" for e in store.events()[previous:])
    assert not (output.parent/"pirc26-handoff.json").exists()
