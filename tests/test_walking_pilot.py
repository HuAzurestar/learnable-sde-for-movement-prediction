"""DEV self-tests: synthetic data only, no shared research-runtime access."""
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from models.walking_pilot import ARMS,PilotConfig,PilotError,PilotModel,radial_step
from estimation.walking_pilot import energy_score,prefix_state,example_from_native,prepare_training,calibrated_model,fit_pilot,training_score
from evaluation.walking_pilot import evaluate
from infrastructure.research_store import ResearchError,digest

torch.set_num_threads(1)


def particles():
    return torch.tensor([[0.,0.,.6,.2]]*8,dtype=torch.float64)


@pytest.mark.parametrize("arm",ARMS)
def test_ten_real_arms_train_gradients_and_json_roundtrip(arm):
    model=PilotModel(PilotConfig(arm,2.,.25,rho=.2))
    assert sum(p.numel() for p in model.parameters())==(124 if model.neural else 10)
    sample=model.predict(particles(),[0.,.2],paths=16)
    score=energy_score(sample[:,-1,:2],torch.tensor([.15,.025],dtype=torch.float64))
    score.backward()
    gradients=torch.cat([p.grad.flatten() for p in model.parameters()])
    assert torch.isfinite(gradients).all() and gradients.abs().sum()>1e-6
    restored=PilotModel.from_checkpoint(json.loads(json.dumps(model.checkpoint())))
    assert torch.equal(restored.predict(particles(),[0.,.2],paths=16),sample)
    before=model.affine.bias.detach().clone()
    torch.optim.SGD(model.parameters(),lr=.01).step()
    assert not torch.equal(before,model.affine.bias)
    if model.neural:
        simple=PilotModel(PilotConfig("M0-UO2" if arm=="M2-UO2" else arm[:-1]+"S",2.,.25,rho=.2))
        with torch.no_grad():
            simple.affine.load_state_dict(model.affine.state_dict())
        model.residual_enabled=False
        assert torch.equal(model.predict(particles(),[0.,.2]),simple.predict(particles(),[0.,.2]))


@pytest.mark.parametrize("family",["C-T-S","C-R-S"])
def test_complete_ito_generator_and_discrete_generator_convergence(family):
    model=PilotModel(PilotConfig(family,2.,.7,rho=.6))
    state=torch.tensor([[.1,.2,.8,-.3]],dtype=torch.float64)
    with torch.no_grad():
        model.affine.bias.copy_(torch.tensor([.2,-.1]))
    b,S=model.coefficients(state)
    v=state[:,2:]
    actual=(v*b[:,2:]).sum(-1)+.5*S[:,2:].square().sum((-1,-2))
    expected=(v*(model.drive(state)-v)).sum(-1)
    assert torch.allclose(actual,expected,atol=1e-13)
    # Discrete generator on first/second moments using exact symmetric normal
    # first two moments. This catches missing radial/Ito drifts, unlike a bound test.
    errors=[]
    for h in (1e-3,1e-5):
        noises=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=torch.float64)[:,:model.noise_dim]*math.sqrt(h)
        nxt=model.step(state.expand(4,-1),h,noises)
        drift=(nxt.mean(0)-state[0])/h
        covariance=(nxt[:,2:]-v).T@(nxt[:,2:]-v)/(4*h)
        errors.append(float((drift-b[0]).abs().max()+ (covariance-S[0,2:]@S[0,2:].T).abs().max()))
    assert errors[-1]<2e-4 and errors[-1]<errors[0]/20


def test_radial_boundary_absorption_nonzero_noise_and_gradcheck():
    v=torch.tensor([[0.,0.],[2.,0.],[.8,.2]],dtype=torch.float64)
    out=radial_step(v,.1,torch.tensor([1.,-1.,.4],dtype=torch.float64),2.,.7)
    assert torch.equal(out[:2],v[:2]) and not torch.equal(out[2],v[2])
    assert torch.linalg.vector_norm(out,dim=-1).max()<=2.
    inside=torch.tensor([[.7,.3]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:radial_step(x,.1,torch.tensor([.3],dtype=torch.float64),2.,.7),(inside,))
    zero=torch.zeros((1,2),dtype=torch.float64,requires_grad=True)
    radial_step(zero,.1,torch.tensor([.3],dtype=torch.float64),2.,.7).sum().backward()
    assert torch.isfinite(zero.grad).all() and zero.grad.abs().sum()>0
    sphere=torch.tensor([[2.,0.]],dtype=torch.float64,requires_grad=True)
    increment=torch.tensor([.3],dtype=torch.float64)
    radial_step(sphere,.1,increment,2.,.7)[0,0].backward()
    eps=1e-6
    interior=sphere.detach()-torch.tensor([[eps,0.]],dtype=torch.float64)
    numerical=(radial_step(sphere.detach(),.1,increment,2.,.7)[0,0]-radial_step(interior,.1,increment,2.,.7)[0,0])/eps
    assert sphere.grad[0,0]==pytest.approx(float(numerical),rel=1e-5)


@pytest.mark.parametrize("arm",["B-S","C-T-S","C-R-S"])
def test_support_rejects_without_clip_and_actual_coupled_step_difference(arm):
    m=PilotModel(PilotConfig(arm,2.,.6,rho=.6))
    bad=particles(); bad[:,2]=2.1
    with pytest.raises(PilotError,match="SUPPORT_REJECTED"):
        m.initialize(bad)
    boundary=particles(); boundary[:,2:]=torch.tensor([2.,0.])
    if arm=="B-S":
        with pytest.raises(PilotError,match="SUPPORT_REJECTED"):
            m.initialize(boundary)
    else:
        m.initialize(boundary)
    fine=torch.randn((20,64,m.noise_dim),generator=torch.Generator().manual_seed(26),dtype=torch.float64)*math.sqrt(.05)
    a=m.predict(particles(),[0.,1.],h=.1,increments=fine.reshape(10,2,64,m.noise_dim).sum(1))
    b=m.predict(particles(),[0.,1.],h=.05,increments=fine)
    assert torch.linalg.vector_norm(m.physical_velocity(a),dim=-1).max()<=2+1e-12
    assert torch.linalg.vector_norm(m.physical_velocity(b),dim=-1).max()<=2+1e-12
    assert torch.linalg.vector_norm(a[:,-1,:2]-b[:,-1,:2]).mean()>1e-8
    assert torch.linalg.vector_norm(a[:,-1,:2]-a[:,0,:2],dim=-1).max()<=2


def test_prefix_causality_units_covariance_and_missing_native_targets():
    t=torch.arange(0,16,dtype=torch.float64)
    p=torch.stack((1000+.7*t,-300+.2*t),-1)
    first=prefix_state(t,p,5.)
    p[6:]+=1e7
    second=prefix_state(t,p,5.)
    assert torch.equal(first["particles"],second["particles"])
    assert torch.allclose(first["velocity"],torch.tensor([.7,.2],dtype=torch.float64),atol=1e-12)
    assert first["particles"].shape==(8,4)
    with pytest.raises(PilotError,match="MISSING_NATIVE_TARGET"):
        example_from_native(t,p,[1.5],origin=5.)
    with pytest.raises(PilotError):
        prefix_state([0,1,1],[[0,0],[1,0],[2,0]],1)


def train_tracks():
    t=torch.arange(0,16,dtype=torch.float64)
    return [{"source_id":f"fake-{i}","times":t.tolist(),
             "positions":torch.stack((.4*t+.01*t.square(),.1*t+.002*t.square()),-1).tolist()} for i in range(4)]


def test_train_only_calibration_and_initial_best_survives_stop():
    data=prepare_training(train_tracks())
    assert data["calibration_increments"]>=4 and data["qv"]>0
    for arm in ARMS:
        m=calibrated_model(arm,data,2.)
        result=fit_pilot(m,data["examples"][:1],data["Rp"],max_steps=1)
        assert result["updates"]==1 and result["monitor_score"] is not None
        with torch.no_grad():
            selected=float(training_score(m,data["examples"][:1],data["Rp"],2602))
        assert selected==pytest.approx(result["monitor_score"])
    m=calibrated_model("C-R-N",data,2.); initial=m.checkpoint()
    stopped=fit_pilot(m,data["examples"][:1],data["Rp"],cancelled=lambda:True)
    assert stopped["checkpoint"]==initial and stopped["updates"]==0 and stopped["termination"]=="BUDGET_STOP"


def test_energy_score_independent_hand_calculation():
    samples=torch.tensor([[0.,0.],[2.,0.]],dtype=torch.float64)
    # E|X-y|=1, half mean over ordered distinct pairs=1.
    assert energy_score(samples,torch.tensor([1.,0.])).item()==pytest.approx(0.)


@pytest.mark.parametrize("arm",ARMS)
def test_all_arms_compute_actual_coupled_h_h2_difference(arm):
    m=PilotModel(PilotConfig(arm,2.5,.3,rho=.3))
    fine=torch.randn((20,64,m.noise_dim),generator=torch.Generator().manual_seed(2603),dtype=torch.float64)*math.sqrt(.05)
    coarse=fine.reshape(10,2,64,m.noise_dim).sum(1)
    with torch.no_grad():
        a=m.predict(particles(),[0.,1.],h=.1,increments=coarse)
        b=m.predict(particles(),[0.,1.],h=.05,increments=fine)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.linalg.vector_norm(a[:,-1,:2]-b[:,-1,:2],dim=-1).mean()>1e-8


def test_horizon_rows_missing_and_numeric_differences_are_reported():
    m=PilotModel(PilotConfig("C-R-S",2.,.3,rho=.4))
    e={"particles":particles(),"targets":{"10":[6.,2.]},"native_times":[1.,10.],"native_targets":{"1.0":[.6,.2],"10.0":[6.,2.]}}
    rows=evaluate(m,e,torch.eye(2,dtype=torch.float64)*.01)
    assert len(rows)==3 and rows[1]["status"]==rows[2]["status"]=="MISSING_NATIVE_TARGET"
    assert rows[0]["mean_h_h2_difference_m"]>0 and rows[0]["score_h_h2_difference_m"]>0
    e["native_times"].append(.123)
    e["native_targets"]["0.123"]=[.0738,.0246]
    assert evaluate(m,e,torch.eye(2,dtype=torch.float64)*.01)[0]["native_count"]==3


def test_missing_configuration_still_produces_ten_status_rows(tmp_path):
    output=tmp_path/"status.json"
    result=subprocess.run([sys.executable,"-m","application.walking_pilot","run","--config",str(tmp_path/"absent.json"),"--report",str(output)],capture_output=True,timeout=30)
    assert result.returncode==2
    report=json.loads(output.read_bytes())
    assert len(report["rows"])==10 and all(r["status"]=="CONFIG_OR_IO_ERROR" for r in report["rows"])
