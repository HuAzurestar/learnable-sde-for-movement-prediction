"""Causal train-only preparation, physical noise calibration and UO2 training."""
from copy import deepcopy
import math
import time

import torch

from models.walking_pilot import PilotConfig, PilotError, PilotModel


def energy_score(samples, target):
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 2 or not torch.isfinite(samples).all():
        raise PilotError("INVALID_ENERGY_POPULATION")
    n = len(samples)
    return torch.linalg.vector_norm(samples-target, dim=-1).mean() - torch.cdist(samples,samples).sum()/(2*n*(n-1))


def prefix_state(times, positions, origin):
    """Centered regression over [origin-5,origin], never future observations.

    Eight symmetric particles exactly represent the estimated joint [p,v]
    regression covariance (denominator eight). Rp is observation residual
    covariance, distinct from uncertainty in the prefix regression.
    """
    times, positions = torch.as_tensor(times,dtype=torch.float64), torch.as_tensor(positions,dtype=torch.float64)
    if (times.ndim != 1 or positions.shape != (len(times),2) or
            not torch.isfinite(times).all() or not torch.isfinite(positions).all() or
            bool((times[1:] <= times[:-1]).any())):
        raise PilotError("INVALID_NATIVE_TRAJECTORY")
    keep = (times >= origin-5) & (times <= origin)
    t, p = times[keep], positions[keep]
    if len(t)<3 or origin-times[0]<4.999999 or abs(float(t[-1])-origin)>1e-8:
        raise PilotError("MISSING_CAUSAL_FIVE_SECOND_PREFIX")
    center=t.mean(); dt=t-center; ss=dt.square().sum()
    velocity=(dt[:,None]*(p-p.mean(0))).sum(0)/ss
    predicted=p.mean(0)+(origin-center)*velocity
    residual=p-p.mean(0)-dt[:,None]*velocity
    Rp=residual.T@residual/(len(t)-2)
    # kron in [p_x,p_y,v_x,v_y] order.
    joint=torch.tensor([[1/len(t)+(origin-center)**2/ss,(origin-center)/ss],
                        [(origin-center)/ss,1/ss]],dtype=torch.float64)
    covariance=torch.kron(joint,Rp)
    values,vectors=torch.linalg.eigh(covariance)
    if float(values.min()) < -1e-9:
        raise PilotError("INVALID_PREFIX_COVARIANCE")
    factor=vectors*values.clamp(min=0).sqrt()[None,:]
    mean=torch.cat((predicted-p[-1],velocity))
    particles=mean+torch.cat((2*factor.T,-2*factor.T))
    return {"particles":particles,"Rp":Rp,"velocity":velocity,
            "velocity_covariance":Rp/ss,"origin_position":p[-1],"origin":origin}


def example_from_native(times, positions, horizons, origin=None):
    times,positions=torch.as_tensor(times,dtype=torch.float64),torch.as_tensor(positions,dtype=torch.float64)
    origins=[origin] if origin is not None else [float(t) for t in times if t>=float(times[0])+5-1e-8]
    for start in origins:
        try:
            prefix=prefix_state(times,positions,start)
            targets=[]
            for horizon in horizons:
                found=torch.where(torch.abs(times-(start+horizon))<1e-8)[0]
                if len(found)!=1:
                    raise PilotError("MISSING_NATIVE_TARGET")
                targets.append(positions[found[0]]-prefix["origin_position"])
            return {**prefix,"times":[0.,*horizons],"targets":torch.stack(targets)}
        except PilotError:
            if origin is not None:
                raise
    raise PilotError("NO_LEGAL_NATIVE_ORIGIN")


def prepare_training(trajectories):
    """Only train trajectories enter normalization, Rp and QV calibration.

    Non-overlapping five-second regressions provide velocity increments.
    Observation-induced velocity covariance is subtracted before nonnegative
    trace aggregation. This is a frozen approximate calibration, not latent
    instantaneous velocity identification.
    """
    examples=[]; residuals=[]; velocities=[]; increments=[]; weights=[]; source_rows=[]
    for track in trajectories:
        source_velocities=[]; source_increments=[]; example=None
        for segment in track.get("segments",[track]):
            t=torch.tensor(segment["times"],dtype=torch.float64)
            p=torch.tensor(segment["positions"],dtype=torch.float64)
            if example is None:
                try:
                    example=example_from_native(t,p,[1.,3.],segment.get("origin"))
                    example["source_id"]=track["source_id"]
                    examples.append(example); residuals.append(example["Rp"])
                except PilotError:
                    pass
            previous=None
            for origin in t:
                origin=float(origin)
                if origin<float(t[0])+5-1e-8 or (previous is not None and origin-previous["origin"]<=5+1e-8):
                    continue
                try:
                    current=prefix_state(t,p,origin)
                except PilotError:
                    continue
                source_velocities.append(current["velocity"])
                if previous is not None:
                    correction=torch.trace(current["velocity_covariance"]+previous["velocity_covariance"])
                    source_increments.append(((current["velocity"]-previous["velocity"]).square().sum()-correction)/(origin-previous["origin"]))
                previous=current
        source_rows.append({"source_id":track["source_id"],"training_origin":"READY" if example else "NO_LEGAL_NATIVE_ORIGIN",
                            "calibration_increments":len(source_increments)})
        if source_velocities:
            velocities.extend(source_velocities); weights.extend([1/len(source_velocities)]*len(source_velocities))
        if source_increments:
            increments.append(torch.stack(source_increments).mean())
    if not examples or not increments:
        raise PilotError("MISSING_TRAIN_NONOVERLAPPING_NOISE_CALIBRATION")
    states=torch.cat([e["particles"] for e in examples])
    return {"examples":examples,"means":states.mean(0),"scales":states.std(0,unbiased=False).clamp(min=1e-6),
            "Rp":torch.stack(residuals).mean(0),"qv":max(0.,float(torch.stack(increments).mean())),
            "calibration_velocities":torch.stack(velocities),"calibration_weights":torch.tensor(weights,dtype=torch.float64)/sum(weights),
            "calibration_increments":sum(r["calibration_increments"] for r in source_rows),"sources":source_rows,
            "scope":"train-only; approximate prefix regression; observation covariance removed"}


def calibrated_model(arm, prepared, speed_limit):
    velocities=prepared["calibration_velocities"]
    r2=velocities.square().sum(-1)
    if arm.startswith("B"):
        if bool((r2>=speed_limit**2).any()):
            raise PilotError("CALIBRATION_SUPPORT_REJECTED")
        z=velocities/torch.sqrt(speed_limit**2-r2)[:,None]
        s=1+z.square().sum(-1)
        J=speed_limit*(torch.eye(2,dtype=torch.float64)[None]/s.sqrt()[:,None,None]-z[:,:,None]*z[:,None,:]/s.pow(1.5)[:,None,None])
        denominator=float((J.square().sum((1,2))*prepared["calibration_weights"]).sum())
    elif arm.startswith("C-R"):
        if bool((r2>speed_limit**2).any()):
            raise PilotError("CALIBRATION_SUPPORT_REJECTED")
        denominator=float((r2*(1+(1-r2/speed_limit**2).square())*prepared["calibration_weights"]).sum())
    elif arm.startswith("C-T"):
        if bool((r2>speed_limit**2).any()):
            raise PilotError("CALIBRATION_SUPPORT_REJECTED")
        denominator=float((r2*prepared["calibration_weights"]).sum())
    else:
        denominator=2.
    if denominator<=0:
        raise PilotError("NO_PHYSICAL_NOISE_MATCH_AT_REST")
    noise=math.sqrt(prepared["qv"]/denominator)
    model=PilotModel(PilotConfig(arm,speed_limit,noise,rho=noise if arm.startswith("C-R") else 0.),
                     means=prepared["means"].tolist(),scales=prepared["scales"].tolist())
    # Whole population support is checked before fitting. No particle replacement.
    for example in prepared["examples"]:
        model.initialize(example["particles"])
    return model


def observation_samples(samples,Rp,seed):
    eig,vec=torch.linalg.eigh(Rp)
    if float(eig.min()) < -1e-9:
        raise PilotError("INVALID_OBSERVATION_COVARIANCE")
    factor=vec*eig.clamp(min=0).sqrt()[None,:]
    noise=torch.randn(samples.shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
    return samples+noise@factor.T


def training_score(model,examples,Rp,seed,*,cancelled=None):
    values=[]
    for i,example in enumerate(examples):
        states=model.predict(example["particles"],example["times"],paths=16,seed=seed+i,h=.1,cancelled=cancelled)
        predicted=observation_samples(states[:,1:,:2],Rp,seed+100+i)
        values.append(torch.stack([energy_score(predicted[:,j],target) for j,target in enumerate(example["targets"])]).mean())
    return torch.stack(values).mean()


def fit_pilot(model,examples,Rp,*,max_steps=100,cancelled=None,progress=None):
    """Initial best is monitored before update zero; selection uses train only.

    Callback emits the currently selected checkpoint after every completed update
    so a supervisor's soft stop can retain useful progress without dev selection.
    """
    if not 0<=max_steps<=100:
        raise PilotError("INVALID_TRAIN_STEPS")
    start=time.monotonic(); best=deepcopy(model.checkpoint()); best_score=None; steps=0
    result={"checkpoint":best,"updates":0,"best_update":0,"monitor_score":None,
            "termination":"INITIAL_CANDIDATE","elapsed_s":0.,"selection":"train-monitor-only-seed-2602"}
    def emit(status):
        result.update(checkpoint=best,updates=steps,monitor_score=best_score,termination=status,elapsed_s=time.monotonic()-start)
        if progress:
            progress(deepcopy(result))
    emit("INITIAL_CANDIDATE")
    try:
        with torch.no_grad():
            initial=training_score(model,examples,Rp,2602,cancelled=cancelled)
        if not torch.isfinite(initial):
            raise PilotError("NONFINITE_INITIAL_MONITOR")
        best_score=float(initial); emit("TRAINING")
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        for step in range(max_steps):
            if cancelled and cancelled():
                raise PilotError("BUDGET_STOP")
            optimizer.zero_grad()
            value=training_score(model,examples,Rp,2601+step*17,cancelled=cancelled)
            if not torch.isfinite(value):
                raise PilotError("NONFINITE_TRAIN_SCORE")
            value.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise PilotError("NONFINITE_GRADIENT")
            optimizer.step(); steps+=1
            if steps%10==0 or steps==max_steps:
                with torch.no_grad():
                    score=float(training_score(model,examples,Rp,2602,cancelled=cancelled))
                if math.isfinite(score) and score<best_score:
                    best_score=score; best=deepcopy(model.checkpoint()); result["best_update"]=steps
            emit("TRAINING")
        emit("MAX_UPDATES" if max_steps else "INITIAL_ONLY")
    except PilotError as exc:
        emit(str(exc))
    model.load_state_dict(PilotModel.from_checkpoint(best).state_dict())
    return result
