"""Position observation scores and actually coupled h/h2 diagnostics."""
import math
import time
import torch
from estimation.walking_pilot import energy_score,observation_samples
from models.walking_pilot import PilotError,integration_grid


def summarize(samples,target):
    groups=torch.stack([energy_score(group,target) for group in samples.split(16)])
    return {"energy_score":float(energy_score(samples,target)),
            "mc_se":float(groups.std(unbiased=True)/math.sqrt(len(groups))),
            "mean_error_m":float(torch.linalg.vector_norm(samples.mean(0)-target)),
            "coordinate_90_coverage":float(((target>=torch.quantile(samples,.05,dim=0)) &
                                           (target<=torch.quantile(samples,.95,dim=0))).double().mean()),
            "coordinate_90_width_m":(torch.quantile(samples,.95,dim=0)-torch.quantile(samples,.05,dim=0)).tolist()}


def evaluate(model,example,Rp):
    rows=[]
    available=[h for h in (10.,30.,60.) if str(int(h)) in example["targets"]]
    if not available:
        return [{"horizon_s":h,"status":"MISSING_NATIVE_TARGET"} for h in (10,30,60)]
    native=[float(t) for t in example["native_times"] if 0<float(t)<=max(available)]
    times=[0.,*sorted(set(native+available))]
    fg,cg=integration_grid(times,.05),integration_grid(times,.1)
    dt=torch.tensor([b-a for a,b in zip(fg,fg[1:])],dtype=torch.float64)
    fine_noise=torch.randn(len(dt),64,model.noise_dim,
                           generator=torch.Generator().manual_seed(2603),dtype=torch.float64)*dt.sqrt()[:,None,None]
    coarse_noise=torch.stack([fine_noise[fg.index(a):fg.index(b)].sum(0) for a,b in zip(cg,cg[1:])])
    with torch.no_grad():
        coarse_start=time.monotonic()
        coarse=model.predict(example["particles"],times,paths=64,h=.1,increments=coarse_noise)
        fine_start=time.monotonic()
        fine=model.predict(example["particles"],times,paths=64,h=.05,increments=fine_noise)
        score_start=time.monotonic()
        yc=observation_samples(coarse[:,:,:2],Rp,3603)
        yf=observation_samples(fine[:,:,:2],Rp,3603)
        for horizon in (10,30,60):
            if str(horizon) not in example["targets"]:
                rows.append({"horizon_s":horizon,"status":"MISSING_NATIVE_TARGET"}); continue
            i=times.index(float(horizon)); target=torch.tensor(example["targets"][str(horizon)],dtype=torch.float64)
            a,b=summarize(yc[:,i],target),summarize(yf[:,i],target)
            mean_difference=float(torch.linalg.vector_norm(coarse[:,i,:2].mean(0)-fine[:,i,:2].mean(0)))
            score_difference=abs(a["energy_score"]-b["energy_score"])
            threshold=max(.1,2*math.hypot(a["mc_se"],b["mc_se"]))
            paired=coarse[:,i,:2]-fine[:,i,:2]
            mean_se=float(torch.linalg.vector_norm(paired.std(0,unbiased=True)/8))
            native_indices=[j for j,t in enumerate(times) if 0<t<=horizon and str(t) in example["native_targets"]]
            errors=[]; native_scores=[]
            for j in native_indices:
                y=torch.tensor(example["native_targets"][str(times[j])],dtype=torch.float64)
                errors.append(torch.linalg.vector_norm(coarse[:,j,:2].mean(0)-y))
                native_scores.append(energy_score(yc[:,j],y))
            velocity=model.physical_velocity(coarse[:,:i+1])
            speed=torch.linalg.vector_norm(velocity,dim=-1)
            rows.append({"horizon_s":horizon,"status":"OK" if score_difference<=threshold and mean_difference<=max(.1,2*mean_se) else "NUMERICAL_NOT_READY",
                         **a,"fine":b,"mean_h_h2_difference_m":mean_difference,"score_h_h2_difference_m":score_difference,
                         "paired_mean_mc_se_m":mean_se,"score_threshold_m":threshold,
                         "ADE_native_m":float(torch.stack(errors).mean()) if errors else None,
                         "native_average_energy":float(torch.stack(native_scores).mean()) if native_scores else None,
                         "native_count":len(errors),"FDE_m":float(torch.linalg.vector_norm(coarse[:,i,:2].mean(0)-target)),
                         "max_speed_m_s":float(speed.max()),"speed_exceedance_fraction":float((speed>model.config.speed_limit+1e-9).double().mean()),
                         "B_saturation_fraction":float((speed>.95*model.config.speed_limit).double().mean()) if model.family=="B" else None,
                         "paths":64,"failed_paths":0,"seed":2603,"coupling":"fine Brownian pairs summed; within arm/dimension"})
            rows[-1]["forecast_call_seconds"]={"coarse":fine_start-coarse_start,"h2":score_start-fine_start,
                "score_elapsed_so_far":time.monotonic()-score_start,"scope":"same forecast call for all horizons; do not sum rows"}
    return rows
