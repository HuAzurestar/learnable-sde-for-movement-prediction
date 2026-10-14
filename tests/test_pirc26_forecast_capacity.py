"""Full-plan unpublished capacity probes, not fitted paths or owner ACKs."""

from copy import deepcopy
from dataclasses import asdict
import json
import sys
import zlib

import pytest
import torch

from application.pirc26_forecast_control import pack_fit,preflight_job_capacity,validate_saved_job,SCHEMA
from application.research_registry import _bounded_json
from estimation.phase_space_checkpoint import encode_state,rng_state,history_metadata
from infrastructure.pirc26_forecast_codec import pack_json,unpack_json,chunks,LIMIT,TEXT_LIMIT
from infrastructure.research_control import canonical,SCHEMA as CONTROL_SCHEMA
from infrastructure.research_store import digest,ResearchError
from inference.phase_space import ForecastRequest,forecast
from inference.phase_space_resume import scope,snapshot,array,moments
from models.phase_space import ModelContractError
from tests.test_pirc26_components_data import declarations
from tests.test_pirc26_dynamics import model


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_complete_two_origin_samples_full_ten_thousand_row_fit_history_and_owner_frames_fit(dtype):
    m = model("M2").to(dtype=dtype)
    config,inputs,*_ = declarations(m)
    config["plan"].update(max_steps=10000,patience=10000,tolerance=0.)
    inputs.update(dtype=str(dtype).split(".")[-1],steps=64,paths=128,origins=2)
    requests = [ForecastRequest((0.,0.,.2,.1),tuple(i/10+j for i in range(64)),float(j),128,root*64,chunk_size=1)
                for j,root in enumerate(("a","b"))]
    config["forecast_request_hashes"] = [digest(asdict(r)) for r in requests]
    job = {"schema_version":"pirc26-worker-job-v1","operation":"fit-and-forecast","initial_checkpoint":m.checkpoint(),
           "o1_result":None,"batch_size":4,"origins":[{"segment_id":"fixture","origin_index":j,
            "time_grid":list(r.time_grid),"sample_count":r.sample_count,"brownian_root_id":r.brownian_root_id,"chunk_size":1}
            for j,r in enumerate(requests)]}
    preflight_job_capacity(job,config,inputs)
    # Probe full lossless publication only: no trained-model/path provenance is
    # asserted for these private-generator arrays and history-column values.
    history_scope = {"objective":"O1","plan":config["plan"],"data_identity":[None]}
    history = [{"step":i+1,"objective":-sys.float_info.max,"gradient_norm":sys.float_info.max,
                **history_metadata(history_scope,i)} for i in range(10000)]
    fit = {"status":"MAX_STEPS","objective":"O1","steps":10000,"history":history,"checkpoint":m.checkpoint()}
    fit_frame = pack_fit(fit,job,config,1)
    generator = torch.Generator().manual_seed(71)
    populations = [torch.randn((128,64,4),dtype=dtype,generator=generator) for _ in requests]
    means,m2s = zip(*(moments(s) for s in populations))
    finished = [{"schema_version":"pirc26-forecast-result-v1","samples":array(populations[0]),
        "sample_ids":list(range(128)),"failed_sample_ids":[],"requested_paths":128,"valid_paths":128,
        "time_grid":list(requests[0].time_grid),"brownian_root_id":requests[0].brownian_root_id,
        "moments":{"estimator_id":"ordered-sample-welford-full-state-v1","sample_count":128,
                   "mean":means[0].tolist(),"covariance":(m2s[0]/127).tolist()}}]
    active = snapshot(scope(m,requests[1]),128,0,populations[1],list(range(128)),[],None,None,means[1],m2s[1])
    method = {"schema_version":SCHEMA,"job_hash":digest(job),"origin_index":1,"fit":fit_frame,"finished":pack_json(finished),"active":active}
    work = 2*128*63
    state = {"step":work,"data_position":{"origin_index":1,"first":128,"step":0},
             "method_state":{**method,"sha256":digest(method)},"rng_state":encode_state(rng_state())}
    receipt = {"cell":{"execution":{"config":config,"inputs":inputs}}}
    assert validate_saved_job(state,job,receipt)[2] == 1
    _bounded_json(state,nodes=65536,depth=32)
    progress = {"completed_steps":work,"total_steps":work,"throughput_per_second":sys.float_info.max,"eta_seconds":0.}
    response = {"schema_version":CONTROL_SCHEMA,"attempt_id":"a"*128,"token":"f"*64,"request_id":"f"*32,"state":state,"progress":progress}
    publication = {"schema_version":"pirc25-checkpoint-v1","parent_attempt_id":"a"*128,"run_id":"a"*128,"plugin_id":"a"*128,
        "plugin_version":"a"*128,"recovery_command_hash":"f"*64,"resume_level":"exact","bindings":{str(i):"f"*64 for i in range(10)},
        "payload_hash":digest(state),"state":state,"admission_hash":"f"*64,"progress":progress}
    _bounded_json(publication,nodes=65536,depth=32)
    assert len(canonical(response,LIMIT)) < LIMIT and len(canonical(publication,LIMIT)) < LIMIT
    public = {"fit":fit,"forecast":{"origins":[{"samples":s.tolist(),"moments":{"mean":mean.tolist(),"covariance":(m2/127).tolist()}}
                  for s,mean,m2 in zip(populations,means,m2s)]}}
    assert len(json.dumps(public,allow_nan=False).encode()) < LIMIT
    assert len(fit["history"]) == 10000 and all(len(s) == 128 for s in populations)


def test_forecast_and_o2_complete_plan_capacity_refuse_before_numerical_work_without_shortening(monkeypatch):
    m = model("M0")
    req = ForecastRequest((0.,0.,.2,.1),tuple(range(10000)),0.,50,"a"*64)
    monkeypatch.setattr(torch,"empty",lambda *_a,**_k:pytest.fail("oversized forecast allocated paths"))
    monkeypatch.setattr(m,"drift",lambda *_a:pytest.fail("oversized forecast reached drift"))
    with torch.no_grad(),pytest.raises(ModelContractError,match="RESOURCE_PLAN_REJECTED"):
        forecast(m,req,checkpoint_requested=lambda:False,checkpoint_handler=lambda *_:None)
    assert req.sample_count == 50 and len(req.time_grid) == 10000
    cfg,inputs,*_ = declarations(m)
    cfg.update(objective="O2",plan={"max_steps":10000,"horizon_indices":list(range(1,33))})
    job = {"initial_checkpoint":m.checkpoint(),"operation":"fit-and-forecast",
           "origins":[{"sample_count":256,"time_grid":list(range(64))}]}
    original = deepcopy((job,cfg))
    with pytest.raises(ResearchError,match="RESOURCE_PLAN_REJECTED"):
        preflight_job_capacity(job,cfg,inputs)
    assert (job,cfg) == original


@pytest.mark.parametrize("fault",["bomb","trailing","length","noncanonical","nonfinite","extra"])
def test_lossless_json_codec_refuses_bounded_decompression_and_noncanonical_state(fault):
    frame = pack_json({"x":1})
    if fault == "bomb":
        frame["size_bytes"] = 16
        frame["data"] = chunks(zlib.compress(b"a"*(TEXT_LIMIT+1)))
    elif fault == "trailing":
        frame["data"] = chunks(zlib.compress(b'{"x":1}')+b"trailing")
    elif fault == "length":
        frame["size_bytes"] += 1
    elif fault in ("noncanonical","nonfinite"):
        raw = b'{ "x":1}' if fault == "noncanonical" else b'{"x":NaN}'
        frame.update(size_bytes=len(raw),data=chunks(zlib.compress(raw)))
    else:
        frame["grant"] = "not-authority"
    with pytest.raises(ResearchError):
        unpack_json(frame)
