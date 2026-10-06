"""Exact inference cursor/moments and malformed scientific state controls."""

from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from inference.phase_space import forecast, ForecastRequest
from infrastructure.research_store import digest, ResearchError
from models.phase_space import ModelContractError
from tests.test_pirc26_dynamics import model


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def request():
    return ForecastRequest((0.,0.,.2,.1), tuple(i/20 for i in range(21)), 0., 17, "a"*64, chunk_size=5)


def interrupted(m, req, boundary):
    polls, saved = 0, []
    def stop():
        nonlocal polls
        polls += 1
        return polls == boundary
    with torch.no_grad():
        result = forecast(m, req, checkpoint_requested=stop, checkpoint_handler=lambda state,row:saved.append((state,row)))
    assert result["status"] == "CHECKPOINTED" and len(saved) == 1
    return saved[0]


@pytest.mark.parametrize("family", ["M0","M2"])
@pytest.mark.parametrize("boundary", [1,7,24,47,80])
def test_exact_resume_preserves_all_ids_paths_and_full_state_moments(family,boundary):
    m, req = model(family), request()
    state, row = interrupted(m,req,boundary)
    with torch.no_grad():
        resumed = forecast(m,req,resume_state=state)
        original = forecast(m,req)
    assert torch.equal(resumed["samples"],original["samples"])
    assert resumed["sample_ids"] == list(range(17)) == original["sample_ids"]
    assert resumed["failed_sample_ids"] == original["failed_sample_ids"]
    assert resumed["moments"] == original["moments"]
    assert row["completed_steps"] <= row["total_steps"] == 17*20
    samples = original["samples"].double()
    mean = samples.mean(0)
    centered = samples - mean
    covariance = torch.einsum("nti,ntj->tij",centered,centered)/(len(samples)-1)
    assert torch.allclose(torch.tensor(resumed["moments"]["mean"],dtype=torch.float64),mean,atol=1e-14,rtol=1e-14)
    assert torch.allclose(torch.tensor(resumed["moments"]["covariance"],dtype=torch.float64),covariance,atol=1e-14,rtol=1e-14)


def test_repeated_resume_and_all_failed_population_are_not_restarted_or_zero_scored():
    m, req = model("M0"), replace(request(),maximum_state_norm=.001)
    state, _ = interrupted(m,req,24)
    saved = []
    polls = 0
    def stop():
        nonlocal polls
        polls += 1
        return polls == 25
    with torch.no_grad():
        assert forecast(m,req,resume_state=state,checkpoint_requested=stop,
                        checkpoint_handler=lambda s,r:saved.append(s))["status"] == "CHECKPOINTED"
        final = forecast(m,req,resume_state=saved[0])
        original = forecast(m,req)
    assert torch.equal(final["samples"],original["samples"])
    assert final["failed_sample_ids"] == list(range(17))
    assert final["moments"]["mean"] is None and final["moments"]["covariance"] is None


@pytest.mark.parametrize("fault", ["hash","request","model","extra","cursor","id","dtype","shape","binary","nonfinite","mask","moment"])
def test_malformed_or_incompatible_forecast_state_is_refused(fault):
    m,req = model("M0"),request()
    state,_ = interrupted(m,req,24)
    state = deepcopy(state)
    if fault == "hash":
        state["sha256"] = "f"*64
    elif fault in {"request","model"}:
        state["scope"][fault+"_hash"] = "f"*64
    elif fault == "extra":
        state["old_balance"] = 99
    elif fault == "cursor":
        state["first"] = True
    elif fault == "id":
        state["sample_ids"].append(0)
    elif fault == "dtype":
        state["mean"]["dtype"] = "float32"
    elif fault == "shape":
        state["completed"]["shape"][0] = True
    elif fault == "binary":
        state["m2"]["data"] = ["not-base64!"]
    elif fault == "nonfinite":
        import struct
        from infrastructure.pirc26_forecast_codec import chunks
        raw = struct.pack("<d",float("nan")) + bytes(21*4*4*8-8)
        state["m2"]["data"] = chunks(raw)
    elif fault == "mask":
        state["alive"][0] = 1
    else:
        from inference.phase_space_resume import array
        state["mean"] = array(torch.ones((21,4),dtype=torch.float64))
    if fault != "hash":
        state["sha256"] = digest({k:v for k,v in state.items() if k != "sha256"})
    with torch.no_grad(),pytest.raises((ResearchError,ModelContractError,ValueError)):
        forecast(m,req,resume_state=state)


def test_f32_collapsing_time_grid_is_refused_before_drift_and_no_ambient_rng_is_consumed(monkeypatch):
    m = model("M0").float()
    bad = replace(request(),time_grid=(1e8,1e8+1),history_cutoff=1e8)
    monkeypatch.setattr(m,"drift",lambda *a,**kw:pytest.fail("collapsed grid reached drift"))
    with pytest.raises(ModelContractError,match="collapses"):
        forecast(m,bad)
    m,req = model("M0"),request()
    py, npstate, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    state,_ = interrupted(m,req,24)
    with torch.no_grad():
        forecast(m,req,resume_state=state)
    assert random.getstate() == py
    now = np.random.get_state()
    assert np.array_equal(now[1],npstate[1]) and now[2:] == npstate[2:]
    assert torch.equal(torch.get_rng_state(),cpu)


@pytest.mark.parametrize("operation", ["forecast", "fit-and-forecast"])
def test_actual_owner_forecast_ack_reopen_and_fresh_cost_preserve_complete_samples_and_moments(tmp_path, operation):
    import json
    import math
    from application.research_budget import BudgetSpec,BudgetLedger
    from application.research_recovery import SharedRecovery
    from experiments.pirc25.runner import SharedRunner
    from infrastructure.research_store import ResearchStore
    from tests.test_pirc26_runtime import prepare
    role = "validation" if operation == "forecast" else "train"
    baseline,value,_,_,registry,recovery,_ = prepare(tmp_path/"baseline",operation=operation,role=role,long_forecast=True)
    expected = SharedRunner(baseline,registry,recovery_registry=recovery).run_cell(value["study_id"],digest(value["cells"][0]),budget=BudgetSpec(90))
    assert expected["state"] == "SUCCEEDED",expected
    target = json.loads((baseline.path/"artifacts"/expected["artifact_id"]).read_bytes())
    # Calibrate a separate finite engineering reservation from observed native
    # baseline cost. No synthetic time/delay/request/ACK or production budget.
    measured = BudgetLedger(baseline).balance("affine")["committed_ms"]/1000
    stopped_seconds = max(5,math.floor(measured*.75))
    store,value,_,_,registry,recovery,grant = prepare(tmp_path/"resumed",operation=operation,role=role,long_forecast=True)
    stopped = SharedRunner(store,registry,recovery_registry=recovery).run_cell(value["study_id"],digest(value["cells"][0]),budget=BudgetSpec(stopped_seconds))
    assert stopped["state"] == "FAILED", (stopped,measured,stopped_seconds)
    assert store.attempts()[stopped["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED"
    saves = [e["payload"] for e in store.events() if e["event_kind"] == "CHECKPOINT_SAVED"]
    assert len(saves) == 1 and 0 < saves[0]["progress"]["completed_steps"] < 256*63
    before = BudgetLedger(store).balance("affine")["committed_ms"]
    assert before > 0
    reopened = ResearchStore(tmp_path/"resumed",store.store_id)
    resumed = SharedRecovery(reopened,registry,recovery).resume(stopped["attempt_id"],saves[0]["artifact_id"],authorization=grant,budget=BudgetSpec(90))
    assert resumed["state"] == "SUCCEEDED",resumed
    actual = json.loads((reopened.path/"artifacts"/resumed["artifact_id"]).read_bytes())
    assert actual["forecast"] == target["forecast"] and actual["metrics"] == target["metrics"]
    assert actual["fit"]["checkpoint"] == target["fit"]["checkpoint"]
    if operation == "fit-and-forecast":
        assert actual["fit"]["history"] == target["fit"]["history"]
        assert actual["fit"]["producer_attempt_id"] == stopped["attempt_id"] != resumed["attempt_id"]
    assert len(actual["forecast"]["origins"][0]["sample_ids"]) == 256
    assert BudgetLedger(reopened).balance("affine")["committed_ms"] > before
    assert reopened.attempts()[resumed["attempt_id"]]["parent_attempt_id"] == stopped["attempt_id"]
    requests = [e["payload"] for e in reopened.events() if e["event_kind"] == "CHECKPOINT_REQUESTED"]
    assert requests and 0 < requests[0]["remaining_seconds"] <= stopped_seconds*.2
