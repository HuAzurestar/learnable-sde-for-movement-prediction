"""Actual Adam continuation, including a fresh interpreter, on synthetic data."""

from dataclasses import replace
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from estimation.phase_space import O1Plan, fit_o1
from estimation.phase_space_checkpoint import decode_state, encode_state
from estimation.phase_space_o2 import HorizonTrainingExample, O2Plan, fit_o2
from infrastructure.research_store import digest
from models.phase_space import ModelContractError, PhaseSpaceSDE
from tests.test_pirc26_dynamics import model
from tests.test_pirc26_training_forecast import batch, request


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def examples(m):
    req = request(8, 8, (0., .2, .4, .6))
    target = torch.tensor([req.initial_state, (.08, -.03, .5, -.1),
                           (.21, -.04, .7, .0), (.39, -.03, .9, .1)], dtype=torch.float64)
    return [HorizonTrainingExample(req, target, m.spec.train_binding_hash)]


def run_fixture(payload, stop=12):
    """Same bounded component invocation on either side of process restart."""
    torch.set_num_threads(1)
    m = PhaseSpaceSDE.from_checkpoint(payload["initial"])
    rows, saved = [], []
    kwargs = {"progress": rows.append, "checkpoint_requested": lambda: rows[-1]["step"] >= stop,
              "checkpoint_handler": lambda state, progress: saved.append(state),
              "resume_state": payload.get("resume")}
    if payload["kind"] == "O1":
        b = batch(m, 64)
        return fit_o1(m, [replace(b, time=b.time[:32], state=b.state[:32], next_state=b.next_state[:32], dt=b.dt[:32]),
                          replace(b, time=b.time[32:], state=b.state[32:], next_state=b.next_state[32:], dt=b.dt[32:])],
                      O1Plan(max_steps=12, patience=12, tolerance=0., fit_diffusion=True), **kwargs)
    return fit_o2(m, examples(m), O2Plan(max_steps=12, patience=12, curriculum_steps=3, tolerance=0.),
                  payload["o1"], **kwargs)


def initial_payload(kind):
    m = model("M2")
    result = {"kind": kind}
    if kind == "O2":
        result["o1"] = fit_o1(m, [batch(m, 64)], O1Plan(max_steps=5, patience=5, fit_diffusion=False))
    return {**result, "initial": m.checkpoint()}


@pytest.mark.parametrize("kind", ["O1", "O2"])
def test_adam_resume_matches_uninterrupted_training_in_fresh_process(kind):
    payload = initial_payload(kind)
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    uninterrupted = run_fixture(payload)
    # The interruption changes ambient RNG; restored state must supersede it.
    partial = run_fixture(payload, stop=4)
    assert partial["status"] == "CHECKPOINTED" and partial["steps"] == 4
    resumed_payload = {**payload, "resume": json.loads(json.dumps(partial["training_state"]))}
    program = "import json,sys; from tests.test_pirc26_training_resume import run_fixture; print(json.dumps(run_fixture(json.load(sys.stdin))))"
    completed = subprocess.run([sys.executable, "-c", program], input=json.dumps(resumed_payload),
                               text=True, capture_output=True, timeout=60,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert completed.returncode == 0, completed.stderr
    resumed = json.loads(completed.stdout)
    assert resumed["checkpoint"] == uninterrupted["checkpoint"]
    assert resumed["history"] == uninterrupted["history"]
    assert resumed["best_train_objective"] == uninterrupted["best_train_objective"]
    first = decode_state(uninterrupted["training_state"]["state"])
    second = decode_state(resumed["training_state"]["state"])
    assert encode_state(first["optimizer"]) == encode_state(second["optimizer"])
    assert encode_state(first["auxiliary"]) == encode_state(second["auxiliary"])
    assert first["step"] == second["step"] == 12


def test_o2_curriculum_lineage_and_training_score_improve_without_mutating_o1():
    payload = initial_payload("O2")
    before = json.dumps(payload["o1"], sort_keys=True)
    result = run_fixture(payload)
    assert result["gradient_route"] == "G1" and result["diffusion_policy"] == "frozen-o1"
    assert result["o1_lineage_hash"] == digest(payload["o1"])
    assert result["history"][0]["horizon_indices"] == [1]
    assert result["history"][3]["horizon_indices"] == [1, 3]
    assert result["best_train_objective"] < result["history"][0]["objective"]
    assert json.dumps(payload["o1"], sort_keys=True) == before
    assert result["checkpoint"]["state"]["velocity_factor"] == payload["initial"]["state"]["velocity_factor"]


@pytest.mark.parametrize("change", ["target", "plan", "lineage", "model", "environment", "state_hash"])
def test_resume_rejects_changed_immutable_scope(change):
    payload = initial_payload("O2")
    partial = run_fixture(payload, 4)
    m = PhaseSpaceSDE.from_checkpoint(payload["initial"])
    train = examples(m)
    plan = O2Plan(max_steps=12, patience=12, curriculum_steps=3, tolerance=0.)
    state = json.loads(json.dumps(partial["training_state"]))
    o1 = payload["o1"]
    if change == "target":
        train[0].target[1, 0] += .01
    elif change == "plan":
        plan = replace(plan, learning_rate=.003)
    elif change == "lineage":
        o1 = {**o1, "wall_seconds": o1["wall_seconds"] + 1}
    elif change == "model":
        with torch.no_grad():
            m.acceleration_model.layers[-1].bias.add_(.01)
    elif change == "environment":
        torch.set_num_threads(2)
    else:
        state["sha256"] = "f" * 64
    with pytest.raises(ModelContractError, match="CHECKPOINT_INCOMPATIBLE|OBJECTIVE_INCOMPATIBLE"):
        fit_o2(m, train, plan, o1, resume_state=state)


def test_o2_refuses_nontrain_unfinished_lineage_and_path_quotas():
    payload = initial_payload("O2")
    m = PhaseSpaceSDE.from_checkpoint(payload["initial"])
    plan = O2Plan(max_steps=12, patience=12, curriculum_steps=3)
    train = examples(m)
    with pytest.raises(ModelContractError, match="UNAUTHORIZED_DATA"):
        fit_o2(m, [replace(train[0], split_role="selection")], plan, payload["o1"])
    with pytest.raises(ModelContractError, match="OBJECTIVE_INCOMPATIBLE"):
        fit_o2(m, train, plan, {**payload["o1"], "status": "CHECKPOINTED"})
    with pytest.raises(ModelContractError, match="RESOURCE_PLAN_REJECTED"):
        fit_o2(m, train, replace(plan, max_gradient_state_elements=8), payload["o1"])
    with pytest.raises(ModelContractError, match="NONFINITE"):
        fit_o2(m, [replace(train[0], request=replace(train[0].request, maximum_state_norm=.01))], plan, payload["o1"])


def test_rng_codec_restores_all_declared_cpu_generators_and_rejects_malformed():
    from estimation.phase_space_checkpoint import rng_state, restore_rng
    state = encode_state(rng_state())
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(decode_state(json.loads(json.dumps(state))))
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    for malformed in ({"map": [["x", 1], ["x", 2]]}, {"tuple": 1},
                      {"tensor": {"dtype": "float64", "shape": [100001], "data": []}}):
        with pytest.raises(ModelContractError):
            decode_state(malformed)


def test_managed_adapter_requires_owner_and_binds_saved_position_rng(monkeypatch):
    from application.pirc26_training_control import managed_fit_o1, restore_training_state
    from infrastructure.research_control import WorkerControl
    from infrastructure.research_store import ResearchError
    payload = initial_payload("O1")
    m = PhaseSpaceSDE.from_checkpoint(payload["initial"])
    monkeypatch.setattr(WorkerControl, "from_environment", lambda: None)
    with pytest.raises(ResearchError, match="owner checkpoint channel"):
        managed_fit_o1(m, [batch(m, 64)], O1Plan(max_steps=12, patience=12))
    partial = run_fixture(payload, stop=4)
    snapshot = partial["training_state"]
    decoded = decode_state(snapshot["state"])
    envelope = {"step": 4, "data_position": 4, "method_state": snapshot, "rng_state": encode_state(decoded["rng"])}
    assert restore_training_state(envelope) == snapshot
    with pytest.raises(ResearchError, match="position/RNG"):
        restore_training_state({**envelope, "rng_state": {}})
    with pytest.raises(ResearchError, match="position"):
        restore_training_state({**envelope, "data_position": 5})
