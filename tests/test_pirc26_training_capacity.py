"""Full-plan lossless capacity; synthetic controls, not scientific qualification."""

import json
import random

import numpy as np
import pytest
import torch

from application.research_registry import _bounded_json
from estimation.phase_space import O1Plan, VelocityCholesky, fit_o1
from estimation.phase_space_checkpoint import (HISTORY_SCHEMA, TRAINING_SCHEMA, LIMIT,
    _checkpoint_capacity, decode_state, encode_state, history_metadata, managed_envelope,
    pack_history, preflight_training_checkpoint, rng_state, training_scope, train_loop, unpack_history)
from estimation.phase_space_o2 import O2Plan
from infrastructure.research_control import canonical
from models.phase_space import ModelContractError, PhaseSpaceSDE
from tests.test_pirc26_checkpoint_preflight import maximum_model
from tests.test_pirc26_dynamics import model
from tests.test_pirc26_training_forecast import batch


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def scope_for(m, kind):
    if kind == "O1":
        plan = O1Plan(max_steps=10000, patience=10000, tolerance=0., fit_diffusion=True)
        identities = [{} for _ in range(256)]
    else:
        plan = O2Plan(max_steps=10000, patience=10000, curriculum_steps=3,
                      horizon_indices=tuple(range(1, 33)), tolerance=0.)
        identities = {"o1_lineage_hash": "f"*64,
                      "examples": [{"request": {"brownian_root_id": str(i)*64}} for i in range(32)]}
    return plan, training_scope(m, plan, identities, kind)


@pytest.mark.parametrize("family,kind", [("M0", "O1"), ("M2", "O1"), ("M2", "O2"),
                                         ("M1-R", "O1"), ("M1-S", "O1")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_maximum_capacity_before_update_without_rng_or_parameter_mutation(family, kind, dtype, monkeypatch):
    m = maximum_model(family, dtype)
    plan, scope = scope_for(m, kind)
    auxiliary = VelocityCholesky(m.velocity_factor, .001) if kind == "O1" else None
    parameters = list(m.acceleration_model.parameters()) + ([] if auxiliary is None else list(auxiliary.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=plan.learning_rate)
    initial, rng = m.checkpoint(), encode_state(rng_state())
    monkeypatch.setattr(optimizer, "step", lambda: pytest.fail("capacity probe updated parameters"))
    capacity = preflight_training_checkpoint(m, optimizer, plan, scope, auxiliary)
    assert capacity["max_steps"] == 10000 and capacity["publication_nodes"] < 65536
    assert capacity["response_bytes"] < LIMIT and capacity["publication_bytes"] < LIMIT
    assert not optimizer.state and m.checkpoint() == initial and encode_state(rng_state()) == rng


@pytest.mark.parametrize("kind", ["O1", "O2"])
def test_all_ten_thousand_history_values_and_metadata_round_trip_exactly(kind):
    _, scope = scope_for(model("M2"), kind)
    history = [{"step": i+1, "objective": (-0. if i == 0 else float(i)/13),
                "gradient_norm": float(i)/17, **history_metadata(scope, i)} for i in range(10000)]
    columns = pack_history(history, scope)
    assert columns["schema_version"] == HISTORY_SCHEMA
    assert unpack_history(json.loads(json.dumps(columns)), scope, 10000) == history
    # Preserve signed zero as well as ordinary equality (no quantization).
    assert json.dumps(unpack_history(columns, scope, 10000)) == json.dumps(history)
    assert len(json.dumps(columns)) < len(json.dumps(history))
    with pytest.raises(ModelContractError, match="history"):
        pack_history([{**history[0], "batch_index": 999}], scope)
    with pytest.raises(ModelContractError, match="history"):
        pack_history([{**history[0], "new_diagnostic": 1}], scope)


@pytest.mark.parametrize("fault", ["length", "float", "extra", "schema", "position", "bool"])
def test_history_refuses_resigned_incomplete_or_changed_columns(fault):
    _, scope = scope_for(model("M0"), "O1")
    columns = {"schema_version": HISTORY_SCHEMA, "objective": [0.], "gradient_norm": [1.]}
    step = 1
    if fault == "length":
        columns["objective"] = []
    elif fault == "float":
        columns["objective"] = [float("inf")]
    elif fault == "extra":
        columns["unknown"] = []
    elif fault == "schema":
        columns["schema_version"] = "pirc26-history-columns-v0"
    elif fault == "bool":
        columns["objective"] = [True]
    else:
        step = 10001
    with pytest.raises(ModelContractError, match="CHECKPOINT_INCOMPATIBLE"):
        unpack_history(columns, scope, step)


def test_capacity_rejection_precedes_first_objective_or_update(monkeypatch):
    m = model("M0")
    plan = O1Plan(max_steps=2, patience=2)
    scope = training_scope(m, plan, [{}], "O1")
    original = m.checkpoint()
    # Fixture a real structural overflow; do not lower production quotas.
    monkeypatch.setattr(m, "checkpoint", lambda: {**original, "fixture_capacity": [0]*45000})
    def forbidden(*args):
        pytest.fail("rejected capacity reached objective/update")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    with pytest.raises(ModelContractError, match="RESOURCE_PLAN_REJECTED"):
        train_loop(m, list(m.acceleration_model.parameters()), plan, scope, forbidden, forbidden)
    monkeypatch.undo()
    assert m.checkpoint() == original


def test_actual_smaller_control_quota_refused_before_first_update(monkeypatch):
    m = model("M0")
    initial = m.checkpoint()
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *args: pytest.fail("small control quota updated model"))
    with pytest.raises(ModelContractError, match="RESOURCE_PLAN_REJECTED"):
        fit_o1(m, [batch(m, 4)], O1Plan(max_steps=2, patience=2), checkpoint_byte_limit=1024)
    assert m.checkpoint() == initial


def test_actual_ten_thousand_step_save_and_last_step_resume_preserve_full_history():
    """Actual O1 + Adam, not a fabricated trained result or shortened plan."""
    initial = model("M0").checkpoint()
    plan = O1Plan(max_steps=10000, patience=10000, tolerance=0., fit_diffusion=False)
    def run(stop, resume=None):
        m, rows, saves = PhaseSpaceSDE.from_checkpoint(initial), [], []
        return fit_o1(m, [batch(m, 4)], plan, resume_state=resume, progress=rows.append,
                      checkpoint_requested=lambda: rows[-1]["step"] >= stop,
                      checkpoint_handler=lambda state, row: saves.append(state))
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    full = run(10000)
    partial = run(9999)
    resumed = run(10000, json.loads(json.dumps(partial["training_state"])))
    assert full["steps"] == resumed["steps"] == 10000
    assert full["history"] == resumed["history"]
    assert full["checkpoint"] == resumed["checkpoint"]
    assert full["training_state"]["schema_version"] == TRAINING_SCHEMA
    for key in ("optimizer", "rng", "model", "best_checkpoint", "history"):
        assert encode_state(decode_state(full["training_state"]["state"])[key]) == encode_state(
            decode_state(resumed["training_state"]["state"])[key])
    state = decode_state(full["training_state"]["state"])
    envelope = managed_envelope(full["training_state"], state["rng"], 10000)
    _bounded_json(envelope, nodes=65536, depth=32)
    assert len(canonical(envelope, LIMIT)) < LIMIT
    _checkpoint_capacity(full["training_state"]["state"], envelope["rng_state"], 10000)
