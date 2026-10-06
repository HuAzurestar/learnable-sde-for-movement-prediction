"""Actual versioned numerical factories and owner-granted synthetic block reads."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math

import pytest
import torch

from application.pirc26_components import (component_bindings, component_registries, composition_contract,
                                         construct_components, entry, STATE, UNITS)
from application.pirc26_data import read_block, decode_block, PURPOSES
from application.registry import ComponentRegistry
from application.research_data import EvaluationExposureLedger
from application.research_execution import execution_binding
from application.research_registry import implementation_hash
from estimation.phase_space import fit_o1, O1Plan
from estimation.phase_space_o2 import O2Plan
from inference.phase_space import forecast
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.research_admission_fixtures import synthetic_plugin
from tests.test_pirc26_dynamics import model
from tests.test_pirc26_training_forecast import batch, request
from tests.test_pirc26_training_resume import examples


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def adapter_command(output, spec, cell):
    # Only a metadata declaration for composition tests, never dispatched.
    raise AssertionError("composition factory tests must not launch a worker")


def declarations(m, objective="O1", o1=None):
    plan = O1Plan(max_steps=3, patience=3, fit_diffusion=False) if objective == "O1" else O2Plan(
        max_steps=3, patience=3, curriculum_steps=1)
    req = request(8, 8) if objective == "O1" else examples(m)[0].request
    config = {"seed": 12, "family": m.model_card()["family"], "objective": objective,
              "plan": json.loads(json.dumps(asdict(plan))), "initial_model_hash": m.checkpoint()["sha256"],
              "dynamics_spec_hash": digest(m.model_card()["spec"]), "configuration_hash": digest(m.acceleration_model.configuration()),
              "forecast_request_hashes": [digest(asdict(req))]}
    if o1 is not None:
        config["o1_lineage_hash"] = digest(o1)
    cfg = m.acceleration_model.configuration()
    capacity = max(4, math.ceil(math.sqrt(sum(p.numel() for p in m.state_dict().values()))),
                   *cfg.get("hidden", []), len(cfg.get("centers", [])),
                   sum(len(k) - cfg.get("degree", 0) - 1 for k in cfg.get("knots", [])))
    profile = {"state_dim": 4, "noise_dim": 2, "diffusion_support": ["vx", "vy"], "dtype": "float64", "device": "cpu",
               "observation_profile": "causal-observed-phase-space-v1", "observations": 64, "batches": 2,
               "origins": 1, "paths": 8, "steps": 4, "components": capacity}
    registries = component_registries(objective, config["family"])
    bindings = component_bindings(config, profile, matrix_cells=1, registries=registries)
    plugin = synthetic_plugin("pirc26-composition-fixture", frozenset({"generic-rollout"}),
                              STATE, UNITS, "exact", adapter_command)
    adapter = replace(plugin.registry_entry, composition=composition_contract(objective))
    return config, profile, registries, bindings, adapter, req


@pytest.mark.parametrize("family", ["M0", "M1-S", "M1-R", "M2"])
def test_registered_factories_fit_actual_dynamics_and_use_common_predictor(family):
    initial = model(family)
    config, profile, registries, bindings, adapter, req = declarations(initial)
    parts, plan = construct_components(adapter, bindings, matrix_cells=1, seed=12, registries=registries,
                                       initial_checkpoint=initial.checkpoint())
    result = parts["trainer"].fit(parts["model"], [batch(initial, 32)])
    direct = fit_o1(initial, [batch(initial, 32)], O1Plan(**config["plan"]))
    assert result["checkpoint"] == direct["checkpoint"] and result["history"] == direct["history"]
    actual = parts["predictor"].predict(parts["model"], req)
    assert torch.equal(actual["samples"], forecast(initial, req)["samples"])
    assert plan["profile"]["noise_dim"] == 2
    assert set(plan["entries"]) == {"model", "trainer", "predictor"}


def test_registered_o2_consumes_exact_o1_lineage_and_curriculum():
    initial = model("M2")
    o1 = fit_o1(initial, [batch(initial, 32)], O1Plan(max_steps=2, patience=2, fit_diffusion=False))
    config, profile, registries, bindings, adapter, req = declarations(initial, "O2", o1)
    parts, _ = construct_components(adapter, bindings, matrix_cells=1, seed=12, registries=registries,
                                    initial_checkpoint=initial.checkpoint())
    result = parts["trainer"].fit(parts["model"], examples(initial), o1_result=o1)
    assert result["o1_lineage_hash"] == digest(o1) and result["gradient_route"] == "G1"
    assert result["history"][0]["horizon_indices"] == [1] and result["history"][1]["horizon_indices"] == [1, 3]
    assert parts["predictor"].predict(parts["model"], req)["valid_paths"] == 8


@pytest.mark.parametrize("fault", ["version", "identity", "seed", "role", "noise", "shared-config"])
def test_whole_composition_refuses_invalid_binding_before_any_constructor(monkeypatch, fault):
    initial = model("M2")
    config, profile, registries, bindings, adapter, req = declarations(initial)
    bindings = deepcopy(bindings)
    seed = 12
    if fault == "version":
        bindings["model"]["component_version"] = "missing"
    elif fault == "identity":
        bindings["trainer"]["registry_entry_hash"] = "f" * 64
    elif fault == "seed":
        seed = 13
    elif fault == "role":
        bindings.pop("predictor")
    elif fault == "noise":
        bindings["predictor"]["inputs"]["noise_dim"] = 1
    else:
        bindings["trainer"]["config"]["plan"]["learning_rate"] *= .5
        # Reseal this individually valid binding: only the complete shared
        # configuration check, not a stale per-role hash, can reject it.
        bindings["trainer"] = execution_binding(entry("trainer", "O1", "M2"),
            bindings["trainer"]["config"], profile, matrix_cells=1)
    monkeypatch.setattr(ComponentRegistry, "create_bound", lambda *a, **k: pytest.fail("invalid combination constructed"))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        construct_components(adapter, bindings, matrix_cells=1, seed=seed, registries=registries,
                             initial_checkpoint=initial.checkpoint())


def test_actual_factory_profiles_refuse_data_capacity_and_changed_forecast_recipe():
    initial = model("M2")
    config, profile, registries, bindings, adapter, req = declarations(initial)
    parts, _ = construct_components(adapter, bindings, matrix_cells=1, seed=12, registries=registries,
                                    initial_checkpoint=initial.checkpoint())
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        parts["trainer"].fit(parts["model"], [batch(initial, 65)])
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        parts["predictor"].predict(parts["model"], replace(req, brownian_root_id="b" * 64))
    parts["model"].float()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        parts["predictor"].predict(parts["model"], req)
    with pytest.raises(ResearchError, match="OBJECTIVE_INCOMPATIBLE"):
        component_registries("O2", "M0")


def document(m, source="backward-difference-v1"):
    segment = {"segment_id": "synthetic-segment", "time": [0., 1., 2., 3., 4.],
               "position": [[float(i*i), float(2*i)] for i in range(5)],
               "condition": [[] for _ in range(5)], "condition_available_at": [0., 1., 2., 3., 4.]}
    if source == "measured":
        segment["velocity"] = [[float(2*i), 2.] for i in range(5)]
    return {"schema_version": "pirc26-observed-block-v1", "block_id": "block", "coordinate_frame": m.spec.coordinate_frame,
            "state_units": ["m", "m", "m/s", "m/s"], "time_unit": "s", "velocity_source": source,
            "train_binding_hash": m.spec.train_binding_hash, "normalizer_hash": m.spec.normalizer_hash,
            "context_hash": m.spec.context_hash, "segments": [segment]}


def admitted(doc, role="train"):
    content = encode(doc).decode()
    identity = hashlib.sha256(content.encode()).hexdigest()
    return {"schema_version": "pirc26-admitted-block-v1", "protocol_hash": digest("synthetic-control"),
            "block_id": doc["block_id"], "source_identity": {"dataset_id": "synthetic", "release_id": "fixture-v1",
            "source_block_id": doc["block_id"], "sha256": identity}, "split_role": role, "purpose": PURPOSES[role],
            "content_sha256": identity, "content_utf8": content}


@pytest.mark.parametrize("source", ["measured", "backward-difference-v1"])
def test_causal_decode_uses_only_history_at_forecast_origin_and_separates_truth(source):
    m = model("M0")
    doc = document(m, source)
    first = decode_block(admitted(doc), m)
    req = first.forecast_request("synthetic-segment", 2, (2., 3., 4.), sample_count=8, brownian_root_id="a" * 64)
    expected_velocity = (4., 2.) if source == "measured" else (3., 2.)
    assert req.initial_state == (4., 4., *expected_velocity) and req.history_cutoff == 2.
    before = forecast(m, req)["samples"]
    changed = deepcopy(doc)
    changed["segments"][0]["position"][3:] = [[999., -999.], [888., -888.]]
    if source == "measured":
        changed["segments"][0]["velocity"][3:] = [[999., 999.], [888., 888.]]
    second = decode_block(admitted(changed), m)
    replay = second.forecast_request("synthetic-segment", 2, (2., 3., 4.), sample_count=8, brownian_root_id="a" * 64)
    assert replay == req and torch.equal(forecast(m, replay)["samples"], before)
    assert not torch.equal(first.truth("synthetic-segment", req), second.truth("synthetic-segment", replay))
    batches = first.transitions(batch_size=2)
    assert sum(len(b.time) for b in batches) == (4 if source == "measured" else 3)
    assert batches[0].state[0, 2].item() == (0. if source == "measured" else 1.)
    if source == "backward-difference-v1":
        with pytest.raises(ResearchError, match="causal observed velocity"):
            first.forecast_request("synthetic-segment", 0, (0., 1.), sample_count=8, brownian_root_id="a" * 64)


@pytest.mark.parametrize("fault", ["units", "frame", "binding", "future-context", "time", "dimensions", "duplicate-segment"])
def test_decode_refuses_invalid_physical_or_causal_contract_before_tensor_allocation(monkeypatch, fault):
    m = model("M0")
    doc = document(m)
    if fault == "units":
        doc["time_unit"] = "minutes"
    elif fault == "frame":
        doc["coordinate_frame"] = "foreign"
    elif fault == "binding":
        doc["train_binding_hash"] = "d" * 64
    elif fault == "future-context":
        doc["segments"][0]["condition_available_at"][2] = 3.
    elif fault == "time":
        doc["segments"][0]["time"][2] = 1.
    elif fault == "dimensions":
        doc["segments"][0]["position"][2].append(1.)
    else:
        doc["segments"].append(deepcopy(doc["segments"][0]))
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("bad input allocated tensors"))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        decode_block(admitted(doc), m)


def authorized_store(tmp_path, role):
    m = model("M0")
    store = ResearchStore(tmp_path, "pirc26-data-fixture", initialize=True)
    content = encode(document(m))
    (tmp_path / "synthetic-observations.json").write_bytes(content)
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "observations", "study_id": "data-fixture",
                "blocks": [{"block_id": "block", "dataset_id": "synthetic", "release_id": "fixture-v1",
                            "source_block_id": "block", "sha256": hashlib.sha256(content).hexdigest(),
                            "size_bytes": len(content), "path": "synthetic-observations.json", "split_role": role,
                            "fit_scope": role == "train"}]}
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    store.authorize({"authorization_id": "synthetic-data-grant", "study_id": "data-fixture", "protocol_hash": digest(protocol),
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("synthetic fixture grant, not production consent"),
        "data_root": str(tmp_path), "block_ids": ["block"], "purposes": [PURPOSES[role]], "visibilities": ["synthetic"],
        "test_authorization": False})
    return store, m


@pytest.mark.parametrize("role", ["train", "selection", "validation"])
def test_real_owner_read_logs_authority_and_nontrain_cannot_become_fitting_data(tmp_path, role):
    store, m = authorized_store(tmp_path, role)
    transport = read_block(store, "observations", "block", authorization_id="synthetic-data-grant", purpose=PURPOSES[role])
    block = decode_block(transport, m)
    completed = [e for e in store.events() if e["event_kind"] == "READ_COMPLETED"]
    assert len(completed) == 2 and all(e["payload"]["purpose"] == PURPOSES[role] for e in completed)
    if role != "train":
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            block.transitions(batch_size=16)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            read_block(store, "observations", "block", authorization_id="synthetic-data-grant", purpose="fit")
        assert store.events()[-1]["event_kind"] == "EXPOSURE_DENIED"


def test_final_eval_read_is_refused_without_explicit_test_authorization_before_file_open(tmp_path, monkeypatch):
    store, m = authorized_store(tmp_path, "final-eval")
    from application import research_data
    monkeypatch.setattr(research_data, "opened_regular_file", lambda *a, **k: pytest.fail("unauthorized final-eval opened"))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read_block(store, "observations", "block", authorization_id="synthetic-data-grant", purpose="evaluate")
    assert not any(e["event_kind"] == "READ_STARTED" for e in store.events())


def test_frozen_context_is_origin_only_and_changes_dynamics_without_future_leakage():
    from models.phase_space import AffineAccelerationDrift, PhaseSpaceSDE
    from tests.test_pirc26_dynamics import spec
    affine = AffineAccelerationDrift(2)
    with torch.no_grad():
        affine.context_weight.copy_(torch.eye(2))
    m = PhaseSpaceSDE(affine, torch.zeros(2, 2), spec(2)).double()
    doc = document(m)
    doc["segments"][0]["condition"] = [[1., -1.] for _ in range(5)]
    first = decode_block(admitted(doc), m)
    req = first.forecast_request("synthetic-segment", 2, (2., 3., 4.), sample_count=8, brownian_root_id="a" * 64)
    assert req.context == (1., -1.)
    doc["segments"][0]["condition"][3:] = [[999., 999.], [888., 888.]]
    second = decode_block(admitted(doc), m)
    assert second.forecast_request("synthetic-segment", 2, (2., 3., 4.), sample_count=8,
                                   brownian_root_id="a" * 64) == req
    endpoint = forecast(m, req)["samples"][0, -1]
    # Two physical one-second EM steps: acceleration is exactly the two
    # frozen origin context values, not future context rows from the file.
    assert torch.equal(endpoint, torch.tensor([11., 7., 5., 0.], dtype=torch.float64))
