"""Real composition factories must consume sealed versions, not only labels.

These are engineering checks, not qualification of a scientific method.
Managed worker integration and the four-state adapter remain separate scope.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from application.experiment import ExperimentApplication
from application.registry import ComponentRegistry
from application.research_execution import execution_binding
from application.research_registry import implementation_hash
from application.synthetic import make_synthetic_em_data
from config import Components, Config
from domain import ForecastRequest, ModelContext
from experiments.pirc25.plugins import affine_configuration
from infrastructure.research_store import ResearchError
import registry as root
from tests.research_admission_fixtures import synthetic_plugin


def config():
    return Config(seed=19, components=Components(model="I1", estimator="EM", inference="exact"),
        model={"I1": {"n_modes": 2, "kappa": 0.0, "dt_ref": 1.0}},
        protocol={"em": {"max_iter": 2}})


def profile():
    return {"state_dim": 2, "noise_dim": 1, "diffusion_support": ["vx"],
        "dtype": "float64", "device": "cpu", "observation_profile": "segment-trajectories",
        "observations": 40, "paths": 8, "steps": 8, "components": 4}


def construct(document, inputs, context):
    context.append((deepcopy(document), deepcopy(inputs)))
    return {"constructed": document}


def component_registry():
    plugin = synthetic_plugin("component-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", construct)
    entry = replace(plugin.registry_entry, component_kind="model", code_hash=implementation_hash(construct))
    value = ComponentRegistry()
    value.register_version(entry, construct)
    return value, entry


def test_actual_single_axis_noise_dimension_matches_execution_declaration():
    model = root.build_model(config())
    declaration, _ = affine_configuration(1)
    assert declaration["noise_dim"] == model.noise_dim, "the actual model uses one velocity noise coordinate"


def test_versioned_component_constructs_only_after_frozen_plan_validation():
    value, entry = component_registry()
    binding = execution_binding(entry, {}, {}, matrix_cells=1)
    calls = []
    assert value.create_bound(binding, matrix_cells=1, context=calls) == {"constructed": {}}
    assert calls == [({}, {})]
    binding["resource_plan_hash"] = "0" * 64
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        value.create_bound(binding, matrix_cells=1, context=calls)
    assert len(calls) == 1


def test_versioned_component_never_falls_back_to_unversioned_builder():
    value, entry = component_registry()
    value.register("component-fixture", lambda _: pytest.fail("legacy fallback must not execute"))
    binding = execution_binding(entry, {}, {}, matrix_cells=1)
    binding["component_version"] = "unavailable"
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        value.create_bound(binding, matrix_cells=1, context=[])


def test_real_application_uses_frozen_versions_without_changing_affine_numerics():
    cfg = config()
    bindings = root.component_bindings(cfg, profile(), matrix_cells=1)
    plan = root.plan_components(cfg, bindings, matrix_cells=1)
    assert {value["component_kind"] for value in plan["entries"].values()} == {"model", "trainer", "predictor"}
    versioned = ExperimentApplication.from_config(cfg, component_bindings=bindings, matrix_cells=1)
    legacy = ExperimentApplication.from_config(cfg)
    request = ForecastRequest(torch.tensor([1.0, 0.25], dtype=torch.float64),
        torch.tensor([1.0, 2.0], dtype=torch.float64), 8, ModelContext(regime=0))
    assert torch.equal(versioned.predict(versioned.model, request).samples, legacy.predict(legacy.model, request).samples)
    data, _ = make_synthetic_em_data(n_segments=4, length=10, dt=1.0, seed=cfg.seed)
    versioned.train(data)
    legacy.train(data)
    # Existing test_shared_runner uses these pre-existing float64 tolerances.
    # Repeated legacy least-squares fitting itself is not bitwise deterministic.
    torch.testing.assert_close(versioned.predict(versioned.model, request).samples,
        legacy.predict(legacy.model, request).samples, rtol=1e-10, atol=1e-12)
    assert versioned.component_plan == plan


@pytest.mark.parametrize("fault", ["version", "identity", "config", "noise", "observations", "resource-plan", "missing-role"])
def test_real_application_rejects_invalid_component_before_any_factory(tmp_path, monkeypatch, fault):
    cfg = config()
    bindings = root.component_bindings(cfg, profile(), matrix_cells=1)
    if fault == "version":
        bindings["model"]["component_version"] = "unavailable"
    elif fault == "identity":
        bindings["predictor"]["registry_entry_hash"] = "0" * 64
    elif fault == "config":
        cfg.model["I1"]["n_modes"] = 3
    elif fault == "noise":
        bindings["model"]["inputs"]["noise_dim"] = 2
    elif fault == "observations":
        bindings["trainer"]["inputs"]["observation_profile"] = "future-evaluation"
    elif fault == "resource-plan":
        bindings["predictor"]["resource_plan_hash"] = "0" * 64
    else:
        bindings.pop("trainer")
    monkeypatch.setattr(ComponentRegistry, "create_bound", lambda *a, **k: pytest.fail("invalid plan invoked a factory"))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH|RESOURCE_PLAN_REJECTED"):
        ExperimentApplication.from_config(cfg, component_bindings=bindings, matrix_cells=1)
