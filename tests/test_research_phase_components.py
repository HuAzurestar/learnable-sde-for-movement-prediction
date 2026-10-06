"""Real four-state registered factories must enforce causal/bounded inputs."""

from dataclasses import replace
import inspect

import numpy as np
import pytest

from application.registry import ComponentRegistry
from experiments.pirc25 import affine
from experiments.pirc25.components import component_registries, fixture_component_bindings
from infrastructure.research_store import ResearchError


def components():
    registries = component_registries(4)
    bindings = fixture_component_bindings(4, 19, matrix_cells=1, registries=registries)
    return {role: registry.create_bound(bindings[role], matrix_cells=1) for role, registry in registries.items()}


def test_real_managed_four_state_predictor_never_receives_future_observations(monkeypatch):
    value = affine.fixture_spec(dimensions=4)
    original = ComponentRegistry.create_bound
    seen = []
    def trace(registry, binding, **kwargs):
        result = original(registry, binding, **kwargs)
        if "predictor" in binding["component_id"]:
            def predictor(*args, **options):
                seen.append((args, options))
                assert len(args) == 1, "registered predictor received a complete evaluation trajectory"
                assert "segment" not in options and "cutoff" not in options
                assert set(options) == {"initial_state", "time_grid", "n_samples", "rng", "condition_field"}
                return result(*args, **options)
            return predictor
        return result
    monkeypatch.setattr(ComponentRegistry, "create_bound", trace)
    affine.execute(value, value["cells"][0])
    assert len(seen) == 1
    segment = affine.synthetic_cohort().splits["evaluation"][0]
    cutoff = max(2, len(segment.time) // 2 - 1)
    expected = np.r_[segment.state[cutoff],
        (segment.state[cutoff] - segment.state[cutoff - 1]) / (segment.time[cutoff] - segment.time[cutoff - 1])]
    np.testing.assert_array_equal(seen[0][1]["initial_state"], expected)
    np.testing.assert_array_equal(seen[0][1]["time_grid"], segment.time[cutoff:])


def test_registered_four_state_preserves_existing_numerical_chain():
    legacy = affine.four_state(19)
    value = affine.fixture_spec(dimensions=4)
    actual = affine.execute(value, value["cells"][0])
    assert actual["source_schema"] == legacy["source_schema"]
    for key in ("metrics", "forecast", "fit"):
        assert actual[key] == legacy[key]


@pytest.mark.parametrize("fault", ["segments", "observations", "shape", "dtype"])
def test_actual_registered_trainer_rejects_unplanned_inputs_before_conversion(monkeypatch, fault):
    trainer = components()["trainer"]
    segment = affine.synthetic_cohort().splits["train"][0]
    rows = 33 if fault == "observations" else 8
    state = np.zeros((rows, 3 if fault == "shape" else 2), dtype=np.float32 if fault == "dtype" else np.float64)
    segment = replace(segment, time=np.arange(rows, dtype=np.float64), state=state)
    segments = (segment,) * (5 if fault == "segments" else 1)
    monkeypatch.setattr(affine.phase_space_api(), "_transition_rows",
        lambda *a, **k: pytest.fail("unplanned actual inputs reached four-state transition allocation"))
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED|CONTRACT_MISMATCH"):
        trainer(segments, condition_names=(), feature_basis="direct", ridge=1e-6)


@pytest.mark.parametrize("fault", ["paths", "steps", "dtype", "shape"])
def test_actual_registered_predictor_rejects_unplanned_inputs_before_rollout(fault):
    factories = components()
    cohort = affine.synthetic_cohort()
    model = factories["trainer"](cohort.splits["train"], condition_names=(), feature_basis="direct", ridge=1e-6,
        model_factory=factories["model"])
    predictor = factories["predictor"]
    segment = cohort.splits["evaluation"][0]
    if "segment" in inspect.signature(predictor).parameters:
        # Baseline uses the existing raw function, so exercise its real small
        # over-cap allocations rather than failing just on absent new syntax.
        if fault == "steps":
            segment = replace(segment, time=np.arange(20, dtype=np.float64), state=np.zeros((20, 2)))
        if fault == "dtype":
            segment = replace(segment, state=segment.state.astype(np.float32))
        if fault == "shape":
            segment = replace(segment, state=np.zeros((8, 3)))
        arguments = {"segment": segment, "cutoff": 3}
    else:
        arguments = {"initial_state": np.ones(3 if fault == "shape" else 4,
            dtype=np.float32 if fault == "dtype" else np.float64),
            "time_grid": np.arange(10 if fault == "steps" else 5, dtype=np.float64)}
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED|CONTRACT_MISMATCH"):
        predictor(model, **arguments, n_samples=9 if fault == "paths" else 8,
            rng=np.random.default_rng(19), condition_field=None)


def test_registered_trainer_cannot_fall_back_to_unversioned_model_constructor():
    trainer = components()["trainer"]
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        trainer(affine.synthetic_cohort().splits["train"], condition_names=(), feature_basis="direct", ridge=1e-6)


@pytest.mark.parametrize("fault", ["weights-dtype", "weights-shape", "diffusion-list", "diffusion-nan"])
def test_predictor_rejects_actual_model_profile_before_any_rollout(monkeypatch, fault):
    factories = components()
    model = factories["trainer"](affine.synthetic_cohort().splits["train"], condition_names=(),
        model_factory=factories["model"])
    changes = {"weights-dtype": {"weights": model.weights.astype(np.float32)},
        "weights-shape": {"weights": np.ones((3, 3))},
        "diffusion-list": {"diffusion_covariance": [[1., 0.], [0., 1.]]},
        "diffusion-nan": {"diffusion_covariance": np.full((2, 2), np.nan)}}
    model = replace(model, **changes[fault])
    monkeypatch.setattr(affine.phase_space_api(), "rollout_phase_space_from_state",
        lambda *a, **k: pytest.fail("invalid model reached the actual rollout allocation"))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        factories["predictor"](model, initial_state=np.ones(4), time_grid=np.arange(5, dtype=np.float64),
            n_samples=8, rng=np.random.default_rng(19))


def test_direct_four_state_adapter_checks_complete_combination_before_any_factory(monkeypatch):
    registries = component_registries(4)
    bindings = fixture_component_bindings(4, 19, matrix_cells=1, registries=registries)
    # All role bindings remain individually valid; adapter seed must still
    # reject the whole combination before the first actual constructor.
    monkeypatch.setattr(ComponentRegistry, "create_bound", lambda *a, **k: pytest.fail("unbound combination constructed"))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        affine.four_state(20, component_bindings=bindings, matrix_cells=1, registries=registries)
