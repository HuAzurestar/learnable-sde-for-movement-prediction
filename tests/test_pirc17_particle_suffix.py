"""Suffix/full-array software oracles; no empirical inheritance or timing claim."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.pirc17 import particle_suffix as suffix
from experiments.pirc17.brownian import BrownianPath, integration_grid
from experiments.pirc17.configurations import terrain_configurations
from experiments.pirc17.origins import VelocityPrior, causal_prefix, known_velocity
from experiments.pirc17.rollout import Features, rollout
from tests.test_pirc17_direct_linear import fitted

HORIZONS = [2.3, 5.9, 10.2, 17.6]
STEPS = [.625, .3125]
SEED = 20260814
STREAM = "suffix-software-origin"


def origin(mode):
    if mode == "known_velocity":
        return known_velocity([3., -2.], 12., [.8, -.1], source="software-observation")
    if mode == "causal_prefix":
        return causal_prefix([[1., 0.], [2., -.8], [3., -2.]], [2., 7., 12.])
    return VelocityPrior([[.8, -.1], [-.3, .4], [1.2, .9], [0., 0.]], "train-fixture").at([3., -2.], 12.)


def driver(count):
    return BrownianPath(HORIZONS, STEPS, history_step_seconds=5., particles=count,
                        seed=SEED, stream_id=STREAM)


def callbacks():
    counts = {"drift_rows": 0, "terrain_rows": 0, "invalid_rows": 0}

    def drift(state):
        counts["drift_rows"] += len(state.positions_m)
        return .7*state.velocities_mps + .03*np.sin(state.positions_m)

    def diffusion(state):
        root = np.array([[.3, .06], [.02, .4]])
        return np.broadcast_to(root, (len(state.positions_m), 2, 2))

    def terrain(state):
        values = np.column_stack((np.sin(state.positions_m[:, 0]), state.history_direction[0][:, 1]))
        valid = np.column_stack((state.positions_m[:, 0] > 2., np.ones(len(values), dtype=bool)))
        counts["terrain_rows"] += len(values)
        counts["invalid_rows"] += int(np.count_nonzero(~valid.all(axis=1)))
        return Features(values, valid)

    def conditioner(state, matrix):
        return np.column_stack((.1*matrix[:, 0] + .03*matrix[:, 2], -.02*matrix[:, 1]))

    return dict(base_drift=drift, diffusion=diffusion, terrain=terrain, conditioner=conditioner), counts


def run_suffix(o, path, prefix, step, **kw):
    return suffix.rollout_suffix(o, HORIZONS, prefix_particles=prefix, driver=path,
        seed=SEED, stream_id=STREAM, max_step_seconds=step, **kw)


@pytest.mark.parametrize("mode", ["known_velocity", "causal_prefix", "point_only"])
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("prefix,total", [(1, 2), (7, 16), (16, 32)])
def test_suffix_is_exact_and_only_new_particles_query_features(mode, step, prefix, total):
    o, path = origin(mode), driver(total)
    preserved = path.values.copy(), o.history_positions_m.copy(), deepcopy(path.identity)
    whole_callbacks, _ = callbacks()
    whole = rollout(o, HORIZONS, particles=total, seed=SEED, max_step_seconds=step,
                    brownian_increments=path, **whole_callbacks)
    partial_callbacks, counts = callbacks()
    actual = run_suffix(o, path, prefix, step, **partial_callbacks)
    np.testing.assert_array_equal(actual.forecast.positions_m, whole.positions_m[prefix:])
    np.testing.assert_array_equal(actual.forecast.elapsed_seconds, whole.elapsed_seconds)
    intervals = len(integration_grid(HORIZONS, step, 5.)) - 1
    assert counts["drift_rows"] == counts["terrain_rows"] == (total-prefix)*intervals
    assert actual.forecast.feature_query_rows == counts["terrain_rows"]
    assert actual.forecast.invalid_feature_rows == counts["invalid_rows"]
    assert actual.new_particles == total-prefix and actual.total_particles == total
    assert actual.prefix_particles == prefix and actual.version == suffix.VERSION
    assert actual.brownian_identity == path.identity and actual.brownian_identity is not path.identity
    assert not actual.forecast.positions_m.flags.writeable
    np.testing.assert_array_equal(path.values, preserved[0])
    np.testing.assert_array_equal(o.history_positions_m, preserved[1])
    assert path.identity == preserved[2]


@pytest.mark.parametrize("mode", ["known_velocity", "causal_prefix", "point_only"])
def test_two_successive_appends_equal_one_full_budget_without_prefix_rollout(mode):
    o = origin(mode)
    first_callbacks, _ = callbacks()
    saved = rollout(o, HORIZONS, particles=7, seed=SEED, max_step_seconds=.3125,
                    brownian_increments=driver(7), **first_callbacks).positions_m
    initial = saved.copy()
    computed = 7
    for count in (16, 32):
        kw, counters = callbacks()
        new = run_suffix(o, driver(count), len(saved), .3125, **kw)
        assert counters["drift_rows"] == new.new_particles*(len(integration_grid(HORIZONS, .3125, 5.))-1)
        computed += new.new_particles
        saved = np.concatenate((saved, new.forecast.positions_m), axis=0)
    whole_callbacks, _ = callbacks()
    whole = rollout(o, HORIZONS, particles=32, seed=SEED, max_step_seconds=.3125,
                    brownian_increments=driver(32), **whole_callbacks)
    assert computed == 32
    np.testing.assert_array_equal(saved[:7], initial)
    np.testing.assert_array_equal(saved, whole.positions_m)


@pytest.mark.parametrize("name", list(terrain_configurations()))
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("total", [32, 2048])
def test_actual_direct_linear_callbacks_match_complete_budget(name, step, total):
    model = fitted(name)
    width = model.conditioner_checkpoint["input_dim"]//2
    o, path = origin("causal_prefix"), driver(total)

    def terrain(state):
        columns = np.arange(1, width+1, dtype=float)[None, :]
        values = np.sin(state.positions_m[:, :1]/columns) + .02*state.history_direction[0][:, 1:2]
        valid = np.broadcast_to(state.positions_m[:, :1] > 0., values.shape)
        return Features(values, valid)

    kw = dict(base_drift=model.base_drift, diffusion=model.diffusion,
              terrain=terrain, conditioner=model.correction)
    whole = rollout(o, HORIZONS, particles=total, seed=SEED, max_step_seconds=step,
                    brownian_increments=path, **kw)
    actual = run_suffix(o, path, total//2, step, **kw)
    np.testing.assert_array_equal(actual.forecast.positions_m, whole.positions_m[total//2:])


@pytest.mark.parametrize("prefix", [None, True, False, 0, -1, 8, 9, 2.5, "2"])
def test_invalid_prefix_rejected_before_callbacks(prefix):
    kw, counts = callbacks()
    with pytest.raises(ValueError, match="prefix"):
        run_suffix(origin("known_velocity"), driver(8), prefix, .625, **kw)
    assert counts == {"drift_rows": 0, "terrain_rows": 0, "invalid_rows": 0}


@pytest.mark.parametrize("changes", [
    {"seed": SEED+1}, {"seed": True}, {"stream_id": "another-origin"},
    {"max_step_seconds": .2}, {"max_step_seconds": 0.}, {"history_step_seconds": 0.},
    {"horizons_seconds": [2.3, 5.9]}, {"horizons_seconds": [2.3, 5.9, 10.2, 20.]},
    {"horizons_seconds": [5., 2.]}, {"horizons_seconds": [np.nan]},
    {"origin": object()}, {"driver": object()},
])
def test_identity_or_grid_mismatch_rejected_before_callbacks(changes):
    kw, counts = callbacks()
    args = dict(origin=origin("known_velocity"), horizons_seconds=HORIZONS,
                prefix_particles=4, driver=driver(8), seed=SEED, stream_id=STREAM,
                max_step_seconds=.625, **kw)
    args.update(changes)
    with pytest.raises(ValueError):
        suffix.rollout_suffix(**args)
    assert counts == {"drift_rows": 0, "terrain_rows": 0, "invalid_rows": 0}


def test_no_terrain_has_no_queries_and_model_failure_is_not_swallowed():
    kw, counts = callbacks()
    actual = run_suffix(origin("known_velocity"), driver(8), 4, .625,
                        base_drift=kw["base_drift"], diffusion=kw["diffusion"])
    assert actual.forecast.feature_query_rows == actual.forecast.invalid_feature_rows == 0
    assert counts["terrain_rows"] == 0

    def failing(state):
        raise MemoryError("software fresh resource refusal")

    with pytest.raises(MemoryError, match="resource refusal"):
        run_suffix(origin("known_velocity"), driver(8), 4, .625,
                   base_drift=failing, diffusion=kw["diffusion"])


def test_counterfactual_coupled_callbacks_are_not_a_supported_equivalence_claim():
    o, path = origin("known_velocity"), driver(16)

    def coupled(state):
        # Explicitly outside the helper's particle-local contract.
        return .7*state.velocities_mps + .1*state.positions_m.mean(axis=0)

    diffusion = lambda s: np.broadcast_to(np.eye(2), (len(s.positions_m), 2, 2))
    whole = rollout(o, HORIZONS, particles=16, seed=SEED, max_step_seconds=.625,
                    brownian_increments=path, base_drift=coupled, diffusion=diffusion)
    actual = run_suffix(o, path, 8, .625, base_drift=coupled, diffusion=diffusion)
    assert not np.array_equal(actual.forecast.positions_m, whole.positions_m[8:])
