"""Numerical qualification fixtures, not registered research or real data."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from domain import ModelContext
from experiments.pirc28.synthetic import ScenarioSpec, generate_scenario, simulate_path
from infrastructure.research_store import digest
from models import ExactTransitionProvider, SDEModel
from models.switching import SwitchingError, SwitchingSDEModel


def fixture_poll():
    """Tiny unit fixtures; the research worker must supply supervisor control."""


def model(k=2, rate=0.4, noise=0.1):
    return SwitchingSDEModel(np.array([[-rate, rate], [rate, -rate]]) if k == 2 else [[0.0]],
        np.zeros((k, 2, 2)), np.zeros((k, 2, 2)), np.array([[1.0, 0], [-1.0, 0]])[:k],
        np.tile(np.eye(2) * noise, (k, 1, 1)))


def scenario(k=2, paths=4, rate=0.4, noise=0.1):
    return ScenarioSpec("qualification-fixture", model(k, rate, noise).to_dict(),
        (0.0, 1.0, 2.0, 4.0), paths, 37, tuple(np.ones(k) / k),
        tuple(np.zeros((k, 4)).tolist()), tuple(np.zeros((k, 4, 4)).tolist()),
        tuple(np.eye(4)[:2].tolist()), tuple((np.eye(2) * 0.01).tolist()))


def test_joint_model_is_not_an_exact_gaussian_kernel():
    m = model()
    assert isinstance(m, SDEModel)
    assert not isinstance(m, ExactTransitionProvider)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert m.drift(torch.tensor(0.0), x, ModelContext(regime=0)).tolist() == [3, 4, 1, 0]
    assert torch.equal(m.diffusion(None, x, ModelContext(regime=1))[:2], torch.zeros((2, 2)))
    with pytest.raises(SwitchingError):
        m.drift(None, x, ModelContext())


@pytest.mark.parametrize("q", [[[0, 0.4], [0.4, -0.4]], [[0.4, -0.4], [0.4, -0.4]],
    [[float('nan'), 0.4], [0.4, -0.4]], [[-101, 101], [0.4, -0.4]]])
def test_invalid_generators_refused(q):
    with pytest.raises(SwitchingError):
        SwitchingSDEModel(q, np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2)), np.zeros((2, 2, 2)))


def test_softplus_and_model_roundtrip():
    m = model()
    fitted = SwitchingSDEModel.from_rate_logits([-2.0, 1.0], m.A.numpy(), m.B.numpy(), m.a.numpy(), m.L.numpy())
    assert np.allclose(fitted.Q.numpy().sum(axis=1), 0)
    assert np.allclose(-fitted.Q.numpy().diagonal(), np.logaddexp(0, [-2, 1]))
    assert SwitchingSDEModel.from_dict(fitted.to_dict()).to_dict() == fitted.to_dict()


def test_conditional_affine_kernel_matches_integrated_acceleration():
    dt, sigma = 2.3, 0.4
    m = model(k=1, noise=sigma)
    f, c, s = (x.numpy() for x in m.conditional_affine_transition(0, dt))
    assert np.allclose(f[:2, 2:], np.eye(2) * dt)
    assert np.allclose(c, [dt**2 / 2, 0, dt, 0])
    expected = np.block([[np.eye(2) * dt**3 / 3, np.eye(2) * dt**2 / 2],
                         [np.eye(2) * dt**2 / 2, np.eye(2) * dt]]) * sigma**2
    assert np.allclose(s, expected, atol=1e-12)


def test_k1_degeneracy_dwell_and_zero_noise():
    s = replace(scenario(k=1, noise=0), R=((0, 0), (0, 0)))
    obs, truth, manifest = generate_scenario(s, check_budget=fixture_poll)
    assert np.all(truth.modes == 0)
    assert all(not times for times in truth.jump_times)
    assert np.allclose(truth.dwell_by_mode, 4)
    assert np.allclose(truth.states[:, :, 0], np.array(s.times)**2 / 2)
    assert np.allclose(obs.values, truth.states[:, :, :2])
    assert manifest['occupied_K'] == 1


def test_replay_observation_noise_independence_and_truth_isolation():
    s = scenario()
    obs, truth, manifest = generate_scenario(s, check_budget=fixture_poll)
    obs2, truth2, manifest2 = generate_scenario(s, check_budget=fixture_poll)
    assert digest(obs.to_dict()) == digest(obs2.to_dict())
    assert digest(truth.to_dict()) == digest(truth2.to_dict())
    assert manifest == manifest2
    noisy = replace(s, R=((10.0, 0), (0, 10.0)))
    obs3, truth3, _ = generate_scenario(noisy, check_budget=fixture_poll)
    assert np.array_equal(truth.states, truth3.states)
    assert truth.jump_times == truth3.jump_times
    assert not np.array_equal(obs.values, obs3.values)
    assert set(obs.to_dict()) == {'schema_version', 'scenario_hash', 'times', 'values', 'H', 'R'}
    assert not hasattr(obs, 'modes') and not hasattr(obs, 'states')


def test_permutation_relabels_all_truth_without_changing_observations():
    s = scenario(paths=6)
    obs, truth, _ = generate_scenario(s, check_budget=fixture_poll)
    obs2, truth2, _ = generate_scenario(replace(s, label_permutation=(1, 0)), check_budget=fixture_poll)
    assert np.array_equal(obs.values, obs2.values)
    assert np.array_equal(truth.states, truth2.states)
    assert np.array_equal(truth2.modes, 1 - truth.modes)
    assert np.array_equal(truth2.dwell_by_mode, truth.dwell_by_mode[:, ::-1])
    assert truth2.jump_modes == tuple(tuple(1 - m for m in row) for row in truth.jump_modes)
    m = model()
    swapped = m.permuted([1, 0])
    for old, new in ((0, 1), (1, 0)):
        for left, right in zip(m.conditional_affine_transition(old, 0.7), swapped.conditional_affine_transition(new, 0.7)):
            assert torch.equal(left, right)
    assert np.allclose(swapped.mode_transition(2).numpy(), m.mode_transition(2).numpy()[::-1, ::-1])


def test_multiple_jumps_do_not_become_a_single_endpoint_segment():
    class Clock:
        def exponential(self, scale):
            return 0.25
        def choice(self, k, p):
            return int(np.argmax(p))
    z, s, jumps, modes, dwell = simulate_path(model(noise=0), [0, 1.1], 0, np.zeros(4),
        Clock(), np.random.default_rng(7), max_jumps=10, check_budget=fixture_poll)
    assert jumps == (0.25, 0.5, 0.75, 1.0)
    assert modes == (0, 1, 0, 1, 0)
    assert s[-1] == 0  # Same endpoint mode still includes four jumps.
    assert np.allclose(dwell, [0.6, 0.5])
    assert np.isclose(z[-1, 2], 0.1)
    assert not np.isclose(z[-1, 0], 1.1**2 / 2)


def test_exponential_clock_endpoint_probability_and_occupancy():
    # Independent unit-fixture paths, not seeds/points treated as replicates.
    s = replace(scenario(paths=300, noise=0), times=(0.0, 4.0), R=((0, 0), (0, 0)), initial_mode_probability=(1.0, 0.0))
    _, truth, _ = generate_scenario(s, check_budget=fixture_poll)
    expected = model().mode_transition(4).numpy()[0, 1]
    assert abs(np.mean(truth.modes[:, -1] == 1) - expected) < 0.08
    assert abs(np.mean([len(row) == 0 for row in truth.jump_times]) - np.exp(-0.4 * 4)) < 0.07
    assert np.allclose(truth.dwell_by_mode.sum(axis=1), 4)


@pytest.mark.parametrize('change', [dict(times=(0, 1, 1)), dict(initial_mode_probability=(0.4, 0.4)),
    dict(label_permutation=(0, 0)), dict(R=((-1, 0), (0, 1))), dict(seed=True),
    dict(sample_count=100_001), dict(max_jumps=1), dict(H=((1, 0),))])
def test_bad_scenario_and_preflight_quotas_refused(change):
    with pytest.raises(SwitchingError):
        replace(scenario(), **change).validated()


def test_actual_jump_limit_refuses_incomplete_path():
    class Clock:
        def exponential(self, scale):
            return 0.1
        def choice(self, k, p):
            return int(np.argmax(p))
    with pytest.raises(SwitchingError, match='incomplete') as error:
        simulate_path(model(), [0, 1], 0, np.zeros(4), Clock(), np.random.default_rng(4),
            max_jumps=2, check_budget=fixture_poll)
    assert error.value.code == 'JUMP_LIMIT'


def test_budget_poll_prevents_computation():
    def stop():
        raise TimeoutError('shared supervisor deadline')
    with pytest.raises(TimeoutError):
        generate_scenario(scenario(), check_budget=stop)


def test_jump_storage_quota_and_invalid_direct_schedule():
    with pytest.raises(SwitchingError) as error:
        replace(scenario(paths=101), max_jumps=100_000).validated()
    assert error.value.code == 'RESOURCE_LIMIT'
    with pytest.raises(SwitchingError):
        simulate_path(model(), [0, 2, 1], 0, np.zeros(4), np.random.default_rng(1),
            np.random.default_rng(2), max_jumps=10, check_budget=fixture_poll)


def test_k1_row_sum_roundoff_is_canonicalized_to_absorbing():
    m = model(k=1)
    saved = m.to_dict()
    saved['Q'] = [[-1e-15]]
    assert SwitchingSDEModel.from_dict(saved).Q.item() == 0
