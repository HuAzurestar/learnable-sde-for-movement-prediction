"""Independent physical-contract, interpolation and trainable-init checks."""

import copy
import hashlib
import json

import pytest
import torch

from domain import ModelContext
from models.phase_space import (AffineAccelerationDrift, DynamicsSpec, ModelContractError,
    NeuralResidualAccelerationDrift, PhaseSpaceSDE, RBFResidualDrift, SplineResidualDrift,
    stability_diagnostics)


def spec(context_dim=0):
    return DynamicsSpec("synthetic-local-cartesian", "a" * 64, "b" * 64, "c" * 64,
                        context_dim, (0.,) * (4 + context_dim), (1.,) * (4 + context_dim))


def model(family, dtype=torch.float64):
    value = spec()
    affine = AffineAccelerationDrift()
    with torch.no_grad():
        affine.A.copy_(torch.tensor([[-.2, .1], [-.1, -.3]]))
        affine.B.copy_(torch.tensor([[.01, 0.], [0., .02]]))
    if family == "M1-R":
        drift = RBFResidualDrift(affine, value, [2, 3], [[0., 0.], [1., -1.]], [1., 2.])
    elif family == "M1-S":
        drift = SplineResidualDrift(affine, value, [2], [[-2., -2., -2., -2., 0., 2., 2., 2., 2.]])
    elif family == "M2":
        drift = NeuralResidualAccelerationDrift(affine, value, [2, 3], hidden=(16, 16), seed=12)
    else:
        drift = affine
    return PhaseSpaceSDE(drift, [[.4, 0.], [.1, .3]], value).to(dtype=dtype)


def inputs(dtype=torch.float64):
    return torch.tensor([0., 2., 5.], dtype=dtype), torch.tensor(
        [[0., 1., 2., 3.], [1., 2., -.2, .4], [5., 7., 0., -1.]], dtype=dtype), ModelContext()


@pytest.mark.parametrize("family", ["M0", "M1-R", "M1-S", "M2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_physical_rows_zero_residual_and_covariance(family, dtype):
    m = model(family, dtype)
    t, z, c = inputs(dtype)
    affine = model("M0", dtype)
    assert torch.equal(m.drift(t, z, c), affine.drift(t, z, c))
    assert torch.equal(m.drift(t, z, c)[:, :2], z[:, 2:])
    noise = m.diffusion(t, z, c)
    assert noise.shape == (3, 4, 2) and torch.count_nonzero(noise[:, :2]) == 0
    assert torch.equal(noise[0, 2:] @ noise[0, 2:].T, m.velocity_factor @ m.velocity_factor.T)
    assert not m.supports("L2", "generic-rollout", "exact")["supported"]


@pytest.mark.parametrize("family", ["M1-R", "M1-S", "M2"])
def test_explicit_residual_off_is_exact_affine(family):
    m = model(family)
    for parameter in (m.acceleration_model.coefficients,) if family != "M2" else (m.acceleration_model.layers[-1].bias,):
        with torch.no_grad():
            parameter.fill_(.7)
    t, z, c = inputs()
    assert not torch.equal(m.drift(t, z, c), model("M0").drift(t, z, c))
    m.acceleration_model.residual_enabled = False
    assert torch.equal(m.drift(t, z, c), model("M0").drift(t, z, c))


def test_neural_zero_initialization_learns_and_bounds_physical_acceleration():
    m = model("M2")
    t, z, c = inputs()
    before = m.acceleration(t, z, c).detach().clone()
    loss = (m.acceleration(t, z, c) - (before + .3)).square().mean()
    loss.backward()
    last = m.acceleration_model.layers[-1]
    assert last.weight.grad.norm() > 0 and torch.isfinite(last.weight.grad).all()
    optimizer = torch.optim.SGD(m.acceleration_model.layers.parameters(), lr=.2)
    optimizer.step()
    assert not torch.equal(m.acceleration(t, z, c), before)
    with torch.no_grad():
        last.weight.fill_(1e6)
        last.bias.fill_(1e6)
    residual = m.acceleration(t, z * 1e4, c) - m.acceleration_model.affine(t, z * 1e4, z.new_empty((3, 0)))
    assert (residual.abs() <= 1. + 1e-12).all()
    report = stability_diagnostics(m, t, z, c)
    assert report["max_jacobian_norm"] >= 0 and not report["drift_clipping"]


def test_construction_preserves_rng_and_seed_replays_hidden_parameters():
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    first = model("M2")
    assert torch.equal(before, torch.random.get_rng_state())
    second = model("M2")
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_spline_partition_endpoint_and_clamped_extrapolation():
    m = model("M1-S").acceleration_model
    u = torch.tensor([[-20.], [-2.], [-1.], [0.], [1.], [2.], [20.]], dtype=torch.float64)
    basis = m.basis(u)
    assert torch.allclose(basis.sum(-1), torch.ones(len(u), dtype=u.dtype), atol=1e-12)
    assert (basis >= 0).all()
    assert torch.equal(basis[0], basis[1]) and torch.equal(basis[-1], basis[-2])
    # A constant coefficient represents the same constant throughout the domain.
    with torch.no_grad():
        m.coefficients.fill_(.25)
    assert torch.allclose(m.residual(u), torch.full((len(u), 2), .25, dtype=u.dtype))


def test_rbf_center_value_decay_and_input_gradients():
    drift = model("M1-R").acceleration_model
    u = torch.tensor([[0., 0.], [100., 100.]], dtype=torch.float64)
    assert drift.basis(u)[0, 0] == 1 and drift.basis(u)[1, 0] == 0
    u = torch.tensor([[.3, -.2]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(drift.basis, (u,))
    spline = model("M1-S").acceleration_model
    assert torch.autograd.gradcheck(spline.basis, (u[:, :1].detach().requires_grad_(True),))


@pytest.mark.parametrize("family", ["M0", "M1-R", "M1-S", "M2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_roundtrip_is_json_and_exact(family, dtype):
    m = model(family, dtype)
    if family != "M0":
        m.acceleration_model.residual_enabled = False
    checkpoint = json.loads(json.dumps(m.checkpoint()))
    restored = PhaseSpaceSDE.from_checkpoint(checkpoint)
    t, z, c = inputs(dtype)
    assert torch.equal(m.drift(t, z, c), restored.drift(t, z, c))
    assert restored.checkpoint() == checkpoint


def resign(checkpoint):
    value = copy.deepcopy(checkpoint)
    value.pop("sha256")
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
    return value


def test_corrupt_or_semantically_incompatible_checkpoint_is_rejected():
    original = model("M2").checkpoint()
    bad = copy.deepcopy(original)
    bad["state"]["velocity_factor"]["data"][0][0] = 1.
    with pytest.raises(ModelContractError):
        PhaseSpaceSDE.from_checkpoint(bad)
    for key, value in [("state_names", ["vx", "vy", "x", "y"]), ("units", ["km"] * 4), ("noise_dim", 4)]:
        bad = copy.deepcopy(original)
        bad["model_card"][key] = value
        with pytest.raises(ModelContractError):
            PhaseSpaceSDE.from_checkpoint(resign(bad))


def test_nonzero_residual_checkpoint_and_nonintegral_normalizer_replay():
    value = DynamicsSpec("local", "a" * 64, "b" * 64, "c" * 64,
                         means=(.1, .2, .3, .4), scales=(.9, 1.1, 1.3, 1.7))
    drift = NeuralResidualAccelerationDrift(AffineAccelerationDrift(), value, [2, 3])
    with torch.no_grad():
        drift.layers[-1].weight.fill_(.1)
    original = PhaseSpaceSDE(drift, torch.eye(2), value).double()
    restored = PhaseSpaceSDE.from_checkpoint(original.checkpoint())
    assert torch.equal(original.drift(*inputs()), restored.drift(*inputs()))


def test_context_and_shape_contract_rejects_broadcasting_and_wrong_dtype():
    value = spec(1)
    m = PhaseSpaceSDE(AffineAccelerationDrift(1), torch.eye(2), value).double()
    t, z, _ = inputs()
    c = ModelContext(torch.ones((3, 1), dtype=torch.float64))
    assert m.drift(t, z, c).shape == (3, 4)
    for bad in (ModelContext(), ModelContext(torch.ones(1, dtype=torch.float64)), ModelContext(c.condition, regime=0)):
        with pytest.raises(ModelContractError):
            m.drift(t, z, bad)
    with pytest.raises(ModelContractError):
        m.drift(t.float(), z.float(), c)
    with pytest.raises(ModelContractError):
        m.drift(t, z[:, :2], c)
    z[0, 0] = float("nan")
    with pytest.raises(ModelContractError):
        m.drift(t, z, c)


def test_unregistered_basis_capacity_and_normalizer_fail_closed():
    affine, value = AffineAccelerationDrift(), spec()
    with pytest.raises(ModelContractError):
        RBFResidualDrift(affine, value, [0], torch.zeros(129, 1), torch.ones(129))
    with pytest.raises(ModelContractError):
        NeuralResidualAccelerationDrift(affine, value, [0], hidden=(64,))
    with pytest.raises(ModelContractError):
        DynamicsSpec("local", "a" * 64, "b" * 64, "c" * 64, scales=(0., 1., 1., 1.))
