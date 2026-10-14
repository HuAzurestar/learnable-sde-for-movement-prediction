"""Analytic covariance, legacy equivalence and nonlinear recovery fixtures."""

import copy

import numpy as np
import pytest
import torch

from application.pirc26_dynamics import (AffinePhaseSpaceOracle, adapt_legacy_affine,
    dynamics_identity, freeze_dynamics, load_frozen_dynamics)
from experiments.nex326.phase_space import AffineVelocityModel
from infrastructure.research_store import ResearchError
from models.phase_space import AffineAccelerationDrift, ModelContractError, PhaseSpaceSDE
from tests.test_pirc26_dynamics import inputs, model, spec
from domain import ModelContext


def test_exact_free_velocity_covariance_matches_integrated_brownian_formula():
    m = PhaseSpaceSDE(AffineAccelerationDrift(), torch.zeros((2, 2)), spec()).double()
    m.velocity_factor.copy_(torch.tensor([[.4, 0.], [.1, .3]], dtype=torch.float64))
    dt = 2.3
    oracle = AffinePhaseSpaceOracle(m)
    transition, offset, covariance = oracle.affine_transition(dt, ModelContext())
    q = m.velocity_factor @ m.velocity_factor.T
    assert torch.allclose(covariance[:2, :2], q * dt**3 / 3, atol=1e-12)
    assert torch.allclose(covariance[:2, 2:], q * dt**2 / 2, atol=1e-12)
    assert torch.allclose(covariance[2:, 2:], q * dt, atol=1e-12)
    assert torch.equal(offset, torch.zeros_like(offset))
    t, z, c = inputs()
    result = oracle.exact_transition(z, dt, c)
    assert torch.allclose(result.mean[:, :2], z[:, :2] + dt * z[:, 2:])
    assert torch.allclose(result.mean[:, 2:], z[:, 2:])


def test_oracle_semigroup_and_off_residual_switch_rechecked_each_call():
    m = model("M2")
    with pytest.raises(ModelContractError):
        AffinePhaseSpaceOracle(m)
    m.acceleration_model.residual_enabled = False
    oracle = AffinePhaseSpaceOracle(m)
    f, c, q = oracle.affine_transition(.7, ModelContext())
    f2, c2, q2 = oracle.affine_transition(1.4, ModelContext())
    assert torch.allclose(f2, f @ f, atol=1e-12)
    assert torch.allclose(c2, f @ c + c, atol=1e-12)
    assert torch.allclose(q2, f @ q @ f.T + q, atol=1e-12)
    m.acceleration_model.residual_enabled = True
    with pytest.raises(ModelContractError):
        oracle.affine_transition(.7, ModelContext())
    for interval in (0., -1., float("nan")):
        with pytest.raises(ModelContractError):
            AffinePhaseSpaceOracle(model("M0")).affine_transition(interval, ModelContext())


def test_legacy_normalized_direct_affine_is_equivalent_in_physical_units():
    legacy = AffineVelocityModel(("known",), "direct", ("vx", "vy", "known"),
        np.array([.2, -.7, 3.]), np.array([1.2, 2.3, .8]),
        np.array([[.1, .2], [-.2, .01], [.03, -.4], [.15, -.3]]),
        np.array([[.5, .1], [.1, .3]]), 100)
    adapted = adapt_legacy_affine(legacy, spec(1))
    t, z, _ = inputs()
    context = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
    expected = legacy.acceleration(z[:, 2:].numpy(), context.numpy())
    assert np.allclose(adapted.acceleration(t, z, ModelContext(context)).detach().numpy(), expected, atol=1e-14)
    factor = adapted.velocity_factor
    assert np.allclose((factor @ factor.T).numpy(), legacy.diffusion_covariance, atol=1e-14)
    transition = AffinePhaseSpaceOracle(adapted).exact_transition(z, 1., ModelContext(context[0]))
    assert torch.isfinite(transition.mean).all()


@pytest.mark.parametrize("family", ["M1-R", "M1-S"])
def test_fixed_basis_recovers_nonlinear_coefficients_and_prediction(family):
    drift = model(family).acceleration_model
    dimensions = len(drift.features)
    train = torch.linspace(-1.8, 1.8, 101, dtype=torch.float64)[:, None]
    if dimensions == 2:
        train = torch.cat((train, .4 * torch.cos(2 * train)), dim=1)
    design = drift.basis(train)
    truth = torch.linspace(-.25, .3, design.shape[1] * 2, dtype=torch.float64).reshape(-1, 2)
    target = design @ truth
    recovered = torch.linalg.lstsq(design, target).solution
    assert torch.linalg.matrix_rank(design) == design.shape[1]
    assert torch.allclose(recovered, truth, atol=1e-10)
    with torch.no_grad():
        drift.coefficients.copy_(recovered)
    selection = train[1::3] + .013
    assert torch.allclose(drift.residual(selection), drift.basis(selection) @ truth, atol=1e-10)
    # Chunking never changes the representation and no N x N matrix is needed.
    assert torch.allclose(torch.cat([drift.residual(part) for part in selection.split(7)]), drift.residual(selection))


def test_neural_fixture_recovers_smooth_acceleration_on_distinct_grid():
    m = model("M2")
    drift = m.acceleration_model
    train = torch.linspace(-1.5, 1.5, 64, dtype=torch.float64)
    features = torch.stack((train, .5 * train.square()), dim=-1)
    target = torch.stack((.15 * torch.sin(train), -.1 * torch.tanh(train)), dim=-1)
    optimizer = torch.optim.Adam(drift.layers.parameters(), lr=.02)
    initial = (drift.residual(features) - target).square().mean().item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = (drift.residual(features) - target).square().mean()
        loss.backward()
        optimizer.step()
    selection = torch.linspace(-1.4, 1.4, 31, dtype=torch.float64)
    predicted = drift.residual(torch.stack((selection, .5 * selection.square()), dim=-1))
    expected = torch.stack((.15 * torch.sin(selection), -.1 * torch.tanh(selection)), dim=-1)
    assert (predicted - expected).square().mean().item() < initial * .05


def test_frozen_package_checks_actual_code_and_cannot_claim_formal_qualification():
    original = model("M0")
    package = freeze_dynamics(original, protocol_hash="d" * 64)
    identity = dynamics_identity()
    assert package["model_card"]["implementation"] == identity
    assert package["model_card"]["code_sha"] == identity["code_sha"]
    restored = load_frozen_dynamics(package, protocol_hash="d" * 64)
    assert torch.equal(original.drift(*inputs()), restored.drift(*inputs()))
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        load_frozen_dynamics(package, protocol_hash="d" * 64, formal=True)
    forged = {**package, "qualification": "qualified", "qualification_hash": "e" * 64,
              "preregistration_hash": "f" * 64}
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        load_frozen_dynamics(forged, protocol_hash="d" * 64, formal=True)
    for key, value in [("output_hash", "e" * 64), ("data_hash", "f" * 64), ("code_hash", "f" * 64)]:
        changed = copy.deepcopy(package)
        changed[key] = value
        with pytest.raises(ResearchError):
            load_frozen_dynamics(changed, protocol_hash="d" * 64)
    with pytest.raises(ResearchError):
        load_frozen_dynamics(package, protocol_hash="e" * 64)


def test_public_reference_exception_is_limited_to_reviewed_adapter_and_name():
    from pathlib import Path
    from scripts.check_public_release import is_approved_public_reference, PUBLIC_PREREGISTRATION_ID
    allowed = Path("application/pirc26_dynamics.py")
    assert is_approved_public_reference("internal work item", allowed, "NEX326")
    assert not is_approved_public_reference("internal work item", allowed, PUBLIC_PREREGISTRATION_ID)
    assert not is_approved_public_reference("internal work item", Path("application/unrelated.py"), "NEX326")
