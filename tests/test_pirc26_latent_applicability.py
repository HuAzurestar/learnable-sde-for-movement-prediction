"""Current-contract refusals and noise-support counterexamples, not an ELBO."""

import subprocess
import sys

import pytest
import torch

from application.pirc26_components import latent_applicability
from domain import ModelContext
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc26_dynamics import model


@pytest.mark.parametrize("objective", ["L1", "L2"])
def test_report_is_scoped_negative_evidence_not_a_production_need_decision(objective):
    report = latent_applicability(objective)
    payload = dict(report)
    assert payload.pop("report_hash") == digest(payload)
    assert report["status"] == "INAPPLICABLE" and report["metric_value"] is None
    assert report["production_latent_need"] == "NOT_ASSESSED"
    assert report["qualification"] == "unqualified"
    assert report["scope"] == "current-registered-observed-contract-only"
    assert report["state_order"] == ["x", "y", "vx", "vy"]
    assert "OBSERVATION_LIKELIHOOD_UNREGISTERED" in report["reason_codes"]
    assert ("FULL_STATE_DIFFUSION_INVERSE_UNAVAILABLE" in report["reason_codes"]) == (objective == "L2")
    report["reason_codes"].clear()
    assert latent_applicability(objective)["reason_codes"]  # caller mutation cannot change the contract


def test_report_does_not_relabel_observed_or_unknown_objectives():
    for objective in ("O1", "O2", "L3"):
        with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
            latent_applicability(objective)


@pytest.mark.parametrize("objective", ["L1", "L2"])
def test_all_actual_registration_and_factory_routes_refuse_before_numerical_import(objective):
    program = r'''
import importlib.abc
import sys
class NoNumericalEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy'}:
            raise AssertionError('numerical allocation before refusal: ' + fullname)
sys.meta_path.insert(0, NoNumericalEngine())
from application import pirc26_components as c
from application.pirc26_runtime import execution_plugin, recovery_plugin
from infrastructure.research_store import ResearchError, ResearchStore
def no_store(*args, **kwargs):
    raise AssertionError('latent refusal must not construct a store')
ResearchStore.__init__ = no_store
objective = sys.argv[1]
calls = [lambda: c.plan_schema(objective, 'M2'), lambda: c.config_schema(objective, 'M2'),
    lambda: c.composition_contract(objective, 'M2'), lambda: c.component_registries(objective, 'M2'),
    lambda: execution_plugin(objective, 'M2'), lambda: recovery_plugin(objective, 'M2'),
    lambda: c.model_factory({'objective': objective}, {}, {}),
    lambda: c.trainer_factory({'objective': objective}, {}, None),
    lambda: c.predictor_factory({'objective': objective}, {}, None)]
for role in ('model', 'trainer', 'predictor'):
    calls.append(lambda role=role: c.entry(role, objective, 'M2'))
for call in calls:
    try:
        call()
    except ResearchError as exc:
        assert exc.code == 'INAPPLICABLE', str(exc)
    else:
        raise AssertionError('unqualified latent route admitted')
assert not {'torch', 'numpy', 'scipy'}.intersection(sys.modules)
print('latent-refusal-ok')
'''
    result = subprocess.run([sys.executable, "-B", "-c", program, objective],
        capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "latent-refusal-ok"


@pytest.mark.parametrize("family", ["M0", "M1-S", "M1-R", "M2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_models_refuse_latent_without_mutation_or_fake_position_noise(family, dtype):
    m = model(family, dtype)
    before = m.checkpoint()
    for objective in ("L1", "L2"):
        support = m.supports(objective, "generic-rollout", "restart-only")
        assert not support["supported"] and support["reason"].startswith("INAPPLICABLE:")
    assert m.checkpoint() == before
    assert m.supports("O1", "generic-rollout", "restart-only")["supported"]


def test_degenerate_noise_allows_velocity_control_but_cannot_project_position_drift():
    m = model("M2")
    t, z = torch.zeros(1, dtype=torch.float64), torch.tensor([[0., 0., 1., 2.]], dtype=torch.float64)
    g = m.diffusion(t, z, ModelContext())[0]
    assert g.shape == (4, 2) and torch.linalg.matrix_rank(g) == 2
    delta = torch.tensor([0., 0., .3, -.2], dtype=z.dtype)
    control = torch.linalg.solve(m.velocity_factor, delta[2:])
    assert torch.allclose(g @ control, delta, atol=1e-15)
    invalid = delta.clone()
    invalid[0] = .2
    # Keeping only the velocity component does not satisfy the full equation.
    assert torch.allclose((g @ control - invalid).norm(), z.new_tensor(.2))
    assert torch.count_nonzero(g[:2]) == 0


def test_unconstrained_invertible_posterior_flow_need_not_preserve_kinematics():
    m = model("M2")
    t, z = torch.zeros(1, dtype=torch.float64), torch.tensor([[0., 0., 1., 2.]], dtype=torch.float64)
    g = m.diffusion(t, z, ModelContext())[0]
    # F(epsilon,t)=epsilon is smooth/invertible and has Gaussian score -z.
    # With constant G its posterior drift is 0 + .5 G G^T score.
    posterior = .5 * (g @ g.T) @ (-z[0])
    prior = m.drift(t, z, ModelContext())[0]
    assert torch.equal(posterior[:2], torch.zeros(2, dtype=z.dtype))
    assert torch.equal(prior[:2], z[0, 2:])
    assert not torch.equal(prior[:2], posterior[:2])
    # This counterexample is not a path-law or latent-model qualification.
