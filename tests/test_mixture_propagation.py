"""Bounded mixture engineering checks, not research qualification."""

from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from domain.errors import DataValidationError, NumericalError
from domain.mixture import MixturePolicy
from experiments.pirc25.affine import code_hash
from inference.mixture_propagation import component, merge, mixture_estimate, mixture_step, strict_root
from inference.propagation_methods import discrete_moments
from tests.test_propagation_methods import inputs
from tests.test_propagation_recovery import Stopped


def setup(**changes):
    package, request = inputs(steps=4)
    policy = MixturePolicy(request.request_hash, package.package_hash, code_hash(),
        1, 0., 0., (1., 1., 1., 1.), 0., 1_000_000, 60.)
    return package, request, replace(policy, **changes)


def test_merge_preserves_absolute_weight_and_first_second_moments_not_density():
    a = component(.2, (-2., 0., 0., 0.), np.eye(4), "a"*64)
    b = component(.3, (3., 0., 0., 0.), 2*np.eye(4), "b"*64)
    c = merge(a, b, policy_hash="c"*64, step=0)
    assert c.weight == .5
    np.testing.assert_allclose(c.mean, (1., 0., 0., 0.), atol=1e-15)
    target = (.2*(np.array(a.covariance)+np.outer(a.mean, a.mean))
        + .3*(np.array(b.covariance)+np.outer(b.mean, b.mean)))/.5
    np.testing.assert_allclose(np.array(c.covariance)+np.outer(c.mean, c.mean), target, atol=1e-14)
    assert c.lineage_id not in (a.lineage_id, b.lineage_id)


def test_one_component_affine_limit_matches_euler_moments_without_claiming_law_exactness():
    package, request, policy = setup()
    actual = mixture_estimate(package, request, policy)
    mean, covariance = discrete_moments(package, request)
    diagnostics = dict(actual.diagnostics)
    np.testing.assert_allclose(diagnostics["mean"], mean, atol=1e-14, rtol=0)
    np.testing.assert_allclose(diagnostics["covariance"], covariance, atol=1e-14, rtol=0)
    assert actual.status == "APPROXIMATION_ONLY"
    assert actual.error_budget.propagation_approximation.value is None
    assert actual.error_budget.time_discretization.value is None
    assert actual.error_budget.model.value is None
    assert actual.sample_count == 0 and actual.interval is None
    assert diagnostics["retained_mass"] == 1 and diagnostics["discarded_mass"] == 0
    assert diagnostics["covariance_projection"] is False
    counts = diagnostics["reduction_counts"]
    assert counts["forced_merges"]+counts["threshold_merges"] == 7*request.steps


def test_checkpoint_replay_is_exact_and_bounded_with_environment_and_policy_bindings():
    package, request, policy = setup()
    saved = []
    def stop(state, total):
        assert total == request.steps*policy.work_per_step
        assert len(json.dumps(state).encode()) < 65536
        saved.append(state)
        raise Stopped
    with pytest.raises(Stopped):
        mixture_estimate(package, request, policy, checkpoint=stop)
    actual = mixture_estimate(package, request, policy, resume_state=saved[0])
    assert actual.manifest() == mixture_estimate(package, request, policy).manifest()
    for field in ("policy_hash", "request_hash", "model_package_hash"):
        bad = copy.deepcopy(saved[0])
        bad["method_state"][field] = "f"*64
        with pytest.raises(DataValidationError):
            mixture_estimate(package, request, policy, resume_state=bad)
    bad = copy.deepcopy(saved[0])
    bad["rng_state"]["numpy_version"] = "other"
    with pytest.raises(DataValidationError):
        mixture_estimate(package, request, policy, resume_state=bad)


@pytest.mark.parametrize("changes", [{"component_cap": 0}, {"component_cap": 33},
    {"component_cap": True}, {"merge_distance": -1}, {"prune_weight": 1},
    {"state_scales": (1., 0., 1., 1.)}, {"maximum_work_units": 1},
    {"maximum_discarded_mass": 1}, {"maximum_job_seconds": 7201}, {"code_hash": "0"*64}])
def test_invalid_policy_refused(changes):
    package, request, policy = setup(**changes)
    with pytest.raises(DataValidationError):
        mixture_estimate(package, request, policy)


def test_policy_manifest_is_complete_and_detached():
    package, request, policy = setup()
    manifest = policy.manifest()
    restored = MixturePolicy.from_manifest(manifest)
    manifest["state_scales"][0] = 999
    assert restored.state_scales == (1.,)*4
    restored.validate(package, request, code_hash())
    incomplete = restored.manifest()
    del incomplete["maximum_work_units"]
    with pytest.raises(DataValidationError):
        MixturePolicy.from_manifest(incomplete)


def test_no_psd_projection_or_silent_all_pruned_fallback():
    with pytest.raises(NumericalError, match="projection"):
        strict_root(np.diag([1., 1., 1., -1e-16]))
    package, request, policy = setup(prune_weight=.2, maximum_discarded_mass=.9)
    with pytest.raises(NumericalError, match="all mixture mass pruned"):
        mixture_estimate(package, request, policy)


def test_pruning_records_absolute_mass_and_component_candidate_caps():
    _, _, policy = setup(component_cap=4, prune_weight=.03, maximum_discarded_mass=.3)
    parents = (component(.8, (-2., 0., 0., 0.), np.zeros((4, 4)), "a"*64),
        component(.2, (2., 0., 0., 0.), np.zeros((4, 4)), "b"*64))
    parameters = np.zeros((4, 4)), np.zeros(4), np.zeros((4, 2)), (0., 0.), (1., 1.)
    reduced, counts, removed = mixture_step(parents, parameters, .1, policy, 0)
    assert removed == .2
    assert sum(c.weight for c in reduced) == pytest.approx(.8)
    assert len(reduced) <= policy.component_cap
    assert counts["generated"] == 16 <= 8*policy.component_cap
    assert counts["pruned"] == 8 and counts["threshold_merges"] == 7


def test_one_step_nonlinear_retains_distinct_centres_not_just_a_gaussian():
    from experiments.pirc27.nonlinear import nonlinear_package
    package = nonlinear_package()
    _, request = inputs(steps=1, functional="endpoint-halfspace", threshold=.5,
        initial_mean=(0.,)*4,
        initial_covariance=((.25, 0., 0., 0.), (0.,)*4, (0.,)*4, (0.,)*4))
    request = replace(request, model_package_hash=package.package_hash)
    policy = MixturePolicy(request.request_hash, package.package_hash, code_hash(),
        8, 0., 0., (1.,)*4, 0., 1_000_000, 60.)
    actual = mixture_estimate(package, request, policy)
    diagnostics = dict(actual.diagnostics)
    assert len(diagnostics["components"]) == 3
    assert sorted(c["mean"][0] for c in diagnostics["components"]) == [-1., 0., 1.]
    assert actual.estimate == .125  # Existing halfspace is normal @ state >= threshold.
    assert actual.error_budget.reference.value is None
    assert actual.error_budget.propagation_approximation.value is None


@pytest.mark.parametrize("field,value", [("step", -1), ("step", True),
    ("chunk_complete", False), ("data_position", {"next_grid_step": 999})])
def test_bad_checkpoint_boundary_is_refused(field, value):
    package, request, policy = setup()
    saved = []
    def capture(state, total):
        saved.append(state)
        raise Stopped
    with pytest.raises(Stopped):
        mixture_estimate(package, request, policy, checkpoint=capture)
    bad = copy.deepcopy(saved[0])
    bad[field] = value
    with pytest.raises(DataValidationError):
        mixture_estimate(package, request, policy, resume_state=bad)


def test_pruning_cap_failure_does_not_return_renormalized_success():
    from experiments.pirc27.nonlinear import nonlinear_package
    package = nonlinear_package()
    _, request = inputs(steps=2, initial_mean=(0.,)*4,
        initial_covariance=((.25, 0., 0., 0.), (0.,)*4, (0.,)*4, (0.,)*4))
    request = replace(request, model_package_hash=package.package_hash)
    policy = MixturePolicy(request.request_hash, package.package_hash, code_hash(),
        8, 0., .02, (1.,)*4, .1, 1_000_000, 60.)
    with pytest.raises(NumericalError, match="pruning cap"):
        mixture_estimate(package, request, policy)
