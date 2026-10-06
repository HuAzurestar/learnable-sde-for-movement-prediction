"""Continued numerical computation, not just checkpoint serialization."""

from dataclasses import replace
import json

import pytest

from domain.errors import DataValidationError
from inference.propagation_methods import importance_sampling, mlmc_estimate, monte_carlo
from tests.test_propagation_methods import inputs


def estimator(method, package, request, **kwargs):
    if method == "mlmc":
        return mlmc_estimate(package, request, level_samples=(13, 11, 9), **kwargs)
    if method == "importance":
        return importance_sampling(package, request, proposal=(1.0, 0.0), **kwargs)
    return monte_carlo(package, request, solver=method, **kwargs)


class Stopped(Exception):
    pass


def saved_state(method="euler", *, boundary=1, functional="endpoint-halfspace", threshold=0.5):
    package, request = inputs(samples=33, steps=4, chunk_size=4, functional=functional, threshold=threshold)
    saved = []
    def stop(state, total):
        assert 0 < state["step"] <= total
        json.dumps(state, allow_nan=False)
        saved.append(state)
        if len(saved) == boundary:
            raise Stopped
    with pytest.raises(Stopped):
        estimator(method, package, request, checkpoint=stop)
    return package, request, saved[-1]


@pytest.mark.parametrize("method", ["euler", "additive-heun", "mlmc", "importance"])
@pytest.mark.parametrize("boundary", [1, 4, 9])
def test_completed_chunks_resume_all_statistics_and_match_uninterrupted_result(method, boundary):
    package, request, state = saved_state(method, boundary=boundary)
    expected = estimator(method, package, request)
    observed = []
    actual = estimator(method, package, request, resume_state=json.loads(json.dumps(state)),
        checkpoint=lambda current, total: observed.append(current["step"]))
    assert actual.manifest() == expected.manifest()
    assert all(step > state["step"] for step in observed)
    # The final partial chunk is restored as completed, not regenerated.
    if method != "mlmc" and boundary == 9:
        assert not observed


def test_repeated_resume_across_mlmc_level_transition_keeps_signed_corrections():
    package, request, state = saved_state("mlmc", boundary=4)
    assert state["data_position"] == {"level": 1, "next_sample": 0}
    saved = []
    def stop(current, total):
        saved.append(current)
        raise Stopped
    with pytest.raises(Stopped):
        estimator("mlmc", package, request, resume_state=state, checkpoint=stop)
    actual = estimator("mlmc", package, request, resume_state=saved[0])
    assert actual.manifest() == estimator("mlmc", package, request).manifest()
    assert actual.sample_count == 33


@pytest.mark.parametrize("method", ["euler", "importance", "mlmc"])
def test_zero_hit_states_remain_finite_and_restore_unknown_uncertainty(method):
    package, request, state = saved_state(method, threshold=100)
    actual = estimator(method, package, request, resume_state=state)
    assert actual.manifest() == estimator(method, package, request).manifest()
    assert actual.standard_error is None
    json.dumps(actual.manifest(), allow_nan=False)


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(step=s["step"]+1),
    lambda s: s.update(remaining_seconds=900),
    lambda s: s.update(chunk_complete=False),
    lambda s: s["data_position"].update(next_sample=5),
    lambda s: s["data_position"].update(level=True),
    lambda s: s["rng_state"].update(numpy_version="different"),
    lambda s: s["rng_state"].update(coupling_id="changed-root"),
    lambda s: s["method_state"]["statistics"][0].update(m2=-1),
    lambda s: s["method_state"]["statistics"][0].update(mean=float("nan")),
    lambda s: s["method_state"]["statistics"][0].update(n=True),
])
def test_incompatible_or_unbounded_state_is_refused(mutation):
    package, request, state = saved_state(functional="endpoint-x")
    mutation(state)
    with pytest.raises(DataValidationError):
        estimator("euler", package, request, resume_state=state)


def test_request_chunk_size_and_importance_proposal_are_checkpoint_bindings():
    package, request, state = saved_state("importance")
    with pytest.raises(DataValidationError):
        estimator("importance", package, replace(request, chunk_size=8), resume_state=state)
    with pytest.raises(DataValidationError):
        importance_sampling(package, request, proposal=(0.0, 0.0), resume_state=state)


def test_callback_cannot_mutate_live_estimator_accumulators():
    package, request = inputs(samples=12, steps=4, chunk_size=4)
    def mutate(state, total):
        state["method_state"]["statistics"][0]["mean"] = 1e9
        state["rng_state"]["seed"] = 0
    assert monte_carlo(package, request, checkpoint=mutate) == monte_carlo(package, request)
