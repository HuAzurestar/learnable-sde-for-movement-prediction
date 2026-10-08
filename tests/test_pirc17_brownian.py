import numpy as np
import pytest

from experiments.pirc17.brownian import BrownianPath, integration_grid
from experiments.pirc17.origins import known_velocity
from experiments.pirc17.rollout import rollout


def test_increments_add_and_particle_prefixes_are_shared():
    path=BrownianPath([1,2],[1,.5],history_step_seconds=5,particles=10000,seed=7,stream_id="fixture-a")
    np.testing.assert_allclose(path(0,2,10000,2),path(0,1,10000,2)+path(1,2,10000,2),atol=1e-14)
    np.testing.assert_array_equal(path(0,1,3,2),path(0,1,10000,2)[:3])
    np.testing.assert_allclose(path(0,2,10000,2).var(axis=0),[2,2],rtol=.04)
    replay=BrownianPath([1,2],[1,.5],history_step_seconds=5,particles=10000,seed=7,stream_id="fixture-a")
    assert path.identity==replay.identity


def test_constant_within_physical_history_tick_is_pathwise_step_invariant():
    origin=known_velocity([0,0],0,[1,0],source="analytic-software-fixture")
    horizons=[1.7,5.3,10.2]
    path=BrownianPath(horizons,[1.,.5,.25],history_step_seconds=5.,particles=32,seed=12,stream_id="fixture-a")
    predictions=[rollout(origin,horizons,particles=32,seed=1,max_step_seconds=step,
        base_drift=lambda s:.8*s.velocities_mps,diffusion=lambda s:np.broadcast_to(np.eye(2),(32,2,2)),
        brownian_increments=path).positions_m for step in (1.,.5,.25)]
    for prediction in predictions[1:]:
        np.testing.assert_allclose(prediction,predictions[0],rtol=0,atol=1e-12)


def test_grid_and_dimension_mismatches_fail_closed():
    path=BrownianPath([2],[1],history_step_seconds=5,particles=3,seed=1,stream_id="fixture-a")
    with pytest.raises(ValueError,match="grid"):
        path(0,.1,3,2)
    with pytest.raises(ValueError,match="dimensions"):
        path(0,1,4,2)
    with pytest.raises(ValueError,match="positive"):
        integration_grid([1],0,5)
    with pytest.raises(ValueError,match="budget"):
        integration_grid([100],.00001,5)


def test_origin_streams_differ_and_particle_budget_extension_preserves_paths():
    def driver(origin,particles):
        return BrownianPath([2],[1,.5],history_step_seconds=5,particles=particles,seed=7,stream_id=origin)
    small=driver("origin-a",8)
    large=driver("origin-a",32)
    other=driver("origin-b",8)
    np.testing.assert_array_equal(small.values,large.values[:,:8])
    assert small.identity["stream_id_sha256"]!=other.identity["stream_id_sha256"]
    assert not np.array_equal(small.values,other.values)
    assert small.identity==driver("origin-a",8).identity


@pytest.mark.parametrize("stream_id",[None,""," ",3])
def test_invalid_origin_namespace_rejected(stream_id):
    with pytest.raises(ValueError,match="stream_id"):
        BrownianPath([1],[1],history_step_seconds=5,particles=2,seed=1,stream_id=stream_id)
