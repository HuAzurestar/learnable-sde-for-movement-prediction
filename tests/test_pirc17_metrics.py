import numpy as np
import pytest

from experiments.pirc17.metrics import EntropyGrid, energy_score, marginal_crps, radial_regions, score_path


def test_energy_exact_discrete_oracle_and_chunk_invariance():
    x = np.array([[0.,0.],[2.,0.],[4.,0.]])
    # E|X-1|=5/3; half of ordered off-diagonal distance mean=4/3.
    assert energy_score(x,[1,0],chunk_size=1) == pytest.approx(1/3)
    assert energy_score(x,[1,0],chunk_size=99) == pytest.approx(1/3)
    np.testing.assert_allclose(marginal_crps(x,[1,0]),[1/3,0])
    assert energy_score(np.tile([3.,4.],(8,1)),[0,0]) == pytest.approx(5)


def test_energy_rotation_translation_and_scale():
    x = np.array([[0.,0.],[1.,3.],[2.,4.]])
    y = np.array([2.,1.])
    rotation = np.array([[0.,-1.],[1.,0.]])
    assert energy_score(x@rotation+7,y@rotation+7) == pytest.approx(energy_score(x,y))
    assert energy_score(x*2,y*2) == pytest.approx(2*energy_score(x,y))


def test_radial_disk_is_not_labelled_hdr_and_keeps_quantile_mass():
    x = np.array([[1.,0.],[-1.,0.],[0.,2.],[0.,-2.]])
    result = radial_regions(x,[1.5,0],levels=[.5,.95])
    assert result["region"] == "ensemble_mean_radial_quantile_disk"
    half, tail = result["levels"]
    assert not half["covered"] and half["area_m2"] == pytest.approx(np.pi)
    assert tail["covered"] and tail["empirical_mass"] >= .95


def test_entropy_has_explicit_overflow_mass_and_fixed_grid_identity():
    grid = EntropyGrid((-1,0,1),(-1,0,1))
    result = grid.entropy([[-.5,-.5],[.5,.5],[20,20],[21,21]])
    assert result["total_mass"] == 1 and result["overflow_mass"] == .5
    assert result["entropy_nats"] == pytest.approx(-2*.25*np.log(.25)-.5*np.log(.5))
    assert result["grid_identity"] != EntropyGrid((-2,0,2),(-2,0,2)).identity
    # Reshuffling/duplicating the exact empirical measure does not change entropy.
    assert grid.entropy([[.5,.5],[-.5,-.5]])["entropy_nats"] == pytest.approx(np.log(2))
    assert grid.entropy([[1,1]])["overflow_mass"] == 0  # histogram rightmost boundary


def test_score_path_uses_mean_prediction_not_oracle_particle():
    x = np.array([[[0,0],[0,0]],[[2,0],[4,0]]],dtype=float)
    y = np.zeros((2,2))
    result = score_path(x,y,[1,2],time_weights=[.25,.75],entropy_grid=EntropyGrid((-10,0,10),(-10,0,10)))
    assert result["ade_grid_mean_m"] == 1.5 and result["fde_m"] == 2
    assert result["time_weighted_displacement_error_m"] == 1.75
    assert result["nll"]["status"] == "unavailable"
    assert result["joint_path_entropy"]["status"] == "unavailable"
    assert result["by_time"][1]["elapsed_seconds"] == 2


def test_joint_path_score_detects_dependency_with_same_marginals():
    same = np.array([[[0,0],[0,0]],[[2,0],[2,0]]],float)
    opposite = same.copy()
    opposite[:,1] = opposite[::-1,1]
    kw = dict(time_weights=[.5,.5],entropy_grid=EntropyGrid((-5,0,5),(-5,0,5)))
    a = score_path(same,np.zeros((2,2)),[1,2],**kw)
    b = score_path(opposite,np.zeros((2,2)),[1,2],**kw)
    assert a["time_weighted_energy_score_m"] == b["time_weighted_energy_score_m"]
    assert a["path_energy_score_m"] != b["path_energy_score_m"]


@pytest.mark.parametrize("bad", [np.nan,np.inf])
def test_nonfinite_prediction_is_never_silently_dropped(bad):
    with pytest.raises(ValueError):
        energy_score([[0,0],[bad,0]],[0,0])
