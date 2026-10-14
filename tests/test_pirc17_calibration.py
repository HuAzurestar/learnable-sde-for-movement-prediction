import numpy as np
import pytest

from experiments.pirc17.calibration import observed_indices


def test_only_actual_timestamps_are_selected_and_ties_prefer_earlier():
    times=np.array([59.,61.,298.,901.,1799.])
    indexes=observed_indices(times)
    np.testing.assert_array_equal(times[indexes],[59,298,901,1799])


def test_sparse_targets_and_duplicate_reuse_are_rejected():
    with pytest.raises(ValueError,match="distinct observed"):
        observed_indices([20,100,200,300])
    with pytest.raises(ValueError,match="distinct observed"):
        observed_indices([100],[90,110])


@pytest.mark.parametrize("times",[[0,60,300,900,1800],[60,60,300,900,1800],[60,np.nan],[60,300,200,1800]])
def test_bad_time_metadata_never_becomes_interpolated_truth(times):
    with pytest.raises(ValueError):
        observed_indices(times)
