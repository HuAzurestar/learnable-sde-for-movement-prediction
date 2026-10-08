import itertools

import numpy as np
import pytest

from experiments.pirc17.inference import PRIMARY_FAMILY, paired_blocks
from experiments.pirc17.bootstrap_tail_check import check, exact_quantiles, exact_two_point_pivots, fixture_rows


def test_exact_binomial_pivot_matches_exhaustive_bootstrap():
    values = np.array([0., 0., 0., 1.])
    draws = values[np.array(list(itertools.product(range(4), repeat=4)))]
    means = draws.mean(axis=1)
    errors = draws.std(axis=1, ddof=1)/2
    with np.errstate(divide="ignore"):
        pivots = (means-values.mean())/errors
    probabilities = [.01, .1, .5, .9, .99]
    np.testing.assert_array_equal(exact_quantiles(4, 1, probabilities),
        np.quantile(pivots, probabilities, method="inverted_cdf"))
    exact, masses = exact_two_point_pivots(4, 1)
    assert masses.sum() == pytest.approx(1)
    assert np.all(np.diff(exact) > 0)


def test_fixture_preserves_five_seeds_and_actual_family_pairs():
    rows, estimates, errors = fixture_rows(30, 15, 1.)
    blocks, names, values, counts = paired_blocks(rows)
    assert len(blocks) == 30 and names == list(PRIMARY_FAMILY)
    assert values.shape == (30, 5, 5) and counts == [1]*30
    np.testing.assert_allclose(estimates, [-2, -1, 0, 1, 2], atol=1e-12)
    assert np.all(errors > 0)


def test_diagnostic_is_reproducible_and_does_not_certify_population_coverage():
    first = check(delta_m=1., repetitions=1, cases=((30, 1), (30, 15)))
    assert first == check(delta_m=1., repetitions=1, cases=((30, 1), (30, 15)))
    assert first["certified"] is False and first["final_eval_label_prediction_metric_reads"] == 0
    assert first["cases"][0]["exact_interval_finite"] is False
    assert first["cases"][1]["exact_interval_finite"] is True
    for row in first["cases"][0]["comparisons"].values():
        assert row["finite_endpoint_comparisons"] == 0
        assert row["maximum_finite_endpoint_error_m"] is None
    for case in first["cases"]:
        for row in case["comparisons"].values():
            assert 0 <= row["silent_reference_exceedance_count"] <= row["tail_check_pass_count"] <= 1


@pytest.mark.parametrize("n,k", [(30, 0), (30, 30), (30, 31), (True, 1), (30, 1.5)])
def test_invalid_exact_samples(n, k):
    with pytest.raises(ValueError):
        exact_two_point_pivots(n, k)


def test_invalid_repetition_budget():
    with pytest.raises(ValueError):
        check(delta_m=1, repetitions=0)
