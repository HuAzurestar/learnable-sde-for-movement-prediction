from __future__ import annotations

import numpy as np
import pytest

from experiments.nex326.confirmation_verdict import (
    BlockTotals,
    ConfirmationVerdictError,
    _mechanism_gates_passed,
    block_totals,
    bootstrap_comparison,
    holm_adjust,
)
from experiments.nex326.runner import Prediction


def _prediction(segment_id: str) -> Prediction:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    return Prediction(
        segment_id=segment_id,
        target=np.asarray([0.5, 0.0]),
        samples=np.column_stack((np.cos(angles), np.sin(angles))),
        prior_mean=None,
        prior_source=None,
        unbridged_prior_distance=None,
        bridged_prior_distance=None,
        bridge_path=None,
        bridge_diagnostics=None,
    )


def test_block_totals_reject_unknown_or_duplicate_segments():
    prediction = _prediction("segment-1")
    with pytest.raises(ConfirmationVerdictError, match="duplicate or unknown"):
        block_totals((prediction,), {})
    with pytest.raises(ConfirmationVerdictError, match="duplicate or unknown"):
        block_totals(
            (prediction, prediction), {"segment-1": "block-1"}
        )


def test_block_bootstrap_resamples_blocks_and_preserves_seed_pairing():
    candidate = {}
    reference = {}
    for seed in (11, 22, 33):
        candidate[seed] = {
            block: BlockTotals(10, 10.0, 9.0, 10.0)
            for block in ("a", "b", "c", "d")
        }
        reference[seed] = {
            block: BlockTotals(10, 20.0, 8.0, 20.0)
            for block in ("a", "b", "c", "d")
        }
    result = bootstrap_comparison(
        candidate, reference, iterations=200, seed=123
    )
    assert result["independent_block_count"] == 4
    assert result["metrics"]["energy_score_d2"]["mean_delta"] == pytest.approx(-1.0)
    assert result["metrics"]["hdr90_abs_error_from_90"]["mean_delta"] == pytest.approx(
        -0.1
    )
    assert result["metrics"]["cep50_error"]["one_sided_95pct_upper"] == pytest.approx(
        -1.0
    )
    assert result["primary_one_sided_p"] == pytest.approx(1 / 201)


def test_holm_adjustment_is_monotone_in_sorted_order():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(ConfirmationVerdictError, match="p-values"):
        holm_adjust([])


def test_mechanism_gate_reader_requires_registered_gate_list():
    assert _mechanism_gates_passed(
        {"mechanism_gates": [{"passed": True}, {"passed": True}]}
    )
    assert not _mechanism_gates_passed(
        {"mechanism_gates": [{"passed": True}, {"passed": False}]}
    )
    with pytest.raises(ConfirmationVerdictError, match="no registered"):
        _mechanism_gates_passed({"mechanism_gates": []})
    with pytest.raises(ConfirmationVerdictError, match="malformed"):
        _mechanism_gates_passed({"mechanism_gates": [{"passed": True}, "invalid"]})
