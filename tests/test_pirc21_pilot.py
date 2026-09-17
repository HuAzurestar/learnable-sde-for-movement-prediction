from __future__ import annotations

import pytest

from experiments.nex326.pirc21_pilot import (
    PILOT_CONFIGURATION_IDS,
    PILOT_VARIANTS_BY_FACTOR,
    PIRC21PilotError,
    build_pilot_receipt,
)


def _record(result_id: str, *, status: str = "succeeded") -> dict[str, object]:
    return {
        "run_id": f"run-{result_id}",
        "result_id": result_id * 64,
        "run_status": status,
        "metrics": {"energy_score_d2": 1.0},
    }


def test_pilot_receipt_covers_every_factor_and_both_configurations():
    records = {
        "no_pirc21_features": _record("a"),
        "all_registered_factors": _record("b"),
    }
    receipt = build_pilot_receipt(
        dataset_id="pirc20-fixture",
        snapshot_id="snapshot-fixture",
        runtime_identity_sha256="c" * 64,
        transition_count=12,
        records=records,
        cohort_fingerprint="d" * 64,
        maximum_segments_per_role=2,
    )

    assert len(PILOT_VARIANTS_BY_FACTOR) == 13
    assert set(receipt["selected_factor_ids"]) == set(PILOT_VARIANTS_BY_FACTOR)
    assert tuple(receipt["configuration_ids"]) == PILOT_CONFIGURATION_IDS
    assert receipt["paper_conclusions_ready"] is False
    assert len(receipt["receipt_identity_sha256"]) == 64


def test_pilot_receipt_rejects_failed_or_incomplete_runs():
    with pytest.raises(PIRC21PilotError, match="both registered"):
        build_pilot_receipt(
            dataset_id="dataset",
            snapshot_id="snapshot",
            runtime_identity_sha256="c" * 64,
            transition_count=1,
            records={"no_pirc21_features": _record("a")},
            cohort_fingerprint="d" * 64,
            maximum_segments_per_role=1,
        )
    with pytest.raises(PIRC21PilotError, match="complete successfully"):
        build_pilot_receipt(
            dataset_id="dataset",
            snapshot_id="snapshot",
            runtime_identity_sha256="c" * 64,
            transition_count=1,
            records={
                "no_pirc21_features": _record("a"),
                "all_registered_factors": _record("b", status="data_unavailable"),
            },
            cohort_fingerprint="d" * 64,
            maximum_segments_per_role=1,
        )
