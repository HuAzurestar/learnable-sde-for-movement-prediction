from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.nex326.candidate_selection import (
    CandidateSelectionError,
    build_candidate_selection,
)


FIELDS = [
    "arm_id",
    "subconfig_id",
    "slot",
    "reference_arm_id",
    "reference_subconfig_id",
    "comparison_status",
    "replicate_status",
    "replicate_count",
    "replicate_seeds",
    "assessment",
    *[
        f"delta_{metric}_{suffix}"
        for metric in ("energy_score_d2", "hdr90_abs_error_from_90", "cep50_error")
        for suffix in ("mean", "min", "max", "sign_consistency")
    ],
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path):
    table = tmp_path / "nex326_cross_replicate_comparisons.csv"
    base = {
        "slot": "M2",
        "reference_arm_id": "6",
        "reference_subconfig_id": "dt60",
        "comparison_status": "exploratory_point_estimate",
        "replicate_status": "succeeded",
        "replicate_count": "3",
        "replicate_seeds": "101;202;303",
        "assessment": "not_assessed",
    }
    rows = []
    for arm_id, subconfig, values in (
        (6, "dt30", (-2.0, -0.1, -1.0)),
        (3, "single_gaussian", (-3.0, -0.2, 0.5)),
    ):
        row = {**base, "arm_id": arm_id, "subconfig_id": subconfig}
        for metric, value in zip(
            ("energy_score_d2", "hdr90_abs_error_from_90", "cep50_error"), values
        ):
            row[f"delta_{metric}_mean"] = value
            row[f"delta_{metric}_min"] = value - 0.2
            row[f"delta_{metric}_max"] = value + 0.2
            row[f"delta_{metric}_sign_consistency"] = (
                "all_negative" if value + 0.2 < 0 else "all_positive"
            )
        rows.append(row)
    with table.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = tmp_path / "nex326_replicate_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "nex326-tsde-replicate-aggregate-v1",
                "artifacts": {
                    table.name: {"path": table.name, "sha256": _sha(table)}
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "nex326-dsde-multi-seed-receipt-v1",
                "scientific_status": "exploratory_only_not_final_scientific_evidence",
                "assessment": "not_assessed",
                "replicate_seeds": [101, 202, 303],
                "cohort": {"sha256": "c" * 64},
                "integrity": {"replicate_aggregate_summary_sha256": _sha(summary)},
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "pirc20-confirmation-policy-v1",
                "discovery_evidence": {
                    "use": "candidate_selection_only_not_confirmation",
                    "cohort_id": "discovery-v1",
                    "cohort_sha256": "c" * 64,
                    "multi_seed_receipt_sha256": _sha(receipt),
                    "replicate_aggregate_summary_sha256": _sha(summary),
                    "cross_replicate_table_sha256": _sha(table),
                },
                "candidate_selection": {
                    "required_replicate_count": 3,
                    "required_comparison_status": "exploratory_point_estimate",
                    "required_run_status": "succeeded",
                    "required_assessment": "not_assessed",
                    "required_improvement_metrics": [
                        "energy_score_d2",
                        "cep50_error",
                    ],
                    "expected_candidate_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return policy, receipt, summary, table


def test_candidate_selection_applies_all_metric_all_seed_rule(tmp_path):
    policy, receipt, summary, _ = _inputs(tmp_path)
    result = build_candidate_selection(policy, receipt, summary, tmp_path / "out.json")
    assert [(row["arm_id"], row["subconfig_id"]) for row in result["candidates"]] == [
        (6, "dt30")
    ]
    assert result["candidates"][0]["reference_subconfig_id"] == "dt60"
    assert result["scientific_status"] == "exploratory_candidate_selection_not_a_verdict"


def test_candidate_selection_rejects_tampered_table(tmp_path):
    policy, receipt, summary, table = _inputs(tmp_path)
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CandidateSelectionError, match="hash chain"):
        build_candidate_selection(policy, receipt, summary, tmp_path / "out.json")


def test_checked_in_candidate_selection_is_policy_and_audit_bound():
    root = Path(__file__).parents[1] / "experiments" / "nex326"
    policy = root / "pirc20_confirmation_policy.json"
    selection = root / "pirc20_candidate_selection.json"
    audit = json.loads((root / "pirc20_runtime_audit.json").read_text(encoding="utf-8"))
    payload = json.loads(selection.read_text(encoding="utf-8"))

    assert payload["policy"]["sha256"] == _sha(policy)
    assert [(item["arm_id"], item["subconfig_id"]) for item in payload["candidates"]] == [
        (6, "dt30"),
        (9, "mixed"),
        (9, "pure_es"),
    ]
    assert audit["future_confirmation"]["candidate_policy"]["sha256"] == _sha(policy)
    assert audit["future_confirmation"]["candidate_selection"]["sha256"] == _sha(
        selection
    )
    assert all(not value for value in payload["privacy"].values())
