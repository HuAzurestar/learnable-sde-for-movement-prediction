"""Frozen CSV equality includes numeric eligibility without weakening imports."""

from copy import deepcopy
import csv
import hashlib
import io
import json

import pytest

from application.research_evidence import accept_evidence_package, expected_metrics_csv
from infrastructure.research_store import ResearchError, encode
from tests.path_paper_helpers import independent_aggregate
from tests.test_research_comparison import comparison_source


def aggregate(*, metrics=True, managed=False):
    arm = {"arm_id": "original-arm", "stratum_id": "stratum", "comparison_dimensions": {"horizon": 1.},
        "independent_n": 1 if metrics else 0, "expected_cells": 2, "successful_cells": 2,
        "comparison_eligible_cells": 2 if metrics else 0,
        "path_output_dispositions": {"PASSED": 2} if metrics else {"FAILED": 1, "UNRESOLVED": 1},
        "status": "complete" if metrics else "incomplete", "metrics": {"error": 1.25} if metrics else {},
        "metric_units": {"error": "m"} if metrics else {},
        "cost": {"charged_ms": 10, "reserved_ms": 0, "measured_ms": 10, "unit": "slot-ms", "scope": "original-arm"},
        "status_rates": {"denominator": 2, "values": {"SUCCEEDED": 1.}}}
    value = {"aggregate_hash": "frozen-hash", "arms": [arm],
        "comparison_eligible_cell_count": arm["comparison_eligible_cells"],
        "path_output_dispositions": arm["path_output_dispositions"]}
    if managed:
        value.update(adjudication={"records": []}, computation_ref={"manifest_id": "frozen-receipt"})
    return value


@pytest.mark.parametrize("metrics", [False, True])
@pytest.mark.parametrize("managed", [False, True])
def test_current_csv_matches_actual_independent_paper_bytes(metrics, managed):
    value = aggregate(metrics=metrics, managed=managed)
    actual = expected_metrics_csv(value)
    assert actual == independent_aggregate().csv_bytes(value)
    row = next(csv.DictReader(io.StringIO(actual.decode())))
    assert int(row["comparison_eligible_cells"]) == value["arms"][0]["comparison_eligible_cells"]
    assert json.loads(row["path_output_dispositions"]) == value["arms"][0]["path_output_dispositions"]


def test_legacy_csv_does_not_invent_eligibility_for_old_aggregates():
    value = aggregate()
    for key in ("comparison_eligible_cell_count", "path_output_dispositions"):
        del value[key]
    for key in ("comparison_eligible_cells", "path_output_dispositions"):
        del value["arms"][0][key]
    row = next(csv.DictReader(io.StringIO(expected_metrics_csv(value).decode())))
    assert "comparison_eligible_cells" not in row and "path_output_dispositions" not in row
    assert row["value"] == "1.25" and row["successful_cells"] == "2"


@pytest.mark.parametrize("case", ["count", "dispositions", "all-arm-fields", "mixed-arms"])
def test_partial_eligibility_schema_has_no_legacy_fallback(case):
    value = aggregate()
    if case == "mixed-arms":
        value["arms"].append(deepcopy(value["arms"][0]))
    arm = value["arms"][-1]
    for key in ("comparison_eligible_cells", "path_output_dispositions"):
        if case in {"all-arm-fields", "mixed-arms"} or key == ("comparison_eligible_cells" if case == "count" else "path_output_dispositions"):
            del arm[key]
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH.*eligibility"):
        expected_metrics_csv(value)


@pytest.mark.parametrize("column", ["comparison_eligible_cells", "path_output_dispositions"])
def test_resealed_real_package_cannot_change_csv_eligibility(tmp_path, column):
    store, _, _, _, _ = comparison_source(tmp_path)
    root = tmp_path/"source-evidence"
    original = json.loads((root/"aggregate.json").read_bytes())
    rows = list(csv.DictReader(io.StringIO((root/"metrics.csv").read_text(encoding="utf-8"))))
    rows[0][column] = "999" if column == "comparison_eligible_cells" else '{"PASSED":999}'
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    table = stream.getvalue().encode()
    index = json.loads((root/"PaperEvidenceIndex.json").read_bytes())
    index["table_sha256"] = hashlib.sha256(table).hexdigest()
    index_bytes = encode(index)
    manifest = json.loads((root/"manifest.json").read_bytes())
    manifest["files"]["metrics.csv"] = hashlib.sha256(table).hexdigest()
    manifest["files"]["PaperEvidenceIndex.json"] = hashlib.sha256(index_bytes).hexdigest()
    (root/"metrics.csv").write_bytes(table)
    (root/"PaperEvidenceIndex.json").write_bytes(index_bytes)
    (root/"manifest.json").write_bytes(encode(manifest))
    before = store.events()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH.*frozen CSV"):
        accept_evidence_package(store, root, original["aggregate_hash"])
    assert store.events() == before
