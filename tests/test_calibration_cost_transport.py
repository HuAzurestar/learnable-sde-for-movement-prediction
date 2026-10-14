"""Shared preparation stays one frozen cost, never per-arm cost or science n."""

from copy import deepcopy
import csv
import hashlib
import io
import json

import pytest

from application.research_evidence import (
    accept_aggregate, accept_evidence_package, expected_metrics_csv, expected_paper_index,
)
from infrastructure.research_store import ResearchError, digest, encode
from tests.path_paper_helpers import independent_aggregate
from tests.test_calibration_export import complete_export, consumers, outcomes, outcome_export
from tests.test_research_csv_eligibility import aggregate as display_aggregate


FIELDS = ("calibration_cost", "calibration_evidence_hash", "calibration_status_counts")


def display_value(*, metrics=True, managed=False, known=True):
    """Serialization-only controls; these objects are not admitted source proofs."""
    value = display_aggregate(metrics=metrics, managed=managed)
    second = deepcopy(value["arms"][0])
    second.update(arm_id="second-arm", stratum_id="second-stratum")
    value["arms"].append(second)
    value.update(study_id="display-only", code_hash=digest("display-code"),
        disclosure_scope="synthetic", cell_dispositions=[],
        calibration_cost={"unit": "slot-ms", "scope": "unique-calibration-source-attempts-including-failures-not-per-target-cell",
            "unique_attempts": 2, "charged_ms": 23 if known else None,
            "reserved_ms": 0 if known else None, "measured_ms": 23 if known else None,
            "sources": [{"attempt_id": "original-failed", "charged_ms": 11},
                        {"attempt_id": "original-other", "charged_ms": 12 if known else None}]},
        calibration_evidence_hash=digest("recorded-preparation-only"),
        calibration_status_counts={"CALIBRATED": 1, "FAILED": 1, "UNAVAILABLE": 1})
    return value


@pytest.mark.parametrize("metrics", [False, True])
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("known", [False, True])
def test_actual_independent_csv_and_index_keep_one_shared_cost(metrics, managed, known):
    value = display_value(metrics=metrics, managed=managed, known=known)
    paper = independent_aggregate()
    table = expected_metrics_csv(value)
    assert table == paper.csv_bytes(value)
    rows = list(csv.DictReader(io.StringIO(table.decode())))
    shared = [row for row in rows if row.get("row_kind") == "shared-calibration-preparation"]
    assert len(shared) == 1
    row = shared[0]
    assert row["arm_id"] == row["independent_n"] == row["metric"] == row["value"] == ""
    assert json.loads(row["calibration_cost"]) == value["calibration_cost"]
    assert row["calibration_evidence_hash"] == value["calibration_evidence_hash"]
    assert json.loads(row["calibration_status_counts"]) == value["calibration_status_counts"]
    assert row["charged_ms"] == ("23" if known else "")
    method_rows = [row for row in rows if row["row_kind"] == "method-metric"]
    assert len(method_rows) == 2
    assert all(all(row[name] == "" for name in FIELDS) for row in method_rows)
    assert all(row["charged_ms"] == "10" for row in method_rows)
    index = expected_paper_index(value, table)
    assert index == paper.evidence_index(value, table)
    assert all(index[name] == value[name] for name in FIELDS)
    assert all(not any(name in claim for name in FIELDS) for claim in index["claims"])


@pytest.mark.parametrize("missing", FIELDS)
def test_partial_shared_cost_cannot_fall_back_to_legacy_serialization(missing):
    value = display_value()
    del value[missing]
    paper = independent_aggregate()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH.*calibration"):
        expected_metrics_csv(value)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH.*calibration"):
        expected_paper_index(value, b"not-published")
    with pytest.raises(ValueError, match="calibration"):
        paper.csv_bytes(value)
    with pytest.raises(ValueError, match="calibration"):
        paper.evidence_index(value, b"not-published")


@pytest.mark.parametrize("field", [*FIELDS, "all-fields", "bool-count"])
def test_resealed_aggregate_cannot_drop_or_substitute_original_preparation(consumers, complete_export, field):
    store, spec, *_ = consumers
    value = independent_aggregate().aggregate(deepcopy(complete_export))
    if field == "all-fields":
        for name in FIELDS:
            del value[name]
    elif field == "bool-count":
        value["calibration_status_counts"]["CALIBRATED"] = True
    elif field == "calibration_cost":
        value[field]["charged_ms"] = 0
    elif field == "calibration_evidence_hash":
        value[field] = "0" * 64
    else:
        value[field] = {"CALIBRATED": 999}
    value["aggregate_hash"] = digest({k: v for k, v in value.items() if k != "aggregate_hash"})
    before = store.events()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH.*calibration"):
        accept_aggregate(store, value, spec["study_id"])
    assert store.events() == before


def test_actual_failed_and_unknown_cost_survives_package_import(outcomes, outcome_export, tmp_path):
    from application.research_query import ResearchQuery
    store, spec, grant, _ = outcomes
    workers = [e for e in store.events() if e["event_kind"] == "WORKER_STARTED"]
    paper = independent_aggregate()
    directory = tmp_path / "original-cost-package"
    value = paper.write_package(outcome_export, directory)
    package = accept_evidence_package(store, directory, value["aggregate_hash"])
    header = outcome_export["probability_calibration"]
    assert value["calibration_cost"] == header["cost"]
    assert value["calibration_cost"]["charged_ms"] is None
    assert value["calibration_cost"]["unique_attempts"] == 4
    assert value["calibration_status_counts"] == {"CALIBRATED": 1, "FAILED": 2, "UNAVAILABLE": 1}
    assert all(arm["independent_n"] == 0 for arm in value["arms"])
    # Target arms have no attempts at all: known no execution is zero. The
    # original unreserved source failure above keeps shared preparation unknown.
    assert all(arm["cost"]["charged_ms"] == 0 and arm["cost"]["source_event_hashes"] == []
               for arm in value["arms"])
    index = json.loads((directory / "PaperEvidenceIndex.json").read_bytes())
    assert index["claims"] == [] and index["calibration_cost"] == header["cost"]
    assert store.manifest("comparison-" + value["aggregate_hash"]) == package
    query = ResearchQuery(store, grant["authorization_id"], authorization_version=grant.get("version"))
    assert query.comparison(value["aggregate_hash"])["aggregate"] == value
    downloaded, media = query.artifact(package["table"], export=True)
    assert media == "text/csv" and downloaded == (directory / "metrics.csv").read_bytes()
    downloaded, media = query.artifact(package["evidence-index"], export=True)
    assert json.loads(downloaded)["calibration_cost"] == header["cost"]
    assert [e for e in store.events() if e["event_kind"] == "WORKER_STARTED"] == workers


@pytest.mark.parametrize("member", ["aggregate.json", "metrics.csv", "PaperEvidenceIndex.json"])
def test_resealed_package_substitution_is_rejected_without_import_writes(consumers, complete_export, tmp_path, member):
    store, spec, *_ = consumers
    paper = independent_aggregate()
    directory = tmp_path / "substituted-package"
    value = paper.write_package(deepcopy(complete_export), directory)
    if member == "aggregate.json":
        value["calibration_cost"]["charged_ms"] = 0
        value["aggregate_hash"] = digest({k: v for k, v in value.items() if k != "aggregate_hash"})
        table = paper.csv_bytes(value)
        (directory / member).write_bytes(encode(value))
        (directory / "metrics.csv").write_bytes(table)
        (directory / "PaperEvidenceIndex.json").write_bytes(encode(paper.evidence_index(value, table)))
    elif member == "metrics.csv":
        rows = list(csv.DictReader(io.StringIO((directory / member).read_text(encoding="utf-8"))))
        row = next(row for row in rows if row["row_kind"] == "shared-calibration-preparation")
        row["charged_ms"] = "0"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        table = stream.getvalue().encode()
        (directory / member).write_bytes(table)
        index = json.loads((directory / "PaperEvidenceIndex.json").read_bytes())
        index["table_sha256"] = hashlib.sha256(table).hexdigest()
        (directory / "PaperEvidenceIndex.json").write_bytes(encode(index))
    else:
        index = json.loads((directory / member).read_bytes())
        index["calibration_status_counts"] = {"CALIBRATED": 999}
        (directory / member).write_bytes(encode(index))
    manifest = {"schema_version": "pirc25-evidence-package-v1", "aggregate_hash": value["aggregate_hash"],
        "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                  for name in ("aggregate.json", "metrics.csv", "PaperEvidenceIndex.json")}}
    (directory / "manifest.json").write_bytes(encode(manifest))
    before = store.events()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        accept_evidence_package(store, directory, value["aggregate_hash"])
    assert store.events() == before


def test_noncalibrated_bundle_cannot_acquire_invented_preparation_cost(tmp_path):
    from application.research_evidence import export_evidence
    from tests.test_research_evidence import setup
    store, spec, grant = setup(tmp_path)
    original = export_evidence(store, spec["study_id"], grant)
    value = independent_aggregate().aggregate(original)
    value.update({name: display_value()[name] for name in FIELDS})
    value["aggregate_hash"] = digest({k: v for k, v in value.items() if k != "aggregate_hash"})
    before = store.events()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH.*calibration"):
        accept_aggregate(store, value, spec["study_id"])
    assert store.events() == before
