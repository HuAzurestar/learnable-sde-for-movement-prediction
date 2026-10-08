"""Full compiled matrix -> shared refusals/export -> unchanged paper consumer."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

from application.research_budget import BudgetLedger
from application.propagation_execution import validate_propagation_cell
from application.research_contracts import CapabilityRegistry
from application.research_dimensions import comparison_dimensions
from application.research_evidence import export_evidence
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.design import StudyFunctional, StudyMethod, freeze_design
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_propagation_study_design import fixture


def test_full_matrix_retains_unavailable_rows_regions_histories_and_denominators(tmp_path):
    functions = (StudyFunctional("right", "endpoint-halfspace"),
                 StudyFunctional("far-right", "endpoint-halfspace", threshold=3.0))
    methods = tuple(StudyMethod(name, samples=8, steps=2) for name in ("euler", "exact", "mixture"))
    frozen = freeze_design(fixture(nonlinear=True, methods=methods, functionals=functions))
    value = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert len(value["cells"]) == value["propagation_design"]["expected_cells"] == 24
    # Registration/disclosure-only disposable test authority, no research job.
    store = ResearchStore(tmp_path / "runtime", "matrix-unit", initialize=True)
    store.register(value, digest(value))
    runner = SharedRunner(store, CapabilityRegistry())
    blocked = [c for c in value["cells"] if "execution_disposition" in c]
    assert len(blocked) == 16
    for cell in blocked:
        refusal = runner.run_cell(value["study_id"], digest(cell))
        assert refusal["state"] == "PREFLIGHT_FAILED"
        assert refusal["error_code"] == cell["execution_disposition"]["status"]
    before_export = store.events()
    assert not {event["event_kind"] for event in before_export} & {"WORKER_STARTED", "RESERVE", "QUEUED", "SETTLE"}
    for arm in value["arms"]:
        assert BudgetLedger(store).balance(arm["arm_id"])["committed_ms"] == 0
    grant = {"authorization_id": "matrix-export", "study_id": value["study_id"],
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("engineering export only"),
        "purposes": ["export"], "visibilities": ["synthetic"], "block_ids": [value["cells"][0]["block_id"]]}
    store.authorize(grant)
    bundle = export_evidence(store, value["study_id"], grant)
    assert len(bundle["cells"]) == len(bundle["expected_cells"]) == 24
    assert sum(row["status"] == "PREFLIGHT_FAILED" for row in bundle["cells"]) == 16
    assert sum(row["status"] == "MISSING" for row in bundle["cells"]) == 8
    for row in bundle["cells"]:
        assert row["metrics"] is None and row["artifact_id"] is None
        assert row["comparison_dimensions"] == comparison_dimensions(row["registered_cell"])
        if row["status"] == "PREFLIGHT_FAILED":
            assert row["history"][0]["error_code"] == row["registered_cell"]["execution_disposition"]["status"]
            assert row["cost"]["charged_ms"] is None  # no fabricated measured cost
    assert len({(row["arm_id"], row["block_id"], row["seed"], digest(row["comparison_dimensions"]))
                for row in bundle["cells"]}) == 24
    # Exercise the actual existing paired paper consumer, not a local imitation.
    paper = Path(__file__).resolve().parents[2] / "TSDE-SDE"
    aggregator = paper / "scripts/pirc25/aggregate.py"
    assert aggregator.is_file(), "paired paper checkout required"
    source = tmp_path / "fixture-bundle.json"
    source.write_bytes(encode(bundle))
    output = tmp_path / "paper-fixture"
    outcome = subprocess.run([sys.executable, "-B", str(aggregator), str(source),
        "--expected-hash", bundle["bundle_hash"], "--output", str(output)],
        cwd=paper, capture_output=True, text=True, timeout=30)
    assert outcome.returncode == 0, outcome.stderr
    aggregate = json.loads((output / "aggregate.json").read_bytes())
    assert aggregate["qualification"] == "descriptive"
    assert aggregate["expected_cell_count"] == 24 and aggregate["successful_cell_count"] == 0
    assert len(aggregate["cell_dispositions"]) == 24
    # Three methods, two horizons, two distinct functional/region strata.
    assert len(aggregate["arms"]) == 12
    assert all(arm["independent_n"] == 0 and arm["expected_cells"] == 2 for arm in aggregate["arms"])
    assert all(arm["status_rates"]["denominator"] == 2 for arm in aggregate["arms"])
    assert {arm["status"] for arm in aggregate["arms"]} == {"incomplete"}
    assert not any(event["event_kind"] == "WORKER_STARTED" for event in store.events())


def test_non_executable_matrix_export_still_requires_a_real_shared_disclosure_grant(tmp_path):
    frozen = freeze_design(fixture(nonlinear=True, methods=(StudyMethod("exact"),), horizons=(1.0,), seeds=(11,)))
    value = frozen.study_spec(expected_hash=frozen.manifest_hash)
    store = ResearchStore(tmp_path, "matrix-denial-unit", initialize=True)
    store.register(value, digest(value))
    grant = {"authorization_id": "not-registered", "study_id": value["study_id"],
        "expires_at": "2099-01-01T00:00:00+00:00", "purposes": ["export"],
        "visibilities": ["synthetic"], "block_ids": [value["cells"][0]["block_id"]]}
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        export_evidence(store, value["study_id"], grant)
    assert not any(event["payload"].get("object_id", "").startswith("bundle-") for event in store.events())


def test_filtering_non_executable_rows_invalidates_the_original_resource_matrix_binding():
    frozen = freeze_design(fixture(nonlinear=True, methods=(StudyMethod("euler"), StudyMethod("exact"))))
    value = frozen.study_spec(expected_hash=frozen.manifest_hash)
    runnable = [cell for cell in value["cells"] if "execution" in cell]
    validate_propagation_cell(value, runnable[0])
    filtered = {**value, "cells": runnable}
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        validate_propagation_cell(filtered, runnable[0])
