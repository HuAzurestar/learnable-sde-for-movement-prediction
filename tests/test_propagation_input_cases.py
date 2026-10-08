"""Frozen case preparation is not source independence or data-read permission."""

from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.propagation_execution import validate_propagation_cell
from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash
from experiments.pirc27.design import StudyMethod, freeze_design
from experiments.pirc27.input_cases import (StudyInputCase, StudyInputPolicy,
    validate_cell_input_binding)
from experiments.pirc27.preparation import reload_design
from infrastructure.research_store import ResearchError, encode
from tests.test_propagation_study_design import fixture


def cases_design(*, private=False, nonlinear=False):
    design = fixture(nonlinear=nonlinear)
    model = design.models[0]
    policy = StudyInputPolicy("frozen-input-policy", content_hash("initial-law-procedure"),
        content_hash("origin-procedure"), design.selection_hash)
    cases = tuple(StudyInputCase("source-"+str(i), "instance-"+str(i), "generator-unit", "release-unit",
        "original-source-"+str(i), content_hash({"recording": i}),
        tuple(x+i for x in model.initial_mean), model.initial_covariance, float(i), float(i),
        source_kind="private-past-prefix" if private else "synthetic-recipe") for i in range(2))
    return replace(design, input_cases=cases, input_policy=policy)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_cases_pair_actual_inputs_without_seeds_windows_or_extra_budget_arms(nonlinear):
    design = cases_design(nonlinear=nonlinear)
    frozen = freeze_design(design)
    doc = frozen.manifest()
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert len(doc["matrix"]) == doc["expected_cells"] == 16
    assert doc["arms"] == freeze_design(fixture(nonlinear=nonlinear)).manifest()["arms"]
    assert doc["axis_manifest"]["input_policy"]["instance_policy"] == "one-preselected-instance-per-source-block"
    assert len({c["block_id"] for c in spec["cells"]}) == 2
    assert len({c["comparison_dimensions"]["initialization"]["policy_hash"] for c in spec["cells"]}) == 1
    for case in design.input_cases:
        selected = [c for c in spec["cells"] if c["block_id"] == case.block_id]
        for cell in selected:
            _, req, _, _ = validate_propagation_cell(spec, cell)
            assert req.initial_mean == case.initial_mean and req.origin == case.origin
            assert req.initial_covariance == case.initial_covariance
            assert cell["instance_id"] == case.instance_id
            assert cell["input_binding"]["qualification"] == "declared-only-not-scientific"
        for h in design.horizons:
            for seed in design.seeds:
                paired = [c for c in selected if c["horizon"] == h and c["seed"] == seed]
                assert len({c["propagation_request"]["coupling_id"] for c in paired}) == 1
    same_stratum = [c["comparison_dimensions"] for c in spec["cells"] if c["horizon"] == 1.]
    assert all(d == same_stratum[0] for d in same_stratum)


def test_private_cases_remain_explicit_unavailable_rows_not_synthetic_relabeled_data():
    design = cases_design(private=True)
    design = replace(design, methods=design.methods+(StudyMethod("pde", samples=8, steps=2),),
        arms=fixture(methods=design.methods+(StudyMethod("pde", samples=8, steps=2),)).arms)
    frozen = freeze_design(design)
    doc = frozen.manifest()
    assert len(doc["matrix"]) == 24
    for row in doc["matrix"]:
        assert row["cell"]["visibility"] == "restricted" and "execution" not in row["cell"]
        assert row["disposition"] == ("NOT_IMPLEMENTED" if row["method"] == "pde" else "INELIGIBLE")
        if row["method"] != "pde":
            assert "past-only" in row["reason"]


@pytest.mark.parametrize("change", ["same-block", "same-instance", "same-source", "same-content", "release-alias"])
def test_aliases_and_correlated_instances_cannot_create_independent_blocks(change):
    design = cases_design()
    a, b = design.input_cases
    fields = {"same-block": {"block_id": a.block_id}, "same-instance": {"instance_id": a.instance_id},
        "same-source": {"source_block_id": a.source_block_id},
        "same-content": {"source_sha256": a.source_sha256, "dataset_id": "renamed-dataset"},
        "release-alias": {"source_block_id": a.source_block_id, "release_id": "new-release"}}[change]
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(replace(design, input_cases=(a, replace(b, **fields))))


@pytest.mark.parametrize("fields", [{"source_sha256": "bad"}, {"history_cutoff": 2.},
    {"initial_mean": (0., 0., 0.)}, {"origin": True}, {"source_kind": "test-is-synthetic"},
    {"split_role": "evaluation"}, {"initial_covariance": ((-1., 0., 0., 0.), (0.,)*4, (0.,)*4, (0.,)*4)}])
def test_invalid_bounded_input_identity_or_law_is_refused_before_expansion(fields):
    design = cases_design()
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(replace(design, input_cases=(replace(design.input_cases[0], **fields),)))


def test_policy_is_required_and_selection_binding_is_not_optional():
    design = cases_design()
    for changed in (replace(design, input_policy=None),
            replace(design, input_policy=replace(design.input_policy, selection_hash=content_hash("other"))),
            replace(design, input_cases=())):
        with pytest.raises((DataValidationError, ResearchError)):
            freeze_design(changed)


def test_runtime_validates_exact_case_request_and_protocol_source_before_bytes():
    frozen = freeze_design(cases_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    cell = spec["cells"][0]
    case = cell["input_binding"]["case"]
    block = {k: case[k] for k in ("block_id", "dataset_id", "release_id", "source_block_id", "split_role")}
    block["sha256"] = case["source_sha256"]
    validate_cell_input_binding(cell, protocol_block=block)
    for key in ("dataset_id", "release_id", "source_block_id", "sha256", "split_role"):
        bad = {**block, key: "changed"}
        with pytest.raises(ResearchError):
            validate_cell_input_binding(cell, protocol_block=bad)
    for key, value in (("block_id", "new-block"), ("instance_id", "new-window")):
        with pytest.raises(ResearchError):
            validate_propagation_cell(spec, {**cell, key: value})
    changed = copy.deepcopy(cell)
    changed["input_binding"]["case"]["origin"] = 99.
    with pytest.raises(ResearchError):
        validate_propagation_cell(spec, changed)


def test_source_binding_also_binds_pairing_root_not_only_numerical_initial_values():
    frozen = freeze_design(cases_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    cell = copy.deepcopy(spec["cells"][0])
    other_source = next(c for c in spec["cells"] if c["block_id"] != cell["block_id"])
    cell["propagation_request"]["coupling_id"] = other_source["propagation_request"]["coupling_id"]
    with pytest.raises(ResearchError, match="pairing"):
        validate_cell_input_binding(cell)


def test_complete_reload_regenerates_cases_and_rejects_rehashed_request_substitution(tmp_path):
    frozen = freeze_design(cases_design())
    path = tmp_path/"source-cases.json"
    path.write_bytes(encode(frozen.manifest()))
    assert reload_design(path, expected_hash=frozen.manifest_hash).manifest_hash == frozen.manifest_hash
    doc = frozen.manifest()
    doc["matrix"][0]["cell"]["propagation_request"]["origin"] = 77.
    path.write_bytes(encode(doc))
    with pytest.raises(ResearchError):
        reload_design(path, expected_hash=content_hash(doc))


def test_case_axis_cap_is_checked_before_package_or_cartesian_allocation(monkeypatch):
    from experiments.pirc27 import design as module
    design = cases_design()
    monkeypatch.setattr(module, "product", lambda *args: pytest.fail("expanded before quota"))
    monkeypatch.setattr(module, "validate_oracle_input", lambda *args, **kwargs: pytest.fail("processed package before quota"))
    with pytest.raises(DataValidationError, match="cell limit"):
        freeze_design(replace(design, input_cases=design.input_cases*32,
            horizons=tuple(float(i) for i in range(1, 65)), seeds=tuple(range(64))))


@pytest.mark.parametrize("private", [False, True])
def test_full_authorized_export_and_actual_paper_cli_preserve_source_blocks_and_missing_denominator(tmp_path, private):
    from application.research_evidence import export_evidence
    from experiments.pirc25.runner import SharedRunner
    from application.research_contracts import CapabilityRegistry
    from infrastructure.research_store import ResearchStore, digest
    design = cases_design(private=private)
    design = replace(design, horizons=(1.,))
    frozen = freeze_design(design)
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    store = ResearchStore(tmp_path/"runtime", "input-cases-export", initialize=True)
    store.register(spec, digest(spec))
    runner = SharedRunner(store, CapabilityRegistry())
    if private:
        for cell in spec["cells"]:
            assert runner.run_cell(spec["study_id"], digest(cell))["state"] == "PREFLIGHT_FAILED"
    grant = {"authorization_id": "export-unit", "study_id": spec["study_id"],
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("engineering export only"),
        "purposes": ["export"], "visibilities": ["restricted" if private else "synthetic"],
        "block_ids": [case.block_id for case in design.input_cases]}
    store.authorize(grant)
    bundle = export_evidence(store, spec["study_id"], grant)
    assert len(bundle["expected_cells"]) == len(bundle["cells"]) == 8
    assert {c["block_id"] for c in bundle["cells"]} == {c.block_id for c in design.input_cases}
    source = tmp_path/"bundle.json"
    source.write_bytes(encode(bundle))
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    output = tmp_path/"paper"
    result = subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/aggregate.py"), str(source),
        "--expected-hash", bundle["bundle_hash"], "--output", str(output)], cwd=paper,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((output/"aggregate.json").read_bytes())
    assert aggregate["expected_cell_count"] == 8 and aggregate["successful_cell_count"] == 0
    assert len(aggregate["arms"]) == 2
    for arm in aggregate["arms"]:
        assert arm["expected_blocks"] == 2 and arm["independent_n"] == 0
        assert arm["status_rates"]["denominator"] == 4 and arm["status"] == "incomplete"
        assert arm["dispositions"] == {"PREFLIGHT_FAILED" if private else "MISSING": 4}
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "SETTLE", "READ_STARTED"} for e in store.events())


@pytest.mark.parametrize("changed_key", ["dataset_id", "release_id", "source_block_id", "sha256", "split_role"])
def test_actual_admission_refuses_source_mismatch_before_grant_package_or_read(changed_key):
    from application.research_admission import AdmissionGate, data_binding
    from infrastructure.research_store import digest
    frozen = freeze_design(cases_design())
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    cell = spec["cells"][0]
    case = cell["input_binding"]["case"]
    block = {k: case[k] for k in ("block_id", "dataset_id", "release_id", "source_block_id", "split_role")}
    block["sha256"] = case["source_sha256"]
    block[changed_key] = "test" if changed_key == "split_role" else "b"*64 if changed_key == "sha256" else "changed"
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "source-unit",
        "study_id": spec["study_id"], "blocks": [block]}
    spec.update(protocol_hash=digest(protocol), data_hash=data_binding(protocol),
        admission={"mode": "fixture", "protocol_id": protocol["protocol_id"]})
    _, _, _, plugin = validate_propagation_cell(spec, cell)

    class UnreadStore:
        def attempts(self):
            return {"attempt-unit": {"run_id": "run-unit"}}

        def publish(self, *args):
            pass  # Registry metadata only, no official store or data bytes.

        def manifest(self, key):
            if key == "run-run-unit":
                return {"spec_hash": digest(spec), "cell_hash": digest(cell), "run_id": "run-unit"}
            if key == "protocol-source-unit":
                return protocol
            pytest.fail("package or other manifest looked up before source refusal")

        def authorization(self, *args, **kwargs):
            pytest.fail("grant looked up before source refusal")

    with pytest.raises(ResearchError, match="case source differs from registered protocol block"):
        AdmissionGate(UnreadStore()).prepare(spec, cell, plugin, "attempt-unit")
