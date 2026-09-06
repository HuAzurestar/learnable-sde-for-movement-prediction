"""Regression tests for Slice C atomic run commits and audit records."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from application.experiment import ExperimentApplication
from config import Components, Config
from domain import (
    ArtifactCommitError,
    ArtifactReference,
    DataValidationError,
    RunRecord,
)
from infrastructure import AtomicRunStore, JsonArtifactStore


def _config(output_root) -> Config:
    return Config(
        seed=17,
        dtype="float64",
        device="cpu",
        paths={"output_root": str(output_root)},
        components=Components(
            model="I1",
            estimator="EM",
            transfer="none",
            condition="none",
            inference="exact",
        ),
        model={"I1": {"n_modes": 2, "kappa": 0.0, "dt_ref": 1.0}},
    )


def _record(run_id: str = "run-001") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        issue_id="9",
        experiment_id="public-i1",
        code_version="bebbc55",
        data_id="synthetic-i1-v1",
        split="evaluation",
        config={"model": "I1", "inference": "exact"},
        seed=17,
        components={
            "model": "I1",
            "estimator": "EM",
            "inference": "exact",
        },
        reproducibility="partial",
        missing_reproducibility=(
            "component_versions",
            "dependency_lock",
            "environment_fingerprint",
        ),
    )


def _write_report(destination, value: float = 1.25) -> None:
    JsonArtifactStore().write(
        {"metric": "energy_score", "value": value},
        destination / "evaluation.json",
    )


def test_application_atomically_commits_artifact_and_run_record(tmp_path):
    output_root = tmp_path / "runs"
    app = ExperimentApplication.from_config(_config(output_root))

    committed = app.commit_run(_record(), _write_report)
    loaded = app.run_store.read("run-001")

    artifact_path = output_root / "run-001" / "evaluation.json"
    expected_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert loaded == committed
    assert loaded.artifacts["evaluation.json"].path == "run-001/evaluation.json"
    assert loaded.artifacts["evaluation.json"].sha256 == expected_hash
    assert loaded.artifacts["evaluation.json"].size_bytes == artifact_path.stat().st_size
    assert loaded.status == "succeeded"
    assert loaded.reproducibility == "partial"
    assert loaded.missing_reproducibility == (
        "component_versions",
        "dependency_lock",
        "environment_fingerprint",
    )


def test_failed_atomic_commit_leaves_no_visible_partial_run(tmp_path):
    store = AtomicRunStore(tmp_path / "runs")

    def fail_after_one_artifact(destination) -> None:
        _write_report(destination)
        raise RuntimeError("simulated writer failure")

    with pytest.raises(ArtifactCommitError, match="simulated writer failure"):
        store.commit(_record("run-failed"), fail_after_one_artifact)

    assert not (tmp_path / "runs" / "run-failed").exists()
    assert list((tmp_path / "runs").glob(".run-failed.*")) == []


def test_atomic_commit_never_overwrites_an_existing_run(tmp_path):
    app = ExperimentApplication.from_config(_config(tmp_path / "runs"))

    def write_existing(destination) -> None:
        app.model_store.save(
            app.model,
            {"run_id": "run-existing"},
            destination / "checkpoint.pt",
        )
        _write_report(destination)

    app.commit_run(_record("run-existing"), write_existing)
    artifact = tmp_path / "runs" / "run-existing" / "evaluation.json"
    checkpoint = tmp_path / "runs" / "run-existing" / "checkpoint.pt"
    original_artifact = artifact.read_bytes()
    original_checkpoint = checkpoint.read_bytes()

    with pytest.raises(ArtifactCommitError, match="already exists"):
        app.commit_run(
            _record("run-existing"),
            lambda destination: _write_report(destination, 99.0),
        )

    assert artifact.read_bytes() == original_artifact
    assert checkpoint.read_bytes() == original_checkpoint
    loaded = app.run_store.read("run-existing")
    assert set(loaded.artifacts) == {"checkpoint.pt", "evaluation.json"}
    assert loaded.artifacts["evaluation.json"].sha256 == (
        hashlib.sha256(original_artifact).hexdigest()
    )


def test_run_store_rejects_a_run_id_that_can_escape_its_root(tmp_path):
    store = AtomicRunStore(tmp_path / "runs")

    with pytest.raises(DataValidationError, match="run_id"):
        store.commit(_record(".."), _write_report)

    assert not (tmp_path / "evaluation.json").exists()


def test_run_store_detects_committed_artifact_tampering(tmp_path):
    store = AtomicRunStore(tmp_path / "runs")
    store.commit(_record("run-tampered"), _write_report)
    artifact = tmp_path / "runs" / "run-tampered" / "evaluation.json"
    artifact.write_text('{"metric": "changed"}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="digest mismatch"):
        store.read("run-tampered")


def test_failed_run_record_round_trips_without_a_success_artifact(tmp_path):
    store = AtomicRunStore(tmp_path / "runs")
    failed = replace(
        _record("run-capability-failure"),
        status="failed",
        failure_stage="predict",
        failure_reason="selected inference engine does not support model",
    )

    committed = store.commit(failed)

    assert committed.artifacts == {}
    assert store.read(failed.run_id) == failed


def test_run_record_rejects_inconsistent_status_and_reproducibility():
    with pytest.raises(DataValidationError, match="failure_stage"):
        replace(_record(), status="failed").validate()
    with pytest.raises(DataValidationError, match="failure_reason"):
        replace(_record(), status="failed", failure_stage="predict").validate()
    with pytest.raises(DataValidationError, match="successful"):
        replace(_record(), failure_reason="not allowed").validate()
    with pytest.raises(DataValidationError, match="reproducibility"):
        replace(
            _record(),
            reproducibility="complete",
            missing_reproducibility=("environment_fingerprint",),
        ).validate()


def test_run_record_rejects_non_mapping_component_and_artifact_fields():
    with pytest.raises(DataValidationError, match="components must be a mapping"):
        replace(_record(), components=[("model", "I1")]).validate()
    with pytest.raises(DataValidationError, match="artifacts must be a mapping"):
        replace(_record(), artifacts=[("forecast", object())]).validate()


def test_run_record_rejects_values_that_cannot_round_trip_through_json():
    reference = ArtifactReference(
        path="run-001/evaluation.json",
        sha256="0" * 64,
        size_bytes=1,
    )
    cases = (
        (replace(_record(), components={1: "I1"}), "component names and values"),
        (replace(_record(), components={"model": 1}), "component names and values"),
        (replace(_record(), config={1: "I1"}), "config must preserve JSON"),
        (replace(_record(), config={"axes": ("model",)}), "config must preserve JSON"),
        (replace(_record(), artifacts={1: reference}), "artifact name"),
        (
            replace(_record(), missing_reproducibility=["dependency_lock"]),
            "missing_reproducibility must be a tuple",
        ),
        (replace(_record(), status=[]), "status must be a string"),
        (
            replace(_record(), reproducibility=[]),
            "reproducibility must be a string",
        ),
        (
            replace(
                _record(),
                status="failed",
                failure_stage=1,
                failure_reason="reason",
            ),
            "failure_stage must be a string",
        ),
        (
            replace(
                _record(),
                status="failed",
                failure_stage="predict",
                failure_reason=1,
            ),
            "failure_reason must be a string",
        ),
    )

    for record, message in cases:
        with pytest.raises(DataValidationError, match=message):
            record.validate()


def test_artifact_reference_from_dict_does_not_coerce_invalid_types():
    with pytest.raises(DataValidationError, match="artifact path"):
        ArtifactReference.from_dict(
            {"path": 123, "sha256": "0" * 64, "size_bytes": 1}
        )
