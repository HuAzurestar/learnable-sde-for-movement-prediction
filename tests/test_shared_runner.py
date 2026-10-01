"""Independent-process affine compatibility and capability rejection."""

import json
import subprocess
import sys

import numpy as np
import pytest

from application.research_budget import BudgetSpec
from application.research_contracts import accept_model, accept_propagation
from experiments.pirc25.affine import execute, fixture_spec, four_state, single_axis
from experiments.pirc25.runner import SharedRunner, affine_registry
from infrastructure.research_store import ResearchStore, ResearchError, digest


@pytest.mark.parametrize("dimensions", [1, 4])
def test_supervised_affine_matches_existing_chain(tmp_path, dimensions):
    store = ResearchStore(tmp_path, "fixture", initialize=True)
    spec = fixture_spec(dimensions=dimensions)
    store.register(spec, digest(spec))
    result = SharedRunner(store).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60, category="smoke"))
    assert result["state"] == "SUCCEEDED"
    stored = json.loads((store.path / "artifacts" / result["artifact_id"]).read_bytes())
    legacy = single_axis(19) if dimensions == 1 else four_state(19)
    for metric, value in legacy["metrics"].items():
        assert stored["metrics"][metric] == pytest.approx(value, rel=1e-10, abs=1e-12)
    assert stored["qualification"] == "fixture"
    assert stored["source_schema"] == legacy["source_schema"]
    reused = SharedRunner(store).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert reused["reused"] and reused["artifact_id"] == result["artifact_id"]


def test_capability_mismatch_is_rejected_before_any_attempt(tmp_path):
    store = ResearchStore(tmp_path, "fixture", initialize=True)
    spec = fixture_spec()
    spec["cells"][0]["capability"] = "rare-event"
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="capability"):
        SharedRunner(store).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert not store.attempts()


def package(kind="FrozenDynamicsPackage"):
    return {"schema_version": "pirc25-package-v1", "kind": kind, "state_order": ["x", "y", "vx", "vy"],
            "units": ["m", "m", "m/s", "m/s"], "capabilities": ["generic-rollout"],
            "resume_level": "restart-only", "qualification": "fixture",
            **{key: digest(key) for key in ("code_hash", "data_hash", "input_hash", "output_hash", "protocol_hash")}}


def test_packages_fail_closed_but_oracle_fixture_needs_no_neural_model():
    value = package()
    assert accept_model(value) == digest(value)
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        accept_model(value, formal=True)
    with pytest.raises(ResearchError, match="state order"):
        accept_model({**value, "state_order": ["x", "vx", "y", "vy"]})
    oracle = package("PropagationResult")
    assert accept_propagation(oracle)
    with pytest.raises(ResearchError, match="qualified frozen model"):
        accept_propagation({**oracle, "requires_frozen_model": True})


def test_cli_independent_process_registration_query_and_error(tmp_path):
    base = [sys.executable, "-m", "experiments.pirc25", "--root", str(tmp_path), "--store-id", "cli-fixture"]
    for args in (["init"], ["fixture"], ["rebuild-index"], ["list"]):
        result = subprocess.run(base + args, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
    assert payload["items"][0]["object_id"] == "study-affine-fixture"
    failure = subprocess.run(base + ["show", "../escape"], text=True, capture_output=True, timeout=60)
    assert failure.returncode == 1
    assert json.loads(failure.stdout)["error"]["code"] == "CONTRACT_MISMATCH"
