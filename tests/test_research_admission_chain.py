"""R1 counterexample: a plugin cannot self-authorize a research execution."""

import sys

import pytest

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry, ExecutionPlugin
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.test_research_store import spec


def test_r1_runner_requires_registered_input_and_qualification_evidence(tmp_path):
    store = ResearchStore(tmp_path, "admission-counterexample", initialize=True)
    value = spec()
    cell = value["cells"][0]
    cell.update(plugin_id="self-qualified", capability="generic-rollout", visibility="synthetic")
    store.register(value, digest(value))
    called = []

    def command(output, registered_spec, registered_cell):
        called.append(True)
        payload = {"metrics": {"error": 1}, "forecast": {}, "fit": {}, "source_schema": "synthetic-counterexample"}
        result = {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED",
            "spec_hash": digest(registered_spec), "cell_hash": digest(registered_cell),
            "protocol_hash": registered_spec["protocol_hash"], "input_hash": registered_spec["data_hash"],
            "output_hash": digest(payload), "state_order": ["x", "y", "vx", "vy"],
            "units": ["m", "m", "m/s", "m/s"], "resume_level": "restart-only",
            "qualification": "qualified", "metric_units": {"error": "m"}, **payload}
        return [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
                str(output), encode(result).decode()]

    registry = CapabilityRegistry()
    registry.register(ExecutionPlugin("self-qualified", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", command))
    with pytest.raises(ResearchError):
        SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))
    assert not called, "unadmitted plugin was invoked before input gate"
    assert not any(a["state"] == "SUCCEEDED" for a in store.attempts().values())
