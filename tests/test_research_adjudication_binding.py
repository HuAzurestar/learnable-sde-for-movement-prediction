"""Actual admission/export and independent paper-side frozen-policy checks."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry, ExecutionPlugin
from application.research_evidence import export_evidence
from application.research_preregistration import PreregistrationGate
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_research_admission_chain import fixture_command
from tests.test_research_store import spec


def policy():
    return {"schema_version": "pirc25-adjudication-spec-v1",
        "primary_metric": {"name": "error", "definition": "admitted-synthetic-error-v1", "unit": "m", "direction": "minimize"},
        "independent_unit": "block_id", "seed_aggregation": "mean-within-block", "minimum_seeds": 1, "minimum_paired_blocks": 2,
        "seed_pairing": "independent-within-block",
        "interval": {"method": "paired-block-percentile-bootstrap", "confidence": 0.95, "replicates": 2000, "seed": 19},
        "multiplicity": "bonferroni", "practical_threshold": 0.5, "attempt_policy": "first-successful-attempt",
        "missing_policy": "exclude-incomplete-paired-blocks", "stopping_rule": "fixed-family-no-test-driven-expansion",
        "quality_gates": {"minimum_ess": None, "maximum_reference_error": None},
        "contrasts": [{"comparison_id": "primary", "reference": "affine", "candidate": "candidate",
                       "stratum_weights": [{"comparison_dimensions": {}, "weight": 1.0}]}]}


def command(output, value, cell):
    args = fixture_command(output, value, cell)
    result = json.loads(args[-1])
    result["metric_definitions"] = {"error": cell.get("definition", "admitted-synthetic-error-v1")}
    return [*args[:-1], encode(result).decode()]


def prepare(root, monkeypatch):
    store = ResearchStore(root, "adjudication", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="adjudication-fixture", capability="generic-rollout", visibility="synthetic")
    value["arms"].append({**value["arms"][0], "arm_id": "candidate", "model_family_id": "candidate"})
    value["cells"].append({**value["cells"][0], "arm_id": "candidate"})
    original = PreregistrationGate.register_preregistration

    def freeze(gate, plan, _expected):
        plan["adjudication_spec"] = policy()
        return original(gate, plan, digest(plan))

    monkeypatch.setattr(PreregistrationGate, "register_preregistration", freeze)
    plugin = ExecutionPlugin("adjudication-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", command)
    registry = CapabilityRegistry()
    registry.register(plugin)
    grant = admit_fixture(store, value, plugin, root, formal=True)
    value["comparison_plan"].update(adjudication_spec=policy(), adjudication_hash=digest(policy()))
    return store, value, registry, grant


def paper_compare(root, bundle):
    paper = Path(__file__).resolve().parents[2] / "TSDE-SDE"
    assert (paper / "scripts/pirc25/adjudication.py").is_file(), "paired paper checkout is required, not skipped"
    source = root / "bundle.json"
    source.write_bytes(encode(bundle))
    code = ("import sys,json; sys.path.insert(0,sys.argv[1]); "
            "from scripts.pirc25.adjudication import adjudicate; "
            "print(json.dumps(adjudicate(json.load(open(sys.argv[2],encoding='utf-8')),formal=True)))")
    return subprocess.run([sys.executable, "-B", "-c", code, str(paper), str(source)],
                          capture_output=True, text=True, timeout=30)


def test_formal_policy_is_frozen_and_actual_export_retains_metric_definition(tmp_path, monkeypatch):
    store, value, registry, grant = prepare(tmp_path, monkeypatch)
    store.register(value, digest(value))
    for cell in value["cells"]:
        assert SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    bundle = export_evidence(store, "synthetic", grant)
    assert all(row.get("metric_definitions") == {"error": "admitted-synthetic-error-v1"} for row in bundle["cells"])
    result = paper_compare(tmp_path, bundle)
    assert result.returncode == 0, result.stderr
    computed = json.loads(result.stdout)
    assert computed["qualification"] == "formal" and computed["adjudication_hash"] == digest(policy())
    assert computed["records"][0]["verdict"] == "INSUFFICIENT_DATA"  # one real block, never two seeds-as-n
    assert computed["records"][0]["interval"] is None

    clipped = deepcopy(bundle)
    clipped["cells"].pop()
    clipped["expected_cells"].pop()
    clipped["bundle_hash"] = digest({k: v for k, v in clipped.items() if k != "bundle_hash"})
    refused = paper_compare(tmp_path, clipped)
    assert refused.returncode != 0 and "registered matrix" in refused.stderr

    # A legacy exporter can rehash the spec/receipts consistently, but it cannot
    # move the policy into the already published preregistration before reads.
    changed = deepcopy(bundle)
    new_spec = deepcopy(value)
    new_spec["comparison_plan"]["adjudication_spec"]["practical_threshold"] = 0.9
    new_spec["comparison_plan"]["adjudication_hash"] = digest(new_spec["comparison_plan"]["adjudication_spec"])
    changed.update(spec_hash=digest(new_spec), comparison_plan=new_spec["comparison_plan"])
    changed["registered_spec"] = new_spec
    for row in changed["cells"]:
        receipt = row["admission"]
        receipt.update(spec=new_spec, spec_hash=digest(new_spec))
        receipt["admission_hash"] = digest({k: v for k, v in receipt.items() if k != "admission_hash"})
        row["admission_hash"] = receipt["admission_hash"]
    changed["bundle_hash"] = digest({k: v for k, v in changed.items() if k != "bundle_hash"})
    refused = paper_compare(tmp_path, changed)
    assert refused.returncode != 0 and "frozen in preregistration" in refused.stderr


def test_changed_policy_is_rejected_before_actual_input_read_or_worker(tmp_path, monkeypatch):
    store, value, registry, _ = prepare(tmp_path, monkeypatch)
    value["comparison_plan"]["adjudication_spec"]["practical_threshold"] = 0.9
    value["comparison_plan"]["adjudication_hash"] = digest(value["comparison_plan"]["adjudication_spec"])
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="adjudication"):
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert not any(e["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for e in store.events())


def test_worker_cannot_substitute_metric_estimator_definition(tmp_path, monkeypatch):
    store, value, registry, _ = prepare(tmp_path, monkeypatch)
    value["cells"][0]["definition"] = "post-hoc-estimator-v1"
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="metric definition"):
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert not any(a["state"] == "SUCCEEDED" for a in store.attempts().values())
