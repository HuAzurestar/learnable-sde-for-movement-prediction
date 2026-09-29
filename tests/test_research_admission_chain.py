"""R1 counterexample: a plugin cannot self-authorize a research execution."""

import sys
import json
from pathlib import Path
import subprocess
from copy import deepcopy

import pytest

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry, ExecutionPlugin
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.test_research_store import spec
from tests.research_admission_fixtures import admit_fixture, attach_foreign_model


def fixture_command(output, value, cell):
    payload = {"metrics": {"error": 1}, "forecast": {}, "fit": {}, "source_schema": "admitted-synthetic-v1"}
    result = {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED", "spec_hash": digest(value),
        "cell_hash": digest(cell), "protocol_hash": value["protocol_hash"], "input_hash": value["data_hash"],
        "output_hash": digest(payload), "state_order": ["x", "y", "vx", "vy"],
        "units": ["m", "m", "m/s", "m/s"], "resume_level": "restart-only",
        "qualification": "qualified" if value["admission"]["mode"] == "formal" else "fixture",
        "metric_units": {"error": "m"}, **payload}
    if cell.get("force_qualification"):
        result["qualification"] = cell["force_qualification"]
    return [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
            str(output), encode(result).decode()]


def prepared(tmp_path, *, formal=False, two_arms=False, package_visibility="synthetic"):
    store = ResearchStore(tmp_path, "admitted", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="admitted-fixture", capability="generic-rollout", visibility="synthetic")
    if two_arms:
        value["arms"].append({**value["arms"][0], "arm_id": "candidate", "model_family_id": "candidate"})
        value["cells"].append({**value["cells"][0], "arm_id": "candidate"})
    plugin = ExecutionPlugin("admitted-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", fixture_command)
    registry = CapabilityRegistry()
    registry.register(plugin)
    grant = admit_fixture(store, value, plugin, tmp_path, formal=formal, package_visibility=package_visibility)
    return store, value, registry, grant


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


@pytest.mark.parametrize("formal", [False, True])
def test_admitted_plugin_binds_actual_input_read_package_and_result(tmp_path, formal):
    from application.research_evidence import export_evidence
    store, value, registry, grant = prepared(tmp_path, formal=formal)
    store.register(value, digest(value))
    result = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert result["state"] == "SUCCEEDED"
    artifact = json.loads(store.read_artifact(result["artifact_id"], purpose="preview", authorization=grant))
    admission = store.manifest("admission-" + artifact["admission_hash"])
    assert admission["attempt_id"] == result["attempt_id"]
    assert admission["documents"]["package"]["input_hash"] == value["data_hash"]
    assert admission["input_evidence"][0]["payload"]["attempt_id"] == result["attempt_id"]
    exported = export_evidence(store, "synthetic", grant)
    assert exported["cells"][0]["admission"] == admission
    assert artifact["qualification"] == ("qualified" if formal else "fixture")


@pytest.mark.parametrize("mutation", ["missing-admission", "missing-package", "bad-code", "bad-data",
    "bad-upstream", "expired-grant", "wrong-protocol", "missing-qualification", "bad-package-payload"])
def test_r1_invalid_admission_never_starts_worker(tmp_path, mutation):
    store, value, registry, grant = prepared(tmp_path, formal=True)
    if mutation == "missing-admission":
        value.pop("admission")
    elif mutation == "missing-package":
        value["admission"]["package_hash"] = "0" * 64
    elif mutation == "bad-code":
        value["code_hash"] = "0" * 64
    elif mutation == "bad-data":
        value["data_hash"] = "0" * 64
    elif mutation == "bad-upstream":
        value["admission"]["upstream_hash"] = "0" * 64
    elif mutation == "expired-grant":
        expired = {**grant, "authorization_id": "expired", "expires_at": "2000-01-01T00:00:00+00:00"}
        # Simulate a once-registered grant that is now expired; authorize()
        # correctly refuses creating an already expired grant today.
        store.publish("authorization-expired", expired)
        value["admission"]["authorization_id"] = "expired"
    elif mutation == "wrong-protocol":
        value["protocol_hash"] = "0" * 64
    else:
        package = store.manifest("package-" + value["admission"]["package_hash"])
        if mutation == "missing-qualification":
            package["qualification_hash"] = "0" * 64
        else:
            package["payload"] = {"changed": True}
        value["admission"]["package_hash"] = store.publish("package-" + digest(package), package)
    store.register(value, digest(value))
    with pytest.raises(ResearchError):
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())
    assert all(a["state"] == "PREFLIGHT_FAILED" for a in store.attempts().values())


def test_r1_worker_cannot_promote_fixture_qualification(tmp_path):
    store, value, registry, _ = prepared(tmp_path)
    value["cells"][0]["force_qualification"] = "qualified"
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert not any(a["state"] == "SUCCEEDED" for a in store.attempts().values())


def test_private_admission_attachments_cannot_be_laundered_by_synthetic_cells(tmp_path):
    from application.research_evidence import export_evidence, accept_aggregate
    store, value, registry, grant = prepared(tmp_path, formal=True, package_visibility="restricted")
    store.register(value, digest(value))
    assert SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    bundle = export_evidence(store, "synthetic", grant)
    aggregate = {"schema_version": "pirc25-aggregate-v1", "study_id": "synthetic", "spec_hash": digest(value),
        "protocol_hash": value["protocol_hash"], "source_bundle_hash": bundle["bundle_hash"],
        "expected_cell_count": len(value["cells"]), "cell_dispositions": bundle["cells"], "arms": []}
    aggregate["aggregate_hash"] = digest(aggregate)
    artifact = accept_aggregate(store, aggregate, "synthetic")
    assert artifact["visibility"] == "restricted", "private evidence inherited synthetic cell visibility"
    limited = {**grant, "authorization_id": "synthetic-only", "visibilities": ["synthetic"], "purposes": ["preview"]}
    store.authorize(limited)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=limited)


def test_r1_recovery_builder_must_match_accepted_package(tmp_path):
    from application.research_admission import AdmissionGate
    store, value, registry, _ = prepared(tmp_path)
    store.register(value, digest(value))
    cell = value["cells"][0]
    attempt = store.new_attempt(store.register_run("synthetic", cell))
    def substituted_builder(*args):
        raise AssertionError("must not run")
    with pytest.raises(ResearchError, match="package differs"):
        AdmissionGate(store).prepare(value, cell, registry.resolve(cell["plugin_id"], cell["capability"]),
                                     attempt, recovery_builder=substituted_builder)
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())


@pytest.mark.parametrize("permission", ["allowed", "missing-consumer", "evaluate-only", "wrong-model-protocol"])
def test_frozen_model_cross_study_reuse_requires_explicit_read_and_export_grants(tmp_path, permission):
    from application.research_evidence import export_evidence
    store, value, registry, grant = prepared(tmp_path, formal=True, two_arms=True)
    attach_foreign_model(store, value, consumer=permission != "missing-consumer", export=permission != "evaluate-only")
    if permission == "wrong-model-protocol":
        value["admission"]["model_protocol_id"] = "inputs"
    store.register(value, digest(value))
    runner = SharedRunner(store, registry)
    if permission == "wrong-model-protocol":
        with pytest.raises(ResearchError, match="source protocol"):
            runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
        assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())
        return
    if permission == "missing-consumer":
        with pytest.raises(ResearchError, match="consumer study"):
            runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
        assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())
        return
    assert runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    if permission == "evaluate-only":
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            export_evidence(store, "synthetic", grant)
    else:
        bundle = export_evidence(store, "synthetic", grant)
        row = bundle["cells"][0]
        assert row["admission"]["documents"]["frozen_model"]["study_id"] == "model-study"
        aggregator = Path(__file__).resolve().parents[2] / "TSDE-SDE/scripts/pirc25/aggregate.py"
        if aggregator.is_file():
            source = tmp_path / "model-bundle.json"
            source.write_bytes(encode(bundle))
            accepted = subprocess.run([sys.executable, "-B", str(aggregator), str(source),
                "--expected-hash", bundle["bundle_hash"], "--formal", "--output", str(tmp_path / "model-evidence")],
                capture_output=True, text=True, timeout=30)
            assert accepted.returncode == 0, accepted.stderr


def test_formal_chain_roundtrip_and_resealed_tampering_rejected_by_tsde(tmp_path):
    from application.research_evidence import export_evidence
    aggregator = Path(__file__).resolve().parents[2] / "TSDE-SDE/scripts/pirc25/aggregate.py"
    if not aggregator.is_file():
        pytest.skip("cross-repository check requires explicit sibling TSDE checkout")
    store, value, registry, grant = prepared(tmp_path, formal=True, two_arms=True)
    store.register(value, digest(value))
    for cell in value["cells"]:
        assert SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    bundle = export_evidence(store, "synthetic", grant)

    def invoke(source, name):
        source["bundle_hash"] = digest({k: v for k, v in source.items() if k != "bundle_hash"})
        path = tmp_path / (name + ".json")
        path.write_bytes(encode(source))
        return subprocess.run([sys.executable, "-B", str(aggregator), str(path), "--expected-hash", source["bundle_hash"],
            "--formal", "--output", str(tmp_path / name)], capture_output=True, text=True, timeout=30)

    accepted = invoke(bundle, "accepted")
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    result = json.loads((tmp_path / "accepted/aggregate.json").read_bytes())
    assert result["qualification"] == "formal" and result["successful_cell_count"] == 2
    assert result["comparisons"][0]["independent_n"] == 1  # Engineering evidence, not scientific power.
    for mutation in ("no-admission", "no-reads", "other-attempt", "changed-package", "expired-grant", "no-qualification"):
        changed = deepcopy(bundle)
        row = changed["cells"][0]
        receipt = row["admission"]
        if mutation == "no-admission":
            row.pop("admission")
        elif mutation == "no-reads":
            receipt["input_evidence"] = []
        elif mutation == "other-attempt":
            receipt["attempt_id"] = "other"
        elif mutation == "changed-package":
            receipt["documents"]["package"]["code_hash"] = "0" * 64
        elif mutation == "expired-grant":
            receipt["documents"]["authorization"]["expires_at"] = "2000-01-01T00:00:00+00:00"
        else:
            receipt["documents"].pop("qualification")
        # Deliberately recompute enclosing hashes: the validator must check
        # cross-object bindings and evidence, not merely a top-level checksum.
        receipt["admission_hash"] = digest({k: v for k, v in receipt.items() if k != "admission_hash"})
        row["admission_hash"] = receipt["admission_hash"]
        rejected = invoke(changed, mutation)
        assert rejected.returncode != 0, mutation
        assert "admission evidence" in rejected.stderr, rejected.stderr
