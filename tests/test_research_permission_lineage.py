"""Adversarial permissions: source labels and pilot mode cannot declassify inputs."""

from dataclasses import replace
import threading

import pytest

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_query import ResearchQuery
from application.research_recovery import RecoveryPlugin, RecoveryRegistry, SharedRecovery
from experiments.pirc25.runner import SharedRunner
from experiments.pirc25.web import make_server
from infrastructure.research_store import ResearchError, digest
from tests.research_admission_fixtures import attach_foreign_model
from tests.test_research_admission_chain import prepared, fixture_command
from tests.test_research_web import request


def limited_grant(store, grant):
    limited = {**grant, "authorization_id": "synthetic-only", "visibilities": ["synthetic"],
               "purposes": ["preview", "export", "resume"]}
    store.authorize(limited)
    return limited


def test_restricted_source_result_is_not_declassified_by_cell(tmp_path):
    store, value, registry, grant = prepared(tmp_path, formal=True, package_visibility="restricted")
    store.register(value, digest(value))
    result = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert result["state"] == "SUCCEEDED"
    assert store.manifest("artifact-" + result["artifact_id"])["visibility"] == "restricted"
    limited = limited_grant(store, grant)
    for purpose in ("preview", "export"):
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            store.read_artifact(result["artifact_id"], purpose=purpose, authorization=limited)
    assert store.read_artifact(result["artifact_id"], purpose="preview", authorization=grant)


def test_old_misclassified_result_and_metadata_are_denied_after_reopen(tmp_path, monkeypatch):
    # Simulate publication by the original supervisor without modifying any
    # immutable metadata after it has been stored.
    store, value, registry, grant = prepared(tmp_path, formal=True, package_visibility="restricted")
    original = store.artifact
    def legacy_publish(content, **metadata):
        if metadata["role"] == "result":
            metadata["visibility"] = "synthetic"
        return original(content, **metadata)
    monkeypatch.setattr(store, "artifact", legacy_publish)
    store.register(value, digest(value))
    result = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    limited = limited_grant(store, grant)
    from infrastructure.research_store import ResearchStore
    reopened = ResearchStore(tmp_path, "admitted")
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        reopened.read_artifact(result["artifact_id"], purpose="preview", authorization=limited)
    query = ResearchQuery(reopened, limited["authorization_id"])
    for kind in ("study", "run", "comparison"):
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            query.list(kind)
    server = make_server(reopened, limited["authorization_id"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for suffix in ("", "?download=1"):
            assert request(server, "/api/artifacts/" + result["artifact_id"] + suffix)[0] == 403
        assert request(server, "/api/runs/" + reopened.attempts()[result["attempt_id"]]["run_id"])[0] == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_checkpoint_inherits_registered_source_visibility_before_admission(tmp_path):
    store, value, registry, grant = prepared(tmp_path, package_visibility="restricted")
    cell = value["cells"][0]
    plugin = registry.resolve(cell["plugin_id"], cell["capability"], version=cell["execution"]["component_version"])
    from tests.research_admission_fixtures import synthetic_plugin, bind_fixture_execution
    plugin = synthetic_plugin(plugin.plugin_id, plugin.capabilities, plugin.state_order, plugin.units, "exact", fixture_command)
    bind_fixture_execution(value, plugin)
    execution = CapabilityRegistry()
    execution.register(plugin)
    adapters = RecoveryRegistry()
    adapters.register(RecoveryPlugin(plugin.plugin_id, "exact", fixture_command, plugin.registry_entry.version))
    store.register(value, digest(value))
    attempt = store.new_attempt(store.register_run("synthetic", cell))
    checkpoint = SharedRecovery(store, execution, adapters).checkpoint(attempt,
        {"step": 1, "data_position": 1, "method_state": {}, "rng_state": {"seed": 1}})
    assert store.manifest("artifact-" + checkpoint)["visibility"] == "restricted"
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        store.read_artifact(checkpoint, purpose="resume", authorization=limited_grant(store, grant))


@pytest.mark.parametrize("mode", ["fixture", "pilot", "formal"])
@pytest.mark.parametrize("mutation", ["missing-grant", "missing-consumer", "expired", "wrong-study",
    "wrong-protocol", "missing-protocol", "wrong-purpose", "missing-protocol-binding"])
def test_foreign_model_permission_is_required_in_every_mode(tmp_path, mode, mutation):
    store, value, registry, _ = prepared(tmp_path, formal=True, two_arms=True)
    attach_foreign_model(store, value, consumer=mutation != "missing-consumer")
    value["admission"]["mode"] = mode
    if mutation == "missing-grant":
        value["admission"].pop("model_authorization_id")
    elif mutation == "wrong-protocol":
        value["admission"]["model_protocol_id"] = "inputs"
    elif mutation == "missing-protocol":
        value["admission"].pop("model_protocol_id")
    elif mutation not in {"missing-consumer"}:
        grant = store.manifest("authorization-model-consumer")
        grant["authorization_id"] = "modified-model-grant"
        if mutation == "expired":
            grant["expires_at"] = "2000-01-01T00:00:00+00:00"
        elif mutation == "wrong-study":
            grant["study_id"] = "unrelated"
        elif mutation == "wrong-purpose":
            grant["purposes"] = ["preview"]
        else:
            grant.pop("protocol_hash", None)
        store.publish("authorization-" + grant["authorization_id"], grant)
        value["admission"]["model_authorization_id"] = grant["authorization_id"]
    store.register(value, digest(value))
    with pytest.raises(ResearchError):
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert not any(event["event_kind"] == "WORKER_STARTED" for event in store.events())


@pytest.mark.parametrize("mode", ["fixture", "pilot", "formal"])
def test_authorized_foreign_model_is_bound_in_every_mode(tmp_path, mode):
    store, value, registry, _ = prepared(tmp_path, formal=True, two_arms=True)
    attach_foreign_model(store, value)
    value["admission"]["mode"] = mode
    store.register(value, digest(value))
    result = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert result["state"] == "SUCCEEDED"
    admissions = [event["payload"]["admission_hash"] for event in store.events() if event["event_kind"] == "ADMISSION"]
    documents = store.manifest("admission-" + admissions[0])["documents"]
    assert documents["frozen_model"]["study_id"] == "model-study"
    assert documents["model_authorization"]["study_id"] == "model-study"


@pytest.mark.parametrize("mode", ["fixture", "pilot", "formal"])
def test_restricted_model_visibility_survives_synthetic_consumer_package(tmp_path, mode):
    store, value, registry, grant = prepared(tmp_path, formal=True, two_arms=True)
    attach_foreign_model(store, value, visibility="restricted")
    value["admission"]["mode"] = mode
    store.register(value, digest(value))
    result = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert result["state"] == "SUCCEEDED"
    assert store.manifest("artifact-" + result["artifact_id"])["visibility"] == "restricted"
    for purpose in ("preview", "export"):
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            store.read_artifact(result["artifact_id"], purpose=purpose, authorization=grant)
    broad = {**grant, "authorization_id": "consumer-restricted", "visibilities": ["synthetic", "restricted"]}
    store.authorize(broad)
    assert store.read_artifact(result["artifact_id"], purpose="preview", authorization=broad)
