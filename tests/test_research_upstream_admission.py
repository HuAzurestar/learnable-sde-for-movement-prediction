"""Actual synthetic execution must consume the frozen metadata snapshot.

No real acceptance, protected trajectories or test-time result stand-ins.
"""
import pytest

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture, synthetic_plugin
from tests.test_research_admission_chain import fixture_command
from tests.test_research_store import spec


def prepared(tmp_path, *, formal=False, upstream_ids=(), two_arms=False, **upstream):
    store = ResearchStore(tmp_path, "snapshot-admission", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="admitted-fixture", capability="generic-rollout", visibility="synthetic")
    if two_arms:
        value["arms"].append({**value["arms"][0], "arm_id": "candidate", "model_family_id": "candidate"})
        value["cells"].append({**value["cells"][0], "arm_id": "candidate"})
    plugin = synthetic_plugin("admitted-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", fixture_command)
    registry = CapabilityRegistry()
    registry.register(plugin)
    grant = admit_fixture(store, value, plugin, tmp_path, formal=formal, upstream_ids=upstream_ids, **upstream)
    return store, value, registry, grant


@pytest.mark.parametrize("formal", [False, True], ids=["fixture", "formal"])
@pytest.mark.parametrize("upstream_ids", [(), ("affine-4d",)], ids=["no-terrain", "public-affine"])
def test_missing_selected_snapshot_refuses_before_provider_and_worker(tmp_path, formal, upstream_ids):
    store, value, registry, _ = prepared(tmp_path, formal=formal, upstream_ids=upstream_ids)
    # Even a prior ready receipt or accepted catalog cannot replace the frozen
    # spec reference. No downstream package/qualification binding is forged.
    value["admission"].pop("upstream_snapshot_hash", None)
    store.register(value, digest(value))
    error, outcome = None, None
    try:
        outcome = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    except ResearchError as exc:
        error = exc.code
    events = store.events()
    observed = {"error": error, "outcome": outcome,
                "states": [attempt["state"] for attempt in store.attempts().values()],
                "reads": [event for event in events if event["event_kind"] in {"READ_STARTED", "READ_COMPLETED"}],
                "workers": [event for event in events if event["event_kind"] == "WORKER_STARTED"]}
    (tmp_path / "missing-snapshot-observed.json").write_bytes(encode(observed))
    assert error == "MISSING_ARTIFACT", observed
    assert not observed["reads"] and not observed["workers"]


def admission(store, outcome):
    import json
    result = json.loads(store._verified_artifact_content(store.manifest("artifact-" + outcome["artifact_id"])))
    return store.manifest("admission-" + result["admission_hash"])


@pytest.mark.parametrize("formal", [False, True], ids=["fixture", "formal"])
def test_actual_selected_snapshot_and_catalog_are_bound_in_execution(tmp_path, formal):
    store, value, registry, _ = prepared(tmp_path, formal=formal, upstream_ids=("affine-4d",))
    store.register(value, digest(value))
    outcome = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert outcome["state"] == "SUCCEEDED"
    evidence = admission(store, outcome)["documents"]["upstream_snapshot"]
    assert digest(evidence["snapshot"]) == value["admission"]["upstream_snapshot_hash"]
    assert digest(evidence["acceptance_catalog"]) == value["admission"]["upstream_acceptance_hash"]
    assert digest(evidence["validation"]) == evidence["validation_hash"]
    assert evidence["validation"]["data_authorization"] == "none"
    assert evidence["validation"]["cells"][0]["cell_id"] == digest(value["cells"][0])
    assert evidence["validation"]["resolved"][0]["physical_size_bytes"] > 0
    assert evidence["validation"]["resolved"][0]["last_validated_at"]


@pytest.mark.parametrize("mutation,expected", [
    ("missing-file", "MISSING_ARTIFACT"), ("changed-file", "IDENTITY_MISMATCH"),
    ("unaccepted", "UNACCEPTED_VERSION"), ("missing-dependency", "MISSING_ARTIFACT"),
    ("bad-cutover", "IDENTITY_MISMATCH"), ("exposed-selection", "UNACCEPTED_VERSION"),
    ("package-snapshot", "IDENTITY_MISMATCH"), ("prereg-binding", "IDENTITY_MISMATCH"),
    ("root-missing", "MISSING_ARTIFACT")])
def test_actual_snapshot_rejections_precede_all_data_and_worker_reads(tmp_path, mutation, expected):
    from tests.test_shared_upstream_snapshot import snapshot_fixture
    from application.research_preregistration import PreregistrationGate

    path, manifest, accepted = snapshot_fixture(tmp_path, terrain=mutation == "exposed-selection")
    cutover = manifest["studies"][0]["pirc22_cutover"]
    dependencies = None
    if mutation == "unaccepted":
        accepted[0]["status"] = "pending"
    elif mutation == "missing-dependency":
        dependencies = {"affine": ["absent"]}
    elif mutation == "bad-cutover":
        cutover = {"mode": "adopted_primary", "selection_hash": digest("unfrozen conditioner")}
    elif mutation == "exposed-selection":
        import hashlib
        import json
        metadata = json.loads(path.read_bytes())
        metadata["final_eval_read_count"] = 1
        path.write_bytes(encode(metadata))
        for record in (manifest["inputs"][0], accepted[0]["input"]):
            record.update(artifact_hash=hashlib.sha256(path.read_bytes()).hexdigest(), artifact_size_bytes=path.stat().st_size)
            record["metadata_checks"]["final_eval_read_count"] = 1
    if mutation == "prereg-binding":
        original = PreregistrationGate.register_preregistration
        # This is a real malformed frozen operator plan; qualification binds
        # the same actual published plan, never a bypassed runtime result.
        def omit_binding(gate, plan, _expected):
            plan.pop("upstream_bindings")
            return original(gate, plan, digest(plan))
        from unittest.mock import patch
        with patch.object(PreregistrationGate, "register_preregistration", omit_binding):
            store, value, registry, _ = prepared(tmp_path, formal=True, upstream_inputs=manifest["inputs"],
                accepted_versions=accepted, upstream_dependencies=dependencies, pirc22_cutover=cutover)
    else:
        store, value, registry, _ = prepared(tmp_path, formal=True, upstream_inputs=manifest["inputs"],
            accepted_versions=accepted, upstream_dependencies=dependencies, pirc22_cutover=cutover)
    if mutation == "missing-file":
        path.rename(tmp_path / "retained-absent-metadata.json")
    elif mutation == "changed-file":
        path.write_bytes(path.read_bytes().replace(b"accepted-control", b"rejected-control"))
    elif mutation == "package-snapshot":
        package = store.manifest("package-" + value["admission"]["package_hash"])
        package["upstream_snapshot_hash"] = "0" * 64
        value["admission"]["package_hash"] = store.publish("package-" + digest(package), package)
    elif mutation == "root-missing":
        value["admission"].pop("upstream_root")
    store.register(value, digest(value))
    with pytest.raises(ResearchError) as refused:
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert refused.value.code == expected
    assert not any(event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "WORKER_STARTED"} for event in store.events())


def test_missing_dependency_rejects_only_its_actual_cell_without_fallback(tmp_path):
    from tests.test_shared_upstream_snapshot import snapshot_fixture

    path, manifest, accepted = snapshot_fixture(tmp_path)
    store, value, registry, _ = prepared(tmp_path, formal=True, two_arms=True, upstream_inputs=manifest["inputs"],
        accepted_versions=accepted, upstream_dependencies={"affine": ["metadata"], "candidate": []})
    path.rename(tmp_path / "retained-missing-metadata.json")
    store.register(value, digest(value))
    runner = SharedRunner(store, registry)
    with pytest.raises(ResearchError, match="MISSING_ARTIFACT"):
        runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert runner.run_cell("synthetic", digest(value["cells"][1]), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    actual = [event["payload"] for event in store.events() if event["event_kind"] == "UPSTREAM_VALIDATION"]
    assert [row["status"] for row in actual] == ["rejected", "ready"]
    rejected = store.manifest("upstream-validation-" + actual[0]["validation_hash"])
    assert rejected["cells"][0]["upstream_ids"] == ["metadata"]
    assert not any(event["event_kind"] == "WORKER_STARTED" and
                   event["payload"].get("run_id") == store.attempts()[next(iter(store.attempts()))]["run_id"]
                   for event in store.events())


def test_old_ready_validation_cannot_hide_actual_late_metadata_change(tmp_path):
    from experiments.pirc25.snapshot import UpstreamSnapshot
    from tests.test_shared_upstream_snapshot import snapshot_fixture

    path, manifest, accepted = snapshot_fixture(tmp_path)
    store, value, registry, _ = prepared(tmp_path, upstream_inputs=manifest["inputs"], accepted_versions=accepted)
    definition = store.manifest("upstream-snapshot-" + value["admission"]["upstream_snapshot_hash"])
    earlier = UpstreamSnapshot(definition).register(store, root=tmp_path, accepted_versions=accepted)
    assert earlier["cells"][0]["status"] == "ready"
    path.write_bytes(path.read_bytes().replace(b"accepted-control", b"rejected-control"))
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="IDENTITY_MISMATCH"):
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert not any(event["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for event in store.events())


def test_complete_cell_identity_cannot_change_after_snapshot_freeze(tmp_path):
    store, value, registry, _ = prepared(tmp_path, formal=True)
    value["cells"][0]["horizon"] = 99
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="IDENTITY_MISMATCH"):
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert not any(event["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for event in store.events())


def test_new_snapshot_route_does_not_require_a_legacy_public_recipe(tmp_path):
    from tests.test_shared_upstream_snapshot import snapshot_fixture
    path, manifest, accepted = snapshot_fixture(tmp_path)
    store, value, registry, _ = prepared(tmp_path, upstream_inputs=manifest["inputs"], accepted_versions=accepted)
    value["admission"].pop("upstream_ids")
    value["admission"].pop("upstream_hash")
    package = store.manifest("package-" + value["admission"]["package_hash"])
    package.pop("upstream_hash")
    value["admission"]["package_hash"] = store.publish("package-" + digest(package), package)
    store.register(value, digest(value))
    outcome = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert outcome["state"] == "SUCCEEDED"
    documents = admission(store, outcome)["documents"]
    assert "upstream" not in documents
    assert documents["upstream_snapshot"]["validation"]["resolved"][0]["artifact_size_bytes"] == path.stat().st_size


@pytest.mark.parametrize("key", ["upstream_snapshot_hash", "upstream_acceptance_hash"])
def test_actual_upstream_manifest_tamper_is_refused_before_data_reads(tmp_path, key):
    store, value, registry, _ = prepared(tmp_path, formal=True)
    prefix = "upstream-snapshot-" if key == "upstream_snapshot_hash" else "upstream-acceptance-"
    reference = prefix + value["admission"][key]
    source = store.path / "manifests" / (reference + ".json")
    original = store.manifest(reference)
    source.write_bytes(encode({**original, "tampered": True}))
    store.register(value, digest(value))
    with pytest.raises(ResearchError, match="IDENTITY_MISMATCH"):
        SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert not any(event["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for event in store.events())


def test_restricted_upstream_metadata_cannot_be_declassified_by_synthetic_cells(tmp_path):
    from application.research_evidence import export_evidence
    from application.research_query import ResearchQuery
    store, value, registry, grant = prepared(tmp_path, formal=True, upstream_visibility="restricted")
    store.register(value, digest(value))
    outcome = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert outcome["state"] == "SUCCEEDED"
    metadata = store.manifest("artifact-" + outcome["artifact_id"])
    observed = {"result_visibility": metadata["visibility"], "grant_visibilities": grant["visibilities"]}
    actions = {"artifact": lambda: store.read_artifact(outcome["artifact_id"], purpose="preview", authorization=grant),
               "export": lambda: export_evidence(store, "synthetic", grant),
               "query": lambda: ResearchQuery(store, grant["authorization_id"]).list("study")}
    for name, action in actions.items():
        try:
            actual = action()
            observed[name] = {"error": None, "disclosed_bytes": len(actual) if isinstance(actual, bytes) else len(encode(actual))}
        except ResearchError as exc:
            observed[name] = {"error": exc.code}
    (tmp_path / "upstream-visibility-observed.json").write_bytes(encode(observed))
    assert metadata["visibility"] == "restricted", observed
    assert all(observed[name]["error"] == "UNAUTHORIZED_DATA" for name in actions), observed


def test_explicit_restricted_metadata_disclosure_grant_is_not_blocked(tmp_path):
    from application.research_evidence import export_evidence
    from application.research_query import ResearchQuery
    store, value, registry, grant = prepared(tmp_path, formal=True, upstream_visibility="restricted")
    store.register(value, digest(value))
    outcome = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    allowed = {**grant, "authorization_id": "explicit-metadata-visible", "visibilities": ["synthetic", "restricted"]}
    store.authorize(allowed)
    assert store.read_artifact(outcome["artifact_id"], purpose="preview", authorization=allowed)
    bundle = export_evidence(store, "synthetic", allowed)
    assert bundle["visibility"] == "restricted"
    assert bundle["cells"][0]["admission"]["documents"]["upstream_snapshot"]["acceptance_catalog"]["visibility"] == "restricted"
    assert ResearchQuery(store, allowed["authorization_id"]).list("study")["items"]


@pytest.mark.parametrize("source", ["snapshot", "catalog", "input", "accepted-input", "unknown-label"])
def test_upstream_metadata_visibility_is_conservative(source):
    from infrastructure.research_visibility import upstream_metadata_visibility
    snapshot = {"inputs": [{"visibility": "synthetic"}], "visibility": "synthetic"}
    catalog = {"entries": [{"input": {"visibility": "synthetic"}}], "visibility": "synthetic"}
    if source == "snapshot":
        snapshot.pop("visibility")
    elif source == "catalog":
        catalog.pop("visibility")
    elif source == "input":
        snapshot["inputs"][0]["visibility"] = "restricted"
    elif source == "accepted-input":
        catalog["entries"][0]["input"]["visibility"] = "restricted"
    else:
        catalog["visibility"] = {"unknown": "not a disclosure class"}
    assert upstream_metadata_visibility(snapshot, catalog) == "restricted"
