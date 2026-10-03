"""Select immutable authorization versions without a moving latest grant."""

import pytest

from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_research_artifact_read_bounds import published
from tests.test_research_read_authorization import provider_source
from tests.test_research_store import spec


def test_same_id_versions_coexist_and_reopen_without_changing_old_grant(tmp_path):
    store, artifact, legacy = published(tmp_path)
    first = {**legacy, "version": "v1"}
    second = {**legacy, "version": "v2", "purposes": ["resume"]}
    store.authorize(first)
    store.authorize(second)
    reopened = ResearchStore(tmp_path, "read-bounds")
    assert reopened.authorization(legacy["authorization_id"], version="v1") == first
    assert reopened.authorization(legacy["authorization_id"], version="v2") == second
    assert reopened.authorization(legacy["authorization_id"]) == legacy
    assert reopened.read_artifact(artifact["artifact_id"], purpose="preview", authorization=first) == b'{"value":2}'
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        reopened.read_artifact(artifact["artifact_id"], purpose="preview", authorization=second)
    assert reopened.events()[-1]["event_kind"] == "EXPOSURE_DENIED"


def test_same_version_is_idempotent_but_changed_content_conflicts(tmp_path):
    store, _, grant = published(tmp_path)
    grant = {**grant, "version": "v1"}
    store.authorize(grant)
    before = store.events()
    store.authorize(grant)
    assert store.events() == before
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        store.authorize({**grant, "purposes": ["resume"]})
    assert store.authorization(grant["authorization_id"], version="v1") == grant


@pytest.mark.parametrize("version", [None, "", 1, True, [], "../v1"])
def test_invalid_explicit_version_is_rejected_without_publication(tmp_path, version):
    store, _, grant = published(tmp_path)
    before = store.events()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        store.authorize({**grant, "authorization_id": "new", "version": version})
    assert store.events() == before


def test_missing_selected_version_never_falls_back_to_legacy_or_latest(tmp_path):
    store, _, legacy = published(tmp_path)
    grant = {**legacy, "authorization_id": "version-only", "version": "v2"}
    store.authorize(grant)
    for version in (None, "v1", "v3"):
        with pytest.raises(ResearchError, match="MISSING_INPUT"):
            store.authorization("version-only", version=version)


def test_provider_selects_exact_version_and_journals_its_binding(tmp_path):
    store, ledger, legacy, content = provider_source(tmp_path, "fit")
    first = {**legacy, "version": "v1", "purposes": ["validate"]}
    second = {**legacy, "version": "v2"}
    store.authorize(first)
    store.authorize(second)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ledger.read("reserved", "test-block", purpose="fit", authorization_id=legacy["authorization_id"],
            authorization_version="v1", data_root=tmp_path)
    assert ledger.read("reserved", "test-block", purpose="fit", authorization_id=legacy["authorization_id"],
        authorization_version="v2", data_root=tmp_path) == content
    request = store.events()[-1]["payload"]
    assert request["authorization_version"] == "v2"
    assert request["authorization_hash"] == digest(second)


def test_query_is_bound_to_selected_version_without_changing_legacy(tmp_path):
    store, _, legacy = published(tmp_path)
    value = spec()
    value["study_id"] = legacy["study_id"]
    value["cells"][0]["visibility"] = "synthetic"
    value["cells"][0]["block_id"] = "b"
    store.register(value, digest(value))
    first = {**legacy, "version": "v1", "purposes": ["resume"]}
    second = {**legacy, "version": "v2"}
    store.authorize(first)
    store.authorize(second)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ResearchQuery(store, legacy["authorization_id"], authorization_version="v1").list("study")
    query = ResearchQuery(store, legacy["authorization_id"], authorization_version="v2")
    assert query.list("study")["items"]
    assert query._grant() == second
    assert ResearchQuery(store, legacy["authorization_id"])._grant() == legacy


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_actual_admission_uses_selected_execution_and_input_version(tmp_path, version):
    from application.research_admission import AdmissionGate
    from tests.test_research_admission_chain import prepared

    store, value, registry, _ = prepared(tmp_path, formal=False)
    legacy = store.authorization(value["admission"]["authorization_id"])
    store.authorize({**legacy, "version": "v1", "purposes": ["preview"]})
    store.authorize({**legacy, "version": "v2"})
    value["admission"]["authorization_version"] = version
    store.register(value, digest(value))
    cell = value["cells"][0]
    attempt = store.new_attempt(store.register_run(value["study_id"], cell))
    binding = cell["execution"]
    plugin = registry.resolve(cell["plugin_id"], cell["capability"],
        version=binding["component_version"], entry_hash=binding["registry_entry_hash"])
    if version == "v1":
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            AdmissionGate(store).prepare(value, cell, plugin, attempt)
        assert not any(event["event_kind"] == "READ_STARTED" for event in store.events())
    else:
        receipt = AdmissionGate(store).prepare(value, cell, plugin, attempt)
        assert receipt["documents"]["authorization"]["version"] == "v2"
        assert receipt["input_evidence"][-1]["payload"]["authorization_version"] == "v2"


def test_actual_loopback_session_cannot_switch_to_other_version_in_query(tmp_path):
    import threading
    from experiments.pirc25.web import make_server
    from tests.test_research_web import service, request

    with service(tmp_path) as (store, _, _):
        legacy = store.authorization("ui")
        store.authorize({**legacy, "version": "v1", "purposes": ["resume"]})
        store.authorize({**legacy, "version": "v2"})
        for version, status in (("v1", 403), ("v2", 200)):
            server = make_server(store, "ui", authorization_version=version)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                assert request(server, "/api/studies?study_id=synthetic&authorization_version=v2")[0] == status
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


def test_independent_cli_export_requires_exact_version(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    store, _, legacy = published(tmp_path)
    value = spec()
    value["cells"][0].update(block_id="b", visibility="synthetic")
    store.register(value, digest(value))
    store.authorize({**legacy, "version": "v1"})
    store.authorize({**legacy, "version": "v2", "purposes": ["export"]})
    for version, status in (("v1", 1), ("v2", 0)):
        output = tmp_path / (version + "-bundle.json")
        result = subprocess.run([sys.executable, "-B", "-m", "experiments.pirc25", "--root", str(tmp_path),
            "--store-id", store.store_id, "export", value["study_id"], "--authorization-id", legacy["authorization_id"],
            "--authorization-version", version, "--output", str(output)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        assert result.returncode == status, result.stdout + result.stderr
        assert output.exists() == (status == 0)
        if status == 0:
            assert json.loads(output.read_bytes())["study_id"] == value["study_id"]


def test_actual_formal_receipt_keeps_version_binding_in_independent_paper_validation(tmp_path):
    from copy import deepcopy
    import json
    from pathlib import Path
    import subprocess
    import sys
    from application.research_budget import BudgetSpec
    from application.research_evidence import export_evidence
    from experiments.pirc25.runner import SharedRunner
    from infrastructure.research_store import encode
    from tests.test_research_admission_chain import prepared

    store, value, registry, legacy = prepared(tmp_path, formal=True, two_arms=True)
    grant = {**legacy, "version": "v2"}
    store.authorize(grant)
    value["admission"]["authorization_version"] = "v2"
    store.register(value, digest(value))
    for cell in value["cells"]:
        assert SharedRunner(store, registry).run_cell(value["study_id"], digest(cell), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    bundle = export_evidence(store, value["study_id"], grant)
    paper = Path(__file__).resolve().parents[2] / "TSDE-SDE"
    source = tmp_path / "paper-input.json"
    code = ("import sys,json;sys.path.insert(0,sys.argv[1]);"
        "from scripts.pirc25.admission import validate_admission;"
        "b=json.load(open(sys.argv[2],encoding='utf-8'));"
        "[validate_admission(b,r) for r in b['cells']]")
    source.write_bytes(encode(bundle))
    result = subprocess.run([sys.executable, "-B", "-c", code, str(paper), str(source)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

    substituted = deepcopy(bundle)
    changed_spec = substituted["registered_spec"]
    changed_spec["admission"]["authorization_version"] = "v1"
    substituted["spec_hash"] = digest(changed_spec)
    for row in substituted["cells"]:
        receipt = row["admission"]
        receipt.update(spec=changed_spec, spec_hash=digest(changed_spec))
        receipt["documents"]["authorization"]["version"] = "v1"
        receipt["admission_hash"] = digest({k: v for k, v in receipt.items() if k != "admission_hash"})
        row["admission_hash"] = receipt["admission_hash"]
    substituted["bundle_hash"] = digest({k: v for k, v in substituted.items() if k != "bundle_hash"})
    source.write_bytes(encode(substituted))
    result = subprocess.run([sys.executable, "-B", "-c", code, str(paper), str(source)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "input authorization version/hash" in result.stderr
