"""Read-only loopback API, browser boundary and immutable export checks."""

from contextlib import contextmanager
import http.client
import json
import threading
import base64

import pytest

from application.research_query import ResearchQuery
from experiments.pirc25.web import make_server
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_research_store import spec


@contextmanager
def service(tmp_path):
    store = ResearchStore(tmp_path, "web", initialize=True)
    value = spec()
    value["cells"][0]["visibility"] = "synthetic"
    store.register(value, digest(value))
    grant = {"authorization_id": "ui", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
             "evidence_hash": digest("fixture permission"), "purposes": ["preview", "export"],
             "visibilities": ["synthetic"], "block_ids": ["fixture-1"]}
    store.authorize(grant)
    server = make_server(store, "ui")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield store, value, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(server, path, *, method="GET", auth=True, extra=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    headers = {"X-Session-Token": server.session_token} if auth else {}
    headers.update(extra or {})
    try:
        connection.request(method, path, headers=headers)
        result = connection.getresponse()
        return result.status, dict(result.getheaders()), result.read()
    finally:
        connection.close()


def test_session_origin_host_and_mutation_boundaries(tmp_path):
    with service(tmp_path) as (store, value, server):
        assert server.server_address[0] == "127.0.0.1"
        assert request(server, "/api/studies", auth=False)[0] == 401
        assert request(server, "/api/studies", extra={"Origin": "https://untrusted.example"})[0] == 403
        assert request(server, "/api/studies", extra={"Host": "untrusted.example"})[0] == 403
        assert request(server, "/api/runs", method="POST")[0] == 405
        assert request(server, "/api/runs?study_id=another-study")[0] == 403
        assert request(server, "/api/artifacts/..%2Fstore.json")[0] != 200
        status, headers, body = request(server, "/")
        assert status == 200 and b"Research results" in body
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_empty_runs_and_failed_attempt_trace_are_visible(tmp_path):
    with service(tmp_path) as (store, value, server):
        status, _, content = request(server, "/api/runs")
        assert status == 200 and json.loads(content)["items"][0]["state"] == "MISSING"
        assert json.loads(content)["items"][0]["expected_cell"] is True
        run = store.register_run("synthetic", value["cells"][0])
        attempt = store.new_attempt(run)
        store.transition(attempt, "TIMEOUT", error_code="TIMEOUT")
        status, _, content = request(server, "/api/runs?state=TIMEOUT")
        assert status == 200
        assert json.loads(content)["items"][0]["state"] == "TIMEOUT"
        status, _, trace = request(server, "/api/runs/" + run)
        assert status == 200 and json.loads(trace)["attempts"][0]["error_code"] == "TIMEOUT"


def test_csp_allows_verified_blob_images_without_blob_scripts_or_frames(tmp_path):
    with service(tmp_path) as (_, _, server):
        status, headers, _ = request(server, '/')
        assert status == 200
        directives = {parts[0]: parts[1:] for directive in headers['Content-Security-Policy'].split(';')
                      if (parts := directive.split())}
        assert directives.get('img-src') == ["'self'", 'blob:']
        assert directives['script-src'] == ["'self'"]
        assert directives['frame-ancestors'] == ["'none'"]


def test_csv_is_exact_frozen_artifact_and_corruption_fails_closed(tmp_path):
    with service(tmp_path) as (store, value, server):
        data = b"aggregate_hash,metric,value,unit\nv1,error,2.5,m\n"
        artifact = store.artifact(data, role="table", visibility="synthetic", block_ids=["fixture-1"],
                                  study_id="synthetic", media_type="text/csv")
        status, headers, content = request(server, "/api/artifacts/" + artifact["artifact_id"] + "?download=1")
        assert status == 200 and content == data
        assert headers["Content-Disposition"].startswith("attachment")
        assert any(e["event_kind"] == "READ_COMPLETED" for e in store.events())
        (store.path / "artifacts" / artifact["artifact_id"]).write_bytes(b"broken")
        status, _, content = request(server, "/api/artifacts/" + artifact["artifact_id"])
        assert status == 409 and json.loads(content)["error"]["code"] == "CORRUPT_ARTIFACT"


def test_result_manifest_requires_export_permission(tmp_path):
    with service(tmp_path) as (store, value, server):
        grant = {**store.manifest("authorization-ui"), "authorization_id": "preview-only", "purposes": ["preview"]}
        store.authorize(grant)
        artifact = store.artifact(b"{}", role="result", visibility="synthetic", block_ids=["fixture-1"], study_id="synthetic")
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            ResearchQuery(store, "preview-only").result_manifest(artifact["artifact_id"])


def test_run_trace_preserves_spec_hashes_and_checkpoint_references(tmp_path):
    with service(tmp_path) as (store, value, server):
        run = store.register_run("synthetic", value["cells"][0])
        attempt = store.new_attempt(run)
        checkpoint = store.artifact(b"{}", role="checkpoint", visibility="synthetic", block_ids=["fixture-1"], study_id="synthetic")
        store.append("CHECKPOINT", {"attempt_id": attempt, "artifact_id": checkpoint["artifact_id"], "resume_level": "exact"})
        status, _, content = request(server, "/api/runs/" + run)
        assert status == 200
        trace = json.loads(content)
        assert trace["provenance"]["code_hash"] == value["code_hash"]
        assert trace["provenance"]["data_hash"] == value["data_hash"]
        assert trace["provenance"]["schema_version"] == value["schema_version"]
        assert trace["checkpoints"][0]["artifact_id"] == checkpoint["artifact_id"]
        assert trace["paper_evidence"] == []


def test_result_manifest_export_has_provenance_without_trajectory_payload(tmp_path):
    with service(tmp_path) as (store, value, server):
        result = {"spec_hash": digest(value), "protocol_hash": value["protocol_hash"],
                  "forecast": {"samples": [[[1, 2]]]}, "metrics": {"error": 1}}
        artifact = store.artifact(json.dumps(result).encode(), role="result", visibility="synthetic",
                                  block_ids=["fixture-1"], study_id="synthetic")
        status, headers, content = request(server, "/api/artifacts/" + artifact["artifact_id"] + "?manifest=1")
        assert status == 200
        manifest = json.loads(content)
        assert manifest["artifact"] == artifact
        assert manifest["bindings"]["spec_hash"] == digest(value)
        assert "forecast" not in manifest and "samples" not in content.decode()
        assert headers["Content-Disposition"].startswith("attachment")


def test_preview_bounds_and_unauthorized_visibility_are_explicit(tmp_path):
    with service(tmp_path) as (store, value, server):
        artifact = store.artifact(json.dumps({"forecast": {"samples": [[[1, 2]]] * 65}}).encode(), role="result",
                                  visibility="synthetic", block_ids=["fixture-1"], study_id="synthetic")
        status, _, content = request(server, "/api/artifacts/" + artifact["artifact_id"])
        assert status == 409 and json.loads(content)["error"]["code"] == "TOO_LARGE"
        secret = store.artifact(b"{}", role="result", visibility="restricted", block_ids=["fixture-1"], study_id="synthetic")
        assert request(server, "/api/artifacts/" + secret["artifact_id"])[0] == 403


def test_comparison_manifests_require_all_artifacts_in_preview_scope(tmp_path):
    with service(tmp_path) as (store, value, server):
        run = store.register_run("synthetic", value["cells"][0])
        packages = {}
        for hidden_key in (None, "aggregate_id", "table", "evidence-index", "block", "study"):
            name = hidden_key or "visible"
            package_hash = digest(name)
            attachments = {}
            for key, role in (("aggregate_id", "aggregate"), ("table", "table"),
                              ("evidence-index", "evidence-index")):
                hidden = key == hidden_key or (hidden_key in {"block", "study"} and key == "aggregate_id")
                artifact = store.artifact(encode({"package": name, "role": role}), role=role,
                                          visibility="restricted" if hidden and hidden_key in {"aggregate_id", "table", "evidence-index"} else "synthetic",
                                          block_ids=["other-block"] if hidden_key == "block" and hidden else ["fixture-1"],
                                          study_id="other-study" if hidden_key == "study" and hidden else "synthetic")
                attachments[key] = artifact["artifact_id"]
            package = {"study_id": "synthetic", "aggregate_hash": package_hash, **attachments}
            store.publish("comparison-" + package_hash, package)
            packages[name] = package

        query = ResearchQuery(store, "ui")
        assert [row["manifest"] for row in query.list("comparison")["items"]] == [packages["visible"]]
        assert query.run(run)["paper_evidence"] == [packages["visible"]]
        assert query.comparison(packages["visible"]["aggregate_hash"])["package"] == packages["visible"]
        for name, package in packages.items():
            if name != "visible":
                with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
                    query.comparison(package["aggregate_hash"])

        status, _, content = request(server, "/api/comparisons")
        assert status == 200
        assert [row["manifest"] for row in json.loads(content)["items"]] == [packages["visible"]]
        status, _, content = request(server, "/api/runs/" + run)
        assert status == 200 and json.loads(content)["paper_evidence"] == [packages["visible"]]

        broad = {**store.manifest("authorization-ui"), "authorization_id": "ui-broad",
                 "visibilities": ["synthetic", "restricted"]}
        store.authorize(broad)
        visible = [row["manifest"] for row in ResearchQuery(store, "ui-broad").list("comparison")["items"]]
        assert {package["aggregate_hash"] for package in visible} == {
            packages[name]["aggregate_hash"] for name in ("visible", "aggregate_id", "table", "evidence-index")}


def test_query_page_limits_and_filter_bound_cursor(tmp_path):
    with service(tmp_path) as (store, value, server):
        query = ResearchQuery(store, "ui")
        with pytest.raises(ResearchError):
            query.list("run", limit=201)
        with pytest.raises(ResearchError, match="CURSOR_STALE"):
            query.list("run", cursor="not-a-valid-cursor")
        with pytest.raises(ResearchError, match="CURSOR_STALE"):
            query.list("run", cursor="W10=")


def test_query_rejects_authority_change_during_snapshot(tmp_path, monkeypatch):
    with service(tmp_path) as (store, value, server):
        query = ResearchQuery(store, "ui")
        original = query._objects
        def publish_during_enumeration(kind, grant):
            rows = original(kind, grant)
            store.register_run("synthetic", value["cells"][0])
            return rows
        monkeypatch.setattr(query, "_objects", publish_during_enumeration)
        with pytest.raises(ResearchError, match="INDEX_STALE"):
            query.list("run")
        monkeypatch.setattr(query, "_objects", original)
        assert query.list("run")["items"][0]["state"] == "REGISTERED"


def test_stable_multi_page_cursor_and_invalid_position(tmp_path):
    store = ResearchStore(tmp_path, "pagination", initialize=True)
    value = spec()
    value["cells"] = [{**value["cells"][0], "seed": seed, "visibility": "synthetic"} for seed in range(3)]
    store.register(value, digest(value))
    store.authorize({"authorization_id": "pages", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
                     "evidence_hash": digest("synthetic pagination grant"), "purposes": ["preview"],
                     "visibilities": ["synthetic"], "block_ids": ["fixture-1"]})
    query = ResearchQuery(store, "pages")
    first = query.list("run", limit=1)
    second = query.list("run", limit=1, cursor=first["next_cursor"])
    third = query.list("run", limit=1, cursor=second["next_cursor"])
    assert len({p["items"][0]["object_id"] for p in (first, second, third)}) == 3
    assert first["watermark"] == second["watermark"] == third["watermark"]
    assert third["next_cursor"] is None
    cursor = json.loads(base64.urlsafe_b64decode(first["next_cursor"]))
    cursor["after"] = None
    with pytest.raises(ResearchError, match="CURSOR_STALE"):
        query.list("run", cursor=base64.urlsafe_b64encode(json.dumps(cursor).encode()).decode())
    store.register_run("synthetic", value["cells"][0])
    with pytest.raises(ResearchError, match="CURSOR_STALE"):
        query.list("run", cursor=first["next_cursor"])


@pytest.mark.parametrize("selector", ["model=unregistered", "version=unregistered", "horizon=99", "seed=99",
                                     "trainer=unregistered", "predictor=unregistered"])
def test_ui_filters_do_not_silently_return_unfiltered_matrix(tmp_path, selector):
    with service(tmp_path) as (_, _, server):
        status, _, content = request(server, "/api/runs?" + selector)
        assert status == 200
        assert json.loads(content)["items"] == []


def test_ui_budget_provenance_distinguishes_unknown_cost_from_zero(tmp_path):
    from application.research_budget import BudgetLedger, BudgetSpec
    with service(tmp_path) as (store, value, server):
        run = store.register_run("synthetic", value["cells"][0])
        attempt = store.new_attempt(run)
        ledger = BudgetLedger(store)
        reservation = ledger.reserve(attempt, BudgetSpec(10))
        ledger.settle(reservation["reservation_id"], None, outcome="INTERRUPTED")
        status, _, content = request(server, "/api/runs")
        assert status == 200
        budget = json.loads(content)["items"][0]["budget"]
        assert budget["unit"] == "slot-ms" and budget["committed_ms"] == 10000
        assert budget["remaining_ms"] == 86400000 - 10000 and budget["closed"] is True
        source = budget["sources"][0]
        assert source["reservation_id"] == reservation["reservation_id"]
        assert source["monotonic_elapsed_ms"] is None and source["charged_ms"] == 10000
        assert source["cost_basis"] == "unknown-conservative-reservation"
        assert source["event_hash"] in {event["hash"] for event in store.events()}


def test_ui_filter_matches_and_budget_changes_invalidate_pagination(tmp_path):
    from application.research_budget import BudgetLedger, BudgetSpec
    store = ResearchStore(tmp_path, "filters", initialize=True)
    value = spec()
    value["arms"][0]["trainer_id"] = "fixture-trainer"
    value["cells"] = [{**value["cells"][0], "seed": seed, "horizon": horizon,
                       "plugin_id": "fixture-predictor", "visibility": "synthetic"}
                      for horizon in (1, 2) for seed in (1, 2)]
    store.register(value, digest(value))
    store.authorize({"authorization_id": "ui", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
                     "evidence_hash": digest("synthetic filters grant"), "purposes": ["preview"],
                     "visibilities": ["synthetic"], "block_ids": ["fixture-1"]})
    query = ResearchQuery(store, "ui")
    selected = query.list("run", model="affine", version=digest(value), horizon="2", seed="1",
                          trainer="fixture-trainer", predictor="fixture-predictor", state="MISSING")
    assert len(selected["items"]) == 1
    assert selected["items"][0]["comparison_dimensions"] == {"horizon": 2}
    run = store.register_run("synthetic", value["cells"][0])
    attempt = store.new_attempt(run)
    first = query.list("run", limit=1)
    with pytest.raises(ResearchError, match="CURSOR_STALE"):
        query.list("run", limit=1, cursor=first["next_cursor"], horizon="2")
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempt, BudgetSpec(10))
    with pytest.raises(ResearchError, match="CURSOR_STALE"):
        query.list("run", limit=1, cursor=first["next_cursor"])
    reserved = query.run(run)["budget"]["sources"][0]
    assert reserved["cost_basis"] == "reservation" and reserved["settled"] is False
    second = query.list("run", limit=1)
    ledger.settle(reservation["reservation_id"], 123, outcome="SUCCEEDED")
    with pytest.raises(ResearchError, match="CURSOR_STALE"):
        query.list("run", limit=1, cursor=second["next_cursor"])
    measured = query.run(run)["budget"]["sources"][0]
    assert measured["cost_basis"] == "measured-monotonic" and measured["charged_ms"] == 123
