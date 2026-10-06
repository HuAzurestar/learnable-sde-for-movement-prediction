"""Actual authorized HTTP projection of frozen upstream checks, not new reads."""
from contextlib import contextmanager
import json
import threading
import sys
import os
from copy import deepcopy

import pytest

from experiments.pirc25.web import make_server
from application.research_query import ResearchQuery
from infrastructure.research_store import digest, encode
from tests.research_upstream_view_fixtures import recorded_view
from tests.test_research_web import request, service
from tests.test_research_metadata_authorization import permission_clock


@contextmanager
def view_service(root, mutation="missing-file", *, visibility="synthetic"):
    store, spec, grant = recorded_view(root, mutation, visibility=visibility)
    server = make_server(store, grant["authorization_id"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield store, spec, grant, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("mutation,code", [("missing-file", "MISSING_ARTIFACT"),
    ("changed-file", "IDENTITY_MISMATCH"), ("unaccepted", "UNACCEPTED_VERSION")])
def test_actual_refusal_is_visible_only_for_affected_registered_cell(tmp_path, mutation, code):
    with view_service(tmp_path, mutation) as (store, spec, grant, server):
        status, _, content = request(server, "/api/upstream")
        (tmp_path / "upstream-http-observed.json").write_bytes(encode({"status": status, "body": content.decode()}))
        assert status == 200, content
        value = json.loads(content)
        assert value["schema_version"] == "pirc25-upstream-view-v1"
        assert value["study_id"] == spec["study_id"] and value["spec_hash"] == digest(spec)
        assert value["data_authorization"] == "none"
        rows = {row["arm_id"]: row for row in value["items"]}
        assert rows["affine"]["status"] == "REJECTED"
        assert rows["affine"]["upstream_ids"] == ["metadata"]
        assert rows["affine"]["rejected_inputs"] == [{"object_id": "metadata", "code": code}]
        assert rows["candidate"]["status"] == "NOT_CHECKED" and rows["candidate"]["rejected_inputs"] == []
        actual = next(event for event in store.events() if event["event_kind"] == "UPSTREAM_VALIDATION")
        assert rows["affine"]["validation_ref"]["event_hash"] == actual["hash"]
        assert rows["affine"]["validation_ref"]["validation_hash"] == actual["payload"]["validation_hash"]
        assert str(tmp_path) not in content.decode() and '"path"' not in content.decode()
        assert not any(event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "WORKER_STARTED"} for event in store.events())
        assert any(event["event_kind"] == "DISCLOSURE_ALLOWED" for event in store.events())


def test_unchecked_snapshot_does_not_create_ready_claim_or_reopen_missing_input(tmp_path):
    with view_service(tmp_path, "not-checked") as (store, spec, grant, server):
        # Displace the original metadata source after operator registration. The UI
        # must use persisted facts, not invent a fresh filesystem result.
        original = tmp_path / "metadata.json"
        assert original.is_file()
        original.rename(tmp_path / "retained-unread-metadata.json")
        before = [event for event in store.events() if event["event_kind"].startswith("UPSTREAM_")]
        status, _, content = request(server, "/api/upstream")
        assert status == 200, content
        assert all(row["status"] == "NOT_CHECKED" and row["validation_ref"] is None for row in json.loads(content)["items"])
        assert [event for event in store.events() if event["event_kind"].startswith("UPSTREAM_")] == before


def test_legacy_no_snapshot_is_explicit_not_configured_and_ui_view_exists(tmp_path):
    with service(tmp_path) as (_, _, server):
        status, _, content = request(server, "/api/upstream")
        assert status == 200, content
        assert json.loads(content)["items"][0]["status"] == "NOT_CONFIGURED"
        status, _, page = request(server, "/")
        assert status == 200 and b'data-view="upstream"' in page


def test_missing_frozen_reference_is_visible_with_explicit_restricted_permission(tmp_path):
    with view_service(tmp_path, "missing-reference") as (store, spec, grant, server):
        status, _, content = request(server, "/api/upstream")
        assert status == 200, content
        assert all(row["status"] == "REFUSED" and row["rejected_inputs"] == [{"code": "MISSING_ARTIFACT"}]
                   for row in json.loads(content)["items"])
        assert str(tmp_path) not in content.decode()


def test_lost_published_definition_is_integrity_failure_even_with_restricted_permission(tmp_path):
    # Unlike an absent declared input/reference, losing an already published
    # authoritative object is corruption. Never relax source lineage checks.
    with view_service(tmp_path, "missing-document") as (store, spec, grant, server):
        status, _, content = request(server, "/api/upstream")
        assert status == 409 and json.loads(content)["error"]["code"] == "CORRUPT_ARTIFACT", content
        assert b'"items"' not in content and str(tmp_path) not in content.decode()


def test_upstream_query_pages_are_stable_and_refusal_changes_invalidate_cursor(tmp_path):
    with view_service(tmp_path, "not-checked") as (store, spec, grant, server):
        first = ResearchQuery(store, grant["authorization_id"]).upstream(limit=1)
        second = ResearchQuery(store, grant["authorization_id"]).upstream(limit=1, cursor=first["next_cursor"])
        assert first["watermark"] == second["watermark"] and second["next_cursor"] is None
        assert {row["cell_id"] for row in first["items"] + second["items"]} == {digest(cell) for cell in spec["cells"]}
        # Real recorded metadata change: no invented source validation status.
        store.publish("pagination-observation", {"fixture": "new immutable metadata changes watermark"})
        status, _, content = request(server, "/api/upstream?cursor=" + first["next_cursor"])
        assert status == 409 and json.loads(content)["error"]["code"] == "CURSOR_STALE"
        for limit in (0, 201):
            assert request(server, "/api/upstream?limit=" + str(limit))[0] == 409
        assert request(server, "/api/upstream?cursor=invalid")[0] != 200


def test_session_cross_study_and_read_only_boundaries_apply_to_upstream(tmp_path):
    with view_service(tmp_path) as (_, _, _, server):
        assert request(server, "/api/upstream", auth=False)[0] == 401
        assert request(server, "/api/upstream?study_id=another-study")[0] == 403
        assert request(server, "/api/upstream", method="POST")[0] == 405


def test_upstream_query_never_opens_actual_source_or_provider_files(tmp_path):
    with view_service(tmp_path, "not-checked") as (store, spec, grant, server):
        forbidden = {os.path.normcase(os.path.abspath(tmp_path / name)) for name in ("metadata.json", "plugin-input.bin")}
        active = [True]
        touched = []
        def audit(event, args):
            if (active[0] and event == "open" and isinstance(args[0], (str, bytes))
                    and os.path.normcase(os.path.abspath(os.fsdecode(args[0]))) in forbidden):
                touched.append(args[0])
                raise RuntimeError("UI attempted source/provider file open")
        sys.addaudithook(audit)
        try:
            with pytest.raises(RuntimeError):
                sys.audit("open", str(tmp_path / "metadata.json"), "r", 0)
            touched.clear()
            status, _, content = request(server, "/api/upstream")
            assert status == 200 and not touched, content
        finally:
            active[0] = False


def test_fresh_grant_corruption_after_final_physical_cannot_return_upstream(tmp_path, monkeypatch):
    from tests.test_research_metadata_authorization import corrupt_grant_after_final_physical
    with view_service(tmp_path) as (store, spec, grant, server):
        original = ResearchQuery._upstream
        ready = []
        def assemble(query, *args):
            result = original(query, *args)
            ready.append(True)
            return result
        monkeypatch.setattr(ResearchQuery, "_upstream", assemble)
        touched = corrupt_grant_after_final_physical(store, grant, ready, monkeypatch)
        status, _, content = request(server, "/api/upstream")
        assert ready and touched and status == 403, content
        assert json.loads(content)["error"]["code"] == "UNAUTHORIZED_DATA"


def test_journal_failure_does_not_return_upstream_rows(tmp_path, monkeypatch):
    with view_service(tmp_path) as (store, spec, grant, server):
        original = store._append
        def fail(kind, *args):
            if kind == "DISCLOSURE_ALLOWED":
                raise OSError("actual injected metadata journal failure")
            return original(kind, *args)
        monkeypatch.setattr(store, "_append", fail)
        status, _, content = request(server, "/api/upstream")
        assert status != 200 and b'"items"' not in content


def test_restricted_metadata_is_not_declassified_by_refusal_view(tmp_path):
    with view_service(tmp_path, visibility="restricted") as (store, spec, grant, server):
        status, _, content = request(server, "/api/upstream")
        assert status == 403 and json.loads(content)["error"]["code"] == "UNAUTHORIZED_DATA"
        assert b'"items"' not in content


def test_actual_same_id_grant_versions_do_not_select_latest_preview_permission(tmp_path):
    store, spec, grant = recorded_view(tmp_path)
    store.authorize({**grant, "version": "ui-v1"})
    store.authorize({**grant, "version": "ui-v2", "purposes": ["execute", "evaluate"]})
    for version, expected in (("ui-v1", 200), ("ui-v2", 403)):
        server = make_server(store, grant["authorization_id"], authorization_version=version)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, _, content = request(server, "/api/upstream")
            assert status == expected, content
            if expected == 403:
                assert b'"items"' not in content
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_recorded_ready_is_historical_and_query_does_not_reopen_now_missing_source(tmp_path):
    with view_service(tmp_path, "ready") as (store, spec, grant, server):
        reads = [event for event in store.events() if event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "WORKER_STARTED"}]
        assert reads
        (tmp_path / "metadata.json").rename(tmp_path / "retained-after-check.json")
        status, _, content = request(server, "/api/upstream")
        assert status == 200, content
        rows = {row["arm_id"]: row for row in json.loads(content)["items"]}
        assert rows["affine"]["status"] == "READY" and rows["candidate"]["status"] == "NOT_CHECKED"
        assert json.loads(content)["data_authorization"] == "none"
        assert [event for event in store.events() if event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "WORKER_STARTED"}] == reads


def test_actual_final_physical_io_crossing_expiry_refuses_upstream(tmp_path, monkeypatch, permission_clock):
    with view_service(tmp_path) as (store, spec, grant, server):
        original_assemble = ResearchQuery._upstream
        ready = []
        def assemble(query, *args):
            result = original_assemble(query, *args)
            ready.append(True)
            return result
        monkeypatch.setattr(ResearchQuery, "_upstream", assemble)
        original = store._events
        crossed = []
        def physical():
            result = original()
            if ready and store._read_snapshot() is None and not crossed:
                crossed.append(True)
                permission_clock[0] = True
            return result
        monkeypatch.setattr(store, "_events", physical)
        status, _, content = request(server, "/api/upstream")
        assert ready and crossed and status == 403, content
        assert json.loads(content)["error"]["code"] == "UNAUTHORIZED_DATA"


@pytest.mark.parametrize("mutation", ["other-consumer", "untracked-attempt", "other-cell", "metadata-grant", "private-detail"])
def test_resealed_operator_metadata_cannot_replace_actual_consumer_or_leak_details(tmp_path, mutation):
    with view_service(tmp_path) as (store, spec, grant, server):
        original = next(event for event in store.events() if event["event_kind"] == "UPSTREAM_VALIDATION")
        payload = deepcopy(original["payload"])
        validation = store.manifest("upstream-validation-" + payload["validation_hash"])
        # Deliberately owner-publish a new adversarial metadata envelope. The
        # original actual attempt/events/immutable receipt remain unchanged;
        # this is not a mocked runtime success or returned provider outcome.
        if mutation == "other-consumer":
            validation["consumer"]["attempt_id"] = digest("another-consumer")
        elif mutation == "untracked-attempt":
            validation["consumer"]["attempt_id"] = payload["attempt_id"] = digest("not-an-actual-attempt")
        elif mutation == "other-cell":
            validation["cells"][0]["cell_id"] = digest("another-cell")
        elif mutation == "metadata-grant":
            validation["data_authorization"] = "granted"
        else:
            validation["cells"][0]["rejected_inputs"][0]["detail"] = str(tmp_path / "private-coordinate-payload")
            payload["rejected_inputs"] = validation["cells"][0]["rejected_inputs"]
        payload["validation_hash"] = digest(validation)
        store.publish("upstream-validation-" + payload["validation_hash"], validation)
        store.append("UPSTREAM_VALIDATION", payload)
        status, _, content = request(server, "/api/upstream")
        assert str(tmp_path) not in content.decode(), content
        if mutation == "private-detail":
            assert status == 200, content
            assert all("detail" not in error for row in json.loads(content)["items"] for error in row["rejected_inputs"])
        else:
            assert status == 409 and json.loads(content)["error"]["code"] == "CONTRACT_MISMATCH", content
