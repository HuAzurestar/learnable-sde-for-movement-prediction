"""Actual authorized HTTP projection of frozen upstream checks, not new reads."""
from contextlib import contextmanager
import json
import threading

import pytest

from experiments.pirc25.web import make_server
from infrastructure.research_store import digest, encode
from tests.research_upstream_view_fixtures import recorded_view
from tests.test_research_web import request, service


@contextmanager
def view_service(root, mutation="missing-file"):
    store, spec, grant = recorded_view(root, mutation)
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
