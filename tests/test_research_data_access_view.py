"""Real protocol/grant/events rendered separately; never open provider bytes."""
from contextlib import contextmanager
import json
import os
import sys
import threading

import pytest

from application.research_data import EvaluationExposureLedger
from application.research_query import ResearchQuery
from experiments.pirc25.web import make_server
from infrastructure.research_store import digest, encode
from tests.test_research_upstream_admission import prepared
from tests.test_research_web import request, service


def access_fixture(root, *, formal=True, version=None, grant_change=None):
    store, spec, _, grant = prepared(root, formal=formal, two_arms=True)
    if version is not None or grant_change:
        grant = {**grant, "authorization_id": "data-scope", **({"version": version} if version else {}),
                 **(grant_change or {})}
        store.authorize(grant)
        spec["admission"].update(authorization_id=grant["authorization_id"])
        if version:
            spec["admission"]["authorization_version"] = version
    store.register(spec, digest(spec))
    preview = {**grant, "authorization_id": "metadata-preview", "purposes": ["preview"],
               "evidence_hash": digest("explicit synthetic metadata-only permission")}
    preview.pop("version", None)
    store.authorize(preview)
    return store, spec, grant, preview


@contextmanager
def access_service(root, **options):
    store, spec, grant, preview = access_fixture(root, **options)
    server = make_server(store, preview["authorization_id"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield store, spec, grant, preview, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def observed(server, root):
    status, _, content = request(server, "/api/data-access")
    (root / "data-access-observed.json").write_bytes(encode({"status": status, "body": content.decode()}))
    assert status == 200, content
    return json.loads(content), content


def test_registered_purpose_grant_and_history_are_independent_columns(tmp_path):
    with access_service(tmp_path) as (store, spec, grant, preview, server):
        value, content = observed(server, tmp_path)
        assert value["schema_version"] == "pirc25-data-access-view-v1"
        assert value["spec_hash"] == digest(spec) and value["permission_decision"] == "not-provided"
        assert len(value["items"]) == 2
        for row in value["items"]:
            assert row["purpose"] == {"split_role": "final-eval", "requested": "evaluate", "fit_scope": False}
            assert row["authorization"]["status"] == "SCOPE_CHECK_PASSED"
            assert row["authorization"]["authorization_id"] == grant["authorization_id"]
            assert row["authorization"]["authorization_hash"] == digest(grant)
            assert row["authorization"]["version"] is None
            assert row["exposure"]["status"] == "NO_RECORDED_EXPOSURE"
            assert row["exposure"]["raw_read_started"] is None and row["exposure"]["result_disclosure_started"] is None
            assert row["exposure"]["external_history_hash"] == store.manifest("protocol-inputs")["history_hash"]
        assert str(tmp_path) not in content.decode() and '"path"' not in content.decode()
        assert not any(e["event_kind"].startswith("READ_") for e in store.events())
        assert any(e["event_kind"] == "DISCLOSURE_ALLOWED" for e in store.events())


def test_execute_permission_without_test_authorization_is_visible_as_refused(tmp_path):
    with access_service(tmp_path, version="v1", grant_change={"test_authorization": False}) as (_, _, grant, _, server):
        value, _ = observed(server, tmp_path)
        for row in value["items"]:
            assert row["purpose"]["split_role"] == "final-eval"
            assert row["authorization"]["status"] == "SCOPE_CHECK_DENIED"
            assert row["authorization"]["reason_codes"] == ["UNAUTHORIZED_DATA"]
            assert row["authorization"]["version"] == "v1"
            assert row["exposure"]["status"] == "NO_RECORDED_EXPOSURE"


def test_actual_raw_read_and_result_disclosure_have_distinct_original_event_refs(tmp_path):
    with access_service(tmp_path) as (store, spec, grant, preview, server):
        ledger = EvaluationExposureLedger(store)
        assert ledger.read("inputs", "fixture-1", purpose="evaluate", authorization_id=grant["authorization_id"],
                           data_root=tmp_path) == b"explicit synthetic plugin input"
        before = store.events()
        raw_start = next(e for e in before if e["event_kind"] == "READ_STARTED")
        first, _ = observed(server, tmp_path)
        assert all(r["exposure"]["status"] == "EXPOSED" and r["exposure"]["result_disclosure_started"] is None
                   for r in first["items"])
        artifact = store.artifact(b"synthetic visible result", role="result", visibility="synthetic",
                                  block_ids=["fixture-1"], study_id=spec["study_id"])
        assert store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=preview) == b"synthetic visible result"
        original_reads = [e for e in store.events() if e["event_kind"].startswith("READ_")]
        result_start = next(e for e in original_reads if e["event_kind"] == "READ_STARTED" and e["payload"].get("artifact_id"))
        second, _ = observed(server, tmp_path)
        for row in second["items"]:
            assert row["exposure"]["raw_read_started"] == {k: raw_start[k] for k in ("event_id", "sequence", "hash", "created_at")}
            assert row["exposure"]["result_disclosure_started"] == {k: result_start[k] for k in ("event_id", "sequence", "hash", "created_at")}
        assert [e for e in store.events() if e["event_kind"].startswith("READ_")] == original_reads


def test_view_never_opens_provider_or_result_content(tmp_path):
    with access_service(tmp_path) as (store, _, _, _, server):
        forbidden = {os.path.normcase(os.path.abspath(tmp_path / "plugin-input.bin"))}
        forbidden.update(os.path.normcase(os.path.abspath(p)) for p in (store.path / "artifacts").iterdir())
        active, touched = [True], []
        def audit(event, args):
            if (active[0] and event == "open" and isinstance(args[0], (str, bytes))
                    and os.path.normcase(os.path.abspath(os.fsdecode(args[0]))) in forbidden):
                touched.append(args[0])
                raise RuntimeError("metadata view opened provider/artifact bytes")
        sys.addaudithook(audit)
        try:
            with pytest.raises(RuntimeError):
                sys.audit("open", str(tmp_path / "plugin-input.bin"), "r", 0)
            touched.clear()
            observed(server, tmp_path)
            assert not touched
        finally:
            active[0] = False


def test_legacy_no_protocol_is_unknown_not_untouched(tmp_path):
    with service(tmp_path) as (_, _, server):
        value, _ = observed(server, tmp_path)
        assert value["items"][0]["purpose"] is None
        assert value["items"][0]["authorization"]["status"] == "NOT_CONFIGURED"
        assert value["items"][0]["exposure"]["status"] == "UNKNOWN"


def test_actual_session_and_read_only_boundaries(tmp_path):
    with access_service(tmp_path) as (_, _, _, _, server):
        assert request(server, "/api/data-access", auth=False)[0] == 401
        assert request(server, "/api/data-access?study_id=foreign")[0] == 403
        assert request(server, "/api/data-access", method="POST")[0] == 405
        value, _ = observed(server, tmp_path)
        assert value["items"]
