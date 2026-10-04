"""Real protocol/grant/events rendered separately; never open provider bytes."""
from contextlib import contextmanager
import json
import os
import sys
import threading

import pytest

from application.research_data import EvaluationExposureLedger
from application.research_preregistration import PreregistrationGate, source_identity
from application.research_query import ResearchQuery
from experiments.pirc25.web import make_server
from infrastructure.research_store import digest, encode
from infrastructure.research_store import ResearchError
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


@pytest.mark.parametrize("status", ["unknown", "exposed"])
def test_shared_source_imported_history_is_inherited_without_private_foreign_payload(tmp_path, status):
    with access_service(tmp_path) as (store, _, _, _, server):
        block = store.manifest("protocol-inputs")["blocks"][0]
        record = {**source_identity(block), "dataset_id": "foreign-renamed-dataset", "source_block_id": "foreign-window", "status": status}
        evidence = {"records": [record], "private_note": "PRIVATE FOREIGN SOURCE DETAILS"}
        report = {"schema_version": "pirc25-exposure-history-v1", "source": "FOREIGN PRIVATE TOOL",
                  "source_evidence": evidence, "source_evidence_hash": digest(evidence)}
        PreregistrationGate(store).import_history(report, digest(report))
        value, content = observed(server, tmp_path)
        assert all(row["exposure"]["status"] == status.upper() for row in value["items"])
        for secret in ("FOREIGN PRIVATE TOOL", "PRIVATE FOREIGN SOURCE DETAILS", "foreign-renamed-dataset", "foreign-window"):
            assert secret not in content.decode()


def foreign_read(store, root, *, artifact=False):
    source = store.manifest("protocol-inputs")["blocks"][0]
    block = {**source, "block_id": "foreign-window", "source_block_id": "foreign-window", "dataset_id": "foreign-dataset",
             "split_role": "train", "fit_scope": True}
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "foreign-inputs",
                "study_id": "private-foreign-study", "blocks": [block]}
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    grant = {"authorization_id": "foreign", "study_id": protocol["study_id"], "purposes": ["fit", "preview"],
             "protocol_hash": digest(protocol), "block_ids": [block["block_id"]], "visibilities": ["synthetic"],
             "evidence_hash": digest("explicit foreign synthetic permission"), "expires_at": "2099-01-01T00:00:00+00:00"}
    store.authorize(grant)
    if artifact:
        metadata = store.artifact(b"FOREIGN PRIVATE RESULT BYTES", role="result", visibility="synthetic",
                                  block_ids=[block["block_id"]], study_id=protocol["study_id"])
        assert store.read_artifact(metadata["artifact_id"], purpose="preview", authorization=grant) == b"FOREIGN PRIVATE RESULT BYTES"
    else:
        assert ledger.read(protocol["protocol_id"], block["block_id"], purpose="fit",
                           authorization_id=grant["authorization_id"], data_root=root) == b"explicit synthetic plugin input"
    return next(e for e in store.events() if e["event_kind"] == "READ_STARTED")


@pytest.mark.parametrize("artifact", [False, True], ids=["raw-alias", "result-alias"])
def test_actual_foreign_reads_follow_content_alias_without_disclosing_foreign_study(tmp_path, artifact):
    with access_service(tmp_path) as (store, _, _, _, server):
        event = foreign_read(store, tmp_path, artifact=artifact)
        value, content = observed(server, tmp_path)
        key = "result_disclosure_started" if artifact else "raw_read_started"
        assert all(r["exposure"]["status"] == "EXPOSED" and r["exposure"][key]["hash"] == event["hash"] for r in value["items"])
        assert "private-foreign-study" not in content.decode() and "FOREIGN PRIVATE RESULT BYTES" not in content.decode()


def test_actual_failed_provider_read_retains_possible_exposure_without_completion(tmp_path):
    with access_service(tmp_path) as (store, _, grant, _, server):
        (tmp_path / "plugin-input.bin").rename(tmp_path / "retained-unavailable-provider.bin")
        with pytest.raises(OSError):
            EvaluationExposureLedger(store).read("inputs", "fixture-1", purpose="evaluate",
                                                 authorization_id=grant["authorization_id"], data_root=tmp_path)
        value, _ = observed(server, tmp_path)
        for row in value["items"]:
            assert row["exposure"]["status"] == "EXPOSED"
            assert row["exposure"]["raw_read_started"] is not None
            assert row["exposure"]["raw_read_completed"] is None and row["exposure"]["read_failed_count"] == 1


def test_current_data_grant_is_exact_version_not_latest_broader_scope(tmp_path):
    with access_service(tmp_path, version="v1", grant_change={"test_authorization": False}) as (store, _, grant, _, server):
        store.authorize({**grant, "version": "v2", "test_authorization": True})
        value, _ = observed(server, tmp_path)
        assert all(r["authorization"]["version"] == "v1" and r["authorization"]["status"] == "SCOPE_CHECK_DENIED"
                   for r in value["items"])


def test_paged_view_does_not_invalidate_itself_but_real_read_invalidates_cursor(tmp_path):
    with access_service(tmp_path) as (store, _, grant, preview, server):
        query = ResearchQuery(store, preview["authorization_id"])
        first = query.data_access(limit=1)
        assert first["next_cursor"]
        second = query.data_access(limit=1, cursor=first["next_cursor"])
        assert len(second["items"]) == 1 and second["next_cursor"] is None
        assert first["watermark"] == second["watermark"]
        EvaluationExposureLedger(store).read("inputs", "fixture-1", purpose="evaluate",
                                             authorization_id=grant["authorization_id"], data_root=tmp_path)
        with pytest.raises(ResearchError) as stale:
            query.data_access(limit=1, cursor=first["next_cursor"])
        assert stale.value.code == "CURSOR_STALE"


@pytest.mark.parametrize("kind", ["data", "preview"])
def test_changed_immutable_grant_cannot_return_old_rows(tmp_path, kind):
    with access_service(tmp_path) as (store, _, grant, preview, server):
        observed(server, tmp_path)
        target = grant if kind == "data" else preview
        path = store.path / "manifests" / ("authorization-" + target["authorization_id"] + ".json")
        path.write_bytes(encode({**target, "purposes": []}))
        status, _, content = request(server, "/api/data-access")
        assert status != 200 and b'"items"' not in content
        assert json.loads(content)["error"]["code"] == "CORRUPT_ARTIFACT"


def test_journal_failure_is_fail_closed_not_unaudited_rows(tmp_path, monkeypatch):
    with access_service(tmp_path) as (store, _, _, _, server):
        original = store._append
        def fail(kind, *args):
            if kind == "DISCLOSURE_ALLOWED":
                raise OSError("injected actual durable journal failure")
            return original(kind, *args)
        monkeypatch.setattr(store, "_append", fail)
        status, _, content = request(server, "/api/data-access")
        assert status != 200 and b'"items"' not in content


@pytest.mark.parametrize("target", ["preview", "data"])
def test_expiry_during_actual_final_physical_read_cannot_disclose_rows(tmp_path, monkeypatch, target):
    from datetime import datetime, timezone
    import application.research_data as data_module
    import application.research_evidence as preview_module
    from application.research_data_view import DataAccessView
    with access_service(tmp_path) as (store, _, _, _, server):
        expired, assembled, touched = [False], [], []
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2100 if expired[0] else 2030, 1, 1, tzinfo=timezone.utc)
        monkeypatch.setattr(preview_module if target == "preview" else data_module, "datetime", Clock)
        original_rows, original_events = DataAccessView.rows, store._events
        def rows(view, cells):
            result = original_rows(view, cells)
            assembled.append(True)
            return result
        def physical():
            result = original_events()
            if assembled and store._read_snapshot() is None and not touched:
                expired[0] = True
                touched.append(True)
            return result
        monkeypatch.setattr(DataAccessView, "rows", rows)
        monkeypatch.setattr(store, "_events", physical)
        status, _, content = request(server, "/api/data-access")
        assert assembled and touched and status == 403, content
        assert json.loads(content)["error"]["code"] == "UNAUTHORIZED_DATA" and b'"items"' not in content


def test_corrupted_source_authority_after_final_physical_cannot_return_scope_claim(tmp_path, monkeypatch):
    from application.research_data_view import DataAccessView
    from tests.test_research_metadata_authorization import corrupt_grant_after_final_physical
    with access_service(tmp_path) as (store, _, grant, _, server):
        original, ready = DataAccessView.rows, []
        def rows(view, cells):
            result = original(view, cells)
            ready.append(True)
            return result
        monkeypatch.setattr(DataAccessView, "rows", rows)
        touched = corrupt_grant_after_final_physical(store, grant, ready, monkeypatch)
        status, _, content = request(server, "/api/data-access")
        assert ready and touched and status == 409, content
        assert json.loads(content)["error"]["code"] == "CORRUPT_ARTIFACT" and b'"items"' not in content


def test_denied_data_scope_cannot_hide_late_exposure_history_corruption(tmp_path, monkeypatch):
    from application.research_data_view import DataAccessView
    with access_service(tmp_path, grant_change={"test_authorization": False}) as (store, _, _, _, server):
        protocol = store.manifest("protocol-inputs")
        object_id = "exposure-history-" + protocol["history_hash"]
        history = store.manifest(object_id)
        original_rows, original_events, ready, touched = DataAccessView.rows, store._events, [], []
        def rows(view, cells):
            result = original_rows(view, cells)
            ready.append(True)
            return result
        def physical():
            result = original_events()
            if ready and store._read_snapshot() is None and not touched:
                (store.path / "manifests" / (object_id + ".json")).write_bytes(encode({**history, "source": "physically changed source"}))
                touched.append(True)
            return result
        monkeypatch.setattr(DataAccessView, "rows", rows)
        monkeypatch.setattr(store, "_events", physical)
        status, _, content = request(server, "/api/data-access")
        (tmp_path / "late-history-observed.json").write_bytes(encode({"status": status, "body": content.decode()}))
        assert ready and touched and status == 409, content
        assert json.loads(content)["error"]["code"] == "CORRUPT_ARTIFACT" and b'"items"' not in content


def test_late_source_package_corruption_cannot_inherit_earlier_visibility_check(tmp_path, monkeypatch):
    from application.research_data_view import DataAccessView
    with access_service(tmp_path) as (store, spec, _, _, server):
        object_id = "package-" + spec["admission"]["package_hash"]
        source = store.manifest(object_id)
        original_rows, original_events, ready, touched = DataAccessView.rows, store._events, [], []
        def rows(view, cells):
            result = original_rows(view, cells)
            ready.append(True)
            return result
        def physical():
            result = original_events()
            if ready and store._read_snapshot() is None and not touched:
                (store.path / "manifests" / (object_id + ".json")).write_bytes(encode({**source, "visibility": "restricted"}))
                touched.append(True)
            return result
        monkeypatch.setattr(DataAccessView, "rows", rows)
        monkeypatch.setattr(store, "_events", physical)
        status, _, content = request(server, "/api/data-access")
        assert ready and touched and status == 409, content
        assert json.loads(content)["error"]["code"] == "CORRUPT_ARTIFACT" and b'"items"' not in content


def test_restricted_source_is_not_declassified_by_synthetic_cell(tmp_path):
    from copy import deepcopy
    from tests.test_research_store import spec as base_spec
    from tests.research_admission_fixtures import admit_fixture, synthetic_plugin
    from tests.test_research_admission_chain import fixture_command
    from infrastructure.research_store import ResearchStore
    value = deepcopy(base_spec())
    value["cells"][0].update(plugin_id="restricted-source", capability="generic-rollout", visibility="synthetic")
    store = ResearchStore(tmp_path, "restricted-data-view", initialize=True)
    plugin = synthetic_plugin("restricted-source", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", fixture_command)
    grant = admit_fixture(store, value, plugin, tmp_path, formal=True, package_visibility="restricted")
    store.register(value, digest(value))
    preview = {**grant, "authorization_id": "limited-preview", "purposes": ["preview"], "visibilities": ["synthetic"]}
    store.authorize(preview)
    with pytest.raises(ResearchError) as denied:
        ResearchQuery(store, preview["authorization_id"]).data_access()
    assert denied.value.code == "UNAUTHORIZED_DATA"


def test_invalid_data_access_bounds_and_foreign_cursor_are_refused(tmp_path):
    with access_service(tmp_path) as (store, _, _, preview, server):
        query = ResearchQuery(store, preview["authorization_id"])
        for limit in (0, 201, True):
            with pytest.raises(ResearchError) as invalid:
                query.data_access(limit=limit)
            assert invalid.value.code == "CONTRACT_MISMATCH"
        for cursor in ("invalid", query.upstream(limit=1)["next_cursor"]):
            with pytest.raises(ResearchError) as stale:
                query.data_access(cursor=cursor)
            assert stale.value.code == "CURSOR_STALE"
        assert request(server, "/api/data-access?authorization_version=unregistered-v2")[0] == 200
