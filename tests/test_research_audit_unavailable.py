"""Typed audit failures on actual staging I/O, with original exposure retained."""
import json

import pytest

from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchError, encode
from tests.research_audit_fixtures import audit_fault, payload_opens
from tests.test_research_data_access_view import access_service
from tests.test_research_web import request


@pytest.mark.parametrize("kind", ["EXPOSURE_ALLOWED", "EXPOSURE_DENIED", "READ_STARTED", "READ_COMPLETED", "READ_FAILED"])
def test_provider_actual_event_fsync_failure_is_typed_without_returning_bytes(tmp_path, monkeypatch, kind):
    with access_service(tmp_path) as (store, _, grant, _, _):
        provider = tmp_path / "plugin-input.bin"
        if kind == "READ_FAILED":
            provider.write_bytes(b"actual changed synthetic source, original frozen digest retained")
        purpose = "fit" if kind == "EXPOSURE_DENIED" else "evaluate"
        with payload_opens([provider]) as opened, audit_fault(store, kind, monkeypatch) as fired:
            try:
                EvaluationExposureLedger(store).read("inputs", "fixture-1", purpose=purpose,
                    authorization_id=grant["authorization_id"], data_root=tmp_path)
            except Exception as exc:
                observed = {"exception_type": type(exc).__name__, "code": getattr(exc, "code", None),
                            "safe_details": getattr(exc, "safe_details", None), "fired": fired, "opens": len(opened)}
            else:
                pytest.fail("provider returned bytes after real audit-fsync failure")
        (tmp_path / "audit-provider-observed.json").write_bytes(encode(observed))
        assert fired and observed["code"] == "EXPOSURE_AUDIT_UNAVAILABLE", observed
        assert str(tmp_path) not in observed["safe_details"] and "PRIVATE" not in observed["safe_details"]
        assert observed["opens"] == (1 if kind in {"READ_COMPLETED", "READ_FAILED"} else 0)
        reads = [e for e in store.events() if e["event_kind"].startswith("READ_")]
        assert [e["event_kind"] for e in reads] == (["READ_STARTED"] if kind in {"READ_COMPLETED", "READ_FAILED"} else [])


@pytest.mark.parametrize("kind", ["DISCLOSURE_ALLOWED", "DISCLOSURE_DENIED", "EXPOSURE_ALLOWED", "READ_STARTED", "READ_COMPLETED"])
def test_http_actual_audit_fsync_failure_is_typed_and_never_returns_rows_or_result(tmp_path, monkeypatch, kind):
    with access_service(tmp_path) as (store, spec, _, preview, server):
        artifact = store.artifact(encode({"private_synthetic_result": "must_not_escape_audit_failure"}),
            role="result", visibility="synthetic", block_ids=["fixture-1"], study_id=spec["study_id"])
        endpoint = "/api/data-access" if kind.startswith("DISCLOSURE_") else "/api/artifacts/" + artifact["artifact_id"]
        if kind == "DISCLOSURE_DENIED":
            # Separate immutable preview grant deliberately lacks purpose, not
            # corrupted content or an invented data-provider success/failure.
            # The same server's selected grant is immutable, so force expiry via
            # the existing transparent clock seam, retaining actual grant bytes.
            from datetime import datetime, timezone
            import application.research_evidence as evidence
            class Expired(datetime):
                @classmethod
                def now(cls, tz=None):
                    return datetime(2100, 1, 1, tzinfo=timezone.utc)
            monkeypatch.setattr(evidence, "datetime", Expired)
        with payload_opens([store.path / "artifacts" / artifact["artifact_id"]]) as opened, audit_fault(store, kind, monkeypatch) as fired:
            status, _, content = request(server, endpoint)
        observed = {"status": status, "body": json.loads(content), "fired": fired, "opens": len(opened)}
        (tmp_path / "audit-http-observed.json").write_bytes(encode(observed))
        assert fired and status == 503, observed
        assert observed["body"]["error"]["code"] == "EXPOSURE_AUDIT_UNAVAILABLE"
        assert observed["body"]["error"]["trace_id"]
        assert str(tmp_path) not in content.decode() and b'"items"' not in content and b'must_not_escape_audit_failure' not in content
        assert observed["opens"] == (1 if kind == "READ_COMPLETED" else 0)
        reads = [e for e in store.events() if e["event_kind"].startswith("READ_")]
        assert [e["event_kind"] for e in reads] == (["READ_STARTED"] if kind == "READ_COMPLETED" else [])


@pytest.mark.parametrize("stage", ["event-rename", "head-fsync"])
def test_actual_journal_publication_phase_failure_preserves_possible_exposure(tmp_path, monkeypatch, stage):
    with access_service(tmp_path) as (store, _, grant, _, server):
        with audit_fault(store, "READ_STARTED", monkeypatch, stage=stage) as fired:
            with pytest.raises(ResearchError) as failure:
                EvaluationExposureLedger(store).read("inputs", "fixture-1", purpose="evaluate",
                    authorization_id=grant["authorization_id"], data_root=tmp_path)
        assert fired and failure.value.code == "EXPOSURE_AUDIT_UNAVAILABLE"
        expected = ["READ_STARTED"] if stage == "head-fsync" else []
        assert [e["event_kind"] for e in store.events() if e["event_kind"].startswith("READ_")] == expected
        status, _, content = request(server, "/api/data-access")
        assert status == 200
        assert all(r["exposure"]["status"] == ("EXPOSED" if expected else "NO_RECORDED_EXPOSURE")
                   for r in json.loads(content)["items"])
