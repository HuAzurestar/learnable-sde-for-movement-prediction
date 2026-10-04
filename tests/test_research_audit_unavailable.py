"""Typed audit failures on actual staging I/O, with original exposure retained."""
import json
import os
import subprocess
import sys

import pytest

from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchError, encode
from tests.research_audit_fixtures import audit_fault, payload_opens
from tests.test_research_data_access_view import access_service
from tests.test_research_web import request
from tests.test_research_export_authorization import source, bundles


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
            # Force actual preview-grant expiry via the existing transparent
            # clock seam; retain original immutable grant bytes and scope.
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


@pytest.mark.parametrize("stage", ["event-rename", "head-fsync", pytest.param("event-directory-sync",
    marks=pytest.mark.skipif(os.name != "posix", reason="actual directory fsync is a POSIX publication boundary"))])
@pytest.mark.parametrize("kind", ["READ_STARTED", "READ_COMPLETED"])
def test_actual_journal_publication_phase_failure_preserves_possible_exposure(tmp_path, monkeypatch, stage, kind):
    with access_service(tmp_path) as (store, _, grant, _, server):
        with audit_fault(store, kind, monkeypatch, stage=stage) as fired:
            with pytest.raises(ResearchError) as failure:
                EvaluationExposureLedger(store).read("inputs", "fixture-1", purpose="evaluate",
                    authorization_id=grant["authorization_id"], data_root=tmp_path)
        assert fired and failure.value.code == "EXPOSURE_AUDIT_UNAVAILABLE"
        expected = (["READ_STARTED"] if kind == "READ_COMPLETED" else [])
        if stage != "event-rename":
            expected.append(kind)
        assert [e["event_kind"] for e in store.events() if e["event_kind"].startswith("READ_")] == expected
        status, _, content = request(server, "/api/data-access")
        assert status == 200
        assert all(r["exposure"]["status"] == ("EXPOSED" if expected else "NO_RECORDED_EXPOSURE")
                   for r in json.loads(content)["items"])


def test_non_audit_storage_error_and_audit_identity_conflict_are_not_reclassified(tmp_path, monkeypatch):
    with access_service(tmp_path) as (store, _, _, _, _):
        with audit_fault(store, "CONTROL", monkeypatch) as fired:
            with pytest.raises(OSError):
                store.append("CONTROL", {})
        assert fired
        event = store.append("READ_STARTED", {"block_id": "fixture-1"}, "original-audit-id")
        with pytest.raises(ResearchError) as failure:
            store.append("READ_STARTED", {"block_id": "different"}, event["event_id"])
        assert failure.value.code == "IDENTITY_CONFLICT"
        assert store.append("READ_STARTED", event["payload"], event["event_id"]) == event


def test_physical_corrupt_chain_remains_integrity_refusal_not_audit_outage(tmp_path):
    with access_service(tmp_path) as (store, _, grant, _, server):
        (store.path / "events/0000000000000001.json").write_bytes(b'{}')
        with payload_opens([tmp_path / "plugin-input.bin"]) as opened:
            with pytest.raises(ResearchError) as failure:
                EvaluationExposureLedger(store).read("inputs", "fixture-1", purpose="evaluate",
                    authorization_id=grant["authorization_id"], data_root=tmp_path)
            status, _, content = request(server, "/api/data-access")
        assert failure.value.code == "CORRUPT_ARTIFACT" and not opened
        assert status == 409 and json.loads(content)["error"]["code"] == "CORRUPT_ARTIFACT"


CLI_AUDIT_FAILURE = r'''
import json
from importlib import import_module
from pathlib import Path
import sys
import pytest
from infrastructure.research_store import ResearchStore
from tests.research_audit_fixtures import audit_fault
store = ResearchStore(Path(sys.argv[1]), sys.argv[2])
cli = import_module('experiments.pirc25.__main__')
output = Path(sys.argv[3])
changes = pytest.MonkeyPatch()
with audit_fault(store, 'DISCLOSURE_ALLOWED', changes, occurrence=int(sys.argv[4])) as fired:
    code = cli.main(['--root', sys.argv[1], '--store-id', sys.argv[2], 'export', 'synthetic',
                    '--authorization-id', 'export', '--output', str(output)])
Path(str(output) + '.fault.json').write_text(json.dumps(fired), encoding='utf-8')
raise SystemExit(code)
'''


@pytest.mark.parametrize("occurrence", [1, 2, 4])
@pytest.mark.parametrize("existing", [False, True])
def test_actual_cli_audit_failure_cannot_publish_or_overwrite_export(source, tmp_path, occurrence, existing):
    store, _, _, _ = source
    output = tmp_path / "audit-denied-bundle.json"
    original = b"old immutable synthetic target"
    if existing:
        output.write_bytes(original)
    completed = subprocess.run([sys.executable, "-B", "-c", CLI_AUDIT_FAILURE,
        str(store.path.parent), store.store_id, str(output), str(occurrence)],
        capture_output=True, text=True, timeout=30)
    observed = {"exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
                "fired": json.loads(output.with_suffix(".json.fault.json").read_bytes())}
    (tmp_path / "audit-cli-observed.json").write_bytes(encode(observed))
    assert observed["fired"] and completed.returncode == 1, observed
    error = json.loads(completed.stdout)["error"]
    assert error["code"] == "EXPOSURE_AUDIT_UNAVAILABLE" and error["trace_id"]
    assert str(tmp_path) not in completed.stdout and "PRIVATE" not in completed.stdout
    assert not completed.stderr and not error["retryable"]
    assert (output.read_bytes() == original) if existing else not output.exists()
    assert not list(output.parent.glob("." + output.name + ".*.staging"))
    events = store.events()
    reads = [e["event_kind"] for e in events if e["event_kind"].startswith("READ_")]
    assert reads == ([] if occurrence == 1 else ["READ_STARTED", "READ_COMPLETED"])
    # The fourth main disclosure is the external guard AFTER target fsync.
    # Its failure preserves the already published private candidate, not output.
    assert len(bundles(store)) == (1 if occurrence == 4 else 0)
