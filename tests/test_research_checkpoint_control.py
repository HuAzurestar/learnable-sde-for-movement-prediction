"""Strict IPC bounds/types and authoritative checkpoint source controls."""

import json
import time

import pytest

from infrastructure import research_control as control
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_research_recovery import setup


def descriptor(tmp_path):
    exchange = control.CheckpointExchange(tmp_path, "attempt-fixture", time.monotonic() + 3, 4096)
    return json.loads(exchange.environment()[control.ENVIRONMENT])


def test_total_frame_byte_quota_precedes_json_serialization(monkeypatch):
    monkeypatch.setattr(control.json, "dumps", lambda *a, **k: pytest.fail("over-byte-quota frame reached encoding allocation"))
    with pytest.raises(control.ControlError):
        control.canonical({"payload": ["x" * 32] * 8}, 128)


@pytest.mark.parametrize("value", [None, True, False, 0, -(2**128), 1e100, -0.0,
    {"中\u0000\n": ["😀\t\\\"", (), {"x": "é"}]}, {"empty": []}])
def test_canonical_accepts_exact_byte_boundary_and_rejects_one_less(value):
    expected = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert control.canonical(value, len(expected)) == expected
    with pytest.raises(control.ControlError):
        control.canonical(value, len(expected) - 1)


@pytest.mark.parametrize("field,value", [("attempt_id", []), ("attempt_id", ""), ("attempt_id", "../escape"),
    ("token", []), ("token", "not-a-token"), ("directory", "relative-runtime")])
def test_worker_descriptor_rejects_invalid_identity_and_path(tmp_path, field, value):
    value_set = descriptor(tmp_path)
    value_set[field] = value
    with pytest.raises(control.ControlError):
        control.WorkerControl(value_set)


def test_invalid_environment_json_is_a_typed_protocol_error(monkeypatch):
    monkeypatch.setenv(control.ENVIRONMENT, "{not-json")
    with pytest.raises(control.ControlError):
        control.WorkerControl.from_environment()


@pytest.mark.parametrize("fault", ["foreign-attempt", "wrong-token", "unrequested", "wrong-request", "over-byte-limit"])
def test_owner_rejects_response_before_any_handler(tmp_path, fault):
    exchange = control.CheckpointExchange(tmp_path, "attempt-fixture", time.monotonic() + 3, 4096)
    exchange.request()
    frame = {"schema_version": control.SCHEMA, "attempt_id": exchange.attempt_id, "token": exchange.token,
        "request_id": exchange.request_id, "state": {}, "progress": {}}
    if fault == "foreign-attempt":
        frame["attempt_id"] = "another-attempt"
    elif fault == "wrong-token":
        frame["token"] = "0" * 64
    elif fault == "unrequested":
        exchange.request_id = None
    elif fault == "wrong-request":
        frame["request_id"] = "0" * 32
    else:
        frame["state"] = {"payload": "x" * 4096}
    control.write_frame(tmp_path / "checkpoint-response.json", frame, 8192)
    with pytest.raises(control.ControlError):
        exchange.response()


@pytest.mark.parametrize("role", ["result", "checkpoint"])
def test_repacked_checkpoint_without_authoritative_save_cannot_resume(tmp_path, role):
    store, recovery, attempt, grant = setup(tmp_path)
    original = recovery.checkpoint(attempt, {"step": 1, "data_position": 1, "method_state": {}, "rng_state": {}})
    checkpoint = json.loads((store.path / "artifacts" / original).read_bytes())
    checkpoint["state"]["step"] = 2
    checkpoint["state"]["data_position"] = 2
    checkpoint["payload_hash"] = digest(checkpoint["state"])
    forged = store.artifact(encode(checkpoint), role=role, visibility="synthetic", block_ids=["fixture-1"], study_id="synthetic")
    store.transition(attempt, "FAILED", error_code="TRANSIENT")
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        recovery.prepare(attempt, forged["artifact_id"], authorization=grant)
    assert len(store.attempts()) == 1
