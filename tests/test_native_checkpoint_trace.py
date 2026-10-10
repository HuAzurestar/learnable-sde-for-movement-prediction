"""Pure diagnostic controls, not replacements for actual native deadline tests."""

import json

import pytest

from tests.native_checkpoint_trace import NativeCheckpointTrace


def test_original_return_identity_and_closed_timing_metadata():
    clock = iter([1., 1.125])
    trace = NativeCheckpointTrace(clock=lambda: next(clock))
    private = {"token": "secret-token", "state": "private-state"}
    assert trace.call("response", lambda: private, deadline=2., frame=True) is private
    row, = trace.report()["observations"]
    assert row == {"operation": "response", "duration_ms": 125., "remaining_ms": 875.,
                   "native_code": None, "frame_present": True, "raised": False}
    assert "secret-token" not in json.dumps(trace.report())
    assert "private-state" not in json.dumps(trace.report())


def test_same_original_exception_not_masked_by_optional_native_probe_failure():
    trace = NativeCheckpointTrace()
    original = AssertionError("private-exception-message")
    def fail():
        raise original
    def probe():
        raise OSError("private-probe-message")
    with pytest.raises(AssertionError) as caught:
        trace.call("native-stop", fail, native_code=probe)
    assert caught.value is original
    row, = trace.report()["observations"]
    assert row["raised"] and row["native_code"] is None
    assert "private" not in json.dumps(trace.report())


def test_missed_response_polls_do_not_evict_actual_phase_evidence():
    trace = NativeCheckpointTrace()
    trace.call("request", lambda: None)
    for _ in range(1000):
        assert trace.call("response", lambda: None, frame=True, omit_empty=True) is None
    assert [row["operation"] for row in trace.report()["observations"]] == ["request"]


def test_metadata_bound_and_snapshot_cannot_mutate_trace():
    trace = NativeCheckpointTrace()
    for _ in range(1000):
        trace.call("ack", lambda: "secret-token", native_code=lambda: -9)
    report = trace.report()
    assert len(report["observations"]) == 32
    assert len(json.dumps(report).encode()) < 16384
    assert "secret-token" not in json.dumps(report)
    report["observations"][0]["native_code"] = 85
    assert trace.report()["observations"][0]["native_code"] == -9


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 10**20, 10**400, True, "private-value"])
def test_invalid_scalars_never_leak_or_invent_clock_evidence(bad):
    trace = NativeCheckpointTrace(clock=lambda: bad)
    trace.call("ack", lambda: None, deadline=bad, native_code=lambda: bad)
    row, = trace.report()["observations"]
    assert row["duration_ms"] is None and row["remaining_ms"] is None
    assert row["native_code"] is None
    json.dumps(trace.report(), allow_nan=False)


def test_unknown_operation_refused_before_callback():
    trace = NativeCheckpointTrace()
    invoked = []
    with pytest.raises(ValueError, match="closed"):
        trace.call("private-token", lambda: invoked.append(True))
    assert not invoked and not trace.report()["observations"]


def test_optional_probe_failure_does_not_change_a_successful_callback(capsys):
    trace = NativeCheckpointTrace()
    original = object()
    def probe():
        raise OSError("private-probe-message")
    assert trace.call("native-stop", lambda: original, native_code=probe) is original
    assert not trace.report()["observations"][0]["raised"]
    assert trace.report()["observations"][0]["native_code"] is None
    assert not capsys.readouterr().out  # Observations alone never print.
