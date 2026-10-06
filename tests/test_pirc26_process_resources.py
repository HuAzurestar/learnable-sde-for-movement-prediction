"""Native observations and explicit unavailable semantics; synthetic controls."""

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from infrastructure import pirc26_process_resources as resources
from infrastructure.research_control import canonical, read_frame, WorkerControl
from estimation.phase_space import O1Plan, fit_o1
from estimation.phase_space_checkpoint import encode_state, rng_state, SOURCE_FILES
from tests.test_pirc26_dynamics import model
from tests.test_pirc26_training_forecast import batch


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_observation(value, *, pid=None):
    assert value["schema_version"] == resources.SCHEMA
    assert value["scope"] == "current-process-lifetime-through-observation" and value["units"] == "bytes"
    assert not value["includes_children"] and not value["includes_previous_attempts"] and not value["is_phase_isolated"]
    assert value["process_id"] == (pid if pid is not None else os.getpid())
    assert value["observed_monotonic_seconds"] >= value["observation_started_monotonic_seconds"]
    assert value["gpu_status"] == "NOT_APPLICABLE" and value["peak_gpu_memory_bytes"] is None
    assert len(canonical(value, 16384)) < 16384
    if value["status"] == "MEASURED":
        assert value["peak_resident_bytes"] > 0 and value["reason"] is None
    else:
        assert value["status"] == "UNAVAILABLE" and value["peak_resident_bytes"] is None and value["reason"]


def test_actual_host_observation_never_changes_rng_or_budget_state():
    before = encode_state(rng_state())
    value = resources.process_resources()
    assert_observation(value)
    assert encode_state(rng_state()) == before
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        assert value["status"] == "MEASURED"
    assert "budget" not in value and "remaining_seconds" not in value
    assert "infrastructure/pirc26_process_resources.py" in SOURCE_FILES


def test_native_process_peak_in_fresh_interpreter_includes_touched_memory_and_is_not_phase_delta():
    program = '''
import gc,json,os
from infrastructure.pirc26_process_resources import process_resources
before=process_resources()
buffer=bytearray(32*1024*1024)
for i in range(0,len(buffer),4096): buffer[i]=1
after=process_resources()
del buffer
gc.collect()
released=process_resources()
print(json.dumps({'pid':os.getpid(),'before':before,'after':after,'released':released}))
'''
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    values = json.loads(done.stdout)
    for name in ("before", "after", "released"):
        assert_observation(values[name], pid=values["pid"])
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        assert values["after"]["peak_resident_bytes"] >= max(32*1024*1024, values["before"]["peak_resident_bytes"])
        assert values["released"]["peak_resident_bytes"] >= values["after"]["peak_resident_bytes"]
    if sys.platform == "win32":
        assert values["after"]["peak_resident_bytes"] > values["before"]["peak_resident_bytes"]


def test_linux_self_kib_conversion_is_explicit_not_children_or_guessed_units(monkeypatch):
    calls = []
    sentinel = object()
    fake = SimpleNamespace(RUSAGE_SELF=sentinel,
        getrusage=lambda who: (calls.append(who) or SimpleNamespace(ru_maxrss=32768)))
    monkeypatch.setitem(sys.modules, "resource", fake)
    value = resources._linux_counters()
    assert calls == [sentinel] and value["peak_resident_bytes"] == 32768*1024
    assert value["peak_private_commit_bytes"] is None and value["commit_reason"]


@pytest.mark.parametrize("bad", [0, -1, True, .5, 1 << 63])
def test_linux_invalid_native_rss_never_becomes_zero_measurement(monkeypatch, bad):
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=1,
        getrusage=lambda who: SimpleNamespace(ru_maxrss=bad)))
    with pytest.raises(ValueError):
        resources._linux_counters()


@pytest.mark.parametrize("kind", ["failure", "zero", "negative", "noninteger", "inconsistent", "missing"])
def test_native_failure_or_invalid_counter_is_explicit_unavailable(monkeypatch, kind):
    monkeypatch.setattr(resources.sys, "platform", "win32")
    value = {"resident_counter": "fixture-native", "peak_resident_bytes": 10, "current_resident_bytes": 8,
             "commit_counter": "fixture-commit", "peak_private_commit_bytes": 12, "current_private_commit_bytes": 9,
             "commit_reason": None}
    if kind == "zero": value["peak_resident_bytes"] = 0
    elif kind == "negative": value["peak_resident_bytes"] = -1
    elif kind == "noninteger": value["peak_resident_bytes"] = True
    elif kind == "inconsistent": value["current_private_commit_bytes"] = 13
    elif kind == "missing": value.pop("peak_resident_bytes")
    def query():
        if kind == "failure": raise OSError("fixture private native message")
        return value
    monkeypatch.setattr(resources, "_windows_counters", query)
    result = resources.process_resources()
    assert_observation(result)
    assert result["status"] == "UNAVAILABLE" and result["reason"] == "NATIVE_COUNTER_UNAVAILABLE"
    assert "private native message" not in json.dumps(result)


def test_unsupported_platform_does_not_query_any_other_pid_or_engine(monkeypatch):
    monkeypatch.setattr(resources.sys, "platform", "unknown-test-platform")
    def forbidden(): pytest.fail("unsupported platform guessed a counter")
    monkeypatch.setattr(resources, "_windows_counters", forbidden)
    monkeypatch.setattr(resources, "_linux_counters", forbidden)
    result = resources.process_resources()
    assert result["status"] == "UNAVAILABLE" and result["reason"] == "UNSUPPORTED_PLATFORM"
    assert_observation(result)


def test_native_observer_is_fresh_not_cached_values(monkeypatch):
    monkeypatch.setattr(resources.sys, "platform", "linux")
    values = iter((10, 20))
    monkeypatch.setattr(resources, "_linux_counters", lambda: {
        "resident_counter": "fixture-native", "peak_resident_bytes": next(values), "current_resident_bytes": None,
        "commit_counter": None, "peak_private_commit_bytes": None, "current_private_commit_bytes": None,
        "commit_reason": "fixture-not-provided"})
    assert resources.process_resources()["peak_resident_bytes"] == 10
    assert resources.process_resources()["peak_resident_bytes"] == 20


def test_real_fit_reports_native_scope_without_serializing_telemetry_into_resumable_state():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        m, rows = model("M0"), []
        result = fit_o1(m, [batch(m, 4)], O1Plan(max_steps=2, patience=2, fit_diffusion=False),
            progress=rows.append, checkpoint_requested=lambda: rows[-1]["step"] == 1,
            checkpoint_handler=lambda state, row: None)
        assert_observation(result["resource_observation"])
        assert result["peak_memory_bytes"] == result["resource_observation"]["peak_resident_bytes"]
        assert result["resource_observation"]["point"] == "fit-return-before-encoding"
        assert "resource" not in json.dumps(result["training_state"])
    finally:
        torch.set_num_threads(previous)


def test_adapter_persists_observation_before_control_save_and_never_restores_memory_counter(tmp_path, monkeypatch):
    """Ordering-only synthetic control; actual owner ACK is tested separately."""
    from application.pirc26_training_control import ManagedTrainingControl
    from tests.test_pirc26_training_resume import initial_payload, run_fixture
    snapshot = run_fixture(initial_payload("O1"), stop=4)["training_state"]
    calls = []
    def save(state, progress):
        observation = read_frame(tmp_path / "pirc26-resources.json", 16384)
        assert_observation(observation)
        assert observation["point"] == "checkpoint-save-before-ack"
        assert observation["attempt_id"] == "synthetic-control-ordering"
        assert "resource" not in json.dumps(state)
        calls.append(progress)
    synthetic = SimpleNamespace(directory=tmp_path, descriptor={"deadline": time.monotonic()+60,
        "byte_limit": 4*1024*1024, "attempt_id": "synthetic-control-ordering"}, save=save)
    monkeypatch.setattr(WorkerControl, "from_environment", lambda: synthetic)
    control = ManagedTrainingControl(12)
    control.progress({"step": 4})
    control.save(snapshot, {"step": 4})
    assert len(calls) == 1 and calls[0]["completed_steps"] == 4
