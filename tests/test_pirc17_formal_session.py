"""Short software/OS fixtures only; no final-eval access or experiments."""
import os
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from experiments.pirc17 import formal_budget as budget, formal_session as session
from experiments.pirc17.protocol_core import digest, envelope, read_json, unpack

NS = 1_000_000_000
NATIVE = pytest.mark.skipif(os.name != "nt", reason="real native Windows Job test")
WORKER = Path(__file__).parent/"fixtures/pirc17_session_worker.py"


def contract(directory):
    return {"schema_version": budget.VERSION, "protocol_sha256": digest("protocol-fixture"),
        "execution_sha256": digest("execution-fixture"), "matrix_sha256": digest("matrix-fixture"),
        "runtime_manifest_sha256": digest("runtime-fixture"), "approval_sha256": digest("NOT-A-REAL-HUMAN-APPROVAL"),
        "ledger_directory": str(directory.resolve()), "phase_caps_ns": {"software": 60*NS},
        "total_cap_ns": 60*NS, "max_generated_forecasts": 0, "max_attempts_per_item": 1,
        "workloads": [{"work_id": digest(i), "phase": "software", "max_active_ns": (20 if i == 0 else 2)*NS,
                       "generated_forecasts": 0} for i in range(3)]}


@pytest.fixture
def ledger(tmp_path):
    with budget.Ledger.create(tmp_path/"ledger", contract(tmp_path/"ledger")) as value:
        yield value


def make_session(tmp_path, ledger, mode="counter", **kwargs):
    return session.OwnedSession(tmp_path/"session", ledger,
        [sys.executable, "-u", str(WORKER.resolve()), "--mode", mode], **kwargs)


def settle(ledger, observed):
    p = unpack(observed)
    # Test consumer only. Actual formal controller must durably publish this
    # observation AND account its overhead before granting the next admission.
    return ledger.settle(p["reservation_sha256"], status=p["status"], elapsed_ns=p["elapsed_ns"],
        completion_evidence_sha256=observed["sha256"], result_sha256=p["result_sha256"], reason="software fixture")


@NATIVE
def test_native_persistent_pid_state_and_exact_barrier_chain(tmp_path, ledger):
    worker = make_session(tmp_path, ledger)
    try:
        start = time.monotonic_ns()
        first = worker.run(ledger.reserve(digest(0)), started_ns=start)
        p = unpack(first)
        assert p["status"] == "success", p
        assert p["started_ns"] == start and p["elapsed_ns"] > 0
        assert p["closure"] is None and p["accounting"]["active_processes"] >= 2
        assert ledger.summary()["pending"] is not None  # Transport cannot self-settle.
        value = unpack(read_json(worker.directory/"outputs/000000/result.json"))["value"]
        assert value == {"fixture": True, "count": 1, "pid": p["worker_pid"]}
        pending = {**ledger.summary()["pending"], "ledger_root_sha256": ledger.root_sha256}
        with pytest.raises(FileExistsError):
            worker.run(pending)  # Even the same live object cannot double-dispatch.
        with pytest.raises(session.UnclosedTree, match="still active"):
            session.recover_pending_closure(ledger)
        settle(ledger, first)
        second = worker.run(ledger.reserve(digest(1)))
        q = unpack(second)
        assert q["status"] == "success", q
        assert q["worker_pid"] == p["worker_pid"]
        assert q["accounting"]["total_processes"] == p["accounting"]["total_processes"]
        assert unpack(read_json(worker.directory/"outputs/000001/result.json"))["value"]["count"] == 2
        assert unpack(read_json(worker.directory/"requests/000001.json"))["previous_barrier_sha256"] == p["barrier_sha256"]
        assert q["inter_call_gap"]["started_ns"] == p["ended_ns"]
        settle(ledger, second)
    finally:
        closed = worker.close()
    assert closed["process_tree_closed"] and closed["accounting"]["active_processes"] == 0
    assert closed["elapsed_ns"] == closed["ended_ns"]-closed["started_ns"]


@NATIVE
def test_native_expired_second_request_kills_entire_tree_and_no_retry(tmp_path, ledger):
    worker = make_session(tmp_path, ledger, "sleep_second")
    try:
        first = worker.run(ledger.reserve(digest(0)))
        assert unpack(first)["status"] == "success", first
        settle(ledger, first)
        second = worker.run(ledger.reserve(digest(1)))
        p = unpack(second)
        assert p["status"] == "timeout", p
        assert p["elapsed_ns"] >= 2*NS and p["ended_ns"] >= p["deadline_ns"]
        assert p["closure"]["process_tree_closed"]
        assert p["closure"]["accounting"]["active_processes"] == 0
        assert p["result_sha256"] is None
        settle(ledger, second)
        assert ledger.summary()["halted_reason"] == "work_deadline_reached"
        with pytest.raises(TimeoutError):
            ledger.reserve(digest(2))
        with pytest.raises(RuntimeError, match="cannot restart"):
            worker.run({})
    finally:
        worker.close()


@NATIVE
@pytest.mark.parametrize("mode", ["exception", "exit", "bad_ready", "oversized_ready",
    "bad_barrier_sequence", "bad_barrier_work_id", "bad_barrier_result_sha256",
    "bad_barrier_request_sha256", "bad_barrier_previous_barrier_sha256", "bad_barrier_worker_pid"])
def test_native_failure_never_becomes_success_and_preserves_pending(tmp_path, ledger, mode):
    worker = make_session(tmp_path, ledger, mode)
    try:
        observed = unpack(worker.run(ledger.reserve(digest(0))))
        assert observed["status"] == "failure", observed
        assert observed["closure"]["process_tree_closed"]
        assert observed["result_sha256"] is None
        assert ledger.summary()["pending"] is not None
        assert worker.failed and worker.closed
        closure = unpack(session.recover_pending_closure(ledger))
        assert closure["process_tree_closed"] and closure["elapsed_ns"] is None
        assert closure["required_conservative_charge_ns"] == 20*NS
    finally:
        worker.close()


@NATIVE
def test_native_unjoined_descendant_is_rejected_and_closed(tmp_path, ledger):
    worker = make_session(tmp_path, ledger, "child_second")
    try:
        first = worker.run(ledger.reserve(digest(0)))
        assert unpack(first)["status"] == "success", first
        settle(ledger, first)
        second = unpack(worker.run(ledger.reserve(digest(1))))
        assert second["status"] == "failure", second
        assert second["closure"]["accounting"]["active_processes"] == 0
        assert second["closure"]["accounting"]["total_processes"] > unpack(first)["accounting"]["total_processes"]
    finally:
        worker.close()


@NATIVE
def test_native_platform_identity_is_primed_before_baseline_and_reused(tmp_path, ledger):
    worker = make_session(tmp_path, ledger, "platform_identity")
    try:
        first = worker.run(ledger.reserve(digest(0)))
        p = unpack(first)
        assert p["status"] == "success", p
        value = unpack(read_json(worker.directory/"outputs/000000/result.json"))["value"]
        assert value["platform"] and value["after_ready_subprocesses"] == []
        assert p["accounting"]["total_processes"] == worker.baseline["total_processes"]
        settle(ledger, first)
        second = worker.run(ledger.reserve(digest(1)))
        q = unpack(second)
        assert q["status"] == "success", q
        again = unpack(read_json(worker.directory/"outputs/000001/result.json"))["value"]
        assert again["platform"] == value["platform"]
        assert again["processor"] == value["processor"]
        assert again["after_ready_subprocesses"] == []
        assert q["accounting"]["total_processes"] == p["accounting"]["total_processes"]
        settle(ledger, second)
    finally:
        closure = worker.close()
    assert closure["accounting"]["active_processes"] == 0


@NATIVE
def test_native_real_configured_runtime_after_ready_keeps_exact_identity(tmp_path):
    # Software-only environment preparation, not an empirical launch. The
    # ordinary CLI seal is then consumed by the REAL retained owned handler.
    metadata = subprocess.run([sys.executable, "-m", "experiments.pirc17.formal_entrypoint",
        "environment", "--output-directory", str(tmp_path/"environment")],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        encoding="utf-8", timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
    assert metadata.returncode == 0, metadata.stderr
    summary = json.loads(metadata.stdout)
    seal = read_json(summary["path"])
    c = contract(tmp_path/"full-runtime-ledger")
    c["phase_caps_ns"]["software"] = c["total_cap_ns"] = 90*NS
    c["workloads"][0]["max_active_ns"] = 40*NS  # Fixed software-import allowance.
    with budget.Ledger.create(tmp_path/"full-runtime-ledger", c) as actual:
        worker = session.OwnedSession(tmp_path/"full-runtime-session", actual,
            [sys.executable, "-u", str(WORKER.resolve()), "--mode", "configured_runtime",
             "--fixture-directory", summary["path"]])
        try:
            observed = worker.run(actual.reserve(digest(0)))
            p = unpack(observed)
            assert p["status"] == "success", p
            value = unpack(read_json(worker.directory/"outputs/000000/result.json"))["value"]
            assert value["environment_sha256"] == seal["sha256"]
            assert value["after_ready_subprocesses"] == []
            assert p["accounting"]["total_processes"] == worker.baseline["total_processes"]
            settle(actual, observed)
        finally:
            closure = worker.close()
    assert closure["accounting"]["active_processes"] == 0


@NATIVE
@pytest.mark.parametrize("mode", ["joined_child", "platform_cache_reset"])
def test_native_completed_descendants_are_still_rejected_and_closed(tmp_path, ledger, mode):
    worker = make_session(tmp_path, ledger, mode)
    try:
        p = unpack(worker.run(ledger.reserve(digest(0))))
        assert p["status"] == "failure", p
        assert p["error_type"] == "ValueError"
        assert p["result_sha256"] is None
        assert p["closure"]["accounting"]["active_processes"] == 0
        assert p["closure"]["accounting"]["total_processes"] > worker.baseline["total_processes"]
        assert ledger.summary()["pending"] is not None
        barrier = unpack(read_json(worker.directory/"barriers/000000.json"))
        assert barrier["status"] == "success"  # Handler return cannot bypass guard.
    finally:
        worker.close()


def test_platform_query_failure_prevents_readiness_and_handler_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_load_session", lambda *args: ({"job_name": "fixture"}, {"workloads": []}))
    monkeypatch.setattr(session, "_membership", lambda *args: None)
    def fail():
        raise ValueError("fixture platform-query failure")
    monkeypatch.setattr(session.platform, "platform", fail)
    monkeypatch.setattr(session, "_write", lambda *args: pytest.fail("ready must not be published"))
    with pytest.raises(ValueError, match="platform-query failure"):
        session.serve(tmp_path, digest("session"), lambda *args: pytest.fail("handler must not start"))


def test_platform_query_finishes_before_ready_but_handler_stays_lazy(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(session, "_load_session", lambda *args: ({"job_name": "fixture"}, {"workloads": []}))
    monkeypatch.setattr(session, "_membership", lambda *args: events.append("membership"))
    monkeypatch.setattr(session.platform, "platform", lambda: events.append("metadata"))
    def stop_at_ready(path, payload):
        assert payload["schema_version"] == session.VERSION+"-ready"
        events.append("ready")
        raise RuntimeError("fixture stop at ready")
    monkeypatch.setattr(session, "_write", stop_at_ready)
    with pytest.raises(RuntimeError, match="stop at ready"):
        session.serve(tmp_path, digest("session"), lambda *args: pytest.fail("handler stays lazy"))
    assert events == ["membership", "metadata", "ready"]


@NATIVE
def test_watchdog_remains_live_during_blocked_supervisor_io(tmp_path, ledger, monkeypatch):
    worker = make_session(tmp_path, ledger, "sleep_second")
    try:
        first = worker.run(ledger.reserve(digest(0)))
        assert unpack(first)["status"] == "success", first
        settle(ledger, first)
        original = session._write
        seen = []
        def slow_write(path, payload):
            result = original(path, payload)
            if Path(path).parent.name == "requests":
                time.sleep(2.3)  # Main supervisor cannot poll while IO is blocked.
                seen.append(worker.job.accounting()["active_processes"])
            return result
        monkeypatch.setattr(session, "_write", slow_write)
        observed = unpack(worker.run(ledger.reserve(digest(1))))
        assert seen == [0]  # Watchdog terminated the tree before main resumed.
        assert observed["status"] == "timeout" and observed["elapsed_ns"] >= 2_300_000_000
    finally:
        worker.close()


def test_prelaunch_memory_failure_never_spawns_and_retains_charge(tmp_path, ledger):
    worker = make_session(tmp_path, ledger, _memory=lambda: 0,
        _job_factory=lambda name: pytest.fail("must not spawn with insufficient memory"))
    observed = unpack(worker.run(ledger.reserve(digest(0))))
    assert observed["status"] == "failure"
    assert observed["stop_observation"]["reason"] == "resource_pressure"
    assert observed["closure"]["process_tree_closed"]
    assert worker.child is None and ledger.summary()["pending"] is not None


def test_prelaunch_expired_reservation_never_starts(tmp_path, ledger):
    worker = make_session(tmp_path, ledger, _job_factory=lambda name: pytest.fail("expired"))
    reservation = ledger.reserve(digest(0))
    observed = unpack(worker.run(reservation, started_ns=time.monotonic_ns()-21*NS))
    assert observed["status"] == "timeout" and observed["elapsed_ns"] >= 21*NS
    assert worker.child is None


def test_memory_query_failure_is_not_assumed_free_memory(tmp_path, ledger):
    def broken(): raise OSError("fixture-memory-query-failed")
    worker = make_session(tmp_path, ledger, _memory=broken)
    observed = unpack(worker.run(ledger.reserve(digest(0))))
    assert observed["status"] == "failure" and worker.child is None


def test_explicit_stop_before_start_closes_without_dispatch(tmp_path, ledger):
    worker = make_session(tmp_path, ledger, _stop=lambda: True)
    observed = unpack(worker.run(ledger.reserve(digest(0))))
    assert observed["status"] == "interrupted" and worker.child is None
    assert observed["stop_observation"]["reason"] == "operator_stop"


def test_future_or_overlapping_clock_does_not_claim_work(tmp_path, ledger):
    worker = make_session(tmp_path, ledger, _clock=lambda: 1000)
    pending = ledger.reserve(digest(0))
    with pytest.raises(ValueError, match="clock interval"):
        worker.run(pending, started_ns=1001)
    worker._last_end = 900
    with pytest.raises(ValueError, match="clock interval"):
        worker.run(pending, started_ns=899)
    assert not (ledger.directory/"dispatches").exists()


@pytest.mark.parametrize("change", ["root", "amount", "work", "missing", "halted", "closed", "bool_count"])
def test_only_exact_current_live_pending_reservation_admitted(tmp_path, ledger, change):
    receipt = ledger.reserve(digest(0))
    if change == "root": receipt["ledger_root_sha256"] = "0"*64
    if change == "amount": receipt["reserved_ns"] += 1
    if change == "work": receipt["work_id"] = digest(1)
    if change == "missing": receipt.pop("reservation_name")
    if change == "halted": ledger.halt("operator_stop", digest("fixture"))
    if change == "closed": ledger.close()
    if change == "bool_count": receipt["generated_forecasts"] = False
    worker = make_session(tmp_path, ledger, _job_factory=lambda name: pytest.fail("invalid"))
    with pytest.raises(ValueError, match="exact current durable"):
        worker.run(receipt)
    assert not worker.directory.exists()


def test_durable_dispatch_claim_blocks_new_session_or_path_replay(tmp_path, ledger):
    receipt = ledger.reserve(digest(0))
    first = make_session(tmp_path, ledger, _memory=lambda: 0)
    first.run(receipt)
    root = ledger.root_sha256
    ledger.close()
    with budget.Ledger.open(ledger.directory, expected_root_sha256=root) as resumed:
        second = session.OwnedSession(tmp_path/"new-session", resumed, ["not-launched"])
        with pytest.raises(FileExistsError):
            second.run(receipt)
        assert second.child is None
        assert resumed.summary()["pending"] is not None


def test_reentrant_request_is_rejected_before_dispatch(tmp_path, ledger):
    worker = make_session(tmp_path, ledger)
    worker._in_call.acquire()
    try:
        with pytest.raises(RuntimeError, match="one in-flight"):
            worker.run({})
    finally:
        worker._in_call.release()


@pytest.mark.parametrize("value", [[], "python -c something", [""], [None], ["a\0b"]])
def test_no_shell_or_ambiguous_command(tmp_path, ledger, value):
    with pytest.raises(ValueError, match="argument vector"):
        session.OwnedSession(tmp_path/"unused", ledger, value)


def test_metadata_publication_is_bounded(tmp_path):
    with pytest.raises(ValueError, match="bounded metadata"):
        session._write(tmp_path/"huge.json", {"data": "x"*session.MAX_MESSAGE_BYTES})
    assert not (tmp_path/"huge.json").exists()


def test_request_validator_rejects_foreign_and_expanded_reservation(tmp_path, ledger):
    receipt = ledger.reserve(digest(0))
    config = {"ledger_directory": str(ledger.directory), "ledger_root_sha256": ledger.root_sha256}
    sid, previous = digest("session"), digest("previous")
    row = {"schema_version": session.VERSION+"-request", "session_sha256": sid,
        "previous_barrier_sha256": previous, "sequence": 0, "work_id": digest(0),
        "reservation_sha256": receipt["reservation_sha256"], "reservation_event_index": 0,
        "started_ns": 100, "deadline_ns": 100+receipt["reserved_ns"]}
    work = {r["work_id"]: r for r in contract(ledger.directory)["workloads"]}
    args = dict(session_sha256=sid, previous_sha256=previous, sequence=0, session=config, work=work)
    assert session._request(envelope(row), **args)[1] == work[digest(0)]
    for key, value in (("sequence", True), ("session_sha256", digest("other")),
            ("previous_barrier_sha256", digest("other")), ("work_id", digest(1)),
            ("reservation_sha256", digest("other")), ("reservation_event_index", True),
            ("started_ns", True), ("deadline_ns", row["deadline_ns"]+1)):
        changed = {**row, key: value}
        with pytest.raises(ValueError):
            session._request(envelope(changed), **args)


def test_unconfirmed_native_closure_is_not_a_completion(tmp_path, ledger, monkeypatch):
    class StuckJob:
        def accounting(self): return {"active_processes": 1}
        def terminate(self): pass
        def close(self): pytest.fail("must retain handle until closure known")
    worker = make_session(tmp_path, ledger)
    worker.job = StuckJob()
    monkeypatch.setattr(session, "CLOSE_LIMIT_NS", 0)
    with pytest.raises(session.UnclosedTree, match="not confirmed"):
        worker.close()
    assert worker.closed and worker.job is not None


@pytest.mark.parametrize("fault", ["missing", "wrong_work", "wrong_job", "active", "query_error"])
def test_recovery_requires_real_exact_closed_job_evidence(tmp_path, ledger, monkeypatch, fault):
    pending = ledger.reserve(digest(0))
    worker = make_session(tmp_path, ledger, _memory=lambda: 0)
    if fault != "missing":
        worker.run(pending)
    path = ledger.directory/"dispatches"/(pending["reservation_sha256"]+".json")
    if fault in {"wrong_work", "wrong_job"}:
        p = unpack(read_json(path))
        p["work_id" if fault == "wrong_work" else "job_name"] = digest(2) if fault == "wrong_work" else "unrelated-job"
        from experiments.pirc17.protocol_core import canonical
        path.write_bytes(canonical(envelope(p)))  # Test-owned temporary evidence only.
    def query(name):
        if fault == "query_error": raise OSError("fixture-query-denied")
        return {"exists": True, "accounting": {"active_processes": 1}}
    monkeypatch.setattr(session, "_query_job", query)
    with pytest.raises((ValueError, OSError, session.UnclosedTree)):
        session.recover_pending_closure(ledger)
    assert ledger.summary()["pending"] is not None


@NATIVE
def test_direct_worker_is_not_an_owned_launch():
    with pytest.raises(PermissionError, match="live reserved process job"):
        session._membership("PIRC17-FORMAL-"+"f"*32)
