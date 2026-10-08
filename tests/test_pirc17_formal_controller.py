"""Software-only cumulative controller, artifact and recovery tests."""
import os
from pathlib import Path
import sys
import time

import pytest

from experiments.pirc17 import formal_budget as budget, formal_controller as control, formal_session as native
from experiments.pirc17.protocol_core import digest, envelope, file_hash, read_json, unpack

NS = 1_000_000_000


def contract(directory):
    return {"schema_version": budget.VERSION, "protocol_sha256": digest("protocol-fixture"),
        "execution_sha256": digest("execution-fixture"), "matrix_sha256": digest("matrix-fixture"),
        "runtime_manifest_sha256": digest("runtime-fixture"), "approval_sha256": digest("NOT-REAL-APPROVAL"),
        "ledger_directory": str(directory.resolve()), "phase_caps_ns": {"first": 60*NS, "second": 60*NS},
        "total_cap_ns": 120*NS, "max_generated_forecasts": 4, "max_attempts_per_item": 1,
        "workloads": [{"work_id": digest(i), "phase": "first" if i < 2 else "second",
            "max_active_ns": 20*NS, "generated_forecasts": 1} for i in range(4)]}


@pytest.fixture
def ledger(tmp_path):
    with budget.Ledger.create(tmp_path/"ledger", contract(tmp_path/"ledger")) as result:
        yield result


def authority(ledger, work):
    return {"schema_version": control.VERSION+"-work-authority", "ledger_root_sha256": ledger.root_sha256,
        "execution_sha256": ledger._state.contract["execution_sha256"],
        "approval_sha256": ledger._state.contract["approval_sha256"], "work_id": work["work_id"], "granted": True}


def verification(work, manifest, output):
    return {"schema_version": control.VERSION+"-artifact-verification", "work_id": work["work_id"],
        "verified": True, "details": {"software_fixture_only": True, "manifest": manifest},
        "artifacts": {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "file_sha256": file_hash(p)}
                      for p in output.rglob("*") if p.is_file() and p.name != "result.json"}}


class Clock:
    def __init__(self): self.now = 100*NS
    def __call__(self): return self.now
    def advance(self, amount): self.now += amount


class FakeSession:
    """Deterministic software seam, never production proof of OS supervision."""
    def __init__(self, directory, ledger, command, *, _clock, **kwargs):
        self.directory, self.ledger, self.command = directory, ledger, list(command)
        self.clock, self.sequence, self.closed = _clock, 0, False
        self.job_name = "PIRC17-FORMAL-"+"a"*32
        self.terminations, self.calls = [], []

    def run(self, reservation, *, started_ns):
        self.calls.append(reservation["work_id"])
        self.clock.advance(NS)
        output = self.directory/"outputs"/f"{self.sequence:06d}"
        output.mkdir(parents=True)
        (output/"fixture.bin").write_bytes(b"bounded software artifact")
        manifest = native._write(output/"result.json", {"value": {"fixture": True, "sequence": self.sequence}})
        result = envelope({"schema_version": native.VERSION+"-observation", "ledger_root_sha256": self.ledger.root_sha256,
            "reservation_sha256": reservation["reservation_sha256"], "work_id": reservation["work_id"],
            "status": "success", "sequence": self.sequence, "result_sha256": manifest["sha256"],
            "started_ns": started_ns, "ended_ns": self.clock(), "elapsed_ns": self.clock()-started_ns,
            "deadline_ns": started_ns+reservation["reserved_ns"]})
        self.sequence += 1
        return result

    def request_termination(self, reason): self.terminations.append(reason)

    def close(self):
        if not self.closed: self.clock.advance(50_000_000)
        self.closed = True
        return {"process_tree_closed": True, "accounting": {"active_processes": 0}, "software_fixture": True}


def make_controller(ledger, clock, **kwargs):
    return control.Controller(ledger, ["software-fixture-not-real-runner"],
        authorize_work=kwargs.pop("authorize_work", lambda w: authority(ledger, w)),
        validate_result=kwargs.pop("validate_result", verification), _clock=clock,
        _memory=lambda: 8*1024**3, _session_factory=kwargs.pop("_session_factory", FakeSession), **kwargs)


def test_complete_two_phase_inventory_keeps_one_worker_and_bills_both_types_of_time(ledger, monkeypatch):
    clock = Clock()
    original = ledger.settle
    def slow_settlement(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.advance(100_000_000)
        return result
    monkeypatch.setattr(ledger, "settle", slow_settlement)
    runner = make_controller(ledger, clock)
    result = runner.run_all(["first", "second"])
    assert runner.session.sequence == 4 and runner.session.closed
    assert set(result["dispositions"].values()) == {"success"}
    summary = result["summary"]
    assert summary["control_charged_ns_by_phase"] == {"first": 5*NS, "second": 5*NS}
    assert summary["control_observed_ns_by_phase"] == {"first": 200_000_000, "second": 250_000_000}
    assert summary["charged_ns_by_phase"] == {"first": 7*NS, "second": 7*NS}
    assert summary["measured_ns_by_phase"] == {"first": 2*NS, "second": 2*NS}
    assert summary["generated_forecasts_reserved"] == 4 and summary["pending"] is None
    assert summary["active_control_sha256"] is None and summary["halted_reason"] is None
    for key in summary["control_spans"]:
        assert control._control_record(ledger, key)["worker_job_name"] == runner.session.job_name
    for row in result["settlements"]:
        assert row["status"] == "success"
    with pytest.raises(control.ControllerStopped, match="one-shot"):
        runner.run_all(["first", "second"])


def test_validation_and_observation_io_are_inside_the_work_clock(ledger, monkeypatch):
    clock = Clock()
    publish = control._publish
    def slow_publish(directory, value):
        result = publish(directory, value)
        if value["schema_version"] == control.VERSION+"-work-observation":
            clock.advance(200_000_000)
        return result
    monkeypatch.setattr(control, "_publish", slow_publish)
    def validate(work, manifest, output):
        clock.advance(300_000_000)
        return verification(work, manifest, output)
    runner = make_controller(ledger, clock, validate_result=validate)
    result = runner.run_all(["first", "second"])
    assert result["summary"]["measured_ns_by_phase"] == {"first": 3*NS, "second": 3*NS}


def test_slow_scientific_manifest_validation_times_out_before_more_work(ledger):
    clock = Clock()
    def too_slow(work, manifest, output):
        clock.advance(20*NS)
        return verification(work, manifest, output)
    runner = make_controller(ledger, clock, validate_result=too_slow)
    with pytest.raises(control.ControllerStopped):
        runner.run_all(["first", "second"])
    assert runner.session.calls == [digest(0)] and runner.session.closed
    assert ledger.dispositions()[digest(0)] == "timeout"
    assert ledger.summary()["measured_ns_by_phase"]["first"] >= 21*NS
    assert ledger.summary()["halted_reason"] is not None


def test_expired_control_credit_during_settle_io_halts_no_retroactive_topup(ledger, monkeypatch):
    clock = Clock()
    settle = ledger.settle
    def stalled(*args, **kwargs):
        value = settle(*args, **kwargs)
        clock.advance(6*NS)
        return value
    monkeypatch.setattr(ledger, "settle", stalled)
    runner = make_controller(ledger, clock)
    with pytest.raises(control.ControllerStopped):
        runner.run_all(["first", "second"])
    assert runner.session.calls == [digest(0)]
    summary = ledger.summary()
    assert summary["halted_reason"] == "control_credit_exhausted"
    assert summary["control_charged_ns_by_phase"]["first"] >= 6*NS
    assert len(summary["control_spans"]) == 1


def test_topup_is_prepaid_before_expiry_and_does_not_expand_phase(ledger, monkeypatch):
    clock = Clock()
    settle = ledger.settle
    def slow(*args, **kwargs):
        value = settle(*args, **kwargs)
        clock.advance(4*NS)
        return value
    monkeypatch.setattr(ledger, "settle", slow)
    runner = make_controller(ledger, clock)
    # Each single topup buys only1s, not a free extension of the5s initial pool.
    # The second4s settlement exceeds the available6s and stops this run.
    with pytest.raises(control.ControllerStopped):
        runner.run_all(["first", "second"])
    events = [unpack(read_json(p)) for p in sorted((ledger.directory/"events").glob("*.json"))]
    topups = [e["row"] for e in events if e["type"] == "control_credit"]
    assert len(topups) == 1 and topups[0]["credit_ns"] == NS
    assert topups[0]["observed_ns"] == 4*NS
    assert ledger.summary()["control_charged_ns_by_phase"]["second"] == 0


@pytest.mark.parametrize("fault", ["false", "wrong_work", "wrong_approval", "bool_int", "missing"])
def test_unauthorized_work_never_reaches_worker(ledger, fault):
    clock = Clock()
    def bad(work):
        value = authority(ledger, work)
        if fault == "false": value["granted"] = False
        if fault == "wrong_work": value["work_id"] = digest(99)
        if fault == "wrong_approval": value["approval_sha256"] = digest("other")
        if fault == "bool_int": value["granted"] = 1
        if fault == "missing": value.pop("execution_sha256")
        return value
    runner = make_controller(ledger, clock, authorize_work=bad)
    with pytest.raises(control.ControllerStopped):
        runner.run_all(["first", "second"])
    assert runner.session.calls == [] and runner.session.closed
    assert ledger.dispositions()[digest(0)] == "failure"
    assert ledger.dispositions()[digest(1)] == "unattempted"


@pytest.mark.parametrize("fault", ["false", "empty", "missing_artifact", "extra_artifact", "hash", "size", "wrong_work"])
def test_verifier_boolean_cannot_replace_real_artifact_integrity(ledger, fault):
    clock = Clock()
    def invalid(work, manifest, output):
        value = verification(work, manifest, output)
        if fault == "false": value["verified"] = False
        if fault == "empty": return {}
        if fault == "missing_artifact": value["artifacts"] = {}
        if fault == "extra_artifact": (output/"unregistered.bin").write_bytes(b"unregistered test fixture")
        if fault == "hash": value["artifacts"]["fixture.bin"]["file_sha256"] = "0"*64
        if fault == "size": value["artifacts"]["fixture.bin"]["bytes"] += 1
        if fault == "wrong_work": value["work_id"] = digest(99)
        return value
    runner = make_controller(ledger, clock, validate_result=invalid)
    with pytest.raises(control.ControllerStopped):
        runner.run_all(["first", "second"])
    assert runner.session.calls == [digest(0)] and runner.session.closed
    assert ledger.dispositions()[digest(0)] == "failure"


@pytest.mark.parametrize("order", [["first"], ["first", "second", "first"], ["first", "unknown"]])
def test_complete_phase_denominator_is_mandatory(ledger, order):
    runner = make_controller(ledger, Clock())
    with pytest.raises(ValueError, match="complete explicit"):
        runner.run_all(order)
    assert ledger.tip["event_count"] == 0 and runner.session.calls == []


def test_claimed_scientific_success_with_foreign_native_clock_is_rejected(ledger):
    class BadSession(FakeSession):
        def run(self, receipt, *, started_ns):
            result = unpack(super().run(receipt, started_ns=started_ns))
            result["reservation_sha256"] = digest("wrong")
            return envelope(result)
    runner = make_controller(ledger, Clock(), _session_factory=BadSession)
    with pytest.raises(control.ControllerStopped):
        runner.run_all(["first", "second"])
    assert ledger.dispositions()[digest(0)] == "failure"


def test_recovery_of_open_controller_requires_closed_original_job_and_charges_pending(ledger, monkeypatch):
    clock = Clock()
    runner = make_controller(ledger, clock)
    runner._open_phase("first")
    pending = ledger.reserve(digest(0))
    runner.meter.close()  # Simulate controller process gone, without altering durable records.
    runner.meter = None
    monkeypatch.setattr(native, "_query_job", lambda name: {"exists": True, "accounting": {"active_processes": 1}})
    with pytest.raises(native.UnclosedTree):
        control.recover_interrupted(ledger)
    monkeypatch.setattr(native, "_query_job", lambda name: {"exists": False, "accounting": None})
    evidence = unpack(control.recover_interrupted(ledger))
    assert evidence["elapsed_ns"] is None
    summary = ledger.summary()
    assert summary["pending"] is None and summary["active_control_sha256"] is None
    assert summary["halted_reason"] == "unknown_controller_interruption"
    assert summary["conservatively_charged_ns_by_phase"]["first"] == 5*NS+pending["reserved_ns"]
    assert ledger.dispositions()[digest(0)] == "interrupted"
    with pytest.raises(TimeoutError): ledger.reserve(digest(1))


def test_disk_failure_keeps_pending_and_control_and_closes_only_our_worker(ledger, monkeypatch):
    clock = Clock()
    original = control._publish
    def broken(directory, value):
        if value["schema_version"] == control.VERSION+"-work-observation":
            raise OSError("software fixture disk failure")
        return original(directory, value)
    monkeypatch.setattr(control, "_publish", broken)
    runner = make_controller(ledger, clock)
    with pytest.raises(OSError): runner.run_all(["first", "second"])
    assert runner.session.closed and runner.meter is None
    assert ledger.summary()["pending"] is not None and ledger.summary()["active_control_sha256"] is not None
    with pytest.raises(RuntimeError): ledger.reserve(digest(1))


@pytest.mark.skipif(os.name != "nt", reason="real native Windows controller integration")
def test_native_controller_reuses_worker_across_complete_two_phase_run(ledger):
    worker = Path(__file__).parent/"fixtures/pirc17_session_worker.py"
    pids = []
    def validate(work, manifest, output):
        pids.append(manifest["pid"])
        return verification(work, manifest, output)
    runner = control.Controller(ledger, [sys.executable, "-u", str(worker.resolve())],
        authorize_work=lambda w: authority(ledger, w), validate_result=validate)
    result = runner.run_all(["first", "second"])
    assert len(pids) == 4 and len(set(pids)) == 1
    assert set(result["dispositions"].values()) == {"success"}
    assert runner.session.closed and runner.meter is None
    assert result["summary"]["control_observed_ns_by_phase"]["first"] > 0
    assert result["summary"]["control_observed_ns_by_phase"]["second"] > 0
    assert native._query_job(runner.session.job_name)["exists"] is False


@pytest.mark.skipif(os.name != "nt", reason="real native cumulative watchdog during controller IO")
def test_native_idle_worker_is_stopped_while_settlement_io_blocks(ledger, monkeypatch):
    monkeypatch.setattr(control, "INITIAL_CREDIT_NS", NS)
    monkeypatch.setattr(control, "TOPUP_MARGIN_NS", 1)
    monkeypatch.setattr(control, "TOPUP_CREDIT_NS", 100_000_000)
    worker = Path(__file__).parent/"fixtures/pirc17_session_worker.py"
    runner = control.Controller(ledger, [sys.executable, "-u", str(worker.resolve())],
        authorize_work=lambda w: authority(ledger, w), validate_result=verification)
    original = ledger.settle
    active = []
    def blocked(*args, **kwargs):
        value = original(*args, **kwargs)
        time.sleep(1.3)
        active.append(runner.session.job.accounting()["active_processes"])
        return value
    monkeypatch.setattr(ledger, "settle", blocked)
    with pytest.raises(control.ControllerStopped): runner.run_all(["first", "second"])
    assert active == [0]
    assert runner.session.closed and runner.meter is None
    assert ledger.summary()["halted_reason"] == "control_credit_exhausted"
    assert ledger.summary()["control_charged_ns_by_phase"]["first"] > NS
    assert ledger.summary()["attempted_work_items"] == 1


def test_phase_transition_gap_is_charged_to_next_phase(ledger, monkeypatch):
    clock = Clock()
    runner = make_controller(ledger, clock)
    original = runner._open_phase
    def delayed(phase):
        if phase == "second": clock.advance(400_000_000)
        original(phase)
    monkeypatch.setattr(runner, "_open_phase", delayed)
    summary = runner.run_all(["first", "second"])["summary"]
    assert summary["control_observed_ns_by_phase"] == {"first": 0, "second": 450_000_000}


def test_late_close_record_io_overrun_prevents_next_phase(ledger, monkeypatch):
    clock = Clock()
    original = ledger.close_control
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.advance(6*NS)
        return result
    monkeypatch.setattr(ledger, "close_control", delayed)
    runner = make_controller(ledger, clock)
    with pytest.raises(control.ControllerStopped): runner.run_all(["first", "second"])
    assert runner.session.calls == [digest(0), digest(1)]
    assert ledger.summary()["control_charged_ns_by_phase"]["first"] == 6*NS
    assert ledger.summary()["halted_reason"] == "control_credit_exhausted"
    assert runner.session.closed


def test_source_denominator_is_copied_once_not_once_per_work(ledger, monkeypatch):
    original, calls = ledger.dispositions, []
    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(ledger, "dispositions", counted)
    make_controller(ledger, Clock()).run_all(["first", "second"])
    assert len(calls) == 2  # Initial inventory and final complete disposition table.


def test_watcher_join_tail_is_not_free_after_closed_record(ledger, monkeypatch):
    clock = Clock()
    original = control._Meter.close
    first = []
    def delayed(meter):
        original(meter)
        if not first:
            first.append(True)
            clock.advance(6*NS)
    monkeypatch.setattr(control._Meter, "close", delayed)
    runner = make_controller(ledger, clock)
    with pytest.raises(control.ControllerStopped): runner.run_all(["first", "second"])
    assert ledger.summary()["control_charged_ns_by_phase"]["first"] == 6*NS
    assert ledger.summary()["halted_reason"] == "control_credit_exhausted"
    assert runner.session.calls == [digest(0), digest(1)] and runner.session.closed


def test_complete_controller_ledger_reopens_with_identical_dispositions_and_credit(ledger):
    runner = make_controller(ledger, Clock())
    runner.run_all(["first", "second"])
    expected, dispositions = ledger.summary(), ledger.dispositions()
    directory, root, tip = ledger.directory, ledger.root_sha256, ledger.tip
    ledger.close()
    with budget.Ledger.open(directory, expected_root_sha256=root, expected_tip=tip) as resumed:
        assert resumed.summary() == expected
        assert resumed.dispositions() == dispositions
        with pytest.raises(ValueError, match="no retry"):
            resumed.reserve(digest(0))


def test_control_start_without_enough_budget_never_launches_worker(ledger, monkeypatch):
    monkeypatch.setattr(control, "INITIAL_CREDIT_NS", 60*NS)
    runner = make_controller(ledger, Clock())
    with pytest.raises(control.ControllerStopped): runner.run_all(["first", "second"])
    assert runner.session.calls == [] and runner.session.closed
    assert ledger.summary()["halted_reason"] == "phase_cap_reached"


def test_real_artifact_path_escape_is_rejected(ledger, tmp_path):
    output = tmp_path/"safe"
    output.mkdir()
    (output/"fixture.bin").write_bytes(b"test")
    work = contract(ledger.directory)["workloads"][0]
    proof = verification(work, {}, output)
    proof["artifacts"]["../outside.bin"] = proof["artifacts"].pop("fixture.bin")
    with pytest.raises(ValueError): control.verify_artifacts(proof, work, output)


def test_failed_start_manifest_cannot_erase_startup_overhead(ledger, monkeypatch):
    clock = Clock()
    original = control._publish
    def broken(directory, value):
        if value["schema_version"] == control.VERSION+"-control":
            clock.advance(6*NS)
            raise OSError("software start-manifest failure")
        return original(directory, value)
    monkeypatch.setattr(control, "_publish", broken)
    runner = make_controller(ledger, clock)
    with pytest.raises(OSError): runner.run_all(["first", "second"])
    assert runner.session.calls == [] and runner.session.closed and runner.meter is None
    assert ledger.summary()["control_charged_ns_by_phase"]["first"] >= 6*NS
    assert ledger.summary()["halted_reason"] == "control_credit_exhausted"


def test_error_journal_failure_still_stops_watcher_and_worker(ledger, monkeypatch):
    original = control._publish
    def broken(directory, value):
        if value["schema_version"] in {control.VERSION+"-control", control.VERSION+"-stopped"}:
            raise OSError("software all-controller-publishing failure")
        return original(directory, value)
    monkeypatch.setattr(control, "_publish", broken)
    runner = make_controller(ledger, Clock())
    with pytest.raises(OSError): runner.run_all(["first", "second"])
    assert runner.session.closed and runner.meter is None
    assert ledger.summary()["active_control_sha256"] is not None  # Not silently repaired/closed.
    assert ledger.summary()["control_charged_ns_by_phase"]["first"] == 5*NS


def test_control_event_write_failure_still_stops_watcher(ledger, monkeypatch):
    original = budget._publish
    def broken(path, value, **kwargs):
        if value.get("type") == "control_open":
            raise OSError("software initial-control-event failure")
        return original(path, value, **kwargs)
    monkeypatch.setattr(budget, "_publish", broken)
    runner = make_controller(ledger, Clock())
    with pytest.raises(OSError): runner.run_all(["first", "second"])
    assert runner.session.calls == [] and runner.session.closed and runner.meter is None
    assert ledger._poisoned


def test_failed_handoff_before_reservation_is_control_time_not_a_free_gap(ledger, monkeypatch):
    clock = Clock()
    original = control._Meter.begin_work
    def broken(meter, allocation):
        original(meter, allocation)
        clock.advance(6*NS)
        raise control.ControllerStopped("software interrupted reservation handoff")
    monkeypatch.setattr(control._Meter, "begin_work", broken)
    runner = make_controller(ledger, clock)
    with pytest.raises(control.ControllerStopped): runner.run_all(["first", "second"])
    summary = ledger.summary()
    assert summary["attempted_work_items"] == 0 and summary["pending"] is None
    assert summary["control_charged_ns_by_phase"]["first"] >= 6*NS
    assert summary["halted_reason"] == "control_credit_exhausted"
    assert runner.session.calls == [] and runner.session.closed and runner.meter is None


def test_entrypoint_prefix_is_charged_only_to_first_phase_not_hidden_or_repeated(ledger):
    clock = Clock()
    runner = make_controller(ledger, clock)
    start = clock()-7*NS  # Software metadata/setup before constructing Ledger/Controller.
    result = runner.run_all(['first', 'second'], started_ns=start)
    summary = result['summary']
    assert summary['control_charged_ns_by_phase'] == {'first': 12*NS, 'second': 5*NS}
    assert summary['control_observed_ns_by_phase']['first'] == 7*NS
    assert summary['charged_ns_by_phase'] == {'first': 14*NS, 'second': 7*NS}
    first = next(iter(summary['control_spans']))
    assert control._control_record(ledger, first)['started_ns'] == start
    assert summary['attempted_work_items'] == 4
    assert ledger._state.contract['total_cap_ns'] == 120*NS  # No new budget.


def test_expired_outer_startup_cannot_launch_worker_or_reset_input_clock(ledger):
    clock = Clock()
    runner = make_controller(ledger, clock)
    with pytest.raises(control.ControllerStopped):
        runner.run_all(['first', 'second'], started_ns=clock()-61*NS)
    summary = ledger.summary()
    assert summary['attempted_work_items'] == 0
    assert summary['control_charged_ns_by_phase']['first'] >= 61*NS
    assert summary['halted_reason'] is not None
    assert runner.session.calls == [] and runner.session.closed
    assert set(ledger.dispositions().values()) == {'unattempted'}


@pytest.mark.parametrize('bad_start', [-1, True, 1.5, 101*NS])
def test_entrypoint_start_rejects_invalid_or_future_clock_before_admission(ledger, bad_start):
    runner = make_controller(ledger, Clock())
    with pytest.raises(ValueError): runner.run_all(['first', 'second'], started_ns=bad_start)
    assert ledger.tip['event_count'] == 0 and runner.session.calls == []


def test_terminal_publication_is_measured_in_last_phase_without_more_credit(ledger):
    clock = Clock()
    runner = make_controller(ledger, clock)
    runner.run_all(['first', 'second'])
    before = ledger.summary()
    clock.advance(NS)  # Actual launcher writes its terminal report here.
    runner.finish_terminal(digest('saved-terminal-bytes'))
    after = ledger.summary()
    assert after['charged_ns_by_phase'] == before['charged_ns_by_phase']
    assert after['control_observed_ns_by_phase']['second'] == before['control_observed_ns_by_phase']['second']+NS
    assert after['control_observed_ns_by_phase']['first'] == before['control_observed_ns_by_phase']['first']
    assert after['terminal_control_sha256'] == next(reversed(after['control_spans']))


def test_terminal_publication_overrun_is_durable_and_never_success(ledger):
    clock = Clock()
    runner = make_controller(ledger, clock)
    runner.run_all(['first', 'second'])
    clock.advance(6*NS)
    with pytest.raises(control.ControllerStopped): runner.finish_terminal(digest('slow-terminal-bytes'))
    summary = ledger.summary()
    assert summary['control_charged_ns_by_phase']['second'] >= 6*NS
    assert summary['halted_reason'] == 'control_credit_exhausted'
    assert summary['terminal_control_sha256'] is not None
    assert runner.session.closed and runner.session.sequence == 4


def test_terminal_append_io_overrun_is_not_hidden_by_an_earlier_success(ledger, monkeypatch):
    clock = Clock()
    runner = make_controller(ledger, clock)
    runner.run_all(['first', 'second'])
    original = ledger.finish_control_tail
    def slow(*args, **kwargs):
        record = original(*args, **kwargs)
        clock.advance(6*NS)
        return record
    monkeypatch.setattr(ledger, 'finish_control_tail', slow)
    with pytest.raises(control.ControllerStopped): runner.finish_terminal(digest('terminal'))
    assert ledger.summary()['control_charged_ns_by_phase']['second'] >= 6*NS
    assert ledger.summary()['halted_reason'] == 'control_credit_exhausted'


def test_terminal_does_not_trust_the_sessions_closed_boolean(ledger, monkeypatch):
    runner = make_controller(ledger, Clock())
    runner.run_all(['first', 'second'])
    assert runner.session.closed
    monkeypatch.setattr(runner.session, 'close',
        lambda: (_ for _ in ()).throw(native.UnclosedTree('software closure failure')))
    with pytest.raises(native.UnclosedTree): runner.finish_terminal(digest('terminal'))
    assert ledger.summary()['terminal_control_sha256'] is None
