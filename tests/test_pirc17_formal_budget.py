"""Accounting/process-lock software fixtures, never research forecasts."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys

import pytest

from experiments.pirc17 import formal_budget as budget
from experiments.pirc17.protocol_core import canonical, digest, envelope, read_json, unpack


def fixture_contract(directory):
    return {"schema_version": budget.VERSION, "protocol_sha256": digest("protocol-fixture"),
        "execution_sha256": digest("execution-fixture"), "matrix_sha256": digest("matrix-fixture"),
        "runtime_manifest_sha256": digest("runtime-fixture"), "approval_sha256": digest("NOT-REAL-APPROVAL"),
        "ledger_directory": str(directory.resolve()), "phase_caps_ns": {"fit": 1000, "forecast": 500},
        "total_cap_ns": 1500, "max_generated_forecasts": 3, "max_attempts_per_item": 1,
        "workloads": [{"work_id": digest(i), "phase": "fit" if i < 2 else "forecast",
                       "max_active_ns": 700 if i < 2 else 300, "generated_forecasts": int(i >= 2)} for i in range(5)]}


@pytest.fixture
def ledger(tmp_path):
    result = budget.Ledger.create(tmp_path/"ledger", fixture_contract(tmp_path/"ledger"))
    yield result
    result.close()


def settle(ledger, receipt, elapsed=100, status="success"):
    return ledger.settle(receipt["reservation_sha256"], status=status, elapsed_ns=elapsed,
        completion_evidence_sha256=digest("SOFTWARE-COMPLETION-FIXTURE-NOT-PROCESS-PROOF"),
        result_sha256=digest("fixture-output") if status == "success" else None, reason="software-fixture")


def test_reserve_debits_before_work_and_settle_releases_only_observed_remainder(ledger):
    r = ledger.reserve(digest(0))
    assert r["reserved_ns"] == 700
    assert ledger.summary()["remaining_ns_by_phase"]["fit"] == 300
    assert ledger.summary()["pending"]["reservation_sha256"] == r["reservation_sha256"]
    stored = read_json(ledger.directory/"events/000000.json")
    assert stored["sha256"] == r["reservation_sha256"]
    assert unpack(read_json(ledger.directory/"head.json"))["last_event_sha256"] == r["reservation_sha256"]
    settle(ledger, r, elapsed=150)
    summary = ledger.summary()
    assert summary["remaining_ns_by_phase"] == {"fit": 850, "forecast": 500}
    assert summary["measured_ns_by_phase"] == {"fit": 150, "forecast": 0}
    assert summary["attempted_work_items"] == 1 and summary["unattempted_work_items"] == 4


def test_same_item_cannot_retry_after_success_or_failure(ledger):
    for i, status in ((0, "success"), (1, "failure")):
        settle(ledger, ledger.reserve(digest(i)), status=status)
        with pytest.raises(ValueError, match="no retry"):
            ledger.reserve(digest(i))
    assert ledger.dispositions()[digest(1)] == "failure"


def test_pending_reservation_blocks_every_phase_not_just_its_own(ledger):
    ledger.reserve(digest(0))
    with pytest.raises(RuntimeError, match="verified completion"):
        ledger.reserve(digest(2))
    assert ledger.summary()["generated_forecasts_reserved"] == 0


def test_reopen_preserves_full_debit_before_any_new_work(ledger):
    r = ledger.reserve(digest(2))
    root, tip, directory = ledger.root_sha256, ledger.tip, ledger.directory
    ledger.close()
    with budget.Ledger.open(directory, expected_root_sha256=root, expected_tip=tip) as resumed:
        assert resumed.summary()["remaining_ns_by_phase"]["forecast"] == 200
        assert resumed.summary()["generated_forecasts_reserved"] == 1
        with pytest.raises(RuntimeError):
            resumed.reserve(digest(3))
        settle(resumed, r, elapsed=None, status="interrupted")
        assert resumed.summary()["conservatively_charged_ns_by_phase"]["forecast"] == 300
        assert resumed.summary()["measured_ns_by_phase"]["forecast"] == 0
        with pytest.raises(ValueError):
            resumed.reserve(digest(2))
        next_receipt = resumed.reserve(digest(3))
        assert next_receipt["reserved_ns"] == 200
        settle(resumed, next_receipt, elapsed=50)
        assert resumed.summary()["remaining_ns_by_phase"]["forecast"] == 150


def test_phase_credit_cannot_be_transferred_and_timeout_halts_globally(ledger):
    settle(ledger, ledger.reserve(digest(2)), elapsed=250)
    r = ledger.reserve(digest(3))
    assert r["reserved_ns"] == 250  # Unused fit1000 cannot augment forecast250.
    settle(ledger, r, elapsed=260, status="timeout")
    summary = ledger.summary()
    assert summary["charged_ns_by_phase"]["forecast"] == 510  # Never clamp actual overrun to hide it.
    assert summary["remaining_ns_by_phase"]["fit"] == 1000
    assert summary["halted_reason"] == "phase_cap_reached"
    with pytest.raises(TimeoutError):
        ledger.reserve(digest(0))


@pytest.mark.parametrize("elapsed", [300, 301])
def test_deadline_boundary_is_not_success(ledger, elapsed):
    r = ledger.reserve(digest(2))
    before = ledger.tip
    with pytest.raises(ValueError, match="timeout"):
        settle(ledger, r, elapsed=elapsed)
    assert ledger.tip == before and ledger.summary()["pending"] is not None


def test_exhausting_work_cap_stops_even_if_phase_has_credit(ledger):
    settle(ledger, ledger.reserve(digest(2)), elapsed=300, status="timeout")
    assert ledger.summary()["halted_reason"] == "work_deadline_reached"
    with pytest.raises(TimeoutError):
        ledger.reserve(digest(3))


@pytest.mark.parametrize("bad", [True, False, -1, .5, float("inf"), float("nan")])
def test_elapsed_must_be_measured_integer_nanoseconds(ledger, bad):
    r = ledger.reserve(digest(0))
    before = ledger.tip
    with pytest.raises(ValueError):
        settle(ledger, r, elapsed=bad)
    assert ledger.tip == before


def test_unknown_elapsed_is_never_free_or_success(ledger):
    r = ledger.reserve(digest(0))
    with pytest.raises(ValueError, match="unknown elapsed"):
        settle(ledger, r, elapsed=None)
    settle(ledger, r, elapsed=None, status="interrupted")
    assert ledger.summary()["charged_ns_by_phase"]["fit"] == 700


@pytest.mark.parametrize("field,value", [("completion_evidence_sha256", None), ("result_sha256", None),
    ("reservation_sha256", "0"*64), ("reason", ""), ("status", "PASS")])
def test_incomplete_or_foreign_settlement_cannot_release_budget(ledger, field, value):
    r = ledger.reserve(digest(0))
    args = {"reservation_sha256": r["reservation_sha256"], "status": "success", "elapsed_ns": 100,
        "completion_evidence_sha256": "1"*64, "result_sha256": "2"*64, "reason": "fixture"}
    args[field] = value
    with pytest.raises(ValueError):
        ledger.settle(**args)
    assert ledger.summary()["remaining_ns_by_phase"]["fit"] == 300


def test_no_new_directory_or_existing_directory_reset(ledger, tmp_path):
    contract = fixture_contract(ledger.directory)
    with pytest.raises(FileExistsError):
        budget.Ledger.create(ledger.directory, contract)
    with pytest.raises(ValueError, match="changing directories"):
        budget.Ledger.create(tmp_path/"reset", contract)
    assert not (tmp_path/"reset").exists()


def test_os_lock_excludes_second_writer_without_waiting(ledger):
    with pytest.raises(OSError):
        budget.Ledger.open(ledger.directory, expected_root_sha256=ledger.root_sha256)


def test_wrong_root_identity_releases_lock_without_mutation(ledger):
    root, directory = ledger.root_sha256, ledger.directory
    ledger.close()
    with pytest.raises(ValueError, match="identity mismatch"):
        budget.Ledger.open(directory, expected_root_sha256="0"*64)
    with budget.Ledger.open(directory, expected_root_sha256=root) as resumed:
        assert resumed.tip["event_count"] == 0


def test_head_deletion_of_committed_suffix_detected(ledger):
    r = ledger.reserve(digest(0))
    settle(ledger, r)
    root, directory = ledger.root_sha256, ledger.directory
    ledger.close()
    (directory/"events/000001.json").unlink()  # Test-owned temp fixture only.
    with pytest.raises(ValueError, match="suffix deleted"):
        budget.Ledger.open(directory, expected_root_sha256=root)


def test_external_pin_detects_rollback_of_both_head_and_tail(ledger):
    old_head = (ledger.directory/"head.json").read_bytes()
    r = ledger.reserve(digest(0))
    settle(ledger, r)
    root, tip, directory = ledger.root_sha256, ledger.tip, ledger.directory
    ledger.close()
    (directory/"events/000000.json").unlink()
    (directory/"events/000001.json").unlink()
    (directory/"head.json").write_bytes(old_head)
    with pytest.raises(ValueError, match="independently pinned"):
        budget.Ledger.open(directory, expected_root_sha256=root, expected_tip=tip)


@pytest.mark.parametrize("fault", ["amount", "previous", "root", "index", "bool_count", "partial", "extra"])
def test_hash_chain_and_recomputed_budget_reject_tampering(ledger, fault):
    ledger.reserve(digest(2))
    root, directory = ledger.root_sha256, ledger.directory
    ledger.close()
    path = directory/"events/000000.json"
    payload = read_json(path)["payload"]
    if fault == "amount": payload["row"]["reserved_ns"] = 1
    if fault == "previous": payload["previous_sha256"] = "0"*64
    if fault == "root": payload["root_sha256"] = "0"*64
    if fault == "index": payload["index"] = 1
    if fault == "bool_count": payload["row"]["generated_forecasts"] = True
    if fault == "partial": path.write_bytes(b'{"payload":')
    elif fault == "extra": (directory/"events/extra.json").write_bytes(b"{}")
    else: path.write_bytes(canonical(envelope(payload)))
    with pytest.raises(ValueError):
        budget.Ledger.open(directory, expected_root_sha256=root)


def test_unanchored_reservation_recovered_without_refund_or_retry(ledger, monkeypatch):
    original = budget._publish
    def fail_head(path, payload, **kwargs):
        if Path(path).name == "head.json":
            raise OSError("software-disk-failure-after-event")
        return original(path, payload, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(budget, "_publish", fail_head)
        with pytest.raises(OSError):
            ledger.reserve(digest(2))
    with pytest.raises(RuntimeError):
        ledger.reserve(digest(3))
    root, directory = ledger.root_sha256, ledger.directory
    ledger.close()
    with budget.Ledger.open(directory, expected_root_sha256=root) as resumed:
        assert resumed.recovered_unanchored_events == 1
        assert resumed.summary()["remaining_ns_by_phase"]["forecast"] == 200
        assert resumed.summary()["generated_forecasts_reserved"] == 1
        assert unpack(read_json(directory/"head.json")) == resumed.tip
        with pytest.raises(RuntimeError):
            resumed.reserve(digest(3))


def test_unpublished_temporary_file_is_retained_but_not_authority(ledger):
    root, directory = ledger.root_sha256, ledger.directory
    ledger.close()
    pending = directory/"events/.000000.json.orphan.pending"
    pending.write_bytes(b"incomplete-software-fixture")
    with budget.Ledger.open(directory, expected_root_sha256=root) as resumed:
        assert resumed.tip["event_count"] == 0
        assert resumed.summary()["orphan_temporary_files"] == [pending.name]
        assert pending.exists()


def test_explicit_resource_halt_does_not_drop_pending_charge(ledger):
    r = ledger.reserve(digest(0))
    ledger.halt("resource_pressure", "9"*64)
    assert ledger.summary()["remaining_ns_by_phase"]["fit"] == 300
    settle(ledger, r, elapsed=123, status="failure")
    assert ledger.summary()["remaining_ns_by_phase"]["fit"] == 877
    assert ledger.summary()["halted_reason"] == "resource_pressure"
    with pytest.raises(TimeoutError):
        ledger.reserve(digest(1))


def test_summaries_are_copies_and_append_never_rescans_history(ledger, monkeypatch):
    def trap(*args, **kwargs):
        pytest.fail("append must not reread the full history")
    monkeypatch.setattr(budget, "read_json", trap)
    for i in range(2):
        r = ledger.reserve(digest(i))
        summary = ledger.summary()
        summary["pending"]["reserved_ns"] = 0
        summary["remaining_ns_by_phase"]["fit"] = 10**9
        settle(ledger, r)
    assert ledger.summary()["charged_ns_by_phase"]["fit"] == 200
    assert len(ledger.dispositions()) == 5


@pytest.mark.parametrize("fault", ["total", "calls", "attempts", "duplicate", "work_cap", "phase", "empty", "unknown", "bool_cap"])
def test_budget_contract_cannot_omit_work_or_move_credits(tmp_path, fault):
    c = fixture_contract(tmp_path/"fixture")
    if fault == "total": c["total_cap_ns"] += 1
    if fault == "calls": c["max_generated_forecasts"] += 1
    if fault == "attempts": c["max_attempts_per_item"] = 2
    if fault == "duplicate": c["workloads"].append(deepcopy(c["workloads"][0]))
    if fault == "work_cap": c["workloads"][0]["max_active_ns"] = 1001
    if fault == "phase": c["workloads"][0]["phase"] = "unknown"
    if fault == "empty": c["workloads"] = []
    if fault == "unknown": c["reset_on_resume"] = True
    if fault == "bool_cap": c["workloads"][0]["max_active_ns"] = True
    with pytest.raises(ValueError):
        budget.validate_contract(c)


def test_lock_is_released_after_real_process_exit_and_reservation_survives(tmp_path):
    """Short stdlib software child, no model/data/prediction workload."""
    directory = tmp_path/"native-fixture"
    c = fixture_contract(directory)
    with budget.Ledger.create(directory, c) as created:
        root = created.root_sha256
    script = """import os,sys
from experiments.pirc17.formal_budget import Ledger
from experiments.pirc17.protocol_core import digest
ledger=Ledger.open(sys.argv[1],expected_root_sha256=sys.argv[2])
ledger.reserve(digest(2))
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", script, str(directory), root], timeout=15,
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 17, result.stderr
    with budget.Ledger.open(directory, expected_root_sha256=root) as resumed:
        assert resumed.summary()["pending"] is not None
        assert resumed.summary()["remaining_ns_by_phase"]["forecast"] == 200
        with pytest.raises(RuntimeError):
            resumed.reserve(digest(3))


def test_control_credit_is_nonrefundable_separate_from_measured_work(ledger):
    key = digest("control")
    ledger.open_control(key, phase="fit", credit_ns=100)
    assert ledger.summary()["remaining_ns_by_phase"]["fit"] == 900
    settle(ledger, ledger.reserve(digest(0)), elapsed=150)
    ledger.close_control(key, observed_ns=40, evidence_sha256=digest("closed"), reason="phase_complete")
    summary = ledger.summary()
    assert summary["charged_ns_by_phase"]["fit"] == 250
    assert summary["measured_ns_by_phase"]["fit"] == 150
    assert summary["conservatively_charged_ns_by_phase"]["fit"] == 100
    assert summary["control_observed_ns_by_phase"]["fit"] == 40
    assert summary["control_charged_ns_by_phase"]["fit"] == 100
    assert summary["active_control_sha256"] is None
    assert summary["generated_forecasts_reserved"] == 0


def test_controller_cannot_cross_phases_or_overlap_spans(ledger):
    key = digest("control")
    ledger.open_control(key, phase="fit", credit_ns=100)
    with pytest.raises(ValueError, match="current control phase"):
        ledger.reserve(digest(2))
    with pytest.raises(ValueError, match="exclusive"):
        ledger.open_control(digest("other"), phase="forecast", credit_ns=50)
    reservation = ledger.reserve(digest(0))
    with pytest.raises(ValueError, match="idle phase"):
        ledger.topup_control(key, credit_ns=20, observed_ns=10)
    with pytest.raises(ValueError, match="idle current"):
        ledger.close_control(key, observed_ns=10, evidence_sha256=digest("proof"), reason="phase_complete")
    settle(ledger, reservation)


@pytest.mark.parametrize("value", [0, -1, True, .5, 1001])
def test_invalid_control_credit_never_mutates_state(ledger, value):
    old = ledger.tip
    with pytest.raises((ValueError, TimeoutError)):
        ledger.open_control(digest("control"), phase="fit", credit_ns=value)
    assert ledger.tip == old and ledger.summary()["active_control_sha256"] is None


def test_control_topups_cannot_retroactively_cover_expiry_or_roll_back_clock(ledger):
    key = digest("control")
    ledger.open_control(key, phase="fit", credit_ns=100)
    ledger.topup_control(key, credit_ns=50, observed_ns=80)
    before = ledger.tip
    for observed in (79, 150, 151):
        with pytest.raises(ValueError):
            ledger.topup_control(key, credit_ns=10, observed_ns=observed)
        assert ledger.tip == before
    assert ledger.summary()["control_charged_ns_by_phase"]["fit"] == 150


def test_control_replay_preserves_credit_and_current_span(ledger):
    key = digest("control")
    ledger.open_control(key, phase="fit", credit_ns=100)
    ledger.topup_control(key, credit_ns=50, observed_ns=80)
    expected, directory, root = ledger.summary(), ledger.directory, ledger.root_sha256
    ledger.close()
    with budget.Ledger.open(directory, expected_root_sha256=root) as reopened:
        assert reopened.summary() == expected
        reopened.close_control(key, observed_ns=100, evidence_sha256=digest("closed"), reason="phase_complete")
        with pytest.raises(ValueError):
            reopened.open_control(key, phase="fit", credit_ns=1)
        assert reopened.summary()["remaining_ns_by_phase"]["fit"] == 850


def test_unknown_control_interruption_is_terminal_not_free_overhead(ledger):
    key = digest("control")
    ledger.open_control(key, phase="fit", credit_ns=100)
    with pytest.raises(ValueError, match="unknown overhead"):
        ledger.close_control(key, observed_ns=None, evidence_sha256=digest("closed"), reason="phase_complete")
    ledger.close_control(key, observed_ns=None, evidence_sha256=digest("closed"), reason="recovered_unknown")
    assert ledger.summary()["charged_ns_by_phase"]["fit"] == 100
    assert ledger.summary()["halted_reason"] == "unknown_controller_interruption"
    with pytest.raises(TimeoutError): ledger.reserve(digest(0))


def test_late_closing_io_overrun_is_debited_once_and_halts(ledger):
    key = digest("control")
    ledger.open_control(key, phase="fit", credit_ns=100)
    ledger.close_control(key, observed_ns=80, evidence_sha256=digest("closed"), reason="phase_complete")
    ledger.retain_control_overrun(key, observed_ns=140, evidence_sha256=digest("late"))
    assert ledger.summary()["charged_ns_by_phase"]["fit"] == 140
    assert ledger.summary()["control_observed_ns_by_phase"]["fit"] == 140
    before = ledger.tip
    with pytest.raises(ValueError, match="newly observed"):
        ledger.retain_control_overrun(key, observed_ns=140, evidence_sha256=digest("late"))
    assert ledger.tip == before
    ledger.retain_control_overrun(key, observed_ns=150, evidence_sha256=digest("later"))
    assert ledger.summary()["charged_ns_by_phase"]["fit"] == 150
    assert ledger.summary()["halted_reason"] == "control_credit_exhausted"


def test_control_cannot_borrow_other_phase_credit(ledger):
    key = digest("control")
    with pytest.raises(TimeoutError, match="cannot expand"):
        ledger.open_control(key, phase="forecast", credit_ns=501)
    ledger.open_control(key, phase="forecast", credit_ns=500)
    assert ledger.summary()["remaining_ns_by_phase"]["fit"] == 1000
    assert ledger.summary()["halted_reason"] == "phase_cap_reached"


def test_terminal_tail_observes_prepaid_time_and_permanently_seals_admission(ledger):
    key = digest('last-control')
    ledger.open_control(key, phase='fit', credit_ns=100)
    ledger.close_control(key, observed_ns=10, evidence_sha256=digest('phase'), reason='phase_complete')
    ledger.finish_control_tail(key, observed_ns=80, evidence_sha256=digest('terminal-bytes'))
    summary = ledger.summary()
    assert summary['control_observed_ns_by_phase']['fit'] == 80
    assert summary['control_charged_ns_by_phase']['fit'] == 100
    assert summary['terminal_control_sha256'] == key
    assert summary['halted_reason'] is None
    with pytest.raises(ValueError, match='terminal execution'): ledger.reserve(digest(0))
    with pytest.raises(ValueError, match='terminal execution'):
        ledger.open_control(digest('reset'), phase='fit', credit_ns=1)
    directory, root = ledger.directory, ledger.root_sha256
    ledger.close()
    with budget.Ledger.open(directory, expected_root_sha256=root) as reopened:
        assert reopened.summary() == summary
        with pytest.raises(ValueError, match='terminal execution'): reopened.reserve(digest(0))


@pytest.mark.parametrize('elapsed', [100, 140])
def test_terminal_tail_at_or_beyond_credit_is_not_complete(ledger, elapsed):
    key = digest('tail')
    ledger.open_control(key, phase='fit', credit_ns=100)
    ledger.close_control(key, observed_ns=10, evidence_sha256=digest('phase'), reason='phase_complete')
    ledger.finish_control_tail(key, observed_ns=elapsed, evidence_sha256=digest('terminal'))
    assert ledger.summary()['charged_ns_by_phase']['fit'] == elapsed
    assert ledger.summary()['halted_reason'] == 'control_credit_exhausted'
    before = ledger.tip
    with pytest.raises(ValueError, match='backward'):
        ledger.finish_control_tail(key, observed_ns=elapsed-1, evidence_sha256=digest('false'))
    assert ledger.tip == before


def test_terminal_tail_cannot_charge_an_earlier_or_open_span(ledger):
    first, last = digest('first'), digest('last')
    ledger.open_control(first, phase='fit', credit_ns=100)
    with pytest.raises(ValueError, match='latest measured closed idle'):
        ledger.finish_control_tail(first, observed_ns=10, evidence_sha256=digest('proof'))
    ledger.close_control(first, observed_ns=10, evidence_sha256=digest('first-closed'), reason='phase_complete')
    ledger.open_control(last, phase='forecast', credit_ns=100)
    ledger.close_control(last, observed_ns=10, evidence_sha256=digest('last-closed'), reason='phase_complete')
    with pytest.raises(ValueError, match='latest measured closed idle'):
        ledger.finish_control_tail(first, observed_ns=50, evidence_sha256=digest('proof'))
