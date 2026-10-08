"""Ordered repeated continuation on synthetic data; no empirical launches."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

from experiments.pirc17 import seed_resume_lineage as lineage
from experiments.pirc17 import seed_resume_session as session
from tests.test_pirc17_seed_resume_execution import (
    candidate_bytes, evidence, inputs, qualification, seed_qualification, closed_seed,
    resume_input, execution, assembled)


def reserve(args):
    return lineage.reserve(source=args["source"], locations={key: args[key] for key in lineage.LOCATION_ARGUMENTS})


def current_args(args, reservation):
    return dict(args, output=reservation["attempt"]/"work", resume_directory=reservation["directory"],
                root_sha256=reservation["root_sha256"], attempt_index=reservation["index"])


def seal_software(reservation, *, interrupted, returncode):
    # This fixture's synchronous function has already returned/unwound. These
    # synthetic process facts test lineage, NOT native job-process supervision.
    elapsed = (lineage.datetime.now(lineage.timezone.utc)-lineage._date(reservation["start"]["started_at"])).total_seconds()
    process = {"worker_pid": os.getpid(), "worker_returncode": returncode, "process_tree_closed": True,
        "accounting": {"active_processes": 0, "total_processes": 1}, "timed_out": False,
        "job_name": reservation["start"]["job_name"],
        "interrupted": interrupted, "supervisor_error": None, "elapsed_seconds": elapsed}
    lineage.journal._publish_json(reservation["attempt"]/"work-process.json", process)
    return lineage.seal(reservation, elapsed=elapsed, returncode=returncode,
                        tree_closed=True, interrupted=interrupted)


def interrupt_after(count):
    def progress(event):
        if event["phase"] == "run_committed" and event["new_committed"] == count:
            raise KeyboardInterrupt("synthetic user stop")
    return progress


def test_three_attempts_compute_missing_ten_once_with_full_exact_ancestry(resume_input):
    args, old_paths, prefix, complete, tracker = resume_input
    before = len(tracker["calls"])
    records = []
    for cut in (3, 2):
        reservation = reserve(args)
        with pytest.raises(KeyboardInterrupt):
            execution.run(**current_args(args, reservation), progress=interrupt_after(cut))
        closed = seal_software(reservation, interrupted=True, returncode=130)
        assert closed["continuable"] and closed["new_committed_count"] == cut
        records.append(closed)
    reservation = reserve(args)
    final_args = current_args(args, reservation)
    result = execution.run(**final_args)
    assert result["status"] == "computed_pending_supervisory_closure", result
    assert result["inherited_success_count"] == 19 and result["new_committed_count"] == 5
    assert len(tracker["calls"])-before == 10
    assert result["resources"]["prior_elapsed_upper_bound_seconds"] == math.fsum([180, *[record["elapsed_seconds"] for record in records]])
    assert result["resources"]["cooperative_remaining_seconds"] < 120
    closure = seal_software(reservation, interrupted=False, returncode=0)
    assert closure["status"] == "computed_closed" and not closure["continuable"]
    rows, saved = assembled(final_args)
    assert len(saved.manifest["ancestry"]["continuation"]["prior_closures"]) == 2
    assert len(set(saved.manifest["ancestry"]["inherited_directories"])) == 3
    for row, baseline in zip(rows, complete[2:-1]):
        x = execution.load_particle_evidence(row, final_args["output"])
        y = execution.load_particle_evidence(baseline, old_paths["envelope"].parent)
        assert all(np.array_equal(a, b) for a, b in zip(x, y))
        assert row["scores"] == baseline["scores"]
        assert row["particle_precision"] == baseline["particle_precision"]
        assert row["brownian_identity"] == baseline["brownian_identity"]
    _, closed, _, _ = lineage.history(reservation["directory"], reservation["root_sha256"])
    assert len(closed) == 3
    independently_checked = session.inspect_closed(reservation["directory"], reservation["root_sha256"])
    assert independently_checked["status"] == "verified_closed_whole_workload"
    assert independently_checked["attempt_count"] == 3 and independently_checked["new_committed_count"] == 10
    assert independently_checked["legacy_inherited_count"] == 14
    assert independently_checked["reference_numerical"]["out_of_tolerance"] > 0
    assert not independently_checked["numerically_qualified"] and not independently_checked["resource_certified"]
    with pytest.raises(ValueError, match="no automatic retry"):
        reserve(args)


def test_live_reservation_blocks_second_launcher_and_worker_path_forks(resume_input):
    args, _, _, _, tracker = resume_input
    reservation = reserve(args)
    before = len(tracker["calls"])
    with pytest.raises(FileExistsError, match="unclosed"):
        reserve(args)
    with pytest.raises(ValueError, match="output"):
        execution.run(**dict(current_args(args, reservation), output=args["output"]))
    with pytest.raises(ValueError, match="sole latest"):
        lineage.history(reservation["directory"], reservation["root_sha256"], current_index=1)
    assert len(tracker["calls"]) == before


def test_interrupted_before_output_charges_attempt_and_can_continue(resume_input):
    args, _, _, _, _ = resume_input
    first = reserve(args)
    closure = seal_software(first, interrupted=True, returncode=130)
    assert closure["continuable"] and closure["journal_tip"] is None
    second = reserve(args)
    assert second["start"]["inherited_count"] == 14
    assert second["start"]["prior_elapsed_seconds"] == 180+closure["elapsed_seconds"]
    result = execution.run(**current_args(args, second))
    assert result["status"] == "computed_pending_supervisory_closure", result


def test_deleted_tail_or_modified_closed_inventory_cannot_be_reused(resume_input):
    args, _, _, _, tracker = resume_input
    first = reserve(args)
    with pytest.raises(KeyboardInterrupt):
        execution.run(**current_args(args, first), progress=interrupt_after(2))
    seal_software(first, interrupted=True, returncode=130)
    path = first["attempt"]/"work/journal/records/000015.json"
    saved_bytes = path.read_bytes()
    path.unlink()
    before = len(tracker["calls"])
    with pytest.raises(ValueError, match="inventory"):
        reserve(args)
    path.write_bytes(saved_bytes)
    source = first["attempt"]/"work-process.json"
    changed = json.loads(source.read_text())
    changed["accounting"]["active_processes"] = 1
    source.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        reserve(args)
    assert len(tracker["calls"]) == before


def test_failed_scientific_row_stays_failed_and_forbids_automatic_retry(resume_input, monkeypatch):
    args, _, _, _, _ = resume_input
    first = reserve(args)
    def fail(*a, **k):
        raise MemoryError("synthetic resource failure inside prediction")
    monkeypatch.setattr(execution.engine, "rollout", fail)
    result = execution.run(**current_args(args, first))
    assert result["new_failure_count"] == 1
    closure = seal_software(first, interrupted=False, returncode=1)
    assert closure["status"] == "failed" and not closure["continuable"]
    with pytest.raises(ValueError, match="no automatic retry"):
        reserve(args)


def test_seal_never_accepts_a_live_process_tree(resume_input):
    args, _, _, _, _ = resume_input
    reservation = reserve(args)
    with pytest.raises(ValueError, match="process-tree closure"):
        lineage.seal(reservation, elapsed=1., returncode=130, tree_closed=False, interrupted=True)
    assert not (reservation["attempt"]/"closure.json").exists()


def test_stop_request_binds_only_latest_live_attempt(resume_input):
    args, _, _, _, _ = resume_input
    first = reserve(args)
    assert not session._stop_requested(first)
    session.request_stop(first["directory"], first["root_sha256"])
    assert session._stop_requested(first)
    with pytest.raises(FileExistsError):
        session.request_stop(first["directory"], first["root_sha256"])
    seal_software(first, interrupted=True, returncode=130)
    second = reserve(args)
    assert not session._stop_requested(second)


@pytest.mark.skipif(os.name != "nt", reason="native Windows ownership contract")
def test_native_job_normal_completion_and_hard_cap_kill_descendants():
    completed = session.run_owned([sys.executable, "-c", "raise SystemExit(0)"], timeout=10)
    assert completed["worker_returncode"] == 0 and completed["process_tree_closed"]
    assert completed["accounting"]["total_processes"] >= 2
    assert not completed["timed_out"] and not completed["supervisor_error"]
    # Parent exits immediately, leaving an actual child behind. Waiting for
    # the launcher alone would incorrectly call this complete.
    command = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], creationflags=subprocess.CREATE_NO_WINDOW)"
    timed = session.run_owned([sys.executable, "-c", command], timeout=2)
    assert timed["timed_out"] and timed["process_tree_closed"]
    assert timed["accounting"]["active_processes"] == 0
    assert timed["accounting"]["total_processes"] >= 3
    assert timed["elapsed_seconds"] < 10


@pytest.mark.skipif(os.name != "nt", reason="native Windows ownership contract")
def test_native_stop_terminates_only_owned_job():
    begin = time.perf_counter()
    outcome = session.run_owned([sys.executable, "-c", "import time; time.sleep(30)"], timeout=10,
                              stop_requested=lambda: time.perf_counter()-begin > .8)
    assert outcome["interrupted"] and not outcome["timed_out"]
    assert outcome["process_tree_closed"] and outcome["worker_returncode"] != 0


def test_nonzero_prefix_slice_requires_valid_offset(resume_input):
    args, paths, prefix, complete, _ = resume_input
    source = args["source"]
    validated = execution.admission.inspect(**source)
    reference = execution.admission.read_closed(source["reference_ledger"], validated.spec["reference_ledger_sha256"], jsonl=True)
    for offset in (True, -1, 25, 1.5):
        with pytest.raises(ValueError, match="denominator"):
            execution.admission.validate_prefix_rows(validated.spec, reference, prefix[:2], paths["forecast"].parent,
                Path(source["reference_ledger"]).parent, guard=lambda: None, offset=offset)


def test_concurrent_reservations_have_one_atomic_winner(resume_input, monkeypatch):
    args, _, _, _, _ = resume_input
    directory, _, _ = lineage.initialize(source=args["source"],
        locations={key: args[key] for key in lineage.LOCATION_ARGUMENTS})
    original = lineage._attempts
    barrier = threading.Barrier(2)
    def simultaneous(path):
        result = original(path)
        barrier.wait(timeout=15)
        return result
    monkeypatch.setattr(lineage, "_attempts", simultaneous)
    def attempt():
        try:
            return reserve(args)
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert sum(value is not None for value in outcomes) == 1
    assert [path.name for path in (directory/"attempts").iterdir()] == ["000000"]


def test_cumulative_elapsed_can_exhaust_engine_cap_before_outer_cap(resume_input, monkeypatch):
    args, _, _, _, _ = resume_input
    first = reserve(args)
    real_datetime = lineage.datetime
    class Later(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz)+timedelta(seconds=121)
    monkeypatch.setattr(lineage, "datetime", Later)
    closed = seal_software(first, interrupted=True, returncode=130)
    assert closed["continuable"] and closed["elapsed_seconds"] >= 121
    with pytest.raises(TimeoutError, match="no renewal"):
        reserve(args)
    assert not (first["directory"]/"attempts/000001").exists()


def test_independent_consumer_rejects_an_interrupted_prefix(resume_input):
    args, _, _, _, _ = resume_input
    first = reserve(args)
    with pytest.raises(KeyboardInterrupt):
        execution.run(**current_args(args, first), progress=interrupt_after(1))
    seal_software(first, interrupted=True, returncode=130)
    with pytest.raises(ValueError, match="complete supervisory closure"):
        session.inspect_closed(first["directory"], first["root_sha256"])


@pytest.mark.skipif(os.name != "nt", reason="native Windows ownership contract")
def test_real_worker_cli_rejects_synthetic_map_contract_and_closes_owned_tree(resume_input):
    args, _, _, _, tracker = resume_input
    before = len(tracker["calls"])
    # The parent's fixture uses synthetic maps. That monkeypatch intentionally
    # does not exist in the actual subprocess: scientific admission must fail
    # before it can predict or produce a journal, then the supervisor must seal
    # a failed whole-process closure, not permit an automatic retry.
    result = session.supervise(source=args["source"],
        locations={key: args[key] for key in lineage.LOCATION_ARGUMENTS})
    assert result == 1 and len(tracker["calls"]) == before
    directory = lineage.directory_for(args["source"])
    root_sha = lineage.native._hash(directory/"root.json")
    _, closed, _, _ = lineage.history(directory, root_sha)
    assert len(closed) == 1 and closed[0]["closure"]["status"] == "failed"
    assert not closed[0]["closure"]["continuable"] and closed[0]["closure"]["process_tree_closed"]
    assert closed[0]["closure"]["journal_tip"] is None
    process = json.loads((closed[0]["directory"]/"work-process.json").read_text())
    assert process["accounting"]["active_processes"] == 0 and process["accounting"]["total_processes"] >= 2
    assert not (closed[0]["directory"]/"work").exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows ownership contract")
def test_direct_worker_cannot_bypass_owned_job_even_with_exact_reservation(resume_input):
    args, _, _, _, tracker = resume_input
    reservation = reserve(args)
    before = len(tracker["calls"])
    with pytest.raises(PermissionError, match="live reserved process job"):
        session.worker(reservation["directory"], reservation["root_sha256"], reservation["index"], reservation["start_sha256"])
    job = session.OwnedJob(reservation["start"]["job_name"])
    try:
        # Existing named job is not enough: this process is not a member.
        with pytest.raises(PermissionError, match="live reserved process job"):
            session.worker(reservation["directory"], reservation["root_sha256"], reservation["index"], reservation["start_sha256"])
        with pytest.raises(FileExistsError, match="existing process job"):
            session.OwnedJob(reservation["start"]["job_name"])
        assert job.accounting()["active_processes"] == 0
    finally:
        job.close()
    assert len(tracker["calls"]) == before and not (reservation["attempt"]/"work").exists()
    command = "from experiments.pirc17.seed_resume_session import require_owned_job; require_owned_job(%r)" % reservation["start"]["job_name"]
    owned = session.run_owned([sys.executable, "-c", command], timeout=20, job_name=reservation["start"]["job_name"])
    assert owned["worker_returncode"] == 0 and owned["process_tree_closed"] and not owned["supervisor_error"]
