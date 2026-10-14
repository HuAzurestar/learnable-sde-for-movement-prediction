"""Durable-record boundary tests using small synthetic arrays only."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.pirc17 import resume_journal as journal


def workloads(count=4):
    return [{"sample_id": f"origin-{index}", "configuration": "base", "seed": 20260815,
             "particles": 4, "max_step_seconds": 2.5} for index in range(count)]


def make_writer(path, *, inherited_count=0, ancestry=None):
    return journal.Writer(path, workloads=workloads(), inherited_count=inherited_count,
        ancestry=ancestry or {"interrupted_forecast_sha256": "a"*64, "interruption_status": "failed"},
        contract={"software_fixture": True, "unchanged_full_workload_count": 4})


def arrays():
    return np.arange(32., dtype=float).reshape(4, 4, 2), np.ones((4, 2)), np.array([60., 300., 900., 1800.])


def row(index=0, *, status="success"):
    record = {"type": "run", **workloads()[index], "status": status, "actual_horizons_seconds": arrays()[2].tolist()}
    if status == "failure":
        record.update(error_type="MemoryError", error_message="software low memory")
    return record


def snapshot(writer):
    return journal.read(writer.directory, writer.manifest_sha256, expected_tip=writer.tip)


def test_exact_array_roundtrip_inherited_denominator_and_hash_chain(tmp_path):
    writer = make_writer(tmp_path/"attempt", inherited_count=2)
    original = row(2)
    first = writer.append(original, arrays())
    second = writer.append(row(3), arrays())
    assert "particle_artifact" not in original
    assert first["index"] == 2 and second["index"] == 3
    assert first["previous_record_sha256"] == writer.manifest_sha256
    assert second["previous_record_sha256"] == journal._hash(writer.directory/"records/000002.json")
    saved = snapshot(writer)
    assert saved.manifest["inherited_count"] == 2 and len(saved.manifest["workloads"]) == 4
    assert saved.records == [first, second] and saved.failure_count == 0 and not saved.uncommitted_files
    assert saved.tip["record_count"] == 2 and not saved.manifest["certified"]
    restored = journal.load_particle_evidence(first["row"], writer.directory)
    assert all(np.array_equal(a, b) for a, b in zip(restored, arrays()))
    with pytest.raises(ValueError, match="denominator"):
        writer.append(row(3), arrays())


def test_no_reopening_clobbering_or_direct_ancestor_particle_import(tmp_path):
    writer = make_writer(tmp_path/"attempt")
    before = (writer.directory/"manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        make_writer(writer.directory)
    with pytest.raises(FileExistsError):
        journal._publish_json(writer.directory/"manifest.json", {"replacement": True})
    assert (writer.directory/"manifest.json").read_bytes() == before
    inherited = dict(row(), particle_artifact={"path": "old.npz", "sha256": "b"*64})
    with pytest.raises(ValueError, match="import"):
        writer.append(inherited, arrays())
    assert not snapshot(writer).records


def test_inflight_particle_without_record_is_not_a_completed_run(tmp_path, monkeypatch):
    writer = make_writer(tmp_path/"interrupted")
    publish = journal._publish_json
    def interrupted(path, value):
        if path.parent.name == "records":
            raise KeyboardInterrupt("software interruption after particle publication")
        return publish(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(journal, "_publish_json", interrupted)
        with pytest.raises(KeyboardInterrupt):
            writer.append(row(), arrays())
    saved = snapshot(writer)
    assert saved.tip["record_count"] == 0 and not saved.records
    assert [value["path"] for value in saved.uncommitted_files] == ["particles/000000.npz"]
    orphan_bytes = (writer.directory/"particles/000000.npz").read_bytes()
    with pytest.raises(RuntimeError, match="new directory"):
        writer.append(row(), arrays())
    following = make_writer(tmp_path/"new-attempt", ancestry={"closed_parent_tip": saved.tip,
        "retained_uncommitted_files": saved.uncommitted_files})
    following.append(row(), arrays())
    assert len(snapshot(following).records) == 1
    assert (writer.directory/"particles/000000.npz").read_bytes() == orphan_bytes


def test_hard_process_exit_keeps_unpublished_temp_but_no_false_commit(tmp_path):
    writer = make_writer(tmp_path/"killed-writer")
    # This harmless owned child never predicts, accesses empirical data, or
    # targets another PID. os._exit simulates interruption without finally.
    code = """
import os, sys
from pathlib import Path
from experiments.pirc17.resume_journal import _publish
def die(target):
    target.write(b'partial software fixture')
    target.flush()
    os.fsync(target.fileno())
    os._exit(17)
_publish(Path(sys.argv[1]), die)
"""
    import os
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run([sys.executable, "-c", code, str(writer.directory/"records/000000.json")],
        capture_output=True, text=True, timeout=60, creationflags=flags)
    assert result.returncode == 17, result.stderr
    saved = snapshot(writer)
    assert saved.tip["record_count"] == 0 and len(saved.uncommitted_files) == 1
    assert saved.uncommitted_files[0]["path"].endswith(".pending")
    assert not (writer.directory/"records/000000.json").exists()


def test_atomic_no_replace_race_has_exactly_one_winner(tmp_path):
    path = tmp_path/"record.json"
    def publish(value):
        try:
            journal._publish_json(path, {"writer": value})
            return value
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, [1, 2]))
    winners = [value for value in outcomes if value is not None]
    assert len(winners) == 1 and json.loads(path.read_text()) == {"writer": winners[0]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]


@pytest.mark.parametrize("fault", ["manifest", "index", "previous", "extra", "row_key", "float_seed", "status", "particle_hash", "particle_path"])
def test_rehashed_record_tampering_still_fails_closed(tmp_path, fault):
    writer = make_writer(tmp_path/"attempt")
    writer.append(row(), arrays())
    path = writer.directory/"records/000000.json"
    record = json.loads(path.read_text())
    if fault == "manifest": record["manifest_sha256"] = "b"*64
    if fault == "index": record["index"] = True
    if fault == "previous": record["previous_record_sha256"] = "b"*64
    if fault == "extra": record["unregistered"] = 1
    if fault == "row_key": record["row"]["sample_id"] = "different"
    if fault == "float_seed": record["row"]["seed"] = 20260815.
    if fault == "status": record["row"]["status"] = "partial"
    if fault == "particle_hash": record["row"]["particle_artifact"]["sha256"] = "b"*64
    if fault == "particle_path": record["row"]["particle_artifact"]["path"] = "../outside.npz"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        journal.read(writer.directory, writer.manifest_sha256)


@pytest.mark.parametrize("removed", [0, 1])
def test_pinned_tip_catches_both_middle_and_last_record_loss(tmp_path, removed):
    writer = make_writer(tmp_path/"attempt")
    writer.append(row(), arrays())
    writer.append(row(1), arrays())
    tip = writer.tip
    (writer.directory/f"records/{removed:06d}.json").unlink()
    with pytest.raises(ValueError):
        journal.read(writer.directory, writer.manifest_sha256, expected_tip=tip)


def test_failure_record_and_original_error_cannot_disappear_from_progress(tmp_path):
    writer = make_writer(tmp_path/"attempt")
    failed = writer.append(row(status="failure"))
    writer.append(row(1), arrays())
    saved = snapshot(writer)
    assert saved.failure_count == 1 and saved.records[0] == failed
    assert saved.tip["record_count"] == 2 and len(saved.manifest["workloads"]) == 4
    assert not (writer.directory/"particles/000000.npz").exists()
    assert saved.records[0]["row"]["error_message"] == "software low memory"


@pytest.mark.parametrize("fault", ["out_of_order", "arrays_missing", "complex", "nan", "shape", "times", "failure_arrays", "failure_error"])
def test_invalid_new_records_leave_writer_and_storage_untouched(tmp_path, fault):
    writer = make_writer(tmp_path/"attempt")
    record, values = row(), arrays()
    if fault == "out_of_order": record = row(1)
    if fault == "arrays_missing": values = None
    if fault == "complex": values = (values[0].astype(complex), *values[1:])
    if fault == "nan": values[0][0, 0, 0] = np.nan
    if fault == "shape": values = (values[0][:2], *values[1:])
    if fault == "times": record["actual_horizons_seconds"][0] += 1
    if fault == "failure_arrays": record = row(status="failure")
    if fault == "failure_error": record, values = row(status="failure"), None; record.pop("error_type")
    with pytest.raises(ValueError):
        writer.append(record, values)
    assert not snapshot(writer).records and not snapshot(writer).uncommitted_files
    writer.append(row(), arrays())
    assert len(snapshot(writer).records) == 1


def test_manifest_isolated_from_mutable_inputs_and_requires_full_unique_workload(tmp_path):
    keys = workloads()
    writer = journal.Writer(tmp_path/"attempt", workloads=keys, inherited_count=0,
        ancestry={"parent": "a"*64}, contract={"software_only": True})
    keys[0]["sample_id"] = "changed-after-reservation"
    writer.append(row(), arrays())
    assert snapshot(writer).manifest["workloads"] == workloads()
    invalid = workloads(); invalid[1] = deepcopy(invalid[0])
    for workload, inherited in ((invalid, 0), (workloads(), -1), (workloads(), 5), ([], 0), (workloads(), True)):
        with pytest.raises(ValueError):
            journal.Writer(tmp_path/"not-created", workloads=workload, inherited_count=inherited,
                ancestry={"parent": "a"*64}, contract={"software_only": True})
        assert not (tmp_path/"not-created").exists()


def test_storage_failure_poisoning_prevents_in_place_retry(tmp_path, monkeypatch):
    writer = make_writer(tmp_path/"attempt")
    def fail(*args):
        raise OSError("software fsync failure")
    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "fsync", fail)
        with pytest.raises(OSError):
            writer.append(row(), arrays())
    assert not snapshot(writer).records and not snapshot(writer).uncommitted_files
    with pytest.raises(RuntimeError):
        writer.append(row(), arrays())


def test_committed_but_unacknowledged_record_is_found_by_closed_inventory(tmp_path, monkeypatch):
    writer = make_writer(tmp_path/"attempt")
    publish = journal._publish_json
    def lose_acknowledgement(path, value):
        publish(path, value)
        raise KeyboardInterrupt("software exit after atomic commit but before counter update")
    with monkeypatch.context() as patch:
        patch.setattr(journal, "_publish_json", lose_acknowledgement)
        with pytest.raises(KeyboardInterrupt):
            writer.append(row(), arrays())
    saved = journal.read(writer.directory, writer.manifest_sha256)
    assert len(saved.records) == saved.tip["record_count"] == 1 and not saved.uncommitted_files
    assert writer.tip["record_count"] == 0  # Never use an in-memory counter as closure proof.
    with pytest.raises(ValueError, match="pinned"):
        snapshot(writer)
    with pytest.raises(RuntimeError):
        writer.append(row(), arrays())


def test_unsupported_publication_fails_without_an_overwrite_fallback(tmp_path, monkeypatch):
    writer = make_writer(tmp_path/"attempt")
    def unsupported(*args):
        raise OSError("software filesystem does not support hard links")
    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "link", unsupported)
        with pytest.raises(OSError, match="hard links"):
            writer.append(row(), arrays())
    assert not snapshot(writer).records and not snapshot(writer).uncommitted_files


def test_evidence_changing_during_snapshot_is_rejected(tmp_path, monkeypatch):
    writer = make_writer(tmp_path/"attempt")
    writer.append(row(), arrays())
    original_hash = journal._hash
    def changed(path):
        return "f"*64 if Path(path).name == "manifest.json" else original_hash(path)
    monkeypatch.setattr(journal, "_hash", changed)
    with pytest.raises(ValueError, match="changed during"):
        snapshot(writer)
