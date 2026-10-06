"""Durable checkpoint events share only their final owned head publication.

No event fsync, current authority or final physical verification is deferred.
Actual abrupt process exit demonstrates recovery from the already-supported
lagging head, rather than treating an in-memory snapshot as crash evidence.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import infrastructure.research_store as store_module
from infrastructure.research_store import ResearchError, ResearchStore, encode
from tests.research_audit_fixtures import audit_fault, payload_opens
from tests.test_research_artifact_read_bounds import published
from tests.test_research_checkpoint_save_work import owner


def released(store):
    assert store._read_snapshot() is None
    assert not hasattr(store._read_scope, "checkpoint_head")


@pytest.mark.parametrize("level", ["exact", "chunk"])
def test_actual_save_publishes_one_final_head_without_deferring_events(tmp_path, monkeypatch, level):
    store, recovery, attempt, receipt, state, progress, _ = owner(tmp_path, level)
    before = store.events()
    original_write, original_sync = store_module.atomic_write, os.fsync
    writes, flushes, current = [], [], []

    def write(path, content, **options):
        current.append(Path(path))
        try:
            result = original_write(path, content, **options)
            writes.append((Path(path), content))
            # Observe the actual published event before the writer returns.
            if Path(path).parent == store.path / "events":
                assert Path(path).read_bytes() == content
            return result
        finally:
            current.pop()

    def sync(fd):
        result = original_sync(fd)
        if current and os.fstat(fd).st_size > 0:
            flushes.append(current[-1])
        return result

    monkeypatch.setattr(store_module, "atomic_write", write)
    monkeypatch.setattr(os, "fsync", sync)
    reference = recovery.checkpoint_handler(attempt)(state, progress, receipt, None)
    after = store.events()
    events = [path for path, _ in writes if path.parent == store.path / "events"]
    heads = [content for path, content in writes if path == store.path / "head.json"]
    assert [event["event_kind"] for event in after[len(before):]] == ["MANIFEST", "CHECKPOINT"]
    assert len(events) == 2 and all(path in flushes for path in events)
    assert len(heads) == 1, "one owned save republishes intermediate journal heads"
    assert json.loads(heads[0]) == {"sequence": after[-1]["sequence"], "hash": after[-1]["hash"]}
    assert store.path / "head.json" in flushes
    assert after[-1]["payload"]["artifact_id"] == reference["artifact_id"]
    released(store)


@pytest.mark.parametrize("stage", ["event-fsync", "head-fsync"])
def test_actual_save_io_failure_retains_chain_and_releases_owned_phase(tmp_path, monkeypatch, stage):
    store, recovery, attempt, receipt, state, progress, _ = owner(tmp_path, "exact")
    before = store.events()
    with audit_fault(store, "CHECKPOINT", monkeypatch, stage=stage) as fired:
        with pytest.raises(OSError):
            recovery.checkpoint_handler(attempt)(state, progress, receipt, None)
    assert len(fired) == 1
    released(store)
    reopened = ResearchStore(tmp_path / "owner", "live-checkpoint")
    after = reopened.events()
    assert after[:len(before)] == before
    assert after[len(before)]["event_kind"] == "MANIFEST"
    assert (after[-1]["event_kind"] == "CHECKPOINT") == (stage == "head-fsync")
    next_event = reopened.append("AFTER_FAILURE", {"fixture": True})
    assert next_event["sequence"] == len(after) + 1
    assert reopened.events()[-2] == after[-1]


def test_corruption_during_final_head_still_refuses_save_return(tmp_path, monkeypatch):
    store, recovery, attempt, receipt, state, progress, _ = owner(tmp_path, "exact")
    original, corrupted = store_module.atomic_write, []

    def write(path, content, **options):
        result = original(path, content, **options)
        if Path(path) == store.path / "head.json":
            (store.path / "events" / "0000000000000001.json").write_bytes(b"{}")
            corrupted.append(True)
        return result

    monkeypatch.setattr(store_module, "atomic_write", write)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        recovery.checkpoint_handler(attempt)(state, progress, receipt, None)
    assert corrupted
    released(store)


def test_exception_after_durable_checkpoint_does_not_erase_events(tmp_path, monkeypatch):
    store, recovery, attempt, receipt, state, progress, _ = owner(tmp_path, "exact")
    original, before = store.append, store.events()

    def append(kind, *args, **kwargs):
        result = original(kind, *args, **kwargs)
        if kind == "CHECKPOINT":
            raise LookupError("actual checkpoint published before owner interruption")
        return result

    monkeypatch.setattr(store, "append", append)
    with pytest.raises(LookupError, match="owner interruption"):
        recovery.checkpoint_handler(attempt)(state, progress, receipt, None)
    released(store)
    reopened = ResearchStore(tmp_path / "owner", "live-checkpoint")
    assert reopened.events()[:len(before)] == before
    assert [event["event_kind"] for event in reopened.events()[len(before):]] == ["MANIFEST", "CHECKPOINT"]
    head = json.loads((reopened.path / "head.json").read_bytes())
    assert head["sequence"] <= len(reopened.events())


def test_actual_abrupt_owner_exit_recovers_all_events_from_original_head(tmp_path):
    program = r'''
import json, os, sys
from pathlib import Path
from tests.test_research_checkpoint_save_work import owner
store, recovery, attempt, receipt, state, progress, _ = owner(Path(sys.argv[1]), 'exact')
head = json.loads((store.path / 'head.json').read_bytes())
print(json.dumps({'head': head, 'events': len(store.events())}), flush=True)
original = store._write_journal
def write(kind, path, content):
    result = original(kind, path, content)
    if kind == 'CHECKPOINT' and path.parent == store.path / 'events':
        os._exit(17)  # AFTER actual event fsync/rename, BEFORE final head.
    return result
store._write_journal = write
recovery.checkpoint_handler(attempt)(state, progress, receipt, None)
raise AssertionError('actual owner crash hook was not reached')
'''
    result = subprocess.run([sys.executable, "-B", "-c", program, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 17, result.stderr
    original = json.loads(result.stdout)
    reopened = ResearchStore(tmp_path / "owner", "live-checkpoint")
    events = reopened.events()
    assert len(events) == original["events"] + 2
    assert [event["event_kind"] for event in events[-2:]] == ["MANIFEST", "CHECKPOINT"]
    assert json.loads((reopened.path / "head.json").read_bytes()) == original["head"], \
        "intermediate head was republished before the owned save completed"
    next_event = reopened.append("AFTER_CRASH", {"fixture": True})
    assert next_event["sequence"] == len(events) + 1
    assert reopened.events()[-2] == events[-1]


def test_nested_owner_save_and_saved_event_share_final_head(tmp_path, monkeypatch):
    """Owner-publication unit control, not live supervisor deadline evidence."""
    store, recovery, attempt, receipt, state, progress, _ = owner(tmp_path, "exact")
    original, heads = store_module.atomic_write, []

    def write(path, content, **options):
        result = original(path, content, **options)
        if Path(path) == store.path / "head.json":
            heads.append(json.loads(content))
        return result

    monkeypatch.setattr(store_module, "atomic_write", write)
    with store._checkpoint_publication():
        reference = recovery.checkpoint_handler(attempt)(state, progress, receipt, None)
        assert not heads, "nested handler published a head before the owner save phase ended"
        saved = store.append("CHECKPOINT_SAVED", {"attempt_id": attempt,
            "request_id": "synthetic-owner-phase", **reference, "progress": progress})
        assert not heads
    assert heads == [{"sequence": saved["sequence"], "hash": saved["hash"]}]
    assert [event["event_kind"] for event in store.events()[-3:]] == ["MANIFEST", "CHECKPOINT", "CHECKPOINT_SAVED"]
    released(store)


def test_ordinary_read_transaction_does_not_coalesce_checkpoint_named_events(tmp_path, monkeypatch):
    store, _, _ = published(tmp_path)
    original, heads = store_module.atomic_write, []

    def write(path, content, **options):
        result = original(path, content, **options)
        if Path(path) == store.path / "head.json":
            heads.append(json.loads(content))
        return result

    monkeypatch.setattr(store_module, "atomic_write", write)
    with store._read_transaction():
        first = store.append("CHECKPOINT", {"synthetic_store_control": 1})
        assert len(heads) == 1 and heads[-1]["hash"] == first["hash"]
        second = store.append("CHECKPOINT_SAVED", {"synthetic_store_control": 2})
        assert len(heads) == 2 and heads[-1]["hash"] == second["hash"]
    released(store)


def test_actual_read_audit_head_failure_is_immediate_inside_checkpoint_phase(tmp_path, monkeypatch):
    store, artifact, grant = published(tmp_path)
    with payload_opens([store.path / "artifacts" / artifact["artifact_id"]]) as opened:
        with store._checkpoint_publication():
            store.append("CHECKPOINT", {"synthetic_store_control": True})
            with audit_fault(store, "READ_STARTED", monkeypatch, stage="head-fsync") as fired:
                with pytest.raises(ResearchError, match="EXPOSURE_AUDIT_UNAVAILABLE"):
                    store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=grant)
                assert len(fired) == 1 and not opened
    assert [event["event_kind"] for event in store.events()[-3:]] == ["CHECKPOINT", "EXPOSURE_ALLOWED", "READ_STARTED"]
    released(store)


def test_caught_immediate_head_failure_cannot_republish_older_pending_head(tmp_path, monkeypatch):
    store, _, _ = published(tmp_path)
    initial = json.loads((store.path / "head.json").read_bytes())
    original, attempts = store_module.atomic_write, []

    def write(path, content, **options):
        if Path(path) == store.path / "head.json":
            attempts.append(json.loads(content))
        return original(path, content, **options)

    monkeypatch.setattr(store_module, "atomic_write", write)
    with store._checkpoint_publication():
        store.append("CHECKPOINT", {"synthetic_store_control": True})
        with audit_fault(store, "READ_STARTED", monkeypatch, stage="head-fsync") as fired:
            with pytest.raises(ResearchError, match="EXPOSURE_AUDIT_UNAVAILABLE"):
                store.append("READ_STARTED", {"synthetic_store_control": True})
        assert fired
    events = store.events()
    assert len(attempts) == 1 and attempts[0]["hash"] == events[-1]["hash"]
    assert json.loads((store.path / "head.json").read_bytes()) == initial
    assert [event["event_kind"] for event in events[-2:]] == ["CHECKPOINT", "READ_STARTED"]
    released(store)
