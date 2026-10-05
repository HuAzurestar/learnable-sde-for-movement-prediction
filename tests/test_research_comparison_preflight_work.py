"""Bound the real owner preflight, not worker time or disclosure permission.

The entry fault below stops before any supervised job. It measures actual
metadata reads and verifies that no preflight snapshot crosses that boundary.
Original computation/deadline and package-publication tests remain separate.
"""
import pytest

from application.research_comparison import ComparisonRunner
from application.research_supervisor import ResearchSupervisor
from infrastructure.research_store import ResearchError
from tests.test_research_comparison import source, compute


class BeforeSupervision(Exception):
    pass


def test_comparison_preflight_verifies_one_initial_and_one_final_chain(source, monkeypatch):
    store = source[0]
    initial_events = len(store.events())
    original_json, reads, boundary = store._json, [], []

    def read(path):
        if path.parent == store.path / "events":
            reads.append(path)
        return original_json(path)

    def stop(supervisor, attempt_id, command, budget, **kwargs):
        assert supervisor.store is store and budget.job_seconds == 10
        assert store._read_snapshot() is None, "preflight lock must not cover supervision"
        boundary.append(len(reads))
        raise BeforeSupervision

    monkeypatch.setattr(store, "_json", read)
    monkeypatch.setattr(ResearchSupervisor, "run", stop)
    with pytest.raises(BeforeSupervision):
        compute(source)
    final_events = store.events()
    assert len(boundary) == 1
    assert boundary[0] == initial_events + len(final_events), (
        f"preflight physically reread {boundary[0]} event files; retain exactly "
        f"the initial {initial_events} and final {len(final_events)} complete prefixes")
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in final_events)


def test_preflight_final_chain_corruption_cannot_reach_supervision(source, monkeypatch):
    store = source[0]
    original_attempt, corrupted, entered = store.new_attempt, [], []

    def new_attempt(*args, **kwargs):
        result = original_attempt(*args, **kwargs)
        corrupted.append(True)
        (store.path / "events/0000000000000001.json").write_bytes(b"{}")
        return result

    def unexpected(*args, **kwargs):
        entered.append(True)
        raise BeforeSupervision

    monkeypatch.setattr(store, "new_attempt", new_attempt)
    monkeypatch.setattr(ResearchSupervisor, "run", unexpected)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        compute(source)
    assert corrupted == [True] and not entered
    assert store._read_snapshot() is None
