"""Measure actual result publication work, excluding the worker lifetime."""

import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_supervisor import ResearchSupervisor
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import digest
from tests.research_file_observation import observe_file
from tests.test_research_budget import registered
from tests.test_research_live_checkpoint import prepared


@pytest.mark.parametrize("admitted", [False, True])
def test_actual_result_publication_verifies_at_most_two_physical_chains(tmp_path, monkeypatch, admitted):
    if admitted:
        store, value, registry, _, _ = prepared(tmp_path / "admitted", "exact", "admitted")
    else:
        store, attempts = registered(tmp_path)
    original_json, original_settle, original_run = store._json, BudgetLedger.settle, ResearchSupervisor.run
    active, reads, measured = [], [], []

    def command(output):
        return [sys.executable, "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'{\"ok\":true}')",
                str(output)]

    def read(path):
        if active and not measured and path.parent == store.path / "events":
            reads.append(path)
        return original_json(path)

    def run(owner, attempt, builder, *args, **kwargs):
        def observed(output):
            observe_file(monkeypatch, output, before_open=lambda: active.append(True))
            return builder(output)
        return original_run(owner, attempt, observed, *args, **kwargs)

    def settle(ledger, *args, **kwargs):
        count = len(reads)
        measured.append(count)  # Stop observing before independent settlement.
        assert active and count, "actual owner result read/publication was not reached"
        assert store._read_snapshot() is None, "result scope leaked into settlement"
        events = store.events()
        measured.append(len(events))
        return original_settle(ledger, *args, **kwargs)

    monkeypatch.setattr(store, "_json", read)
    monkeypatch.setattr(BudgetLedger, "settle", settle)
    monkeypatch.setattr(ResearchSupervisor, "run", run)
    if admitted:
        result = SharedRunner(store, registry).run_cell("admitted", digest(value["cells"][0]), budget=BudgetSpec(10))
    else:
        result = ResearchSupervisor(store).run(attempts[0], command, BudgetSpec(10))
    assert result["state"] == "SUCCEEDED", result
    assert len(measured) == 2
    count, total = measured
    assert count <= 2 * total, (
        f"owner result publication reread {count} actual event files for {total} events")
    assert store.attempts()[result["attempt_id"]]["artifact_id"] == result["artifact_id"]
