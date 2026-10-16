"""Scheduling/deadline boundaries only; full real-lock controls remain intact."""

import pytest

from experiments.pirc28 import source_metadata, source_preflight
from infrastructure.research_store import ResearchError
from tests.test_pirc28_source_preflight import admitted_sources


def test_each_cold_read_yields_without_writer_ownership_and_checks_budget(admitted_sources, monkeypatch):
    store, _, _, sources = admitted_sources
    calls, polls = [], []
    native_sleep = source_metadata._handoff_wait

    def poll():
        assert store._read_snapshot() is None
        polls.append(len(calls))

    def observed_sleep(seconds):
        # A sleep spy forwards the ACTUAL scheduler wait. It neither substitutes
        # a lock nor changes its timeout/heartbeat/scientific deadline.
        if seconds == source_metadata.READER_HANDOFF_SECONDS:
            assert store._read_snapshot() is None
            assert polls[-1] == len(calls)
            calls.append(seconds)
        native_sleep(seconds)

    monkeypatch.setattr(source_metadata, '_handoff_wait', observed_sleep)
    result = source_preflight.validate_scientific_source(store, *sources[0], check_budget=poll)
    assert result['qualified_identifiability'] is False
    assert calls and all(value == 0.02 for value in calls)
    assert polls[-1] == len(calls)


def test_deadline_crossed_during_reader_handoff_prevents_disclosure(admitted_sources, monkeypatch):
    store, _, _, sources = admitted_sources
    crossed = []
    native_sleep = source_metadata._handoff_wait

    def observed_sleep(seconds):
        native_sleep(seconds)
        if seconds == source_metadata.READER_HANDOFF_SECONDS:
            assert store._read_snapshot() is None
            crossed.append(True)

    def poll():
        assert store._read_snapshot() is None
        if crossed:
            raise ResearchError('TIMEOUT', 'original deadline crossed during reader handoff')

    monkeypatch.setattr(source_metadata, '_handoff_wait', observed_sleep)
    with pytest.raises(ResearchError) as error:
        source_preflight.validate_scientific_source(store, *sources[0], check_budget=poll)
    assert error.value.code == 'TIMEOUT'
    assert crossed == [True]
