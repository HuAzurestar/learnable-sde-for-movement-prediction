"""Actual synthetic execution must consume the frozen metadata snapshot.

No real acceptance, protected trajectories or test-time result stand-ins.
"""
import pytest

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture, synthetic_plugin
from tests.test_research_admission_chain import fixture_command
from tests.test_research_store import spec


def prepared(tmp_path, *, formal=False, upstream_ids=()):
    store = ResearchStore(tmp_path, "snapshot-admission", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="admitted-fixture", capability="generic-rollout", visibility="synthetic")
    plugin = synthetic_plugin("admitted-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", fixture_command)
    registry = CapabilityRegistry()
    registry.register(plugin)
    grant = admit_fixture(store, value, plugin, tmp_path, formal=formal, upstream_ids=upstream_ids)
    return store, value, registry, grant


@pytest.mark.parametrize("formal", [False, True], ids=["fixture", "formal"])
@pytest.mark.parametrize("upstream_ids", [(), ("affine-4d",)], ids=["no-terrain", "public-affine"])
def test_missing_selected_snapshot_refuses_before_provider_and_worker(tmp_path, formal, upstream_ids):
    store, value, registry, _ = prepared(tmp_path, formal=formal, upstream_ids=upstream_ids)
    # Even a prior ready receipt or accepted catalog cannot replace the frozen
    # spec reference. No downstream package/qualification binding is forged.
    value["admission"].pop("upstream_snapshot_hash", None)
    store.register(value, digest(value))
    error, outcome = None, None
    try:
        outcome = SharedRunner(store, registry).run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    except ResearchError as exc:
        error = exc.code
    events = store.events()
    observed = {"error": error, "outcome": outcome,
                "states": [attempt["state"] for attempt in store.attempts().values()],
                "reads": [event for event in events if event["event_kind"] in {"READ_STARTED", "READ_COMPLETED"}],
                "workers": [event for event in events if event["event_kind"] == "WORKER_STARTED"]}
    (tmp_path / "missing-snapshot-observed.json").write_bytes(encode(observed))
    assert error == "MISSING_ARTIFACT", observed
    assert not observed["reads"] and not observed["workers"]
