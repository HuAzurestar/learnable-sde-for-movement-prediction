"""Six-method compiler and real admission controls in isolated fixture stores."""
import importlib.util
from pathlib import Path
import sys

import pytest

from application.research_admission import AdmissionGate
from application.research_registry import plan_resources
from infrastructure.research_store import ResearchStore, ResearchError, digest


@pytest.fixture(scope="module")
def pilot():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pirc27_method_pilot.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("pirc27_method_pilot_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compiled(pilot):
    arms = {m: {"arm_id": "pirc27-affine-"+m+"-mean-x", "model_family_id": pilot.original.FAMILY,
        "method_family_id": m, "objective_id": "endpoint-x", "budget_seconds": 86400} for m in pilot.METHODS}
    return arms, pilot.frozen(arms)


def test_fixed_methods_original_arms_and_total_prospective_budget(pilot, compiled):
    arms, (case, cells, plugins) = compiled
    assert len(cells) == 6 and len(cells)*pilot.JOB_SECONDS == 1800
    assert tuple(c["comparison_dimensions"]["method"] for c in cells) == pilot.METHODS
    assert {c["arm_id"] for c in cells} == {a["arm_id"] for a in arms.values()}
    assert {c["block_id"] for c in cells} == {pilot.original.FAMILY}
    assert {c["propagation_request"]["seed"] for c in cells} == {1147}
    assert {c["comparison_dimensions"]["model_configuration"] for c in cells} == {"stable-v1"}
    assert all(c["propagation_request"]["tolerance"] == .05 for c in cells)
    assert all(c["frozen_dynamics"]["state_names"] == ["x", "y", "vx", "vy"] for c in cells)
    assert all(c["frozen_dynamics"]["training_source"]["training_data_hash"] is None for c in cells)
    assert len({digest(c) for c in cells}) == 6
    for cell in cells:
        plugin = plugins[cell["plugin_id"]]
        plan_resources(plugin.registry_entry, cell["execution"]["config"], cell["execution"]["inputs"], matrix_cells=6)


def test_exact_per_cell_packages_and_actual_pilot_admission(pilot, compiled, tmp_path):
    arms, _ = compiled
    store = ResearchStore(tmp_path, "pirc27-method-pilot-control", initialize=True)
    with store._read_transaction():
        store.publish(pilot.original.ORIGINAL, {"arms": list(arms.values())})
        result = pilot.prepare(store)
        assert result["cells"] == 6 and result["charged_ms"] == 0
        spec = store.manifest("study-" + pilot.STUDY)["spec"]
        assert spec["scientific_independent_n"] == 0
        table = spec["admission"]["cell_packages"]["bindings"]
        assert len(table) == 6 and len({p["package_hash"] for p in table}) == 4
        _, _, plugins = pilot.frozen(arms)
        for cell in spec["cells"]:
            attempt = store.new_attempt(store.register_run(pilot.STUDY, cell))
            receipt = AdmissionGate(store).prepare(spec, cell, plugins[cell["plugin_id"]], attempt)
            assert receipt["mode"] == "pilot" and receipt["qualification"] == "fixture"
            assert receipt["admission_selection"]["cell_hash"] == digest(cell)
        assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED"} for e in store.events())


@pytest.mark.parametrize("fault", ["closed", "foreign-arm", "aggregate-cap"])
def test_refuses_closed_or_relabelled_or_unfunded_preparation(pilot, compiled, tmp_path, monkeypatch, fault):
    from copy import deepcopy
    arms, _ = compiled
    arms = deepcopy(arms)
    store = ResearchStore(tmp_path, "pirc27-method-pilot-refusal", initialize=True)
    if fault == "foreign-arm":
        arms["euler"]["objective_id"] = "different-target"
    store.publish(pilot.original.ORIGINAL, {"arms": list(arms.values())})
    if fault == "closed":
        store.append("ARM_CLOSED", {"arm_id": arms["euler"]["arm_id"], "reason": "fixture"})
    if fault == "aggregate-cap":
        monkeypatch.setattr(pilot, "accounting", lambda store: {"charged_ms": 14400000, "reserved_ms": 0})
    with pytest.raises(ResearchError):
        pilot.prepare(store)
    assert not any(e["event_kind"] == "RESERVE" for e in store.events())
