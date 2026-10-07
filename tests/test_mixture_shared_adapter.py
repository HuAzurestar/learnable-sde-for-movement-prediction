"""Disposable shared-worker controls, never an official research ledger."""

from dataclasses import replace
import json
import math
import subprocess
import sys
import time

import pytest

from application.propagation_execution import request_from_manifest, validate_propagation_cell
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_plan
from domain.mixture import MixturePolicy, MixtureSettings
from domain.errors import DataValidationError
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.design import StudyMethod, freeze_design
from experiments.pirc27.mixture_plugin import mixture_config, mixture_plugin, mixture_recovery_plugin
from experiments.pirc27.plugin import execution_inputs, propagation_resume_command
from infrastructure.research_store import ResearchError, digest
from tests.research_admission_fixtures import admit_fixture
from tests.test_propagation_shared_adapter import prepare
from tests.test_propagation_study_design import fixture


def _mixture_worker_startup_seconds():
    # Measure fresh imports AND real mixture request/source/resource validation,
    # not just generic path imports. One-step unprotected engineering input;
    # no store, grant, reservation, owner clock or scientific evidence exists.
    settings = MixtureSettings(1, 0., 0., (1.,)*4, 0., 1_000_000, 60.)
    frozen = freeze_design(fixture(methods=(StudyMethod("mixture", steps=1,
        samples=8, chunk_size=1, recovery=True, mixture_settings=settings),),
        horizons=(1.,), seeds=(11,)))
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    command = [sys.executable, "-c", "import json,sys; "
        "from experiments.pirc27.worker import execute_propagation; "
        "spec=json.load(sys.stdin); execute_propagation(spec,spec['cells'][0])"]
    started = time.monotonic()
    subprocess.run(command, input=json.dumps(spec).encode("utf-8"), check=True,
        timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return time.monotonic()-started


def test_mixture_startup_calibration_uses_full_bounded_pipeline_without_store(monkeypatch):
    from infrastructure.research_store import ResearchStore
    def no_store(*args, **kwargs):
        pytest.fail("startup calibration created a store")
    monkeypatch.setattr(ResearchStore, "__init__", no_store)
    calls = []
    def cold_process(command, **kwargs):
        calls.append((command, kwargs))
        spec = json.loads(kwargs["input"])
        assert len(spec["cells"]) == 1 and "admission" not in spec and "runtime_binding" not in spec
        cell = spec["cells"][0]
        assert cell["propagation_request"]["steps"] == 1
        assert cell["execution"]["config"]["work_steps"] == 528
        assert "execute_propagation(spec,spec['cells'][0])" in command[-1]
        assert kwargs["check"] is True and kwargs["timeout"] == 30
    monkeypatch.setattr(subprocess, "run", cold_process)
    # The no-op process mock can finish within one Windows monotonic clock tick;
    # real positive measurements are still required by duration sizing below.
    assert _mixture_worker_startup_seconds() >= 0 and len(calls) == 1


def _mixture_checkpoint_job_seconds(compute, startup):
    # Target the real80% signal at measured startup plus half the numerical
    # work. A3s floor can outlast the whole kernel on a fast host. Freeze this
    # initial duration BEFORE reservation; the workload/caps/watchdog do not
    # change. The20s bound is the same first-job engineering cap as before.
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (compute, startup)):
        raise ValueError("positive finite measured mixture calibration required")
    return min(20., (startup+.5*compute)/.8)


@pytest.mark.parametrize("compute,startup,expected", [(.5, .5, .9375), (1., .5, 1.25),
    (12., 6., 15.), (16., .5, 10.625), (100., 10., 20.)])
def test_mixture_signal_calibration_is_bounded_and_not_after_fast_work_completion(compute, startup, expected):
    job = _mixture_checkpoint_job_seconds(compute, startup)
    assert job == pytest.approx(expected) and 0 < job <= 20
    assert startup < .8*job < startup+compute


@pytest.mark.parametrize("compute,startup", [(0., 1.), (1., 0.), (-1., 1.), (True, 1.),
    (float("nan"), 1.), (1., float("inf"))])
def test_invalid_mixture_calibration_refused(compute, startup):
    with pytest.raises(ValueError):
        _mixture_checkpoint_job_seconds(compute, startup)


def prepared(root, *, synthetic=False, formal=False, maximum_job_seconds=60, request_changes=None):
    changes = ({"steps": 1, "initial_mean": (0.,)*4,
        "initial_covariance": ((.25, 0., 0., 0.), (0.,)*4, (0.,)*4, (0.,)*4)} if synthetic else {})
    changes.update(request_changes or {})
    store, spec, _ = prepare(root, recovery=True, synthetic=synthetic, changes=changes)
    cell = spec["cells"][0]
    request = request_from_manifest(cell["propagation_request"])
    plugin = mixture_plugin(synthetic=synthetic)
    policy = MixturePolicy(request.request_hash, request.model_package_hash, code_hash(),
        8 if synthetic else 1, 0., 0., (1.,)*4, 0., 1_000_000, maximum_job_seconds)
    cell.update(plugin_id=plugin.plugin_id, mixture_policy=policy.manifest())
    spec["arms"][0]["method_family_id"] = "mixture"
    config = mixture_config(request, policy, synthetic=synthetic)
    admit_fixture(store, spec, plugin, root, formal=formal, execution_config=config,
        execution_inputs=execution_inputs(request), recovery_command_builder=propagation_resume_command,
        legacy_upstream=False, fixture_prefix="mixture-")
    registry = CapabilityRegistry()
    registry.register(plugin)
    return store, spec, registry


@pytest.mark.parametrize("synthetic", [False, True])
def test_actual_mixture_worker_uses_shared_owner_cost_and_reuse_without_qualification(tmp_path, synthetic):
    store, spec, registry = prepared(tmp_path, synthetic=synthetic)
    cell = spec["cells"][0]
    package, request, config, plugin = validate_propagation_cell(spec, cell)
    plan = execution_plan(spec, cell, plugin)
    assert plan["counts"]["state_dim"] == 4
    assert plan["counts"]["paths"] == 1
    assert plan["counts"]["components"] == config["component_cap"]
    assert plan["counts"]["observations"] == 8*config["component_cap"]
    store.register(spec, digest(spec))
    runner = SharedRunner(store, registry)
    outcome = runner.run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60, category="smoke"))
    assert outcome["state"] == "SUCCEEDED", outcome
    artifact = json.loads((store.path / "artifacts" / outcome["artifact_id"]).read_bytes())
    functional = artifact["forecast"]["functional"]
    assert artifact["qualification"] == "fixture" and artifact["resume_level"] == "chunk"
    assert functional["status"] == "APPROXIMATION_ONLY"
    assert functional["error_budget"]["propagation_approximation"]["value"] is None
    assert functional["sample_count"] == 0 and functional["interval"] is None
    assert artifact["admission_hash"]
    charged = BudgetLedger(store).balance(request.arm_id)["committed_ms"]
    assert charged > 0
    reused = runner.run_cell(spec["study_id"], digest(cell))
    assert reused["reused"] and reused["artifact_id"] == outcome["artifact_id"]
    assert BudgetLedger(store).balance(request.arm_id)["committed_ms"] == charged
    recovery = mixture_recovery_plugin(synthetic=synthetic)
    assert recovery.plugin_id == plugin.plugin_id and recovery.version == "1.0.0"


@pytest.mark.parametrize("synthetic", [False, True])
def test_generic_formal_report_cannot_promote_mixture_or_read_held_out_inputs(tmp_path, synthetic):
    store, spec, registry = prepared(tmp_path, synthetic=synthetic, formal=True)
    store.register(spec, digest(spec))
    before = len(store.events())
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
            budget=BudgetSpec(60, category="smoke"))
    events = store.events()[before:]
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "READ_STARTED", "READ_COMPLETED"} for e in events)
    assert next(iter(store.attempts().values()))["state"] == "PREFLIGHT_FAILED"


def test_frozen_job_cap_refuses_before_reservation_or_worker(tmp_path):
    store, spec, registry = prepared(tmp_path, maximum_job_seconds=1)
    store.register(spec, digest(spec))
    before = len(store.events())
    with pytest.raises(ResearchError, match="frozen job budget"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
            budget=BudgetSpec(60, category="smoke"))
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "READ_STARTED", "READ_COMPLETED"}
        for e in store.events()[before:])


def test_fixture_mode_cannot_consume_heldout_mixture_block(tmp_path):
    store, spec, registry = prepared(tmp_path, formal=True)
    spec["admission"]["mode"] = "fixture"
    store.register(spec, digest(spec))
    before = len(store.events())
    with pytest.raises(ResearchError, match="dedicated held-out qualification"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
            budget=BudgetSpec(60, category="smoke"))
    assert not any(e["event_kind"] in {"WORKER_STARTED", "READ_STARTED", "READ_COMPLETED"}
        for e in store.events()[before:])


def test_actual_mixture_save_ack_reopened_resume_keeps_policy_lineage_and_all_costs(tmp_path):
    from application.research_recovery import RecoveryRegistry, SharedRecovery
    from application.propagation_execution import execute_propagation
    from infrastructure.research_store import ResearchStore, encode

    # Freeze the engineering workload within the unchanged million-work quota
    # and predeclare at most two60s continuations before creating any store.
    maximum_resumes = 2
    steps = 1_000_000 // (8*(1+1+4**3))
    startup = _mixture_worker_startup_seconds()
    store, spec, registry = prepared(tmp_path, request_changes={"steps": steps, "chunk_size": 1})
    cell = spec["cells"][0]
    started = time.monotonic()
    expected = execute_propagation(spec, cell)  # Pure unprotected unit control.
    compute = time.monotonic()-started
    job_seconds = _mixture_checkpoint_job_seconds(compute, startup)
    store.register(spec, digest(spec))
    adapters = RecoveryRegistry()
    adapters.register(mixture_recovery_plugin())
    interrupted = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
        spec["study_id"], digest(cell), budget=BudgetSpec(job_seconds, category="smoke"))
    assert interrupted["state"] == "FAILED", {**interrupted,
        "engineering_calibration_seconds": {"startup": startup, "compute": compute, "job": job_seconds}}
    parent = store.attempts()[interrupted["attempt_id"]]
    assert parent["error_code"] == "CHECKPOINT_SAVED", interrupted
    saved = interrupted["checkpoint"]
    progress = saved["progress"]
    assert 0 < progress["completed_steps"] < progress["total_steps"] == cell["execution"]["config"]["work_steps"]
    assert progress["throughput_per_second"] > 0 and progress["eta_seconds"] > 0
    assert progress["eta_seconds"] == pytest.approx((progress["total_steps"]-progress["completed_steps"])/progress["throughput_per_second"])
    events = [e for e in store.events() if e["payload"].get("attempt_id") == parent["attempt_id"]]
    kinds = [e["event_kind"] for e in events]
    assert kinds.index("CHECKPOINT_REQUESTED") < kinds.index("CHECKPOINT") < kinds.index("CHECKPOINT_SAVED")
    assert kinds.index("CHECKPOINT_SAVED") < kinds.index("WORKER_TREE_STOPPED") < kinds.index("SETTLE")
    request = next(e["payload"] for e in events if e["event_kind"] == "CHECKPOINT_REQUESTED")
    assert 0 < request["remaining_seconds"] <= .2*job_seconds
    saved_value = json.loads((store.path/"artifacts"/saved["artifact_id"]).read_bytes())
    state = saved_value["state"]
    assert state["method_state"]["policy_hash"] == cell["execution"]["config"]["mixture_policy_hash"]
    assert state["rng_state"]["scheme"] == "deterministic-cubature-no-sampled-rng-v1"
    assert state["chunk_complete"] is True and "budget" not in state and "remaining_seconds" not in state
    arm = cell["arm_id"]
    before = BudgetLedger(store).balance(arm)
    assert before["committed_ms"] == interrupted["elapsed_ms"] and not before["closed"]
    reopened = ResearchStore(tmp_path, store.store_id)
    grant = reopened.manifest("authorization-"+spec["admission"]["authorization_id"])
    with pytest.raises(ResearchError):
        SharedRecovery(reopened, registry, adapters).prepare(parent["attempt_id"], saved["artifact_id"],
            authorization={**grant, "purposes": ["evaluate"]})
    assert BudgetLedger(reopened).balance(arm) == before
    continuations = []
    previous, checkpoint, completed = parent["attempt_id"], saved["artifact_id"], progress["completed_steps"]
    for _ in range(maximum_resumes):
        reopened = ResearchStore(tmp_path, store.store_id)
        resumed = SharedRecovery(reopened, registry, adapters).resume(previous, checkpoint,
            authorization=grant, budget=BudgetSpec(60, category="smoke"))
        continuations.append(resumed)
        current = reopened.attempts()[resumed["attempt_id"]]
        assert current["parent_attempt_id"] == previous
        if resumed["state"] == "SUCCEEDED":
            break
        # Never continue timeout, fused, numerical or admission failures.
        assert resumed["state"] == "FAILED" and current["error_code"] == "CHECKPOINT_SAVED", resumed
        saved = resumed["checkpoint"]
        assert completed < saved["progress"]["completed_steps"] < progress["total_steps"]
        previous, checkpoint, completed = resumed["attempt_id"], saved["artifact_id"], saved["progress"]["completed_steps"]
    assert resumed["state"] == "SUCCEEDED", resumed
    actual = json.loads((reopened.path/"artifacts"/resumed["artifact_id"]).read_bytes())
    assert actual["forecast"] == json.loads(encode(expected["forecast"]))
    assert actual["metrics"] == expected["metrics"] and actual["output_hash"] == expected["output_hash"]
    assert actual["qualification"] == "fixture" and actual["admission_hash"] != saved_value["admission_hash"]
    receipt = reopened.manifest("admission-"+actual["admission_hash"])
    assert receipt["attempt_id"] == resumed["attempt_id"]
    assert BudgetLedger(reopened).balance(arm)["committed_ms"] == interrupted["elapsed_ms"]+sum(c["elapsed_ms"] for c in continuations)
    assert not BudgetLedger(reopened).balance(arm)["closed"]


@pytest.mark.parametrize("nonlinear", [False, True])
def test_frozen_compiler_binds_explicit_mixture_per_cell_without_new_arms_or_admission(nonlinear):
    settings = MixtureSettings(8, 0., 0., (1.,)*4, 0., 1_000_000, 60.)
    design = fixture(nonlinear=nonlinear, methods=(StudyMethod("mixture", steps=2, samples=8,
        recovery=True, mixture_settings=settings), StudyMethod("euler", steps=2, samples=8)))
    frozen = freeze_design(design)
    document = frozen.manifest()
    spec = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert "admission" not in spec and "runtime_binding" not in spec
    assert len(document["matrix"]) == 8 and len(spec["arms"]) == 2
    mixture_cells = [cell for cell in spec["cells"] if "mixture_policy" in cell]
    assert len({cell["arm_id"] for cell in mixture_cells}) == 1
    assert len({cell["mixture_policy"]["request_hash"] for cell in mixture_cells}) == 4
    for cell in spec["cells"]:
        validate_propagation_cell(spec, cell)
    for horizon in design.horizons:
        for seed in design.seeds:
            assert len({cell["propagation_request"]["coupling_id"] for cell in spec["cells"]
                if cell["horizon"] == horizon and cell["seed"] == seed}) == 1


@pytest.mark.parametrize("changes", [{"maximum_work_units": 1}, {"state_scales": [1., 1., 1., 1.]},
    {"component_cap": 33}, {"maximum_job_seconds": 7201}])
def test_compiler_refuses_invalid_settings_not_silently_smaller_axes(changes):
    settings = replace(MixtureSettings(8, 0., 0., (1.,)*4, 0., 1_000_000, 60.), **changes)
    design = fixture(methods=(StudyMethod("mixture", steps=2, samples=8, recovery=True, mixture_settings=settings),))
    with pytest.raises(DataValidationError):
        freeze_design(design)
