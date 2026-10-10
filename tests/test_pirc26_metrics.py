"""Consumer semantics and real pre-read owner boundaries; synthetic data only."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from application.pirc26_metrics import PRIMARY, TARGET_DEFINITION, FIXTURE_DEFINITION, metric_binding, aggregate_metric
from application.pirc26_runtime import command, validate_job
from application.research_admission import AdmissionGate, plugin_binding
from application.research_budget import BudgetSpec
from application.research_preregistration import PreregistrationGate
from evaluation.phase_space import evaluate_forecast
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_pirc26_runtime import prepare, owner_admit
from tests.test_research_adjudication_binding import policy


def target_policy(horizon=2.):
    value = policy()
    value["primary_metric"] = deepcopy(PRIMARY)
    value["contrasts"][0]["stratum_weights"] = [{"comparison_dimensions": {"horizon": horizon}, "weight": 1.}]
    return value


def freeze_synthetic(monkeypatch, frozen):
    original = PreregistrationGate.register_preregistration
    def freeze(gate, plan, _expected):
        plan["adjudication_spec"] = deepcopy(frozen)
        plan["primary_metrics"] = ["energy_score"]
        return original(gate, plan, digest(plan))
    monkeypatch.setattr(PreregistrationGate, "register_preregistration", freeze)


@pytest.mark.parametrize("fault", ["definition", "unit", "direction", "name", "hash", "horizon", "multi-horizon",
    "endpoint", "ambiguous", "grid-order", "stratum", "duplicate-origin", "extra-primary", "prereg", "path-count"])
def test_supported_binding_refuses_before_engines_or_new_exposure(tmp_path, monkeypatch, fault):
    frozen = target_policy()
    store, value, plugin, job, _, _, _ = prepare(tmp_path, metric_policy=frozen)
    _, receipt = owner_admit(store, value, plugin)
    job, receipt = deepcopy(job), deepcopy(receipt)
    p = receipt["spec"]["comparison_plan"]["adjudication_spec"]
    if fault in {"definition", "unit", "direction", "name"}:
        p["primary_metric"][fault] = "unimplemented"
    elif fault == "hash":
        receipt["spec"]["comparison_plan"]["adjudication_hash"] = "f" * 64
    elif fault == "horizon":
        receipt["cell"]["comparison_dimensions"]["horizon"] = 99
    elif fault == "multi-horizon":
        receipt["cell"]["comparison_dimensions"]["horizons"] = [.1, .2]
    elif fault == "endpoint":
        job["origins"][0]["time_grid"][-1] = 5.
    elif fault == "ambiguous":
        job["origins"][0]["time_grid"] = [2., 4. - 1e-13, 4.]
    elif fault == "grid-order":
        job["origins"][0]["time_grid"] = [.2, .4, .3]
    elif fault == "stratum":
        p["contrasts"][0]["stratum_weights"][0]["comparison_dimensions"] = {"horizon": 8}
    elif fault == "duplicate-origin":
        job["origins"].append(deepcopy(job["origins"][0]))
        receipt["cell"]["execution"]["inputs"]["origins"] = 2
    elif fault == "path-count":
        job["origins"][0]["sample_count"] = 257
    else:
        receipt["mode"] = "formal"
        receipt["documents"]["protocol"]["blocks"][0]["split_role"] = "final-eval"
        receipt["documents"]["preregistration"] = {"adjudication_spec": deepcopy(p), "primary_metrics": ["energy_score"]}
        if fault == "extra-primary":
            receipt["documents"]["preregistration"]["primary_metrics"].append("calibration")
        else:
            receipt["documents"]["preregistration"]["adjudication_spec"]["practical_threshold"] = 99
        prereg_hash = digest(receipt["documents"]["preregistration"])
        receipt["documents"]["protocol"]["preregistration_hash"] = prereg_hash
        receipt["spec"]["comparison_plan"]["preregistration_hash"] = prereg_hash
    if fault != "hash":
        receipt["spec"]["comparison_plan"]["adjudication_hash"] = digest(p)
    before = store.events()
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("metric preflight allocated tensors"))
    with pytest.raises(ResearchError):
        metric_binding(job, receipt)
    assert store.events() == before


@pytest.mark.parametrize("formal", [False, True])
def test_actual_admission_refusal_precedes_first_scientific_read(tmp_path, monkeypatch, formal):
    frozen = target_policy()
    if formal:
        # Qualified fixtures are deliberately synthetic attestations, not study acceptance.
        frozen["primary_metric"]["definition"] = "unimplemented"
        freeze_synthetic(monkeypatch, frozen)
    else:
        frozen["contrasts"][0]["stratum_weights"][0]["comparison_dimensions"] = {"horizon": 8}
    store, value, _, _, registry, recovery, _ = prepare(tmp_path, operation="forecast", role="final-eval" if formal else "validation",
                                                      formal=formal, metric_policy=frozen)
    before = [e for e in store.events() if e["event_kind"] in {"READ_STARTED", "READ_COMPLETED"}
              and e["payload"].get("block_id") == "fixture-1"]
    with pytest.raises(ResearchError):
        SharedRunner(store, registry, recovery_registry=recovery).run_cell(value["study_id"], digest(value["cells"][0]), budget=BudgetSpec(70))
    after = [e for e in store.events() if e["event_kind"] in {"READ_STARTED", "READ_COMPLETED"}
             and e["payload"].get("block_id") == "fixture-1"]
    assert after == before == []
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in store.events())


def test_target_endpoint_uses_manual_exact_energy_not_grid_average_or_stratum_weights(tmp_path):
    frozen = target_policy()
    frozen["contrasts"][0]["stratum_weights"] = [
        {"comparison_dimensions": {"horizon": 2.}, "weight": .25},
        {"comparison_dimensions": {"horizon": 4.}, "weight": .75}]
    store, value, plugin, job, _, _, _ = prepare(tmp_path, metric_policy=frozen)
    _, receipt = owner_admit(store, value, plugin)
    job["origins"][0]["sample_count"] = 3  # independent three-path manual score population.
    binding = metric_binding(job, receipt)
    assert binding["definition"] == TARGET_DEFINITION and binding["selected_grid_indices"] == [[2]]
    assert not binding["cross_stratum_weights_applied"]
    paths = torch.tensor([[[0.,0.,0.,0.],[100.,0.,0.,0.],[0.,0.,0.,0.]],
                          [[0.,0.,0.,0.],[101.,0.,0.,0.],[2.,0.,0.,0.]],
                          [[0.,0.,0.,0.],[102.,0.,0.,0.],[0.,2.,0.,0.]]], dtype=torch.float64)
    result = {"samples": paths, "time_grid": [2.,3.,4.], "failed_sample_ids": [], "failure_rate": 0., "requested_paths": 3}
    evaluation = evaluate_forecast(result, torch.zeros((3,4), dtype=torch.float64))
    actual = aggregate_metric([{"evaluation": evaluation}], binding)["energy_score"]
    expected = 4 / 3 - (8 + 4 * math.sqrt(2)) / 12  # independent ordered-pair formula.
    assert actual == pytest.approx(expected, abs=1e-14)
    assert actual != pytest.approx(sum(r["metrics"]["energy_score"] for r in evaluation["rows"][1:]) / 2)
    broken = deepcopy(evaluation)
    broken["rows"][-1]["estimator"]["estimator_id"] = "energy-u-uniform-pairs-v1"
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        aggregate_metric([{"evaluation": broken}], binding)
    # Different solver-grid densities still give each distinct origin one
    # endpoint vote. The registered .25 stratum weight is NOT applied here.
    job["origins"].append({**job["origins"][0], "segment_id": "another-origin", "time_grid": [2.,2.5,3.,4.]})
    two = metric_binding(job, receipt)
    second_paths = torch.cat([paths[:, :1], paths[:, 1:2], paths[:, 1:]], dim=1) * 2
    second = evaluate_forecast({**result, "samples": second_paths, "time_grid": [2.,2.5,3.,4.]}, torch.zeros((4,4), dtype=torch.float64))
    assert aggregate_metric([{"evaluation": evaluation}, {"evaluation": second}], two)["energy_score"] == pytest.approx(expected * 1.5)
    fewer = deepcopy(evaluation)
    fewer["rows"][-1]["sample_count"] = 2
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        aggregate_metric([{"evaluation": fewer}], binding)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        aggregate_metric([{"evaluation": evaluation}], {**binding, "sample_counts": []})


def test_pre_read_hook_is_content_bound_but_legacy_plugin_hash_is_unchanged(tmp_path):
    _, _, plugin, _, registry, _, _ = prepare(tmp_path)
    assert callable(registry.resolve(plugin.plugin_id, "generic-rollout", version=plugin.registry_entry.version).pre_read_validator)
    assert plugin_binding(plugin) != plugin_binding(replace(plugin, pre_read_validator=None))


def test_real_formal_worker_publishes_supported_frozen_target_definition(tmp_path, monkeypatch):
    frozen = target_policy()
    freeze_synthetic(monkeypatch, frozen)
    store, value, _, _, registry, recovery, _ = prepare(tmp_path, operation="forecast", role="final-eval", formal=True, metric_policy=frozen)
    result = SharedRunner(store, registry, recovery_registry=recovery).run_cell(value["study_id"], digest(value["cells"][0]), budget=BudgetSpec(70))
    assert result["state"] == "SUCCEEDED", result
    payload = json.loads((store.path / "artifacts" / result["artifact_id"]).read_bytes())
    assert payload["metric_definitions"] == {"energy_score": TARGET_DEFINITION}
    binding = payload["forecast"]["metric_binding"]
    assert binding["target_horizon_seconds"] == 2. and binding["adjudication_hash"] == digest(frozen)
    assert payload["metrics"] == aggregate_metric(payload["forecast"]["origins"], binding)
    assert payload["metrics"]["energy_score"] == payload["forecast"]["origins"][0]["evaluation"]["rows"][-1]["metrics"]["energy_score"]
    assert binding["definition"] != FIXTURE_DEFINITION


@pytest.mark.parametrize("mode,protected", [("pilot", False), ("pilot", True), ("formal", True)])
def test_pilot_diagnostics_do_not_invent_a_formal_adjudication_policy(tmp_path, mode, protected):
    store, value, plugin, job, _, _, _ = prepare(tmp_path)
    _, receipt = owner_admit(store, value, plugin)
    receipt["mode"] = mode
    receipt["cell"]["comparison_dimensions"] = {"horizon": 2.}
    if protected:
        receipt["documents"]["protocol"]["blocks"][0]["split_role"] = "final-eval"
        with pytest.raises(ResearchError, match="NEEDS_PREREGISTRATION"):
            metric_binding(job, receipt)
    else:
        binding = metric_binding(job, receipt)
        assert binding["definition"] == TARGET_DEFINITION and binding["adjudication_hash"] is None
        receipt["cell"]["comparison_dimensions"] = {}
        with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
            metric_binding(job, receipt)


def test_real_job_preflight_succeeds_in_cold_interpreter_without_numerical_engines(tmp_path):
    store, value, _, job, _, _, _ = prepare(tmp_path, metric_policy=target_policy())
    receipt = {"mode": "fixture", "spec": value, "cell": value["cells"][0],
               "documents": {"protocol": store.manifest("protocol-inputs")}}
    script = """
import importlib.abc,json,sys
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy'}:
            raise AssertionError('metric preflight initialized an engine')
sys.meta_path.insert(0,NoEngine())
from application.pirc26_runtime import validate_job
job,receipt=json.load(sys.stdin)
assert validate_job(job,receipt)==job
assert not {'torch','numpy','scipy'}.intersection(sys.modules)
print('cold-preflight-ok')
"""
    process = subprocess.run([sys.executable, "-B", "-c", script], input=encode([job,receipt]).decode(),
        capture_output=True, text=True, timeout=40, cwd=Path(__file__).resolve().parents[1])
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip() == "cold-preflight-ok"


def test_real_pilot_worker_uses_explicit_horizon_without_formal_policy(tmp_path):
    store, value, _, _, registry, recovery, _ = prepare(tmp_path, operation="forecast", role="validation", pilot=True)
    assert "adjudication_spec" not in value.get("comparison_plan", {})
    result = SharedRunner(store, registry, recovery_registry=recovery).run_cell(value["study_id"], digest(value["cells"][0]), budget=BudgetSpec(70))
    assert result["state"] == "SUCCEEDED", result
    payload = json.loads((store.path / "artifacts" / result["artifact_id"]).read_bytes())
    assert payload["qualification"] == "fixture"  # no formal qualification promoted by diagnostics.
    assert payload["metric_definitions"] == {"energy_score": TARGET_DEFINITION}
    assert payload["forecast"]["metric_binding"]["adjudication_hash"] is None
