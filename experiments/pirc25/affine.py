"""Adapters delegate to the existing single-axis and four-state execution chains."""

from __future__ import annotations

from pathlib import Path

from infrastructure.research_store import digest
from infrastructure.research_files import source_file_hash

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_VERSION = "shared-affine-synthetic-v1"


def phase_space_api():
    """Keep the reviewed four-state dependency behind its public adapter."""
    from experiments.nex326 import phase_space
    return phase_space


def code_hash():
    files = [ROOT / name for name in ("config.py", "numerics.py", "registry.py")]
    for directory in ("application", "data", "domain", "estimation", "evaluation", "experiments", "inference", "infrastructure", "models"):
        files.extend((ROOT / directory).rglob("*.py"))
    return digest({path.relative_to(ROOT).as_posix(): source_file_hash(ROOT, path)
                   for path in sorted(files)})


def fixture_spec(study_id="affine-fixture", dimensions=4, seeds=(19,)):
    if dimensions not in (1, 4):
        raise ValueError("supported fixture dimensions are 1 and 4")
    arm = "affine-" + str(dimensions)
    value = {"schema_version": "pirc25-contract-v1", "study_id": study_id,
            "experiment_id": "shared-fixture-v1", "comparison_family": "synthetic-affine",
            "protocol_hash": digest({"fixture": FIXTURE_VERSION, "fit": "train-only", "dimensions": dimensions}),
            "code_hash": code_hash(), "data_hash": digest(FIXTURE_VERSION),
            "feature_hash": digest("no-covariates"), "selection_hash": digest("no-selection"),
            "arms": [{"arm_id": arm, "model_family_id": arm, "method_family_id": "existing-affine",
                      "objective_id": "energy-score", "budget_seconds": 86400}],
            "cells": [{"arm_id": arm, "seed": seed, "block_id": "synthetic-block-1",
                       "plugin_id": arm, "capability": "exact-transition" if dimensions == 1 else "generic-rollout",
                       "visibility": "synthetic", "dimensions": dimensions} for seed in seeds]}
    from .plugins import affine_plugin, affine_configuration
    from application.research_execution import execution_binding
    plugin = affine_plugin(dimensions)
    config, inputs = affine_configuration(dimensions)
    from .components import fixture_component_bindings
    for cell in value["cells"]:
        cell["resource_class"] = plugin.registry_entry.resource_class
        components = fixture_component_bindings(dimensions, cell["seed"], matrix_cells=len(value["cells"]),
            registries=plugin.component_registries)
        cell["execution"] = execution_binding(plugin.registry_entry, config, inputs, matrix_cells=len(value["cells"]),
            components=components, component_registries=plugin.component_registries)
    return value


def single_axis(seed, *, component_bindings=None, matrix_cells=1):
    import torch
    from application.experiment import ExperimentApplication
    from application.synthetic import make_synthetic_em_data
    from .components import single_axis_config
    from domain import ForecastRequest, ModelContext, ObservationSet

    config = single_axis_config(seed)
    data, _ = make_synthetic_em_data(n_segments=4, length=10, dt=1.0, seed=seed)
    app = ExperimentApplication.from_config(config, component_bindings=component_bindings, matrix_cells=matrix_cells)
    trained = app.train(data)
    request = ForecastRequest(torch.tensor([1.0, 0.25], dtype=torch.float64),
                              torch.tensor([1.0, 2.0], dtype=torch.float64), 8, ModelContext(regime=0))
    forecast = app.predict(trained.model, request)
    report = app.evaluate(forecast, ObservationSet(torch.zeros((2, 2), dtype=torch.float64)))
    return {"metrics": report.aggregate, "forecast": {"samples": forecast.samples.tolist(), "horizons": [1.0, 2.0],
            "preview": {"case_selection_rule": "registered-synthetic-fixture", "sample_selection_rule": "all-generated-samples",
                        "sample_ids": list(range(len(forecast.samples))), "n_samples": len(forecast.samples), "generation_version": FIXTURE_VERSION}},
            "fit": trained.fit.to_dict(), "source_schema": "ExperimentApplication"}


def synthetic_cohort():
    import numpy as np
    from experiments.nex326.cohort import Cohort, Segment

    def segment(name, offset):
        time = np.arange(8, dtype=float)
        state = np.column_stack([time + 0.07 * time ** 2 + offset,
                                 0.5 * time + 0.02 * time ** 2 - offset])
        return Segment(name, "synthetic", "fixture", time, state, {}, False)

    cohort = Cohort("nex326-cohort-v1", FIXTURE_VERSION, "v1", "engineering-fixture",
                    {"train": (segment("train-1", 0), segment("train-2", 1)),
                     "adapt": (), "validation": (segment("validation-1", 2),),
                     "evaluation": (segment("evaluation-1", 3),), "animal_pretrain": ()},
                    {"adapt": "fit uses train only", "animal_pretrain": "synthetic fixture"}, digest(FIXTURE_VERSION))
    cohort.validate()
    return cohort


def four_state(seed, *, component_bindings=None, matrix_cells=1, registries=None):
    from experiments.nex326.phase_space import load_phase_space_spec, run_phase_space_benchmark

    composition = None
    benchmark = load_phase_space_spec()
    if component_bindings is not None:
        from application.research_composition import component_plan
        from .plugins import affine_plugin
        from infrastructure.research_store import encode
        import json
        entry = affine_plugin(4).registry_entry
        component_plan(entry, registries, component_bindings, matrix_cells=matrix_cells, seed=seed)
        component_bindings = json.loads(encode(component_bindings))
        registries = dict(registries)
        component_plan(entry, registries, component_bindings, matrix_cells=matrix_cells, seed=seed)
        # Freeze the benchmark before any factory callback can inspect inputs.
        benchmark = json.loads(encode(component_bindings["model"]["config"]["benchmark"]))
        composition = {role: registry.create_bound(component_bindings[role], matrix_cells=matrix_cells)
            for role, registry in registries.items()}
    report = run_phase_space_benchmark(synthetic_cohort(), benchmark, n_samples=8, seed=seed, composition=composition)
    return {"metrics": report["metrics"], "forecast": {"per_segment": report["per_segment"]},
            "fit": report["model"], "source_schema": report["schema_version"]}


def execute(spec, cell):
    if spec["code_hash"] != code_hash() or spec["data_hash"] != digest(FIXTURE_VERSION):
        raise ValueError("fixture code/data hash changed")
    dimensions = cell["dimensions"]
    from .plugins import affine_plugin
    from application.research_execution import execution_plan
    plugin = affine_plugin(dimensions)
    plan = execution_plan(spec, cell, plugin)
    components = cell["execution"]["components"]
    output = single_axis(cell["seed"], component_bindings=components, matrix_cells=len(spec["cells"])) if dimensions == 1 else four_state(
        cell["seed"], component_bindings=components, matrix_cells=len(spec["cells"]), registries=plugin.component_registries)
    return {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED",
            "spec_hash": digest(spec), "cell_hash": digest(cell), "protocol_hash": spec["protocol_hash"],
            "state_order": ["x", "vx"] if dimensions == 1 else ["x", "y", "vx", "vy"],
            "units": ["m", "m/s"] if dimensions == 1 else ["m", "m", "m/s", "m/s"],
            "time_unit": "s", "resume_level": "restart-only", "qualification": "fixture",
            "component_plan_hash": plan["composition"]["component_plan_hash"],
            "metric_units": {key: ("legacy-state-norm" if dimensions == 1 else {
                "position_energy_score_d2": "m", "position_hdr90_coverage": "1", "position_cep50_error": "m",
                "velocity_endpoint_rmse": "m/s", "evaluation_segment_count": "count", "kinematic_identity_max_error": "m"
            }[key]) for key in output["metrics"]},
            "input_hash": spec["data_hash"], "output_hash": digest(output), **output}
