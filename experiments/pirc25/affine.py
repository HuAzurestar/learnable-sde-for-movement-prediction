"""Adapters delegate to the existing single-axis and four-state execution chains."""

from __future__ import annotations

import hashlib
from pathlib import Path

from infrastructure.research_store import digest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_VERSION = "shared-affine-synthetic-v1"


def code_hash():
    files = [ROOT / name for name in ("config.py", "numerics.py", "registry.py")]
    for directory in ("application", "data", "domain", "estimation", "evaluation", "experiments", "inference", "infrastructure", "models"):
        files.extend((ROOT / directory).rglob("*.py"))
    return digest({path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
                   for path in sorted(files)})


def fixture_spec(study_id="affine-fixture", dimensions=4, seeds=(19,)):
    if dimensions not in (1, 4):
        raise ValueError("supported fixture dimensions are 1 and 4")
    arm = "affine-" + str(dimensions)
    return {"schema_version": "pirc25-contract-v1", "study_id": study_id,
            "experiment_id": "shared-fixture-v1", "comparison_family": "synthetic-affine",
            "protocol_hash": digest({"fixture": FIXTURE_VERSION, "fit": "train-only", "dimensions": dimensions}),
            "code_hash": code_hash(), "data_hash": digest(FIXTURE_VERSION),
            "feature_hash": digest("no-covariates"), "selection_hash": digest("no-selection"),
            "arms": [{"arm_id": arm, "model_family_id": arm, "method_family_id": "existing-affine",
                      "objective_id": "energy-score", "budget_seconds": 86400}],
            "cells": [{"arm_id": arm, "seed": seed, "block_id": "synthetic-block-1",
                       "plugin_id": arm, "capability": "exact-transition" if dimensions == 1 else "generic-rollout",
                       "visibility": "synthetic", "dimensions": dimensions} for seed in seeds]}


def single_axis(seed):
    import torch
    from application.experiment import ExperimentApplication
    from application.synthetic import make_synthetic_em_data
    from config import Components, Config
    from domain import ForecastRequest, ModelContext, ObservationSet

    config = Config(seed=seed, components=Components(model="I1", estimator="EM", inference="exact"),
                    model={"I1": {"n_modes": 2, "kappa": 0.0, "dt_ref": 1.0}},
                    protocol={"em": {"max_iter": 2}})
    data, _ = make_synthetic_em_data(n_segments=4, length=10, dt=1.0, seed=seed)
    app = ExperimentApplication.from_config(config)
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


def four_state(seed):
    from experiments.nex326.phase_space import load_phase_space_spec, run_phase_space_benchmark

    report = run_phase_space_benchmark(synthetic_cohort(), load_phase_space_spec(), n_samples=8, seed=seed)
    return {"metrics": report["metrics"], "forecast": {"per_segment": report["per_segment"]},
            "fit": report["model"], "source_schema": report["schema_version"]}


def execute(spec, cell):
    if spec["code_hash"] != code_hash() or spec["data_hash"] != digest(FIXTURE_VERSION):
        raise ValueError("fixture code/data hash changed")
    dimensions = cell["dimensions"]
    output = single_axis(cell["seed"]) if dimensions == 1 else four_state(cell["seed"])
    return {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED",
            "spec_hash": digest(spec), "cell_hash": digest(cell), "protocol_hash": spec["protocol_hash"],
            "state_order": ["x", "vx"] if dimensions == 1 else ["x", "y", "vx", "vy"],
            "units": ["m", "m/s"] if dimensions == 1 else ["m", "m", "m/s", "m/s"],
            "time_unit": "s", "resume_level": "restart-only", "qualification": "fixture",
            "metric_units": {key: ("legacy-state-norm" if dimensions == 1 else {
                "position_energy_score_d2": "m", "position_hdr90_coverage": "1", "position_cep50_error": "m",
                "velocity_endpoint_rmse": "m/s", "evaluation_segment_count": "count", "kinematic_identity_max_error": "m"
            }[key]) for key in output["metrics"]},
            "input_hash": spec["data_hash"], "output_hash": digest(output), **output}
