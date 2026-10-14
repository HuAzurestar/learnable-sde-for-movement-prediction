"""Fit the entire distinct linear candidate on the original development population.

No forecast or final-eval access. An exclusive JSONL ledger retains all attempts;
the separate complete model bundle is created only if all fifty fits succeed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import psutil
import torch

from experiments.nex326 import pirc21_adapter
from experiments.pirc22 import conditioners, consumer, representations
from .checkpoints import restore_dynamics
from .configurations import configuration_encoder, terrain_configurations
from .development import load_development, transition_rows
from .direct_linear import (configuration_width, digest, fit_direct_dynamics,
                            restore_direct_dynamics, training_policy)
from .features import CanonicalEncoder
from .inference import SEEDS
from .qualification import _hash

FIT_VERSION = "pirc17-direct-linear-development-fit-v1"
MINIMUM_FREE_BYTES = 2 * 1024**3


def source_hashes():
    names = ("direct_linear_fit.py", "direct_linear.py", "development.py", "dynamics.py",
             "checkpoints.py", "configurations.py", "features.py", "origins.py")
    paths = {"pirc17/"+name: Path(__file__).with_name(name) for name in names}
    paths.update({module.__name__: Path(module.__file__) for module in
                  (conditioners, consumer, representations, pirc21_adapter)})
    return {name: _hash(path) for name, path in paths.items()}


def parameter_identity(identity):
    return digest({"base": identity["base_weights"],
                   "theta": identity["conditioner_checkpoint"]["theta_feature_rows_then_bias"],
                   "diffusion": identity["diffusion_covariance_m2_per_s"]})


def losses(model, train, validation):
    result = {}
    for role, batch in (("train", train), ("validation", validation)):
        target = batch.displacement_m/batch.elapsed_seconds[:, None]
        base = batch.design() @ model.base_weights
        fitted = base + model.correction(None, batch.feature_matrix)
        result[role] = {"base_only_mse_m2_per_s2": float(np.mean((target-base)**2)),
                        "fitted_mse_m2_per_s2": float(np.mean((fitted-target)**2))}
    return result


def prepare(reference_fit, reference_fit_sha256, eligibility, eligibility_sha256,
            release, snapshot, data_root):
    if _hash(reference_fit) != reference_fit_sha256:
        raise ValueError("original fit artifact hash mismatch")
    reference = json.loads(reference_fit.read_text(encoding="utf-8"))
    configs = terrain_configurations()
    if (reference.get("schema_version") != "pirc17-development-fit-v1"
            or reference.get("purpose") != "paired_pilot_checkpoints_not_final_model_acceptance"
            or reference.get("configurations") != configs or set(reference["models"]) != set(configs)
            or reference.get("data_roles") != {"gradient": "train", "checkpoint_selection": "validation", "final_eval_reads": 0}):
        raise ValueError("original fit schema, population role or configuration registry differs")
    encoder = CanonicalEncoder.frozen_pirc22(snapshot)
    windows, parents, identity = load_development(eligibility, eligibility_sha256, release, data_root, encoder)
    if identity != reference["development_identity"] or parents != reference["snapshot_parent_assets"]:
        raise ValueError("original development identities or snapshot parents differ")
    population = {"samples": {r:len(w) for r,w in windows.items()},
                  "independent_blocks": {r:len({x.block_id for x in w}) for r,w in windows.items()},
                  "transition_rule": "one first-future transition per qualified origin; no interpolated observations"}
    if set(windows) != {"train", "validation"} or population != reference["pilot_population"]:
        raise ValueError("original train/validation population differs")
    prepared, reference_designs, original_base = {}, {}, None
    for name, config in configs.items():
        selected = configuration_encoder(encoder, name)
        if len(selected.columns)*2 != configuration_width(name):
            raise ValueError("snapshot and frozen configuration widths differ")
        train, validation = (transition_rows(windows[role], encoder, selected) for role in ("train", "validation"))
        for role, batch in (("train", train), ("validation", validation)):
            if batch.role != role or len(batch.elapsed_seconds) != population["samples"][role]:
                raise ValueError("transition row count or role differs")
            if role in reference_designs:
                np.testing.assert_array_equal(batch.design(), reference_designs[role])
            else:
                reference_designs[role] = batch.design()
        if set(train.independent_blocks) & set(validation.independent_blocks):
            raise ValueError("train and validation independent blocks overlap")
        if set(reference["models"][name]) != {str(s) for s in SEEDS}:
            raise ValueError("original fit must contain all five seeds")
        old_models, old_losses = {}, {}
        for seed in SEEDS:
            old = restore_dynamics(reference["models"][name][str(seed)])
            if (old.identity["training_identity"] != identity["sha256"]
                    or old.identity["configuration_identity"] != config["sha256"] or old.identity["seed"] != seed
                    or old.identity["train_transition_count"] != len(train.elapsed_seconds)
                    or old.identity["validation_transition_count"] != len(validation.elapsed_seconds)
                    or old.conditioner_model.continuous_input_dim != configuration_width(name)):
                raise ValueError("original model mislabeled by population, configuration, seed or dimensions")
            if original_base is None:
                original_base = old.base_weights
            np.testing.assert_array_equal(old.base_weights, original_base)
            measured = losses(old, train, validation)
            cp = old.conditioner_checkpoint
            chosen = [r for r in cp["learning_curve"] if r["epoch"] == cp["best_epoch"]]
            if len(chosen) != 1:
                raise ValueError("original selected learning-curve row is ambiguous")
            for role in ("train", "validation"):
                np.testing.assert_allclose(measured[role]["fitted_mse_m2_per_s2"],
                                           chosen[0][role+"_loss"], rtol=1e-12, atol=1e-12)
            old_models[seed], old_losses[seed] = old, measured
        prepared[name] = train, validation, old_models, old_losses
    return reference, identity, parents, population, prepared


def run(*, reference_fit, reference_fit_sha256, eligibility, eligibility_sha256,
        release, snapshot, data_root, output, available_memory=None, progress=None):
    output = Path(output).resolve()
    if output.suffix != ".json":
        raise ValueError("candidate output must use .json; a sibling .jsonl is its ledger")
    ledger = output.with_suffix(".jsonl")
    if output.exists() or ledger.exists():
        raise FileExistsError("refusing to overwrite candidate output or attempt ledger")
    output.parent.mkdir(parents=True, exist_ok=True)
    available_memory = available_memory or (lambda: psutil.virtual_memory().available)
    started = time.perf_counter()
    expected = len(terrain_configurations()) * len(SEEDS)
    attempted = successful = failures = 0
    models, comparisons = {}, {}
    resources_stopped = False
    terminal_errors = 0
    complete = None
    with ledger.open("x", encoding="utf-8") as target:
        def emit(row):
            target.write(json.dumps(row, allow_nan=False)+"\n")
            target.flush()

        def memory_guard():
            if available_memory() < MINIMUM_FREE_BYTES:
                raise MemoryError("candidate requires at least 2 GiB available system RAM")

        emit({"type": "initialization", "schema_version": FIT_VERSION,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "expected_model_count": expected, "reference_fit_sha256": reference_fit_sha256,
              "eligibility_sha256": eligibility_sha256, "certified": False})
        try:
            memory_guard()
            sources = source_hashes()
            policy = training_policy()
            _, identity, parents, population, prepared = prepare(
                Path(reference_fit), reference_fit_sha256, Path(eligibility), eligibility_sha256,
                Path(release), Path(snapshot), Path(data_root))
            emit({"type": "header", "source_sha256": sources, "training_policy": policy,
                  "development_identity_sha256": identity["sha256"], "population": population,
                  "input_validation_seconds": time.perf_counter()-started})
            if progress:
                progress({"phase": "inputs_verified", "expected_models": expected})
            for name, (train, validation, old_models, old_losses) in prepared.items():
                models[name], comparisons[name] = {}, {}
                for seed in SEEDS:
                    memory_guard()
                    attempted += 1
                    begin = time.perf_counter()
                    record = {"type": "fit", "configuration": name, "seed": seed, "status": "failure"}
                    try:
                        candidate = fit_direct_dynamics(train, validation, seed=seed,
                            training_identity=identity["sha256"], configuration=name)
                        np.testing.assert_array_equal(candidate.base_weights, old_models[seed].base_weights)
                        restored = restore_direct_dynamics(json.loads(json.dumps(candidate.identity, allow_nan=False)))
                        measured = losses(restored, train, validation)
                        for role in ("train", "validation"):
                            for key in measured[role]:
                                np.testing.assert_allclose(measured[role][key], candidate.identity["diagnostics"][role][key],
                                                           rtol=1e-12, atol=1e-12)
                        np.testing.assert_array_equal(restored.base_weights, candidate.base_weights)
                        np.testing.assert_array_equal(restored.diffusion_root, candidate.diffusion_root)
                        np.testing.assert_array_equal(restored.correction(None, train.feature_matrix),
                                                      candidate.correction(None, train.feature_matrix))
                        models[name][str(seed)] = candidate.identity
                        comparisons[name][str(seed)] = {"original": old_losses[seed], "candidate": measured}
                        record.update(status="success", model_identity_sha256=candidate.identity["sha256"],
                            parameter_identity_sha256=parameter_identity(candidate.identity),
                            comparison=comparisons[name][str(seed)],
                            solver_report=candidate.conditioner_checkpoint["solver_report"])
                        successful += 1
                    except Exception as exc:
                        failures += 1
                        record.update(error_type=type(exc).__name__, error=str(exc))
                        resources_stopped = isinstance(exc, MemoryError)
                    record["fit_and_roundtrip_wall_seconds"] = time.perf_counter()-begin
                    emit(record)
                    if resources_stopped:
                        raise MemoryError("candidate fit exhausted resources; remaining fits not attempted")
                    if progress:
                        progress({"phase": "fit", "configuration": name, "seed": seed, "status": record["status"],
                                  "attempted": attempted, "successful": successful})
            if source_hashes() != sources:
                raise ValueError("candidate source files changed during execution")
            if failures or attempted != expected or successful != expected:
                raise ValueError("incomplete candidate fits; no complete model bundle emitted")
            uniqueness = {name:len({parameter_identity(model) for model in group.values()}) for name,group in models.items()}
            if any(n != 1 for n in uniqueness.values()):
                raise ValueError("deterministic fit unexpectedly depends on seed")
            memory = psutil.Process().memory_info()
            complete = {"schema_version": FIT_VERSION, "purpose": "development_training_candidate_not_final_acceptance",
                "training_policy": policy, "reference_fit_sha256": reference_fit_sha256,
                "development_identity": identity, "snapshot_parent_assets": parents,
                "population": population, "configurations": terrain_configurations(), "models": models,
                "comparisons": comparisons, "parameter_identity_count_by_configuration": uniqueness,
                "expected_model_count": expected, "attempted_model_count": attempted, "successful_model_count": successful,
                "failure_count": failures, "source_sha256": sources,
                "data_roles": {"solve": "train", "diagnostic": "validation", "selection": "none", "final_eval_reads": 0},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                            "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                            "rss_bytes": memory.rss, "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss)},
                "elapsed_seconds": time.perf_counter()-started, "formal_training_accepted": False, "certified": False}
            with output.open("x", encoding="utf-8") as result:
                json.dump(complete, result, indent=2, allow_nan=False)
            emit({"type": "artifact", "path": output.name, "sha256": _hash(output)})
        except Exception as exc:
            terminal_errors += 1
            resources_stopped = isinstance(exc, MemoryError)
            complete = None
            emit({"type": "failure", "error_type": type(exc).__name__, "error": str(exc)})
        completion = {"type": "completion", "status": "complete" if complete is not None else "failed",
            "expected_model_count": expected, "attempted_model_count": attempted, "successful_model_count": successful,
            "failure_count": failures, "unattempted_model_count": expected-attempted,
            "terminal_error_count": terminal_errors, "resource_stopped": resources_stopped, "certified": False,
            "elapsed_seconds": time.perf_counter()-started}
        emit(completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-fit", "eligibility", "release", "snapshot", "data-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reference-fit-sha256", required=True)
    parser.add_argument("--eligibility-sha256", required=True)
    args = parser.parse_args()
    completion = run(**vars(args), progress=lambda r: print(json.dumps(r), flush=True))
    print(json.dumps(completion), flush=True)
    return 0 if completion["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
