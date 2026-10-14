"""Recomputable PIRC-17 method diagnostics, never final-eval authorization.

No fitting, data loading or forecasts are performed here. The score diagnostic
explicitly generates 256 *scoring-only* Gaussian draws per saved scoring slot;
it never generates 8192 draws implicitly or changes the N512 forecast budget.
Integration checks consume a separately registered, same-grid exact-kernel
reference. That reference is NOT an exact nonlinear SDE solution. CRN checks
use actual paired forecast-score differences, not fabricated correlated draws.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from experiments.nex326.specification import load_experiment_spec
from .inference import SEEDS
from .comparison_registry import ORIGIN_MODES
from .method_rollout import MethodDynamics, MethodForecast
from .workload import method_inventory

VERSION = "pirc17-causal-method-mechanisms-v1"
ROOT = Path(__file__).resolve().parents[2]
SCORE_DRAWS = 256
QUADRATURE_ORDER = 12
NUMERICAL_SLOTS = {"arm-18/full", "arm-19/em", "arm-19/euler"}
SCORE_SLOTS = {"arm-10/d2_mc", "arm-10/d2_closed"}
VARIANCE_SLOTS = {"arm-21/mc", "arm-21/crn"}
EXACT_REFERENCE = "diagnostic/full-exact-kernel"
STREAM_GROUPS = {
    "Full-replay": ("arm-01/full", "arm-06/dt60"),
    "score-only": ("arm-07/full", "arm-10/d2_mc", "arm-10/d2_closed"),
    "integration": ("arm-18/full", "arm-19/em", "arm-19/euler", EXACT_REFERENCE),
    "MC-CRN": ("arm-20/full", "arm-21/crn"),
}
DIRECT_UNITS = {
    "mode_entropy": "nats", "mode_switch_rate": "probability_per_mode_redraw",
    "covariance_min_eigenvalue": "m^2/s^2", "component_separation": "m/s",
    "decomposition_coefficient_norm": "heterogeneous_coefficient_norm",
    "effective_mode_count": "modes", "validation_energy_improvement": "m/s",
    "objective_improvement": "estimator_specific_objective_units",
    "mean_log_likelihood": "log_density_of_velocity_rate",
    "adaptation_parameter_delta": "heterogeneous_coefficient_norm",
    "adaptation_sample_count": "training_transitions", "meta_task_count": "training_regions",
    "condition_coefficient_norm": "heterogeneous_coefficient_norm", "density_mass_error": "probability",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    a = np.asarray(value, dtype="<f8")
    return _digest({"shape": list(a.shape), "bytes": hashlib.sha256(a.tobytes()).hexdigest()})


def _finite(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("finite real statistic required; missing is not zero")
    return float(value)


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit lowercase SHA256 evidence identity required")
    return value


def forecast_stream_binding(slot_id, origin_mode):
    required = {r["slot_id"] for r in method_inventory()["slots"] if r["disposition"] == "REQUIRED"}
    if slot_id not in required | {EXACT_REFERENCE} or origin_mode not in ORIGIN_MODES:
        raise ValueError("registered slot/reference and origin mode required")
    group = next((name for name, slots in STREAM_GROUPS.items() if slot_id in slots), None)
    return {"run_id": slot_id, "crn_pair_id": f"{VERSION}:{group}:{origin_mode}" if group else None,
            "origin_identity_required_separately": True, "forecast_reuse_authorized": False}


def mechanism_registry():
    slots = {r["slot_id"]: r for r in method_inventory()["slots"]}
    definitions = {}
    for arm, subconfig in load_experiment_spec().executions:
        name = f"arm-{arm.arm_id:02d}/{subconfig['subconfig_id']}"
        row = slots[name]
        legacy = {**arm.mechanism_gate, **subconfig.get("mechanism_gate", {})}
        if row["disposition"] == "EXCLUDED":
            definitions[name] = {"slot_id": name, "disposition": "EXCLUDED",
                "reason": row["exclusion_reason"], "historical_gate": legacy,
                "scientific_rejection_implied": False}
            continue
        statistic = legacy["statistic"]
        source, units, limit = "fitted-model", DIRECT_UNITS.get(statistic), "Parameter/fit diagnostic, not predictive-effect evidence."
        if name in SCORE_SLOTS:
            source, units = "saved-forecast-and-scoring-only-Gaussian-draws", "relative_error"
            limit = "12x12 Gauss-Hermite versus 256-draw cyclic ES on a moment-matched Gaussian. Not closed-form mixture ES, not NLL qualification; not the common primary scorer."
        elif name in NUMERICAL_SLOTS:
            source, units = "same-grid-coupled-full-horizon-reference", "m"
            limit = "Exact affine kernels at unchanged h with the same causal conditions/closure. Not a global nonlinear exact solution or step-refinement certificate. EM >=0 is only a nonnegative smoke check, not accuracy qualification."
        elif name in VARIANCE_SLOTS:
            source, units = "complete-block-origin-five-seed-score-triplets", "variance_ratio"
            limit = "One joint MC/CRN comparison shared by both ledger rows. Conditional on five forecast seeds, not independent replication or a universal CRN guarantee."
        elif statistic == "mode_switch_rate":
            limit = "1-sum(p^2), expected switch probability for iid redraws; not observed switches per second."
        elif statistic == "density_mass_error":
            limit = "Mixture-weight normalization (also constrained at model binding), not independent PDE mass-conservation evidence."
        elif statistic in {"validation_energy_improvement", "objective_improvement", "mean_log_likelihood"}:
            units = ("dimensionless_normalized_objective" if row["components"]["estimator"] == "mixed"
                     else "log_density_of_velocity_rate" if statistic == "mean_log_likelihood" else "m/s")
            limit = "Recomputed from immutable fitted-objective scalars. Development/calibration diagnostic, not final forecasting ES or out-of-sample predictive likelihood."
        if units is None:
            raise ValueError("unbound required method mechanism")
        definitions[name] = {"slot_id": name, "disposition": "REQUIRED", "historical_gate": legacy,
            "statistic": statistic, "operator": legacy.get("operator", "ge"),
            "threshold": float(legacy["threshold"]), "units": units, "source": source,
            "definition_version": VERSION, "scope_limit": limit,
            "historical_empirical_results_reusable_without_recomputation": False}
    sources = ("experiments/pirc17/method_mechanisms.py", "experiments/pirc17/method_rollout.py",
               "experiments/nex326/model.py", "experiments/nex326/experiment.json")
    base_path = "experiments/pirc17/plans/full-delivery-policy-v1.json"
    base_raw = (ROOT/base_path).read_bytes()
    if hashlib.sha256(base_raw).hexdigest() != "0eeb276f4659f7d4f0e242caaa2567ffe17ae51999503d093c863ca6d99cdb9e":
        raise ValueError("finite delivery candidate changed; re-register diagnostic workload")
    base = json.loads(base_raw)
    repeats = base["counts"]["origin_cases_all_modes"]*len(SEEDS)
    payload = {"schema_version": VERSION, "state": "candidate-unsealed", "slots": definitions,
        "counts": {"slots": 36, "required": 28, "excluded": 8},
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        "numerical_rule": "Maximum over the four registered instants of sqrt(mean_i(||mu_i-mu_ref_i||^2+||sqrt(C_i)-sqrt(C_ref_i)||_F^2)), using principal symmetric PSD roots. This is an explicit Gaussian-coupling RMS upper bound for the paired empirical conditional mixture, not exact mixture W2 or global SDE error. Mean and covariance discrepancies are also reported. Aggregate the le gate by maximum over the complete required origin/seed set; retain the historical2m threshold with this explicitly changed distributional definition.",
        "numerical_reference_workload": "One additional Full FP/exact-kernel reference per origin/seed shared by arm18/19, with identical N,h,H,model and mode/normal streams; not a new NEX326 arm. Must be budgeted and sealed before execution.",
        "score_rule": "All four original target slots; worst relative Gaussian diagnostic error for the gate. d2_mc supplementary forecast score uses the first256 of saved N512 paths; d2_closed uses moment-Gaussian quadrature. Common primary ES remains N512 U-statistic for both.",
        "score_draws_per_time": SCORE_DRAWS, "quadrature_order": QUADRATURE_ORDER,
        "score_covariance_jitter_m2": 1e-9, "relative_error_floor_m": 1e-9,
        "variance_rule": "Within each block/seed average matched origin score differences, compute unbiased variance across the five seeds, then ratio of equal-block mean variances (MC-FP)/(CRN-FP). Zero denominator is unavailable, never floored.",
        "variance_interval": "Descriptive paired-block percentile bootstrap, B2000, seed20260926, central95%; not the primary ES interval or an uncertainty estimate over newly generated seeds. Undefined resamples retain null interval endpoints.",
        "pre_seal_workload_supplement": {"base_policy": base_path, "base_content_sha256": base["sha256"],
            "additional_exact_kernel_forecasts": repeats,
            "additional_scoring_only_Gaussian_draws": repeats*2*4*SCORE_DRAWS,
            "total_stochastic_forecasts_including_audits": base["counts"]["max_generated_stochastic_forecasts_including_audits"]+repeats,
            "scientific_forecasts_unchanged": base["counts"]["scientific_forecasts_all_modes"],
            "registered_human_exemptions_unchanged": True, "extra_fits": 0,
            "formal_phase_caps_unchanged": base["resource_budget"]["phase_caps_seconds"],
            "reference_charged_to": "method_forecasts", "score_draws_charged_to": "mechanism_and_paired_inference",
            "automatic_expansion": False,
            "cost_scope": "Exact-reference runtime unmeasured, still charged to the existing3h method phase/48h overall ceiling; no additional cost probe. This supplement must be included in DEV-03/04 and ACCEPT-01, not launched by this registry."},
        "paired_stream_groups": {name: list(slots) for name, slots in STREAM_GROUPS.items()},
        "stream_policy": "Explicitly supersedes unbound legacy arm-seed assumptions for these diagnostic pairs only, before any final-eval run. All28 slot forecasts remain physical executions. Origins and origin modes remain in stream identity; ordinary MC stays independent. Identical aliases/score-only predictors are not independent replication or predictive improvements.",
        "gate_is_global_numerical_qualification": False, "scientific_claim_authorized": False,
        "final_eval_authorized": False}
    return {**payload, "sha256": _digest(payload)}


def _definition(slot_id):
    result = mechanism_registry()["slots"].get(slot_id)
    if result is None or result["disposition"] != "REQUIRED":
        raise ValueError("exact required method slot needed; exclusions are not gates")
    return result


def _receipt(slot_id, value, evidence, *, unavailable_reason=None):
    definition = _definition(slot_id)
    if value is None:
        if not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
            raise ValueError("unavailable statistic needs an explicit reason")
        passed = None
    else:
        value = _finite(value)
        if unavailable_reason is not None:
            raise ValueError("available statistic cannot hide an unavailable reason")
        passed = value >= definition["threshold"] if definition["operator"] == "ge" else value <= definition["threshold"]
    result = {"schema_version": VERSION, "slot_id": slot_id, "definition_sha256": _digest(definition),
        "statistic": definition["statistic"], "operator": definition["operator"],
        "threshold": definition["threshold"], "units": definition["units"],
        "status": "unavailable" if value is None else "computed", "value": value,
        "passed": passed, "unavailable_reason": unavailable_reason, "evidence": evidence,
        "scope_limit": definition["scope_limit"], "scientific_claim_authorized": False}
    result["sha256"] = _digest(result)
    return result


def validate_gate(receipt):
    """Recompute the predicate and definition/hash; raw-source replay is separate."""
    expected = _receipt(receipt["slot_id"], receipt["value"], receipt["evidence"],
                        unavailable_reason=receipt["unavailable_reason"])
    if receipt != expected:
        raise ValueError("gate differs from its statistic/operator/threshold or bound definition")


def model_gate(slot_id, dynamics):
    definition = _definition(slot_id)
    if definition["source"] != "fitted-model" or not isinstance(dynamics, MethodDynamics):
        raise ValueError("this gate requires its registered evidence source")
    dynamics.validate()
    model = dynamics.model
    slot = next(r for r in method_inventory()["slots"] if r["slot_id"] == slot_id)
    c = slot["components"]
    if (model.model_kind != c["model"] or model.condition_names != tuple(c["condition"])
            or model.estimator_method != c["estimator"] or model.transfer_method != c["transfer"]
            or model.finetune_method != c.get("finetune", "all")
            or dynamics.reference_interval_seconds != c["dt_seconds"]):
        raise ValueError("model/training mechanism differs from requested slot")
    stat, p = definition["statistic"], model.mode_probabilities
    if stat == "mode_entropy":
        value = -np.sum(p*np.log(p+1e-15))
    elif stat == "mode_switch_rate":
        value = 1.-np.sum(p*p)
    elif stat == "covariance_min_eigenvalue":
        value = min(np.linalg.eigvalsh(v).min() for v in model.covariances)
    elif stat == "component_separation":
        value = max(np.linalg.norm(a[0]-b[0]) for a in model.weights for b in model.weights)
    elif stat == "decomposition_coefficient_norm":
        value = sum(np.linalg.norm(w[3:6]) for w in model.weights)
    elif stat == "effective_mode_count":
        value = np.sum(p >= .05)
    elif stat in {"validation_energy_improvement", "objective_improvement"}:
        value = model.validation_objective_before-model.validation_objective_after
    elif stat == "mean_log_likelihood":
        value = -model.validation_objective_after
    elif stat == "adaptation_parameter_delta":
        value = model.adaptation_parameter_delta
    elif stat == "adaptation_sample_count":
        value = model.training_sample_count
    elif stat == "meta_task_count":
        value = model.meta_task_count
    elif stat == "condition_coefficient_norm":
        start = 6 if model.model_kind == "explicit_decomp" else 3
        value = sum(np.linalg.norm(w[start:]) for w in model.weights)
    elif stat == "density_mass_error":
        value = abs(np.sum(p)-1.)
    else:
        raise ValueError("no fitted-model implementation for this gate")
    return _receipt(slot_id, float(value), {"dynamics": dynamics.identity(),
        "model_sha256": _digest(model.to_dict()), "evidence_unit": "one deterministic fitted model",
        "validation_objective_before": model.validation_objective_before,
        "validation_objective_after": model.validation_objective_after})


def _cyclic_es(samples, target):
    return float(np.linalg.norm(samples-target, axis=1).mean()
                 - .5*np.linalg.norm(samples-np.roll(samples, 1, axis=0), axis=1).mean())


def _gaussian_score(samples, target):
    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False)+1e-9*np.eye(2)
    roots, weights = np.polynomial.hermite.hermgauss(QUADRATURE_ORDER)
    grid = np.stack(np.meshgrid(roots, roots), axis=-1).reshape(-1, 2)
    w = np.outer(weights, weights).reshape(-1)/math.pi
    root = np.linalg.cholesky(covariance)
    score = np.sum(w*np.linalg.norm(mean+np.sqrt(2)*grid@root.T-target, axis=1))
    score -= .5*np.sum(w*np.linalg.norm(2*grid@root.T, axis=1))
    return float(score), mean, root


def score_diagnostic(slot_id, positions_m, targets_m, *, seed, origin_id, input_identity_sha256):
    if slot_id not in SCORE_SLOTS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("registered score slot and original forecast seed required")
    if not isinstance(origin_id, str) or not origin_id.strip():
        raise ValueError("origin identity required")
    _sha(input_identity_sha256)
    positions, targets = np.asarray(positions_m, dtype=float), np.asarray(targets_m, dtype=float)
    if (positions.shape != (512, 4, 2) or targets.shape != (4, 2)
            or not np.isfinite(positions).all() or not np.isfinite(targets).all()):
        raise ValueError("saved N512/four-time finite forecast and original targets required")
    rows = []
    for t in range(4):
        closed, mean, root = _gaussian_score(positions[:, t], targets[t])
        key = _digest([VERSION, "scoring-Gaussian-not-forecast", origin_id, t])
        raw = bytes.fromhex(key)
        # Python ints preserve uint32 entropy on Windows (NumPy int is int32).
        words = [int.from_bytes(raw[i:i+4], "little") for i in range(0, 32, 4)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, *words]))
        draws = mean+rng.standard_normal((SCORE_DRAWS, 2))@root.T
        mc = _cyclic_es(draws, targets[t])
        rows.append({"time_index": t, "moment_gaussian_quadrature_es_m": closed,
            "gaussian_mc_es_m": mc, "relative_error": abs(closed-mc)/max(abs(closed), 1e-9),
            "saved_forecast_cyclic_es_256_m": _cyclic_es(positions[:SCORE_DRAWS, t], targets[t]),
            "diagnostic_draws_sha256": _array_sha(draws)})
    return _receipt(slot_id, max(r["relative_error"] for r in rows), {
        "origin_id": origin_id, "seed": seed, "input_identity_sha256": input_identity_sha256,
        "positions_sha256": _array_sha(positions), "targets_sha256": _array_sha(targets),
        "per_time": rows, "extra_forecasts": 0, "diagnostic_draws_per_time": SCORE_DRAWS,
        "quadrature_order": QUADRATURE_ORDER, "numpy_version": np.__version__,
        "predictor_changed_by_score_choice": False})


_MATCH = ("dynamics", "model_kind", "condition_names", "origin_mode", "origin_id",
          "origin_epoch_seconds", "velocity_source", "velocity_observed_at_seconds", "velocity_error_mps",
          "particles", "seed", "max_step_seconds", "history_step_seconds", "mode_step_seconds",
          "integration_steps", "history_ticks_seconds", "mode_resampling_times_seconds")


def _provenance(diagnostics):
    required = set(_MATCH) | {"integrator", "propagation", "random_stream_sha256", "crn_pair_id"}
    if not isinstance(diagnostics, dict) or not required <= diagnostics.keys():
        raise ValueError("complete forecast provenance required")
    _sha(diagnostics["random_stream_sha256"])
    if (type(diagnostics["particles"]) is not int or diagnostics["particles"] < 1
            or type(diagnostics["seed"]) is not int or diagnostics["seed"] not in SEEDS
            or diagnostics["origin_mode"] not in ORIGIN_MODES
            or not isinstance(diagnostics["origin_id"], str) or not diagnostics["origin_id"].strip()):
        raise ValueError("registered forecast counts, seed and origin provenance required")
    dynamics = diagnostics["dynamics"]
    if not isinstance(dynamics, dict) or not {"model_sha256", "fit_identity", "reference_interval_seconds"} <= dynamics.keys():
        raise ValueError("fitted model provenance required")
    _sha(dynamics["model_sha256"])
    if (not isinstance(dynamics["fit_identity"], str) or not dynamics["fit_identity"]
            or diagnostics["model_kind"] != "seg_constant_mode" or diagnostics["condition_names"] != ["solar_elev"]
            or dynamics["reference_interval_seconds"] != 60 or diagnostics["mode_step_seconds"] != 60):
        raise ValueError("these integration/variance gates require the registered Full model family and conditions")


def integration_diagnostic(slot_id, approximate, reference, *, approximate_context_sha256, reference_context_sha256):
    if slot_id not in NUMERICAL_SLOTS or not all(isinstance(x, MethodForecast) for x in (approximate, reference)):
        raise ValueError("registered integration slot and complete method forecasts required")
    if _sha(approximate_context_sha256) != _sha(reference_context_sha256):
        raise ValueError("origin/input/condition context changed")
    a, r = approximate.diagnostics, reference.diagnostics
    _provenance(a)
    _provenance(r)
    if any(a.get(k) != r.get(k) for k in _MATCH):
        raise ValueError("paired integration inputs, model or grid differ")
    expected_integrator = "split" if slot_id == "arm-18/full" else "euler_maruyama"
    if (a["integrator"] != expected_integrator or r["integrator"] != "exact"
            or a["propagation"] != "fp" or r["propagation"] != "fp"
            or not a.get("crn_pair_id") or a["crn_pair_id"] != r.get("crn_pair_id")
            or a["random_stream_sha256"] != r["random_stream_sha256"]):
        raise ValueError("same-stream FP approximation and exact-kernel reference required")
    times = approximate.forecast.elapsed_seconds
    if (np.asarray(times).shape != (4,) or not np.isfinite(times).all() or times[0] <= 0
            or np.any(np.diff(times) <= 0) or not np.array_equal(times, reference.forecast.elapsed_seconds)):
        raise ValueError("actual scoring times differ")
    am, rm = approximate.conditional_means_m, reference.conditional_means_m
    ac, rc = approximate.conditional_covariances_m2, reference.conditional_covariances_m2
    n, t = a["particles"], len(times)
    if (not all(isinstance(x, np.ndarray) and np.isfinite(x).all() for x in (am, rm, ac, rc))
            or am.shape != (n, t, 2) or rm.shape != am.shape
            or ac.shape != (n, t, 2, 2) or rc.shape != ac.shape):
        raise ValueError("complete finite paired conditional moments required")
    roots = []
    for covariance in (ac, rc):
        tolerance = 128*np.finfo(float).eps*np.maximum(1., np.max(np.abs(covariance), axis=(2, 3)))
        if (np.any(np.max(np.abs(covariance-covariance.swapaxes(2, 3)), axis=(2, 3)) > tolerance)
                or np.any(np.linalg.eigvalsh(covariance)[..., 0] < -tolerance)):
            raise ValueError("symmetric positive-semidefinite conditional covariances required")
        eigenvalues, eigenvectors = np.linalg.eigh((covariance+covariance.swapaxes(2, 3))/2)
        roots.append((eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))[..., None, :])@eigenvectors.swapaxes(2, 3))
    rms_mean = np.sqrt(np.mean(np.sum((am-rm)**2, axis=2), axis=0))
    rms_cov = np.sqrt(np.mean(np.sum((ac-rc)**2, axis=(2, 3)), axis=0))
    root_difference = np.sum((roots[0]-roots[1])**2, axis=(2, 3))
    coupled_rms = np.sqrt(np.mean(np.sum((am-rm)**2, axis=2)+root_difference, axis=0))
    return _receipt(slot_id, float(coupled_rms.max()), {
        "context_sha256": approximate_context_sha256, "approximate_diagnostics": copy.deepcopy(a),
        "reference_diagnostics": copy.deepcopy(r), "actual_horizons_seconds": times.tolist(),
        "per_time_rms_conditional_mean_error_m": rms_mean.tolist(),
        "per_time_rms_covariance_difference_m2": rms_cov.tolist(),
        "per_time_gaussian_coupling_rms_m": coupled_rms.tolist(),
        "conditional_moment_sha256": [_array_sha(x) for x in (am, rm, ac, rc)],
        "global_exact_nonlinear_solution": False})


def variance_diagnostic(rows, *, expected_origins_by_block):
    """Every expected origin x 5 seeds x MC/CRN/FP row is required.

    No intersection-based deletion or seed pseudoreplication. Bootstrap varies
    independent blocks only; it does not pretend that five seeds are ample for
    precise simulation-variance inference.
    """
    rows = tuple(rows)
    slots = {"arm-20/full", "arm-21/mc", "arm-21/crn"}
    if not expected_origins_by_block or any(not v for v in expected_origins_by_block.values()):
        raise ValueError("nonempty prespecified independent blocks/origins required")
    origins = [o for values in expected_origins_by_block.values() for o in values]
    if (len(origins) != len(set(origins)) or any(not isinstance(o, str) or not o.strip() for o in origins)
            or any(not isinstance(b, str) or not b.strip() for b in expected_origins_by_block)):
        raise ValueError("unique nonempty origins and independent block identities required")
    expected = {(b, o, s, slot) for b, os in expected_origins_by_block.items()
                for o in os for s in SEEDS for slot in slots}
    lookup = {}
    for row in rows:
        key = row["block_id"], row["origin_id"], row["seed"], row["slot_id"]
        if (type(row["seed"]) is not int or key not in expected or key in lookup
                or row["status"] != "success"):
            raise ValueError("missing, failed, duplicate or extra variance evidence row")
        _finite(row["score_m"])
        _sha(row["context_sha256"])
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError("incomplete variance evidence; no intersection or new seeds")
    records = []
    partitions = set()
    for block, origin_ids in sorted(expected_origins_by_block.items()):
        seed_differences = []
        for seed in SEEDS:
            differences = []
            for origin in sorted(origin_ids):
                f, m, c = (lookup[block, origin, seed, s] for s in ("arm-20/full", "arm-21/mc", "arm-21/crn"))
                triplet = (f, m, c)
                partitions.update(x["split"] for x in triplet)
                times = np.asarray(f["elapsed_seconds"], dtype=float)
                if (times.shape != (4,) or not np.isfinite(times).all() or times[0] <= 0
                        or np.any(np.diff(times) <= 0)):
                    raise ValueError("complete positive increasing scoring times required")
                if len({x["context_sha256"] for x in triplet}) != 1 or any(
                    x["elapsed_seconds"] != f["elapsed_seconds"] for x in triplet):
                    raise ValueError("variance triplet context/scoring grid differs")
                fd, md, cd = (x["diagnostics"] for x in triplet)
                for d in (fd, md, cd):
                    _provenance(d)
                if any(d.get(k) != fd.get(k) for d in (md, cd) for k in _MATCH):
                    raise ValueError("variance triplet must share model, origin, seed and grid")
                if any(d["seed"] != seed or d["origin_id"] != origin or d["integrator"] != "split" for d in (fd, md, cd)):
                    raise ValueError("variance row labels disagree with forecast provenance")
                if (fd["propagation"] != "fp" or md["propagation"] != "mc" or cd["propagation"] != "crn"
                        or not fd.get("crn_pair_id") or fd["crn_pair_id"] != cd.get("crn_pair_id")
                        or fd["random_stream_sha256"] != cd["random_stream_sha256"]
                        or md.get("crn_pair_id") is not None or md["random_stream_sha256"] == fd["random_stream_sha256"]):
                    raise ValueError("actual shared FP/CRN streams and independent MC stream required")
                differences.append([m["score_m"]-f["score_m"], c["score_m"]-f["score_m"]])
            seed_differences.append(np.mean(differences, axis=0))
        variances = np.var(seed_differences, axis=0, ddof=1)
        records.append({"block_id": block, "origin_count": len(origin_ids),
            "paired_seed_differences_m": np.asarray(seed_differences).tolist(),
            "mc_minus_fp_variance_m2": float(variances[0]), "crn_minus_fp_variance_m2": float(variances[1])})
    if len(partitions) != 1 or not partitions <= {"train", "validation", "final_eval"}:
        raise ValueError("one admissible, unmixed evidence partition required")
    values = np.array([[r["mc_minus_fp_variance_m2"], r["crn_minus_fp_variance_m2"]] for r in records])
    averages = values.mean(axis=0)
    denominator_zero = averages[1] == 0
    value = None if denominator_zero else float(averages[0]/averages[1])
    rng = np.random.default_rng(20260926)
    draws = values[rng.integers(len(records), size=(2000, len(records)))].mean(axis=1)
    defined = draws[:, 1] > 0
    interval = None
    if defined.all():
        interval = np.quantile(draws[:, 0]/draws[:, 1], [.025, .975], method="inverted_cdf").tolist()
    evidence = {"partition": next(iter(partitions)), "blocks": records, "independent_block_count": len(records),
        "seed_count": len(SEEDS), "variance_ddof": 1, "numerator_m2": float(averages[0]),
        "denominator_m2": float(averages[1]), "descriptive_block_bootstrap_interval": interval,
        "undefined_bootstrap_replicates": int(np.count_nonzero(~defined)), "bootstrap_iterations": 2000,
        "bootstrap_seed": 20260926, "uncertainty_scope": "Block resampling conditional on the five fixed simulation seeds; no global CRN guarantee.",
        "complete_input_sha256": _digest(sorted(rows, key=lambda r: (r["block_id"], r["origin_id"], r["seed"], r["slot_id"]))),
        "same_joint_statistic_shared_by_two_slots": True, "new_forecasts": 0}
    reason = "zero_observed_CRN_difference_variance; ratio_undefined_not_floored" if denominator_zero else None
    return {slot: _receipt(slot, value, evidence, unavailable_reason=reason) for slot in sorted(VARIANCE_SLOTS)}


if __name__ == "__main__":
    print(json.dumps(mechanism_registry(), indent=2, allow_nan=False))
