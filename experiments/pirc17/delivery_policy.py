"""Finite pre-seal workload proposal, not permission to read evaluation data.

This module only reads versioned specifications and public cost summaries.
Selection operates on caller-supplied identity metadata, never trajectories or
scores. DEV-03/04 must bind the actual input/code identities and executable
receipts; TEST-01/REVIEW-01 and exact ACCEPT-01 remain mandatory.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .comparison_registry import comparison_registry
from .delivery_budget import BINDINGS, combined_scenario, read_evidence
from .inference import InferenceConfig, SEEDS
from .method_comparisons import DELTA_M, method_comparison_registry
from .method_training import TRAINING_FIELDS
from .workload import method_inventory

VERSION = "pirc17-finite-delivery-policy-v1"
ROOT = Path(__file__).resolve().parents[2]
BLOCK_LIMIT = 46
SECONDARY_BLOCK_LIMIT = 6
SELECTION_DOMAIN = "pirc17-final-block-order-v1"
PHASE_CAP_SECONDS = {
    "input_qualification_and_binding": 3600,
    "method_training": 1800,
    "terrain_training": 1800,
    "method_forecasts": 10800,
    "terrain_forecasts": 115200,
    "offline_common_scores": 7200,
    "mechanism_and_paired_inference": 7200,
    "runtime_and_forecast_replay": 7200,
    "independent_saved_output_reanalysis": 10800,
    "aggregate_export_and_integrity": 7200,
}
BENCHMARK_METHODS = (
    "arm-01/full", "arm-04/gmm_kernel", "arm-05/explicit_decomp", "arm-21/mc", "arm-21/crn",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def training_groups():
    """Share deterministic fits, NOT arm/seed forecasts or evidence rows."""
    groups = defaultdict(list)
    components_by_key = {}
    for row in method_inventory()["slots"]:
        if row["disposition"] != "REQUIRED":
            continue
        components = {k: row["components"][k] for k in TRAINING_FIELDS if k in row["components"]}
        key = digest(components)
        components_by_key[key] = components
        groups[key].append(row["slot_id"])
    if len(groups) != 16 or sum(map(len, groups.values())) != 28:
        raise ValueError("registered deterministic training groups changed")
    return [{"training_components_sha256": k, "training_components": components_by_key[k],
             "representative_slot": sorted(v)[0], "slots": sorted(v)}
            for k, v in sorted(groups.items())]


def select_identity_metadata(rows):
    """Apply the proposed outcome-blind rule after separate read authorization.

    Caller must establish and seal eligibility BEFORE supplying these rows.
    This is a pure selector, not an eligibility checker or final-eval guard.
    Exactly these fields are accepted so scores cannot enter a ranking rule.
    """
    by_block, seen = defaultdict(list), set()
    fields = {"sample_id", "independent_block_id", "split"}
    for row in rows:
        if set(row) != fields or row["split"] != "final_eval":
            raise ValueError("final-eval identity metadata only; no outcomes or eligibility flags")
        sample, block = row["sample_id"], row["independent_block_id"]
        if any(not isinstance(x, str) or not x.strip() for x in (sample, block)) or sample in seen:
            raise ValueError("unique nonempty sample and block identities required")
        seen.add(sample)
        by_block[block].append(sample)
    order = sorted(by_block, key=lambda b: (digest([SELECTION_DOMAIN, b]), b))
    selected = [{"independent_block_id": b, "sample_id": min(by_block[b]), "split": "final_eval"}
                for b in order[:BLOCK_LIMIT]]
    return {"eligible_block_count": len(order), "selected": selected,
            "secondary_selected": selected[:SECONDARY_BLOCK_LIMIT],
            "requested_blocks": BLOCK_LIMIT, "shortfall_blocks": max(0, BLOCK_LIMIT-len(selected)),
            "minimum_30_blocks_met": len(selected) >= 30,
            "selection_sha256": digest(selected), "execution_admitted": False}


def workload_counts():
    origin_cases = BLOCK_LIMIT + 2*SECONDARY_BLOCK_LIMIT
    repeats = origin_cases*len(SEEDS)
    return {
        "method_slots_required": 28, "method_slots_excluded": 8, "terrain_configurations": 10,
        "primary_origins": BLOCK_LIMIT, "secondary_origins_per_mode": SECONDARY_BLOCK_LIMIT,
        "origin_cases_all_modes": origin_cases, "seeds_per_origin": len(SEEDS),
        "primary_method_forecasts": BLOCK_LIMIT*len(SEEDS)*28,
        "primary_terrain_forecasts": BLOCK_LIMIT*len(SEEDS)*10,
        "secondary_method_forecasts": 2*SECONDARY_BLOCK_LIMIT*len(SEEDS)*28,
        "secondary_terrain_forecasts": 2*SECONDARY_BLOCK_LIMIT*len(SEEDS)*10,
        "method_forecasts_all_modes": repeats*28, "terrain_forecasts_all_modes": repeats*10,
        "scientific_forecasts_all_modes": repeats*38,
        "inertial_deterministic_paths": origin_cases,
        "independent_method_fits": 16, "independent_terrain_fits": 10,
        "forecast_replay_items": 38,
        # Each subject: 5 recreated providers, 1 unscored warm-up, 5 warm trials.
        "runtime_subjects": 15, "runtime_forecasts_including_warmup": 15*11,
        "max_generated_stochastic_forecasts_including_audits": repeats*38+38+15*11,
        "full_prediction_regeneration_passes": 0,
        "full_saved_output_reanalysis_passes": 1,
    }


def projected_compute():
    evidence = read_evidence()
    primary = combined_scenario(evidence, independent_blocks=BLOCK_LIMIT)
    # The price is a planning extrapolation from causal-prefix observations.
    # Neither extra origin mode has a measured unit price: no claimed bound.
    all_modes = combined_scenario(evidence, independent_blocks=BLOCK_LIMIT+2*SECONDARY_BLOCK_LIMIT)
    terrain_per_origin_seed = primary["terrain_prediction_seconds"]/(BLOCK_LIMIT*len(SEEDS))
    method_per_origin_seed = primary["method_prediction_seconds"]/(BLOCK_LIMIT*len(SEEDS))
    measured = evidence["methods"]["by_forecast"]
    benchmark_prediction_seconds = 11*(terrain_per_origin_seed + sum(
        measured[k]["forecast_seconds"] for k in BENCHMARK_METHODS))
    replay_prediction_seconds = terrain_per_origin_seed+method_per_origin_seed
    return {
        "primary_prediction_and_common_score_hours": primary["prediction_and_common_scoring_hours"],
        "all_modes_prediction_and_common_score_hours": all_modes["prediction_and_common_scoring_hours"],
        "finite_runtime_and_reforecast_prediction_hours": (benchmark_prediction_seconds+replay_prediction_seconds)/3600,
        "evidence": {k: {"path": p, "sha256": h} for k, (p, h) in BINDINGS.items()},
        "secondary_origin_cost_measured": False,
        "assumptions": ["same serial CPU runtime, N512/h5 and nominal 30-minute horizon",
                        "all 26 FP slots use the maximum of the three observed FP prices",
                        "secondary origin modes use causal-prefix unit prices without a new cost probe",
                        "cold/warm/replay use existing prices, not measured benchmark distributions"],
        "not_an_upper_bound_or_project_eta": True,
        "excluded_from_estimate": ["input binding/loading", "all formal fits", "extra mechanism scores",
            "independent reanalysis", "serialization and integrity", "failure overhead",
            "software implementation/review", "full manuscript/PDF", "human decision time"],
    }


def policy():
    terrain_registry = comparison_registry()
    methods = method_comparison_registry()
    stats = asdict(InferenceConfig(delta_m=DELTA_M))
    payload = {
        "schema_version": VERSION, "state": "proposed-for-pre-eval-review",
        "execution_admitted": False, "final_eval_authorized": False,
        "new_empirical_fits": 0, "new_empirical_forecasts": 0,
        "purpose": "Close finite scope/parameter decisions for DEV-03/04; not a sealed executable manifest or approval.",
        "training": {
            "population_policy": "Explicitly adopt the entire existing task-qualified development subset, not the full raw release and not only the three pilot blocks.",
            "eligibility_sha256": "7b6773ec628b7a14f586400ab315965d1f88b2227365b0b9a6cb35be9a7f9690",
            "population_identity": "82ad16075261508d5362c17ebd1355c10f58e698d937fdc4ee2105c423308afd",
            "prepared_inputs_sha256": "eec69a9d9ee502e6f5f574a1b7ebc9ea0ec561c7789bf23b1e3b142aa9e3201e",
            "outer_train_windows": 404, "validation_windows": 81,
            "method_roles": {"train": 328, "adapt": 76, "validation": 81},
            "method_assignment": "Preserve the global PIRC-20/NEX326 train/adapt assignment before subset selection.",
            "justification": "Same frozen 30-minute time/terrain-coverage eligibility across both matrices; all eligible development windows, no error-based subsampling. Bounded task-specific experiment, not full-release training.",
            "selection_bias_limit": "Coverage/duration-selected training subset; no claim of optimal full-release training or global geographic generalization.",
            "methods": {"fit_policy": "Refit each of 16 deterministic training identities in the causal frame/UTC solar policy; no historical legacy coefficient reuse.",
                "groups": training_groups(), "max_transitions_per_fit": 80000,
                "training_intervals_seconds": [30, 60, 120, 300, 600],
                "tail_rule": "Trim at most one incomplete terminal training interval, never scoring observations.",
                "reuse_check": "Identical input, training components, sources, environment, interval and serialized-model identity; one physical fit per key, all 28 slot bindings retained.",
                "seed_interpretation": "Five forecast seeds, not five independent deterministic trainings."},
            "terrain": {"fit_policy": "Refit ten distinct configurations with the previously evaluated direct-linear solver and frozen PIRC-22 representation/capacity.",
                "training_policy_sha256": "560a6f0776287ab7a31a91c88d029e37f25a1964f8039a39554cd267e202d0ee",
                "transition_rule": "One original first-future transition per qualified train origin; 81 validation transitions diagnostic only.",
                "independent_configurations": 10, "base_ridge": 1e-6, "weight_decay": 1e-4,
                "reuse_check": "Seed labels may bind an identical deterministic fit only after exact parameter identity checks; configurations never share a fitted conditioner.",
                "historical_candidates": "Existing Adam failures and pilot-only model flags retained; candidate artifacts are not silently promoted to accepted final models."},
        },
        "selection": {"requested_independent_blocks": BLOCK_LIMIT, "origins_per_block": 1,
            "algorithm": "Hash-order independent block IDs with the fixed domain; choose lexically smallest eligible sample ID per block.",
            "domain": SELECTION_DOMAIN, "secondary_prefix_blocks": SECONDARY_BLOCK_LIMIT,
            "eligibility_timing": "After ACCEPT-01 input access, freeze the complete eligibility/missingness denominator and exact selected IDs before any prediction or performance scoring.",
            "shortfall": "Use all eligible blocks up to 46, record shortfall; fewer than 30 forbids a conclusive population verdict. No replacement, horizon shortening or expansion based on results.",
            "same_paired_set": "All required method slots and ten terrain configurations use the identical primary set and identical secondary subsets.",
            "power_basis": {"source_sha256": "29e92c11d2642f0ec334920bb51fafaad8b4b24dec0b0be7431a2497aed3a771",
                "target": .8, "assumed_true_improvement_m": 2*DELTA_M,
                "required_primary_blocks_at_pilot_sd": {"all": 30, "road": 33, "river": 46, "worldcover": 11, "surface": 9},
                "rule": "46 is the largest registered primary-family 1x pilot-SD planning requirement, not a guaranteed power certificate.",
                "limit": "Only three development blocks. The 2x-SD 182-block sensitivity is disclosed, not an expansion instruction. No transfer of terrain power to method comparisons."}},
        "origin_modes": {
            "causal_prefix": {"role": "primary", "blocks": BLOCK_LIMIT,
                "velocity": "Last-three-visible-observation secant; original observation time; measurement error unknown, not zero."},
            "known_velocity": {"role": "secondary-descriptive", "blocks": SECONDARY_BLOCK_LIMIT,
                "velocity": "Supply the same causal-prefix secant at t0, but only position+velocity to the predictor; retain explicit source/time and unknown error.",
                "limitation": "Causal derived velocity, not an independently measured exact sensor velocity; no new information or sensor-accuracy claim."},
            "point_only": {"role": "secondary-descriptive", "blocks": SECONDARY_BLOCK_LIMIT,
                "velocity": "Empirical velocity prior fitted only on the 404 outer-train prefix origins; sample initial velocities per particle.",
                "history": "History channel contains prior-initialized/predicted direction, explicitly labelled unobserved at t0; no hidden prefix enters the predictor."},
            "secondary_inference": "All 38 configurations/slots, all five seeds and metrics, but no hypothesis tests or factor verdicts; six blocks are a bounded descriptive diagnostic, not a representative or powered population sample.",
        },
        "forecast": {"particles": 512, "maximum_step_seconds": 5., "nominal_horizon_seconds": 1800.,
            "nominal_score_seconds": [60., 300., 900., 1800.], "time_weights": [.25]*4,
            "target_tolerance_seconds": 30., "maximum_observation_gap_seconds": 60.,
            "target_rule": "Distinct original observations nearest each slot; earlier tie; no interpolation. Use actual elapsed times (last slot can be 1770..1830 seconds).",
            "terrain_history_clock_seconds": 5., "method_history_clock_seconds": "Each registered training/observation interval tau, not integration h.",
            "method_noise": "R is calibrated velocity-rate covariance; fixed Q=R*tau independent of h; not global continuous-time MLE.",
            "method_fp_scope": "Conditional Gaussian propagation with sampled mode histories and mean-evaluated nonlinear conditions; not a full PDE solution or global exact nonlinear SDE.",
            "forecast_reuse": "None between method slots. Preserve slot/stream identity and explicit paired CRN/control stream mapping; do not count Full anchors as independent evidence.",
            "longer_horizons": "60-minute common validation support is four blocks; not selected. No supported 72-hour claim or extra full forecast per scoring slot."},
        "statistics": {"inference_config": stats, "terrain_registry_sha256": terrain_registry["sha256"],
            "method_registry_sha256": methods["sha256"], "units": "metres for ES/delta/intervals",
            "weights": "Equal independent blocks, equal origins within block, equal five seeds; seeds/windows are not independent blocks.",
            "multiplicity": "Five terrain primary and four LIO contrasts in separate families; five method families separately. Holm zero tests and Bonferroni bootstrap-t intervals, not global FWER.",
            "power_and_numerical_disposition": "Use the actual admissible development qualification per comparison; missing power/numerical qualification must remain explicit and cannot be manufactured from final scores or transferred from another matrix.",
            "failure_precedence": "Missing/failed required evidence -> unavailable; unresolved conflict, missing power, instability or boundary overlap -> inconclusive; otherwise the frozen guarded interval rule.",
            "tail_failure": "No automatic B/seed increase; retain the unstable result as inconclusive."},
        "metrics": {"common_registry": "experiments/pirc17/metric_registry.json",
            "primary": "Same unbiased 2D ensemble U-statistic ES for every slot, equal four-time weights.",
            "inertial_baseline": "One deterministic x0+v0*t path per origin/mode, using the identical supplied causal velocity or the train-prior mean for point_only. Score as a point mass, not fitted uncertainty; no additional NEX326 arm and no fivefold seed replication.",
            "required_secondary": "All registered ADE/FDE, endpoint error quantiles, marginal CRPS, radial 50/80/90/95 coverage and area, path ES, fixed-grid position entropy and overflow.",
            "entropy_grid": {"east_min_m": -10000, "east_max_m": 10000, "north_min_m": -10000, "north_max_m": 10000, "cell_width_m": 250, "overflow_bins": 1},
            "nll": "Unavailable: no qualified predictive density; no unqualified Gaussian shortcut.",
            "mode_entropy": "Separate diagnostic for labelled method modes; unavailable for unlabelled terrain dynamics, never inferred from position entropy.",
            "method_gates": "Bind every historical gate to its actual statistic/operator/threshold and evidence. Corrected causal diagnostic definitions must be versioned, never silently equated to legacy future-state checks.",
            "extra_sampling": "No implicit 8192-sample legacy diagnostic or unregistered forecasts. Saved N512 arrays and fitted parameters first; unresolved required diagnostic binding prevents pre-eval acceptance."},
        "numerical_evidence": {"fixed_setting": "N512/h5, no automatic refinement",
            "terrain_primary": "Existing three-block/five-seed primary-family total margins below delta/4 versus finite N512/h0.3125 reference.",
            "epsilon_m": DELTA_M/4, "reference_is_exact": False,
            "limitations": ["No five-seed N512-to-larger-N certificate", "No global/all-block convergence proof",
                "LIO lacks the same fine-reference study", "Method kernel software or cost trials are not a terrain numerical certificate",
                "Secondary origin modes are descriptive and not covered by the primary numerical study"],
            "extra_cost_probes": 0, "automatic_parameter_expansion": False,
            "remaining_preparation_budget": {"cumulative_empirical_cap_seconds": 1800,
                "closed_probe_seconds": 382.890189,
                "remaining_seconds": 1417.109811,
                "scope": "Existing bounded full-delivery preparation envelope, not a new grant or a cap on all historical PIRC-17 work. Any necessary non-cost qualification needs its own exact preregistered plan; no new cost probes or automatic retries."},
            "unresolved_pre_eval_work": "Method-specific causal gate binding and bounded allowed-data qualification must be resolved in DEV-03/04/TEST-01; no fabricated blanket numerically_qualified=true."},
        "replay": {"all_outputs": "One independent complete pass over immutable saved arrays, receipts, metrics, pairing, bootstrap/Holm, gates and verdicts. Not another full forecast generation.",
            "prediction_replay": "First selected primary block, first registered seed, all 28 required method slots and ten terrain configs; compare output identity and retained failures, not an extra scientific replication.",
            "runtime": {"terrain": "All ten configurations", "methods": list(BENCHMARK_METHODS),
                "origin": "First selected primary origin", "seed": SEEDS[0], "cold_trials": 5, "warmup_trials": 1, "warm_trials": 5,
                "units": "ms per actual forecast minute; p50/p95 total latency, failure rate, sampled RSS/lifetime peak separately",
                "scope": "Provider-cold/OS-warm versus warmed provider, not machine cold; five trials per condition give descriptive quantiles, not precise tail latency claims.",
                "exclude_from_inference": "Fitting, offline scoring, file export and paper build."}},
        "resource_budget": {"phase_caps_seconds": dict(PHASE_CAP_SECONDS),
            "total_active_compute_cap_seconds": sum(PHASE_CAP_SECONDS.values()),
            "measurement": "Cumulative elapsed active job time across resumes, including failed/aborted attempts. No reset on interruption; preserve original receipts and remaining caps.",
            "per_method_fit_seconds": 90, "per_terrain_fit_seconds": 90,
            "per_method_forecast_seconds": 30, "per_terrain_forecast_seconds": 180,
            "max_attempts_per_work_item": 1, "minimum_free_memory_bytes": 2*1024**3,
            "parallel_empirical_workers": 1, "precision": "float64 dynamics/scoring; canonical map values retain frozen float32 semantics",
            "device": "CPU", "batch_particles": 512,
            "runtime_binding": {"platform": "Windows-10-10.0.19045-SP0",
                "processor": "Intel64 Family 6 Model 141 Stepping 1, GenuineIntel",
                "physical_cpus": 6, "logical_cpus": 12, "ram_total_bytes": 34071752704,
                "python": "3.10.13", "numpy": "1.26.4", "torch": "2.3.1+cu121",
                "torch_threads": 6, "torch_interop_threads": 12, "numpy_openblas_threads": 12,
                "numpy_openblas_version": "0.3.23.dev", "numba_threads_when_loaded": 12,
                "duckdb_input_threads": 2, "duckdb_eligibility_threads": 4,
                "map_packages": {"duckdb": "1.5.5", "numba": "0.65.1", "pyarrow": "24.0.0", "rasterio": "1.4.4"},
                "thread_policy": "Set/verify these effective counts and record all loaded thread pools per worker; environment variables alone are not proof. No migration or parallel speedup assumed.",
                "source": "Closed method cost evidence for Python/NumPy/Torch and 6/12 Torch threads; read-only current-host runtime inventory for other fields. Numba 12 is an explicit proposed count, not a measured cost-trial observation."},
            "cap_enforcement_required": "DEV-04 owned-process deadlines plus durable cumulative ledger; declarations alone are not enforcement.",
            "on_cap": "Stop owned work and record every unattempted/failed item. No silent phase transfer, cheaper fallback, retry or cap extension; missing required evidence is not completion.",
            "outside_this_cap": "Pre-eval software/qualification preparation, manuscript/PDF and review/human latency; not a 48-hour project-completion promise."},
        "counts": workload_counts(), "cost_projection": projected_compute(),
        "admission_requirements": ["DEV-03 exact protocol/input/code seal", "DEV-04 executable full matrix and enforced finite budgets",
            "Required mechanism/power/numerical qualification dispositions with evidence",
            "TEST-01 and REVIEW-01 all pre-eval HC/EC checks pass", "Human ACCEPT-01 on the exact version"],
    }
    if sum(PHASE_CAP_SECONDS.values()) != 48*3600:
        raise ValueError("phase budgets must equal the explicit 48-hour compute ceiling")
    return {**payload, "sha256": digest(payload)}


def validate_policy(value):
    if value != policy():
        raise ValueError("candidate policy differs from canonical scope, parameters, budget or evidence")


if __name__ == "__main__":
    print(json.dumps(policy(), indent=2, allow_nan=False))
