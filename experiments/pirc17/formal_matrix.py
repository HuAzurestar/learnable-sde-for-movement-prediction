"""Full pre-evaluation work inventory; ranks are not sampled final-eval IDs.

This builder reads only the sealed protocol and public source specifications.
It produces a capacity manifest, not an executable/authorized evaluation. The
actual frozen population binds rank placeholders after ACCEPT01 and before
prediction. A shortfall remains explicit; it never selects replacements.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from experiments.pirc22.consumer import COMPOSITION_DEPENDENCIES, FACTOR_GROUPS
from experiments.pirc22.representations import _COMPOSITIONS, _VARIANT_DIMENSIONS
from .comparison_registry import ORIGIN_MODES
from .inference import SEEDS
from .method_mechanisms import EXACT_REFERENCE, forecast_stream_binding
from .protocol import validate_protocol
from .protocol_core import digest, envelope, publish, read_json, unpack
from .workload import method_inventory

VERSION = "pirc17-full-formal-matrix-v1"
PROTOCOL_SHA256 = "2267ab84fa77ff4039234a61e8f44be1c3713317d8ad2d1bb4ced9af0c5b293a"
PROTOCOL_FILE_SHA256 = "8cd3aa35049a397bf691d18661db30f3873903d6cb2a621d1806b91f0d64e3dd"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT/"experiments/pirc17/protocols"/(PROTOCOL_SHA256+".json")
FACTORS = ("road", "river", "worldcover", "surface")


def load_protocol():
    value = read_json(PROTOCOL_PATH, expected_file_sha256=PROTOCOL_FILE_SHA256)
    validate_protocol(value, expected_sha256=PROTOCOL_SHA256)
    return value


def terrain_ownership(configurations):
    full = configurations["all-terrain"]
    output = {}
    for name, config in configurations.items():
        factors = set(config["terrain_factors"])
        allowed = {v for f in (*factors, "history") for v in FACTOR_GROUPS[f]}
        expected_variants = [v for v in full["variant_ids"] if v in allowed]
        expected_compositions = [c for c in full["composition_ids"] if set(COMPOSITION_DEPENDENCIES[c]) <= allowed]
        if (config["variant_ids"] != expected_variants or config["composition_ids"] != expected_compositions
                or "history.direction" not in expected_variants):
            raise ValueError("terrain owner closure or baseline history changed")
        numeric = sum(_VARIANT_DIMENSIONS[v] for v in expected_variants) + sum(_COMPOSITIONS[c][0] for c in expected_compositions)
        output[name] = {"configuration_sha256": config["sha256"], "included_factors": list(config["terrain_factors"]),
            "removed_factors": [f for f in FACTORS if f not in factors],
            "variant_ids": expected_variants, "composition_ids": expected_compositions,
            "removed_variant_ids": [v for v in full["variant_ids"] if v not in expected_variants],
            "removed_composition_ids": [c for c in full["composition_ids"] if c not in expected_compositions],
            "composition_owners": {c: sorted(f for f, variants in FACTOR_GROUPS.items()
                if set(variants) & set(COMPOSITION_DEPENDENCIES[c])) for c in expected_compositions},
            "numeric_input_dimension": numeric, "validity_input_dimension": numeric,
            "conditioner_input_dimension": 2*numeric, "baseline_history_retained": True,
            "fit_identity": "terrain-fit:"+name, "retrain_independently": True,
            "card_ids": ["terrain-overall", *["terrain-factor:"+f for f in config["terrain_factors"]]]}
    if len(output) != 10 or output["base"]["conditioner_input_dimension"] != 4 or output["all-terrain"]["conditioner_input_dimension"] != 56:
        raise ValueError("fixed ten-config/4-to56-channel capacity changed")
    return output


def definition(protocol):
    p = validate_protocol(protocol, expected_sha256=PROTOCOL_SHA256)
    delivery = p["components"]["finite_delivery"]
    budget = p["resource_contract"]
    groups = delivery["training"]["methods"]["groups"]
    fits = {slot: g["training_components_sha256"] for g in groups for slot in g["slots"]}
    methods = []
    for row in method_inventory()["slots"]:
        required = row["disposition"] == "REQUIRED"
        methods.append({"slot_id": row["slot_id"], "arm_id": row["arm_id"], "components": row["components"],
            "full_anchor": row["full_anchor"], "disposition": row["disposition"],
            "exclusion_reason": row["exclusion_reason"],
            "disposition_source": "MPA/project/PIRC-17/gists/ACCEPTANCE-CRITERIA.md#hc-04" if not required else None,
            "fit_identity": "method-fit:"+fits[row["slot_id"]] if required else None,
            "mechanism_definition": p["components"]["method_mechanisms"]["slots"][row["slot_id"]],
            "card_id": "method:"+row["slot_id"], "scientific_rejection_implied_by_exclusion": False})
    terrain = terrain_ownership(p["components"]["terrain_configurations"])
    required_methods = sorted(r["slot_id"] for r in methods if r["disposition"] == "REQUIRED")
    work = []

    def add(kind, phase, *, subject="all", mode=None, rank=None, seed=None, repeat=None,
            matrix=None, forecasts=0, maximum=None, fit=None, scientific=False):
        cap = budget["phase_caps_seconds"][phase] if maximum is None else maximum
        descriptor = {"kind": kind, "phase": phase, "matrix": matrix, "subject": subject,
            "origin_mode": mode, "origin_rank": rank, "seed": seed, "repetition": repeat,
            "generated_forecasts": forecasts, "max_active_seconds": cap, "fit_identity": fit,
            "scientific": scientific}
        # IDs include all axes and budget routing. No score-dependent selection.
        work.append({"work_id": digest(descriptor), **descriptor})

    add("input_qualification_and_population", "input_qualification_and_binding")
    for group in groups:
        add("method_fit", "method_training", subject=group["representative_slot"], matrix="NEX326-methods",
            fit="method-fit:"+group["training_components_sha256"], maximum=budget["per_method_fit_seconds"])
    for name in sorted(terrain):
        add("terrain_fit", "terrain_training", subject=name, matrix="terrain",
            fit="terrain-fit:"+name, maximum=budget["per_terrain_fit_seconds"])

    subjects = [("NEX326-methods", s, "method-fit:"+fits[s], budget["per_method_forecast_seconds"]) for s in required_methods]
    subjects += [("terrain", s, "terrain-fit:"+s, budget["per_terrain_forecast_seconds"]) for s in sorted(terrain)]
    rank_counts = {m: delivery["origin_modes"][m]["blocks"] for m in ORIGIN_MODES}
    for mode, count in rank_counts.items():
        for rank in range(count):
            for matrix, subject, fit, maximum in subjects:
                phase = "method_forecasts" if matrix == "NEX326-methods" else "terrain_forecasts"
                for seed in SEEDS:
                    add("scientific_forecast", phase, subject=subject, mode=mode, rank=rank, seed=seed,
                        matrix=matrix, forecasts=1, maximum=maximum, fit=fit, scientific=True)
            for seed in SEEDS:
                add("same_grid_reference", "method_forecasts", subject=EXACT_REFERENCE, mode=mode, rank=rank,
                    seed=seed, matrix="NEX326-diagnostic", forecasts=1,
                    maximum=budget["per_method_forecast_seconds"], fit="method-fit:"+fits["arm-01/full"])
            add("inertial_path", "offline_common_scores", mode=mode, rank=rank, matrix="inertial")
            add("common_scores", "offline_common_scores", mode=mode, rank=rank, matrix="both-plus-inertial")
    benchmark = set(delivery["replay"]["runtime"]["methods"])
    for matrix, subject, fit, maximum in subjects:
        add("forecast_replay", "runtime_and_forecast_replay", subject=subject, mode="causal_prefix", rank=0,
            seed=SEEDS[0], matrix=matrix, forecasts=1, maximum=maximum, fit=fit)
        if matrix == "terrain" or subject in benchmark:
            for temperature, count in (("cold", 5), ("warmup", 1), ("warm", 5)):
                for repeat in range(count):
                    add("runtime_"+temperature, "runtime_and_forecast_replay", subject=subject, mode="causal_prefix",
                        rank=0, seed=SEEDS[0], repeat=repeat, matrix=matrix, forecasts=1, maximum=maximum, fit=fit)
    for kind, phase in (("mechanisms_and_inference", "mechanism_and_paired_inference"),
                        ("independent_reanalysis", "independent_saved_output_reanalysis"),
                        ("aggregate_export", "aggregate_export_and_integrity")):
        add(kind, phase)
    counts = dict(Counter(r["kind"] for r in work))
    calls = sum(r["generated_forecasts"] for r in work)
    if (counts["scientific_forecast"] != 11020 or counts["same_grid_reference"] != 290
            or calls != p["generation_counts"]["total_stochastic_forecasts_including_audits"]
            or len(work) != len({r["work_id"] for r in work}) or len(work) != 11659):
        raise ValueError("full matrix workload differs from sealed count supplement")
    return {"schema_version": VERSION, "protocol_sha256": protocol["sha256"],
        "state": "full-capacity-inventory-not-execution-authorization", "method_ledger": methods,
        "terrain_ledger": terrain, "training_groups": groups, "rank_counts": rank_counts,
        "seeds": list(SEEDS), "workloads": work, "counts_by_kind": counts, "max_generated_forecasts": calls,
        "counts_by_phase": dict(Counter(r["phase"] for r in work)),
        "phase_caps_seconds": budget["phase_caps_seconds"],
        "scoring_only_Gaussian_draws": p["generation_counts"]["additional_scoring_only_Gaussian_draws"],
        "paired_streams": {m: {s: forecast_stream_binding(s, m) for s in required_methods+[EXACT_REFERENCE]} for m in ORIGIN_MODES},
        "population_binding": "Zero-based ranks index the outcome-blind frozen population; secondary ranks index its first6 primary blocks. No final IDs are loaded by this builder. Missing ranks retain NOT_ADMITTED dispositions, never replacements.",
        "failed_fit_policy": "All dependent items retain their denominator and explicit dependency-unavailable dispositions. No legacy model fallback or implicit refit.",
        "ledger_not_cartesian_cross": "28 required method slots plus10 terrain configurations; four score times share each forecast.16 method fits and10 independent terrain fits, not5 trainings per forecast seed.",
        "execution_seal_required": True, "final_eval_authorized": False,
        "new_fits": 0, "new_forecasts": 0, "final_eval_reads": 0}


def build_matrix(protocol):
    return envelope(definition(protocol))


def validate_matrix(value, protocol):
    payload = unpack(value)
    if digest(payload) != digest(definition(protocol)):
        raise ValueError("matrix inventory changed slots/ownership/paired axes/budget")
    return payload


if __name__ == "__main__":
    protocol = load_protocol()
    value = build_matrix(protocol)
    path, value = publish(ROOT/"experiments/pirc17/matrices", unpack(value))
    print(json.dumps({"path": str(path.relative_to(ROOT)), "sha256": value["sha256"],
        "work_items": len(value["payload"]["workloads"]), "generated_forecast_cap": value["payload"]["max_generated_forecasts"],
        "final_eval_reads": 0, "final_eval_authorized": False}))
