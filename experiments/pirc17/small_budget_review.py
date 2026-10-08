"""Bounded offline small-ensemble review; never calls a forecast or map loader.

Whole original audits are reproduced before slicing fixed particle prefixes.
The old failures remain failures. New conditional precision diagnostics are not
scientific confidence intervals, continuous-time error bounds or acceptance.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np

from . import candidate_power as parent
from .direct_linear_evidence import read_bound
from .inference import PRIMARY_FAMILY, SEEDS
from .nested_precision import energy_delete_groups
from .precision import energy_delete_one, summarize_delete_one
from .precision_check import KEYS, aggregate_precision, load_particle_evidence
from .qualification import _hash

VERSION = "pirc17-small-budget-offline-v1"
FAMILY_SIZE = 55  # 15 MC + 10 particle + 30 possible full-horizon step diagnostics.
ALPHA = .05
Q = NormalDist().inv_cdf(1-ALPHA/(2*FAMILY_SIZE))


def source_hashes():
    return {**parent.source_hashes(), "small_budget_review.py": _hash(Path(__file__))}


def check_budgets(budgets):
    if (not isinstance(budgets, list) or len(budgets) not in (2, 3)
            or any(type(n) is not int or n < 3 for n in budgets)
            or budgets != sorted(set(budgets)) or any(budgets[-1] % n for n in budgets)):
        raise ValueError("two or three ordered unique integer-ratio budgets >=3 required")


def cached_statistics(particles, target, budgets):
    check_budgets(budgets)
    if len(particles) != budgets[-1]:
        raise ValueError("largest budget must equal the complete saved reference ensemble")
    ordinary = {n: energy_delete_one(particles[:n], target) for n in budgets}
    grouped = {n: energy_delete_groups(particles, target, n) for n in budgets[:-1]}
    return ordinary, grouped


def contrast_statistics(candidate, control, budgets, weights):
    """Combine cached same-path scores/deletions, never independent variances."""
    a, ag = candidate
    b, bg = control
    rows = []
    for n in budgets:
        p = summarize_delete_one(a[n][0]-b[n][0], a[n][1]-b[n][1], weights)
        rows.append(("small_budget_contrast", n, n, p))
    for n in budgets[:-1]:
        p = summarize_delete_one((ag[n][0]-bg[n][0])-(a[n][0]-b[n][0]),
                                 (ag[n][1]-bg[n][1])-(a[n][1]-b[n][1]), weights)
        p.pop("particle_count")
        p.update(jackknife_group_count=n, reference_particle_count=budgets[-1], prefix_particle_count=n)
        rows.append(("small_budget_refinement", budgets[-1], n, p))
    return rows


def summarize(records, header, epsilon):
    aggregates = aggregate_precision(records, header)
    for row in aggregates:
        difference = abs(row["time_weighted_difference_m"])
        margin = Q*row["time_weighted_standard_error_m"]
        row["simultaneous_normal_mc_margin_m"] = margin
        row["epsilon_m"] = epsilon
        row["diagnostic_family_size"] = FAMILY_SIZE
        row["normal_quantile"] = Q
        if row["kind"] == "small_budget_refinement":
            row["sign_convention"] = "reference_budget_model_contrast_minus_prefix_budget_model_contrast"
            row["absolute_refinement_plus_margin_m"] = difference+margin
            row["within_particle_refinement_budget"] = difference+margin <= epsilon
        else:
            row["remaining_budget_for_step_check_m"] = epsilon-margin
            row["mc_margin_within_total_budget"] = margin <= epsilon
        row["five_minute_slot"] = {
            "nominal_seconds": 300,
            "difference_m": row["by_scoring_slot_difference_m"][1],
            "standard_error_m": row["by_scoring_slot_standard_error_m"][1],
            "role": "descriptive short-horizon screening, not a primary-family pass"}
        row["certified"] = False
    return aggregates


def analyze(entries, header, budgets, epsilon, progress=None, guard=lambda: None):
    """Entries have already passed complete model-bound audits and NPZ checks."""
    check_budgets(budgets)
    configurations = {name for pair in PRIMARY_FAMILY.values() for name in pair}
    expected = set(itertools.product(header["sample_ids"], configurations, SEEDS))
    if set(entries) != expected or list(header["seeds"]) != list(SEEDS):
        raise ValueError("complete origin/configuration/five-seed grid required")
    records = []
    for sample, seed in itertools.product(header["sample_ids"], SEEDS):
        guard()
        loaded = {name: entries[(sample, name, seed)] for name in configurations}
        first_row, (_, first_target, first_times) = next(iter(loaded.values()))
        cache = {}
        for name, (row, (particles, target, times)) in loaded.items():
            if (any(row.get(k) != first_row.get(k) for k in
                    ("sample_id", "seed", "independent_block_id", "brownian_identity",
                     "actual_horizons_seconds", "max_step_seconds", "particles"))
                    or not np.array_equal(target, first_target) or not np.array_equal(times, first_times)):
                raise ValueError("paired models change stream, targets, times, block or resolution")
            cache[name] = cached_statistics(particles, target, budgets)
            if not np.allclose(cache[name][0][budgets[-1]][0],
                               [s["energy_score_m"] for s in row["scores"]["by_time"]],
                               rtol=0, atol=1e-8):
                raise ValueError("saved reference scores do not reproduce")
        for comparison, (candidate, control) in PRIMARY_FAMILY.items():
            for kind, large, small, precision in contrast_statistics(
                    cache[candidate], cache[control], budgets, header["time_weights"]):
                axis_candidate = dict(zip(KEYS, (sample, candidate, seed, large, first_row["max_step_seconds"])))
                axis_control = dict(zip(KEYS, (sample,
                    control if kind == "small_budget_contrast" else candidate,
                    seed, small, first_row["max_step_seconds"])))
                records.append({"kind": kind, "comparison": comparison,
                    "candidate_workload": axis_candidate, "control_workload": axis_control,
                    "model_pair": [candidate, control],
                    "independent_block_id": first_row["independent_block_id"],
                    "precision": precision})
        if progress:
            progress({"phase": "origin_seed_analyzed", "completed_pairs":
                len(records)//(len(PRIMARY_FAMILY)*(2*len(budgets)-1)),
                "expected_pairs": len(header["sample_ids"])*len(SEEDS)})
    return {"paired_diagnostics": records, "aggregate_precision": summarize(records, header, epsilon)}


def run(*, plan_path, plan_sha256, output, progress=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite small-budget evidence")
    plan = read_bound(plan_path, plan_sha256)
    if (plan.get("schema_version") != VERSION or plan.get("new_rollouts") != 0
            or plan.get("final_eval_authorized") is not False
            or plan.get("diagnostic_family_size") != FAMILY_SIZE or plan.get("alpha") != ALPHA
            or not 0 < plan.get("wall_seconds", 0) <= 600):
        raise ValueError("registered offline-only bounded plan required")
    check_budgets(plan["particle_budgets"])
    started = time.perf_counter()

    def guard():
        if time.perf_counter()-started > plan["wall_seconds"]:
            raise TimeoutError("fixed offline review cap reached; no extension")

    sources = source_hashes()
    reference = read_bound(plan["reference_power"], plan["reference_power_sha256"])
    bindings = [(Path(plan_path), plan_sha256), (Path(plan["reference_power"]), plan["reference_power_sha256"])]
    evidence = plan["candidate_evidence"]
    binding = {k: v for k, v in evidence.items() if k.endswith("sha256")}
    if (reference.get("schema_version") != parent.VERSION or reference.get("status") != "complete"
            or reference.get("source_sha256") != parent.source_hashes()
            or reference.get("candidate_binding") != binding
            or reference.get("final_eval_label_prediction_metric_reads") != 0
            or reference["parameters"]["particles"] != plan["particle_budgets"][-1]
            or reference["parameters"]["step_seconds"] != plan["step_seconds"]
            or reference["parameters"]["delta_m"]/4 != plan["epsilon_m"]):
        raise ValueError("reference power evidence, candidate or numerical identity differs")
    inputs = plan["inputs"]
    if ([r["ledger_sha256"] for r in inputs] != reference["input_ledger_sha256"]
            or [r["audit_sha256"] for r in inputs] != reference["input_audit_sha256"]):
        raise ValueError("every complete parent source in its registered order is required")
    originals, reports, entries = [], [], {}
    identity = header = None
    seen = set()
    for spec in inputs:
        guard()
        path = Path(spec["ledger"])
        rows = read_bound(path, spec["ledger_sha256"], jsonl=True)
        original = read_bound(spec["audit"], spec["audit_sha256"])
        report = parent.audited(rows, path.parent, spec["ledger_sha256"], plan["epsilon_m"], evidence)
        if report != original or report["status"] != "complete":
            raise ValueError("complete original audit does not reproduce")
        init, current = rows[:2]
        current_identity = (parent._without(init, parent.WORKLOAD_FIELDS | {"started_at", "wall_seconds", "limit_origins"}),
            parent._without(current, parent.WORKLOAD_FIELDS | {"sample_ids", "selected_independent_block_count", "input_validation_seconds"}))
        if identity is not None and identity != current_identity:
            raise ValueError("sources mix model/map/runtime/selection identities")
        identity = current_identity
        if header is None:
            header = dict(current, seeds=list(SEEDS))
        for row in rows[2:-1]:
            key = tuple(row[k] for k in KEYS)
            if key in seen:
                raise ValueError("overlapping source workloads")
            seen.add(key)
            if row["particles"] == plan["particle_budgets"][-1] and row["max_step_seconds"] == plan["step_seconds"]:
                entries[(row["sample_id"], row["configuration"], row["seed"])] = (row, load_particle_evidence(row, path.parent))
        originals.append((path, rows))
        reports.append(report)
        bindings.extend(((path, spec["ledger_sha256"]), (Path(spec["audit"]), spec["audit_sha256"])))
        if progress:
            progress({"phase": "whole_parent_revalidated", "sources": len(reports), "source_runs": len(rows)-3})
    if [r["counts"] for r in reports] != reference["source_counts"]:
        raise ValueError("parent workload denominator changed")
    result = analyze(entries, header, plan["particle_budgets"], plan["epsilon_m"], progress, guard)
    guard()
    for path, rows in originals:
        for row in rows[2:-1]:
            load_particle_evidence(row, path.parent)
    bindings.extend((Path(evidence[k]), evidence[k+"_sha256"]) for k in ("fit", "fit_ledger"))
    if source_hashes() != sources or any(_hash(path) != digest for path, digest in bindings):
        raise ValueError("bound source or evidence changed during review")
    guard()
    result.update(schema_version=VERSION, status="complete", certified=False,
        numerically_qualified=False, formal_training_accepted=False,
        new_rollouts=0, final_eval_label_prediction_metric_reads=0,
        source_sha256=sources, plan_sha256=plan_sha256, reference_power_sha256=plan["reference_power_sha256"],
        candidate_binding=binding, source_counts=reference["source_counts"],
        original_numerical_failures=reference["source_numerical"],
        parameters={k: plan[k] for k in ("particle_budgets", "step_seconds", "epsilon_m", "alpha", "diagnostic_family_size", "wall_seconds")},
        selected_runs=len(entries), independent_blocks=header["selected_independent_block_count"],
        seeds=list(SEEDS), elapsed_seconds=time.perf_counter()-started,
        scope="conditional small-N diagnostics on the complete existing pilot; no new forecasts or general numerical/scientific acceptance",
        interval_caveat="Bonferroni normal margins on asymptotic paired jackknife SEs; not exact coverage or continuous-time bias bounds")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", dest="plan_path", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(**vars(parser.parse_args()), progress=lambda x: print(json.dumps(x), flush=True))
    print(json.dumps({k: result[k] for k in ("status", "selected_runs", "new_rollouts", "elapsed_seconds", "certified")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
