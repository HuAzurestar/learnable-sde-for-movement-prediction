"""Offline identity audit for a complete candidate configuration/budget extension.

Both original audits must reproduce in full before any intersection is examined.
The new workload retains the old maximum budget and adds larger budgets/models.
Every old budget is checked against every compatible new ensemble; unmatched
new configurations remain in the complete audit, not silently discarded.
This establishes saved-path identity only, never numerical qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .direct_linear_evidence import add_evidence_arguments, read_bound
from .direct_linear_rollout import source_hashes as engine_sources
from .particle_extension_check import DRIVER_REQUIRED, PAIR_KEYS
from .precision_check import KEYS, load_particle_evidence
from .qualification import _hash
from .resource_replay import audited, numerical_summary, source_hashes as audit_sources

VERSION = "pirc17-candidate-overlap-extension-audit-v1"
WORKLOAD_CHANGES = {"configurations", "particle_counts", "expected_run_count"}
DRIVER_CHANGES = {"max_particles", "path_sha256"}


def source_hashes():
    return {**engine_sources(), **audit_sources(), **{name: _hash(Path(__file__).with_name(name))
        for name in ("candidate_extension_check.py", "particle_extension_check.py")}}


def _without(row, names):
    return {key: value for key, value in row.items() if key not in names}


def _compare(candidate, reference, directory, reference_directory):
    """Internal: both complete model-bound ledgers have already been audited."""
    new, old = candidate[1], reference[1]
    if (_without(candidate[0], WORKLOAD_CHANGES | {"started_at", "wall_seconds"}) !=
            _without(reference[0], WORKLOAD_CHANGES | {"started_at", "wall_seconds"})
            or _without(new, WORKLOAD_CHANGES | {"input_validation_seconds"}) !=
               _without(old, WORKLOAD_CHANGES | {"input_validation_seconds"})):
        raise ValueError("extension changes a non-extension protocol, model, runtime, grid or source binding")
    if (not set(old["configurations"]) <= set(new["configurations"])
            or min(new["particle_counts"]) != max(old["particle_counts"])
            or max(new["particle_counts"]) <= max(old["particle_counts"])):
        raise ValueError("extension must retain all reference models and its maximum budget, then add larger budgets")

    originals = {tuple(row[k] for k in PAIR_KEYS)+(row["particles"],): row for row in reference[2:-1]}
    checked_reference = set()
    comparisons = []
    for row in candidate[2:-1]:
        if row["configuration"] not in old["configurations"]:
            continue  # Still included in the full candidate audit and final rehash.
        large, target, times = load_particle_evidence(row, directory)
        for budget in old["particle_counts"]:
            key = tuple(row[k] for k in PAIR_KEYS)+(budget,)
            original = originals[key]
            checked_reference.add(key)
            for name in ("independent_block_id", "actual_horizons_seconds", "model_identity_sha256"):
                if row[name] != original[name]:
                    raise ValueError("extension changes paired blocks, observed times or fitted model")
            a, b = row["brownian_identity"], original["brownian_identity"]
            if (not DRIVER_REQUIRED <= a.keys() or not DRIVER_REQUIRED <= b.keys()
                    or a["max_particles"] != max(new["particle_counts"])
                    or b["max_particles"] != max(old["particle_counts"])
                    or _without(a, DRIVER_CHANGES) != _without(b, DRIVER_CHANGES)):
                raise ValueError("extension changes Brownian grid, origin stream or draw order")
            small, old_target, old_times = load_particle_evidence(original, reference_directory)
            equal = {"prefix_positions_m": bool(np.array_equal(large[:budget], small)),
                "target_positions_m": bool(np.array_equal(target, old_target)),
                "elapsed_seconds": bool(np.array_equal(times, old_times))}
            overlap = row["particles"] == budget
            ignored = {"brownian_identity", "particle_artifact", "rollout_wall_seconds_including_lazy_map_initialization"}
            same_fields = _without(row, ignored) == _without(original, ignored) if overlap else None
            comparisons.append({"candidate_workload": {k: row[k] for k in KEYS},
                "reference_workload": {k: original[k] for k in KEYS}, "same_budget_overlap": overlap,
                "arrays_exactly_equal": equal, "same_budget_other_run_fields_exact": same_fields,
                "max_prefix_coordinate_difference_m": float(np.max(np.abs(large[:budget]-small))),
                "passed": all(equal.values()) and (same_fields if overlap else True)})
    if checked_reference != set(originals) or not comparisons:
        raise ValueError("every original workload must participate; no selected successful intersection")
    overlap_count = sum(row["same_budget_overlap"] for row in comparisons)
    return {"passed": all(row["passed"] for row in comparisons), "comparisons": comparisons,
        "comparison_count": len(comparisons), "same_budget_overlap_count": overlap_count,
        "strict_prefix_comparison_count": len(comparisons)-overlap_count,
        "reference_runs_covered": len(checked_reference),
        "new_configuration_run_count": sum(r["configuration"] not in old["configurations"] for r in candidate[2:-1]),
        "added_configurations": sorted(set(new["configurations"])-set(old["configurations"])),
        "reference_particle_budgets": old["particle_counts"], "candidate_particle_budgets": new["particle_counts"],
        "reference_wall_budget_seconds": reference[0]["wall_seconds"],
        "candidate_wall_budget_seconds": candidate[0]["wall_seconds"],
        "allowed_driver_changes": sorted(DRIVER_CHANGES)}


def run(*, ledger, ledger_sha256, audit, audit_sha256, reference_ledger, reference_ledger_sha256,
        reference_audit, reference_audit_sha256, fit, fit_sha256, fit_ledger, fit_ledger_sha256,
        training_policy_sha256, output):
    """No forecasts, tolerance override, subset selection or partial-run salvage."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite candidate extension audit")
    sources = source_hashes()
    evidence = dict(fit=fit, fit_sha256=fit_sha256, fit_ledger=fit_ledger,
        fit_ledger_sha256=fit_ledger_sha256, training_policy_sha256=training_policy_sha256)
    bound = {Path(ledger): ledger_sha256, Path(audit): audit_sha256,
        Path(reference_ledger): reference_ledger_sha256, Path(reference_audit): reference_audit_sha256,
        Path(fit): fit_sha256, Path(fit_ledger): fit_ledger_sha256}
    records, reports = [], []
    for path, digest, report_path, report_digest in (
            (ledger, ledger_sha256, audit, audit_sha256),
            (reference_ledger, reference_ledger_sha256, reference_audit, reference_audit_sha256)):
        rows = read_bound(Path(path), digest, jsonl=True)
        original = read_bound(Path(report_path), report_digest)
        if (original.get("status") != "complete" or original.get("ledger_sha256") != digest
                or not original.get("numerical_audit")):
            raise ValueError("both whole complete candidate audits are required before comparing any intersection")
        tolerance = original["numerical_audit"]["tolerance_m"]
        checked = audited(rows, Path(path).parent, digest, tolerance, evidence)
        if checked != original:
            raise ValueError("complete bound candidate audit does not exactly reproduce")
        records.append(rows)
        reports.append(checked)
    if reports[0]["numerical_audit"]["tolerance_m"] != reports[1]["numerical_audit"]["tolerance_m"]:
        raise ValueError("extension must preserve the original numerical tolerance")
    result = _compare(records[0], records[1], Path(ledger).parent, Path(reference_ledger).parent)
    result.update(schema_version=VERSION, status="complete", certified=False,
        formal_training_accepted=False, final_eval_label_prediction_metric_reads=0,
        ledger_sha256=ledger_sha256, audit_sha256=audit_sha256,
        reference_ledger_sha256=reference_ledger_sha256, reference_audit_sha256=reference_audit_sha256,
        candidate_binding=reports[0]["candidate_binding"], source_sha256=sources,
        candidate_counts=reports[0]["counts"], reference_counts=reports[1]["counts"],
        candidate_numerical=numerical_summary(reports[0]), reference_numerical=numerical_summary(reports[1]),
        scope="exact saved overlapping/prefix path, target and time identity only; not convergence, scientific uncertainty, power or efficacy")
    # Revalidate ALL arrays, including new configurations with no old counterpart.
    for rows, path in zip(records, (ledger, reference_ledger)):
        for row in rows[2:-1]:
            load_particle_evidence(row, Path(path).parent)
    if source_hashes() != sources or any(_hash(path) != digest for path, digest in bound.items()):
        raise ValueError("extension auditor sources or bound evidence changed during analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_arguments(parser)
    for name in ("ledger", "audit", "reference-ledger", "reference-audit"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("status", "passed", "comparison_count",
        "same_budget_overlap_count", "strict_prefix_comparison_count", "reference_runs_covered",
        "candidate_numerical", "reference_numerical", "certified")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
