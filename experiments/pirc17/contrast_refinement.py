"""Offline joint numerical-refinement diagnostics for registered model contrasts.

Four ensembles share a random path/group deletion. This preserves covariance
between the two models AND between numerical resolutions. The result measures
conditional particle error, not a scientific confidence interval, convergence
certificate or replacement for the original per-model sensitivity audit.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .direct_linear_evidence import add_evidence_arguments, read_bound
from .direct_linear_rollout import source_hashes as engine_source_hashes
from .inference import PRIMARY_FAMILY
from .nested_precision import energy_delete_groups
from .precision import energy_delete_one, summarize_delete_one
from .precision_check import KEYS, aggregate_precision, load_particle_evidence
from .qualification import _hash
from .resource_replay import audited, source_hashes as audit_source_hashes

VERSION = "pirc17-joint-contrast-refinement-v1"
SIGN = "(candidate_refined_ES-control_refined_ES)-(candidate_reference_ES-control_reference_ES)"
SCOPE = "conditional simulated-path sampling only; not research-block, training or scientific uncertainty"


def source_hashes():
    return {**engine_source_hashes(), **audit_source_hashes(), **{name: _hash(Path(__file__).with_name(name))
            for name in ("contrast_refinement.py", "inference.py")}}


def joint_precision(candidate_refined, control_refined, candidate_reference,
                    control_reference, target, time_weights, *, kind):
    """Caller must verify shared origin/seed/Brownian, targets and time identity.

    Integration uses the same path index in all four arrays. Particle-budget
    refinement deletes group i=(i,i+n,...) from both N-path ensembles and path i
    from both n-path prefixes. Neither pairing treats the models as independent.
    """
    arrays = [np.asarray(value, dtype=float) for value in
              (candidate_refined, control_refined, candidate_reference, control_reference)]
    if (any(x.ndim != 3 or len(x) < 3 for x in arrays)
            or arrays[0].shape != arrays[1].shape or arrays[2].shape != arrays[3].shape
            or arrays[0].shape[1:] != arrays[2].shape[1:]):
        raise ValueError("four paired path/time arrays with at least three paths required")
    large, small = len(arrays[0]), len(arrays[2])
    if kind == "integration":
        if large != small:
            raise ValueError("integration contrasts require matching particle budgets")
        computed = [energy_delete_one(x, target) for x in arrays]
        method = "joint four-ensemble delete-one-path jackknife; asymptotic diagnostic"
    elif kind == "particles":
        if large <= small or large % small:
            raise ValueError("particle refinement requires a strictly larger integer-ratio budget")
        if any(not np.array_equal(arrays[i][:small], arrays[i+2]) for i in (0, 1)):
            raise ValueError("both model ensembles must preserve their exact shared prefix")
        computed = [energy_delete_groups(x, target, small) for x in arrays[:2]]
        computed += [energy_delete_one(x, target) for x in arrays[2:]]
        method = "joint four-ensemble delete-one-iid-compound-path-group jackknife; asymptotic diagnostic"
    else:
        raise ValueError("refinement kind must be integration or particles")
    scores, deleted = zip(*computed)
    values = (scores[0]-scores[1])-(scores[2]-scores[3])
    leave_out = (deleted[0]-deleted[1])-(deleted[2]-deleted[3])
    result = summarize_delete_one(values, leave_out, time_weights)
    result.pop("particle_count")  # Group count is not a research sample size.
    result.update(schema_version=VERSION, refinement_kind=kind,
        refined_particle_count=large, reference_particle_count=small,
        jackknife_group_count=small, refined_paths_per_group=large//small,
        method=method, scope=SCOPE, sign_convention=SIGN, certified=False)
    return result


def _diagnostics(rows, directory):
    """Consume only a whole ledger already accepted by the bound original audit."""
    header = rows[1]
    metadata = {tuple(row[k] for k in KEYS): row for row in rows[2:-1]}
    records, unavailable = [], []
    steps, budgets = sorted(header["max_steps_seconds"]), sorted(header["particle_counts"])
    missing = []
    for comparison, (candidate, control) in PRIMARY_FAMILY.items():
        absent = sorted({candidate, control}-set(header["configurations"]))
        if absent:
            missing.append({"comparison": comparison, "missing_configurations": absent,
                            "reason": "required model contrast not in the complete registered source workload"})
            continue
        if len(steps) < 2:
            unavailable.append({"comparison": comparison, "kind": "integration",
                                "reason": "at least two registered integration steps required"})
        if len(budgets) < 2:
            unavailable.append({"comparison": comparison, "kind": "particles",
                                "reason": "at least two registered particle budgets required"})

        def record(sample, seed, refined_count, reference_count, refined_step, reference_step, kind):
            keys = [(sample, candidate, seed, refined_count, refined_step),
                    (sample, control, seed, refined_count, refined_step),
                    (sample, candidate, seed, reference_count, reference_step),
                    (sample, control, seed, reference_count, reference_step)]
            group = [metadata[key] for key in keys]
            loaded = [load_particle_evidence(row, directory) for row in group]
            for row, (_, target, times) in zip(group[1:], loaded[1:]):
                if (any(row[key] != group[0][key] for key in
                        ("sample_id", "seed", "independent_block_id", "brownian_identity", "actual_horizons_seconds"))
                        or not np.array_equal(target, loaded[0][1]) or not np.array_equal(times, loaded[0][2])):
                    raise ValueError("joint refinement changes paired stream, block, targets or times")
            precision = joint_precision(*(item[0] for item in loaded), loaded[0][1], header["time_weights"], kind=kind)
            records.append({"kind": "joint_"+kind+"_refinement", "comparison": comparison,
                # The aggregate helper uses the candidate MODEL's refined/reference axes.
                # All four full identities are also retained, without relabeling models.
                "candidate_workload": dict(zip(KEYS, keys[0])), "control_workload": dict(zip(KEYS, keys[2])),
                "four_workloads": {name: dict(zip(KEYS, key)) for name, key in zip(
                    ("candidate_refined", "control_refined", "candidate_reference", "control_reference"), keys)},
                "independent_block_id": group[0]["independent_block_id"],
                "sign_convention": SIGN, "precision": precision})

        for sample, seed in itertools.product(header["sample_ids"], header["seeds"]):
            for count, (fine, coarse) in itertools.product(budgets, zip(steps, steps[1:])):
                record(sample, seed, count, count, fine, coarse, "integration")
            for step, (small, large) in itertools.product(steps, zip(budgets, budgets[1:])):
                if large % small:
                    unavailable.append({"comparison": comparison, "kind": "particles", "sample_id": sample,
                        "seed": seed, "step_seconds": step, "reference_particles": small, "refined_particles": large,
                        "reason": "joint compound-path deletion requires an integer budget ratio"})
                else:
                    record(sample, seed, large, small, step, step, "particles")
    aggregates = aggregate_precision(records, header)
    for row in aggregates:
        row.pop("particles")
        for old, new in (("candidate_particles", "refined_particles"), ("control_particles", "reference_particles"),
                         ("candidate_step_seconds", "refined_step_seconds"), ("control_step_seconds", "reference_step_seconds")):
            row[new] = row.pop(old)
        row.update(sign_convention=SIGN, scope=SCOPE, certified=False,
            method="sum squared registered estimand weights times joint four-ensemble path/group covariance; independent origin/seed streams")
    return {"comparisons": records, "aggregate_precision": aggregates,
            "missing_primary_comparisons": missing, "unavailable_refinements": unavailable,
            "full_primary_family_available": not missing,
            "refinement_diagnostics_available": bool(records) and not unavailable}


def run(*, ledger, ledger_sha256, audit, audit_sha256, fit, fit_sha256,
        fit_ledger, fit_ledger_sha256, training_policy_sha256, output):
    """No forecast, tolerance override, final-eval loader or partial-run salvage."""
    output, ledger, audit = Path(output), Path(ledger).resolve(), Path(audit).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite joint refinement diagnostic")
    sources = source_hashes()
    rows = read_bound(ledger, ledger_sha256, jsonl=True)
    original = read_bound(audit, audit_sha256)
    if (original.get("ledger_sha256") != ledger_sha256 or not original.get("numerical_audit")
            or original.get("status") != "complete"):
        raise ValueError("complete bound original candidate audit required")
    tolerance = original["numerical_audit"]["tolerance_m"]
    evidence = dict(fit=fit, fit_sha256=fit_sha256, fit_ledger=fit_ledger,
                    fit_ledger_sha256=fit_ledger_sha256, training_policy_sha256=training_policy_sha256)
    checked = audited(rows, ledger.parent, ledger_sha256, tolerance, evidence)
    if checked != original:
        raise ValueError("original candidate audit does not exactly reproduce")
    result = _diagnostics(rows, ledger.parent)
    sensitivities = original["numerical_audit"]["sensitivities"]
    result.update(schema_version=VERSION, status="complete", certified=False,
        formal_training_accepted=False, final_eval_label_prediction_metric_reads=0,
        ledger_sha256=ledger_sha256, audit_sha256=audit_sha256, source_sha256=sources,
        candidate_binding=checked["candidate_binding"], source_counts=checked["counts"],
        original_numerical_diagnostic={"tolerance_m": tolerance, "checked": len(sensitivities),
            "out_of_tolerance": sum(not r["all_scoring_times_within_tolerance"] for r in sensitivities),
            "unchanged": True, "certified": False},
        scope="additional conditional model-contrast diagnostic; no tolerance, denominator, protocol or scientific verdict change")
    # Revalidate immutable sources/evidence after all calculations, before output.
    if (source_hashes() != sources or _hash(ledger) != ledger_sha256 or _hash(audit) != audit_sha256
            or _hash(Path(fit)) != fit_sha256 or _hash(Path(fit_ledger)) != fit_ledger_sha256):
        raise ValueError("joint diagnostic source or bound evidence changed during analysis")
    for row in rows[2:-1]:
        load_particle_evidence(row, ledger.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_arguments(parser)
    for name in ("ledger", "audit"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "joint_comparisons": len(result["comparisons"]),
        "aggregate_comparisons": len(result["aggregate_precision"]),
        "missing_primary_comparisons": len(result["missing_primary_comparisons"]),
        "original_numerical_diagnostic": result["original_numerical_diagnostic"], "certified": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
