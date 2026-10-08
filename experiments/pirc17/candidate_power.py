"""Model-bound, complete-primary-family development power planning.

Reproduce every complete candidate audit before adapting in-memory records to
the existing equal-block/five-seed planning math. Never merge partial attempts,
drop failed settings, infer final efficacy or treat forecast seeds as blocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .direct_linear_evidence import add_evidence_arguments, read_bound
from .direct_linear_rollout import source_hashes as engine_sources
from .inference import PRIMARY_FAMILY, SEEDS
from .power_calibration import calibrate
from .precision_check import KEYS, load_particle_evidence
from .qualification import _hash
from .resource_replay import audited, numerical_summary, source_hashes as audit_sources

VERSION = "pirc17-candidate-primary-paired-power-v1"
WORKLOAD_FIELDS = {"configurations", "seeds", "particle_counts", "max_steps_seconds", "expected_run_count"}


def source_hashes():
    return {**engine_sources(), **audit_sources(), **{name: _hash(Path(__file__).with_name(name))
        for name in ("candidate_power.py", "power_calibration.py")}}


def _without(row, fields):
    return {key: value for key, value in row.items() if key not in fields}


def run(*, ledgers, ledger_sha256, audits, audit_sha256, fit, fit_sha256, fit_ledger,
        fit_ledger_sha256, training_policy_sha256, step_seconds, particles, delta_m,
        planning_block_counts, output):
    """No forecast, final-eval access, comparison subset or tolerance override."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite candidate power planning evidence")
    if not ledgers or not len(ledgers) == len(ledger_sha256) == len(audits) == len(audit_sha256):
        raise ValueError("one whole candidate ledger/audit and both hashes per source required")
    if (isinstance(delta_m, bool) or not isinstance(delta_m, (int, float)) or not np.isfinite(delta_m) or delta_m <= 0
            or isinstance(step_seconds, bool) or not isinstance(step_seconds, (int, float))
            or not np.isfinite(step_seconds) or step_seconds <= 0
            or type(particles) is not int or particles < 3
            or not planning_block_counts or any(type(n) is not int or n < 2 for n in planning_block_counts)
            or len(set(planning_block_counts)) != len(planning_block_counts)):
        raise ValueError("explicit finite planning margin, numerical settings and unique block scenarios required")
    sources = source_hashes()
    evidence = dict(fit=fit, fit_sha256=fit_sha256, fit_ledger=fit_ledger,
        fit_ledger_sha256=fit_ledger_sha256, training_policy_sha256=training_policy_sha256)
    bound = [(Path(fit), fit_sha256), (Path(fit_ledger), fit_ledger_sha256)]
    originals, generic, reports = [], [], []
    first_identity = None
    seen, seeds = set(), set()
    family_configurations = {name for pair in PRIMARY_FAMILY.values() for name in pair}
    for path, digest, audit_path, audit_digest in zip(ledgers, ledger_sha256, audits, audit_sha256):
        path, audit_path = Path(path), Path(audit_path)
        rows = read_bound(path, digest, jsonl=True)
        original = read_bound(audit_path, audit_digest)
        if (original.get("status") != "complete" or original.get("ledger_sha256") != digest
                or not original.get("numerical_audit")):
            raise ValueError("whole complete candidate audits required; no successful-subset power analysis")
        # The unchanged planning helper uses delta/4 for its engineering audit.
        # Require that relation explicitly, never silently replace a source tolerance.
        tolerance = original["numerical_audit"]["tolerance_m"]
        if tolerance != delta_m/4:
            raise ValueError("planning margin/4 must preserve the original registered numerical tolerance")
        checked = audited(rows, path.parent, digest, tolerance, evidence)
        if checked != original:
            raise ValueError("whole bound candidate audit does not exactly reproduce")
        init, header = rows[:2]
        if not family_configurations <= set(header["configurations"]):
            raise ValueError("each complete source must contain all five primary model comparisons")
        identity = (_without(init, WORKLOAD_FIELDS | {"started_at", "wall_seconds", "limit_origins"}),
            _without(header, WORKLOAD_FIELDS | {"sample_ids", "selected_independent_block_count", "input_validation_seconds"}))
        if first_identity is not None and identity != first_identity:
            raise ValueError("candidate power sources mix model, runtime, map, source or selection identities")
        first_identity = identity
        for row in rows[2:-1]:
            key = tuple(row[k] for k in KEYS)
            if key in seen:
                raise ValueError("overlapping workload across complete sources; no duplicate batch merging")
            seen.add(key)
        seeds.update(header["seeds"])
        originals.append(rows)
        generic.append([dict(header, schema_version="pirc17-development-rollout-v1",
            purpose="bounded_validation_engineering_pilot"), *rows[2:]])
        reports.append(checked)
        bound.extend(((path, digest), (audit_path, audit_digest)))
    if seeds != set(SEEDS):
        raise ValueError("candidate power planning requires all five fixed forecast/model seed labels")
    planning = calibrate(generic, step_seconds=step_seconds, particles=particles, delta_m=delta_m,
        planning_block_counts=planning_block_counts, comparisons=tuple(PRIMARY_FAMILY), family_size=len(PRIMARY_FAMILY))
    result = {"schema_version": VERSION, "status": "complete", "certified": False,
        "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0,
        "source_sha256": sources, "candidate_binding": reports[0]["candidate_binding"],
        "input_ledger_sha256": list(ledger_sha256), "input_audit_sha256": list(audit_sha256),
        "source_counts": [r["counts"] for r in reports],
        "source_numerical": [{**numerical_summary(report),
            "available_sensitivity_axes": sorted({r["axis"] for r in report["numerical_audit"]["sensitivities"]}),
            "numerically_qualified": False} for report in reports],
        "parameters": {"step_seconds": step_seconds, "particles": particles, "delta_m": delta_m,
            "planning_block_counts": list(planning_block_counts), "family_size": len(PRIMARY_FAMILY), "target_power": .8},
        "planning": planning,
        "adapter": "fully audited candidate records -> in-memory existing planning math; no legacy ledger written",
        "seed_interpretation": "five registered forecast/model labels, never independent research blocks or proof of distinct fitted parameters; deterministic fitting can yield equal coefficients",
        "scope": "development SD and prospective normal-planning scenarios only; not bootstrap power, convergence, final efficacy or protocol acceptance"}
    for rows, path in zip(originals, ledgers):
        for row in rows[2:-1]:
            load_particle_evidence(row, Path(path).parent)
    if source_hashes() != sources or any(_hash(path) != digest for path, digest in bound):
        raise ValueError("candidate power source or bound evidence changed during analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_arguments(parser)
    for name, destination in (("ledger", "ledgers"), ("audit", "audits")):
        parser.add_argument("--"+name, dest=destination, type=Path, action="append", required=True)
        parser.add_argument("--"+name+"-sha256", action="append", required=True)
    parser.add_argument("--step-seconds", type=float, required=True)
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--delta-m", type=float, required=True)
    parser.add_argument("--planning-blocks", dest="planning_block_counts", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "source_independent_blocks": result["planning"]["source_independent_blocks"],
        "source_origin_count": result["planning"]["source_origin_count"], "source_numerical": result["source_numerical"],
        "primary_comparisons": len(result["planning"]["results"]), "certified": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
