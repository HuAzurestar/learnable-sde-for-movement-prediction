"""Verify a larger-budget run preserves every registered smaller-budget path.

Both original ledgers must be complete and successfully replay every saved
array/score. This is prefix identity evidence, not numerical qualification.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .brownian import BROWNIAN_VERSION
from .numerical_check import audit
from .precision_check import KEYS, load_ledger, load_particle_evidence, replay
from .qualification import _hash

EXTENSION_VERSION = "pirc17-particle-extension-audit-v1"
PAIR_KEYS = tuple(k for k in KEYS if k != "particles")
HEADER_REQUIRED = {
    "fit_sha256", "development_identity", "source_sha256", "map_backend",
    "map_source_sha256", "rollout_version", "physical_history_step_seconds",
    "brownian_driver", "scoring_grid", "time_weights", "entropy_grid_m", "save_particles",
}
DRIVER_REQUIRED = {
    "version", "seed", "stream_tag", "stream_id", "stream_id_sha256",
    "seed_derivation", "draw_order", "numpy_version", "max_particles",
    "noise_dimensions", "atomic_intervals", "time_grid_sha256", "path_sha256",
}


def compare_extension(rows, reference, *, directory, reference_directory, tolerance_m):
    for ledger in (rows, reference):
        if audit(ledger, tolerance_m=tolerance_m)["failures"]:
            raise ValueError("every workload must succeed; no partial-success prefix audit")
        header = ledger[0]
        if not HEADER_REQUIRED <= header.keys() or header["save_particles"] is not True:
            raise ValueError("complete prediction provenance and saved particles required")
        if header["brownian_driver"] != BROWNIAN_VERSION:
            raise ValueError("particle extension requires the particle-major origin-bound driver")
        if any(type(n) is not int or n < 3 for n in header["particle_counts"]):
            raise ValueError("positive integer particle budgets of at least three required")
        if not header["source_sha256"] or not header["map_source_sha256"]:
            raise ValueError("prediction and map source bindings required")
    first, second = rows[0], reference[0]
    changed = {k for k in first.keys() | second.keys() if first.get(k) != second.get(k)}
    non_budget_changes = changed - {"particle_counts", "expected_run_count"}
    if non_budget_changes:
        raise ValueError("particle extension changes non-budget protocol fields: " + ", ".join(sorted(non_budget_changes)))
    reference_budget = max(second["particle_counts"])
    if min(first["particle_counts"]) <= reference_budget:
        raise ValueError("new budgets must all strictly exceed the complete reference maximum")

    # No bad workload or sidecar can be silently dropped from either original
    # denominator, including the smallest registered reference budget.
    for ledger, folder in ((rows, directory), (reference, reference_directory)):
        replay(ledger, folder, tolerance_m=tolerance_m)
    old = {tuple(row[k] for k in PAIR_KEYS)+(row["particles"],): row for row in reference[1:-1]}
    comparisons = []
    for row, prefix_budget in itertools.product(rows[1:-1], second["particle_counts"]):
        original = old[tuple(row[k] for k in PAIR_KEYS)+(prefix_budget,)]
        for name in ("independent_block_id", "actual_horizons_seconds"):
            if row[name] != original[name]:
                raise ValueError("particle extension changes blocks or scoring times")
        a, b = row["brownian_identity"], original["brownian_identity"]
        if not DRIVER_REQUIRED <= a.keys() or not DRIVER_REQUIRED <= b.keys():
            raise ValueError("complete Brownian grid/draw identity required")
        if (a["max_particles"] != max(first["particle_counts"])
                or b["max_particles"] != reference_budget):
            raise ValueError("driver maximum differs from registered particle budget")
        changed_driver = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        if changed_driver - {"max_particles", "path_sha256"}:
            raise ValueError("particle extension changes the random grid or draw identity")
        large, target, times = load_particle_evidence(row, directory)
        small, old_target, old_times = load_particle_evidence(original, reference_directory)
        equal = {"prefix_positions_m": np.array_equal(large[:prefix_budget], small),
                 "target_positions_m": np.array_equal(target, old_target),
                 "elapsed_seconds": np.array_equal(times, old_times)}
        comparisons.append({"candidate_workload": {k: row[k] for k in KEYS},
            "reference_workload": {k: original[k] for k in KEYS},
            "arrays_exactly_equal": equal, "passed": all(equal.values()),
            "max_prefix_coordinate_difference_m": float(np.max(np.abs(large[:prefix_budget]-small)))})
    return {"schema_version": EXTENSION_VERSION, "certified": False,
        "passed": all(row["passed"] for row in comparisons),
        "scope": "exact saved prefix/target/time identity only; NOT convergence, Monte Carlo precision or final efficacy",
        "reference_particle_budget": reference_budget,
        "reference_particle_budgets": second["particle_counts"],
        "candidate_particle_budgets": first["particle_counts"],
        "candidate_complete_run_count": len(rows)-2, "reference_complete_run_count": len(reference)-2,
        "reference_budgets_replayed": second["particle_counts"],
        "allowed_driver_changes": ["max_particles", "path_sha256"], "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ledger", "reference", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--tolerance-m", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new; original evidence is never overwritten")
    result = compare_extension(load_ledger(args.ledger, args.ledger_sha256),
        load_ledger(args.reference, args.reference_sha256), directory=args.ledger.parent,
        reference_directory=args.reference.parent, tolerance_m=args.tolerance_m)
    result["ledger_sha256"], result["reference_sha256"] = args.ledger_sha256, args.reference_sha256
    result["audit_source_sha256"] = {name: _hash(Path(__file__).with_name(name)) for name in
        ("particle_extension_check.py", "precision_check.py", "numerical_check.py", "precision.py",
         "nested_precision.py", "brownian.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "passed": result["passed"],
                      "comparisons": len(result["comparisons"]), "certified": False}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
