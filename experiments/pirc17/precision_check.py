"""Replay private development particles and audit paired numerical uncertainty.

Only aggregate diagnostics are exported. No final-eval loader is called and no
factor verdict or general numerical certificate is produced.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .inference import PRIMARY_FAMILY
from .nested_precision import nested_energy_precision
from .numerical_check import audit
from .precision import energy_delete_one, summarize_delete_one
from .qualification import _hash

KEYS = ("sample_id", "configuration", "seed", "particles", "max_step_seconds")


def aggregate_precision(comparisons, header):
    """Propagate particle covariance for the registered equal-block estimand.

    This conditions on the selected observations AND fitted models. It does not
    use the variation of block means as Monte Carlo variance or scientific CI.
    The caller has already replayed every paired array and checked its identity.
    """
    if header["brownian_driver"] != "pirc17-coupled-brownian-v2":
        raise ValueError("aggregate precision requires independent origin/seed streams")
    expected = set(itertools.product(header["sample_ids"], header["seeds"]))
    grouped = {}
    for row in comparisons:
        first, second = row["candidate_workload"], row["control_workload"]
        key = (row["kind"], row["comparison"], first["particles"], second["particles"],
               first["max_step_seconds"], second["max_step_seconds"])
        grouped.setdefault(key, []).append(row)
    output = []
    time_weights = np.asarray(header["time_weights"])
    for key, group in grouped.items():
        seen, origin_blocks = set(), {}
        for row in group:
            workload = row["candidate_workload"]
            origin, seed = workload["sample_id"], workload["seed"]
            if (origin, seed) in seen:
                raise ValueError("duplicate aggregate precision workload")
            seen.add((origin, seed))
            block = row["independent_block_id"]
            if origin in origin_blocks and origin_blocks[origin] != block:
                raise ValueError("aggregate origin changes independent block")
            origin_blocks[origin] = block
        if seen != expected:
            raise ValueError("aggregate precision requires every registered origin/seed")
        blocks = set(origin_blocks.values())
        counts = {block: list(origin_blocks.values()).count(block) for block in blocks}
        values = np.zeros(len(time_weights))
        covariance = np.zeros((len(time_weights), len(time_weights)))
        for row in group:
            weight = 1/(len(blocks)*counts[row["independent_block_id"]]*len(header["seeds"]))
            precision = row["precision"]
            values += weight*np.asarray(precision["by_time_energy_score_m"])
            covariance += weight**2*np.asarray(precision["time_covariance_m2"])
        output.append({"kind": key[0], "comparison": key[1], "particles": key[2],
            "candidate_particles": key[2], "control_particles": key[3],
            "candidate_step_seconds": key[4], "control_step_seconds": key[5],
            "independent_block_count": len(blocks), "origin_count": len(origin_blocks),
            "seeds": header["seeds"], "by_scoring_slot_difference_m": values.tolist(),
            "by_scoring_slot_standard_error_m": np.sqrt(np.maximum(0, covariance.diagonal())).tolist(),
            "time_weighted_difference_m": float(time_weights@values),
            "time_weighted_standard_error_m": float(np.sqrt(max(0, time_weights@covariance@time_weights))),
            "slot_covariance_m2": covariance.tolist(),
            "estimand": "equal selected blocks, equal registered seeds, equal origins within block",
            "scope": "conditional particle sampling only; NOT block sampling or training-seed uncertainty",
            "method": "sum squared estimand weights times paired path or compound-path-group jackknife covariance; independent origin/seed streams",
            "sign_convention": "candidate_ES_minus_control_ES"})
    return output


def load_ledger(path, digest):
    if _hash(path) != digest:
        raise ValueError("ledger hash mismatch")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_particle_evidence(row, directory):
    """Validate private sidecars without trusting stored paths, types or shapes."""
    directory = Path(directory).resolve()
    identity = row["particle_artifact"]
    relative = Path(identity["path"])
    path = (directory/relative).resolve()
    if (relative.is_absolute() or not path.is_relative_to(directory) or path.suffix != ".npz"
            or _hash(path) != identity["sha256"]):
        raise ValueError("particle artifact path or hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"positions_m", "target_positions_m", "elapsed_seconds"}:
            raise ValueError("unexpected particle artifact fields")
        particles, targets, horizons = (arrays[name] for name in
            ("positions_m", "target_positions_m", "elapsed_seconds"))
    times = row["actual_horizons_seconds"]
    if (particles.shape != (row["particles"], len(times), 2) or targets.shape != (len(times), 2)
            or horizons.shape != (len(times),) or not np.array_equal(horizons, times)):
        raise ValueError("particle dimensions or scoring times differ from ledger")
    if any(not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a) or not np.isfinite(a).all()
           for a in (particles, targets, horizons)):
        raise ValueError("finite real particle evidence required")
    return particles, targets, horizons


def replay(rows, directory, *, tolerance_m):
    audited = audit(rows, tolerance_m=tolerance_m)
    output = {"schema_version": "pirc17-particle-precision-audit-v3", "certified": False,
              "ledger_audit": audited, "runs": [], "paired_comparisons": [],
              "aggregate_precision": [], "particle_budget_unavailable": [],
              "scope": "conditional simulated-path sampling error; no final efficacy or block power certificate"}
    if audited["failures"]:
        return output
    header = rows[0]
    if header.get("save_particles") is not True:
        raise ValueError("ledger must declare saved particle evidence")
    weights = header["time_weights"]
    directory = Path(directory).resolve()
    data, metadata = {}, {}
    for row in rows[1:-1]:
        particles, targets, _ = load_particle_evidence(row, directory)
        score, deleted = energy_delete_one(particles, targets)
        expected = [v["energy_score_m"] for v in row["scores"]["by_time"]]
        if (not np.allclose(score, expected, rtol=1e-12, atol=1e-9)
                or not np.isclose(np.asarray(weights)@score,
                                  row["scores"]["time_weighted_energy_score_m"], rtol=1e-12, atol=1e-9)):
            raise ValueError("particle replay differs from recorded energy scores")
        key = tuple(row[name] for name in KEYS)
        data[key] = (score, deleted, targets)
        metadata[key] = row
        output["runs"].append({"workload": dict(zip(KEYS, key)),
                               "precision": summarize_delete_one(score, deleted, weights)})

    def paired_record(first, second, kind, comparison):
        first_targets, second_targets = data[first][2], data[second][2]
        left, right = metadata[first], metadata[second]
        if (left["brownian_identity"] != right["brownian_identity"]
                or left["independent_block_id"] != right["independent_block_id"]
                or left["actual_horizons_seconds"] != right["actual_horizons_seconds"]
                or not np.array_equal(first_targets, second_targets)):
            raise ValueError("paired evidence changes random paths, targets, times or block")
        return {"kind": kind, "comparison": comparison,
            "candidate_workload": dict(zip(KEYS, first)), "control_workload": dict(zip(KEYS, second)),
            "independent_block_id": left["independent_block_id"],
            "sign_convention": "candidate_ES_minus_control_ES"}

    def compare(first, second, kind, comparison):
        record = paired_record(first, second, kind, comparison)
        first_score, first_deleted, _ = data[first]
        second_score, second_deleted, _ = data[second]
        record["precision"] = summarize_delete_one(first_score-second_score, first_deleted-second_deleted, weights)
        output["paired_comparisons"].append(record)

    for sample, seed, particles in itertools.product(header["sample_ids"], header["seeds"], header["particle_counts"]):
        steps = sorted(header["max_steps_seconds"])
        for configuration in header["configurations"]:
            for finer, coarser in zip(steps, steps[1:]):
                compare((sample, configuration, seed, particles, finer),
                        (sample, configuration, seed, particles, coarser), "integration_refinement", configuration)
        for step in steps:
            for comparison, (candidate, control) in PRIMARY_FAMILY.items():
                if candidate in header["configurations"] and control in header["configurations"]:
                    compare((sample, candidate, seed, particles, step), (sample, control, seed, particles, step),
                            "development_model_contrast", comparison)
    budgets = sorted(header["particle_counts"])
    for sample, seed, configuration, step in itertools.product(
            header["sample_ids"], header["seeds"], header["configurations"], header["max_steps_seconds"]):
        for smaller, larger in zip(budgets, budgets[1:]):
            first = (sample, configuration, seed, larger, step)
            second = (sample, configuration, seed, smaller, step)
            record = paired_record(first, second, "particle_budget_refinement", configuration)
            if larger % smaller:
                record["reason"] = "compound-path jackknife requires an integer budget ratio; no independent-ensemble substitute"
                output["particle_budget_unavailable"].append(record)
                continue
            large, targets, _ = load_particle_evidence(metadata[first], directory)
            small, _, _ = load_particle_evidence(metadata[second], directory)
            record["precision"] = nested_energy_precision(large, small, targets, weights)
            output["paired_comparisons"].append(record)
    if header["brownian_driver"] == "pirc17-coupled-brownian-v2":
        output["aggregate_precision"] = aggregate_precision(output["paired_comparisons"], header)
    else:
        output["aggregate_unavailable_reason"] = "legacy driver does not establish independent origin streams"
    return output


def compare_reference(rows, reference, *, tolerance_m, directory=None, reference_directory=None):
    """Compare scores and, when explicitly supplied, every saved particle array."""
    if (directory is None) != (reference_directory is None):
        raise ValueError("both particle evidence directories are required")
    compare_arrays = directory is not None
    for ledger in (rows, reference):
        if audit(ledger, tolerance_m=tolerance_m)["failures"]:
            raise ValueError("failed workloads cannot establish backend equivalence")
        if compare_arrays and ledger[0].get("save_particles") is not True:
            raise ValueError("full array comparison requires both saved particle ledgers")
    for name in ("fit_sha256", "development_identity", "rollout_version", "sample_ids", "configurations",
                 "seeds", "particle_counts", "max_steps_seconds", "physical_history_step_seconds",
                 "brownian_driver", "scoring_grid", "time_weights", "entropy_grid_m"):
        if name not in rows[0] or rows[0][name] != reference[0].get(name):
            raise ValueError(f"backend comparison changes registered {name}")
    for name in ("rollout.py", "brownian.py", "dynamics.py", "checkpoints.py", "features.py",
                 "configurations.py", "metrics.py", "development.py"):
        if rows[0]["source_sha256"][name] != reference[0]["source_sha256"].get(name):
            raise ValueError("backend comparison changes predictor or scoring source")
    old = {tuple(row[k] for k in KEYS): row for row in reference[1:-1]}
    comparisons = []
    for row in rows[1:-1]:
        key = tuple(row[k] for k in KEYS)
        original = old[key]
        for name in ("brownian_identity", "actual_horizons_seconds", "independent_block_id"):
            if row[name] != original[name]:
                raise ValueError("backend comparison changes pairing")
        comparison = {"workload": dict(zip(KEYS, key)),
            "all_score_fields_exactly_equal": row["scores"] == original["scores"],
            "missing_feature_rows_equal": row["invalid_feature_rows"] == original["invalid_feature_rows"],
            "reference_wall_seconds": original["rollout_wall_seconds_including_lazy_map_initialization"],
            "candidate_wall_seconds": row["rollout_wall_seconds_including_lazy_map_initialization"]}
        if compare_arrays:
            first = load_particle_evidence(row, directory)
            second = load_particle_evidence(original, reference_directory)
            equal = [np.array_equal(a, b) for a, b in zip(first, second)]
            comparison["arrays_exactly_equal"] = dict(zip(
                ("positions_m", "target_positions_m", "elapsed_seconds"), equal))
            comparison["all_particle_arrays_exactly_equal"] = all(equal)
            comparison["max_coordinate_difference_m"] = float(np.max(np.abs(first[0]-second[0])))
        comparisons.append(comparison)
    return {"passed": all(r["all_score_fields_exactly_equal"] and r["missing_feature_rows_equal"]
                          and r.get("all_particle_arrays_exactly_equal", True) for r in comparisons),
            "particle_arrays_checked": compare_arrays,
            "scope": ("same registered predictor, draws, scores and full hash-bound positions/targets/times"
                      if compare_arrays else "same registered predictor, draws and aggregate scores only; particle arrays not compared"),
            "runs": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--tolerance-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_sha256) or args.output.exists():
        parser.error("reference requires a hash; output must be new")
    rows = load_ledger(args.ledger, args.ledger_sha256)
    result = replay(rows, args.ledger.parent, tolerance_m=args.tolerance_m)
    result["ledger_sha256"] = args.ledger_sha256
    result["source_sha256"] = {name: _hash(Path(__file__).with_name(name))
                               for name in ("precision_check.py", "precision.py", "nested_precision.py", "numerical_check.py")}
    if args.reference:
        result["reference_sha256"] = args.reference_sha256
        reference = load_ledger(args.reference, args.reference_sha256)
        arrays = (rows[0].get("save_particles") is True and reference[0].get("save_particles") is True)
        result["backend_replay"] = compare_reference(rows, reference, tolerance_m=args.tolerance_m,
            directory=args.ledger.parent if arrays else None,
            reference_directory=args.reference.parent if arrays else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "paired_comparisons": len(result["paired_comparisons"]),
                      "particle_budget_unavailable": len(result["particle_budget_unavailable"]),
                      "certified": False, "backend_replay": result.get("backend_replay", {}).get("passed")}))
    if (result["ledger_audit"]["failures"] or result["particle_budget_unavailable"]
            or result.get("backend_replay", {}).get("passed") is False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
