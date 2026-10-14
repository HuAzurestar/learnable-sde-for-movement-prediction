"""Model-bound numerical audit for the separate direct-linear candidate ledger.

Validate the candidate protocol first; only then adapt an in-memory copy to the
existing generic score/particle math. No legacy ledger is emitted or relabeled,
and different trained models are never compared as integration refinements.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .direct_linear import MODEL_VERSION
from .direct_linear_evidence import add_evidence_arguments, load_candidate, read_bound
from .direct_linear_fit import FIT_VERSION, MINIMUM_FREE_BYTES
from . import direct_linear_rollout as runner
from .numerical_check import audit as numerical_audit
from .precision_check import KEYS, replay
from .qualification import _hash

VERSION = "pirc17-direct-linear-numerical-audit-v1"


def check(rows, directory, *, tolerance_m, **evidence):
    if isinstance(tolerance_m, bool) or not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("positive finite sensitivity tolerance required")
    bundle, models, eligibility = load_candidate(**evidence)
    if len(rows) < 2 or rows[0].get("type") != "initialization" or rows[-1].get("type") != "completion":
        raise ValueError("candidate initialization and terminal completion required")
    init, end = rows[0], rows[-1]
    binding = {k:v for k,v in evidence.items() if k.endswith("sha256")}
    if (init.get("schema_version") != runner.VERSION or init.get("purpose") != runner.PURPOSE
            or any(init.get(k) != v for k,v in binding.items()) or init.get("eligibility_sha256") != eligibility
            or init.get("certified") is not False or init.get("final_eval_label_prediction_metric_reads") != 0
            or init.get("minimum_free_bytes") != MINIMUM_FREE_BYTES):
        raise ValueError("candidate initialization identity or resource contract differs")
    runner.validate_workload(init["configurations"], init["seeds"], init["particle_counts"], init["max_steps_seconds"],
                             init["limit_origins"], init["selection_policy"], init["map_backend"], init["wall_seconds"])
    count = init["limit_origins"]*len(init["configurations"])*len(init["seeds"])*len(init["particle_counts"])*len(init["max_steps_seconds"])
    if type(init.get("expected_run_count")) is not int or init["expected_run_count"] != count:
        raise ValueError("candidate registered count differs from its complete workload")
    headers = [row for row in rows[1:-1] if row.get("type") == "header"]
    runs = [row for row in rows[1:-1] if row.get("type") == "run"]
    errors = [row for row in rows[1:-1] if row.get("type") == "failure"]
    if len(headers) > 1 or len(headers)+len(runs)+len(errors) != len(rows)-2:
        raise ValueError("unexpected candidate ledger record")
    if headers and rows[1] != headers[0]:
        raise ValueError("candidate header must precede every attempted forecast")
    failed = sum(row.get("status") == "failure" for row in runs)
    success = sum(row.get("status") == "success" for row in runs)
    expected_counts = {"expected_run_count":count, "attempted_run_count":len(runs), "success_count":success,
                       "failure_count":failed, "unattempted_run_count":count-len(runs), "terminal_error_count":len(errors)}
    complete = success == count and not errors
    stopped = any(row.get("error_type") in {"MemoryError", "TimeoutError"} for row in runs+errors)
    if (success+failed != len(runs) or len(runs) > count
            or any(type(end.get(k)) is not int or end[k] != v for k,v in expected_counts.items())
            or end.get("status") != ("complete" if complete else "failed")
            or end.get("certified") is not False or end.get("formal_training_accepted") is not False
            or end.get("resource_stopped") is not stopped):
        raise ValueError("candidate completion does not preserve failures or the registered denominator")
    report = {"schema_version":VERSION, "candidate_binding":binding, "status":end["status"],
        "counts":expected_counts, "resource_stopped":stopped, "certified":False, "formal_training_accepted":False,
        "failures":errors+[r for r in runs if r["status"] == "failure"], "numerical_audit":None, "particle_precision":None,
        "scope":"candidate-only engineering sensitivity and conditional particle precision; not scientific power, efficacy or acceptance"}
    if not headers:
        if runs or not errors:
            raise ValueError("candidate forecasts or silent completion without verified header")
        return report
    header = headers[0]
    model_ids = {name:{str(seed):model.identity["sha256"] for seed,model in group.items()} for name,group in models.items()}
    _, modules = runner.resolve_map_backend(header["map_backend"])
    if (header.get("schema_version") != runner.VERSION or header.get("purpose") != runner.PURPOSE
            or any(header.get(k) != v for k,v in binding.items())
            or any(header.get(k) != init[k] for k in ("configurations", "seeds", "particle_counts", "max_steps_seconds",
                                                    "selection_policy", "map_backend", "expected_run_count"))
            or header.get("model_version") != MODEL_VERSION or header.get("fit_version") != FIT_VERSION
            or header.get("model_identities") != model_ids or header.get("development_identity") != bundle["development_identity"]["sha256"]
            or header.get("source_sha256") != runner.source_hashes()
            or header.get("map_source_sha256") != {m.__name__:_hash(Path(m.__file__)) for m in modules}
            or header.get("rollout_version") != runner.ROLLOUT_VERSION or header.get("brownian_driver") != runner.BROWNIAN_VERSION
            or header.get("physical_history_step_seconds") != 5. or header.get("scoring_grid") != runner.SCORING_GRID
            or header.get("time_weights") != list(runner.TIME_WEIGHTS) or header.get("entropy_grid_m") != runner.ENTROPY_GRID
            or header.get("selection") != runner.SELECTIONS[init["selection_policy"]]
            or header.get("save_particles") is not True or header.get("certified") is not False
            or header.get("formal_training_accepted") is not False or header.get("final_eval_label_prediction_metric_reads") != 0):
        raise ValueError("candidate header changes bound models, predictor, selection or scoring")
    samples = header["sample_ids"]
    if (len(samples) != init["limit_origins"] or len(set(samples)) != len(samples)
            or not set(samples) <= set(bundle["development_identity"]["sample_ids"]["validation"])):
        raise ValueError("candidate origins not in the bound validation population")
    expected = set(itertools.product(samples, init["configurations"], init["seeds"], init["particle_counts"], init["max_steps_seconds"]))
    seen, blocks = set(), {}
    for row in runs:
        key = tuple(row[k] for k in KEYS)
        if (type(row["seed"]) is not int or type(row["particles"]) is not int
                or key not in expected or key in seen
                or row.get("model_identity_sha256") != model_ids[row["configuration"]][str(row["seed"])]):
            raise ValueError("mixed, duplicate or unregistered candidate forecast")
        seen.add(key)
        block = row["independent_block_id"]
        if not block or blocks.setdefault(row["sample_id"], block) != block:
            raise ValueError("candidate origin changes independent block")
        times = np.asarray(row["actual_horizons_seconds"], dtype=float)
        if times.shape != (4,) or not np.isfinite(times).all() or np.any(np.abs(times-[60,300,900,1800]) > 30):
            raise ValueError("candidate forecast changes registered observed scoring horizon")
    if len(runs) == count:
        if seen != expected or len(set(blocks.values())) != header["selected_independent_block_count"]:
            raise ValueError("candidate origin/block denominator differs")
        if init["selection_policy"] == "lexical_independent_blocks" and len(set(blocks.values())) != len(samples):
            raise ValueError("candidate independent-block selection repeats a block")
    if errors or len(runs) != count:
        # An interrupted/failed source check cannot become a numerical result
        # merely because some earlier runs happened to save arrays successfully.
        return report
    generic = [dict(header, schema_version="pirc17-development-rollout-v1", purpose="bounded_validation_engineering_pilot"),
               *runs, end]
    report["engine_adapter"] = "validated candidate -> in-memory generic score/particle schema; original ledger unchanged"
    report["numerical_audit"] = numerical_audit(generic, tolerance_m=tolerance_m)
    report["particle_precision"] = replay(generic, directory, tolerance_m=tolerance_m)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_arguments(parser)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--tolerance-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output, path, sha = args.pop("output"), args.pop("ledger"), args.pop("ledger_sha256")
    if output.exists():
        raise FileExistsError("refusing to overwrite candidate numerical audit")
    result = check(read_bound(path, sha, jsonl=True), path.parent, **args)
    result["ledger_sha256"] = sha
    result["audit_source_sha256"] = {name:_hash(Path(__file__).with_name(name)) for name in
        ("direct_linear_check.py", "direct_linear_evidence.py", "numerical_check.py", "precision_check.py", "precision.py", "nested_precision.py")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(json.dumps({"status":result["status"], "counts":result["counts"], "certified":False}), flush=True)
    precision = result["particle_precision"] or {}
    return 1 if result["status"] != "complete" or precision.get("particle_budget_unavailable") else 0


if __name__ == "__main__":
    raise SystemExit(main())
