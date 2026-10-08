"""Read-only admission checks for an interrupted native seed-planning batch.

This is NOT a completed ledger, a resource certificate, or an executor. It
proves which exact successful prefix can be inherited by a separately recorded
attempt. Nothing is appended to the old files and missing terminal records are
never synthesized. Failed, reordered and torn legacy ledgers fail closed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import itertools
import json
from pathlib import Path
import re

from . import native_qualification as native
from .precision_check import KEYS, load_particle_evidence

VERSION = "pirc17-interrupted-seed-prefix-v1"
engine = native.engine
RUN_FIELDS = {"type", *KEYS, "independent_block_id", "model_identity_sha256", "brownian_identity",
    "actual_horizons_seconds", "status", "rollout_wall_seconds_including_lazy_map_initialization",
    "invalid_feature_rows", "feature_query_rows", "scores", "particle_precision", "particle_artifact"}


def source_hashes():
    return {**native.source_hashes(), "seed_resume.py": native._hash(Path(__file__))}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member in interrupted evidence")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite JSON in interrupted evidence")


def read_closed(path, digest, *, jsonl=False):
    """Strict snapshot read; even a torn final row is not silently discarded."""
    path = Path(path)
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("explicit SHA-256 of closed evidence required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("closed evidence hash mismatch")
    text = raw.decode("utf-8")
    if jsonl and (not text or not text.endswith("\n")):
        raise ValueError("closed JSONL must end at a complete newline boundary")
    decode = lambda value: json.loads(value, object_pairs_hook=_object, parse_constant=_nonfinite)
    return [decode(line) for line in text.splitlines()] if jsonl else decode(text)


def workload_keys(spec, samples):
    """Original engine order, including full axes even when a prefix is saved."""
    return [(sample, name, seed, count, step) for sample, seed, name, count, step in
            itertools.product(samples, spec["seeds"], spec["configurations"], spec["particles"], spec["steps"])]


def entropy_grid():
    edges = tuple(engine.np.arange(-10000, 10001, 250))
    return engine.EntropyGrid(edges, edges)


def validate_prefix_rows(spec, reference, candidate, directory, reference_directory, *, guard, offset=0):
    """Validate a contiguous slice after a separately reproduced whole-reference audit.

    Legacy admission always starts at zero. Closed continuation journals may
    supply a nonzero offset, established by their complete ordered ancestry.
    """
    if (len(candidate) < 2 or candidate[0].get("type") != "initialization"
            or candidate[1].get("type") != "header"
            or any(row.get("type") != "run" or row.get("status") != "success" for row in candidate[2:])):
        raise ValueError("only an initialization/header and complete successful prefix can be resumed")
    originals = native.validate_seed_result(spec, reference, candidate)
    init, header = candidate[:2]
    for record in (init, header):
        engine.validate_workload(record["configurations"], record["seeds"], record["particle_counts"],
            record["max_steps_seconds"], init["limit_origins"], record["selection_policy"],
            record["map_backend"], init["wall_seconds"])
        if type(record.get("expected_run_count")) is not int:
            raise ValueError("integer prefix workload denominator required")
    for name, value in (("seeds", spec["seeds"]), ("particle_counts", spec["particles"]),
                        ("max_steps_seconds", spec["steps"]), ("expected_run_count", spec["expected_run_count"]),
                        ("wall_seconds", spec["wall_seconds"])):
        if init.get(name) != value:
            raise ValueError("interrupted initialization differs from the registered seed plan")
    keys = workload_keys(spec, header["sample_ids"])
    rows = candidate[2:]
    if (type(offset) is not int or not 0 <= offset <= len(keys)
            or len(keys) != spec["expected_run_count"] or len(rows)+offset > len(keys)):
        raise ValueError("interrupted prefix exceeds the original denominator")
    drivers = {}
    grid = entropy_grid()
    for index, row in enumerate(rows):
        guard()
        if (set(row) != RUN_FIELDS or tuple(row.get(key) for key in KEYS) != keys[index+offset]
                or type(row.get("seed")) is not int or type(row.get("particles")) is not int
                or isinstance(row.get("max_step_seconds"), bool)
                or row.get("model_identity_sha256") != header["model_identities"][row["configuration"]][str(row["seed"])]
                or any(key in row for key in ("error_type", "error_message", "elapsed_seconds_before_failure"))):
            raise ValueError("mixed, reordered, duplicate, failed or unregistered prefix row")
        previous = originals[(row["sample_id"], row["configuration"])]
        arrays = load_particle_evidence(row, directory)
        original_arrays = load_particle_evidence(previous, reference_directory)
        if (row.get("independent_block_id") != previous["independent_block_id"]
                or row.get("actual_horizons_seconds") != previous["actual_horizons_seconds"]
                or not all(engine.np.array_equal(a, b) for a, b in zip(arrays[1:], original_arrays[1:]))):
            raise ValueError("prefix changes reference block, observed targets or times")
        stream = (row["sample_id"], row["seed"])
        if stream not in drivers:
            # Retain only small identities, never all resident Brownian arrays.
            driver = engine.BrownianPath(arrays[2], spec["steps"], history_step_seconds=5.,
                particles=max(spec["particles"]), seed=row["seed"], stream_id=row["sample_id"])
            drivers[stream] = driver.identity
            del driver
        if row.get("brownian_identity") != drivers[stream]:
            raise ValueError("prefix Brownian stream does not reproduce on the complete original grid")
        scores = engine.score_path(*arrays, time_weights=engine.TIME_WEIGHTS, entropy_grid=grid)
        precision = engine.energy_precision(arrays[0], arrays[1], engine.TIME_WEIGHTS)
        if row.get("scores") != scores or row.get("particle_precision") != precision:
            raise ValueError("prefix scores or conditional particle precision do not exactly reproduce")
        if (any(type(row.get(key)) is not int or row[key] < 0 for key in
                ("invalid_feature_rows", "feature_query_rows"))
                or row["invalid_feature_rows"] > row["feature_query_rows"]):
            raise ValueError("invalid prefix feature counters")
        wall = row.get("rollout_wall_seconds_including_lazy_map_initialization")
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not 0 <= wall < float("inf"):
            raise ValueError("invalid measured prefix runtime")
    return keys


@dataclass
class ValidatedPrefix:
    spec: dict
    initialization: dict
    header: dict
    rows: list
    remaining_keys: list
    report: dict


def inspect(*, plan, plan_sha256, reference_ledger, reference_audit, fit, fit_ledger,
            envelope, envelope_sha256, forecast_sha256, supervisor_sha256, guard=None):
    """Validate closed legacy evidence without starting or reserving any output.

    A later executor MUST independently recheck development/map inputs, reserve
    a new attempt, enforce its registered resource limits, and validate all rows
    before publishing a whole-workload result. This report cannot authorize it.
    """
    guard = guard or native.MemoryObserver().guard
    guard()
    spec = native.load_plan(plan, plan_sha256)
    if spec["schema_version"] != native.SEED_PLAN_VERSION:
        raise ValueError("interrupted-prefix import is restricted to the registered seed profile")
    paths = native.output_paths(envelope)
    bound = {Path(plan).resolve(): plan_sha256, paths["envelope"]: envelope_sha256,
        paths["forecast"]: forecast_sha256, paths["supervisor"]: supervisor_sha256,
        Path(reference_ledger).resolve(): spec["reference_ledger_sha256"],
        Path(reference_audit).resolve(): spec["reference_audit_sha256"],
        Path(fit).resolve(): spec["fit_sha256"], Path(fit_ledger).resolve(): spec["fit_ledger_sha256"]}
    sources = source_hashes()
    supervisor = read_closed(paths["supervisor"], supervisor_sha256)
    if (supervisor.get("schema_version") != native.VERSION+"-supervisor"
            or supervisor.get("plan_sha256") != plan_sha256 or supervisor.get("status") != "failed"
            or supervisor.get("termination_confirmed") is not True
            or type(supervisor.get("worker_returncode")) is not int or supervisor["worker_returncode"] == 0
            or any(type(supervisor.get(key)) is not int or supervisor[key] <= 0 for key in ("supervisor_pid", "worker_pid"))
            or type(supervisor.get("timed_out")) is not bool
            or supervisor.get("outer_wall_seconds") != spec["outer_wall_seconds"]
            or any(supervisor.get(key) is not False for key in ("certified", "formal_training_accepted"))
            or supervisor.get("final_eval_label_prediction_metric_reads") != 0):
        raise ValueError("matching closed, failed supervisor with confirmed worker termination required")
    started, deadline, finished = [datetime.fromisoformat(supervisor[key]) for key in
                                 ("started_at", "deadline_at", "finished_at")]
    if (any(value.tzinfo is None for value in (started, deadline, finished)) or finished < started
            or deadline != started+timedelta(seconds=spec["outer_wall_seconds"])
            or not isinstance(supervisor.get("elapsed_seconds"), (int, float))
            or isinstance(supervisor["elapsed_seconds"], bool)
            or not 0 <= supervisor["elapsed_seconds"] < float("inf")):
        raise ValueError("invalid interrupted attempt time/deadline evidence")
    outer = read_closed(paths["envelope"], envelope_sha256, jsonl=True)
    if (len(outer) != 2 or [row.get("type") for row in outer] != ["initialization", "reference_verified"]
            or paths["audit"].exists()):
        raise ValueError("only an interrupted pre-completion envelope without a candidate audit is supported")
    initial = outer[0]
    if (initial.get("schema_version") != native.VERSION or initial.get("purpose") != spec["purpose"]
            or initial.get("plan") != spec or initial.get("plan_sha256") != plan_sha256
            or initial.get("source_sha256") != native.source_hashes()
            or initial.get("observer") != native.physical_memory.identity()
            or initial.get("forecast_path") != paths["forecast"].name
            or initial.get("audit_path") != paths["audit"].name
            or initial.get("expected_run_count") != spec["expected_run_count"]
            or any(initial.get(key) is not False for key in ("certified", "formal_training_accepted"))
            or initial.get("final_eval_label_prediction_metric_reads") != 0):
        raise ValueError("interrupted envelope changes source, input, resource or workload bindings")
    guard()
    evidence = dict(fit=fit, fit_ledger=fit_ledger, **{key: spec[key] for key in
                    ("fit_sha256", "fit_ledger_sha256", "training_policy_sha256")})
    reference = read_closed(reference_ledger, spec["reference_ledger_sha256"], jsonl=True)
    original = read_closed(reference_audit, spec["reference_audit_sha256"])
    checked = native.audited(reference, Path(reference_ledger).resolve().parent,
                             spec["reference_ledger_sha256"], spec["tolerance_m"], evidence)
    if (checked != original or checked["status"] != "complete"
            or reference[0]["eligibility_sha256"] != spec["eligibility_sha256"]):
        raise ValueError("whole reference audit and original fit eligibility must reproduce unchanged")
    contract = native.seed_reference_contract(spec, reference, checked)
    if outer[1] != {"type": "reference_verified", "numerical": native.numerical_summary(checked),
                    "seed_planning_contract": contract}:
        raise ValueError("interrupted envelope changes whole-reference failures or seed contract")
    runtime = {"python": engine.platform.python_version(), "numpy": engine.np.__version__,
        "torch": engine.torch.__version__, "torch_intraop_threads": engine.torch.get_num_threads(),
        "torch_interop_threads": engine.torch.get_num_interop_threads()}
    if reference[0].get("runtime") != runtime:
        raise ValueError("prefix replay requires the original runtime and thread counts")
    candidate = read_closed(paths["forecast"], forecast_sha256, jsonl=True)
    keys = validate_prefix_rows(spec, reference, candidate, paths["forecast"].parent,
                                Path(reference_ledger).resolve().parent, guard=guard)
    claimed = set()
    for row in candidate[2:]:
        key = {name: row[name] for name in KEYS}
        filename = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()+".npz"
        path = paths["particles"]/filename
        if row["particle_artifact"]["path"] != path.relative_to(paths["forecast"].parent).as_posix():
            raise ValueError("prefix particle path differs from the original exclusive writer")
        bound[path] = row["particle_artifact"]["sha256"]
        claimed.add(path)
    orphans = []
    for path in sorted(paths["particles"].glob("*.npz")):
        if path not in claimed:
            # Saved before a missing row: retain for forensic review, never reuse.
            digest = native._hash(path)
            bound[path] = digest
            orphans.append({"path": path.name, "sha256": digest})
    for row in reference[2:-1]:
        bound[(Path(reference_ledger).resolve().parent/row["particle_artifact"]["path"]).resolve()] = row["particle_artifact"]["sha256"]
    guard()
    if source_hashes() != sources or any(native._hash(path) != digest for path, digest in bound.items()):
        raise ValueError("bound evidence or sources changed during prefix validation")
    count = len(candidate)-2
    report = {"schema_version": VERSION, "status": "validated_prefix_only", "plan_sha256": plan_sha256,
        "source_sha256": sources, "origin_envelope_sha256": envelope_sha256,
        "origin_forecast_sha256": forecast_sha256, "origin_supervisor_sha256": supervisor_sha256,
        "expected_run_count": spec["expected_run_count"], "reusable_success_count": count,
        "missing_completed_record_count": len(keys)-count, "remaining_workloads": [dict(zip(KEYS, key)) for key in keys[count:]],
        "interrupted_supervisor": supervisor, "reference_numerical": native.numerical_summary(checked),
        "uncommitted_particle_files": orphans, "inherited_inner_completion": None,
        "inherited_final_maps_identity": None, "inherited_final_memory_summary": None,
        "numerically_qualified": False, "certified": False, "formal_training_accepted": False,
        "final_eval_label_prediction_metric_reads": 0,
        "scope": "read-only admission evidence; no resumed execution, completed workload or new resource authorization"}
    return ValidatedPrefix(spec, candidate[0], candidate[1], candidate[2:], keys[count:], report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "reference-ledger", "reference-audit", "fit", "fit-ledger", "envelope"):
        parser.add_argument("--"+name, required=True, type=Path)
    for name in ("plan-sha256", "envelope-sha256", "forecast-sha256", "supervisor-sha256"):
        parser.add_argument("--"+name, required=True)
    result = inspect(**vars(parser.parse_args()))
    print(json.dumps(result.report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
