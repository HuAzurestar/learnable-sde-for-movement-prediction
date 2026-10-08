"""Complete, model-bound replay using a fresh Windows RAM observer.

The immutable candidate engine already accepts an available-memory callable.
This entry uses that seam without changing its source or ledger schema. A
separate versioned envelope binds the observer and both audited executions.
Every reference workload is replayed; numerical failures remain failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from . import direct_linear_check as checker
from . import direct_linear_rollout as engine
from . import physical_memory
from .direct_linear_evidence import add_evidence_arguments, read_bound
from .precision_check import KEYS, load_particle_evidence
from .qualification import _hash

VERSION = "pirc17-native-resource-replay-v1"
AUDIT_SOURCES = ("direct_linear_check.py", "direct_linear_evidence.py", "numerical_check.py",
                 "precision_check.py", "precision.py", "nested_precision.py")


def source_hashes():
    return {name: _hash(Path(__file__).with_name(name)) for name in
            ("resource_replay.py", "physical_memory.py", *AUDIT_SOURCES)}


class MemoryObserver:
    """Fresh observation on every call, plus bounded aggregate instrumentation."""

    def __init__(self):
        self.calls = self.failures = 0
        self.elapsed_seconds = 0.
        self.minimum_available_bytes = None

    def __call__(self):
        self.calls += 1
        begin = time.perf_counter()
        try:
            value = physical_memory.available_physical_bytes()
            if type(value) is not int or value < 0:
                raise ValueError("invalid available physical bytes")
        except Exception as exc:
            self.failures += 1
            # The engine stops its entire remaining workload on MemoryError.
            # Unknown RAM must not permit subsequent forecasts after a failed
            # native API read. No availability value is invented or cached.
            raise MemoryError(f"available RAM is unknown; {type(exc).__name__}: {str(exc)[:160]}") from exc
        finally:
            self.elapsed_seconds += time.perf_counter()-begin
        self.minimum_available_bytes = value if self.minimum_available_bytes is None else min(
            self.minimum_available_bytes, value)
        return value

    def guard(self):
        if self() < engine.MINIMUM_FREE_BYTES:
            raise MemoryError("resource replay requires at least 2 GiB available RAM")

    def summary(self):
        return {"calls": self.calls, "failed_calls": self.failures,
                "observation_wall_seconds": self.elapsed_seconds,
                "minimum_observed_available_bytes": self.minimum_available_bytes,
                "measurement_cache": False}


def audited(rows, directory, ledger_sha256, tolerance_m, evidence):
    result = checker.check(rows, directory, tolerance_m=tolerance_m, **evidence)
    result["ledger_sha256"] = ledger_sha256
    result["audit_source_sha256"] = {name: _hash(Path(__file__).with_name(name)) for name in AUDIT_SOURCES}
    return result


def rollout_seconds(rows):
    values = [row["rollout_wall_seconds_including_lazy_map_initialization"] for row in rows[2:-1]]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("finite nonnegative measured rollout wall times required")
    return sum(values)


def compare(reference, candidate, reference_directory, candidate_directory):
    """Strict whole-workload comparison after both independent ledger audits."""
    def without(record, ignored):
        return {key: value for key, value in record.items() if key not in ignored}

    if (without(reference[0], {"started_at"}) != without(candidate[0], {"started_at"})
            or without(reference[1], {"input_validation_seconds"}) !=
               without(candidate[1], {"input_validation_seconds"})
            or without(reference[-1], {"elapsed_seconds"}) != without(candidate[-1], {"elapsed_seconds"})):
        raise ValueError("replay changed workload, runtime, models, source bindings, resources or map evidence")
    previous = {tuple(row[key] for key in KEYS): row for row in reference[2:-1]}
    current = {tuple(row[key] for key in KEYS): row for row in candidate[2:-1]}
    if (len(previous) != len(reference)-3 or len(current) != len(candidate)-3 or set(previous) != set(current)
            or len(previous) != reference[0]["expected_run_count"]):
        raise ValueError("complete matching replay workloads required")
    output = []
    ignored = {"rollout_wall_seconds_including_lazy_map_initialization", "particle_artifact"}
    for key, left in previous.items():
        right = current[key]
        old = load_particle_evidence(left, reference_directory)
        new = load_particle_evidence(right, candidate_directory)
        equal = [bool(np.array_equal(a, b)) for a, b in zip(old, new)]
        delta = float(np.max(np.abs(old[0]-new[0])))
        output.append({"workload": dict(zip(KEYS, key)), "positions_exact": equal[0],
            "targets_exact": equal[1], "times_exact": equal[2], "maximum_position_difference_m": delta,
            "all_other_run_fields_exact": without(left, ignored) == without(right, ignored)})
    before, after = rollout_seconds(reference), rollout_seconds(candidate)
    return {"passed": all(all(row[key] for key in ("positions_exact", "targets_exact", "times_exact",
                                                  "all_other_run_fields_exact")) for row in output),
            "compared_run_count": len(output), "runs": output,
            "reference_rollout_wall_seconds": before, "replay_rollout_wall_seconds": after,
            "observed_reference_over_replay_wall_ratio": before/after if after > 0 else None,
            "runtime_scope": "one complete sequential replay pair, not a general speed benchmark",
            "scope": "prediction equivalence only; not numerical precision, scientific efficacy or acceptance"}


def numerical_summary(report):
    numerical = report.get("numerical_audit") or {}
    sensitivities = numerical.get("sensitivities", [])
    return {"checked": len(sensitivities),
            "out_of_tolerance": sum(not row["all_scoring_times_within_tolerance"] for row in sensitivities),
            "tolerance_m": numerical.get("tolerance_m"), "certified": False}


def run(*, reference_ledger, reference_ledger_sha256, reference_audit, reference_audit_sha256,
        fit, fit_sha256, fit_ledger, fit_ledger_sha256, training_policy_sha256,
        eligibility, release, snapshot, data_root, output, progress=None):
    output = Path(output).resolve()
    forecast_path = output.with_suffix(".forecast.jsonl")
    audit_path = output.with_suffix(".audit.json")
    if output.suffix != ".jsonl":
        raise ValueError("resource replay envelope must use .jsonl")
    if any(path.exists() for path in (output, forecast_path, audit_path, forecast_path.with_suffix(".particles"))):
        raise FileExistsError("refusing to overwrite replay envelope, forecasts, audit or particles")
    reference_ledger, reference_audit = Path(reference_ledger).resolve(), Path(reference_audit).resolve()
    reference = read_bound(reference_ledger, reference_ledger_sha256, jsonl=True)
    original_audit = read_bound(reference_audit, reference_audit_sha256)
    if (not reference or reference[0].get("schema_version") != engine.VERSION
            or original_audit.get("schema_version") != checker.VERSION
            or original_audit.get("ledger_sha256") != reference_ledger_sha256
            or not original_audit.get("numerical_audit")):
        raise ValueError("bound complete candidate reference and its numerical audit required")
    init = reference[0]
    expected = init["expected_run_count"]
    if type(expected) is not int or expected < 1:
        raise ValueError("positive registered reference workload count required")
    tolerance = original_audit["numerical_audit"]["tolerance_m"]
    evidence = dict(fit=fit, fit_sha256=fit_sha256, fit_ledger=fit_ledger,
                    fit_ledger_sha256=fit_ledger_sha256, training_policy_sha256=training_policy_sha256)
    binding = {key: value for key, value in evidence.items() if key.endswith("sha256")}
    sources, observer_identity = source_hashes(), physical_memory.identity()
    engine_sources = engine.source_hashes()
    observer = MemoryObserver()
    started = time.perf_counter()
    execution, comparison, candidate_audit = None, None, None
    forecast_sha = audit_sha = None
    errors = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        def emit(row):
            target.write(json.dumps(row, allow_nan=False)+"\n")
            target.flush()

        def fail(exc):
            errors.append({"type": "failure", "error_type": type(exc).__name__, "error_message": str(exc)[:300]})
            emit(errors[-1])

        emit({"type": "initialization", "schema_version": VERSION,
            "purpose": "complete_native_memory_observer_equivalence_replay", **binding,
            "started_at": datetime.now(timezone.utc).isoformat(), "expected_run_count": expected,
            "reference_ledger_sha256": reference_ledger_sha256, "reference_audit_sha256": reference_audit_sha256,
            "reference_ledger": str(reference_ledger), "reference_audit": str(reference_audit),
            "forecast_ledger_path": forecast_path.name, "audit_path": audit_path.name,
            "observer": observer_identity, "source_sha256": sources, "engine_source_sha256": engine_sources,
            "engine_version": engine.VERSION, "engine_wall_seconds": init["wall_seconds"],
            "minimum_free_bytes": engine.MINIMUM_FREE_BYTES, "tolerance_m": tolerance,
            "resource_policy": "unchanged engine guard call sites and floor; extra fresh reads before and after audits; native failures become MemoryError and stop remaining forecasts",
            "certified": False, "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0})
        try:
            observer.guard()
            validated = audited(reference, reference_ledger.parent, reference_ledger_sha256, tolerance, evidence)
            if validated != original_audit or validated["status"] != "complete":
                raise ValueError("reference audit does not exactly reproduce the complete bound evidence")
            rollout_seconds(reference)
            emit({"type": "reference_verified", "numerical": numerical_summary(validated)})
            if progress:
                progress({"phase": "reference_verified", "expected_runs": expected})
            observer.guard()
            execution = engine.run(**evidence, eligibility=eligibility,
                eligibility_sha256=init["eligibility_sha256"], release=release, snapshot=snapshot,
                data_root=data_root, output=forecast_path, configurations=init["configurations"],
                seeds=init["seeds"], particles=init["particle_counts"], steps=init["max_steps_seconds"],
                limit_origins=init["limit_origins"], wall_seconds=init["wall_seconds"],
                selection_policy=init["selection_policy"], map_backend=init["map_backend"],
                available_memory=observer, progress=progress)
            forecast_sha = _hash(forecast_path)
            emit({"type": "engine_completed", "ledger_sha256": forecast_sha, "completion": execution})
            candidate = read_bound(forecast_path, forecast_sha, jsonl=True)
            observer.guard()
            candidate_audit = audited(candidate, forecast_path.parent, forecast_sha, tolerance, evidence)
            with audit_path.open("x", encoding="utf-8") as audit_target:
                json.dump(candidate_audit, audit_target, indent=2, allow_nan=False)
            audit_sha = _hash(audit_path)
            emit({"type": "audit", "path": audit_path.name, "sha256": audit_sha,
                  "numerical": numerical_summary(candidate_audit)})
            if candidate_audit["status"] != "complete":
                raise ValueError("replay execution is incomplete or failed; no equivalence conclusion")
            observer.guard()
            comparison = compare(reference, candidate, reference_ledger.parent, forecast_path.parent)
            # Precision reports are deterministic functions of the same saved
            # arrays; compare every field, not just pass counts or mean ES.
            if any(candidate_audit[key] != original_audit[key] for key in ("numerical_audit", "particle_precision")):
                comparison["passed"] = False
                comparison["audit_fields_exact"] = False
            else:
                comparison["audit_fields_exact"] = True
            emit({"type": "comparison", **comparison})
            if not comparison["passed"]:
                raise ValueError("full replay prediction, score or uncertainty evidence differs")
            observer.guard()
        except Exception as exc:
            fail(exc)
        finally:
            try:
                if (source_hashes() != sources or physical_memory.identity() != observer_identity
                        or engine.source_hashes() != engine_sources
                        or _hash(reference_ledger) != reference_ledger_sha256
                        or _hash(reference_audit) != reference_audit_sha256
                        or (forecast_sha is not None and _hash(forecast_path) != forecast_sha)
                        or (audit_sha is not None and _hash(audit_path) != audit_sha)):
                    raise ValueError("bound replay sources, reference or output evidence changed during execution")
            except Exception as exc:
                fail(exc)
        counts = {key: execution[key] if execution else default for key, default in
            (("attempted_run_count", 0), ("success_count", 0), ("failure_count", 0), ("unattempted_run_count", expected))}
        passed = bool(not errors and comparison and comparison["passed"] and observer.calls and not observer.failures)
        completion = {"type": "completion", "status": "complete" if passed else "failed",
            "expected_run_count": expected, **counts, "terminal_error_count": len(errors),
            "engine_terminal_error_count": execution["terminal_error_count"] if execution else 0,
            "resource_stopped": bool((execution and execution["resource_stopped"]) or
                any(row["error_type"] in {"MemoryError", "TimeoutError"} for row in errors)),
            "prediction_equivalence_passed": passed, "observer": observer.summary(),
            "candidate_numerical": numerical_summary(candidate_audit) if candidate_audit else None,
            "elapsed_seconds": time.perf_counter()-started, "certified": False,
            "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0}
        emit(completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_arguments(parser)
    for name in ("reference-ledger", "reference-audit"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    for name in ("eligibility", "release", "snapshot", "data-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    result = run(**vars(parser.parse_args()), progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps(result), flush=True)
    return 0 if result["prediction_equivalence_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
