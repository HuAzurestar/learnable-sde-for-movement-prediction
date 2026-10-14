"""Scientific contract for a missing-only, resumable particle-budget extension.

The whole closed reference remains immutable. New rows have a separate schema
and denominator; inherited reference particles are checked, never recalculated
or relabeled as newly executed work. This module alone is NOT an executor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from pathlib import Path
import re

from . import seed_resume as admission
from .brownian import integration_grid
from .inference import PRIMARY_FAMILY
from .numerical_check import audit as numerical_audit
from .precision_check import KEYS, load_particle_evidence, replay
from .resource_replay import audited, numerical_summary

native, engine = admission.native, admission.engine
VERSION = "pirc17-missing-particle-extension-v1"
PLAN_VERSION = VERSION+"-plan"
PURPOSE = "complete_primary_family_validation_missing_particle_extension"
EXTRA_DIGESTS = ("reference_envelope_sha256", "reference_supervisor_sha256")
PLAN_FIELDS = native.PLAN_KEYS | {*EXTRA_DIGESTS, "output_name"}
SOURCE_ARGUMENTS = {"plan", "plan_sha256", "reference_envelope", "reference_supervisor",
                    "reference_ledger", "reference_audit", "fit", "fit_ledger"}
UNQUALIFIED = {"certified": False, "numerically_qualified": False, "formal_training_accepted": False,
               "final_eval_label_prediction_metric_reads": 0}


def source_hashes():
    return {**admission.source_hashes(), "particle_extension.py": native._hash(Path(__file__))}


def runtime_identity():
    return {"python": engine.platform.python_version(), "numpy": engine.np.__version__,
        "torch": engine.torch.__version__, "torch_intraop_threads": engine.torch.get_num_threads(),
        "torch_interop_threads": engine.torch.get_num_interop_threads()}


def load_plan(path, digest):
    spec = admission.read_closed(path, digest)
    if (set(spec) != PLAN_FIELDS or spec["schema_version"] != PLAN_VERSION or spec["purpose"] != PURPOSE
            or any(spec[k] is not False for k in ("certified", "formal_training_accepted", "final_eval_authorized"))
            or any(not isinstance(spec[k], str) or not re.fullmatch("[0-9a-f]{64}", spec[k])
                   for k in (*native.DIGESTS, *EXTRA_DIGESTS))
            or not isinstance(spec["output_name"], str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", spec["output_name"])):
        raise ValueError("exact unsealed particle-extension plan and immutable reference hashes required")
    engine.validate_workload(**{k: spec[k] for k in native.AXES})
    expected = spec["limit_origins"]*math.prod(len(spec[k]) for k in ("configurations", "seeds", "particles", "steps"))
    if (set(spec["configurations"]) != {n for pair in PRIMARY_FAMILY.values() for n in pair}
            or len(spec["particles"]) != 1 or len(spec["steps"]) < 2
            or spec["selection_policy"] != "lexical_independent_blocks" or spec["map_backend"] != "multicell"
            or type(spec["expected_run_count"]) is not int or spec["expected_run_count"] != expected
            or type(spec["minimum_free_bytes"]) is not int or spec["minimum_free_bytes"] != engine.MINIMUM_FREE_BYTES
            or type(spec["outer_wall_seconds"]) is not int
            or not 0 < spec["wall_seconds"] < spec["outer_wall_seconds"] <= 29220 or spec["wall_seconds"] > 28800
            or type(spec["tolerance_m"]) not in (int, float) or not 0 < spec["tolerance_m"] < float("inf")):
        raise ValueError("complete family, one new budget, unchanged refinement grid and bounded resource limits required")
    return spec


def normalize(source):
    if set(source) != SOURCE_ARGUMENTS:
        raise ValueError("exact closed-reference particle-extension arguments required")
    return {k: v if k == "plan_sha256" else str(Path(v).resolve()) for k, v in source.items()}


def directory_for(source, spec):
    return Path(source["reference_ledger"]).resolve().parent/(spec["output_name"]+".extension")


def _closed_reference(spec, source, reference, report):
    envelope = admission.read_closed(source["reference_envelope"], spec["reference_envelope_sha256"], jsonl=True)
    supervisor = admission.read_closed(source["reference_supervisor"], spec["reference_supervisor_sha256"])
    if [r.get("type") for r in envelope] != ["initialization", "reference_verified", "engine_completed", "audit", "completion"]:
        raise ValueError("whole closed native reference envelope required")
    start, _, execution, audit_event, end = envelope
    versions = {"pirc17-native-primary-qualification-v1", "pirc17-native-primary-qualification-v2"}
    if (start.get("schema_version") not in versions
            or supervisor.get("schema_version") != start["schema_version"]+"-supervisor"
            or supervisor.get("plan_sha256") != start.get("plan_sha256")
            or supervisor.get("status") != "complete" or type(supervisor.get("worker_returncode")) is not int
            or supervisor["worker_returncode"] != 0 or supervisor.get("timed_out") is not False
            or supervisor.get("termination_confirmed") is not True
            or execution.get("ledger_sha256") != spec["reference_ledger_sha256"]
            or execution.get("completion") != reference[-1]
            or audit_event.get("sha256") != spec["reference_audit_sha256"]
            or audit_event.get("numerical") != numerical_summary(report)
            or end.get("status") != "complete" or end.get("resource_stopped") is not False):
        raise ValueError("reference process, whole ledger and audit closure do not agree")
    for item in (start, end, supervisor):
        if (any(item.get(k) is not False for k in ("certified", "formal_training_accepted"))
                or type(item.get("final_eval_label_prediction_metric_reads")) is not int
                or item["final_eval_label_prediction_metric_reads"] != 0):
            raise ValueError("reference evidence crosses the development-only boundary")
    counts = report["counts"]
    if (any(type(end.get(k)) is not int or end[k] != v for k, v in counts.items())
            or end.get("engine_terminal_error_count") != 0):
        raise ValueError("whole reference counts/failures differ from supervisory closure")
    original_plan = start["plan"]
    if (any(original_plan[k] != reference[0][k] for k in native.AXES if k not in {"particles", "steps"})
            or original_plan["particles"] != reference[1]["particle_counts"]
            or original_plan["steps"] != reference[1]["max_steps_seconds"]
            or original_plan["tolerance_m"] != spec["tolerance_m"]
            or any(original_plan[k] != spec[k] for k in
                   ("fit_sha256", "fit_ledger_sha256", "training_policy_sha256", "eligibility_sha256"))
            or supervisor["outer_wall_seconds"] != original_plan["outer_wall_seconds"]
            or spec["wall_seconds"] > reference[0]["wall_seconds"]
            or spec["outer_wall_seconds"] > supervisor["outer_wall_seconds"]):
        raise ValueError("reference plan/model bindings differ or extension raises its registered caps")
    begun, deadline, finished = (datetime.fromisoformat(supervisor[k]) for k in ("started_at", "deadline_at", "finished_at"))
    elapsed = supervisor["elapsed_seconds"]
    if (any(t.tzinfo is None for t in (begun, deadline, finished)) or not begun <= finished <= deadline
            or deadline != begun+timedelta(seconds=supervisor["outer_wall_seconds"])
            or type(elapsed) not in (int, float) or not 0 <= elapsed <= supervisor["outer_wall_seconds"]
            or abs((finished-begun).total_seconds()-elapsed) > 2):
        raise ValueError("reference is not a bounded terminal execution")
    return envelope, supervisor


@dataclass
class Prepared:
    source: dict
    spec: dict
    reference: list
    reference_report: dict
    scientific_header: dict
    bound: dict
    sources: dict

    @property
    def keys(self):
        return admission.workload_keys(self.spec, self.scientific_header["sample_ids"])

    def recheck(self, guard=lambda: None):
        guard()
        if source_hashes() != self.sources or any(native._hash(p) != sha for p, sha in self.bound.items()):
            raise ValueError("particle-extension source or whole-reference evidence changed")


def prepare(*, source, guard=lambda: None):
    """No new forecast, raw-data access, file publication or acceptance."""
    guard()
    source = normalize(source)
    sources = source_hashes()
    spec = load_plan(source["plan"], source["plan_sha256"])
    evidence = {"fit": source["fit"], "fit_ledger": source["fit_ledger"], **{k: spec[k] for k in
                ("fit_sha256", "fit_ledger_sha256", "training_policy_sha256")}}
    reference = admission.read_closed(source["reference_ledger"], spec["reference_ledger_sha256"], jsonl=True)
    report = admission.read_closed(source["reference_audit"], spec["reference_audit_sha256"])
    if (report.get("status") != "complete" or not report.get("numerical_audit")
            or report["numerical_audit"]["tolerance_m"] != spec["tolerance_m"]
            or audited(reference, Path(source["reference_ledger"]).parent,
                       spec["reference_ledger_sha256"], spec["tolerance_m"], evidence) != report):
        raise ValueError("complete original audit must reproduce at its unchanged tolerance")
    _closed_reference(spec, source, reference, report)
    init, header = reference[:2]
    if (len(header["particle_counts"]) < 2 or len(header["max_steps_seconds"]) < 2
            or spec["particles"][0] <= max(header["particle_counts"])
            or any(spec[k] != init[k] for k in ("configurations", "seeds", "limit_origins", "selection_policy", "map_backend"))
            or spec["steps"] != header["max_steps_seconds"] or init["runtime"] != runtime_identity()):
        raise ValueError("missing-only extension must retain the entire reference family, origins, seeds, grid and runtime")
    scientific_header = dict(deepcopy(header), particle_counts=spec["particles"], expected_run_count=spec["expected_run_count"])
    bound = {Path(source[k]): spec[k+"_sha256"] for k in SOURCE_ARGUMENTS-{ "plan", "plan_sha256"}}
    bound[Path(source["plan"])] = source["plan_sha256"]
    _, modules = engine.resolve_map_backend(spec["map_backend"])
    bound.update({Path(m.__file__): header["map_source_sha256"][m.__name__] for m in modules})
    for row in reference[2:-1]:
        guard()
        path = Path(source["reference_ledger"]).parent/row["particle_artifact"]["path"]
        bound[path] = row["particle_artifact"]["sha256"]
    prepared = Prepared(source, spec, reference, report, scientific_header, bound, sources)
    if len(prepared.keys) != spec["expected_run_count"]:
        raise ValueError("particle-extension denominator differs from complete reference origins")
    prepared.recheck(guard)
    return prepared


def validate_rows(prepared, rows, directories, *, offset=0, guard=lambda: None):
    """Replay an exact committed slice; closure and ancestry are separate gates."""
    spec, header, keys = prepared.spec, prepared.scientific_header, prepared.keys
    if type(offset) is not int or offset < 0 or offset+len(rows) > len(keys) or len(rows) != len(directories):
        raise ValueError("exact ordered bounded particle-extension slice required")
    old = {tuple(row[k] for k in KEYS): row for row in prepared.reference[2:-1]}
    reference_dir = Path(prepared.source["reference_ledger"]).parent
    drivers = {}
    covered = set()
    for index, (row, directory) in enumerate(zip(rows, directories), start=offset):
        guard()
        if (set(row) != admission.RUN_FIELDS or row.get("type") != "run" or row.get("status") != "success"
                or tuple(row.get(k) for k in KEYS) != keys[index]
                or type(row.get("seed")) is not int or type(row.get("particles")) is not int
                or isinstance(row.get("max_step_seconds"), bool)
                or row["model_identity_sha256"] != header["model_identities"][row["configuration"]][str(row["seed"])]):
            raise ValueError("mixed, failed, duplicate or reordered extension row")
        arrays = load_particle_evidence(row, directory)
        stream = row["sample_id"], row["seed"]
        if stream not in drivers:
            driver = engine.BrownianPath(arrays[2], spec["steps"], history_step_seconds=5.,
                particles=spec["particles"][0], seed=row["seed"], stream_id=row["sample_id"])
            drivers[stream] = driver.identity
            del driver
        if row["brownian_identity"] != drivers[stream]:
            raise ValueError("extension Brownian identity does not reproduce on the full reference grid")
        for count in prepared.reference[1]["particle_counts"]:
            original_key = tuple(count if k == "particles" else row[k] for k in KEYS)
            original = old[original_key]
            original_arrays = load_particle_evidence(original, reference_dir)
            if (row["independent_block_id"] != original["independent_block_id"]
                    or row["actual_horizons_seconds"] != original["actual_horizons_seconds"]
                    or not engine.np.array_equal(arrays[0][:count], original_arrays[0])
                    or not all(engine.np.array_equal(a, b) for a, b in zip(arrays[1:], original_arrays[1:]))):
                raise ValueError("extension changes a saved reference particle prefix, block, target or time")
            covered.add(original_key)
        if (row["scores"] != engine.score_path(*arrays, time_weights=engine.TIME_WEIGHTS, entropy_grid=admission.entropy_grid())
                or row["particle_precision"] != engine.energy_precision(arrays[0], arrays[1], engine.TIME_WEIGHTS)):
            raise ValueError("extension scores or conditional particle precision do not exactly reproduce")
        intervals = len(integration_grid(arrays[2], row["max_step_seconds"], 5.))-1
        wall = row["rollout_wall_seconds_including_lazy_map_initialization"]
        if (type(row["feature_query_rows"]) is not int or row["feature_query_rows"] != row["particles"]*intervals
                or type(row["invalid_feature_rows"]) is not int or not 0 <= row["invalid_feature_rows"] <= row["feature_query_rows"]
                or type(wall) not in (int, float) or not 0 <= wall < float("inf")):
            raise ValueError("extension feature counters or measured rollout time differ")
    prepared.recheck(guard)
    return covered


def whole_audit(prepared, rows, directory, *, guard=lambda: None):
    """Audit only after the complete new workload; no legacy ledger is written."""
    if len(rows) != len(prepared.keys):
        raise ValueError("whole extension requires every registered new workload")
    covered = validate_rows(prepared, rows, [directory]*len(rows), guard=guard)
    old_rows = prepared.reference[2:-1]
    if len(covered) != len(old_rows):
        raise ValueError("whole extension must cover every original workload")
    # Generic math sees the new full grid only. The terminal counts express a
    # scientific denominator, never a fictitious uninterrupted engine attempt.
    generic = [dict(prepared.scientific_header, schema_version="pirc17-development-rollout-v1",
        purpose="bounded_validation_engineering_pilot"), *rows,
        {"type": "completion", "attempted_run_count": len(rows), "success_count": len(rows), "failure_count": 0}]
    old = {tuple(row[k] for k in KEYS): row for row in old_rows}
    prior = max(prepared.reference[1]["particle_counts"])
    comparisons = []
    def vector(row):
        return engine.np.array([row["scores"]["time_weighted_energy_score_m"],
            *[v["energy_score_m"] for v in row["scores"]["by_time"]]])
    for row in rows:
        previous = old[tuple(prior if k == "particles" else row[k] for k in KEYS)]
        a, b = vector(previous), vector(row)
        delta = b-a
        comparisons.append({"axis": "particle_count", "sample_id": row["sample_id"],
            "configuration": row["configuration"], "seed": row["seed"],
            "fixed_setting": {"step_seconds": row["max_step_seconds"]}, "settings": [prior, row["particles"]],
            "weighted_ES_m": [float(a[0]), float(b[0])], "second_minus_first_ES_m": float(delta[0]),
            "second_minus_first_by_time_ES_m": delta[1:].tolist(),
            "all_scoring_times_within_tolerance": bool(engine.np.max(engine.np.abs(delta)) <= prepared.spec["tolerance_m"])})
    result = {"schema_version": VERSION+"-whole-science", "new_run_count": len(rows),
        "reference_runs_covered": len(covered), "exact_reference_prefix_checks": len(covered),
        "reference_numerical": numerical_summary(prepared.reference_report),
        "new_numerical": numerical_audit(generic, tolerance_m=prepared.spec["tolerance_m"]),
        "new_particle_precision": replay(generic, directory, tolerance_m=prepared.spec["tolerance_m"]),
        "cross_source_adjacent_particle_sensitivities": comparisons,
        "tolerance_m": prepared.spec["tolerance_m"], **UNQUALIFIED,
        "scope": "complete original/new numerical evidence only; process closure and resource ancestry are separate gates"}
    for row in rows:
        guard()
        load_particle_evidence(row, directory)
    prepared.recheck(guard)
    return result
