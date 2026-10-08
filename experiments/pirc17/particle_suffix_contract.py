"""Closed-prefix admission and whole science for particle-suffix computation.

The immutable missing-particle profile is an explicitly supported parent, not
a partial ledger to salvage. This module performs no prediction or publication
and provides no process-ownership, interruption or launch certificate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from . import particle_extension_session as parent_session
from .brownian import integration_grid
from .nested_precision import nested_energy_precision
from .precision_check import KEYS, aggregate_precision, load_particle_evidence, replay

lineage = parent_session.lineage
science = parent_session.execution.science
native, engine, admission = science.native, science.engine, science.admission
VERSION = "pirc17-particle-suffix-extension-v1"
PLAN_VERSION = VERSION + "-plan"
PURPOSE = "complete_primary_family_validation_particle_suffix"
PARENT_DIGESTS = {"parent_root_sha256", "parent_closure_sha256", "parent_whole_science_sha256"}
PLAN_FIELDS = science.PLAN_FIELDS | PARENT_DIGESTS | {"prefix_particles"}
SOURCE_FIELDS = {"plan", "plan_sha256", "parent_directory"}
MEASURED_FIELDS = {"feature_query_rows", "invalid_feature_rows",
                  "rollout_wall_seconds_including_lazy_map_initialization"}
SCIENTIFIC_FIELDS = admission.RUN_FIELDS - MEASURED_FIELDS
RUN_FIELDS = SCIENTIFIC_FIELDS | {"schema_version", "prefix_binding", "suffix_accounting"}
ACCOUNT_FIELDS = {"particle_start", "particle_stop", "computed_particles", "new_feature_query_rows",
    "new_invalid_feature_rows", "new_raw_map_query_rows", "inherited_feature_query_rows",
    "inherited_invalid_feature_rows", "ensemble_feature_rows", "ensemble_invalid_rows",
    "new_rollout_wall_seconds_including_lazy_map_initialization"}
UNQUALIFIED = {**science.UNQUALIFIED, "resource_certified": False, "production_resume_ready": False}


def source_hashes():
    return {**lineage.source_hashes(), **{name: native._hash(Path(__file__).with_name(name))
            for name in ("particle_suffix.py", "particle_suffix_contract.py")}}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def normalize(source):
    if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
        raise ValueError("exact suffix plan/hash and closed-parent directory required")
    return {k: v if k == "plan_sha256" else str(Path(v).resolve()) for k, v in source.items()}


def load_plan(path, digest):
    spec = admission.read_closed(path, digest)
    if (set(spec) != PLAN_FIELDS or spec["schema_version"] != PLAN_VERSION or spec["purpose"] != PURPOSE
            or any(spec[k] is not False for k in ("certified", "formal_training_accepted", "final_eval_authorized"))
            or any(not isinstance(spec[k], str) or not re.fullmatch("[0-9a-f]{64}", spec[k])
                   for k in {*native.DIGESTS, *science.EXTRA_DIGESTS, *PARENT_DIGESTS})
            or not isinstance(spec["output_name"], str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", spec["output_name"])):
        raise ValueError("exact unsealed suffix plan and immutable parent hashes required")
    engine.validate_workload(**{k: spec[k] for k in native.AXES})
    if (type(spec["prefix_particles"]) is not int or spec["prefix_particles"] < 2
            or len(spec["particles"]) != 1 or type(spec["particles"][0]) is not int
            or spec["particles"][0] <= spec["prefix_particles"]
            or spec["particles"][0] % spec["prefix_particles"]
            or type(spec["expected_run_count"]) is not int
            or type(spec["minimum_free_bytes"]) is not int
            or type(spec["outer_wall_seconds"]) is not int
            or type(spec["wall_seconds"]) not in (int, float)
            or not 0 < spec["wall_seconds"] < spec["outer_wall_seconds"] <= 29220
            or spec["wall_seconds"] > 28800):
        raise ValueError("one larger integer-ratio ensemble and bounded original limits required")
    return spec


def directory_for(source, spec):
    return Path(source["parent_directory"]).resolve().parent / (spec["output_name"] + ".suffix")


@dataclass
class Prepared:
    source: dict
    spec: dict
    parent_root: dict
    parent_inspection: dict
    parent_rows: list
    parent_work: Path
    parent_science: dict
    original_reference: list
    scientific_header: dict
    bound: dict
    inventory: list
    sources: dict
    reference_resource_charges: dict

    @property
    def keys(self):
        return admission.workload_keys(self.spec, self.scientific_header["sample_ids"])

    def recheck(self, guard=lambda: None):
        guard()
        if (source_hashes() != self.sources
                or lineage.storage.inventory(Path(self.source["parent_directory"])) != self.inventory
                or any(native._hash(p) != sha for p, sha in self.bound.items())):
            raise ValueError("suffix source, closed parent, maps or reference evidence changed")


def prepare(*, source, guard=lambda: None):
    """Independently inspect the entire parent before allowing prefix use."""
    guard()
    source = normalize(source)
    spec = load_plan(source["plan"], source["plan_sha256"])
    sources = source_hashes()
    parent = Path(source["parent_directory"])
    inventory = lineage.storage.inventory(parent)
    root, closed, current, _ = lineage.history(parent, spec["parent_root_sha256"])
    if current is not None or not closed or closed[-1]["closure"]["status"] != "computed_closed":
        raise ValueError("suffix requires an entire successfully closed parent workload")
    last = closed[-1]
    if last["closure_sha256"] != spec["parent_closure_sha256"]:
        raise ValueError("suffix parent closure changed")
    fixed = science.PLAN_FIELDS - {"schema_version", "purpose", "particles", "output_name"}
    if (any(spec[k] != root["plan"][k] for k in fixed)
            or root["plan"]["particles"] != [spec["prefix_particles"]]
            or spec["output_name"] == root["plan"]["output_name"]):
        raise ValueError("suffix changes the full parent family, fit, grid, tolerance or limits")
    # The immutable consumer has its own fresh-memory guards. The future owned
    # caller must charge this whole admission and enforce its outer deadline.
    inspection = parent_session.inspect_closed(parent, spec["parent_root_sha256"])
    guard()
    if (inspection["last_closure_sha256"] != spec["parent_closure_sha256"]
            or inspection["whole_science_sha256"] != spec["parent_whole_science_sha256"]
            or inspection["expected_run_count"] != spec["expected_run_count"]):
        raise ValueError("suffix parent whole-science or complete denominator changed")
    work = last["directory"] / "work"
    rows = parent_session.execution.assembled(work, last["saved"])
    old_source = root["source"]
    reference = admission.read_closed(old_source["reference_ledger"], spec["reference_ledger_sha256"], jsonl=True)
    whole = admission.read_closed(work/"whole-science.json", spec["parent_whole_science_sha256"])
    core = admission.read_closed(work/"core-result.json", native._hash(work/"core-result.json"))
    header = dict(deepcopy(reference[1]), particle_counts=spec["particles"],
                  expected_run_count=spec["expected_run_count"])
    bound = {parent/item["path"]: item["sha256"] for item in inventory}
    bound[Path(source["plan"])] = source["plan_sha256"]
    bound.update({Path(old_source[k]): root["plan"][k+"_sha256"]
                  for k in science.SOURCE_ARGUMENTS - {"plan", "plan_sha256"}})
    bound[Path(old_source["plan"])] = old_source["plan_sha256"]
    bound[Path(root["data_locations"]["eligibility"])] = spec["eligibility_sha256"]
    for row in reference[2:-1]:
        bound[Path(old_source["reference_ledger"]).parent/row["particle_artifact"]["path"]] = row["particle_artifact"]["sha256"]
    _, modules = engine.resolve_map_backend(spec["map_backend"])
    bound.update({Path(m.__file__): header["map_source_sha256"][m.__name__] for m in modules})
    maps = core["new_attempt_maps"]
    if (not isinstance(maps, dict) or maps != reference[-1]["maps"]
            or any(not isinstance(maps.get(k), dict) for k in ("receipt_sha256", "verified_assets"))):
        raise ValueError("suffix requires complete unchanged parent map identity")
    bound.update({Path(p): sha for p, sha in maps["receipt_sha256"].items()})
    bound.update({Path(root["data_locations"]["data_root"])/p: sha for p, sha in maps["verified_assets"].items()})
    supervisor = admission.read_closed(old_source["reference_supervisor"], spec["reference_supervisor_sha256"])
    charges = {"original_native_reference_seconds": supervisor["elapsed_seconds"],
               "parent_extension_charged_seconds": inspection["total_charged_elapsed_seconds"],
               "scope": "separate already incurred reference charges; not a new uninterrupted execution"}
    prepared = Prepared(source, spec, root, inspection, rows, work, whole, reference,
                        header, bound, inventory, sources, charges)
    if len(rows) != len(prepared.keys) or len(rows) != spec["expected_run_count"]:
        raise ValueError("suffix requires every ordered parent workload")
    prepared.recheck(guard)
    return prepared


def prefix_binding(prepared, index):
    row = prepared.parent_rows[index]
    return {"root_sha256": prepared.spec["parent_root_sha256"],
            "closure_sha256": prepared.spec["parent_closure_sha256"],
            "row_sha256": _digest(row), "particle_artifact": deepcopy(row["particle_artifact"])}


def _accounting(prepared, row, original, intervals):
    account = row["suffix_accounting"]
    prefix, total = prepared.spec["prefix_particles"], row["particles"]
    expected = {"particle_start": prefix, "particle_stop": total, "computed_particles": total-prefix,
        "new_feature_query_rows": (total-prefix)*intervals,
        "new_raw_map_query_rows": 0 if row["configuration"] == "base" else (total-prefix)*intervals,
        "inherited_feature_query_rows": original["feature_query_rows"],
        "inherited_invalid_feature_rows": original["invalid_feature_rows"], "ensemble_feature_rows": total*intervals}
    wall_key = "new_rollout_wall_seconds_including_lazy_map_initialization"
    if (set(account) != ACCOUNT_FIELDS
            or any(type(account[k]) is not int or account[k] < 0 for k in ACCOUNT_FIELDS - {wall_key})
            or any(account[k] != v for k, v in expected.items())
            or not 0 <= account["new_invalid_feature_rows"] <= account["new_feature_query_rows"]
            or account["ensemble_invalid_rows"] != account["new_invalid_feature_rows"] + original["invalid_feature_rows"]
            or type(account[wall_key]) not in (int, float) or not 0 <= account[wall_key] < float("inf")):
        raise ValueError("suffix new/inherited particle, query, missing-row or time accounting changed")


def validate_rows(prepared, rows, directories, *, offset=0, guard=lambda: None):
    """Check a committed ordered slice; not a completion or process claim."""
    if (type(offset) is not int or offset < 0 or offset+len(rows) > len(prepared.keys)
            or len(rows) != len(directories)):
        raise ValueError("exact ordered bounded suffix slice required")
    prepared.recheck(guard)
    identities = {}
    for index, (row, directory) in enumerate(zip(rows, directories), start=offset):
        guard()
        original = prepared.parent_rows[index]
        if (set(row) != RUN_FIELDS or row["type"] != "run" or row["status"] != "success"
                or row["schema_version"] != VERSION+"-run"
                or tuple(row[k] for k in KEYS) != prepared.keys[index]
                or type(row["particles"]) is not int or type(row["seed"]) is not int
                or isinstance(row["max_step_seconds"], bool)
                or row["model_identity_sha256"] != original["model_identity_sha256"]
                or row["prefix_binding"] != prefix_binding(prepared, index)):
            raise ValueError("suffix row changes ordered workload, model or exact parent binding")
        arrays = load_particle_evidence(row, directory)
        previous = load_particle_evidence(original, prepared.parent_work)
        prefix = prepared.spec["prefix_particles"]
        if (row["independent_block_id"] != original["independent_block_id"]
                or row["actual_horizons_seconds"] != original["actual_horizons_seconds"]
                or not engine.np.array_equal(arrays[0][:prefix], previous[0])
                or not all(engine.np.array_equal(a, b) for a, b in zip(arrays[1:], previous[1:]))):
            raise ValueError("suffix changed a saved parent particle, target, time or independent block")
        stream = row["sample_id"], row["seed"]
        if stream not in identities:
            driver = engine.BrownianPath(arrays[2], prepared.spec["steps"], history_step_seconds=5.,
                particles=row["particles"], seed=row["seed"], stream_id=row["sample_id"])
            parent_identity = dict(driver.identity, max_particles=prefix,
                path_sha256=hashlib.sha256(driver.values[:, :prefix].astype('<f8').tobytes()).hexdigest())
            if parent_identity != original["brownian_identity"]:
                raise ValueError("suffix full Brownian driver does not retain the exact parent prefix")
            identities[stream] = driver.identity
            del driver
        if row["brownian_identity"] != identities[stream]:
            raise ValueError("suffix complete Brownian identity does not reproduce")
        if (row["scores"] != engine.score_path(*arrays, time_weights=engine.TIME_WEIGHTS, entropy_grid=admission.entropy_grid())
                or row["particle_precision"] != engine.energy_precision(arrays[0], arrays[1], engine.TIME_WEIGHTS)):
            raise ValueError("suffix whole-ensemble scores or particle precision do not reproduce")
        intervals = len(integration_grid(arrays[2], row["max_step_seconds"], 5.))-1
        _accounting(prepared, row, original, intervals)
    for row, directory in zip(rows, directories):
        guard()
        load_particle_evidence(row, directory)
    prepared.recheck(guard)
    return len(rows)


def whole_audit(prepared, rows, directory, *, guard=lambda: None):
    """Complete old/new science with separate execution counters and costs."""
    if len(rows) != len(prepared.keys):
        raise ValueError("whole suffix audit requires every registered workload")
    rows = deepcopy(rows)
    validate_rows(prepared, rows, [directory]*len(rows), guard=guard)
    # The existing pure math consumes only scientific fields, in memory. No
    # legacy ledger or fictitious full-ensemble fresh-query/wall count is made.
    generic = [dict(prepared.scientific_header, schema_version="pirc17-development-rollout-v1",
        purpose="bounded_validation_engineering_pilot"),
        *[{k: row[k] for k in SCIENTIFIC_FIELDS} for row in rows],
        {"type": "completion", "attempted_run_count": len(rows), "success_count": len(rows), "failure_count": 0}]
    comparisons, pairs = [], []
    for row, original in zip(rows, prepared.parent_rows):
        guard()
        def vector(value):
            return engine.np.array([value["scores"]["time_weighted_energy_score_m"],
                *[s["energy_score_m"] for s in value["scores"]["by_time"]]])
        a, b = vector(original), vector(row)
        delta = b-a
        comparisons.append({"axis": "particle_count", "sample_id": row["sample_id"],
            "configuration": row["configuration"], "seed": row["seed"],
            "fixed_setting": {"step_seconds": row["max_step_seconds"]},
            "settings": [original["particles"], row["particles"]], "weighted_ES_m": [float(a[0]), float(b[0])],
            "second_minus_first_ES_m": float(delta[0]), "second_minus_first_by_time_ES_m": delta[1:].tolist(),
            "all_scoring_times_within_tolerance": bool(engine.np.max(engine.np.abs(delta)) <= prepared.spec["tolerance_m"])})
        large, targets, _ = load_particle_evidence(row, directory)
        small, _, _ = load_particle_evidence(original, prepared.parent_work)
        pairs.append({"kind": "particle_budget_refinement", "comparison": row["configuration"],
            "candidate_workload": {k: row[k] for k in KEYS}, "control_workload": {k: original[k] for k in KEYS},
            "independent_block_id": row["independent_block_id"], "sign_convention": "candidate_ES_minus_control_ES",
            "precision": nested_energy_precision(large, small, targets, engine.TIME_WEIGHTS)})
    result = {"schema_version": VERSION+"-whole-science", "new_run_count": len(rows),
        "parent_runs_covered": len(prepared.parent_rows), "exact_parent_prefix_checks": len(rows),
        "reference_runs_covered": prepared.parent_inspection["reference_runs_covered"],
        "parent_whole_science_sha256": prepared.spec["parent_whole_science_sha256"],
        "original_numerical": deepcopy(prepared.parent_science["reference_numerical"]),
        "parent_numerical": deepcopy(prepared.parent_science["new_numerical"]),
        "parent_particle_sensitivities": deepcopy(prepared.parent_science["cross_source_adjacent_particle_sensitivities"]),
        "new_numerical": science.numerical_audit(generic, tolerance_m=prepared.spec["tolerance_m"]),
        "new_particle_precision": replay(generic, directory, tolerance_m=prepared.spec["tolerance_m"]),
        "cross_parent_particle_sensitivities": comparisons, "cross_parent_particle_precision": pairs,
        "cross_parent_aggregate_precision": aggregate_precision(pairs, prepared.scientific_header),
        "execution_accounting": {k: (math.fsum(row["suffix_accounting"][k] for row in rows) if "seconds" in k
            else sum(row["suffix_accounting"][k] for row in rows)) for k in ACCOUNT_FIELDS
            if k not in {"particle_start", "particle_stop", "computed_particles"}},
        "reference_resource_charges": deepcopy(prepared.reference_resource_charges),
        "tolerance_m": prepared.spec["tolerance_m"], **UNQUALIFIED,
        "scope": "whole nested-particle science; current owned closure and full-resource qualification are separate gates"}
    for row in rows:
        guard()
        load_particle_evidence(row, directory)
    prepared.recheck(guard)
    return result
