"""Whole-source power planning with explicitly bound, closed resumptions.

This entry accepts complete original candidate evidence and complete ordered
resume histories. It never joins partial attempts or writes a legacy ledger.
Scientific qualification and missing historical resource evidence stay separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from . import candidate_power as candidate
from . import seed_resume_lineage as lineage
from . import seed_resume_session as session
from .direct_linear_evidence import add_evidence_arguments, load_candidate

VERSION = "pirc17-resumed-candidate-primary-paired-power-v1"
FIT_DIGESTS = ("fit_sha256", "fit_ledger_sha256", "training_policy_sha256")


def source_hashes():
    return {**candidate.source_hashes(), **lineage.source_hashes(),
            "resumed_candidate_power.py": candidate._hash(Path(__file__))}


@dataclass
class _WholeSource:
    initialization: dict
    header: dict
    rows: list
    completion: dict
    counts: dict
    numerical: dict
    provenance: dict
    recheck: Callable[[], None]

    def generic(self):
        # A generic mathematical denominator only, not an old-engine completion.
        return [dict(self.header, schema_version="pirc17-development-rollout-v1",
                     purpose="bounded_validation_engineering_pilot"), *self.rows, self.completion]


def _numerical(report):
    return {**candidate.numerical_summary(report), "numerically_qualified": False,
            "available_sensitivity_axes": sorted({row["axis"] for row in report["numerical_audit"]["sensitivities"]})}


def _complete(path, digest, audit_path, audit_digest, evidence, tolerance):
    path, audit_path = Path(path).resolve(), Path(audit_path).resolve()
    rows = lineage.admission.read_closed(path, digest, jsonl=True)
    original = lineage.admission.read_closed(audit_path, audit_digest)
    if (original.get("status") != "complete" or original.get("ledger_sha256") != digest
            or not original.get("numerical_audit")):
        raise ValueError("whole complete candidate audit required; no partial-source salvage")
    if original["numerical_audit"]["tolerance_m"] != tolerance:
        raise ValueError("planning margin/4 must preserve every original numerical tolerance")
    checked = candidate.audited(rows, path.parent, digest, tolerance, evidence)
    if checked != original:
        raise ValueError("whole bound candidate audit does not reproduce")

    def recheck():
        lineage.admission.read_closed(path, digest, jsonl=True)
        lineage.admission.read_closed(audit_path, audit_digest)
        for row in rows[2:-1]:
            candidate.load_particle_evidence(row, path.parent)

    return _WholeSource(rows[0], rows[1], rows[2:-1], rows[-1], checked["counts"], _numerical(checked),
        {"kind": "complete_candidate", "ledger": str(path), "ledger_sha256": digest,
         "audit": str(audit_path), "audit_sha256": audit_digest}, recheck)


def _resumed(directory, root_digest, closure_digest, evidence, tolerance):
    directory = Path(directory).resolve()
    before = lineage.inventory(directory)
    inspection = session.inspect_closed(directory, root_digest)
    if inspection["last_closure_sha256"] != closure_digest:
        raise ValueError("explicit final closure digest must bind the whole resumed source")
    root, closed, _, _ = lineage.history(directory, root_digest)
    if (not closed or closed[-1]["closure_sha256"] != closure_digest
            or closed[-1]["closure"]["status"] != "computed_closed"):
        raise ValueError("resumed source closure changed after inspection")
    if any(root["plan"][key] != evidence[key] for key in FIT_DIGESTS):
        raise ValueError("resumed source uses a different whole candidate fit or training policy")
    if root["plan"]["tolerance_m"] != tolerance:
        raise ValueError("planning margin/4 must preserve every original numerical tolerance")
    last = closed[-1]
    work, saved = last["directory"]/"work", last["saved"]
    inherited_path = work/"inherited.json"
    inherited = lineage.admission.read_closed(inherited_path, candidate._hash(inherited_path))
    rows = deepcopy(inherited["rows"])
    for record in saved.records:
        row = deepcopy(record["row"])
        row["particle_artifact"]["path"] = "journal/"+row["particle_artifact"]["path"]
        rows.append(row)
    whole = lineage.admission.read_closed(work/"whole-math.json", inspection["whole_math_sha256"])
    count = inspection["expected_run_count"]
    if (len(rows) != count or count != root["plan"]["expected_run_count"]
            or any(row["status"] != "success" for row in rows)
            or before != lineage.inventory(directory)):
        raise ValueError("resumed source changed its complete successful denominator")

    def recheck():
        # Replays original reference/prefix, all ancestors, maps and all arrays,
        # not just the final materialization or the root directory's hashes.
        if (session.inspect_closed(directory, root_digest) != inspection
                or before != lineage.inventory(directory)):
            raise ValueError("resumed evidence changed during planning")

    counts = {"expected_run_count": count, "attempted_run_count": count,
              "success_count": count, "failure_count": 0, "unattempted_run_count": 0}
    completion = {"type": "completion", "attempted_run_count": count, "success_count": count, "failure_count": 0}
    return _WholeSource(root["initialization"], root["header"], rows, completion, counts, _numerical(whole),
        {"kind": "closed_resumption", "directory": str(directory), "root_sha256": root_digest,
         "last_closure_sha256": closure_digest, "plan_sha256": root["source"]["plan_sha256"],
         "reference_ledger_sha256": root["plan"]["reference_ledger_sha256"],
         "reference_audit_sha256": root["plan"]["reference_audit_sha256"],
         "original_interrupted_source": root["source"], "inspection": inspection}, recheck)


def run(*, ledgers, ledger_sha256, audits, audit_sha256, resume_directories, resume_root_sha256,
        resume_closure_sha256, fit, fit_sha256, fit_ledger, fit_ledger_sha256, training_policy_sha256,
        step_seconds, particles, delta_m, planning_block_counts, output):
    """Read-only whole-evidence admission, then unchanged equal-block planning."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite resumed candidate power evidence")
    if not len(ledgers) == len(ledger_sha256) == len(audits) == len(audit_sha256):
        raise ValueError("one complete candidate ledger/audit and both hashes per source required")
    if (not resume_directories or not len(resume_directories) == len(resume_root_sha256) == len(resume_closure_sha256)):
        raise ValueError("each whole resumed source requires its directory, root and final closure hashes")
    if (type(delta_m) not in (int, float) or not np.isfinite(delta_m) or delta_m <= 0
            or type(step_seconds) not in (int, float) or not np.isfinite(step_seconds) or step_seconds <= 0
            or type(particles) is not int or particles < 3 or not planning_block_counts
            or any(type(n) is not int or n < 2 for n in planning_block_counts)
            or len(set(planning_block_counts)) != len(planning_block_counts)):
        raise ValueError("explicit finite planning margin, numerical settings and unique block scenarios required")
    if any(output.is_relative_to(Path(directory).resolve()) for directory in resume_directories):
        raise ValueError("planning output must not mutate an immutable resume directory")
    sources = source_hashes()
    evidence = dict(fit=Path(fit).resolve(), fit_sha256=fit_sha256, fit_ledger=Path(fit_ledger).resolve(),
                    fit_ledger_sha256=fit_ledger_sha256, training_policy_sha256=training_policy_sha256)
    load_candidate(**evidence)
    whole_sources = [_complete(*args, evidence, delta_m/4)
        for args in zip(ledgers, ledger_sha256, audits, audit_sha256)]
    whole_sources.extend(_resumed(*args, evidence, delta_m/4)
        for args in zip(resume_directories, resume_root_sha256, resume_closure_sha256))
    required = {name for pair in candidate.PRIMARY_FAMILY.values() for name in pair}
    identity, seen, seeds = None, set(), set()
    for source in whole_sources:
        if not required <= set(source.header["configurations"]):
            raise ValueError("every source must retain all five primary model comparisons")
        current = (candidate._without(source.initialization,
            candidate.WORKLOAD_FIELDS | {"started_at", "wall_seconds", "limit_origins"}),
            candidate._without(source.header, candidate.WORKLOAD_FIELDS |
                {"sample_ids", "selected_independent_block_count", "input_validation_seconds"}))
        if identity is not None and current != identity:
            raise ValueError("whole planning sources mix candidate, runtime, map, source or selection identities")
        identity = current
        for row in source.rows:
            key = tuple(row[name] for name in candidate.KEYS)
            if key in seen:
                raise ValueError("overlapping workload across whole sources; no original/resumed double counting")
            seen.add(key)
        seeds.update(source.header["seeds"])
    if seeds != set(candidate.SEEDS):
        raise ValueError("whole planning sources require all five fixed forecast/model seed labels")
    planning = candidate.calibrate([source.generic() for source in whole_sources], step_seconds=step_seconds,
        particles=particles, delta_m=delta_m, planning_block_counts=planning_block_counts,
        comparisons=tuple(candidate.PRIMARY_FAMILY), family_size=len(candidate.PRIMARY_FAMILY))
    result = {"schema_version": VERSION, "status": "complete", **lineage.UNQUALIFIED,
        "numerically_qualified": False, "resource_certified": False, "source_sha256": sources,
        "candidate_binding": {key: evidence[key] for key in FIT_DIGESTS},
        "input_sources": [source.provenance for source in whole_sources],
        "source_counts": [source.counts for source in whole_sources],
        "source_numerical": [source.numerical for source in whole_sources],
        "parameters": {"step_seconds": step_seconds, "particles": particles, "delta_m": delta_m,
            "planning_block_counts": list(planning_block_counts), "family_size": len(candidate.PRIMARY_FAMILY),
            "target_power": .8}, "planning": planning,
        "adapter": "whole audited candidate and closed resumed scientific sources -> in-memory generic planning; no legacy ledger written",
        "seed_interpretation": "five forecast/model labels, not independent blocks or proof of distinct fitted coefficients",
        "scope": "development SD and prospective normal-planning scenarios only; no bootstrap power, convergence, efficacy or protocol acceptance"}
    for source in whole_sources:
        source.recheck()
    if (source_hashes() != sources or candidate._hash(evidence["fit"]) != fit_sha256
            or candidate._hash(evidence["fit_ledger"]) != fit_ledger_sha256):
        raise ValueError("planning implementation or bound fit evidence changed during analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    lineage.journal._publish_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_arguments(parser)
    for singular, plural in (("ledger", "ledgers"), ("audit", "audits")):
        parser.add_argument("--"+singular, dest=plural, type=Path, action="append", default=[])
        parser.add_argument("--"+singular+"-sha256", action="append", default=[])
    parser.add_argument("--resume-directory", dest="resume_directories", type=Path, action="append", required=True)
    for name in ("root", "closure"):
        parser.add_argument("--resume-"+name+"-sha256", action="append", required=True)
    parser.add_argument("--step-seconds", type=float, required=True)
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--delta-m", type=float, required=True)
    parser.add_argument("--planning-blocks", dest="planning_block_counts", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "source_independent_blocks": result["planning"]["source_independent_blocks"],
        "source_origin_count": result["planning"]["source_origin_count"], "input_source_count": len(result["input_sources"]),
        "numerically_qualified": False, "resource_certified": False, "certified": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
