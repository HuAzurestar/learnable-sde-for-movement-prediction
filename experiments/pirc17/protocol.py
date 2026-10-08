"""Content-addressed PIRC-17 protocol candidate; a seal is NOT authorization.

DEV03 seals the scientific/input contract and current source snapshot. DEV04
must additionally bind its new executor and full matrix in an execution seal.
ACCEPT01 must name BOTH exact identities, plus successful TEST01/REVIEW01.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.pirc22.consumer import load_benchmark_selection_binding
from .comparison_registry import ORIGIN_MODES, comparison_registry
from .configurations import terrain_configurations
from .decision_policy import POLICY_PATH, decision_policy, validate_policy
from .delivery_policy import policy as delivery_policy
from .direct_linear import training_policy
from .method_comparisons import method_comparison_registry
from .method_mechanisms import mechanism_registry
from .protocol_core import digest, envelope, file_hash, publish, read_json, relative_path, sha256, under, unpack

VERSION = "pirc17-sealed-protocol-v1"
EXECUTION_VERSION = "pirc17-formal-execution-binding-v1"
ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = "experiments/pirc17/evidence/protocol-inputs-v2.json"
INPUT_FILE_SHA256 = "e2e1228603984f95306d9b9f14301da422f6a5c4bb3d01dca58559045ba0bdde"
DECISION_FILE_SHA256 = "ae81d2fd72d80253ee40bf22b4e9cc7b584b7f879231930712ef04b55cfa405f"
SOURCE_PREFIXES = {"PSDE-SDE": ("data", "experiments/pirc17", "experiments/pirc22", "experiments/nex326"),
                   "DSDE-SDE": ("trajectory", "map_data")}
JSON_SOURCES = ("experiments/nex326/experiment.json", "experiments/nex326/pirc19_scope_policy.json",
    "experiments/pirc22/benchmark_selection.consumer.json", "experiments/pirc22/representation_matrix.json",
    "experiments/pirc22/representation_matrix.lock.json", "experiments/pirc17/metric_registry.json",
    "experiments/pirc17/plans/full-delivery-policy-v1.json",
    "experiments/pirc17/plans/pre-evaluation-decision-policy-v1.json",
    "experiments/pirc17/evidence/closed-qualification-v1.json",
    "experiments/pirc17/evidence/protocol-inputs-v1.json", INPUT_PATH)
MANAGEMENT_SOURCES = ("REQUIREMENT.md", "SOLUTION.md", "gists/ACCEPTANCE-CRITERIA.md",
    "gists/EXECUTION-CHECKLIST.md", "gists/DEV-07-exposure.md", "gists/DEV-07-qualification.md")


def repository_roots():
    return {"PSDE-SDE": ROOT, "DSDE-SDE": ROOT.parent/"DSDE-SDE", "MPA": ROOT.parent/"MPA"}


def source_catalog(roots=None):
    roots = repository_roots() if roots is None else roots
    entries = {}
    for repository, prefixes in SOURCE_PREFIXES.items():
        for prefix in prefixes:
            directory = under(roots[repository], prefix)
            if not directory.is_dir():
                raise ValueError("required source directory missing")
            for path in sorted(directory.rglob("*.py")):
                relative = path.relative_to(roots[repository]).as_posix()
                # under() rejects symlink escapes, including changed ancestors.
                entries[f"{repository}/{relative}"] = file_hash(under(roots[repository], relative))
    for relative in JSON_SOURCES:
        entries["PSDE-SDE/"+relative] = file_hash(under(roots["PSDE-SDE"], relative))
    return entries


def verify_sources(bindings, roots=None):
    roots = repository_roots() if roots is None else roots
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("nonempty bound source catalog required")
    for logical, checksum in bindings.items():
        relative_path(logical)
        repository, relative = logical.split("/", 1)
        if repository not in roots or file_hash(under(roots[repository], relative)) != sha256(checksum):
            raise ValueError("registered source identity changed: "+logical)
    return len(bindings)


def input_binding():
    result = read_json(ROOT/INPUT_PATH, expected_file_sha256=INPUT_FILE_SHA256)
    if result["sha256"] != digest({k: v for k, v in result.items() if k != "sha256"}):
        raise ValueError("input binding content identity mismatch")
    return result


def protocol_semantics():
    """Only public metadata/specifications are read. No raw data or adapters."""
    decision = read_json(POLICY_PATH, expected_file_sha256=DECISION_FILE_SHA256)
    validate_policy(decision)
    inputs, delivery, mechanisms = input_binding(), delivery_policy(), mechanism_registry()
    training = training_policy()
    if training["sha256"] != delivery["training"]["terrain"]["training_policy_sha256"]:
        raise ValueError("fixed terrain training recipe changed")
    management = {"MPA/project/PIRC-17/"+p: file_hash(ROOT.parent/"MPA/project/PIRC-17"/p)
                  for p in MANAGEMENT_SOURCES}
    return {"schema_version": VERSION, "state": "sealed-candidate-not-authorized",
        "dataset_inputs": inputs,
        "components": {"finite_delivery": delivery, "decision_policy": decision,
            "method_mechanisms": mechanisms, "terrain_training": training,
            "pirc22_selection": load_benchmark_selection_binding(),
            "terrain_configurations": terrain_configurations(),
            "terrain_comparisons": {mode: comparison_registry(origin_mode=mode) for mode in ORIGIN_MODES},
            "method_comparisons": {mode: method_comparison_registry(origin_mode=mode) for mode in ORIGIN_MODES},
            "metrics": read_json(ROOT/"experiments/pirc17/metric_registry.json")},
        "management_source_sha256": management,
        "forecast_contract": {"input_modes": delivery["origin_modes"], "forecast": delivery["forecast"],
            "training": delivery["training"], "primary_origin_mode": "causal_prefix",
            "coordinate_frames": {"terrain_scoring": "origin-anchored local east/north metres",
                "method_training_and_prediction": "first-visible-history-point-equirectangular-v1; transform to shared scoring frame without future centering",
                "earth_radius_m": 6371008.8, "polar_exclusion_absolute_latitude_degrees": 89.},
            "method_solar": "NOAA geometric UTC-subsecond solar at predicted positions and actual origin-relative elapsed time; stored historical columns not substituted",
            "history_points": 3, "terrain_history_seconds": 5.,
            "missing_features": "Use frozen canonical missing masks/neutral transforms at predicted positions, retain invalid query counts; do not delete forecasts on score/coverage outcomes.",
            "truth": "Original observed positions at distinct nearest scoring instants; no interpolation or future-state conditioning.",
            "inertial": delivery["metrics"]["inertial_baseline"]},
        "eligibility_contract": {"source_rule": "Complete registered original indexes, at least two prefix points, strict positive times, <=60s adjacent gap, >=1800s followup; all frozen factor+history validity from origin through registered target_end, shared by all configurations.",
            "score_slots": "Require four distinct original observations nearest60/300/900/1800s within30s, earlier tie. Eligibility failures are frozen before prediction, never outcome-based.",
            "selection": delivery["selection"],
            "selection_timing": "Only after exact ACCEPT01 access permission: seal all final eligibility dispositions and hash-selected primary/secondary IDs before predictions or scores.",
            "split_separation": "Use unchanged published split; train/validation/final sample,segment,block overlaps must remain zero. Method adapt is the already frozen subdivision of outer train, not final eval.",
            "failure_policy": "Preserve original denominator. Failed/partial/absent required rows produce explicit unavailable; no successful-subset intersection or replacements."},
        "resource_contract": delivery["resource_budget"],
        "generation_counts": mechanisms["pre_seal_workload_supplement"],
        "component_precedence": {"counts": "generation_counts includes290 same-grid references:11513 stochastic calls,11020 scientific forecasts; it supersedes the earlier base11223 ceiling without changing phase budgets.",
            "decision_rules": "decision_policy is the completed DEV10 layer over the retained earlier finite_delivery candidate. Its closed evidence and claim limits supersede older unresolved-qualification notes.",
            "preparation": "All empirical qualification and cost probes are closed. Historical preparation remaining-seconds fields grant no further work; no prechecks, retries or automatic parameter growth are authorized.",
            "allowed_new_precheck_forecasts": 0},
        "source_binding_scope": "This seal binds existing core sources. DEV04 adds an execution seal containing the exact full current catalog, including new executor/guard/matrix code, before TEST01/REVIEW01/ACCEPT01. Changes to an already-bound source require a new protocol identity and acceptance; no in-place edits.",
        "access_contract": {"human_gate": "ACCEPT-01", "dataset_id_alone_unlocks": False,
            "required": ["this exact protocol content SHA256", "exact execution-manifest content SHA256",
                "TEST-01 and REVIEW-01 PASS records on both identities", "human confirmation provenance on both identities"],
            "protected": ["final_eval_eligibility", "final_eval_positions", "final_eval_features", "final_eval_metrics"],
            "population_seal_required_for": ["final_eval_positions", "final_eval_features", "final_eval_metrics"],
            "legacy_bypass": "No formal path may call a legacy final_eval_unlock interface directly; obtain the adapter acknowledgement only inside the verified guard callback.",
            "trust_boundary": "Authorized operator records the actual human decision and pins its receipt hash. This is a workflow guard against absent/stale/wrong-scope approval, not an OS sandbox or proof of human identity against a malicious filesystem owner."},
        "completion_boundary": "A protocol seal is not RUN01 completion, scientific acceptance, or paper delivery. Both full experiment matrices, independent reanalysis, all cards, full editable paper/PDF and human ACCEPT02 remain required.",
        "final_eval_label_prediction_metric_reads": 0, "final_eval_authorized": False,
        "global_numerical_qualification_claimed": False}


def build_protocol():
    return envelope({**protocol_semantics(), "source_sha256": source_catalog()})


def validate_protocol(value, *, expected_sha256=None):
    payload = unpack(value, expected_sha256=expected_sha256)
    if digest({k: v for k, v in payload.items() if k != "source_sha256"}) != digest(protocol_semantics()):
        raise ValueError("protocol differs from the fixed scientific/input/authority contract")
    catalog = payload.get("source_sha256")
    # Existing core bindings are immutable. Additional DEV04 files must appear
    # in the later full execution seal, not be silently treated as old code.
    current = source_catalog()
    required = {f"PSDE-SDE/{p}" for p in JSON_SOURCES} | {
        "PSDE-SDE/experiments/pirc17/protocol.py", "PSDE-SDE/experiments/pirc17/protocol_core.py",
        "PSDE-SDE/experiments/pirc17/final_eval_guard.py"}
    if not isinstance(catalog, dict) or not required <= catalog.keys() or not catalog.keys() <= current.keys():
        raise ValueError("protocol core source bindings are incomplete")
    verify_sources(catalog)
    verify_sources(payload["management_source_sha256"])
    return payload


def validate_execution(value, protocol):
    """Schema for DEV04's later concrete runner/matrix seal, not a stub pass."""
    sealed = validate_protocol(protocol)
    payload = unpack(value)
    required = {"schema_version", "protocol_sha256", "source_sha256", "matrix_sha256",
        "runtime_manifest_sha256", "budget_ledger_schema_version", "entrypoint",
        "phase_caps_seconds", "max_generated_forecasts", "scientific_forecasts",
        "method_required_slots", "terrain_configurations", "final_eval_authorized"}
    if (set(payload) != required or payload["schema_version"] != EXECUTION_VERSION
            or payload["protocol_sha256"] != protocol["sha256"] or payload["final_eval_authorized"] is not False
            or payload["phase_caps_seconds"] != sealed["resource_contract"]["phase_caps_seconds"]
            or type(payload["max_generated_forecasts"]) is not int or payload["max_generated_forecasts"] != 11513
            or type(payload["scientific_forecasts"]) is not int or payload["scientific_forecasts"] != 11020):
        raise ValueError("execution seal changed protocol/scope/budget")
    required_slots = sorted(name for name, r in sealed["components"]["method_mechanisms"]["slots"].items()
                            if r["disposition"] == "REQUIRED")
    if (payload["method_required_slots"] != required_slots
            or payload["terrain_configurations"] != sorted(sealed["components"]["terrain_configurations"])):
        raise ValueError("execution must retain all28 required slots and ten terrain configurations")
    for name in ("matrix_sha256", "runtime_manifest_sha256"):
        sha256(payload[name])
    if (payload["source_sha256"] != source_catalog()
            or any(payload["source_sha256"].get(k) != v for k, v in sealed["source_sha256"].items())):
        raise ValueError("execution source catalog incomplete or core changed")
    entrypoint = relative_path(payload["entrypoint"])
    if entrypoint not in payload["source_sha256"] or payload["budget_ledger_schema_version"] != "pirc17-cumulative-phase-budget-v1":
        raise ValueError("bound actual executor and durable budget contract required")
    verify_sources(payload["source_sha256"])
    return payload


if __name__ == "__main__":
    value = build_protocol()
    validate_protocol(value)
    path, written = publish(ROOT/"experiments/pirc17/protocols", unpack(value))
    print(json.dumps({"path": str(path.relative_to(ROOT)), "protocol_sha256": written["sha256"],
        "source_bindings": len(value["payload"]["source_sha256"]), "final_eval_authorized": False,
        "final_eval_label_prediction_metric_reads": 0}))
