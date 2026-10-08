"""Strict reader for the complete candidate fit AND its original attempt ledger.

This is not a final-training acceptance. Reading a subset of configurations never
permits an incomplete or mixed-version fifty-model bundle to pass preflight.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from .configurations import terrain_configurations
from .direct_linear import digest, restore_direct_dynamics, training_policy
from .direct_linear_fit import FIT_VERSION, parameter_identity, source_hashes
from .inference import SEEDS
from .qualification import _hash


def read_bound(path, sha256, *, jsonl=False):
    path = Path(path)
    if (not isinstance(sha256, str) or len(sha256) != 64
            or any(c not in "0123456789abcdef" for c in sha256) or _hash(path) != sha256):
        raise ValueError("candidate evidence hash mismatch")
    raw = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines()] if jsonl else json.loads(raw)


def load_candidate(*, fit, fit_sha256, fit_ledger, fit_ledger_sha256, training_policy_sha256):
    bundle = read_bound(fit, fit_sha256)
    records = read_bound(fit_ledger, fit_ledger_sha256, jsonl=True)
    policy, configs = training_policy(), terrain_configurations()
    expected = set(itertools.product(configs, SEEDS))
    count = len(expected)
    if (bundle.get("schema_version") != FIT_VERSION
            or bundle.get("purpose") != "development_training_candidate_not_final_acceptance"
            or bundle.get("training_policy") != policy or policy["sha256"] != training_policy_sha256
            or bundle.get("configurations") != configs or bundle.get("source_sha256") != source_hashes()
            or bundle.get("data_roles") != {"solve":"train", "diagnostic":"validation", "selection":"none", "final_eval_reads":0}
            or bundle.get("certified") is not False or bundle.get("formal_training_accepted") is not False
            or any(type(bundle.get(k)) is not int or bundle[k] != count for k in
                   ("expected_model_count", "attempted_model_count", "successful_model_count"))
            or type(bundle.get("failure_count")) is not int or bundle["failure_count"] != 0):
        raise ValueError("candidate fit schema, policy, sources, roles or completion differs")
    if (len(records) != count+4 or [r.get("type") for r in records] !=
            ["initialization", "header"]+["fit"]*count+["artifact", "completion"]):
        raise ValueError("complete original candidate fit ledger required")
    init, header, artifact, completion = records[0], records[1], records[-2], records[-1]
    if (init.get("schema_version") != FIT_VERSION or init.get("expected_model_count") != count
            or init.get("reference_fit_sha256") != bundle["reference_fit_sha256"] or init.get("certified") is not False
            or header.get("source_sha256") != bundle["source_sha256"] or header.get("training_policy") != policy
            or header.get("development_identity_sha256") != bundle["development_identity"]["sha256"]
            or header.get("population") != bundle["population"]
            or artifact != {"type":"artifact", "path":Path(fit).name, "sha256":fit_sha256}
            or completion.get("status") != "complete" or completion.get("certified") is not False
            or completion.get("resource_stopped") is not False
            or any(type(completion.get(k)) is not int or completion[k] != count for k in
                   ("expected_model_count", "attempted_model_count", "successful_model_count"))
            or any(type(completion.get(k)) is not int or completion[k] != 0 for k in
                   ("failure_count", "unattempted_model_count", "terminal_error_count"))):
        raise ValueError("candidate bundle and original fit ledger disagree")
    population = bundle["population"]
    if (set(population["samples"]) != {"train", "validation"}
            or set(population["independent_blocks"]) != {"train", "validation"}
            or any(type(population["samples"][r]) is not int or population["samples"][r] < 2
                   or type(population["independent_blocks"][r]) is not int
                   or not 1 <= population["independent_blocks"][r] <= population["samples"][r]
                   for r in ("train", "validation"))):
        raise ValueError("invalid candidate train/validation population")
    development = dict(bundle["development_identity"])
    development_hash = development.pop("sha256", None)
    if (development_hash != digest(development) or development.get("version") != "pirc17-development-windows-v1"
            or development.get("eligibility_sha256") != init.get("eligibility_sha256")
            or set(development.get("sample_ids", {})) != {"train", "validation"}
            or any(len(development["sample_ids"][r]) != population["samples"][r]
                   or len(set(development["sample_ids"][r])) != population["samples"][r]
                   for r in ("train", "validation"))
            or set(development["sample_ids"]["train"]) & set(development["sample_ids"]["validation"])):
        raise ValueError("candidate development identity or role membership differs")
    if (set(bundle["models"]) != set(configs) or set(bundle["comparisons"]) != set(configs)
            or bundle["parameter_identity_count_by_configuration"] != {name:1 for name in configs}):
        raise ValueError("candidate requires all ten complete configurations")
    ledger = {}
    for record in records[2:-2]:
        key = record["configuration"], record["seed"]
        if type(key[1]) is not int or key not in expected or key in ledger or record.get("status") != "success":
            raise ValueError("duplicate, failed or unexpected candidate fit")
        ledger[key] = record
    if set(ledger) != expected:
        raise ValueError("missing candidate fits")
    models, base = {}, None
    for name, config in configs.items():
        seeds = {str(seed) for seed in SEEDS}
        if set(bundle["models"][name]) != seeds or set(bundle["comparisons"][name]) != seeds:
            raise ValueError("candidate requires every fixed seed for every configuration")
        models[name], parameter_ids = {}, set()
        for seed in SEEDS:
            identity = bundle["models"][name][str(seed)]
            model = restore_direct_dynamics(identity)
            if (identity["configuration"] != name or identity["configuration_identity"] != config["sha256"]
                    or identity["seed"] != seed or identity["training_identity"] != bundle["development_identity"]["sha256"]
                    or any(identity[r+"_transition_count"] != population["samples"][r] for r in ("train", "validation"))):
                raise ValueError("candidate model mislabeled by configuration, seed or population")
            if base is None:
                base = model.base_weights
            np.testing.assert_array_equal(base, model.base_weights)
            record = ledger[name, seed]
            parameter_id = parameter_identity(identity)
            comparison = bundle["comparisons"][name][str(seed)]
            if (record.get("model_identity_sha256") != identity["sha256"]
                    or record.get("parameter_identity_sha256") != parameter_id
                    or record.get("comparison") != comparison
                    or record.get("solver_report") != identity["conditioner_checkpoint"]["solver_report"]):
                raise ValueError("candidate fit row does not bind its model and diagnostics")
            if set(comparison["candidate"]) != {"train", "validation"}:
                raise ValueError("candidate diagnostics require both development roles")
            for role, diagnostic in identity["diagnostics"].items():
                if set(comparison["candidate"][role]) != set(diagnostic):
                    raise ValueError("candidate diagnostic fields differ")
                for key, value in diagnostic.items():
                    # NumPy fitting and Torch roundtrip use different summation
                    # orders; retain the fitter's original replay tolerance.
                    np.testing.assert_allclose(comparison["candidate"][role][key], value, rtol=1e-12, atol=1e-12)
            models[name][seed] = model
            parameter_ids.add(parameter_id)
        if len(parameter_ids) != 1:
            raise ValueError("candidate deterministic parameters unexpectedly differ by seed")
    return bundle, models, init["eligibility_sha256"]


def add_evidence_arguments(parser):
    for name in ("fit", "fit-ledger"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--training-policy-sha256", required=True)
