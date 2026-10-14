"""Candidate regular-interval method fitting, not formal-run authorization.

Keep the registered NEX326 model/estimator/transfer mechanisms. Refit in the
causal coordinate/solar policy; never reuse historical coefficients by shape.
Dropping ONLY an incomplete terminal training interval makes the reference tau
explicit. Original observations and evaluation targets remain untouched.
Callers must register empirical inputs/cost caps before invoking the fitter and
own any process deadline; this module does not start jobs or retry failures.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.nex326.model import train_model
from .method_rollout import MethodDynamics
from .origins import frozen_array
from .workload import method_inventory

VERSION = "pirc17-regular-method-training-v1"
TRAINING_FIELDS = ("model", "estimator", "condition", "transfer", "finetune", "objective_lambda", "dt_seconds")
FIT_SOURCES = ("experiments/pirc17/method_training.py", "experiments/pirc17/method_development.py",
               "experiments/pirc17/method_inputs.py", "experiments/nex326/cohort.py", "experiments/nex326/model.py")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def required_slot(slot_id):
    matches = [s for s in method_inventory()["slots"] if s["slot_id"] == slot_id]
    if len(matches) != 1 or matches[0]["disposition"] != "REQUIRED":
        raise ValueError("fitting requires an exact REQUIRED method slot")
    return matches[0]


def uniform_training_segment(segment, interval_seconds):
    """Reject internal off-grid steps; trim at most one short final step."""
    if type(interval_seconds) not in (int, float) or not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("positive finite reference interval required")
    segment.validate()
    dt = np.diff(segment.time)
    full = np.isclose(dt, interval_seconds, rtol=0, atol=1e-9)
    off = np.flatnonzero(~full)
    drop = int(len(off) != 0)
    if drop and (len(off) != 1 or off[0] != len(dt)-1 or not 0 < dt[-1] < interval_seconds):
        raise ValueError("only one incomplete terminal training interval may be removed")
    end = len(segment.time) - drop
    if end < 4:
        raise ValueError("uniform training segment has fewer than four points; no silent sample exclusion")
    output = replace(segment, time=frozen_array(segment.time[:end]), state=frozen_array(segment.state[:end]),
                     conditions={n: frozen_array(v[:end]) for n, v in segment.conditions.items()})
    output.validate()
    return output, {"segment_id": segment.segment_id, "removed_partial_tail": bool(drop),
        "removed_tail_seconds": float(dt[-1]) if drop else 0., "transition_count": end-1,
        "reference_interval_seconds": float(interval_seconds), "scoring_observations_changed": False}


def prepare_method_training(prepared, slot_id, *, max_transitions):
    if type(max_transitions) is not int or max_transitions <= 0:
        raise ValueError("positive explicit total transition cap required")
    slot = required_slot(slot_id)
    components = slot["components"]
    interval = components["dt_seconds"]
    roles, resampling = prepared.resample(interval)
    if set(roles) != {"train", "adapt", "validation"} or any(not rows for rows in roles.values()):
        raise ValueError("nonempty separate train/adapt/validation roles required")
    output, details, seen = {}, [], set()
    for role, segments in roles.items():
        output[role] = []
        for segment in segments:
            if segment.segment_id in seen:
                raise ValueError("training segment repeats or crosses roles")
            seen.add(segment.segment_id)
            uniform, report = uniform_training_segment(segment, interval)
            if any(name not in uniform.conditions for name in components["condition"]):
                raise ValueError("registered condition column missing")
            output[role].append(uniform)
            details.append({"role": role, **report})
    count = sum(r["transition_count"] for r in details)
    if count > max_transitions:
        raise ValueError("uniform training transition budget exceeded")
    training_components = {name: components[name] for name in TRAINING_FIELDS if name in components}
    identity = {"version": VERSION, "input_sha256": prepared.identity["sha256"],
        "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[2]/name).read_bytes()).hexdigest()
                          for name in FIT_SOURCES}, "numpy_version": np.__version__,
        "training_components": training_components, "interval_policy": "complete-uniform-intervals-only",
        "solar_policy": prepared.identity.get("solar_policy"), "frame_policy": prepared.identity.get("frame_policy"),
        "per_segment": details,
        "sample_counts": {role: len(rows) for role, rows in output.items()},
        "transitions_by_role": {role: sum(r["transition_count"] for r in details if r["role"] == role) for role in output},
        "removed_tails_by_role": {role: sum(r["removed_partial_tail"] for r in details if r["role"] == role) for role in output},
        "reference_interval_seconds": float(interval), "recommended_method_history_clock_seconds": float(interval),
        "history_semantics": "legacy training uses the preceding observation-interval secant; forecast starts from its explicit Origin velocity, then predicted reference-interval secants",
        "noise_policy": "fitted/calibrated velocity-rate covariance R; declared diffusion Q=R*tau, independent of numerical h",
        "continuous_time_MLE_equivalence_claimed": False, "formal_training_accepted": False}
    # Fitting is deterministic and does not use propagation/integrator/arm seed.
    # Equal keys identify candidates for fit reuse ONLY, never forecast reuse.
    identity["training_identity_sha256"] = _digest(identity)
    return {role: tuple(rows) for role, rows in output.items()}, identity, slot, resampling


@dataclass(frozen=True)
class FittedMethod:
    dynamics: MethodDynamics
    training: dict
    slot_id: str
    fit_seconds: float


def fit_development_method(prepared, slot_id, *, max_transitions):
    """Explicit empirical fitting entry; no data loading, prediction or retry."""
    roles, training, slot, _ = prepare_method_training(prepared, slot_id, max_transitions=max_transitions)
    started = time.perf_counter()
    model = train_model(roles["train"], roles["validation"], roles["adapt"], (), slot["components"])
    seconds = time.perf_counter()-started
    fit_identity = _digest({"training_identity_sha256": training["training_identity_sha256"], "model": model.to_dict()})
    dynamics = MethodDynamics.bind(model, reference_interval_seconds=training["reference_interval_seconds"],
        fit_identity=fit_identity,
        noise_binding_rationale="All retained fitting/calibration transitions have explicit duration tau. R is already estimator-scaled velocity-rate covariance; Q=R*tau matches R*tau^2 displacement variance at tau for constant coefficients. This is a discrete-regression embedding, not global continuous-time MLE equivalence.")
    return FittedMethod(dynamics, training, slot_id, seconds)
