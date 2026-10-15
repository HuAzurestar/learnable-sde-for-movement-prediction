"""Scoped measurement packets, not actual research or fabricated native truth."""
import sys

import pytest

from application import propagation_resources as resources
from infrastructure.research_store import ResearchError, digest


def fixture():
    cell = {"resource_class": "cpu", "plugin_id": "affine-propagation",
        "propagation_request": {"fixture": "metadata-only"},
        "resource_observation_policy": resources.resource_policy()}
    spec = {"study_id": "resource-unit", "code_hash": digest("fixture source"), "cells": [cell]}
    return spec, cell, "a"*32


def packet():
    spec, cell, attempt = fixture()
    probe = resources.PropagationResourceProbe(spec, cell, attempt)
    return spec, cell, attempt, probe.snapshot("SUCCEEDED")


def test_actual_observer_has_explicit_units_scope_and_no_authority_or_private_ids():
    spec, cell, attempt, value = packet()
    resources.validate_packet(value, spec, cell, attempt, execution_state="SUCCEEDED")
    assert value["duration"]["wall_ns"] > 0 and value["duration"]["cpu_ns"] >= 0
    assert value["duration"]["scope"] == resources.DURATION_SCOPE
    memory = value["memory"]
    assert memory["scope"] == resources.MEMORY_SCOPE and memory["units"] == "bytes"
    assert memory["includes_children"] is False and memory["is_phase_isolated"] is False
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        assert memory["status"] == "MEASURED" and memory["peak_resident_bytes"] > 0
    serialized = str(value)
    for forbidden in ("process_id", "parent_process_id", "remaining_seconds", "budget", "deadline"):
        assert forbidden not in serialized
    assert value["resumed"] is False and value["includes_previous_attempts"] is False
    assert value["policy_hash"] == digest(cell["resource_observation_policy"])


@pytest.mark.parametrize("state,point", [("SUCCEEDED", "job-payload-before-encoding"),
    ("FAILED", "observer-call"), ("CHECKPOINTED", "checkpoint-save-before-ack")])
def test_real_elapsed_time_and_existing_observer_point_are_forwarded(monkeypatch, state, point):
    spec, cell, attempt = fixture()
    wall, cpu = iter((100, 150)), iter((20, 140))
    monkeypatch.setattr(resources.time, "perf_counter_ns", lambda: next(wall))
    monkeypatch.setattr(resources.time, "process_time_ns", lambda: next(cpu))
    original = resources.process_resources
    seen = []
    def observe(p):
        seen.append(p)
        return original(p)
    monkeypatch.setattr(resources, "process_resources", observe)
    value = resources.PropagationResourceProbe(spec, cell, attempt, resumed=True).snapshot(state)
    assert value["duration"] == {"wall_ns": 50, "cpu_ns": 120, "units": "ns",
        "scope": resources.DURATION_SCOPE, "cpu_scope": "current-process-all-threads"}
    assert seen == [point] and value["resumed"] is True


def test_unavailable_native_counter_is_null_not_zero_or_declared_limit(monkeypatch):
    from infrastructure import pirc26_process_resources as observer
    monkeypatch.setattr(observer.sys, "platform", "fixture-unsupported")
    spec, cell, attempt, value = packet()
    assert value["memory"]["status"] == "UNAVAILABLE"
    assert value["memory"]["peak_resident_bytes"] is None
    assert value["memory"]["reason"] == "UNSUPPORTED_PLATFORM"
    resources.validate_packet(value, spec, cell, attempt)


@pytest.mark.parametrize("fault", ["extra", "unknown-policy", "bool-required", "wrong-unit"])
def test_only_exact_explicit_frozen_policy_enables_collection(fault):
    spec, cell, attempt = fixture()
    if fault == "extra": cell["resource_observation_policy"]["extra"] = True
    elif fault == "unknown-policy": cell["resource_observation_policy"]["schema_version"] += "-unknown"
    elif fault == "bool-required": cell["resource_observation_policy"]["required"] = 1
    else: cell["resource_observation_policy"]["memory_scope"] = "whole-process-tree"
    with pytest.raises(ResearchError):
        resources.PropagationResourceProbe(spec, cell, attempt)
    del cell["resource_observation_policy"]
    assert resources.observation_enabled(cell) is False


@pytest.mark.parametrize("fault", ["attempt", "spec", "cell", "code", "policy", "resumed", "state",
    "negative-wall", "bool-wall", "negative-cpu", "extra", "zero-memory", "bool-memory", "unit",
    "tree", "phase", "memory-extra", "backend", "reason"])
def test_changed_or_misrepresented_packet_is_refused(fault):
    spec, cell, attempt, value = packet()
    if fault in {"attempt", "spec", "cell", "code", "policy"}:
        key = {"attempt": "attempt_id", "spec": "spec_hash", "cell": "cell_hash", "code": "code_hash", "policy": "policy_hash"}[fault]
        value[key] = "b"*len(value[key])
    elif fault == "resumed": value["resumed"] = 1
    elif fault == "state": value["execution_state"] = "SCIENTIFICALLY_QUALIFIED"
    elif fault == "negative-wall": value["duration"]["wall_ns"] = -1
    elif fault == "bool-wall": value["duration"]["wall_ns"] = True
    elif fault == "negative-cpu": value["duration"]["cpu_ns"] = -1
    elif fault == "extra": value["remaining_seconds"] = 30
    elif fault == "zero-memory": value["memory"]["peak_resident_bytes"] = 0
    elif fault == "bool-memory": value["memory"]["peak_resident_bytes"] = True
    elif fault == "unit": value["memory"]["units"] = "KiB"
    elif fault == "tree": value["memory"]["includes_children"] = True
    elif fault == "phase": value["memory"]["is_phase_isolated"] = True
    elif fault == "memory-extra": value["memory"]["process_id"] = 123
    elif fault == "backend": value["memory"]["resident_counter"] = "declared-resource-plan"
    else: value["memory"]["reason"] = "pretend-success"
    with pytest.raises(ResearchError):
        resources.validate_packet(value, spec, cell, attempt)


def test_separate_attempt_snapshots_are_not_summed_or_restored():
    spec, cell, first, value = packet()
    second = "b"*32
    resumed = resources.PropagationResourceProbe(spec, cell, second, resumed=True).snapshot("CHECKPOINTED")
    assert value["attempt_id"] == first and resumed["attempt_id"] == second
    assert resumed["includes_previous_attempts"] is False
    with pytest.raises(ResearchError):
        resources.validate_packet(resumed, spec, cell, first)


@pytest.mark.parametrize("fault", ["spec", "cell", "policy"])
def test_frozen_context_cannot_be_rebound_after_the_work_started(fault):
    spec, cell, attempt = fixture()
    probe = resources.PropagationResourceProbe(spec, cell, attempt)
    if fault == "spec": spec["code_hash"] = digest("different source")
    elif fault == "cell": cell["propagation_request"]["fixture"] = "changed request"
    else: cell["resource_observation_policy"]["required"] = 1
    with pytest.raises(ResearchError):
        probe.snapshot("SUCCEEDED")


def test_output_binding_is_lexical_and_never_authorizes_a_foreign_attempt_root(tmp_path):
    spec, cell, attempt = fixture()
    spec["runtime_binding"] = {"root": str(tmp_path), "store_id": "fixture"}
    output = tmp_path/"pirc25"/"artifacts"/(".attempt-"+attempt)/"result.json"
    assert resources.attempt_from_output(spec, output) == attempt
    for other in (output.with_name("not-result.json"), tmp_path/"foreign"/"result.json",
                  output.parent.parent/".attempt-not-an-identity"/"result.json"):
        with pytest.raises(ResearchError):
            resources.attempt_from_output(spec, other)
    del spec["runtime_binding"]
    with pytest.raises(ResearchError):
        resources.attempt_from_output(spec, output)


@pytest.mark.parametrize("change", [{"resource_class": "gpu"}, {"plugin_id": "foreign-component"}])
def test_policy_cannot_relabel_a_foreign_component_or_gpu_as_this_pipeline(change):
    spec, cell, attempt = fixture()
    cell.update(change)
    with pytest.raises(ResearchError):
        resources.PropagationResourceProbe(spec, cell, attempt)
