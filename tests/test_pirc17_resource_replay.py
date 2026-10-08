"""Software-only tests; no extra empirical trajectories or forecast evidence."""
from copy import deepcopy
import json

import pytest

from experiments.pirc17 import resource_replay as replay
from experiments.pirc17.rollout import Forecast, rollout as real_rollout
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows


def test_observer_is_fresh_and_keeps_exact_zero_and_threshold_values(monkeypatch):
    values = iter([replay.engine.MINIMUM_FREE_BYTES, replay.engine.MINIMUM_FREE_BYTES-1, 0])
    monkeypatch.setattr(replay.physical_memory, "available_physical_bytes", lambda: next(values))
    observer = replay.MemoryObserver()
    observer.guard()
    with pytest.raises(MemoryError, match="2 GiB"):
        observer.guard()
    assert observer() == 0
    summary = observer.summary()
    assert summary["calls"] == 3 and summary["failed_calls"] == 0
    assert summary["minimum_observed_available_bytes"] == 0 and not summary["measurement_cache"]


@pytest.mark.parametrize("value", [True, -1, 1.0, float("nan"), None])
def test_invalid_native_value_is_unknown_ram_not_an_available_value(monkeypatch, value):
    monkeypatch.setattr(replay.physical_memory, "available_physical_bytes", lambda: value)
    observer = replay.MemoryObserver()
    with pytest.raises(MemoryError, match="unknown"):
        observer()
    assert observer.failures == 1 and observer.minimum_available_bytes is None


def test_native_api_failure_never_reuses_prior_measurement(monkeypatch):
    calls = []
    def observe():
        calls.append(1)
        if len(calls) == 1:
            return replay.engine.MINIMUM_FREE_BYTES
        raise OSError("software native failure")
    monkeypatch.setattr(replay.physical_memory, "available_physical_bytes", observe)
    observer = replay.MemoryObserver()
    assert observer() == replay.engine.MINIMUM_FREE_BYTES
    with pytest.raises(MemoryError, match="unknown") as caught:
        observer()
    assert isinstance(caught.value.__cause__, OSError)
    assert observer.calls == 2 and observer.failures == 1


@pytest.fixture
def replay_inputs(inputs, evidence, monkeypatch):
    original, tracker = inputs
    calls = []
    def old_observer():
        calls.append(1)
        return replay.engine.MINIMUM_FREE_BYTES
    original["available_memory"] = old_observer
    assert replay.engine.run(**original)["status"] == "complete"
    reference = original["output"]
    reference_sha = replay._hash(reference)
    audit = reference.with_suffix(".audit.json")
    report = replay.audited(read_rows(reference), reference.parent, reference_sha, 10.27506475, evidence)
    audit.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(replay.physical_memory, "available_physical_bytes", lambda: replay.engine.MINIMUM_FREE_BYTES)
    args = {key: original[key] for key in ("eligibility", "release", "snapshot", "data_root")}
    args.update(**evidence, reference_ledger=reference, reference_ledger_sha256=reference_sha,
                reference_audit=audit, reference_audit_sha256=replay._hash(audit),
                output=reference.with_name("native-replay.jsonl"))
    return args, tracker, report, len(calls)


def test_complete_replay_uses_unchanged_engine_and_retains_numerical_failures(replay_inputs):
    args, tracker, original_audit, original_calls = replay_inputs
    # Deliberately failing software-only precision fixture. This does not
    # change the real reference audit's registered 10.27506475 m tolerance.
    original_audit = replay.audited(read_rows(args["reference_ledger"]), args["reference_ledger"].parent,
        args["reference_ledger_sha256"], 1e-12, {key: args[key] for key in
            ("fit", "fit_sha256", "fit_ledger", "fit_ledger_sha256", "training_policy_sha256")})
    args["reference_audit"].write_text(json.dumps(original_audit), encoding="utf-8")
    args["reference_audit_sha256"] = replay._hash(args["reference_audit"])
    sources = replay.engine.source_hashes()
    before = args["reference_ledger"].read_bytes()
    result = replay.run(**args)
    assert result["status"] == "complete" and result["prediction_equivalence_passed"]
    assert result["success_count"] == result["attempted_run_count"] == result["expected_run_count"] == 32
    assert result["failure_count"] == result["unattempted_run_count"] == result["terminal_error_count"] == 0
    assert not result["resource_stopped"] and result["engine_terminal_error_count"] == 0
    assert result["observer"]["calls"] == original_calls+5
    assert result["observer"]["failed_calls"] == 0
    assert not result["certified"] and not result["formal_training_accepted"]
    assert result["final_eval_label_prediction_metric_reads"] == 0
    rows = read_rows(args["output"])
    comparison = next(row for row in rows if row["type"] == "comparison")
    assert comparison["compared_run_count"] == 32 and comparison["audit_fields_exact"]
    assert all(row["maximum_position_difference_m"] == 0 for row in comparison["runs"])
    assert result["candidate_numerical"] == replay.numerical_summary(original_audit)
    assert result["candidate_numerical"]["out_of_tolerance"] > 0
    assert replay.engine.source_hashes() == sources and args["reference_ledger"].read_bytes() == before
    assert len(tracker["calls"]) == 64 and all(m.closed for m in tracker["maps"])
    assert rows[0]["observer"] == replay.physical_memory.identity()
    assert rows[0]["source_sha256"] == replay.source_hashes()
    assert rows[0]["minimum_free_bytes"] == replay.engine.MINIMUM_FREE_BYTES


def test_passing_reference_also_replays_without_numerical_relabeling(replay_inputs):
    args, _, original_audit, _ = replay_inputs
    result = replay.run(**args)
    assert result["prediction_equivalence_passed"]
    assert result["candidate_numerical"] == replay.numerical_summary(original_audit)
    assert result["candidate_numerical"]["out_of_tolerance"] == 0
    assert not result["certified"]


@pytest.mark.parametrize("name", ["particles", "steps", "seeds", "configurations", "limit_origins", "wall_seconds", "tolerance_m", "available_memory"])
def test_replay_has_no_scope_threshold_or_observer_override(replay_inputs, name):
    args, tracker, _, _ = replay_inputs
    with pytest.raises(TypeError):
        replay.run(**args, **{name: None})
    assert len(tracker["calls"]) == 32 and not args["output"].exists()


@pytest.mark.parametrize("kind", ["envelope", "forecast", "audit", "particles"])
def test_existing_outputs_are_not_overwritten(replay_inputs, kind):
    args, tracker, _, _ = replay_inputs
    path = {"envelope": args["output"], "forecast": args["output"].with_suffix(".forecast.jsonl"),
            "audit": args["output"].with_suffix(".audit.json"),
            "particles": args["output"].with_suffix(".forecast.particles")}[kind]
    if kind == "particles":
        path.mkdir()
    else:
        path.write_text("owned-software-marker", encoding="utf-8")
    with pytest.raises(FileExistsError):
        replay.run(**args)
    assert len(tracker["calls"]) == 32
    assert path.is_dir() if kind == "particles" else path.read_text(encoding="utf-8") == "owned-software-marker"


@pytest.mark.parametrize("field", ["reference_ledger_sha256", "reference_audit_sha256"])
def test_bad_reference_hash_cannot_start_forecasting(replay_inputs, field):
    args, tracker, _, _ = replay_inputs
    args[field] = "f"*64
    with pytest.raises(ValueError, match="hash"):
        replay.run(**args)
    assert len(tracker["calls"]) == 32 and not args["output"].exists()


@pytest.mark.parametrize("field", ["numerical_audit", "particle_precision", "audit_source_sha256"])
def test_rehashed_but_false_reference_audit_is_rejected_before_forecast(replay_inputs, field):
    args, tracker, report, _ = replay_inputs
    changed = deepcopy(report)
    if field == "audit_source_sha256":
        changed[field]["direct_linear_check.py"] = "f"*64
    else:
        changed[field]["certified"] = True
    args["reference_audit"].write_text(json.dumps(changed), encoding="utf-8")
    args["reference_audit_sha256"] = replay._hash(args["reference_audit"])
    result = replay.run(**args)
    assert result["status"] == "failed" and result["unattempted_run_count"] == 32
    assert result["attempted_run_count"] == 0 and not result["prediction_equivalence_passed"]
    assert len(tracker["calls"]) == 32
    assert not args["output"].with_suffix(".forecast.jsonl").exists()


@pytest.mark.parametrize("fault", ["low", "api_failure"])
@pytest.mark.parametrize("stop_call,attempted,engine_errors", [(1, 0, 0), (3, 0, 1), (7, 1, 1)])
def test_low_or_unknown_ram_stops_complete_remaining_workload(replay_inputs, monkeypatch, fault, stop_call, attempted, engine_errors):
    args, tracker, _, _ = replay_inputs
    calls = []
    def observe():
        calls.append(1)
        if len(calls) == stop_call:
            if fault == "api_failure":
                raise OSError("software native failure")
            return 0
        return replay.engine.MINIMUM_FREE_BYTES
    monkeypatch.setattr(replay.physical_memory, "available_physical_bytes", observe)
    result = replay.run(**args)
    assert result["status"] == "failed" and not result["prediction_equivalence_passed"]
    assert result["resource_stopped"] and result["engine_terminal_error_count"] == engine_errors
    assert result["attempted_run_count"] == attempted and result["unattempted_run_count"] == 32-attempted
    assert result["success_count"] == 0 and result["failure_count"] == attempted
    assert len(tracker["calls"]) == 32+attempted
    inner_path = args["output"].with_suffix(".forecast.jsonl")
    if stop_call == 1:
        assert not inner_path.exists()
    else:
        inner = read_rows(inner_path)
        assert inner[-1]["resource_stopped"] and inner[-1]["status"] == "failed"
    assert all(m.closed for m in tracker["maps"])
    assert result["observer"]["failed_calls"] == int(fault == "api_failure")


@pytest.mark.parametrize("fault", ["positions", "targets"])
def test_replay_compares_complete_arrays_not_just_success_counts(replay_inputs, monkeypatch, fault):
    args, tracker, _, _ = replay_inputs
    if fault == "targets":
        tracker["windows"]["validation"][0].target_positions_m[:] = 100000.
    else:
        original = replay.engine.rollout
        def changed(*a, **k):
            prediction = original(*a, **k)
            positions = prediction.positions_m.copy()
            positions[-1, -1, -1] += 1.
            return Forecast(prediction.elapsed_seconds, positions, prediction.invalid_feature_rows, prediction.feature_query_rows)
        monkeypatch.setattr(replay.engine, "rollout", changed)
    result = replay.run(**args)
    assert result["status"] == "failed" and not result["prediction_equivalence_passed"]
    assert result["success_count"] == 32
    if fault == "targets":
        comparison = next(row for row in read_rows(args["output"]) if row["type"] == "comparison")
        assert all(row["positions_exact"] for row in comparison["runs"])
        assert any(not row["targets_exact"] for row in comparison["runs"])


@pytest.mark.parametrize("fault", ["consumer_source", "reference_bytes"])
def test_changed_source_or_reference_cannot_pass_even_after_equal_forecasts(replay_inputs, monkeypatch, fault):
    args, _, _, _ = replay_inputs
    if fault == "consumer_source":
        original, calls = replay.source_hashes(), []
        def sources():
            calls.append(1)
            return original if len(calls) == 1 else {**original, "resource_replay.py": "f"*64}
        monkeypatch.setattr(replay, "source_hashes", sources)
    else:
        def progress(row):
            if row["phase"] == "reference_verified":
                path = args["reference_ledger"]
                path.write_bytes(path.read_bytes()+b"\n")
        args["progress"] = progress
    result = replay.run(**args)
    assert result["status"] == "failed" and result["success_count"] == 32
    assert not result["prediction_equivalence_passed"] and result["terminal_error_count"] >= 1


def test_real_kernel_memory_observer_does_not_change_predictions(inputs, evidence, monkeypatch):
    # Real causal kernel and models, software windows/map fixture only.
    original, _ = inputs
    monkeypatch.setattr(replay.engine, "rollout", real_rollout)
    original.update(configurations=["base"], seeds=[replay.engine.SEEDS[0]], limit_origins=1)
    assert replay.engine.run(**original)["status"] == "complete"
    reference = original["output"]
    reference_sha = replay._hash(reference)
    audit = reference.with_suffix(".audit.json")
    audit.write_text(json.dumps(replay.audited(read_rows(reference), reference.parent,
        reference_sha, 10.27506475, evidence)), encoding="utf-8")
    monkeypatch.setattr(replay.physical_memory, "available_physical_bytes", lambda: replay.engine.MINIMUM_FREE_BYTES)
    result = replay.run(**evidence, reference_ledger=reference, reference_ledger_sha256=reference_sha,
        reference_audit=audit, reference_audit_sha256=replay._hash(audit),
        **{key: original[key] for key in ("eligibility", "release", "snapshot", "data_root")},
        output=reference.with_name("real-kernel-replay.jsonl"))
    assert result["prediction_equivalence_passed"] and result["success_count"] == 4
    assert result["observer"]["calls"] > 1000


@pytest.mark.parametrize("value", [-1., float("inf"), float("nan"), True, "1.0"])
def test_invalid_wall_time_is_not_runtime_evidence(value):
    with pytest.raises(ValueError, match="wall times"):
        replay.rollout_seconds([{}, {}, {"rollout_wall_seconds_including_lazy_map_initialization": value}, {}])


def test_bad_reference_wall_time_is_rejected_before_forecasting(replay_inputs):
    args, tracker, _, _ = replay_inputs
    rows = read_rows(args["reference_ledger"])
    rows[2]["rollout_wall_seconds_including_lazy_map_initialization"] = -1.
    args["reference_ledger"].write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    args["reference_ledger_sha256"] = replay._hash(args["reference_ledger"])
    report = replay.audited(rows, args["reference_ledger"].parent, args["reference_ledger_sha256"], 10.27506475,
        {key: args[key] for key in ("fit", "fit_sha256", "fit_ledger", "fit_ledger_sha256", "training_policy_sha256")})
    args["reference_audit"].write_text(json.dumps(report), encoding="utf-8")
    args["reference_audit_sha256"] = replay._hash(args["reference_audit"])
    result = replay.run(**args)
    assert result["status"] == "failed" and result["attempted_run_count"] == 0
    assert result["unattempted_run_count"] == 32 and len(tracker["calls"]) == 32


@pytest.mark.parametrize("suffix", [".forecast.jsonl", ".audit.json"])
def test_output_mutation_after_equal_comparison_cannot_pass(replay_inputs, monkeypatch, suffix):
    args, _, _, _ = replay_inputs
    original = replay.compare
    def compare_then_corrupt(*a):
        comparison = original(*a)
        path = args["output"].with_suffix(suffix)
        path.write_bytes(path.read_bytes()+b"\n")
        return comparison
    monkeypatch.setattr(replay, "compare", compare_then_corrupt)
    result = replay.run(**args)
    assert result["success_count"] == 32
    assert result["status"] == "failed" and not result["prediction_equivalence_passed"]
    assert result["terminal_error_count"] >= 1


def test_final_guard_failure_cannot_be_hidden_by_equal_saved_arrays(replay_inputs, monkeypatch):
    args, _, _, original_calls = replay_inputs
    calls = []
    def observe():
        calls.append(1)
        return 0 if len(calls) == original_calls+5 else replay.engine.MINIMUM_FREE_BYTES
    monkeypatch.setattr(replay.physical_memory, "available_physical_bytes", observe)
    result = replay.run(**args)
    assert result["success_count"] == 32 and result["failure_count"] == 0
    assert result["resource_stopped"] and result["engine_terminal_error_count"] == 0
    assert result["status"] == "failed" and not result["prediction_equivalence_passed"]
