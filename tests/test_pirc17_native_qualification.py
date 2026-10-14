"""Software-only bounded primary-family orchestration and owned-process tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.pirc17 import native_qualification as native
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows


@pytest.fixture
def qualification(inputs, evidence, monkeypatch):
    original, tracker = inputs
    assert native.engine.run(**original)["status"] == "complete"
    ledger = original["output"]
    audit = ledger.with_suffix(".audit.json")
    audit.write_text(json.dumps(native.audited(read_rows(ledger), ledger.parent,
        native._hash(ledger), 1e-12, evidence)), encoding="utf-8")
    spec = {"schema_version": native.PLAN_VERSION,
        "purpose": "complete_primary_family_validation_numerical_pilot",
        "configurations": sorted({c for pair in native.PRIMARY_FAMILY.values() for c in pair}),
        "seeds": [native.engine.SEEDS[0]], "particles": [4, 8], "steps": [5., 2.5], "limit_origins": 1,
        "selection_policy": "lexical_independent_blocks", "map_backend": "multicell",
        "wall_seconds": 300., "outer_wall_seconds": 420, "minimum_free_bytes": native.engine.MINIMUM_FREE_BYTES,
        "tolerance_m": 1e-12, "expected_run_count": 24,
        "certified": False, "formal_training_accepted": False, "final_eval_authorized": False,
        "reference_ledger_sha256": native._hash(ledger), "reference_audit_sha256": native._hash(audit),
        "eligibility_sha256": original["eligibility_sha256"], **{k: v for k, v in evidence.items() if k.endswith("sha256")}}
    plan = ledger.with_name("plan.json")
    plan.write_text(json.dumps(spec), encoding="utf-8")
    args = {key: original[key] for key in ("fit", "fit_ledger", "eligibility", "release", "snapshot", "data_root")}
    args.update(plan=plan, plan_sha256=native._hash(plan), reference_ledger=ledger,
                reference_audit=audit, output=ledger.with_name("native.jsonl"))
    # The upstream software fixture uses a synthetic eligibility identity and
    # loader. Only that nonexistent fixture input is mapped; all other hashes
    # and the production plan/audit/particle writers remain real.
    original_hash = native._hash
    monkeypatch.setattr(native, "_hash", lambda p: original["eligibility_sha256"]
        if Path(p) == original["eligibility"] else original_hash(Path(p)))
    monkeypatch.setattr(native.physical_memory, "available_physical_bytes", lambda: native.engine.MINIMUM_FREE_BYTES)
    return args, spec, tracker


def replan(args, spec):
    args["plan"].write_text(json.dumps(spec), encoding="utf-8")
    args["plan_sha256"] = native._hash(args["plan"])


def test_complete_family_preserves_original_and_new_numerical_failures(qualification):
    args, spec, tracker = qualification
    old = args["reference_ledger"].read_bytes()
    result = native.run(**args)
    assert result["status"] == "complete" and result["success_count"] == result["expected_run_count"] == 24
    assert result["failure_count"] == result["unattempted_run_count"] == result["terminal_error_count"] == 0
    assert not result["certified"] and not result["formal_training_accepted"] and not result["resource_stopped"]
    assert result["final_eval_label_prediction_metric_reads"] == 0 and result["candidate_numerical"]["out_of_tolerance"] > 0
    assert result["observer"]["calls"] == 103 and result["observer"]["failed_calls"] == 0
    assert not result["observer"]["measurement_cache"]
    rows = read_rows(args["output"])
    assert [r["type"] for r in rows] == ["initialization", "reference_verified", "engine_completed", "audit", "completion"]
    assert rows[0]["plan"] == spec and rows[0]["plan_sha256"] == args["plan_sha256"]
    assert rows[1]["numerical"]["out_of_tolerance"] > 0 and rows[1]["numerical"]["tolerance_m"] == 1e-12
    assert args["reference_ledger"].read_bytes() == old and len(tracker["calls"]) == 32+24
    report = json.loads(native.output_paths(args["output"])["audit"].read_text())
    paired = report["particle_precision"]["paired_comparisons"]
    assert {r["comparison"] for r in paired if r["kind"] == "development_model_contrast"} == set(native.PRIMARY_FAMILY)
    assert len(report["numerical_audit"]["sensitivities"]) == 24


@pytest.mark.parametrize("fault", ["extra", "missing", "final", "certified", "training", "floor", "digest", "count",
    "family", "seed", "particles", "steps", "policy", "backend", "outer_cap", "outer_order", "tolerance"])
def test_invalid_or_scope_reduced_plan_cannot_launch(qualification, fault):
    args, spec, tracker = qualification
    if fault == "extra": spec["ignored_flag"] = True
    if fault == "missing": spec.pop("purpose")
    if fault == "final": spec["final_eval_authorized"] = True
    if fault == "certified": spec["certified"] = True
    if fault == "training": spec["formal_training_accepted"] = True
    if fault == "floor": spec["minimum_free_bytes"] = 0
    if fault == "digest": spec["fit_sha256"] = "invalid"
    if fault == "count": spec["expected_run_count"] -= 1
    if fault == "family": spec["configurations"].remove("loo-road")
    if fault == "seed": spec["seeds"] = [1]
    if fault == "particles": spec["particles"] = [8]
    if fault == "steps": spec["steps"] = [2.5]
    if fault == "policy": spec["selection_policy"] = "lexical_origins"
    if fault == "backend": spec["map_backend"] = "batched"
    if fault == "outer_cap": spec["outer_wall_seconds"] = 12*3600+1
    if fault == "outer_order": spec["outer_wall_seconds"] = 299
    if fault == "tolerance": spec["tolerance_m"] = float("nan")
    replan(args, spec)
    with pytest.raises(ValueError):
        native.run(**args)
    assert not args["output"].exists() and len(tracker["calls"]) == 32


@pytest.mark.parametrize("key", ["envelope", "forecast", "particles", "audit", "supervisor"])
def test_no_existing_artifact_is_overwritten(qualification, key):
    args, _, tracker = qualification
    path = native.output_paths(args["output"])[key]
    if key == "particles": path.mkdir()
    else: path.write_text("owned-marker")
    with pytest.raises(FileExistsError): native.run(**args)
    assert path.is_dir() if key == "particles" else path.read_text() == "owned-marker"
    assert len(tracker["calls"]) == 32


@pytest.mark.parametrize("field", ["reference_ledger_sha256", "reference_audit_sha256", "fit_sha256", "fit_ledger_sha256",
                                  "training_policy_sha256", "eligibility_sha256", "tolerance_m"])
def test_reference_binding_or_tolerance_change_cannot_start_new_forecast(qualification, field):
    args, spec, tracker = qualification
    spec[field] = 100. if field == "tolerance_m" else "f"*64
    replan(args, spec)
    result = native.run(**args)
    assert result["status"] == "failed" and result["attempted_run_count"] == 0
    assert result["unattempted_run_count"] == 24 and len(tracker["calls"]) == 32
    assert not native.output_paths(args["output"])["forecast"].exists()


def test_changed_runtime_is_rejected_before_new_forecasts(qualification):
    args, spec, tracker = qualification
    rows = read_rows(args["reference_ledger"])
    rows[0]["runtime"]["torch_intraop_threads"] += 1
    args["reference_ledger"].write_text("".join(json.dumps(r)+"\n" for r in rows))
    spec["reference_ledger_sha256"] = native._hash(args["reference_ledger"])
    audit = json.loads(args["reference_audit"].read_text())
    audit["ledger_sha256"] = spec["reference_ledger_sha256"]
    args["reference_audit"].write_text(json.dumps(audit))
    spec["reference_audit_sha256"] = native._hash(args["reference_audit"])
    replan(args, spec)
    result = native.run(**args)
    assert result["status"] == "failed" and result["attempted_run_count"] == 0
    assert len(tracker["calls"]) == 32
    assert "runtime" in read_rows(args["output"])[1]["error_message"]


@pytest.mark.parametrize("fault", ["low", "unknown"])
def test_resource_failure_retains_full_unattempted_denominator(qualification, monkeypatch, fault):
    args, _, tracker = qualification
    def bad_observation():
        if fault == "unknown": raise OSError("software API failure")
        return 0
    monkeypatch.setattr(native.physical_memory, "available_physical_bytes", bad_observation)
    result = native.run(**args)
    assert result["status"] == "failed" and result["resource_stopped"]
    assert result["unattempted_run_count"] == 24 and result["attempted_run_count"] == 0
    assert result["observer"]["failed_calls"] == int(fault == "unknown") and len(tracker["calls"]) == 32


def test_engine_failure_is_retained_and_not_salvaged_as_smaller_complete_family(qualification, monkeypatch):
    args, _, _ = qualification
    def fail(*a, **k): raise MemoryError("software worker memory stop")
    monkeypatch.setattr(native.engine, "rollout", fail)
    result = native.run(**args)
    assert result["status"] == "failed" and result["resource_stopped"]
    assert result["attempted_run_count"] == result["failure_count"] == 1
    assert result["success_count"] == 0 and result["unattempted_run_count"] == 23
    assert result["engine_terminal_error_count"] == 1
    audit = json.loads(native.output_paths(args["output"])["audit"].read_text())
    assert audit["status"] == "failed" and audit["counts"]["unattempted_run_count"] == 23


@pytest.mark.parametrize("fault", ["source", "plan"])
def test_changed_sources_or_plan_fail_even_after_successful_predictions(qualification, monkeypatch, fault):
    args, _, _ = qualification
    sources, calls = native.source_hashes(), []
    if fault == "source":
        def changed():
            calls.append(1)
            return sources if len(calls) == 1 else {**sources, "native_qualification.py": "f"*64}
        monkeypatch.setattr(native, "source_hashes", changed)
    else:
        def changed(_):
            args["plan"].write_bytes(args["plan"].read_bytes()+b"\n")
        args["progress"] = changed
    result = native.run(**args)
    assert result["status"] == "failed" and result["success_count"] == 24 and result["terminal_error_count"] >= 1
    assert not result["certified"]


@pytest.mark.parametrize("outcome", ["complete", "failure", "timeout", "interrupt"])
def test_supervisor_owns_exact_worker_and_preserves_timeout_or_interrupt(qualification, monkeypatch, outcome):
    args, spec, _ = qualification
    launched, stopped = [], []
    class Child:
        pid = 900001
        code = None
        def poll(self): return self.code
        def wait(self, timeout):
            assert 0 < timeout <= spec["outer_wall_seconds"]
            if outcome == "timeout": raise native.subprocess.TimeoutExpired("software-worker", timeout)
            if outcome == "interrupt": raise KeyboardInterrupt()
            self.code = 0 if outcome == "complete" else 1
            return self.code
    child = Child()
    def launch(command, **kwargs):
        reserved = json.loads(native.output_paths(args["output"])["supervisor"].read_text())
        assert reserved["status"] == "running" and reserved["plan_sha256"] == args["plan_sha256"]
        with pytest.raises(FileExistsError):
            native.supervise(args, ["duplicate-software-worker"])
        launched.append((command, kwargs))
        return child
    def stop(process):
        assert process is child and process.poll() is None
        stopped.append(process.pid)
        process.code = 1
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(native.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(native.subprocess, "Popen", launch)
    monkeypatch.setattr(native, "terminate_owned", stop)
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt): native.supervise(args, ["software-arguments"])
    else:
        assert native.supervise(args, ["software-arguments"]) == {"complete": 0, "failure": 1, "timeout": 124}[outcome]
    assert launched[0][0][-2:] == ["--worker", "software-arguments"]
    assert launched[0][1]["creationflags"] == 0x08000000
    assert stopped == ([900001] if outcome in {"timeout", "interrupt"} else [])
    record = json.loads(native.output_paths(args["output"])["supervisor"].read_text())
    assert record["worker_pid"] == 900001 and record["termination_confirmed"]
    assert record["timed_out"] is (outcome == "timeout") and not record["certified"]
    assert not args["output"].exists()  # Supervisor never invents a worker completion.


def test_supervised_worker_requires_matching_reservation(qualification):
    args, _, tracker = qualification
    path = native.output_paths(args["output"])["supervisor"]
    path.write_text(json.dumps({"schema_version": native.VERSION+"-supervisor", "status": "running", "plan_sha256": "f"*64}))
    with pytest.raises(ValueError, match="reservation"):
        native.run(**args, _supervised=True)
    assert len(tracker["calls"]) == 32 and not args["output"].exists()


def test_supervised_worker_preserves_its_reservation_until_parent_finishes(qualification):
    args, _, _ = qualification
    path = native.output_paths(args["output"])["supervisor"]
    marker = {"schema_version": native.VERSION+"-supervisor", "status": "running", "plan_sha256": args["plan_sha256"]}
    path.write_text(json.dumps(marker))
    result = native.run(**args, _supervised=True)
    assert result["status"] == "complete" and json.loads(path.read_text()) == marker


def test_supervisor_subtracts_process_startup_from_fixed_outer_cap(qualification, monkeypatch):
    args, spec, _ = qualification
    measured, values = [], iter([0., 30., 31.])
    child = SimpleNamespace(pid=900003, poll=lambda: 0, wait=lambda timeout: measured.append(timeout))
    monkeypatch.setattr(native, "time", SimpleNamespace(perf_counter=lambda: next(values)))
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(native.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(native.subprocess, "Popen", lambda *a, **k: child)
    assert native.supervise(args, ["software-arguments"]) == 0
    assert measured == [spec["outer_wall_seconds"]-30.]
    report = json.loads(native.output_paths(args["output"])["supervisor"].read_text())
    assert report["elapsed_seconds"] == 31. and report["deadline_at"]


def test_unconfirmed_termination_is_reported_not_falsely_closed(qualification, monkeypatch):
    args, _, _ = qualification
    class Child:
        pid = 900004
        def poll(self): return None
        def wait(self, timeout): raise native.subprocess.TimeoutExpired("software-worker", timeout)
    def cannot_stop(child): raise RuntimeError("software unconfirmed stop")
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(native.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(native.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(native, "terminate_owned", cannot_stop)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        native.supervise(args, ["software-arguments"])
    record = json.loads(native.output_paths(args["output"])["supervisor"].read_text())
    assert record["timed_out"] and not record["termination_confirmed"] and record["worker_returncode"] is None
    assert record["termination_error"] and not args["output"].exists()


def test_tree_termination_never_targets_a_closed_handle(monkeypatch):
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **k: pytest.fail("closed handle must not be killed"))
    native.terminate_owned(SimpleNamespace(poll=lambda: 0))


def test_tree_termination_targets_only_the_owned_live_pid(monkeypatch):
    calls = []
    child = SimpleNamespace(pid=900002, code=None)
    child.poll = lambda: child.code
    child.wait = lambda timeout: child.code
    def taskkill(command, **kwargs):
        calls.append(command)
        child.code = 1
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(native.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(native.subprocess, "run", taskkill)
    native.terminate_owned(child)
    assert calls == [["taskkill", "/PID", "900002", "/T", "/F"]] and child.poll() == 1


@pytest.mark.skipif(native.os.name != "nt", reason="Windows process supervisor")
def test_real_windows_supervisor_runs_worker_and_rejects_bad_bound_reference(qualification, monkeypatch):
    args, spec, _ = qualification
    spec["reference_ledger_sha256"] = "0"*64
    replan(args, spec)
    monkeypatch.chdir(Path(native.__file__).parents[2])
    arguments = [value for key, item in args.items() for value in ("--"+key.replace("_", "-"), str(item))]
    assert native.supervise(args, arguments) == 1
    rows = read_rows(args["output"])
    assert rows[-1]["status"] == "failed" and rows[-1]["attempted_run_count"] == 0
    assert rows[-1]["unattempted_run_count"] == 24
    record = json.loads(native.output_paths(args["output"])["supervisor"].read_text())
    assert record["termination_confirmed"] and not record["timed_out"] and record["worker_returncode"] == 1


@pytest.mark.skipif(native.os.name != "nt", reason="Windows process-tree timeout")
def test_real_windows_supervisor_times_out_only_its_harmless_sleeping_child(qualification, monkeypatch):
    args, spec, _ = qualification
    spec.update(wall_seconds=1., outer_wall_seconds=3)
    replan(args, spec)
    original_popen, created = native.subprocess.Popen, []
    def spawn(command, **kwargs):
        # A real, bounded harmless child stands in for a hung worker. The real
        # taskkill invocation is forwarded unchanged, not mocked or broadened.
        if len(command) > 3 and command[3] == "experiments.pirc17.native_qualification":
            child = original_popen([native.sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
            created.append(child)
            return child
        return original_popen(command, **kwargs)
    monkeypatch.setattr(native.subprocess, "Popen", spawn)
    try:
        assert native.supervise(args, ["software-sleep-substitute"]) == 124
        assert len(created) == 1 and created[0].poll() is not None
        record = json.loads(native.output_paths(args["output"])["supervisor"].read_text())
        assert record["timed_out"] and record["termination_confirmed"]
        assert record["worker_pid"] == created[0].pid and not args["output"].exists()
    finally:
        for child in created:
            if child.poll() is None:
                native.terminate_owned(child)
