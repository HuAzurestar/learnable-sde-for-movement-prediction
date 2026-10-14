"""Software-only, closed-prefix import tests; no empirical forecasts."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from experiments.pirc17 import seed_resume as resume
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows
from tests.test_pirc17_native_qualification import qualification
from tests.test_pirc17_native_seed_planning import seed_qualification

native = resume.native


def write_json(path, value, *, jsonl=False):
    path.write_text("".join(json.dumps(row)+"\n" for row in value) if jsonl else json.dumps(value), encoding="utf-8")


@pytest.fixture
def closed_seed(seed_qualification):
    args, spec, tracker = seed_qualification
    assert native.run(**args)["status"] == "complete"
    full = native.output_paths(args["output"])
    original = read_rows(full["forecast"])
    paths = native.output_paths(args["output"].with_name("interrupted.jsonl"))
    paths["particles"].mkdir()
    prefix = deepcopy(original[:16])  # 14/24: one whole origin plus two rows.
    for row in prefix[2:]:
        relative = row["particle_artifact"]["path"]
        dest = paths["particles"]/Path(relative).name
        shutil.copyfile(full["forecast"].parent/relative, dest)
        row["particle_artifact"]["path"] = dest.relative_to(paths["forecast"].parent).as_posix()
    write_json(paths["forecast"], prefix, jsonl=True)
    outer = read_rows(full["envelope"])[:2]
    outer[0].update(forecast_path=paths["forecast"].name, audit_path=paths["audit"].name)
    write_json(paths["envelope"], outer, jsonl=True)
    supervisor = {"schema_version": native.VERSION+"-supervisor", "plan_sha256": args["plan_sha256"],
        "status": "failed", "supervisor_pid": 1, "worker_pid": 2, "worker_returncode": 1,
        "termination_confirmed": True, "timed_out": False, "outer_wall_seconds": 420,
        "started_at": "2026-09-28T00:00:00+00:00", "deadline_at": "2026-09-28T00:07:00+00:00",
        "finished_at": "2026-09-28T00:03:00+00:00", "elapsed_seconds": 180.,
        "certified": False, "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0}
    write_json(paths["supervisor"], supervisor)
    inspect_args = {key: args[key] for key in ("plan", "plan_sha256", "reference_ledger", "reference_audit", "fit", "fit_ledger")}
    inspect_args.update(envelope=paths["envelope"], envelope_sha256=native._hash(paths["envelope"]),
        forecast_sha256=native._hash(paths["forecast"]), supervisor_sha256=native._hash(paths["supervisor"]), guard=lambda: None)
    return inspect_args, paths, prefix, original, tracker


def prefix_check(case, candidate):
    args, paths, _, _, _ = case
    spec = native.load_plan(args["plan"], args["plan_sha256"])
    return resume.validate_prefix_rows(spec, read_rows(args["reference_ledger"]), candidate,
        paths["forecast"].parent, args["reference_ledger"].parent, guard=lambda: None)


def test_closed_partial_admission_replays_but_never_predicts_writes_or_certifies(closed_seed):
    args, paths, prefix, _, tracker = closed_seed
    before = {p: p.read_bytes() for p in paths["envelope"].parent.rglob("*") if p.is_file()}
    calls = len(tracker["calls"])
    result = resume.inspect(**args)
    report = result.report
    assert report["status"] == "validated_prefix_only"
    assert report["expected_run_count"] == 24 and report["reusable_success_count"] == 14
    assert report["missing_completed_record_count"] == len(result.remaining_keys) == 10
    assert result.remaining_keys[0] == ("validation-004", "loo-river", 20260815, 8, 2.5)
    assert result.rows == prefix[2:] and len(tracker["calls"]) == calls
    assert report["reference_numerical"]["out_of_tolerance"] > 0
    assert report["inherited_inner_completion"] is report["inherited_final_maps_identity"] is None
    assert report["inherited_final_memory_summary"] is None
    assert not report["numerically_qualified"] and not report["certified"] and not report["formal_training_accepted"]
    assert report["final_eval_label_prediction_metric_reads"] == 0 and not report["uncommitted_particle_files"]
    assert report["interrupted_supervisor"]["status"] == "failed"
    assert {p: p.read_bytes() for p in paths["envelope"].parent.rglob("*") if p.is_file()} == before


def test_every_canonical_cut_keeps_full_denominator_and_origin_seed_order(closed_seed):
    _, _, prefix, _, _ = closed_seed
    for cut in (0, 1, 6, 7, 12, 13, 14):
        candidate = prefix[:2+cut]
        keys = prefix_check(closed_seed, candidate)
        assert len(keys) == 24
        assert [tuple(row[k] for k in resume.KEYS) for row in candidate[2:]] == keys[:cut]


def test_rehashed_row_and_header_tampering_cannot_be_admitted(closed_seed):
    _, _, prefix, _, _ = closed_seed
    faults = ("duplicate", "missing", "order", "origin", "seed", "particles", "step", "model", "block", "times",
        "scores", "precision", "path_hash", "path_escape", "brownian", "grid", "runtime", "source", "fit",
        "init_seed", "init_count", "header_float", "wall", "feature_count", "failed", "failure_event", "completion", "error_field", "extra_field")
    for fault in faults:
        candidate = deepcopy(prefix)
        row = candidate[2]
        if fault == "duplicate": candidate[3] = deepcopy(row)
        if fault == "missing": candidate.pop(2)
        if fault == "order": candidate[2], candidate[3] = candidate[3], candidate[2]
        if fault == "origin": row["sample_id"] = "another-origin"
        if fault == "seed": row["seed"] = float(row["seed"])
        if fault == "particles": row["particles"] = float(row["particles"])
        if fault == "step": row["max_step_seconds"] = True
        if fault == "model": row["model_identity_sha256"] = "f"*64
        if fault == "block": row["independent_block_id"] = "another-block"
        if fault == "times": row["actual_horizons_seconds"][0] += 1
        if fault == "scores": row["scores"]["fde_m"] += 1
        if fault == "precision": row["particle_precision"]["time_weighted_standard_error_m"] += 1
        if fault == "path_hash": row["particle_artifact"]["sha256"] = "f"*64
        if fault == "path_escape": row["particle_artifact"]["path"] = "../outside.npz"
        if fault == "brownian": row["brownian_identity"]["path_sha256"] = "f"*64
        if fault == "grid": row["brownian_identity"]["time_grid_sha256"] = "f"*64
        if fault == "runtime": candidate[0]["runtime"]["torch_intraop_threads"] += 1
        if fault == "source": candidate[1]["source_sha256"]["rollout.py"] = "f"*64
        if fault == "fit": candidate[0]["fit_sha256"] = "f"*64
        if fault == "init_seed": candidate[0]["seeds"] = [20260817, 20260818]
        if fault == "init_count": candidate[0]["expected_run_count"] -= 1
        if fault == "header_float": candidate[1]["particle_counts"] = [8.]
        if fault == "wall": row["rollout_wall_seconds_including_lazy_map_initialization"] = float("nan")
        if fault == "feature_count": row["invalid_feature_rows"] = True
        if fault == "failed": row["status"] = "failure"
        if fault == "failure_event": candidate.append({"type": "failure", "error_type": "TimeoutError"})
        if fault == "completion": candidate.append({"type": "completion", "status": "complete"})
        if fault == "error_field": row["error_type"] = "ValueError"
        if fault == "extra_field": row["certified"] = True
        with pytest.raises((ValueError, FileNotFoundError), match="."):
            prefix_check(closed_seed, candidate)


def test_rehashed_particle_targets_are_compared_to_full_reference(closed_seed):
    _, paths, prefix, _, _ = closed_seed
    row = prefix[2]
    path = paths["forecast"].parent/row["particle_artifact"]["path"]
    x, y, t = resume.load_particle_evidence(row, paths["forecast"].parent)
    with path.open("wb") as target:
        resume.engine.np.savez_compressed(target, positions_m=x, target_positions_m=y+1, elapsed_seconds=t)
    row["particle_artifact"]["sha256"] = native._hash(path)
    with pytest.raises(ValueError, match="targets or times"):
        prefix_check(closed_seed, prefix)


def test_live_or_unproven_supervisor_and_modified_envelope_are_rejected(closed_seed):
    args, paths, _, _, _ = closed_seed
    supervisor = json.loads(paths["supervisor"].read_text())
    for key, value in (("status", "running"), ("termination_confirmed", False), ("worker_returncode", None),
                       ("worker_returncode", 0), ("worker_returncode", True), ("worker_pid", None),
                       ("deadline_at", "2026-09-28T00:08:00+00:00"), ("elapsed_seconds", -1),
                       ("outer_wall_seconds", 999), ("plan_sha256", "f"*64), ("certified", True)):
        changed = dict(supervisor, **{key: value})
        write_json(paths["supervisor"], changed)
        with pytest.raises(ValueError):
            resume.inspect(**dict(args, supervisor_sha256=native._hash(paths["supervisor"])))
    write_json(paths["supervisor"], supervisor)
    outer = read_rows(paths["envelope"])
    for fault in ("sources", "plan", "observer", "forecast", "numerical", "extra"):
        changed = deepcopy(outer)
        if fault == "sources": changed[0]["source_sha256"]["rollout.py"] = "f"*64
        if fault == "plan": changed[0]["plan"]["particles"] = [1024]
        if fault == "observer": changed[0]["observer"]["measurement_cache"] = True
        if fault == "forecast": changed[0]["forecast_path"] = "other.jsonl"
        if fault == "numerical": changed[1]["numerical"]["out_of_tolerance"] = 0
        if fault == "extra": changed.append({"type": "completion", "status": "complete"})
        write_json(paths["envelope"], changed, jsonl=True)
        with pytest.raises(ValueError):
            resume.inspect(**dict(args, envelope_sha256=native._hash(paths["envelope"])))


def test_orphan_is_retained_and_disclosed_never_reused(closed_seed):
    args, paths, _, _, _ = closed_seed
    orphan = paths["particles"]/"uncommitted.npz"
    orphan.write_bytes(b"software torn particle save")
    result = resume.inspect(**args)
    assert result.report["reusable_success_count"] == 14
    assert result.report["uncommitted_particle_files"] == [{"path": orphan.name, "sha256": native._hash(orphan)}]
    assert orphan.read_bytes() == b"software torn particle save"


def test_resource_guard_aborts_without_forecasts_or_rewriting(closed_seed):
    args, paths, _, _, tracker = closed_seed
    calls = len(tracker["calls"])
    original = paths["forecast"].read_bytes()
    def stop():
        raise MemoryError("software low memory")
    with pytest.raises(MemoryError):
        resume.inspect(**dict(args, guard=stop))
    assert len(tracker["calls"]) == calls and paths["forecast"].read_bytes() == original


@pytest.mark.parametrize("payload,jsonl", [(b'{"x":1,"x":2}\n', True), (b'{"x":NaN}\n', True),
    (b'{"x":Infinity}\n', True), (b'{"x":1}', True), (b'{"x":1}\n{', True),
    (b'{"x":1}\n\n', True), (b'{"x":1,"x":2}', False)])
def test_strict_closed_reader_rejects_torn_ambiguous_and_nonfinite_json(tmp_path, payload, jsonl):
    path = tmp_path/"software.jsonl"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        resume.read_closed(path, native._hash(path), jsonl=jsonl)


def test_closed_reader_requires_exact_hash(tmp_path):
    path = tmp_path/"software.jsonl"
    path.write_bytes(b'{"type":"run"}\n')
    for digest in ("", "F"*64, "f"*64, None):
        with pytest.raises(ValueError):
            resume.read_closed(path, digest, jsonl=True)


def test_sources_changing_during_validation_are_rejected(closed_seed, monkeypatch):
    args, _, _, _, _ = closed_seed
    original, count = resume.source_hashes(), []
    def changed():
        count.append(1)
        return original if len(count) == 1 else {**original, "seed_resume.py": "f"*64}
    monkeypatch.setattr(resume, "source_hashes", changed)
    with pytest.raises(ValueError, match="changed during"):
        resume.inspect(**args)
