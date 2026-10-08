"""Software equivalence of skipped-prefix execution, not empirical qualification."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.pirc17 import seed_resume_execution as execution
from experiments.pirc17.rollout import rollout as real_rollout
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows
from tests.test_pirc17_native_qualification import qualification
from tests.test_pirc17_native_seed_planning import seed_qualification
from tests.test_pirc17_seed_resume import closed_seed


@pytest.fixture
def resume_input(closed_seed, seed_qualification):
    source, old_paths, prefix, complete, tracker = closed_seed
    original = seed_qualification[0]
    args = {key: original[key] for key in ("eligibility", "release", "snapshot", "data_root")}
    args.update(source={key: value for key, value in source.items() if key != "guard"},
                output=old_paths["envelope"].parent/"resumed-core")
    return args, old_paths, prefix, complete, tracker


def assembled(args):
    output = args["output"]
    inherited = json.loads((output/"inherited.json").read_text())["rows"]
    result = json.loads((output/"core-result.json").read_text())
    saved = execution.journal.read(output/"journal", result["journal_tip"]["manifest_sha256"], expected_tip=result["journal_tip"])
    rows = deepcopy(inherited)
    for record in saved.records:
        row = deepcopy(record["row"])
        if "particle_artifact" in row:
            row["particle_artifact"]["path"] = "journal/"+row["particle_artifact"]["path"]
        rows.append(row)
    return rows, saved


def test_only_missing_ten_are_predicted_and_all_twenty_four_match_full_run(resume_input):
    args, old_paths, prefix, complete, tracker = resume_input
    protected = {path: path.read_bytes() for path in old_paths["envelope"].parent.rglob("*") if path.is_file()}
    calls = len(tracker["calls"])
    result = execution.run(**args)
    assert result["status"] == "computed_pending_supervisory_closure", result
    assert result["expected_run_count"] == 24 and result["inherited_success_count"] == 14
    assert result["new_committed_count"] == len(tracker["calls"])-calls == 10
    assert result["new_failure_count"] == result["missing_completed_record_count"] == 0
    assert not result["production_resume_ready"] and not result["resources"]["outer_cap_enforced"]
    assert result["resources"]["cooperative_remaining_seconds"] == 120
    assert result["resources"]["outer_remaining_seconds"] == 240
    assert result["observer"]["calls"] > 0 and not result["observer"]["failed_calls"]
    assert tracker["maps"][-1].closed
    rows, saved = assembled(args)
    assert saved.manifest["ancestry"]["inherited_rows"] == prefix[2:]
    assert len(rows) == 24 and len(saved.records) == 10
    for row, baseline in zip(rows, complete[2:-1]):
        first = execution.load_particle_evidence(row, args["output"])
        second = execution.load_particle_evidence(baseline, old_paths["envelope"].parent)
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
        for key in ("scores", "particle_precision", "brownian_identity", "model_identity_sha256",
                    "invalid_feature_rows", "feature_query_rows", "independent_block_id", "actual_horizons_seconds"):
            assert row[key] == baseline[key]
    audit = json.loads((args["output"]/"whole-math.json").read_text())
    assert audit["expected_run_count"] == 24
    assert len(audit["particle_precision"]["runs"]) == 24
    assert len(audit["particle_precision"]["paired_comparisons"]) == 20
    assert not audit["numerical_audit"]["sensitivities"] and not audit["numerically_qualified"]
    assert audit["reference_numerical"]["out_of_tolerance"] > 0
    assert audit["historical_resource_gaps"] == {"inner_completion": None, "maps": None, "memory_summary": None}
    assert not list(args["output"].glob("*.jsonl"))  # No counterfeit old-engine ledger.
    assert all(path.read_bytes() == data for path, data in protected.items())


def test_interruption_preserves_committed_work_and_closes_maps_without_fake_completion(resume_input):
    args, _, _, _, tracker = resume_input
    calls = len(tracker["calls"])
    def interrupt(event):
        if event["phase"] == "run_committed" and event["new_committed"] == 3:
            raise KeyboardInterrupt("software user interruption")
    with pytest.raises(KeyboardInterrupt):
        execution.run(**args, progress=interrupt)
    assert len(tracker["calls"])-calls == 3 and tracker["maps"][-1].closed
    manifest = args["output"]/"journal/manifest.json"
    saved = execution.journal.read(manifest.parent, execution.native._hash(manifest))
    assert len(saved.records) == 3 and [row["index"] for row in saved.records] == [14, 15, 16]
    assert not saved.failure_count and not saved.uncommitted_files
    assert not (args["output"]/"core-result.json").exists()
    assert not (args["output"]/"whole-math.json").exists()
    with pytest.raises(FileExistsError):
        execution.run(**args)


@pytest.mark.parametrize("exception", [ValueError, MemoryError])
def test_prediction_failure_keeps_error_and_full_denominator(resume_input, monkeypatch, exception):
    args, _, _, _, tracker = resume_input
    rollout, count = execution.engine.rollout, []
    def fail_once(*a, **k):
        count.append(1)
        if len(count) == 1:
            raise exception("software injected failure")
        return rollout(*a, **k)
    monkeypatch.setattr(execution.engine, "rollout", fail_once)
    result = execution.run(**args)
    assert result["status"] == "failed" and result["new_failure_count"] == 1
    assert result["inherited_success_count"] == 14 and result["expected_run_count"] == 24
    assert result["new_committed_count"] == (1 if exception is MemoryError else 10)
    assert result["missing_completed_record_count"] == (9 if exception is MemoryError else 0)
    assert not (args["output"]/"whole-math.json").exists() and tracker["maps"][-1].closed
    rows, _ = assembled(args)
    assert rows[14]["status"] == "failure" and rows[14]["error_type"] == exception.__name__


def test_changed_actual_targets_stop_before_any_output_or_prediction(resume_input):
    args, _, _, _, tracker = resume_input
    calls = len(tracker["calls"])
    tracker["windows"]["validation"][0].target_positions_m[:] = 123456.
    with pytest.raises(ValueError, match="targets, times or block"):
        execution.run(**args)
    assert len(tracker["calls"]) == calls and not args["output"].exists()


def test_original_budget_cannot_be_reset_by_starting_a_new_directory(resume_input):
    args, old_paths, _, _, tracker = resume_input
    calls = len(tracker["calls"])
    prior = json.loads(old_paths["supervisor"].read_text())
    prior["elapsed_seconds"] = 300.
    old_paths["supervisor"].write_text(json.dumps(prior), encoding="utf-8")
    args["source"]["supervisor_sha256"] = execution.native._hash(old_paths["supervisor"])
    with pytest.raises(TimeoutError, match="no renewal"):
        execution.run(**args)
    assert len(tracker["calls"]) == calls and not args["output"].exists()


def test_source_change_retains_successes_but_forbids_whole_math_result(resume_input, monkeypatch):
    args, _, _, _, _ = resume_input
    sources, calls = execution.source_hashes(), []
    def changed():
        calls.append(1)
        return sources if len(calls) == 1 else {**sources, "seed_resume_execution.py": "f"*64}
    monkeypatch.setattr(execution, "source_hashes", changed)
    result = execution.run(**args)
    assert result["status"] == "failed" and result["new_committed_count"] == 10
    assert not result["new_failure_count"] and not (args["output"]/"whole-math.json").exists()
    assert any("sources" in error["error_message"] for error in result["errors"])


def test_unregistered_source_seam_cannot_bypass_revalidation(resume_input):
    args, _, _, _, _ = resume_input
    args["source"]["guard"] = lambda: None
    with pytest.raises(ValueError, match="guard override"):
        execution.run(**args)
    assert not args["output"].exists()


def test_real_integrator_resumed_suffix_matches_every_uninterrupted_array(resume_input, monkeypatch):
    args, old_paths, _, _, _ = resume_input
    source = args["source"]
    spec = execution.native.load_plan(source["plan"], source["plan_sha256"])
    calls = []
    def tracked_real(*a, **k):
        calls.append((k["seed"], k["particles"], k["max_step_seconds"]))
        return real_rollout(*a, **k)
    monkeypatch.setattr(execution.engine, "rollout", tracked_real)
    baseline = args["output"].with_name("real-kernel-baseline.jsonl")
    evidence = {"fit": source["fit"], "fit_ledger": source["fit_ledger"], **{key: spec[key] for key in
        ("fit_sha256", "fit_ledger_sha256", "training_policy_sha256")}}
    result = execution.engine.run(**evidence, **{key: args[key] for key in
        ("eligibility", "release", "snapshot", "data_root")}, eligibility_sha256=spec["eligibility_sha256"],
        output=baseline, **{key: spec[key] for key in execution.native.AXES},
        available_memory=lambda: execution.engine.MINIMUM_FREE_BYTES)
    assert result["status"] == "complete" and len(calls) == 24
    complete = read_rows(baseline)
    # Replace ONLY this synthetic fixture's legacy prefix with real-kernel
    # output; no empirical artifact or production source is touched.
    prefix = deepcopy(complete[:16])
    for row in prefix[2:]:
        relative = row["particle_artifact"]["path"]
        destination = old_paths["particles"]/Path(relative).name
        shutil.copyfile(baseline.parent/relative, destination)
        row["particle_artifact"]["path"] = destination.relative_to(old_paths["forecast"].parent).as_posix()
    old_paths["forecast"].write_text("".join(json.dumps(row)+"\n" for row in prefix), encoding="utf-8")
    source["forecast_sha256"] = execution.native._hash(old_paths["forecast"])
    result = execution.run(**args)
    assert result["status"] == "computed_pending_supervisory_closure", result
    assert len(calls) == 24+10 and result["new_committed_count"] == 10
    rows, _ = assembled(args)
    for row, original in zip(rows, complete[2:-1]):
        first = execution.load_particle_evidence(row, args["output"])
        second = execution.load_particle_evidence(original, baseline.parent)
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
        assert row["scores"] == original["scores"] and row["particle_precision"] == original["particle_precision"]
        assert row["brownian_identity"] == original["brownian_identity"]


def test_runtime_change_forbids_whole_result_even_when_saved_rows_succeed(resume_input, monkeypatch):
    args, _, _, _, _ = resume_input
    original = execution.runtime_identity()
    monkeypatch.setattr(execution, "runtime_identity", lambda: {**original, "torch_intraop_threads": 999})
    result = execution.run(**args)
    assert result["status"] == "failed" and result["new_committed_count"] == 10
    assert not (args["output"]/"whole-math.json").exists()


def test_relative_input_locations_are_frozen_as_absolute_ancestry(resume_input, monkeypatch):
    args, _, _, _, _ = resume_input
    directory = args["output"].parent
    monkeypatch.chdir(directory)
    args["source"] = {key: value if key.endswith("_sha256") else str(Path(value).relative_to(directory))
                      for key, value in args["source"].items()}
    for key in ("eligibility", "release", "snapshot", "data_root"):
        args[key] = str(Path(args[key]).relative_to(directory))
    maps, modules = execution.engine.resolve_map_backend("multicell")
    class AbsoluteMaps(maps):
        def __init__(self, root, receipts):
            assert root == directory and root.is_absolute()
            super().__init__(root, receipts)
    monkeypatch.setattr(execution.engine, "resolve_map_backend", lambda _: (AbsoluteMaps, modules))
    other = directory/"other-working-directory"
    other.mkdir()
    def move_working_directory(event):
        if event["phase"] == "prefix_inherited":
            monkeypatch.chdir(other)
    result = execution.run(**args, progress=move_working_directory)
    assert result["status"] == "computed_pending_supervisory_closure"
    _, saved = assembled(args)
    assert all(Path(value).is_absolute() for key, value in saved.manifest["ancestry"]["legacy_arguments"].items()
               if not key.endswith("_sha256"))
    assert all(Path(value).is_absolute() for value in saved.manifest["contract"]["data_locations"].values())


def test_empty_stream_boundary_and_fully_saved_prefixes_skip_exactly_their_rows(resume_input):
    args, old_paths, _, complete, tracker = resume_input
    # Synthetic fixtures only: the complete baseline supplies every immutable
    # particle file; varying the closed row boundary must never change the grid.
    full = deepcopy(complete[:-1])
    for row in full[2:]:
        relative = row["particle_artifact"]["path"]
        destination = old_paths["particles"]/Path(relative).name
        shutil.copyfile(old_paths["forecast"].parent/relative, destination)
        row["particle_artifact"]["path"] = destination.relative_to(old_paths["forecast"].parent).as_posix()
    for cut in (0, 6, 12, 24):
        old_paths["forecast"].write_text("".join(json.dumps(row)+"\n" for row in full[:2+cut]), encoding="utf-8")
        source = dict(args["source"], forecast_sha256=execution.native._hash(old_paths["forecast"]))
        current = dict(args, source=source, output=args["output"].with_name(f"prefix-cut-{cut}"))
        before = len(tracker["calls"])
        result = execution.run(**current)
        assert result["status"] == "computed_pending_supervisory_closure", result
        assert result["inherited_success_count"] == cut
        assert result["new_committed_count"] == len(tracker["calls"])-before == 24-cut
        rows, _ = assembled(current)
        assert len(rows) == 24
        assert [row["scores"] for row in rows] == [row["scores"] for row in complete[2:-1]]
