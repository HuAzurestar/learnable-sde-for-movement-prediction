"""Software-only complete-ledger overlap/extension evidence tests."""
import json
from pathlib import Path
import sys

import pytest

from experiments.pirc17 import candidate_extension_check as extension
from experiments.pirc17 import direct_linear_rollout as engine
from experiments.pirc17.inference import PRIMARY_FAMILY
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows


def write_audit(path, evidence, tolerance=1e-12):
    report = extension.audited(read_rows(path), path.parent, extension._hash(path), tolerance, evidence)
    audit = path.with_suffix(".audit.json")
    audit.write_text(json.dumps(report), encoding="utf-8")
    return audit


@pytest.fixture
def extension_inputs(inputs, evidence):
    args, tracker = inputs
    assert engine.run(**args)["status"] == "complete"
    reference = args["output"]
    reference_audit = write_audit(reference, evidence)
    candidate = reference.with_name("extended.jsonl")
    new = dict(args, output=candidate, particles=[8, 16], wall_seconds=420.,
        configurations=sorted({name for pair in PRIMARY_FAMILY.values() for name in pair}))
    assert engine.run(**new)["status"] == "complete"
    audit = write_audit(candidate, evidence)
    return dict(**evidence, ledger=candidate, ledger_sha256=extension._hash(candidate),
        audit=audit, audit_sha256=extension._hash(audit), reference_ledger=reference,
        reference_ledger_sha256=extension._hash(reference), reference_audit=reference_audit,
        reference_audit_sha256=extension._hash(reference_audit), output=reference.with_name("overlap.json")), tracker


def compare_args(args):
    return (read_rows(args["ledger"]), read_rows(args["reference_ledger"]),
        args["ledger"].parent, args["reference_ledger"].parent)


def replace_ledger(args, key, rows):
    path = args[key]
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    args[key+"_sha256"] = extension._hash(path)
    audit_key = "audit" if key == "ledger" else "reference_audit"
    report = json.loads(args[audit_key].read_text(encoding="utf-8"))
    report["ledger_sha256"] = args[key+"_sha256"]
    args[audit_key].write_text(json.dumps(report), encoding="utf-8")
    args[audit_key+"_sha256"] = extension._hash(args[audit_key])


def test_complete_intersection_checks_every_old_budget_and_retains_full_new_denominator(extension_inputs):
    args, tracker = extension_inputs
    bound = {key: Path(args[key]).read_bytes() for key in
        ("ledger", "audit", "reference_ledger", "reference_audit", "fit", "fit_ledger")}
    result = extension.run(**args)
    assert result["passed"] and result["status"] == "complete"
    assert result["candidate_counts"]["success_count"] == 96 and result["reference_counts"]["success_count"] == 32
    assert result["reference_runs_covered"] == 32 and result["new_configuration_run_count"] == 64
    assert result["comparison_count"] == 64 and result["same_budget_overlap_count"] == 16
    assert result["strict_prefix_comparison_count"] == 48 and len(result["added_configurations"]) == 4
    assert result["reference_particle_budgets"] == [4, 8] and result["candidate_particle_budgets"] == [8, 16]
    assert result["reference_wall_budget_seconds"] == 300. and result["candidate_wall_budget_seconds"] == 420.
    assert result["reference_numerical"]["out_of_tolerance"] > 0 and result["candidate_numerical"]["out_of_tolerance"] > 0
    assert result["reference_numerical"]["tolerance_m"] == result["candidate_numerical"]["tolerance_m"] == 1e-12
    assert all(r["passed"] and r["max_prefix_coordinate_difference_m"] == 0 for r in result["comparisons"])
    assert not result["certified"] and not result["formal_training_accepted"] and result["final_eval_label_prediction_metric_reads"] == 0
    assert len(tracker["calls"]) == 128  # Auditing cannot launch additional forecasts.
    assert all(Path(args[key]).read_bytes() == value for key, value in bound.items())
    assert json.loads(args["output"].read_text()) == result


@pytest.mark.parametrize("fault", ["runtime", "seed", "origin", "step", "source", "map_source", "model",
    "weights", "history", "unknown_header", "lost_model", "lost_overlap", "no_growth", "driver", "block"])
def test_only_declared_configuration_budget_and_wall_changes_are_allowed(extension_inputs, fault):
    args, _ = extension_inputs
    new, old, folder, old_folder = compare_args(args)
    if fault == "runtime": new[0]["runtime"]["numpy"] = "different"
    if fault == "seed": new[1]["seeds"] = [1]
    if fault == "origin": new[1]["sample_ids"][0] = "different-origin"
    if fault == "step": new[1]["max_steps_seconds"] = [10., 5.]
    if fault == "source": new[1]["source_sha256"]["rollout.py"] = "f"*64
    if fault == "map_source": new[1]["map_source_sha256"]["extra"] = "f"*64
    if fault == "model": new[1]["model_identities"]["base"][str(engine.SEEDS[0])] = "f"*64
    if fault == "weights": new[1]["time_weights"] = [0., 0., 0., 1.]
    if fault == "history": new[1]["physical_history_step_seconds"] = 10.
    if fault == "unknown_header": new[1]["unrecognized_policy"] = "changed"
    if fault == "lost_model": new[1]["configurations"].remove("base")
    if fault == "lost_overlap": new[1]["particle_counts"] = [16, 32]
    if fault == "no_growth": new[1]["particle_counts"] = [8]
    if fault == "driver": new[2]["brownian_identity"]["time_grid_sha256"] = "f"*64
    if fault == "block": new[2]["independent_block_id"] = "other-block"
    with pytest.raises(ValueError): extension._compare(new, old, folder, old_folder)


@pytest.mark.parametrize("side", ["ledger", "reference_ledger"])
def test_incomplete_original_denominator_cannot_be_salvaged(extension_inputs, side):
    args, _ = extension_inputs
    rows = read_rows(args[side])
    # Remove a run, not the completion or registered denominator. For the new
    # ledger this is a new LOO model outside the old-model intersection.
    rows.pop(-2)
    replace_ledger(args, side, rows)
    with pytest.raises(ValueError): extension.run(**args)
    assert not args["output"].exists()


def test_running_ledger_without_terminal_completion_is_rejected(extension_inputs):
    args, _ = extension_inputs
    replace_ledger(args, "ledger", read_rows(args["ledger"])[:-1])
    with pytest.raises(ValueError, match="terminal completion"):
        extension.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("field", ["ledger_sha256", "audit_sha256", "reference_ledger_sha256", "reference_audit_sha256",
    "fit_sha256", "fit_ledger_sha256", "training_policy_sha256"])
def test_wrong_binding_never_produces_evidence(extension_inputs, field):
    args, _ = extension_inputs
    args[field] = "f"*64
    with pytest.raises(ValueError): extension.run(**args)
    assert not args["output"].exists()


def test_numerical_failure_deletion_from_rehashed_audit_is_detected(extension_inputs):
    args, _ = extension_inputs
    report = json.loads(args["reference_audit"].read_text())
    report["numerical_audit"]["sensitivities"].pop()
    args["reference_audit"].write_text(json.dumps(report))
    args["reference_audit_sha256"] = extension._hash(args["reference_audit"])
    with pytest.raises(ValueError, match="exactly reproduce"): extension.run(**args)
    assert not args["output"].exists()


def test_changed_tolerance_is_rejected_even_if_both_audits_reproduce(extension_inputs, evidence):
    args, _ = extension_inputs
    write_audit(args["ledger"], evidence, tolerance=1.)
    args["audit_sha256"] = extension._hash(args["audit"])
    with pytest.raises(ValueError, match="original numerical tolerance"): extension.run(**args)
    assert not args["output"].exists()


def test_failed_new_configuration_is_not_hidden_by_successful_old_model_intersection(extension_inputs, evidence):
    args, _ = extension_inputs
    rows = read_rows(args["ledger"])
    assert rows[-2]["configuration"] not in read_rows(args["reference_ledger"])[1]["configurations"]
    rows[-2].update(status="failure", error_type="ValueError", error_message="software failure")
    rows[-1].update(status="failed", success_count=95, failure_count=1)
    replace_ledger(args, "ledger", rows)
    write_audit(args["ledger"], evidence)
    args["audit_sha256"] = extension._hash(args["audit"])
    with pytest.raises(ValueError, match="whole complete"): extension.run(**args)
    assert not args["output"].exists()


def test_same_budget_metadata_difference_is_a_failed_comparison(extension_inputs):
    args, _ = extension_inputs
    new, old, folder, old_folder = compare_args(args)
    new[2]["feature_query_rows"] += 1
    result = extension._compare(new, old, folder, old_folder)
    assert not result["passed"]
    failed = [r for r in result["comparisons"] if not r["passed"]]
    assert len(failed) == 1 and failed[0]["same_budget_overlap"]
    assert failed[0]["same_budget_other_run_fields_exact"] is False


@pytest.mark.parametrize("array", [0, 1, 2])
def test_prefix_target_or_time_mismatch_is_retained_as_failure(extension_inputs, monkeypatch, array):
    args, _ = extension_inputs
    new, old, folder, old_folder = compare_args(args)
    real_load = extension.load_particle_evidence
    target_row = new[2]
    def changed(row, directory):
        values = [a.copy() for a in real_load(row, directory)]
        if row is target_row: values[array].flat[0] += 1.
        return values
    monkeypatch.setattr(extension, "load_particle_evidence", changed)
    result = extension._compare(new, old, folder, old_folder)
    assert not result["passed"] and result["comparison_count"] == 64
    assert sum(not r["passed"] for r in result["comparisons"]) == 2


@pytest.mark.parametrize("fault", ["new_model_array", "source", "fit"])
def test_post_calculation_revalidation_covers_nonoverlapping_arrays_and_sources(extension_inputs, monkeypatch, fault):
    args, _ = extension_inputs
    original = extension._compare
    def changed(*items):
        result = original(*items)
        if fault == "source":
            monkeypatch.setattr(extension, "source_hashes", lambda: {"changed": "f"*64})
        elif fault == "fit":
            args["fit"].write_bytes(args["fit"].read_bytes()+b"\n")
        else:
            row = read_rows(args["ledger"])[-2]
            assert row["configuration"] not in read_rows(args["reference_ledger"])[1]["configurations"]
            path = args["ledger"].parent/row["particle_artifact"]["path"]
            path.write_bytes(path.read_bytes()+b"software corruption")
        return result
    monkeypatch.setattr(extension, "_compare", changed)
    with pytest.raises(ValueError): extension.run(**args)
    assert not args["output"].exists()


def test_existing_output_cannot_be_overwritten(extension_inputs):
    args, _ = extension_inputs
    args["output"].write_text("owned marker")
    with pytest.raises(FileExistsError): extension.run(**args)
    assert args["output"].read_text() == "owned marker"


def test_cli_has_no_tolerance_or_subset_override(extension_inputs, monkeypatch):
    args, tracker = extension_inputs
    command = [value for key, item in args.items() for value in ("--"+key.replace("_", "-"), str(item))]
    monkeypatch.setattr(sys, "argv", ["candidate_extension_check", *command, "--tolerance-m", "1000"])
    with pytest.raises(SystemExit) as caught: extension.main()
    assert caught.value.code == 2 and len(tracker["calls"]) == 128 and not args["output"].exists()
