"""Software-only evidence adapter, complete family and equal-block oracles."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.pirc17 import candidate_power as power
from experiments.pirc17 import direct_linear_rollout as engine
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows


def build(args, evidence, *, groups=None, overrides=None, stem="power"):
    groups = groups or (power.SEEDS[:2], power.SEEDS[2:])
    ledgers, audits = [], []
    for index, seeds in enumerate(groups):
        path = args["output"].with_name(f"{stem}-{index}.jsonl")
        options = dict(args, output=path, seeds=list(seeds), particles=[4], steps=[5.],
            configurations=sorted({name for pair in power.PRIMARY_FAMILY.values() for name in pair}))
        options.update((overrides or {}).get(index, {}))
        assert engine.run(**options)["status"] == "complete"
        audit = path.with_suffix(".audit.json")
        audit.write_text(json.dumps(power.audited(read_rows(path), path.parent, power._hash(path), 1., evidence)))
        ledgers.append(path)
        audits.append(audit)
    return dict(**evidence, ledgers=ledgers, ledger_sha256=[power._hash(p) for p in ledgers],
        audits=audits, audit_sha256=[power._hash(p) for p in audits], step_seconds=5., particles=4,
        delta_m=4., planning_block_counts=[30, 62], output=args["output"].with_name(f"{stem}-result.json"))


@pytest.fixture
def power_inputs(inputs, evidence):
    args, tracker = inputs
    return build(args, evidence), tracker


def rewrite(args, index, rows, evidence):
    path = args["ledgers"][index]
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    args["ledger_sha256"][index] = power._hash(path)
    audit = args["audits"][index]
    audit.write_text(json.dumps(power.audited(rows, path.parent, power._hash(path), 1., evidence)))
    args["audit_sha256"][index] = power._hash(audit)


def check_oracle(result, args):
    rows = [row for path in args["ledgers"] for row in read_rows(path)[2:-1]
        if row["max_step_seconds"] == args["step_seconds"] and row["particles"] == args["particles"]]
    planning = result["planning"]
    blocks = sorted({r["independent_block_id"] for r in rows})
    for contrast, (candidate, control) in power.PRIMARY_FAMILY.items():
        block_seed = []
        for block in blocks:
            by_seed = []
            for seed in power.SEEDS:
                part = [r for r in rows if r["independent_block_id"] == block and r["seed"] == seed]
                model_means = {name: sum(r["scores"]["time_weighted_energy_score_m"] for r in part
                    if r["configuration"] == name)/sum(r["configuration"] == name for r in part)
                    for name in (candidate, control)}
                by_seed.append(model_means[candidate]-model_means[control])
            block_seed.append(by_seed)
        values = np.array(block_seed)
        report = planning["results"][contrast]
        assert report["paired_block_sd_m"] == pytest.approx(values.mean(axis=1).std(ddof=1))
        assert report["observed_development_improvement_m"] == pytest.approx(-values.mean())
        assert report["seed_mean_improvement_m"] == pytest.approx(-values.mean(axis=0))
        assert all(s["assumed_true_improvement_m"] == 2*args["delta_m"] for s in report["planning_scenarios"])
        assert [(s["planning_blocks"], s["sd_multiplier"]) for s in report["planning_scenarios"]] == [
            (30, 1.), (30, 2.), (62, 1.), (62, 2.)]


def test_complete_candidate_sources_keep_five_seeds_and_five_contrasts_without_new_forecasts(power_inputs):
    args, tracker = power_inputs
    saved = {p: p.read_bytes() for p in [*args["ledgers"], *args["audits"], args["fit"], args["fit_ledger"]]}
    result = power.run(**args)
    assert result["status"] == "complete" and not result["certified"] and not result["formal_training_accepted"]
    assert result["final_eval_label_prediction_metric_reads"] == 0
    planning = result["planning"]
    assert planning["source_independent_blocks"] == planning["source_origin_count"] == 2
    assert planning["registered_seeds"] == list(power.SEEDS) and planning["primary_family_size"] == 5
    assert set(planning["results"]) == set(power.PRIMARY_FAMILY) and not planning["missing_primary_comparisons"]
    assert planning["source_attempted_runs"] == [24, 36]
    assert [r["success_count"] for r in result["source_counts"]] == [24, 36]
    assert all(r["checked"] == 0 and r["available_sensitivity_axes"] == [] and not r["numerically_qualified"]
        for r in result["source_numerical"])  # Empty sensitivity is never convergence.
    assert "distinct fitted parameters" in result["seed_interpretation"]
    assert len(tracker["calls"]) == 60 and all(p.read_bytes() == data for p, data in saved.items())
    assert json.loads(args["output"].read_text()) == result
    check_oracle(result, args)


def test_unequal_origins_in_two_blocks_are_not_promoted_to_independent_samples(inputs, evidence):
    original, tracker = inputs
    original.update(selection_policy="lexical_origins", limit_origins=5)
    args = build(original, evidence)
    result = power.run(**args)
    assert result["planning"]["source_origin_count"] == 5
    assert result["planning"]["source_independent_blocks"] == 2
    assert result["planning"]["origin_count_by_block"] == [4, 1] and len(tracker["calls"]) == 150
    check_oracle(result, args)


def test_split_seed_batches_and_one_whole_batch_give_identical_planning(power_inputs, inputs, evidence):
    args, _ = power_inputs
    split = power.run(**args)
    combined_args = build(inputs[0], evidence, groups=(power.SEEDS,), stem="combined")
    whole = power.run(**combined_args)
    assert split["planning"]["results"] == whole["planning"]["results"]
    assert split["planning"]["independent_block_ids"] == whole["planning"]["independent_block_ids"]
    assert split["planning"]["source_attempted_runs"] == [24, 36]
    assert whole["planning"]["source_attempted_runs"] == [60]


def test_mixed_resolution_batches_preserve_every_unselected_numerical_failure(inputs, evidence):
    args = build(inputs[0], evidence, overrides={0: {"particles": [4, 8]}})
    result = power.run(**args)
    assert [r["success_count"] for r in result["source_counts"]] == [48, 36]
    first = result["source_numerical"][0]
    assert first["available_sensitivity_axes"] == ["particle_count"] and first["checked"] == 24
    assert first["out_of_tolerance"] > 0 and first["tolerance_m"] == 1. and not first["numerically_qualified"]
    check_oracle(result, args)


def test_complete_one_seed_pilot_cannot_become_five_seed_power(inputs, evidence):
    args = build(inputs[0], evidence, groups=(power.SEEDS[:1],))
    with pytest.raises(ValueError, match="all five fixed"):
        power.run(**args)
    assert not args["output"].exists()


def test_missing_origin_seed_pairs_cannot_be_reduced_to_a_complete_intersection(inputs, evidence):
    args = build(inputs[0], evidence, overrides={0: {"limit_origins": 1}})
    with pytest.raises(ValueError, match="complete matched origin/seed"):
        power.run(**args)
    assert not args["output"].exists()


def test_missing_primary_comparison_cannot_shrink_the_family(inputs, evidence):
    args = build(inputs[0], evidence, overrides={0: {"configurations": ["base", "all-terrain"]}})
    with pytest.raises(ValueError, match="all five primary"):
        power.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("field,value", [("delta_m", 0.), ("delta_m", float("nan")), ("delta_m", True),
    ("step_seconds", 0.), ("step_seconds", True), ("step_seconds", float("inf")),
    ("particles", 2), ("particles", True), ("planning_block_counts", []),
    ("planning_block_counts", [30, 30]), ("planning_block_counts", [True])])
def test_invalid_parameters_fail_before_reading_any_evidence(tmp_path, field, value):
    args = dict(ledgers=[Path("absent")], ledger_sha256=["0"*64], audits=[Path("absent")], audit_sha256=["0"*64],
        fit=Path("absent"), fit_sha256="0"*64, fit_ledger=Path("absent"), fit_ledger_sha256="0"*64,
        training_policy_sha256="0"*64, step_seconds=5., particles=4, delta_m=4.,
        planning_block_counts=[30, 62], output=tmp_path/"not-created.json")
    args[field] = value
    with pytest.raises(ValueError, match="explicit finite"): power.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("field", ["ledgers", "ledger_sha256", "audits", "audit_sha256"])
def test_incomplete_input_binding_lists_fail_before_calculation(power_inputs, field):
    args, _ = power_inputs
    args[field].pop()
    with pytest.raises(ValueError, match="both hashes"): power.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("fault", ["partial", "missing", "failure", "changed_runtime", "changed_audit"])
def test_incomplete_or_changed_evidence_is_never_silently_filtered(power_inputs, evidence, fault):
    args, _ = power_inputs
    rows = read_rows(args["ledgers"][1])
    if fault == "partial": rows.pop()
    if fault == "missing": rows.pop(-2)
    if fault in {"partial", "missing"}:
        path = args["ledgers"][1]
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        args["ledger_sha256"][1] = power._hash(path)
        report = json.loads(args["audits"][1].read_text())
        report["ledger_sha256"] = args["ledger_sha256"][1]
        args["audits"][1].write_text(json.dumps(report))
        args["audit_sha256"][1] = power._hash(args["audits"][1])
    if fault == "failure":
        rows[-2].update(status="failure", error_type="ValueError", error_message="software fixture failure")
        rows[-1].update(status="failed", success_count=35, failure_count=1)
        rewrite(args, 1, rows, evidence)
    if fault == "changed_runtime":
        rows[0]["runtime"]["numpy"] = "changed"
        rewrite(args, 1, rows, evidence)
    if fault == "changed_audit":
        report = json.loads(args["audits"][1].read_text())
        report["particle_precision"]["runs"].pop()
        args["audits"][1].write_text(json.dumps(report))
        args["audit_sha256"][1] = power._hash(args["audits"][1])
    with pytest.raises(ValueError): power.run(**args)
    assert not args["output"].exists()


def test_duplicate_complete_batches_are_not_extra_evidence(power_inputs):
    args, _ = power_inputs
    for key in ("ledgers", "ledger_sha256", "audits", "audit_sha256"): args[key].append(args[key][0])
    with pytest.raises(ValueError, match="overlapping workload"): power.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("field,value", [("delta_m", 8.), ("step_seconds", 2.5), ("particles", 8)])
def test_planning_cannot_override_source_tolerance_or_invent_missing_settings(power_inputs, field, value):
    args, _ = power_inputs
    args[field] = value
    with pytest.raises(ValueError): power.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("field", ["ledger_sha256", "audit_sha256", "fit_sha256", "fit_ledger_sha256", "training_policy_sha256"])
def test_wrong_source_bindings_are_rejected(power_inputs, field):
    args, _ = power_inputs
    if isinstance(args[field], list): args[field][0] = "f"*64
    else: args[field] = "f"*64
    with pytest.raises(ValueError): power.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("fault", ["array", "source", "fit"])
def test_sources_and_particles_are_revalidated_after_planning(power_inputs, monkeypatch, fault):
    args, _ = power_inputs
    original = power.calibrate
    def changed(*a, **k):
        result = original(*a, **k)
        if fault == "source": monkeypatch.setattr(power, "source_hashes", lambda: {"changed": "f"*64})
        elif fault == "fit": args["fit"].write_bytes(args["fit"].read_bytes()+b"\n")
        else:
            row = read_rows(args["ledgers"][1])[-2]
            path = args["ledgers"][1].parent/row["particle_artifact"]["path"]
            path.write_bytes(path.read_bytes()+b"software corruption")
        return result
    monkeypatch.setattr(power, "calibrate", changed)
    with pytest.raises(ValueError): power.run(**args)
    assert not args["output"].exists()


def test_existing_result_is_not_overwritten(power_inputs):
    args, _ = power_inputs
    args["output"].write_text("owned marker")
    with pytest.raises(FileExistsError): power.run(**args)
    assert args["output"].read_text() == "owned marker"


@pytest.mark.parametrize("option", ["--comparisons", "--tolerance-m"])
def test_cli_refuses_family_or_tolerance_override(power_inputs, monkeypatch, option):
    args, _ = power_inputs
    command = []
    for key, value in args.items():
        if key == "planning_block_counts": command.extend(["--planning-blocks", *map(str, value)])
        elif isinstance(value, list):
            flag = "--"+{"ledgers": "ledger", "audits": "audit"}.get(key, key.replace("_", "-"))
            for item in value: command.extend([flag, str(item)])
        else: command.extend(["--"+key.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", ["candidate_power", *command, option, "all-vs-base"])
    with pytest.raises(SystemExit) as caught: power.main()
    assert caught.value.code == 2 and not args["output"].exists()
