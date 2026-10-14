"""Software-only seed-plan isolation, complete evidence and existing power integration."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.pirc17 import candidate_power, native_qualification as native
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows
from tests.test_pirc17_native_qualification import qualification, replan


@pytest.fixture
def seed_spec():
    return {"schema_version": native.SEED_PLAN_VERSION,
        "purpose": native.PLAN_PURPOSES[native.SEED_PLAN_VERSION],
        "configurations": sorted({c for pair in native.PRIMARY_FAMILY.values() for c in pair}),
        "seeds": list(native.engine.SEEDS[1:3]), "particles": [8], "steps": [2.5], "limit_origins": 2,
        "selection_policy": "lexical_independent_blocks", "map_backend": "multicell",
        "wall_seconds": 300., "outer_wall_seconds": 420, "minimum_free_bytes": native.engine.MINIMUM_FREE_BYTES,
        "tolerance_m": 1e-12, "expected_run_count": 24,
        "certified": False, "formal_training_accepted": False, "final_eval_authorized": False,
        **{key: "a"*64 for key in native.DIGESTS}}


def checked_plan(tmp_path, spec):
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return native.load_plan(path, native._hash(path))


@pytest.mark.parametrize("batch", [slice(1, 3), slice(3, 5)])
def test_exact_two_disjoint_seed_batches_are_valid(tmp_path, seed_spec, batch):
    seed_spec["seeds"] = list(native.engine.SEEDS[batch])
    assert checked_plan(tmp_path, seed_spec) == seed_spec


def test_registered_plans_keep_one_reference_and_complete_disjoint_five_seed_union():
    directory = Path(native.__file__).parent/"plans"
    plans = []
    for labels in ("20260815-16", "20260817-18"):
        path = directory/f"native-primary-seeds-{labels}-p512-v1.json"
        plans.append(native.load_plan(path, native._hash(path)))
    assert {k: v for k, v in plans[0].items() if k != "seeds"} == {
        k: v for k, v in plans[1].items() if k != "seeds"}
    assert not set(plans[0]["seeds"]) & set(plans[1]["seeds"])
    assert [native.engine.SEEDS[0]]+plans[0]["seeds"]+plans[1]["seeds"] == list(native.engine.SEEDS)
    assert plans[0]["particles"] == [512] and plans[0]["steps"] == [0.3125]
    assert plans[0]["limit_origins"] == 3 and plans[0]["expected_run_count"] == 36
    assert plans[0]["tolerance_m"] == 10.27506475


@pytest.mark.parametrize("fault", ["old_schema", "wrong_purpose", "unknown_schema", "one_seed", "first_seed",
    "mixed_batch", "reversed", "all_seeds", "duplicate_seed", "two_budgets", "two_steps", "missing_model",
    "denominator", "final", "qualified", "unknown_field"])
def test_seed_profile_cannot_reduce_or_relabel_the_registered_contract(tmp_path, seed_spec, fault):
    if fault == "old_schema": seed_spec["schema_version"] = native.PLAN_VERSION
    if fault == "wrong_purpose": seed_spec["purpose"] = native.PLAN_PURPOSES[native.PLAN_VERSION]
    if fault == "unknown_schema": seed_spec["schema_version"] = "unknown"
    if fault == "one_seed": seed_spec["seeds"] = [native.engine.SEEDS[1]]
    if fault == "first_seed": seed_spec["seeds"] = list(native.engine.SEEDS[:2])
    if fault == "mixed_batch": seed_spec["seeds"] = [native.engine.SEEDS[1], native.engine.SEEDS[3]]
    if fault == "reversed": seed_spec["seeds"].reverse()
    if fault == "all_seeds": seed_spec["seeds"] = list(native.engine.SEEDS)
    if fault == "duplicate_seed": seed_spec["seeds"] = [native.engine.SEEDS[1]]*2
    if fault == "two_budgets": seed_spec["particles"] = [4, 8]
    if fault == "two_steps": seed_spec["steps"] = [5., 2.5]
    if fault == "missing_model": seed_spec["configurations"].remove("loo-river")
    if fault == "denominator": seed_spec["expected_run_count"] = 12
    if fault == "final": seed_spec["final_eval_authorized"] = True
    if fault == "qualified": seed_spec["certified"] = True
    if fault == "unknown_field": seed_spec["ignore_reference_failures"] = True
    with pytest.raises(ValueError): checked_plan(tmp_path, seed_spec)


@pytest.fixture
def reference_contract(seed_spec):
    init = {"limit_origins": 2, "selection_policy": "lexical_independent_blocks", "map_backend": "multicell"}
    header = {"seeds": [native.engine.SEEDS[0]], "particle_counts": [4, 8], "max_steps_seconds": [5., 2.5],
        "configurations": seed_spec["configurations"].copy(), "sample_ids": ["software-a", "software-b"]}
    report = {"numerical_audit": {"tolerance_m": seed_spec["tolerance_m"],
        "sensitivities": [{"all_scoring_times_within_tolerance": False}]}}
    return [init, header], report


@pytest.mark.parametrize("fault", ["reference_seeds", "reference_budgets", "reference_steps", "model_order",
    "origin_count", "policy", "backend", "smaller_budget", "larger_budget", "coarser_step", "finer_step"])
def test_seed_reference_contract_cannot_choose_an_easier_setting(seed_spec, reference_contract, fault):
    reference, report = reference_contract
    if fault == "reference_seeds": reference[1]["seeds"] = list(native.engine.SEEDS[:2])
    if fault == "reference_budgets": reference[1]["particle_counts"] = [8]
    if fault == "reference_steps": reference[1]["max_steps_seconds"] = [2.5]
    if fault == "model_order": seed_spec["configurations"].reverse()
    if fault == "origin_count": seed_spec["limit_origins"] = 1
    if fault == "policy": seed_spec["selection_policy"] = "lexical_origins"
    if fault == "backend": seed_spec["map_backend"] = "batched"
    if fault == "smaller_budget": seed_spec["particles"] = [4]
    if fault == "larger_budget": seed_spec["particles"] = [16]
    if fault == "coarser_step": seed_spec["steps"] = [5.]
    if fault == "finer_step": seed_spec["steps"] = [1.25]
    with pytest.raises(ValueError, match="whole first-seed"):
        native.seed_reference_contract(seed_spec, reference, report)


def test_seed_contract_reports_reference_failures_without_qualification(seed_spec, reference_contract):
    reference, report = reference_contract
    contract = native.seed_reference_contract(seed_spec, reference, report)
    assert contract["reference_numerical"]["out_of_tolerance"] == 1
    assert contract["required_union_seeds"] == list(native.engine.SEEDS)
    assert not contract["numerically_qualified"]


@pytest.mark.parametrize("changed", ["sample_ids", "model_identities", "runtime", "map_source_sha256", "scoring_grid"])
def test_seed_result_rejects_changed_reference_identity(seed_spec, reference_contract, changed):
    reference, _ = reference_contract
    candidate = deepcopy(reference)
    candidate[1].update(seeds=seed_spec["seeds"], particle_counts=seed_spec["particles"],
        max_steps_seconds=seed_spec["steps"], expected_run_count=seed_spec["expected_run_count"])
    candidate[1][changed] = "software-identity-mutation"
    with pytest.raises(ValueError, match="cohort, model, runtime"):
        native.validate_seed_result(seed_spec, reference, candidate)


@pytest.mark.parametrize("changed,value", [("seeds", [20260818]), ("particle_counts", [16]),
    ("max_steps_seconds", [1.25]), ("expected_run_count", 12)])
def test_seed_result_requires_exact_registered_axes(seed_spec, reference_contract, changed, value):
    reference, _ = reference_contract
    candidate = deepcopy(reference)
    candidate[1].update(seeds=seed_spec["seeds"], particle_counts=seed_spec["particles"],
        max_steps_seconds=seed_spec["steps"], expected_run_count=seed_spec["expected_run_count"])
    candidate[1][changed] = value
    with pytest.raises(ValueError, match="exact registered workload"):
        native.validate_seed_result(seed_spec, reference, candidate)


@pytest.fixture
def seed_qualification(qualification):
    args, numerical, tracker = qualification
    numerical.update(limit_origins=2, expected_run_count=48)
    replan(args, numerical)
    assert native.run(**args)["status"] == "complete"
    paths = native.output_paths(args["output"])
    seed = dict(numerical, schema_version=native.SEED_PLAN_VERSION,
        purpose=native.PLAN_PURPOSES[native.SEED_PLAN_VERSION], seeds=list(native.engine.SEEDS[1:3]),
        particles=[8], steps=[2.5], expected_run_count=24,
        reference_ledger_sha256=native._hash(paths["forecast"]), reference_audit_sha256=native._hash(paths["audit"]))
    updated = dict(args, plan=args["plan"].with_name("seed-plan.json"), reference_ledger=paths["forecast"],
        reference_audit=paths["audit"], output=args["output"].with_name("seed-planning.jsonl"))
    replan(updated, seed)
    return updated, seed, tracker


def test_complete_seed_batch_keeps_whole_reference_and_empty_sensitivity_unqualified(seed_qualification):
    args, spec, _ = seed_qualification
    original = args["reference_ledger"].read_bytes()
    result = native.run(**args)
    assert result["status"] == "complete" and result["success_count"] == 24
    assert result["candidate_numerical"]["checked"] == 0
    assert not result["numerically_qualified"] and not result["certified"]
    assert result["seed_planning_contract"]["reference_numerical"]["out_of_tolerance"] > 0
    assert args["reference_ledger"].read_bytes() == original
    rows = read_rows(args["output"])
    assert rows[0]["schema_version"] == native.VERSION and rows[0]["purpose"] == spec["purpose"]
    assert rows[1]["seed_planning_contract"] == result["seed_planning_contract"]
    report = json.loads(native.output_paths(args["output"])["audit"].read_text())
    assert report["numerical_audit"]["sensitivities"] == []
    assert len(report["particle_precision"]["paired_comparisons"]) == 20


def test_both_whole_batches_feed_existing_five_seed_planner(seed_qualification, evidence):
    args, spec, _ = seed_qualification
    ledgers, audits = [args["reference_ledger"]], [args["reference_audit"]]
    for index, batch in enumerate((native.engine.SEEDS[1:3], native.engine.SEEDS[3:5])):
        attempt = dict(args, plan=args["plan"].with_name(f"batch-{index}.json"),
            output=args["output"].with_name(f"batch-{index}.jsonl"))
        plan = dict(spec, seeds=list(batch))
        replan(attempt, plan)
        assert native.run(**attempt)["status"] == "complete"
        paths = native.output_paths(attempt["output"])
        ledgers.append(paths["forecast"])
        audits.append(paths["audit"])
    result = candidate_power.run(ledgers=ledgers, ledger_sha256=[native._hash(p) for p in ledgers],
        audits=audits, audit_sha256=[native._hash(p) for p in audits], **evidence,
        step_seconds=2.5, particles=8, delta_m=spec["tolerance_m"]*4, planning_block_counts=[30, 62],
        output=args["output"].with_name("software-power.json"))
    assert result["status"] == "complete" and result["planning"]["source_independent_blocks"] == 2
    assert len(result["planning"]["results"]) == 5
    assert [r["expected_run_count"] for r in result["source_counts"]] == [48, 24, 24]
    assert result["source_numerical"][0]["out_of_tolerance"] > 0
    assert all(not r["numerically_qualified"] for r in result["source_numerical"])
    assert [r["checked"] for r in result["source_numerical"]] == [48, 0, 0]


def test_seed_batch_failure_keeps_the_entire_registered_denominator(seed_qualification, monkeypatch):
    args, _, _ = seed_qualification
    def fail(*a, **k): raise MemoryError("software seed-batch stop")
    monkeypatch.setattr(native.engine, "rollout", fail)
    result = native.run(**args)
    assert result["status"] == "failed" and result["resource_stopped"]
    assert result["expected_run_count"] == 24 and result["failure_count"] == 1
    assert result["unattempted_run_count"] == 23 and not result["numerically_qualified"]


def test_seed_plan_cannot_relax_the_original_tolerance(seed_qualification):
    args, spec, tracker = seed_qualification
    before = len(tracker["calls"])
    spec["tolerance_m"] *= 2
    replan(args, spec)
    result = native.run(**args)
    assert result["status"] == "failed" and result["attempted_run_count"] == 0
    assert result["unattempted_run_count"] == 24 and len(tracker["calls"]) == before


@pytest.mark.parametrize("fault", ["targets", "times", "block", "horizons"])
def test_seed_targets_are_checked_against_the_complete_reference(seed_qualification, monkeypatch, fault):
    args, _, _ = seed_qualification
    original = native.load_particle_evidence
    def altered(row, directory):
        positions, targets, times = original(row, directory)
        if row["seed"] != native.engine.SEEDS[0]:
            if fault == "targets": targets = targets+1
            if fault == "times": times = times+1
            if fault == "block": row["independent_block_id"] = "different-software-block"
            if fault == "horizons": row["actual_horizons_seconds"] = [t+1 for t in row["actual_horizons_seconds"]]
        return positions, targets, times
    monkeypatch.setattr(native, "load_particle_evidence", altered)
    result = native.run(**args)
    assert result["status"] == "failed" and result["success_count"] == 24
    assert result["terminal_error_count"] == 1 and not result["numerically_qualified"]
    assert "observed targets" in read_rows(args["output"])[-2]["error_message"]
