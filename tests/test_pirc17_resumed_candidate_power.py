"""Mixed complete/resumed evidence; software fixtures, no empirical forecasts."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from experiments.pirc17 import resumed_candidate_power as power
from tests.test_pirc17_candidate_power import check_oracle
from tests.test_pirc17_seed_resume_lineage import (
    candidate_bytes, evidence, inputs, qualification, seed_qualification, closed_seed, resume_input,
    reserve, current_args, seal_software, interrupt_after, execution, lineage)


def options(resume_args, reservation, seed_args, seed_spec, evidence):
    native = execution.native
    last_args = dict(seed_args, plan=seed_args["plan"].with_name("last-seeds-plan.json"),
                    output=seed_args["output"].with_name("last-seeds.jsonl"))
    last_spec = dict(seed_spec, seeds=list(power.candidate.SEEDS[3:]))
    last_args["plan"].write_text(json.dumps(last_spec), encoding="utf-8")
    last_args["plan_sha256"] = native._hash(last_args["plan"])
    assert native.run(**last_args)["status"] == "complete"
    final = native.output_paths(last_args["output"])
    ledgers = [seed_args["reference_ledger"], final["forecast"]]
    audits = [seed_args["reference_audit"], final["audit"]]
    closed = lineage.history(reservation["directory"], reservation["root_sha256"])[1]
    args = dict(**evidence, ledgers=ledgers, ledger_sha256=[native._hash(p) for p in ledgers],
        audits=audits, audit_sha256=[native._hash(p) for p in audits],
        resume_directories=[reservation["directory"]], resume_root_sha256=[reservation["root_sha256"]],
        resume_closure_sha256=[closed[-1]["closure_sha256"]], step_seconds=2.5, particles=8,
        delta_m=4*seed_spec["tolerance_m"], planning_block_counts=[30, 62],
        output=resume_args["output"].with_name("resumed-power.json"))
    old_complete = native.output_paths(seed_args["output"])
    baseline = {key: value for key, value in args.items() if not key.startswith("resume_")}
    baseline.update(ledgers=[ledgers[0], old_complete["forecast"], ledgers[1]],
        audits=[audits[0], old_complete["audit"], audits[1]], output=args["output"].with_name("full-power.json"))
    baseline["ledger_sha256"] = [native._hash(p) for p in baseline["ledgers"]]
    baseline["audit_sha256"] = [native._hash(p) for p in baseline["audits"]]
    return args, baseline


@pytest.fixture
def mixed(resume_input, seed_qualification, evidence):
    resume_args, old_paths, _, _, tracker = resume_input
    first = reserve(resume_args)
    with pytest.raises(KeyboardInterrupt):
        execution.run(**current_args(resume_args, first), progress=interrupt_after(3))
    seal_software(first, interrupted=True, returncode=130)
    last = reserve(resume_args)
    assert execution.run(**current_args(resume_args, last))["status"] == "computed_pending_supervisory_closure"
    seal_software(last, interrupted=False, returncode=0)
    args, baseline = options(resume_args, last, *seed_qualification[:2], evidence)
    return args, baseline, tracker, old_paths, first, last


def test_repeated_resume_gives_exact_whole_five_seed_planning_without_forecasting(mixed):
    args, baseline, tracker, old_paths, first, last = mixed
    protected = {p: p.read_bytes() for p in old_paths["envelope"].parent.rglob("*") if p.is_file()}
    calls = len(tracker["calls"])
    result = power.run(**args)
    expected = power.candidate.run(**baseline)
    assert result["planning"]["results"] == expected["planning"]["results"]
    assert result["planning"]["source_independent_blocks"] == result["planning"]["source_origin_count"] == 2
    assert result["planning"]["registered_seeds"] == list(power.candidate.SEEDS)
    assert result["planning"]["source_attempted_runs"] == [48, 24, 24]
    assert set(result["planning"]["results"]) == set(power.candidate.PRIMARY_FAMILY)
    check_oracle(result, baseline)
    assert len(tracker["calls"]) == calls
    assert all(p.read_bytes() == data for p, data in protected.items())
    assert not list(first["directory"].rglob("*.jsonl"))
    assert json.loads(args["output"].read_text()) == result
    assert result["source_numerical"][0]["out_of_tolerance"] > 0
    assert result["source_numerical"][2]["checked"] == 0
    assert result["source_numerical"][2]["available_sensitivity_axes"] == []
    assert all(not result[key] for key in ("certified", "numerically_qualified", "resource_certified", "formal_training_accepted"))
    inspection = result["input_sources"][2]["inspection"]
    assert inspection["attempt_count"] == 2 and inspection["legacy_inherited_count"] == 14
    assert inspection["new_committed_count"] == 10 and inspection["total_charged_elapsed_seconds"] > 180
    assert inspection["historical_resource_gaps"] == {"inner_completion": None, "maps": None, "memory_summary": None}
    assert inspection["reference_numerical"]["out_of_tolerance"] > 0
    assert result["final_eval_label_prediction_metric_reads"] == 0


def test_live_or_incomplete_chain_cannot_enter_planning(resume_input, seed_qualification, evidence):
    args, _, _, _, tracker = resume_input
    first = reserve(args)
    with pytest.raises(KeyboardInterrupt):
        execution.run(**current_args(args, first), progress=interrupt_after(3))
    seal_software(first, interrupted=True, returncode=130)
    planning, _ = options(args, first, *seed_qualification[:2], evidence)
    calls = len(tracker["calls"])
    with pytest.raises(ValueError, match="complete supervisory closure"):
        power.run(**planning)
    reserve(args)
    with pytest.raises(FileExistsError, match="unclosed"):
        power.run(**planning)
    assert len(tracker["calls"]) == calls and not planning["output"].exists()


def test_bound_hashes_and_full_grid_fail_closed(mixed):
    original, baseline, _, _, _, _ = mixed
    for field in ("resume_root_sha256", "resume_closure_sha256", "ledger_sha256", "audit_sha256",
                  "fit_sha256", "fit_ledger_sha256", "training_policy_sha256"):
        args = deepcopy(original)
        if isinstance(args[field], list):
            args[field][0] = "f"*64
        else:
            args[field] = "f"*64
        with pytest.raises(ValueError):
            power.run(**args)
        assert not args["output"].exists()
    for fields in (("ledgers", "ledger_sha256", "audits", "audit_sha256"),
                   ("resume_directories", "resume_root_sha256", "resume_closure_sha256")):
        args = deepcopy(original)
        for field in fields:
            args[field].append(args[field][0])
        with pytest.raises(ValueError, match="overlapping workload"):
            power.run(**args)
    args = deepcopy(original)
    for field in ("ledgers", "ledger_sha256", "audits", "audit_sha256"):
        args[field].insert(1, baseline[field][1])
    with pytest.raises(ValueError, match="overlapping workload"):
        power.run(**args)
    args = deepcopy(original)
    for field in ("ledgers", "ledger_sha256", "audits", "audit_sha256"):
        args[field].pop()
    with pytest.raises(ValueError, match="all five fixed"):
        power.run(**args)
    assert not original["output"].exists()


def test_corruption_of_any_ancestor_or_materialization_is_rejected(mixed):
    args, _, _, old, first, last = mixed
    files = [old["forecast"], first["attempt"]/"closure.json",
             first["attempt"]/"work/journal/records/000014.json",
             first["attempt"]/"work/journal/particles/000014.npz",
             last["attempt"]/"work/inherited.json", last["attempt"]/"work/whole-math.json"]
    for path in files:
        saved = path.read_bytes()
        try:
            path.write_bytes(saved+b"corruption")
            with pytest.raises(ValueError):
                power.run(**args)
            assert not args["output"].exists()
        finally:
            path.write_bytes(saved)


@pytest.mark.parametrize("fault", ["ancestor", "reference_particle", "complete_particle", "fit", "source"])
def test_mutation_during_planning_is_detected_before_publication(mixed, monkeypatch, fault):
    args, _, _, old, first, _ = mixed
    original_calibrate = power.candidate.calibrate
    original_sources = power.source_hashes
    modified = []
    def changed(*a, **k):
        result = original_calibrate(*a, **k)
        if fault == "source":
            monkeypatch.setattr(power, "source_hashes", lambda: {**original_sources(), "changed": "f"*64})
            return result
        if fault == "ancestor":
            path = first["attempt"]/"work/journal/particles/000014.npz"
        elif fault == "fit":
            path = args["fit"]
        else:
            ledger = args["ledgers"][0 if fault == "reference_particle" else 1]
            rows = json.loads(ledger.read_text().splitlines()[2])
            path = ledger.parent/rows["particle_artifact"]["path"]
        modified.append((path, path.read_bytes()))
        path.write_bytes(path.read_bytes()+b"changed")
        return result
    monkeypatch.setattr(power.candidate, "calibrate", changed)
    try:
        with pytest.raises(ValueError):
            power.run(**args)
        assert not args["output"].exists()
    finally:
        for path, data in modified:
            path.write_bytes(data)


def test_outputs_cannot_overwrite_or_mutate_the_resume_chain(mixed):
    args, _, _, _, first, _ = mixed
    with pytest.raises(ValueError, match="immutable resume directory"):
        power.run(**dict(args, output=first["directory"]/"new-report.json"))
    args["output"].write_text("preserved marker")
    with pytest.raises(FileExistsError):
        power.run(**args)
    assert args["output"].read_text() == "preserved marker"


def test_registered_tolerance_and_numerical_settings_cannot_be_overridden(mixed):
    args, _, _, _, _, _ = mixed
    for field, value in (("delta_m", args["delta_m"]*2), ("step_seconds", 5.), ("particles", 4)):
        with pytest.raises(ValueError):
            power.run(**dict(args, **{field: value}))
        assert not args["output"].exists()


def test_complete_source_contract_cannot_shrink_or_change_identity(mixed, inputs, evidence):
    args, _, _, _, _, _ = mixed
    base, tracker = inputs
    # These are independently generated complete SOFTWARE batches, not edited
    # empirical rows or a narrowed intersection of the full matched grid.
    for index, override in enumerate(({"limit_origins": 1}, {"configurations": ["base", "all-terrain"]})):
        ledger = base["output"].with_name(f"smaller-source-{index}.jsonl")
        options = dict(base, output=ledger, seeds=list(power.candidate.SEEDS[3:]),
            configurations=sorted({name for pair in power.candidate.PRIMARY_FAMILY.values() for name in pair}),
            particles=[8], steps=[2.5])
        options.update(override)
        assert execution.engine.run(**options)["status"] == "complete"
        audit = ledger.with_suffix(".audit.json")
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        audit.write_text(json.dumps(power.candidate.audited(rows, ledger.parent,
            power.candidate._hash(ledger), args["delta_m"]/4, evidence)), encoding="utf-8")
        changed = deepcopy(args)
        changed["ledgers"][1], changed["audits"][1] = ledger, audit
        changed["ledger_sha256"][1], changed["audit_sha256"][1] = map(power.candidate._hash, (ledger, audit))
        with pytest.raises(ValueError, match="complete matched origin/seed|all five primary"):
            power.run(**changed)
        assert not args["output"].exists()
    ledger, audit = args["ledgers"][1], args["audits"][1]
    saved = {path: path.read_bytes() for path in (ledger, audit)}
    try:
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        rows[0]["runtime"]["numpy"] = "other-runtime"
        ledger.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        digest = power.candidate._hash(ledger)
        audit.write_text(json.dumps(power.candidate.audited(rows, ledger.parent, digest,
            args["delta_m"]/4, evidence)), encoding="utf-8")
        changed = deepcopy(args)
        changed["ledger_sha256"][1], changed["audit_sha256"][1] = digest, power.candidate._hash(audit)
        calls = len(tracker["calls"])
        with pytest.raises(ValueError, match="mix candidate, runtime"):
            power.run(**changed)
        assert len(tracker["calls"]) == calls and not args["output"].exists()
    finally:
        for path, data in saved.items():
            path.write_bytes(data)


def test_invalid_planning_or_bindings_rejected_before_reading_evidence(tmp_path):
    args = dict(ledgers=[], ledger_sha256=[], audits=[], audit_sha256=[], resume_directories=[tmp_path/"absent"],
        resume_root_sha256=["f"*64], resume_closure_sha256=["f"*64], fit=tmp_path/"absent-fit",
        fit_sha256="f"*64, fit_ledger=tmp_path/"absent-fit-ledger", fit_ledger_sha256="f"*64,
        training_policy_sha256="f"*64, step_seconds=2.5, particles=8, delta_m=4e-12,
        planning_block_counts=[30, 62], output=tmp_path/"absent-output.json")
    for field, value in (("delta_m", True), ("delta_m", float("nan")), ("delta_m", 0),
            ("step_seconds", False), ("step_seconds", float("inf")), ("particles", 2), ("particles", True),
            ("planning_block_counts", []), ("planning_block_counts", [30, 30]), ("planning_block_counts", [True])):
        with pytest.raises(ValueError, match="explicit finite"):
            power.run(**dict(args, **{field: value}))
    for field in ("resume_directories", "resume_root_sha256", "resume_closure_sha256"):
        with pytest.raises(ValueError, match="whole resumed source"):
            power.run(**dict(args, **{field: []}))
    with pytest.raises(ValueError, match="both hashes"):
        power.run(**dict(args, ledgers=[tmp_path/"absent"]))
    assert not args["output"].exists()


def test_cli_binds_full_sources_and_refuses_scientific_overrides(mixed, monkeypatch, capsys):
    args, _, _, _, _, _ = mixed
    flags = {"ledgers": "ledger", "audits": "audit", "resume_directories": "resume-directory",
             "planning_block_counts": "planning-blocks"}
    command = []
    for key, value in args.items():
        flag = "--"+flags.get(key, key.replace("_", "-"))
        if key == "planning_block_counts":
            command.extend([flag, *map(str, value)])
        elif isinstance(value, list):
            for item in value:
                command.extend([flag, str(item)])
        else:
            command.extend([flag, str(value)])
    for option in ("--comparisons", "--tolerance-m", "--allow-partial", "--wall-seconds"):
        monkeypatch.setattr(sys, "argv", ["resumed-power", *command, option, "1"])
        with pytest.raises(SystemExit) as caught:
            power.main()
        assert caught.value.code == 2 and not args["output"].exists()
    monkeypatch.setattr(sys, "argv", ["resumed-power", *command])
    assert power.main() == 0
    assert json.loads(capsys.readouterr().out)["source_independent_blocks"] == 2
