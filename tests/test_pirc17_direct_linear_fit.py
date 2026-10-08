"""Complete software-only candidate fit and retained-failure ledger tests."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.pirc17 import direct_linear_fit as cli
from experiments.pirc17.direct_linear import digest, restore_direct_dynamics
from experiments.pirc17.dynamics import fit_dynamics
from tests.test_pirc17_direct_linear import rows


@pytest.fixture(scope="module")
def original_fit():
    configs = cli.terrain_configurations()
    identity = {"sha256":"a"*64, "population":"software-only"}
    models = {}
    for name, config in configs.items():
        models[name] = {str(seed):fit_dynamics(rows("train", name), rows("validation", name),
            seed=seed, training_identity=identity["sha256"], configuration_identity=config["sha256"]).identity
            for seed in cli.SEEDS}
    return {"schema_version":"pirc17-development-fit-v1",
        "purpose":"paired_pilot_checkpoints_not_final_model_acceptance", "configurations":configs,
        "models":models, "development_identity":identity, "snapshot_parent_assets":["software-parent"],
        "data_roles":{"gradient":"train", "checkpoint_selection":"validation", "final_eval_reads":0},
        "pilot_population":{"samples":{"train":24,"validation":24},
            "independent_blocks":{"train":6,"validation":6},
            "transition_rule":"one first-future transition per qualified origin; no interpolated observations"}}


@pytest.fixture
def inputs(tmp_path, monkeypatch, original_fit):
    reference = tmp_path/"reference.json"
    reference.write_text(json.dumps(original_fit, allow_nan=False), encoding="utf-8")
    windows = {r:[SimpleNamespace(role=r, block_id=b) for b in rows(r).independent_blocks]
               for r in ("train", "validation")}
    monkeypatch.setattr(cli.CanonicalEncoder, "frozen_pirc22", lambda _: SimpleNamespace())
    monkeypatch.setattr(cli, "load_development", lambda *args:
        (windows, deepcopy(original_fit["snapshot_parent_assets"]), deepcopy(original_fit["development_identity"])))
    monkeypatch.setattr(cli, "configuration_encoder", lambda full,name:
        SimpleNamespace(name=name, columns=list(range(cli.configuration_width(name)//2))))
    monkeypatch.setattr(cli, "transition_rows", lambda ws,full,selected: rows(ws[0].role, selected.name))
    return dict(reference_fit=reference, reference_fit_sha256=cli._hash(reference),
        eligibility=tmp_path/"eligibility.json", eligibility_sha256="b"*64,
        release=tmp_path/"release", snapshot=tmp_path/"snapshot", data_root=tmp_path/"data",
        output=tmp_path/"candidate.json", available_memory=lambda:cli.MINIMUM_FREE_BYTES)


def ledger(inputs):
    return [json.loads(line) for line in inputs["output"].with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]


def test_complete_fifty_fit_roundtrips_and_distinct_seed_identities(inputs):
    completion = cli.run(**inputs)
    assert completion["status"] == "complete" and completion["attempted_model_count"] == 50
    assert completion["successful_model_count"] == 50 and completion["failure_count"] == 0
    assert completion["unattempted_model_count"] == completion["terminal_error_count"] == 0
    records = ledger(inputs)
    assert len(records) == 54 and sum(r["type"] == "fit" for r in records) == 50
    assert records[-2]["sha256"] == cli._hash(inputs["output"])
    bundle = json.loads(inputs["output"].read_text(encoding="utf-8"))
    assert not bundle["certified"] and not bundle["formal_training_accepted"]
    assert bundle["data_roles"] == {"solve":"train","diagnostic":"validation","selection":"none","final_eval_reads":0}
    assert set(bundle["models"]) == set(cli.terrain_configurations())
    assert all(n == 1 for n in bundle["parameter_identity_count_by_configuration"].values())
    assert len({m["sha256"] for group in bundle["models"].values() for m in group.values()}) == 50
    for name, group in bundle["models"].items():
        for seed, identity in group.items():
            model = restore_direct_dynamics(identity)
            assert identity["seed"] == int(seed) and identity["configuration"] == name
            assert model.conditioner_checkpoint["solver_report"]["normal_equation_penalty"] == pytest.approx(.0024, rel=1e-14)
    assert not any(r["type"] == "failure" for r in records)


def test_one_fit_failure_keeps_all_attempts_and_emits_no_complete_bundle(inputs, monkeypatch):
    original = cli.fit_direct_dynamics
    calls = []
    def fail_first(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("software injected fit failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(cli, "fit_direct_dynamics", fail_first)
    completion = cli.run(**inputs)
    assert completion["status"] == "failed" and not inputs["output"].exists()
    assert completion["attempted_model_count"] == len(calls) == 50
    assert completion["successful_model_count"] == 49 and completion["failure_count"] == 1
    records = [r for r in ledger(inputs) if r["type"] == "fit"]
    assert len(records) == 50 and records[0]["status"] == "failure"
    assert records[0]["error"] == "software injected fit failure"


@pytest.mark.parametrize("fault", ["hash", "schema", "role", "purpose", "config", "seed", "population",
                                   "identity", "parents", "model_label", "curve"])
def test_preflight_rejects_incomplete_or_changed_original_evidence(inputs, monkeypatch, fault):
    reference = json.loads(inputs["reference_fit"].read_text(encoding="utf-8"))
    if fault == "schema": reference["schema_version"] = "other"
    if fault == "role": reference["data_roles"]["final_eval_reads"] = 1
    if fault == "purpose": reference["purpose"] = "final_model"
    if fault == "config": reference["models"].pop("lio-road")
    if fault == "seed": reference["models"]["base"].pop(str(cli.SEEDS[0]))
    if fault == "population": reference["pilot_population"]["samples"]["train"] -= 1
    if fault == "identity": reference["development_identity"]["sha256"] = "f"*64
    if fault == "parents": reference["snapshot_parent_assets"] = []
    if fault in {"model_label", "curve"}:
        model = reference["models"]["base"][str(cli.SEEDS[0])]
        if fault == "model_label": model["seed"] = cli.SEEDS[1]
        else:
            cp = model["conditioner_checkpoint"]
            for record in cp["learning_curve"]:
                if record["epoch"] == cp["best_epoch"]: record["train_loss"] += 1
            cp.pop("checkpoint_identity_sha256")
            cp["checkpoint_identity_sha256"] = digest(cp)
        model.pop("sha256")
        model["sha256"] = digest(model)
    inputs["reference_fit"].write_text(json.dumps(reference), encoding="utf-8")
    inputs["reference_fit_sha256"] = "f"*64 if fault == "hash" else cli._hash(inputs["reference_fit"])
    monkeypatch.setattr(cli, "fit_direct_dynamics", lambda *a,**k: pytest.fail("fit before complete preflight"))
    completion = cli.run(**inputs)
    assert completion["status"] == "failed" and completion["attempted_model_count"] == 0
    assert completion["unattempted_model_count"] == 50 and not inputs["output"].exists()
    assert ledger(inputs)[-2]["type"] == "failure"


@pytest.mark.parametrize("existing", ["output", "ledger"])
def test_outputs_are_exclusive_even_before_preflight(inputs, monkeypatch, existing):
    path = inputs["output"] if existing == "output" else inputs["output"].with_suffix(".jsonl")
    path.write_text("untouched", encoding="utf-8")
    monkeypatch.setattr(cli, "prepare", lambda *a: pytest.fail("preflight after existing output"))
    with pytest.raises(FileExistsError): cli.run(**inputs)
    assert path.read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("guard_call,expected_attempted", [(1,0), (4,2)])
def test_memory_stops_preserve_the_expected_denominator(inputs, guard_call, expected_attempted):
    calls = []
    def memory():
        calls.append(1)
        return 0 if len(calls) == guard_call else cli.MINIMUM_FREE_BYTES
    inputs["available_memory"] = memory
    completion = cli.run(**inputs)
    assert completion["resource_stopped"] and completion["status"] == "failed"
    assert completion["attempted_model_count"] == completion["successful_model_count"] == expected_attempted
    assert completion["unattempted_model_count"] == 50-expected_attempted
    assert not inputs["output"].exists()


def test_memory_error_inside_fit_is_recorded_before_stopping(inputs, monkeypatch):
    def failed(*a, **k): raise MemoryError("software out of memory")
    monkeypatch.setattr(cli, "fit_direct_dynamics", failed)
    completion = cli.run(**inputs)
    assert completion["resource_stopped"] and completion["attempted_model_count"] == 1
    assert completion["failure_count"] == 1 and completion["unattempted_model_count"] == 49
    assert [r for r in ledger(inputs) if r["type"] == "fit"][0]["error_type"] == "MemoryError"


def test_concurrent_source_change_cannot_produce_complete_bundle(inputs, monkeypatch):
    calls = []
    def changed():
        calls.append(1)
        return {"software-source":str(len(calls))}
    monkeypatch.setattr(cli, "source_hashes", changed)
    completion = cli.run(**inputs)
    assert completion["attempted_model_count"] == completion["successful_model_count"] == 50
    assert completion["status"] == "failed" and not inputs["output"].exists()
    assert "source files changed" in ledger(inputs)[-2]["error"]
