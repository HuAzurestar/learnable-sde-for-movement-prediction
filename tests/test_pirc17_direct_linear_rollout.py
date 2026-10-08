"""Software-only candidate orchestration, isolation and numerical audit tests."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pirc17 import direct_linear_fit as fitter
from experiments.pirc17 import direct_linear_rollout as cli
from experiments.pirc17.direct_linear import digest, fit_direct_dynamics, training_policy
from experiments.pirc17.direct_linear_check import check
from experiments.pirc17.direct_linear_evidence import load_candidate
from experiments.pirc17.origins import causal_prefix
from experiments.pirc17.precision_check import load_particle_evidence
from experiments.pirc17.rollout import Features, Forecast, PredictedState, rollout as real_rollout
from tests.test_pirc17_direct_linear import rows as transition_fixture


@pytest.fixture(scope="module")
def candidate_bytes(tmp_path_factory):
    directory = tmp_path_factory.mktemp("software-linear-candidate")
    identity = {"version":"pirc17-development-windows-v1", "eligibility_sha256":"b"*64,
        "fit_population":"one first-future transition per qualified origin; pilot only", "sources":{},
        "sample_ids":{r:[f"{r}-{i:03}" for i in range(24)] for r in ("train", "validation")}}
    identity["sha256"] = digest(identity)
    population = {"samples":{"train":24,"validation":24}, "independent_blocks":{"train":6,"validation":6},
                  "transition_rule":"one first-future transition per qualified origin; no interpolated observations"}
    prepared = {}
    for name in cli.terrain_configurations():
        train, validation = transition_fixture("train", name), transition_fixture("validation", name)
        # Software upstream evidence, not an empirical comparison with Adam.
        # Use the actual production writer so the new reader sees its schema.
        old = {seed:fit_direct_dynamics(train, validation, seed=seed,
            training_identity=identity["sha256"], configuration=name) for seed in cli.SEEDS}
        prepared[name] = train, validation, old, {seed:fitter.losses(m, train, validation) for seed,m in old.items()}
    output = directory/"candidate.json"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fitter, "prepare", lambda *a:({}, identity, ["software-parent"], population, prepared))
        result = fitter.run(reference_fit=directory/"reference.json", reference_fit_sha256="a"*64,
            eligibility=directory/"eligibility.json", eligibility_sha256="b"*64, release=directory,
            snapshot=directory, data_root=directory, output=output, available_memory=lambda:cli.MINIMUM_FREE_BYTES)
    assert result["status"] == "complete"
    return output.read_bytes(), output.with_suffix(".jsonl").read_bytes(), identity


@pytest.fixture
def evidence(tmp_path, candidate_bytes):
    fit, ledger = tmp_path/"candidate.json", tmp_path/"candidate.jsonl"
    fit.write_bytes(candidate_bytes[0])
    ledger.write_bytes(candidate_bytes[1])
    return dict(fit=fit, fit_sha256=cli._hash(fit), fit_ledger=ledger, fit_ledger_sha256=cli._hash(ledger),
                training_policy_sha256=training_policy()["sha256"])


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rebind(evidence, bundle, ledger):
    evidence["fit"].write_text(json.dumps(bundle), encoding="utf-8")
    evidence["fit_sha256"] = cli._hash(evidence["fit"])
    ledger[-2]["sha256"] = evidence["fit_sha256"]
    evidence["fit_ledger"].write_text("".join(json.dumps(row)+"\n" for row in ledger), encoding="utf-8")
    evidence["fit_ledger_sha256"] = cli._hash(evidence["fit_ledger"])


def test_complete_bundle_reader_restores_all_fifty(evidence):
    bundle, models, eligibility = load_candidate(**evidence)
    assert len(models) == 10 and sum(map(len, models.values())) == 50
    assert eligibility == "b"*64 and not bundle["formal_training_accepted"]


@pytest.mark.parametrize("fault", ["fit_hash", "ledger_hash", "policy", "old_schema", "source", "final_role", "accepted",
    "missing_configuration", "missing_seed", "old_model", "seed_label", "duplicate_fit", "failed_fit", "missing_fit",
    "ledger_count", "ledger_error", "reference", "eligibility", "diagnostic", "parameter", "population", "development_hash"])
def test_reader_rejects_incomplete_mixed_or_rehashed_evidence(evidence, fault):
    bundle = json.loads(evidence["fit"].read_text(encoding="utf-8"))
    records = read_rows(evidence["fit_ledger"])
    if fault == "old_schema": bundle["schema_version"] = "pirc17-development-fit-v1"
    if fault == "source": bundle["source_sha256"]["pirc17/dynamics.py"] = "f"*64
    if fault == "final_role": bundle["data_roles"]["final_eval_reads"] = 1
    if fault == "accepted": bundle["formal_training_accepted"] = True
    if fault == "missing_configuration": bundle["models"].pop("lio-surface")
    if fault == "missing_seed": bundle["models"]["lio-surface"].pop(str(cli.SEEDS[-1]))
    if fault == "old_model": bundle["models"]["lio-surface"][str(cli.SEEDS[-1])]["version"] = "old-adam"
    if fault == "seed_label":
        group = bundle["models"]["lio-surface"]
        group[str(cli.SEEDS[-1])] = deepcopy(group[str(cli.SEEDS[0])])
    if fault == "duplicate_fit": records[3] = deepcopy(records[2])
    if fault == "failed_fit": records[2]["status"] = "failure"
    if fault == "missing_fit": records.pop(2)
    if fault == "ledger_count": records[-1]["successful_model_count"] = 49
    if fault == "ledger_error": records[-1]["terminal_error_count"] = 1
    if fault == "reference": records[0]["reference_fit_sha256"] = "f"*64
    if fault == "eligibility": records[0]["eligibility_sha256"] = "f"*64
    if fault == "diagnostic": records[2]["comparison"]["candidate"]["train"]["fitted_mse_m2_per_s2"] += 1
    if fault == "parameter": records[2]["parameter_identity_sha256"] = "f"*64
    if fault == "population": bundle["population"]["samples"]["validation"] = 23
    if fault == "development_hash": bundle["development_identity"]["sample_ids"]["validation"][0] = "other"
    rebind(evidence, bundle, records)
    if fault == "fit_hash": evidence["fit_sha256"] = "f"*64
    if fault == "ledger_hash": evidence["fit_ledger_sha256"] = "f"*64
    if fault == "policy": evidence["training_policy_sha256"] = "f"*64
    with pytest.raises((ValueError, AssertionError)):
        load_candidate(**evidence)


@pytest.fixture
def inputs(tmp_path, monkeypatch, evidence, candidate_bytes):
    origin = causal_prefix([[-2.,-1.],[-1.,-.5],[0.,0.]], [-10.,-5.,0.])
    windows = {r:[SimpleNamespace(sample_id=sample, block_id=r+str(i//4), role=r, origin=origin,
        frame=SimpleNamespace(), horizon_seconds=np.array([60.,300.,900.,1800.]),
        target_positions_m=np.zeros((4,2))) for i,sample in enumerate(candidate_bytes[2]["sample_ids"][r])]
        for r in ("train", "validation")}
    tracker = {"windows":windows, "queries":[], "maps":[], "calls":[], "features":[]}

    class Maps:
        def __init__(self, *a):
            self.parents, self.closed = [], False
            tracker["maps"].append(self)
        def register_snapshot_parent(self, parent): self.parents.append(parent)
        def __call__(self, positions):
            tracker["queries"].append(positions.copy())
            return [{} for _ in positions]
        @property
        def identity(self): return {"software_only":True, "parents":self.parents}
        def close(self): self.closed = True

    class Provider:
        def __init__(self, encoder, query, frame): self.encoder, self.query = encoder, query
        def __call__(self, state):
            assert not hasattr(state, "target_positions_m")
            self.query(state.positions_m)
            tracker["features"].append((self.encoder.name, state.elapsed_seconds))
            shape = len(state.positions_m), len(self.encoder.columns)
            return Features(np.zeros(shape), np.ones(shape, dtype=bool))

    def fake_rollout(origin, horizons, **kwargs):
        tracker["calls"].append(kwargs)
        assert set(kwargs) == {"particles", "seed", "max_step_seconds", "history_step_seconds", "base_drift",
                              "diffusion", "terrain", "conditioner", "brownian_increments"}
        assert kwargs["history_step_seconds"] == 5.
        count = kwargs["particles"]
        state = PredictedState(0., np.broadcast_to(origin.position_m, (count,2)),
            np.broadcast_to(origin.velocity_mps, (count,2)), np.broadcast_to(origin.history_positions_m, (count,3,2)),
            origin.history_times_seconds)
        matrix = kwargs["terrain"](state).model_matrix(count)
        drift = kwargs["base_drift"](state) + kwargs["conditioner"](state, matrix)
        draws = np.stack([kwargs["brownian_increments"](0.,float(t),count,2) for t in horizons], axis=1)
        positions = origin.position_m + drift[:,None,:]*horizons[None,:,None] + np.einsum("pij,ptj->pti",kwargs["diffusion"](state),draws)
        return Forecast(horizons, positions, 0, count)

    monkeypatch.setattr(cli.CanonicalEncoder, "frozen_pirc22", lambda _:SimpleNamespace())
    monkeypatch.setattr(cli, "load_development", lambda *a:(windows, ["software-parent"], deepcopy(candidate_bytes[2])))
    monkeypatch.setattr(cli, "configuration_encoder", lambda full,name:
        SimpleNamespace(name=name, columns=list(range(cli.configuration_width(name)//2))))
    monkeypatch.setattr(cli, "resolve_map_backend", lambda _:(Maps, []))
    monkeypatch.setattr(cli, "PredictedPositionFeatures", Provider)
    monkeypatch.setattr(cli, "rollout", fake_rollout)
    args = dict(**evidence, eligibility=tmp_path/"eligibility.json", eligibility_sha256="b"*64,
        release=tmp_path, snapshot=tmp_path, data_root=tmp_path, output=tmp_path/"rollout.jsonl",
        configurations=["base","all-terrain"], seeds=list(cli.SEEDS[:2]), particles=[4,8], steps=[5.,2.5],
        limit_origins=2, wall_seconds=300., available_memory=lambda:cli.MINIMUM_FREE_BYTES)
    return args, tracker


def test_candidate_forecasts_and_replays_every_pair_with_exact_particle_prefixes(inputs, evidence):
    args, tracker = inputs
    result = cli.run(**args)
    assert result["status"] == "complete" and result["success_count"] == 32
    assert result["failure_count"] == result["unattempted_run_count"] == result["terminal_error_count"] == 0
    records = read_rows(args["output"])
    assert len(records) == 35 and records[1]["sample_ids"] == ["validation-000", "validation-004"]
    assert len(records[1]["model_identities"]) == 10
    assert len(tracker["calls"]) == 32 and len(tracker["queries"]) == 16
    assert tracker["maps"][0].closed and tracker["maps"][0].parents == ["software-parent"]
    before = deepcopy(records)
    report = check(records, args["output"].parent, tolerance_m=10.27506475, **evidence)
    assert records == before
    assert report["status"] == "complete" and not report["certified"]
    assert len(report["numerical_audit"]["sensitivities"]) == 32
    assert len(report["particle_precision"]["runs"]) == 32
    assert not report["particle_precision"]["particle_budget_unavailable"]
    streams = {(r["sample_id"],r["seed"]):r["brownian_identity"]["path_sha256"] for r in records if r["type"] == "run"}
    assert len(set(streams.values())) == 4


def test_all_configurations_and_all_seeds_use_distinct_bound_models(inputs, evidence):
    args, tracker = inputs
    args.update(configurations=list(cli.terrain_configurations()), seeds=list(cli.SEEDS), particles=[4], steps=[5.], limit_origins=1)
    result = cli.run(**args)
    assert result["success_count"] == 50
    records = read_rows(args["output"])
    assert len({r["model_identity_sha256"] for r in records if r["type"] == "run"}) == 50
    assert len(tracker["queries"]) == 45
    assert check(records, args["output"].parent, tolerance_m=10.27506475, **evidence)["status"] == "complete"


@pytest.mark.parametrize("name,value", [("particles",[2]),("particles",[True]),("seeds",[1]),("steps",[0.]),
    ("configurations",["base","base"]),("configurations",["loo-history"]),("limit_origins",True),("wall_seconds",float("inf"))])
def test_invalid_workload_is_rejected_before_output(inputs, name, value):
    args, _ = inputs
    args[name] = value
    with pytest.raises(ValueError): cli.run(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("existing", ["ledger", "particles"])
def test_existing_ledger_or_particle_directory_is_never_overwritten(inputs, existing):
    args, _ = inputs
    path = args["output"] if existing == "ledger" else args["output"].with_suffix(".particles")
    if existing == "ledger": path.write_text("untouched", encoding="utf-8")
    else: path.mkdir()
    with pytest.raises(FileExistsError): cli.run(**args)
    assert path.exists()


@pytest.mark.parametrize("fault", ["eligibility", "role", "block", "sample", "new_sample", "count", "parents", "identity"])
def test_population_preflight_fails_before_any_forecast(inputs, monkeypatch, fault):
    args, tracker = inputs
    windows = tracker["windows"]
    if fault == "eligibility": args["eligibility_sha256"] = "f"*64
    if fault == "role": windows["validation"][0].role = "final_eval"
    if fault == "block": windows["validation"][0].block_id = "train0"
    if fault == "sample": windows["validation"][0].sample_id = windows["validation"][1].sample_id
    if fault == "new_sample": windows["validation"][0].sample_id = "unregistered-software-origin"
    if fault == "count": windows["validation"].pop()
    if fault in {"parents", "identity"}:
        original = cli.load_development
        def wrong(*a):
            w,p,i = original(*a)
            if fault == "parents": p = []
            else: i["sha256"] = "f"*64
            return w,p,i
        monkeypatch.setattr(cli, "load_development", wrong)
    result = cli.run(**args)
    assert result["status"] == "failed" and result["attempted_run_count"] == 0
    assert result["unattempted_run_count"] == 32 and not tracker["calls"] and not tracker["maps"]


@pytest.mark.parametrize("exception", [ValueError, MemoryError, TimeoutError])
def test_forecast_failure_preserves_attempts_and_resource_stop_denominator(inputs, evidence, monkeypatch, exception):
    args, tracker = inputs
    original = cli.rollout
    count = []
    def failing(*a, **k):
        count.append(1)
        if len(count) == 1: raise exception("software injected failure")
        return original(*a, **k)
    monkeypatch.setattr(cli, "rollout", failing)
    result = cli.run(**args)
    stopped = exception is not ValueError
    assert result["status"] == "failed" and result["resource_stopped"] is stopped
    assert result["failure_count"] == 1 and result["attempted_run_count"] == (1 if stopped else 32)
    assert result["unattempted_run_count"] == (31 if stopped else 0)
    assert result["terminal_error_count"] == int(stopped) and tracker["maps"][0].closed
    report = check(read_rows(args["output"]), args["output"].parent, tolerance_m=10.27506475, **evidence)
    assert report["status"] == "failed" and not report["certified"]
    if stopped: assert report["numerical_audit"] is None and report["particle_precision"] is None
    else: assert not report["numerical_audit"]["sensitivities"]


@pytest.mark.parametrize("resource", ["memory", "time"])
def test_resource_stop_before_preflight_keeps_complete_count(inputs, evidence, resource):
    args, tracker = inputs
    if resource == "memory": args["available_memory"] = lambda:0
    else: args["wall_seconds"] = 1e-12
    result = cli.run(**args)
    assert result["resource_stopped"] and result["unattempted_run_count"] == 32
    assert not tracker["calls"]
    assert check(read_rows(args["output"]), args["output"].parent, tolerance_m=10., **evidence)["numerical_audit"] is None


@pytest.mark.parametrize("fault", ["model", "header_model", "policy", "history", "source", "count", "old_schema",
    "missing_run", "duplicate_run", "origin", "horizon", "particle_hash", "particle_float"])
def test_candidate_auditor_rejects_relabeling_and_corrupt_saved_evidence(inputs, evidence, fault):
    args, _ = inputs
    assert cli.run(**args)["status"] == "complete"
    records = read_rows(args["output"])
    if fault == "model": records[2]["model_identity_sha256"] = "f"*64
    if fault == "header_model": records[1]["model_version"] = "old-adam"
    if fault == "policy": records[1]["training_policy_sha256"] = "f"*64
    if fault == "history": records[1]["physical_history_step_seconds"] = 2.5
    if fault == "source": records[1]["source_sha256"]["rollout.py"] = "f"*64
    if fault == "count": records[-1]["unattempted_run_count"] = 1
    if fault == "old_schema": records[1]["schema_version"] = "pirc17-development-rollout-v1"
    if fault == "missing_run": records.pop(2)
    if fault == "duplicate_run": records[3] = deepcopy(records[2])
    if fault == "origin": records[1]["sample_ids"][0] = "train-000"
    if fault == "horizon": records[2]["actual_horizons_seconds"][-1] = 900.
    if fault == "particle_hash": records[2]["particle_artifact"]["sha256"] = "f"*64
    if fault == "particle_float": records[2]["particles"] = float(records[2]["particles"])
    with pytest.raises(ValueError): check(records, args["output"].parent, tolerance_m=10.27506475, **evidence)


def test_source_change_retains_all_work_but_forbids_complete_result(inputs, monkeypatch):
    args, _ = inputs
    original = cli.source_hashes()
    calls = []
    def sources():
        calls.append(1)
        return original if len(calls) == 1 else {**original,"rollout.py":"f"*64}
    monkeypatch.setattr(cli, "source_hashes", sources)
    result = cli.run(**args)
    assert result["status"] == "failed" and result["success_count"] == 32 and result["terminal_error_count"] == 1


def test_real_causal_kernel_predictions_do_not_change_when_hidden_targets_change(inputs, monkeypatch):
    args, tracker = inputs
    monkeypatch.setattr(cli, "rollout", real_rollout)
    args.update(configurations=["base"], seeds=[cli.SEEDS[0]], particles=[4], steps=[5.], limit_origins=1)
    assert cli.run(**args)["status"] == "complete"
    before = read_rows(args["output"])[2]
    first = load_particle_evidence(before, args["output"].parent)
    tracker["windows"]["validation"][0].target_positions_m[:] = 100000.
    args["output"] = args["output"].with_name("changed-targets.jsonl")
    assert cli.run(**args)["status"] == "complete"
    after = read_rows(args["output"])[2]
    second = load_particle_evidence(after, args["output"].parent)
    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[1], second[1]) and before["scores"] != after["scores"]
    assert not tracker["queries"] and len(tracker["features"]) == 720


@pytest.mark.parametrize("fault", ["register", "identity", "close"])
def test_map_failure_still_closes_resources_and_cannot_certify(inputs, monkeypatch, fault):
    args, tracker = inputs
    maps, modules = cli.resolve_map_backend("multicell")
    original_close = maps.close
    if fault == "register":
        def register(self, parent): raise ValueError("software parent failure")
        monkeypatch.setattr(maps, "register_snapshot_parent", register)
    if fault == "identity":
        def identity(self): raise ValueError("software map identity failure")
        monkeypatch.setattr(maps, "identity", property(identity))
    if fault == "close":
        def close(self):
            original_close(self)
            raise ValueError("software close failure")
        monkeypatch.setattr(maps, "close", close)
    result = cli.run(**args)
    assert result["status"] == "failed" and result["terminal_error_count"] == 1
    assert tracker["maps"][0].closed
    assert result["attempted_run_count"] == (0 if fault == "register" else 32)


def test_particle_save_failure_is_retained_without_discarding_other_attempts(inputs, monkeypatch):
    args, _ = inputs
    save, calls = cli.save_particle_artifact, []
    def fail_first(*a, **k):
        calls.append(1)
        if len(calls) == 1: raise OSError("software disk failure")
        return save(*a, **k)
    monkeypatch.setattr(cli, "save_particle_artifact", fail_first)
    result = cli.run(**args)
    assert result["status"] == "failed" and result["attempted_run_count"] == 32
    assert result["failure_count"] == 1 and result["success_count"] == 31


def test_resource_budget_is_checked_after_the_final_particle_save(inputs, monkeypatch):
    args, _ = inputs
    args.update(configurations=["base"], seeds=[cli.SEEDS[0]], particles=[4], steps=[5.], limit_origins=1)
    saved, original = [], cli.save_particle_artifact
    def save(*a, **k):
        result = original(*a, **k)
        saved.append(1)
        return result
    monkeypatch.setattr(cli, "save_particle_artifact", save)
    args["available_memory"] = lambda:0 if saved else cli.MINIMUM_FREE_BYTES
    result = cli.run(**args)
    assert result["status"] == "failed" and result["resource_stopped"]
    assert result["success_count"] == 0 and result["failure_count"] == result["attempted_run_count"] == 1
    records = read_rows(args["output"])
    assert records[2]["status"] == "failure" and "particle_artifact" in records[2]
