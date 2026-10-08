"""Real fit consumers on small synthetic data, never empirical authorization.

The internal fit fixture is deliberately NOT a production protocol/approval.
Actual public guard tests below use independently named synthetic files and
the existing real guard harness; no real dataset or experiment is accessed.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pirc17 import formal_training as module
from experiments.pirc17.configurations import terrain_configurations
from experiments.pirc17.delivery_policy import training_groups
from experiments.pirc17.development import DevelopmentWindow
from experiments.pirc17.direct_linear import configuration_width, restore_direct_dynamics, training_policy
from experiments.pirc17.dynamics import TransitionRows
from experiments.pirc17.features import LocalFrame
from experiments.pirc17.formal_origins import build_velocity_prior
from experiments.pirc17.inference import SEEDS
from experiments.pirc17.origins import causal_prefix, motion
from experiments.pirc17.protocol_core import digest, envelope, read_json, unpack
from experiments.pirc17.protocol import ROOT
from experiments.pirc17.protocol_core import file_hash
from experiments.pirc17.method_training import FIT_SOURCES
from tests.test_pirc17_formal_inputs import qualified_fixture
from tests.test_pirc17_formal_origins import development_fixture


def terrain_rows(role, name):
    n = 12 if role == "train" else 6
    rng = np.random.default_rng(8 if role == "train" else 19)
    velocity = rng.normal(size=(n, 2))
    _, direction, valid = motion(velocity)
    dt = np.linspace(1, 4, n)
    return TransitionRows(role, tuple(role+str(i) for i in range(n)), velocity, direction, valid,
        (velocity+.2*np.sin(3*velocity))*dt[:, None], dt, rng.normal(size=(n, configuration_width(name))))


def fit_fixture():
    # Six per role give each of two synthetic Reptile regions three train
    # segments, and all three heading modes. Never weaken the actual fitter.
    prepared, contract = development_fixture(per_role=6)
    groups = training_groups()
    contract["methods"]["groups"] = groups
    contract["terrain"] = {"training_policy_sha256": training_policy()["sha256"]}
    prior = build_velocity_prior(prepared, training_contract=contract, expected_input_sha256=prepared.identity["sha256"])
    configs = terrain_configurations()
    protocol = envelope({"fixture": "NOT A FORMAL PROTOCOL OR AUTHORIZATION",
        "forecast_contract": {"training": contract}, "components": {"terrain_configurations": configs},
        "dataset_inputs": {"development": {"method_input_sha256": prepared.identity["sha256"]}},
        "source_sha256": {"PSDE-SDE/"+name: file_hash(ROOT/name) for name in FIT_SOURCES},
        "resource_contract": {"per_method_fit_seconds": 90, "per_terrain_fit_seconds": 90}})
    workloads = []
    def add(kind, phase, subject, matrix, fit):
        row = dict(kind=kind, phase=phase, subject=subject, matrix=matrix, fit_identity=fit,
            origin_mode=None, origin_rank=None, seed=None, repetition=None, generated_forecasts=0,
            max_active_seconds=90, scientific=False)
        workloads.append({"work_id": digest(row), **row})
    for g in groups:
        add("method_fit", "method_training", g["representative_slot"], "NEX326-methods",
            "method-fit:"+g["training_components_sha256"])
    for name in sorted(configs):
        add("terrain_fit", "terrain_training", name, "terrain", "terrain-fit:"+name)
    matrix = envelope({"protocol_sha256": protocol["sha256"], "training_groups": groups, "workloads": workloads})
    execution = envelope({"protocol_sha256": protocol["sha256"], "matrix_sha256": matrix["sha256"]})
    encoders = {k: SimpleNamespace(columns=tuple(f"fixture-{i}" for i in range(configuration_width(k)//2))) for k in configs}
    identity = envelope(dict(schema_version=module.INPUT_VERSION, protocol_sha256=protocol["sha256"],
        execution_sha256=execution["sha256"], approval_sha256=digest("NO HUMAN APPROVAL: SOFTWARE ONLY"),
        population_sha256=digest("SOFTWARE POPULATION"), method_input_sha256=prepared.identity["sha256"],
        sample_counts=contract["method_roles"], outer_train_windows=12, validation_windows=6,
        terrain_training_policy_sha256=training_policy()["sha256"], prior_evidence_sha256=prior.evidence["sha256"],
        configuration_columns={k: list(e.columns) for k, e in encoders.items()},
        final_eval_numeric_training_rows=0, new_predictive_model_fits=0))
    inputs = module.RegisteredTrainingInputs(prepared,
        {k: {r: terrain_rows(r, k) for r in ("train", "validation")} for k in configs}, encoders, prior, identity)
    return dict(protocol=protocol, execution=execution, matrix=matrix, inputs=inputs)


def windows_fixture(*, per_role=2):
    prepared, _ = development_fixture(per_role=per_role)
    windows = {"train": [], "validation": []}
    for p, s in zip(prepared.prefixes, prepared.segments):
        lonlat = p.condition_at.frame.to_lonlat(s.state)
        n = len(p.visible_epoch_ns)
        frame = LocalFrame(*lonlat[n-1])
        xy = frame.from_lonlat(lonlat)
        origin = causal_prefix(xy[n-3:n], s.time[n-3:n])
        sample = p.assignment.sample
        windows[sample.split].append(DevelopmentWindow(sample.sample_id, sample.independent_block_id, sample.split,
            origin, frame, np.array([60., 300., 900., 1800.]), xy[[n, n+4, n+14, n+29]],
            xy[n]-origin.position_m, float(s.time[n]), {}))
    return prepared, windows


def test_all_registered_fits_are_actual_distinct_models_with_complete_slot_bindings(tmp_path):
    args = fit_fixture()
    consumer = module.FitConsumers(**args)
    bound_slots, terrains = [], []
    for work in args["matrix"]["payload"]["workloads"]:
        result = consumer.execute(work, output_directory=tmp_path/work["work_id"])
        record = read_json(result["artifact_path"])
        p = unpack(record, expected_sha256=result["artifact_sha256"])
        assert p["schema_version"] == module.FIT_VERSION and p["work_id"] == work["work_id"]
        assert p["protocol_sha256"] == args["protocol"]["sha256"]
        assert p["input_sha256"] == args["inputs"].identity["sha256"]
        artifact = p["artifact"]
        restored_model = module.restore_registered_fit(record, work=work, protocol=args["protocol"],
            execution=args["execution"], matrix=args["matrix"], input_identity=args["inputs"].identity)
        if work["kind"] == "method_fit":
            bound_slots.extend(artifact["slot_bindings"])
            assert artifact["training"]["sample_counts"] == {"train": 6, "adapt": 6, "validation": 6}
            assert artifact["training"]["formal_training_accepted"] is False  # never rewrite historical flags
            assert artifact["fit_seconds"] >= 0
            expected = digest({"training_identity_sha256": artifact["training"]["training_identity_sha256"],
                               "model": artifact["model"]})
            assert p["parameter_identity"] == expected
            assert restored_model.dynamics.identity() == consumer.models[work["fit_identity"]].dynamics.identity()
        else:
            terrains.append(artifact["configuration"])
            restored = restore_direct_dynamics(artifact["model"])
            assert restored.identity["configuration"] == work["subject"]
            assert restored.identity["train_transition_count"] == 12
            assert restored.identity["validation_transition_count"] == 6
            assert artifact["forecast_seed_labels"] == list(SEEDS)
            assert p["parameter_identity"] == restored.identity["sha256"]
            assert restored_model.identity == consumer.models[work["fit_identity"]].identity
    assert len(bound_slots) == len(set(bound_slots)) == 28
    assert set(terrains) == set(terrain_configurations())
    assert len(consumer.models) == len(consumer.receipts) == len(consumer.attempted) == 26
    assert len({id(m) for k, m in consumer.models.items() if k.startswith("terrain-fit:")}) == 10
    assert len(list(tmp_path.rglob("*.json"))) == 26


@pytest.mark.parametrize("failure", [False, True])
def test_success_and_failure_both_forbid_same_live_handler_retry(tmp_path, monkeypatch, failure):
    args = fit_fixture(); consumer = module.FitConsumers(**args)
    work = args["matrix"]["payload"]["workloads"][0]
    if failure:
        def broken(*a, **k): raise RuntimeError("synthetic fit failure")
        monkeypatch.setattr(module, "fit_development_method", broken)
        with pytest.raises(RuntimeError, match="synthetic fit failure"):
            consumer.execute(work, output_directory=tmp_path)
        assert not consumer.models and not list(tmp_path.glob("*.json"))
    else:
        consumer.execute(work, output_directory=tmp_path)
    monkeypatch.setattr(module, "fit_development_method", lambda *a, **k: pytest.fail("attempt repeated"))
    with pytest.raises(ValueError, match="already attempted"):
        consumer.execute(work, output_directory=tmp_path)


@pytest.mark.parametrize("field,value", [("subject", "arm-13/fake"), ("phase", "terrain_training"),
                                        ("work_id", "0"*64), ("fit_identity", "terrain-fit:base")])
def test_replaced_work_fails_before_model_fit_or_publication(tmp_path, monkeypatch, field, value):
    args = fit_fixture(); consumer = module.FitConsumers(**args)
    work = {**args["matrix"]["payload"]["workloads"][0], field: value}
    monkeypatch.setattr(module, "fit_development_method", lambda *a, **k: pytest.fail("unregistered fit"))
    with pytest.raises(ValueError, match="exact registered"):
        consumer.execute(work, output_directory=tmp_path/"output")
    assert not (tmp_path/"output").exists() and not consumer.attempted


@pytest.mark.parametrize("fault", ["method-input", "population", "counts", "policy", "prior", "final-rows", "already-fitted", "columns"])
def test_fit_scope_rejects_rehashed_changes(fault):
    args = fit_fixture(); p = deepcopy(unpack(args["inputs"].identity))
    if fault == "method-input": p["method_input_sha256"] = "0"*64
    elif fault == "population": p["population_sha256"] = "not-an-identity"
    elif fault == "counts": p["outer_train_windows"] -= 1
    elif fault == "policy": p["terrain_training_policy_sha256"] = "0"*64
    elif fault == "prior": p["prior_evidence_sha256"] = "0"*64
    elif fault == "final-rows": p["final_eval_numeric_training_rows"] = 1
    elif fault == "already-fitted": p["new_predictive_model_fits"] = 1
    else: p["configuration_columns"]["base"] = []
    args["inputs"] = replace(args["inputs"], identity=envelope(p))
    with pytest.raises(ValueError): module.FitConsumers(**args)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "group", "fit-id", "changed-cap", "extra-seed"])
def test_rehashed_matrix_cannot_omit_or_change_registered_fits(fault):
    args = fit_fixture(); m = deepcopy(unpack(args["matrix"]))
    if fault == "missing": m["workloads"].pop()
    elif fault == "duplicate": m["workloads"][-1] = deepcopy(m["workloads"][-2])
    elif fault == "group": m["training_groups"][0]["slots"] = []
    else:
        w = m["workloads"][0]
        if fault == "fit-id": w["fit_identity"] = "terrain-fit:base"
        elif fault == "changed-cap": w["max_active_seconds"] = 91
        else: w["seed"] = SEEDS[0]
        w["work_id"] = digest({k: v for k, v in w.items() if k != "work_id"})
    args["matrix"] = envelope(m)
    e = deepcopy(unpack(args["execution"])); e["matrix_sha256"] = args["matrix"]["sha256"]
    args["execution"] = envelope(e)
    scope = deepcopy(unpack(args["inputs"].identity)); scope["execution_sha256"] = args["execution"]["sha256"]
    args["inputs"] = replace(args["inputs"], identity=envelope(scope))
    with pytest.raises(ValueError): module.FitConsumers(**args)


def test_caller_mutation_cannot_rewrite_captured_fit_descriptors(tmp_path):
    args = fit_fixture(); consumer = module.FitConsumers(**args)
    work = deepcopy(args["matrix"]["payload"]["workloads"][0])
    args["matrix"]["payload"]["workloads"][0]["subject"] = "arm-13/not-required"
    args["protocol"]["payload"]["forecast_contract"]["training"]["methods"]["max_transitions_per_fit"] = 1
    result = consumer.execute(work, output_directory=tmp_path)
    assert unpack(read_json(result["artifact_path"]))["artifact"]["slot_bindings"]


def test_crosscheck_uses_same_original_prefix_and_first_transition_in_both_frames():
    prepared, windows = windows_fixture()
    module._crosscheck_development(prepared, windows)
    assert len(windows["train"]) == 4 and len(windows["validation"]) == 2


@pytest.mark.parametrize("fault", ["missing-segment", "segment-order", "duplicate-prefix", "missing-window", "duplicate-window",
                                        "unknown-window", "role", "block", "time", "history", "displacement"])
def test_crosscheck_rejects_missing_or_relabelled_development(fault):
    prepared, windows = windows_fixture()
    w = windows["train"][0]
    if fault == "missing-segment": prepared = replace(prepared, segments=prepared.segments[:-1])
    elif fault == "segment-order": prepared = replace(prepared, segments=prepared.segments[::-1])
    elif fault == "duplicate-prefix": prepared = replace(prepared, prefixes=(prepared.prefixes[0],)*len(prepared.prefixes))
    elif fault == "missing-window": windows["train"].pop()
    elif fault == "duplicate-window": windows["train"].append(w)
    elif fault == "unknown-window": windows["train"][0] = replace(w, sample_id="unknown")
    elif fault == "role": windows["train"][0] = replace(w, role="validation")
    elif fault == "block": windows["train"][0] = replace(w, block_id="changed")
    elif fault == "time": windows["train"][0] = replace(w, transition_seconds=w.transition_seconds+1)
    elif fault == "history":
        xy = w.origin.history_positions_m.copy(); xy[:-1] += 1
        windows["train"][0] = replace(w, origin=replace(w.origin, history_positions_m=xy))
    else: windows["train"][0] = replace(w, transition_displacement_m=w.transition_displacement_m+1)
    with pytest.raises(ValueError): module._crosscheck_development(prepared, windows)


@pytest.mark.parametrize("missing", ["approval_sha256", "population_sha256"])
def test_public_training_entry_cannot_open_snapshot_without_approval(tmp_path, monkeypatch, missing):
    existing = qualified_fixture(tmp_path, monkeypatch)
    args = {k: existing[k] for k in ("protocol", "execution", "approval_path", "approval_sha256", "test_path",
        "review_path", "journal_directory", "population_path", "population_sha256", "release", "data_root")}
    args.update(final_eligibility_path=existing["eligibility_path"], development_eligibility_path=tmp_path/"never-open-dev.json",
                snapshot=tmp_path/"never-open-snapshot", trajectory_path=tmp_path/"never-open-trajectories.parquet")
    args[missing] = None
    monkeypatch.setattr(module, "_prepare", lambda *a, **k: pytest.fail("training opened before approval"))
    monkeypatch.setattr(module.CanonicalEncoder, "frozen_pirc22", lambda *a: pytest.fail("snapshot opened before approval"))
    with pytest.raises(ValueError): module.prepare_formal_training(**args)


def test_prepare_rejects_wrong_access_kind_before_any_raw_file(monkeypatch):
    args = fit_fixture()
    monkeypatch.setattr(module, "_release_sources", lambda *a: pytest.fail("unapproved raw inputs"))
    access = SimpleNamespace(access_kind="final_eval_positions", population_sha256="0"*64, legacy_cohort_ack="fixture")
    with pytest.raises(ValueError, match="population-bound"):
        module._prepare(access, protocol=args["protocol"], execution=args["execution"],
                        eligibility_path=None, release=None, snapshot=None, data_root=None, trajectory_path=None)


def test_training_preparation_wiring_loads_full_development_once_and_never_fits(tmp_path, monkeypatch):
    """IO-adapter doubles isolate orchestration, not real data qualification.

Real loader/encoder IO and transformations have their own temporary-file
tests. Here prior derivation, cross-frame joins, transition extraction and
ten-way matrix slicing are real; no real approval is represented by the token.
    """
    from experiments.pirc17.rollout import Features
    args = fit_fixture()
    prepared, windows = windows_fixture(per_role=6)
    identity = {k: v for k, v in prepared.identity.items() if k != "sha256"}
    identity["observed_points"] = sum(len(s.time) for s in prepared.segments)
    prepared = replace(prepared, identity={**identity, "sha256": digest(identity)})
    p = deepcopy(unpack(args["protocol"]))
    contract = p["forecast_contract"]["training"]
    contract["eligibility_sha256"] = digest("software eligibility")
    p["resource_contract"]["phase_caps_seconds"] = {"input_qualification_and_binding": 3600}
    binding = p["dataset_inputs"]
    binding.update(dataset_id="software-dataset", dataset_file_sha256=digest("software dataset file"),
        source_trajectory_sha256=digest("software trajectory"),
        snapshot={"manifest_sha256": digest("manifest"), "feature_spec_file_sha256": digest("spec")},
        release_artifact_sha256={"condition_file_manifest.jsonl": digest("conditions")})
    binding["development"] = {"method_input_sha256": prepared.identity["sha256"], "observed_points": identity["observed_points"]}
    protocol = envelope(p)
    execution = envelope({**unpack(args["execution"]), "protocol_sha256": protocol["sha256"]})
    access = SimpleNamespace(access_kind="final_eval_features", population_sha256=digest("SYNTHETIC POPULATION"),
        legacy_cohort_ack="software-dataset", approval_sha256=digest("NO HUMAN APPROVAL"), access_started_sha256=digest("synthetic start"),
        protocol_sha256=protocol["sha256"], execution_sha256=execution["sha256"])
    eligibility = {"dataset_id": "software-dataset", "eligibility": {"horizon_minutes": 30,
        "roles": ["train", "validation"], "rows": [{"sample_id": w.sample_id, "split": w.role}
                                                   for group in windows.values() for w in group]}}
    checks, encoded = [], []
    def json_input(path, *, expected_file_sha256):
        checks.append((path.name, expected_file_sha256))
        return {"dataset.json": {"source": {"trajectory": {"sha256": binding["source_trajectory_sha256"]}}},
                "manifest.json": {}, "feature_spec.json": {}, "eligible.json": eligibility}[path.name]
    monkeypatch.setattr(module, "read_json", json_input)
    monkeypatch.setattr(module, "_release_sources", lambda release, b: checks.append(("release", b["dataset_id"])))
    monkeypatch.setattr(module, "_bound_file", lambda *a: checks.append(("condition-manifest", a[-1])))
    def method_loader(*a, sample_ids, budget):
        assert sample_ids == prepared.identity["sample_ids"]
        assert budget.max_samples == 18 and budget.max_total_points == 18*64 and budget.wall_seconds == 180
        checks.append(("methods", sample_ids))
        return prepared
    monkeypatch.setattr(module, "load_method_development", method_loader)
    class FullEncoder:
        columns = tuple(f"fixture-{i}" for i in range(28))
        def encode(self, rows):
            encoded.append(len(rows))
            values = np.arange(len(rows)*28).reshape(len(rows), 28)/100
            return Features(values, np.ones(values.shape, dtype=bool))
    full = FullEncoder()
    monkeypatch.setattr(module.CanonicalEncoder, "frozen_pirc22", lambda path: full)
    monkeypatch.setattr(module, "configuration_encoder", lambda encoder, name:
                        SimpleNamespace(columns=encoder.columns[:configuration_width(name)//2]))
    monkeypatch.setattr(module, "load_development", lambda *a: (windows, ["registered-file:software-map"], {"sha256": digest("terrain inputs")}))
    monkeypatch.setattr(module, "fit_development_method", lambda *a, **k: pytest.fail("input preparation fitted a model"))
    monkeypatch.setattr(module, "fit_direct_dynamics", lambda *a, **k: pytest.fail("input preparation fitted a model"))
    result = module._prepare(access, protocol=protocol, execution=execution,
        eligibility_path=tmp_path/"eligible.json", release=tmp_path/"release", snapshot=tmp_path/"snapshot",
        data_root=tmp_path/"data", trajectory_path=tmp_path/"trajectory.parquet")
    assert encoded == [12, 6]  # encode once per role, not ten snapshot passes
    assert sum(name == "methods" for name, _ in checks) == 1
    assert len(result.prior.prior.velocities_mps) == 12
    scope = unpack(result.identity)
    assert scope["final_eval_numeric_training_rows"] == scope["new_predictive_model_fits"] == 0
    assert scope["method_input_sha256"] == prepared.identity["sha256"]
    for name, roles in result.terrain.items():
        assert roles["train"].feature_matrix.shape == (12, configuration_width(name))
        assert roles["validation"].feature_matrix.shape == (6, configuration_width(name))
        np.testing.assert_array_equal(roles["train"].displacement_m,
                                      np.stack([w.transition_displacement_m for w in windows["train"]]))


@pytest.fixture(scope="module")
def saved_fit_records(tmp_path_factory):
    args = fit_fixture(); consumer = module.FitConsumers(**args)
    directory = tmp_path_factory.mktemp("synthetic-formal-fit-records")
    output = {}
    for work in args["matrix"]["payload"]["workloads"]:
        if work["kind"] == "method_fit" and "method" not in output:
            family = "method"
        elif work["fit_identity"] == "terrain-fit:all-terrain":
            family = "terrain"
        else:
            continue
        saved = consumer.execute(work, output_directory=directory/work["work_id"])
        output[family] = work, read_json(saved["artifact_path"])
    return args, output


@pytest.mark.parametrize("fault", ["input", "work", "slots", "source", "clock", "counts", "duplicate-segment",
    "partial-tail", "model-family", "estimator", "covariance", "fit-time", "historical-flag", "parameter"])
def test_method_record_rejects_rehashed_scope_and_semantic_changes(saved_fit_records, fault):
    from experiments.pirc17.method_training import _digest as tdigest
    args, records = saved_fit_records
    work, original = records["method"]
    p = deepcopy(unpack(original)); a = p["artifact"]; t = a["training"]
    if fault == "input": p["input_sha256"] = "0"*64
    elif fault == "work": p["work_id"] = "0"*64
    elif fault == "slots": a["slot_bindings"] = []
    elif fault == "source": t["source_sha256"][FIT_SOURCES[0]] = "0"*64
    elif fault == "clock": t["reference_interval_seconds"] *= 2
    elif fault == "counts": t["sample_counts"]["train"] -= 1
    elif fault == "duplicate-segment": t["per_segment"][1]["segment_id"] = t["per_segment"][0]["segment_id"]
    elif fault == "partial-tail": t["per_segment"][0]["removed_tail_seconds"] = 999
    elif fault == "model-family": a["model"]["model_kind"] = "single_gaussian"
    elif fault == "estimator": a["model"]["estimator_method"] = "fake-estimator"
    elif fault == "covariance": a["model"]["covariance_scale"] = -1
    elif fault == "fit-time": a["fit_seconds"] = -1
    elif fault == "historical-flag": t["formal_training_accepted"] = True
    else: p["parameter_identity"] = "0"*64
    t["training_identity_sha256"] = tdigest({k: v for k, v in t.items() if k != "training_identity_sha256"})
    with pytest.raises(ValueError):
        module.restore_registered_fit(envelope(p), work=work, protocol=args["protocol"], execution=args["execution"],
                                      matrix=args["matrix"], input_identity=args["inputs"].identity)


@pytest.mark.parametrize("fault", ["config", "seed-labels", "input", "counts", "seed", "parameter"])
def test_terrain_record_rejects_rehashed_configuration_and_population_changes(saved_fit_records, fault):
    args, records = saved_fit_records
    work, original = records["terrain"]
    p = deepcopy(unpack(original)); a = p["artifact"]; model = a["model"]
    if fault == "config": a["configuration"] = "base"
    elif fault == "seed-labels": a["forecast_seed_labels"] = [SEEDS[0]]
    elif fault == "input": model["training_identity"] = "0"*64
    elif fault == "counts": model["validation_transition_count"] += 1
    elif fault == "seed": model["seed"] = SEEDS[1]
    else: p["parameter_identity"] = "0"*64
    model["sha256"] = digest({k: v for k, v in model.items() if k != "sha256"})
    with pytest.raises(ValueError):
        module.restore_registered_fit(envelope(p), work=work, protocol=args["protocol"], execution=args["execution"],
                                      matrix=args["matrix"], input_identity=args["inputs"].identity)


def test_saved_models_restore_without_calling_fitter(saved_fit_records, monkeypatch):
    args, records = saved_fit_records
    monkeypatch.setattr(module, "fit_development_method", lambda *a, **k: pytest.fail("restoration refitted method"))
    monkeypatch.setattr(module, "fit_direct_dynamics", lambda *a, **k: pytest.fail("restoration refitted terrain"))
    for work, record in records.values():
        assert module.restore_registered_fit(record, work=work, protocol=args["protocol"], execution=args["execution"],
            matrix=args["matrix"], input_identity=args["inputs"].identity) is not None
