from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.nex326.cohort import Cohort, Segment, load_cohort
from experiments.nex326.completion import (
    CompletionAuditError,
    build_completion_report,
    write_completion_report,
)
from experiments.nex326.dsde_pilot import materialize_dsde_zhejiang_pilot
from experiments.nex326.endpoint_prior import (
    EndpointPriorError,
    attach_endpoint_priors,
    write_endpoint_prior_request,
)
from experiments.nex326.fidelity import build_fidelity_report, write_fidelity_report
from experiments.nex326.model import build_transition_data, train_model
from experiments.nex326.pilot_receipt import PilotReceiptError, build_pilot_receipt
from experiments.nex326.phase_space import (
    PhaseSpaceError,
    TerrainAlignedResidualModel,
    TerrainAlignedVelocityModel,
    directional_terrain_velocity_terms,
    fit_affine_velocity_model,
    fit_terrain_aligned_residual_model,
    fit_terrain_aligned_velocity_model,
    load_phase_space_spec,
    phase_space_state,
    rollout_phase_space,
    terrain_aligned_velocity_components,
    write_phase_space_report,
)
from experiments.nex326.runner import (
    NEX326Runner,
    RunError,
    STAGES,
    _apply_bridge,
    predict_segments,
    validate_run_record,
)
from experiments.nex326.schrodinger import (
    SchrodingerBridgeError,
    solve_particle_schrodinger_bridge,
)
from experiments.nex326.spatial_conditions import (
    DIRECTIONAL_TERRAIN_CONDITION_NAMES,
    DSDERasterConditionResolver,
)
from experiments.nex326_process import main
from experiments.nex326_phase_space_multi_seed import (
    PhaseSpaceReplicateError,
    run_phase_space_replicates,
    write_phase_space_contrast,
    write_phase_space_receipt,
    write_phase_space_segment_bootstrap,
)
from experiments.nex326_multi_seed import MultiSeedError, validate_replicate_seeds
from experiments.nex326.specification import FULL_ANCHORS, GROUP_COUNTS, load_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
NEX326 = ROOT / "experiments" / "nex326"


def test_frozen_spec_has_exact_lineage_and_numbering_contract():
    spec = load_experiment_spec()
    assert {arm.arm_id for arm in spec.arms} == set(range(1, 23))
    assert {
        group: sum(arm.group == group for arm in spec.arms)
        for group in GROUP_COUNTS
    } == GROUP_COUNTS
    assert {arm.arm_id for arm in spec.arms if arm.control["full_anchor"]} == FULL_ANCHORS
    assert {arm.control["config_id"] for arm in spec.arms if arm.control["full_anchor"]} == {
        "NEX326-FULL-v2"
    }
    assert len(spec.executions) == 36
    assert {item["idea_id"]: item["role"] for item in spec.unnumbered_ideas} == {
        "C-4": "diagnostic_only",
        "C-8": "theory_only",
    }
    arm13 = next(arm for arm in spec.arms if arm.arm_id == 13)
    assert arm13.source_correction == {
        "original_ref": "C-4/NEX-97",
        "canonical_ref": "C-3/NEX-95",
        "basis": "NEX-311 v0.3 names animal-to-human transfer; C-4 is an unnumbered ABM diagnostic.",
    }
    assert (17, "terrain") in {
        (arm.arm_id, subconfig["subconfig_id"])
        for arm, subconfig in spec.executions
    }


def test_registered_fixture_is_versioned_disjoint_and_has_non_oracle_priors():
    cohort = load_cohort(NEX326 / "fixtures" / "registered_cohort.json")
    assert cohort.purpose == "implementation_fixture_not_scientific_evidence"
    ids = [
        segment.segment_id
        for segments in cohort.splits.values()
        for segment in segments
    ]
    assert len(ids) == len(set(ids))
    for segment in cohort.splits["evaluation"]:
        assert segment.endpoint_prior_mean is not None
        assert segment.endpoint_prior_source == "simulated_external_planning_feed_v1"
        assert not segment.endpoint_prior_derived_from_truth


def test_all_arm_subconfigs_run_the_same_artifact_producing_pipeline(tmp_path, monkeypatch):
    import experiments.nex326.runner as runner_module

    called = {name: 0 for name in ("build_transition_data", "train_model", "predict_segments", "compute_metrics", "mechanism_statistics")}
    for name in tuple(called):
        original = getattr(runner_module, name)

        def traced(*args, _name=name, _original=original, **kwargs):
            called[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(runner_module, name, traced)
    runner = NEX326Runner.from_paths(
        NEX326 / "fixtures" / "registered_cohort.json",
        tmp_path / "runs",
        n_samples=24,
    )
    records = runner.run_all()
    assert len(records) == 36
    assert called == {name: 36 for name in called}
    assert {record["arm_id"] for record in records} == set(range(1, 23))
    assert {(record["implementation_status"], record["run_status"], record["verdict"]) for record in records} == {
        ("implemented", "succeeded", "not_assessed")
    }
    source_bundles = {
        record["implementation"]["source_bundle_sha256"] for record in records
    }
    execution_identities = {
        record["implementation"]["execution_identity_sha256"] for record in records
    }
    assert len(source_bundles) == 1
    assert len(execution_identities) == 1
    assert all(len(bundle) == 64 for bundle in source_bundles)
    environment_conformance = {
        record["implementation"]["environment_lock"]["conformant"]
        for record in records
    }
    assert environment_conformance == {
        runner.implementation["environment_lock"]["conformant"]
    }
    for record in records:
        assert [stage["name"] for stage in record["stages"]] == list(STAGES)
        assert all(stage["status"] == "completed" for stage in record["stages"])
        assert record["mechanism_gates"]
        assert record["runtime"]["requested_prediction_samples"] == 24
        assert record["runtime"]["effective_prediction_samples"] >= 24
        assert record["seed"] == 20260814
        assert record["replicate_seed"] == 20260814
        gate = record["mechanism_gates"][0]
        assert set(gate) >= {"statistic", "value", "operator", "threshold", "passed", "sample_size"}
        assert gate["passed"]
        run_dir = tmp_path / "runs" / record["run_id"]
        for reference in record["artifacts"].values():
            artifact = tmp_path / "runs" / reference["path"]
            assert artifact.is_file() and artifact.stat().st_size > 0
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == reference["sha256"]
        persisted = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
        assert persisted["result_id"] == record["result_id"]
        expected_result_id = hashlib.sha256(
            (
                f"{record['spec_version']}:{record['run_id']}:"
                f"{record['dataset']['fingerprint']}:"
                f"{record['implementation']['execution_identity_sha256']}"
            ).encode()
        ).hexdigest()
        assert record["result_id"] == expected_result_id
        expected_execution_identity = hashlib.sha256(
            json.dumps(
                {
                    "source_bundle_sha256": record["implementation"][
                        "source_bundle_sha256"
                    ],
                    "runtime": record["implementation"]["runtime"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        assert (
            record["implementation"]["execution_identity_sha256"]
            == expected_execution_identity
        )
        for source in record["implementation"]["files"]:
            path = NEX326 / source["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    anchors = [record for record in records if record["is_full_anchor"]]
    assert {record["arm_id"] for record in anchors} == FULL_ANCHORS
    assert len({json.dumps(record["config"], sort_keys=True) for record in anchors}) == 1


def test_arm22_uses_distinct_non_oracle_bridge_implementations(tmp_path):
    spec = load_experiment_spec()
    cohort = load_cohort(NEX326 / "fixtures" / "registered_cohort.json")
    runner = NEX326Runner(spec, cohort, tmp_path / "runs", n_samples=32)
    arm = next(item for item in spec.arms if item.arm_id == 22)
    records = [runner.run_one(arm, subconfig) for subconfig in arm.subconfigs]
    assert {record["config"]["bridge"] for record in records} == {
        "doob",
        "gaussian_schrodinger",
        "soft_endpoint",
    }
    assert all(record["mechanism_gates"][0]["value"] > 0 for record in records)
    sb_record = next(record for record in records if record["subconfig_id"] == "sb")
    assert sb_record["config"] == {
        **spec.full_components,
        "bridge": "gaussian_schrodinger",
        "bridge_epsilon_scale": 0.5,
        "bridge_time_steps": 8,
        "bridge_sinkhorn_iterations": 500,
        "bridge_sinkhorn_tolerance": 1e-8,
    }
    prediction_path = tmp_path / "runs" / sb_record["artifacts"]["predictions"]["path"]
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))["predictions"][0]
    assert np.asarray(prediction["bridge_path"]).shape == (32, 9, 2)
    assert prediction["bridge_diagnostics"]["converged"] is True
    assert prediction["bridge_diagnostics"]["solver"] == (
        "log_sinkhorn_brownian_particle_bridge"
    )
    assert prediction["bridge_diagnostics"]["max_marginal_error"] <= 1e-8


def test_bridge_runtime_paths_are_distinct_for_identical_inputs():
    cohort = load_cohort(NEX326 / "fixtures" / "registered_cohort.json")
    segment = cohort.splits["evaluation"][0]
    samples = np.asarray([[0.0, 0.0], [0.5, -0.25], [1.0, 0.75], [-0.5, 0.25]])
    outputs = {
        bridge: _apply_bridge(samples, segment, bridge, np.random.default_rng(123))[0]
        for bridge in ("doob", "gaussian_schrodinger", "soft_endpoint")
    }
    assert not np.allclose(outputs["doob"], outputs["gaussian_schrodinger"])
    assert not np.allclose(outputs["doob"], outputs["soft_endpoint"])
    assert not np.allclose(outputs["gaussian_schrodinger"], outputs["soft_endpoint"])

    solved = _apply_bridge(
        samples, segment, "gaussian_schrodinger", np.random.default_rng(123)
    )
    assert solved[3] is not None and solved[3].shape == (4, 9, 2)
    assert np.allclose(solved[3][:, -1], solved[0])
    assert solved[4] is not None and solved[4]["converged"] is True
    assert solved[4]["realized_terminal_particle_coverage"] == 1.0
    assert len(np.unique(solved[0], axis=0)) == len(samples)


def test_particle_schrodinger_bridge_refuses_nonconvergence():
    rng = np.random.default_rng(17)
    samples = rng.normal(size=(8, 2))
    with pytest.raises(SchrodingerBridgeError, match="did not converge"):
        solve_particle_schrodinger_bridge(
            samples,
            np.asarray([10.0, -10.0]),
            np.asarray([[1.0, 0.2], [0.2, 1.0]]),
            rng,
            epsilon_scale=0.01,
            time_steps=8,
            max_iterations=1,
            tolerance=1e-15,
        )


def test_phase_space_contract_uses_causal_four_dimensional_state():
    spec = load_phase_space_spec()
    assert spec["state_contract"]["layout"] == ["x", "y", "vx", "vy"]
    assert spec["state_contract"]["position_dynamics"] == "dX=Vdt"
    assert spec["condition_contract"]["condition_is_dynamic_state"] is False

    cohort = load_cohort(NEX326 / "fixtures" / "registered_cohort.json")
    segment = cohort.splits["train"][0]
    phase = phase_space_state(segment)
    expected_velocity = np.diff(segment.state, axis=0) / np.diff(segment.time)[:, None]
    assert phase.shape == (len(segment.time), 4)
    assert np.allclose(phase[1:, 2:], expected_velocity)
    assert np.allclose(phase[0, 2:], expected_velocity[0])


def test_directional_terrain_terms_distinguish_uphill_downhill_and_tangent():
    signed, distance = directional_terrain_velocity_terms(
        np.asarray([[2.0, 3.0], [-2.0, 3.0], [0.0, -4.0]]),
        np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
    )
    assert np.allclose(signed, [2.0, -2.0, 0.0])
    assert np.allclose(distance, [2.0, 2.0, 0.0])
    spec = load_phase_space_spec(
        NEX326 / "phase_space_directional_terrain_benchmark.json"
    )
    assert spec["velocity_model"]["feature_basis"] == "directional_terrain_v2"
    assert spec["protocol"]["primary_comparator"] == (
        "NEX326-PHASE-SPACE-4D-TERRAIN-v1"
    )
    assert "t and -t are equivalent" in spec["condition_contract"][
        "contour_tangent_semantics"
    ]


def test_terrain_aligned_projection_resolves_normal_and_tangent_velocity():
    velocity = np.asarray([[3.0, 4.0], [-2.0, 5.0], [7.0, -1.0]])
    gradient = np.asarray([[2.0, 0.0], [0.0, -3.0], [0.0, 0.0]])
    normal, tangent = terrain_aligned_velocity_components(velocity, gradient)
    assert np.allclose(normal, [[3.0, 0.0], [0.0, 5.0], [0.0, 0.0]])
    assert np.allclose(tangent, [[0.0, 4.0], [-2.0, 0.0], [7.0, -1.0]])
    assert np.allclose(normal + tangent, velocity)
    flipped_normal, flipped_tangent = terrain_aligned_velocity_components(
        velocity, -gradient
    )
    assert np.allclose(flipped_normal, normal)
    assert np.allclose(flipped_tangent, tangent)

    model = TerrainAlignedVelocityModel(
        condition_names=DIRECTIONAL_TERRAIN_CONDITION_NAMES,
        intercept=np.asarray([0.5, -0.25]),
        normal_response=-2.0,
        tangent_response=-0.5,
        gradient_force=3.0,
        design_scales=np.ones(5),
        diffusion_covariance=np.eye(2),
        transition_count=3,
    )
    conditions = np.column_stack([np.zeros(3), gradient])
    expected = 0.5 * np.asarray([[1.0, -0.5]]) - 2.0 * normal - 0.5 * tangent + 3.0 * gradient
    assert np.allclose(model.acceleration(velocity, conditions), expected)


def test_terrain_aligned_v4_is_preregistered_as_structural_followup():
    spec = load_phase_space_spec(
        NEX326 / "phase_space_terrain_aligned_benchmark.json"
    )
    assert spec["velocity_model"]["kind"] == "terrain_aligned_projection_drift"
    assert spec["velocity_model"]["learned_parameters"] == [
        "b_x",
        "b_y",
        "lambda_normal",
        "lambda_tangent",
        "kappa",
    ]
    assert spec["protocol"]["primary_comparator"] == (
        "NEX326-PHASE-SPACE-4D-CONTOUR-DISTANCE-v3"
    )
    assert "t and -t" in spec["condition_contract"]["contour_tangent_semantics"]


def test_terrain_aligned_v5_is_nested_and_has_a_stopping_rule():
    spec = load_phase_space_spec(
        NEX326 / "phase_space_terrain_aligned_residual_benchmark.json"
    )
    assert spec["velocity_model"]["kind"] == "terrain_aligned_residual_drift"
    assert spec["velocity_model"]["base_feature_basis"] == (
        "directional_terrain_contour_distance"
    )
    assert spec["velocity_model"]["learned_structural_parameters"] == [
        "lambda_normal"
    ]
    assert spec["protocol"]["primary_comparator"] == (
        "NEX326-PHASE-SPACE-4D-CONTOUR-DISTANCE-v3"
    )
    assert "do not add further terrain feature variants" in spec["protocol"][
        "stopping_rule"
    ]


def test_phase_space_benchmark_runs_without_rewriting_frozen_arms(tmp_path):
    cohort_path = NEX326 / "fixtures" / "registered_cohort.json"
    output = tmp_path / "phase-space.json"
    report = write_phase_space_report(cohort_path, output, n_samples=16)
    assert report["benchmark_id"] == "NEX326-PHASE-SPACE-4D-v1"
    assert report["scientific_role"] == "supplemental_benchmark_not_a_frozen_arm"
    assert report["state_contract"]["layout"] == ["x", "y", "vx", "vy"]
    assert report["condition_contract"]["registered_condition_names"] == []
    assert report["model"]["diffusion_state_support"] == ["vx", "vy"]
    assert report["metrics"]["evaluation_segment_count"] == 6
    assert report["metrics"]["kinematic_identity_max_error"] < 1e-10
    assert all(
        math.isfinite(value)
        for key, value in report["metrics"].items()
        if key != "evaluation_segment_count"
    )
    with pytest.raises(PhaseSpaceError, match="output already exists"):
        write_phase_space_report(cohort_path, output, n_samples=16)


def test_conditional_phase_space_rollout_requires_a_spatial_field():
    cohort = load_cohort(NEX326 / "fixtures" / "registered_cohort.json")
    model = fit_affine_velocity_model(
        cohort.splits["train"], condition_names=("solar_elev",)
    )
    segment = cohort.splits["evaluation"][0]
    with pytest.raises(PhaseSpaceError, match="spatial condition field"):
        rollout_phase_space(
            model,
            segment,
            cutoff=max(2, len(segment.time) // 2 - 1),
            n_samples=8,
            rng=np.random.default_rng(3),
        )


def test_dsde_raster_conditions_query_position_without_route_point_index(tmp_path):
    condition_root = tmp_path / "cond_slices"
    srtm_root = tmp_path / "srtm"
    srtm_root.mkdir()
    terrain = np.asarray(
        [
            [14, 15, 16, 17, 18],
            [13, 14, 15, 16, 17],
            [12, 13, 14, 15, 16],
            [11, 12, 13, 14, 15],
            [10, 11, 12, 13, 14],
        ],
        dtype=">i2",
    )
    (srtm_root / "N30E120.hgt").write_bytes(terrain.tobytes())

    splits = {}
    for split, directory in {
        "train": "zhejiang_finetune",
        "adapt": "zhejiang_finetune",
        "validation": "zhejiang_val",
        "evaluation": "zhejiang_eval",
    }.items():
        target = condition_root / directory
        target.mkdir(parents=True, exist_ok=True)
        condition_path = target / "track_cond.parquet"
        if not condition_path.exists():
            pd.DataFrame(
                {
                    "file_id": ["track", "track"],
                    "lat": [30.49, 30.51],
                    "lon": [120.49, 120.51],
                }
            ).to_parquet(condition_path, index=False)
        splits[split] = (
            Segment(
                segment_id=f"{split}:track:0_0",
                source_domain="human",
                region="zhejiang",
                time=np.arange(4, dtype=float),
                state=np.column_stack([np.arange(4, dtype=float), np.zeros(4)]),
                conditions={},
                has_terrain=True,
            ),
        )
    splits["animal_pretrain"] = ()
    cohort = Cohort(
        schema_version="nex326-cohort-v1",
        dataset_id="terrain-test",
        data_version="v1",
        purpose="test",
        splits=splits,
        unavailable_reasons={"animal_pretrain": "not used"},
        fingerprint="test",
    )
    resolver = DSDERasterConditionResolver(cohort, condition_root, srtm_root)
    field = resolver.for_segment(splits["evaluation"][0])
    values = field.evaluate(np.asarray([[0.0, 0.0], [10.0, 10.0]]), 0.0)
    assert field.names == ("terrain_elevation", "terrain_slope")
    assert values.shape == (2, 2)
    assert np.isfinite(values).all()
    assert values[0, 0] == pytest.approx(14.0)
    assert values[0, 1] > 0.0
    identity = resolver.identity()
    assert identity["future_route_point_index_used"] is False
    assert len(identity["terrain_tiles"]) == 1

    directional_resolver = DSDERasterConditionResolver(
        cohort,
        condition_root,
        srtm_root,
        names=DIRECTIONAL_TERRAIN_CONDITION_NAMES,
    )
    directional = directional_resolver.for_segment(splits["evaluation"][0]).evaluate(
        np.asarray([[0.0, 0.0]]), 0.0
    )
    assert directional.shape == (1, 3)
    assert directional[0, 1] > 0.0
    assert directional[0, 2] > 0.0
    model = fit_affine_velocity_model(
        splits["train"] + splits["adapt"],
        condition_names=DIRECTIONAL_TERRAIN_CONDITION_NAMES,
        condition_resolver=directional_resolver,
        feature_basis="directional_terrain_v2",
    )
    assert model.feature_names[-2:] == (
        "signed_uphill_speed",
        "velocity_to_contour_line_distance",
    )
    terrain_aligned = fit_terrain_aligned_velocity_model(
        splits["train"] + splits["adapt"],
        condition_names=DIRECTIONAL_TERRAIN_CONDITION_NAMES,
        condition_resolver=directional_resolver,
    )
    assert terrain_aligned.transition_count == 4
    assert terrain_aligned.to_dict()["kind"] == "terrain_aligned_projection_drift"
    assert np.linalg.eigvalsh(terrain_aligned.diffusion_covariance).min() > 0.0
    residual_model = fit_terrain_aligned_residual_model(
        splits["train"] + splits["adapt"],
        condition_names=DIRECTIONAL_TERRAIN_CONDITION_NAMES,
        condition_resolver=directional_resolver,
    )
    assert isinstance(residual_model, TerrainAlignedResidualModel)
    assert residual_model.transition_count == 4
    assert residual_model.feature_names[-1] == "velocity_to_contour_line_distance"
    assert residual_model.to_dict()["nested_baseline"].startswith(
        "lambda_normal=0"
    )
    expected_derived = {
        "directional_terrain_gradient_only": (),
        "directional_terrain_signed_uphill": ("signed_uphill_speed",),
        "directional_terrain_contour_distance": (
            "velocity_to_contour_line_distance",
        ),
    }
    for basis, derived in expected_derived.items():
        ablated = fit_affine_velocity_model(
            splits["train"] + splits["adapt"],
            condition_names=DIRECTIONAL_TERRAIN_CONDITION_NAMES,
            condition_resolver=directional_resolver,
            feature_basis=basis,
        )
        assert ablated.feature_names[5:] == derived


def test_arm17_terrain_uses_dynamic_position_lookup_without_future_route_leakage(
    tmp_path,
):
    class PositionField:
        names = ("terrain_elevation", "terrain_slope")

        def __init__(self, calls):
            self.calls = calls

        def evaluate(self, position, time):
            del time
            values = np.atleast_2d(np.asarray(position, dtype=float))
            self.calls.append(values.copy())
            return np.column_stack(
                [100.0 + 0.02 * values[:, 0], 2.0 + 0.01 * np.abs(values[:, 1])]
            )

    class PositionResolver:
        names = PositionField.names

        def __init__(self):
            self.calls = []

        def for_segment(self, segment):
            del segment
            return PositionField(self.calls)

        def identity(self):
            return {
                "schema_version": "test-position-resolver-v1",
                "names": list(self.names),
                "future_route_point_index_used": False,
            }

    source = load_cohort(NEX326 / "fixtures" / "registered_cohort.json")
    stripped_splits = {}
    for split, segments in source.splits.items():
        stripped_splits[split] = tuple(
            Segment(
                segment_id=segment.segment_id,
                source_domain=segment.source_domain,
                region=segment.region,
                time=segment.time,
                state=segment.state,
                conditions={
                    name: values
                    for name, values in segment.conditions.items()
                    if name not in PositionResolver.names
                },
                has_terrain=False,
                endpoint_prior_mean=segment.endpoint_prior_mean,
                endpoint_prior_covariance=segment.endpoint_prior_covariance,
                endpoint_prior_source=segment.endpoint_prior_source,
                endpoint_prior_derived_from_truth=segment.endpoint_prior_derived_from_truth,
            )
            for segment in segments
        )
    cohort = Cohort(
        schema_version=source.schema_version,
        dataset_id="terrain-resolver-test",
        data_version="v1",
        purpose="test",
        splits=stripped_splits,
        unavailable_reasons={},
        fingerprint="a" * 64,
    )
    resolver = PositionResolver()
    runner = NEX326Runner(
        load_experiment_spec(),
        cohort,
        tmp_path / "runs",
        n_samples=8,
        condition_resolver=resolver,
    )
    arm17 = next(arm for arm in runner.spec.arms if arm.arm_id == 17)
    terrain = next(
        item for item in arm17.subconfigs if item["subconfig_id"] == "terrain"
    )
    record = runner.run_one(arm17, terrain)

    assert record["run_status"] == "succeeded"
    assert record["dataset"]["spatial_conditions"][
        "future_route_point_index_used"
    ] is False
    assert record["runtime"]["spatial_condition_propagation"] == (
        "local_gaussian_mean_closure"
    )
    assert len(record["dataset"]["selected_segment_ids_sha256"]["evaluation"]) == 64
    assert any(call.shape == (1, 2) for call in resolver.calls)

    config = runner._config(terrain)
    prepared = {
        split: runner._resolve_spatial_conditions(segments, config["condition"])
        for split, segments in stripped_splits.items()
        if split != "animal_pretrain"
    }
    model = train_model(
        prepared["train"],
        prepared["validation"],
        prepared["adapt"],
        (),
        config,
    )
    original = prepared["evaluation"][0]
    cutoff = max(1, len(original.time) // 2 - 1)
    changed_state = original.state.copy()
    changed_state[cutoff + 1 :] += np.asarray([10_000.0, -20_000.0])
    changed_future = Segment(
        segment_id=original.segment_id,
        source_domain=original.source_domain,
        region=original.region,
        time=original.time,
        state=changed_state,
        conditions=original.conditions,
        has_terrain=True,
        endpoint_prior_mean=original.endpoint_prior_mean,
        endpoint_prior_covariance=original.endpoint_prior_covariance,
        endpoint_prior_source=original.endpoint_prior_source,
    )
    baseline_prediction = predict_segments(
        model, (original,), config, seed=41, n_samples=8, condition_resolver=resolver
    )[0]
    changed_prediction = predict_segments(
        model,
        (changed_future,),
        config,
        seed=41,
        n_samples=8,
        condition_resolver=resolver,
    )[0]
    assert np.array_equal(baseline_prediction.samples, changed_prediction.samples)
    assert not np.array_equal(baseline_prediction.target, changed_prediction.target)


def test_directional_ablation_matrix_registers_one_feature_change_per_contrast():
    matrix = json.loads(
        (NEX326 / "phase_space_directional_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    assert matrix["shared_protocol"]["replicate_seeds"] == [
        20260814,
        20260815,
        20260816,
    ]
    assert len(matrix["registered_contrasts"]) == 4
    specs = {
        item["benchmark_id"]: load_phase_space_spec(NEX326 / item["spec"])
        for item in matrix["configurations"]
        if "spec" in item
    }
    assert specs["NEX326-PHASE-SPACE-4D-GRADIENT-ONLY-v3"]["velocity_model"][
        "feature_basis"
    ] == "directional_terrain_gradient_only"
    assert specs["NEX326-PHASE-SPACE-4D-SIGNED-UPHILL-v3"]["velocity_model"][
        "features"
    ][-1] == "signed_uphill_speed"
    assert specs["NEX326-PHASE-SPACE-4D-CONTOUR-DISTANCE-v3"][
        "velocity_model"
    ]["features"][-1] == "velocity_to_contour_line_distance"


def test_phase_space_multi_seed_manifest_and_receipt_are_hash_bound(tmp_path):
    root = tmp_path / "phase-space-replicates"
    manifest = run_phase_space_replicates(
        NEX326 / "fixtures" / "registered_cohort.json",
        root,
        (101, 202),
        n_samples=8,
    )
    assert manifest["replicate_seeds"] == [101, 202]
    assert manifest["replicate_count"] == 2
    assert len(manifest["reports"]) == 2
    assert set(manifest["metric_summary"]) == set(load_phase_space_spec()["metrics"])
    receipt = write_phase_space_receipt(
        root / "phase_space_multi_seed_manifest.json",
        tmp_path / "phase-space-receipt.json",
    )
    assert receipt["metric_summary"] == manifest["metric_summary"]
    assert len(receipt["integrity"]["reports"]) == 2
    candidate_root = tmp_path / "phase-space-candidate"
    run_phase_space_replicates(
        NEX326 / "fixtures" / "registered_cohort.json",
        candidate_root,
        (101, 202),
        n_samples=8,
    )
    contrast = write_phase_space_contrast(
        root / "phase_space_multi_seed_manifest.json",
        candidate_root / "phase_space_multi_seed_manifest.json",
        tmp_path / "phase-space-contrast.json",
    )
    assert contrast["conclusion"] == "no_observed_primary_metric_gain"
    assert contrast["delta_summary"]["position_energy_score_d2"]["mean"] == 0.0
    uncertainty_protocol = {
        "schema_version": "nex326-phase-space-uncertainty-protocol-v1",
        "analysis_id": "fixture-paired-segment-bootstrap",
        "target_contrast": {
            "baseline_benchmark_id": manifest["benchmark_id"],
            "candidate_benchmark_id": manifest["benchmark_id"],
        },
        "bootstrap_unit": "paired_evaluation_segment",
        "bootstrap_iterations": 100,
        "bootstrap_seed": 17,
        "confidence_level": 0.95,
        "sampling_seed_handling": "average matched sampling-seed deltas",
        "metrics": [
            "position_energy_score_d2",
            "position_cep50_error",
            "velocity_endpoint_rmse",
        ],
    }
    protocol_path = tmp_path / "uncertainty-protocol.json"
    protocol_path.write_text(json.dumps(uncertainty_protocol), encoding="utf-8")
    uncertainty = write_phase_space_segment_bootstrap(
        root / "phase_space_multi_seed_manifest.json",
        candidate_root / "phase_space_multi_seed_manifest.json",
        tmp_path / "phase-space-contrast.json",
        tmp_path / "phase-space-uncertainty.json",
        protocol_path=protocol_path,
    )
    assert uncertainty["bootstrap_unit"] == "paired_evaluation_segment"
    assert uncertainty["evaluation_segment_count"] == 6
    assert uncertainty["uncertainty"]["position_energy_score_d2"] == {
        "candidate_minus_baseline": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "interval_excludes_zero": False,
        "bootstrap_fraction_below_zero": 0.0,
    }
    assert uncertainty["coverage_uncertainty"]["status"] == (
        "not_computable_from_compact_reports"
    )
    manifest_path = root / "phase_space_multi_seed_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["metric_summary"]["position_energy_score_d2"]["mean"] += 1.0
    tampered_path = root / "tampered-manifest.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PhaseSpaceReplicateError, match="metric summary mismatch"):
        write_phase_space_receipt(tampered_path, tmp_path / "tampered-receipt.json")
    with pytest.raises(PhaseSpaceReplicateError, match="at least two unique"):
        run_phase_space_replicates(
            NEX326 / "fixtures" / "registered_cohort.json",
            tmp_path / "invalid",
            (101, 101),
            n_samples=8,
        )


def test_critical_algorithm_paths_are_checkpointed_and_auditable(tmp_path):
    spec = load_experiment_spec()
    cohort = load_cohort(NEX326 / "fixtures" / "registered_cohort.json")
    execution = {
        (arm.arm_id, subconfig["subconfig_id"]): {
            **spec.full_components,
            **subconfig.get("components", {}),
        }
        for arm, subconfig in spec.executions
    }

    qmle = train_model(
        cohort.splits["train"],
        cohort.splits["validation"],
        cohort.splits["adapt"],
        cohort.splits["animal_pretrain"],
        execution[(8, "qmle")],
    )
    assert qmle.estimator_method == "qmle"
    assert qmle.estimator_optimization_scope == "closed_form_linear_gaussian_fit"

    for subconfig_id in ("mixed", "pure_es"):
        joint = train_model(
            cohort.splits["train"],
            cohort.splits["validation"],
            cohort.splits["adapt"],
            cohort.splits["animal_pretrain"],
            execution[(9, subconfig_id)],
        )
        assert joint.estimator_method == subconfig_id
        assert joint.estimator_optimization_scope == (
            "joint_drift_covariance_validation_grid"
        )
        assert joint.estimator_candidate_count == 20
        assert joint.estimator_drift_fraction in {0.0, 0.1, 0.25, 0.5}
        assert joint.validation_objective_after <= joint.validation_objective_before

    reptile = train_model(
        cohort.splits["train"],
        cohort.splits["validation"],
        cohort.splits["adapt"],
        cohort.splits["animal_pretrain"],
        execution[(14, "reptile")],
    )
    assert reptile.transfer_method == "meta_reptile"
    assert reptile.meta_algorithm == "first_order_reptile_gradient_inner_loop"
    assert reptile.meta_task_count >= 2
    assert reptile.meta_outer_epochs == 4
    assert reptile.meta_inner_steps == 5
    assert reptile.meta_inner_rate == 0.5
    assert reptile.meta_inner_objective_after < reptile.meta_inner_objective_before
    assert reptile.meta_inner_objective_after < reptile.meta_inner_objective_before

    report = write_fidelity_report(tmp_path / "fidelity.json")
    assert report == build_fidelity_report(spec)
    terrain = next(
        item
        for item in report["capabilities"]
        if item["capability"] == "spatial_terrain_conditioning"
    )
    assert terrain["implementation"]["future_route_point_index_used"] is False
    endpoint = next(
        item
        for item in report["capabilities"]
        if item["capability"] == "endpoint_conditioning"
    )
    assert endpoint["fidelity"].endswith("_extension")
    assert report["failed_route_count"] == 0
    assert report["overall_status"] == "bounded_reconstruction_with_declared_approximations"
    assert report["paper_equivalent"] is False


def test_pirc19_completion_audit_separates_engineering_and_scientific_status(
    tmp_path,
):
    report = write_completion_report(tmp_path / "completion.json")
    assert report == build_completion_report()
    assert report["check_summary"] == {
        "passed": 13,
        "total": 13,
        "all_passed": True,
    }
    assert report["overall_status"] == (
        "pirc19_complete_with_approved_arm_exclusions"
    )
    assert report["dimensions"]["frozen_contract_implementation"]["ratio"] == 1.0
    assert report["dimensions"]["current_dsde_execution"]["numerator"] == 31
    assert report["dimensions"]["required_empirical_reproduction_scope"] == {
        "status": "complete",
        "numerator": 28,
        "denominator": 28,
        "ratio": 1.0,
    }
    assert report["dimensions"]["replicated_current_dsde_execution"] == {
        "status": "complete_for_available_inputs",
        "numerator": 93,
        "denominator": 108,
        "ratio": 93 / 108,
    }
    assert report["dimensions"][
        "scientifically_assessed_succeeded_executions"
    ]["numerator"] == 0
    assert report["task_completion"]["status"] == "complete"
    assert report["next_core_step_requires_external_input"] is False

    broken_multi = json.loads(
        (NEX326 / "dsde_20pct_multi_seed_receipt.json").read_text(encoding="utf-8")
    )
    broken_multi["total_execution_count"] = 107
    broken_path = tmp_path / "broken-multi.json"
    broken_path.write_text(json.dumps(broken_multi), encoding="utf-8")
    with pytest.raises(CompletionAuditError, match="replicate_matrix"):
        write_completion_report(
            tmp_path / "broken-completion.json",
            multi_seed_path=broken_path,
        )

    broken_scope = json.loads(
        (NEX326 / "pirc19_scope_policy.json").read_text(encoding="utf-8")
    )
    broken_scope["approved_excluded_arms"].append(
        {"arm_id": 12, "reason": "not approved"}
    )
    broken_scope_path = tmp_path / "broken-scope.json"
    broken_scope_path.write_text(json.dumps(broken_scope), encoding="utf-8")
    with pytest.raises(CompletionAuditError, match="approved_scope_policy"):
        write_completion_report(
            tmp_path / "broken-scope-completion.json",
            scope_policy_path=broken_scope_path,
        )


def test_cli_requires_an_explicit_scientific_cohort_or_fixture(tmp_path, capsys):
    assert main(["--validate-only", "--output", str(tmp_path / "unused")]) == 0
    assert '"status": "valid"' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="2"):
        main(["--output", str(tmp_path / "runs")])


def test_external_split_unavailability_produces_an_auditable_record(tmp_path):
    fixture = json.loads(
        (NEX326 / "fixtures" / "registered_cohort.json").read_text(encoding="utf-8")
    )
    fixture["generator"]["split_counts"]["animal_pretrain"] = 0
    fixture["split_status"] = {
        "animal_pretrain": {
            "status": "unavailable",
            "reason": "licensed Movebank export is absent",
        }
    }
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(fixture), encoding="utf-8")

    spec = load_experiment_spec()
    runner = NEX326Runner.from_paths(cohort_path, tmp_path / "runs", n_samples=24)
    arm13 = next(arm for arm in spec.arms if arm.arm_id == 13)
    record = runner.run_one(arm13, arm13.subconfigs[0])

    assert record["implementation_status"] == "implemented"
    assert record["run_status"] == "data_unavailable"
    assert record["verdict"] == "unavailable"
    assert record["runtime"] == {
        "requested_prediction_samples": 24,
        "effective_prediction_samples": None,
    }
    assert record["failure"] == {
        "stage": "load_versioned_data",
        "reason": "licensed Movebank export is absent",
        "category": "external_data_unavailable",
    }
    assert record["stages"][0]["status"] == "data_unavailable"
    assert all(stage["status"] == "not_run" for stage in record["stages"][1:])
    availability = tmp_path / "runs" / record["artifacts"]["availability"]["path"]
    assert availability.is_file() and availability.stat().st_size > 0


def test_unavailable_optional_data_does_not_abort_the_36_execution_batch(tmp_path):
    fixture = json.loads(
        (NEX326 / "fixtures" / "registered_cohort.json").read_text(encoding="utf-8")
    )
    fixture["generator"]["split_counts"]["animal_pretrain"] = 0
    fixture["split_status"] = {
        "animal_pretrain": {
            "status": "unavailable",
            "reason": "licensed Movebank export is absent",
        }
    }
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(fixture), encoding="utf-8")

    runner = NEX326Runner.from_paths(cohort_path, tmp_path / "runs", n_samples=12)
    records = runner.run_all()

    assert len(records) == 36
    assert sum(record["run_status"] == "succeeded" for record in records) == 35
    unavailable = [record for record in records if record["run_status"] == "data_unavailable"]
    assert [(record["arm_id"], record["subconfig_id"]) for record in unavailable] == [
        (13, "animal_pretrain")
    ]
    manifest = json.loads((tmp_path / "runs" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["record_count"] == 36
    assert manifest["requested_prediction_samples"] == 12
    assert manifest["run_status"] == {"data_unavailable": 1, "succeeded": 35}


def test_runtime_record_validation_rejects_incomplete_success_record(tmp_path):
    runner = NEX326Runner.from_paths(
        NEX326 / "fixtures" / "registered_cohort.json",
        tmp_path / "runs",
        n_samples=24,
    )
    arm = runner.spec.arms[0]
    record = runner.run_one(arm, arm.subconfigs[0])
    record["artifacts"].pop("checkpoint")
    with pytest.raises(RunError, match="required artifacts"):
        validate_run_record(record)


def test_runner_rejects_non_positive_prediction_sample_count(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        NEX326Runner.from_paths(
            NEX326 / "fixtures" / "registered_cohort.json",
            tmp_path / "runs",
            n_samples=0,
        )


def test_strict_environment_rejects_a_nonconformant_runtime(tmp_path, monkeypatch):
    import experiments.nex326.runner as runner_module

    identity = runner_module.implementation_identity()
    identity["environment_lock"] = {
        **identity["environment_lock"],
        "conformant": False,
    }
    monkeypatch.setattr(runner_module, "implementation_identity", lambda: identity)
    with pytest.raises(RunError, match="does not conform"):
        NEX326Runner.from_paths(
            NEX326 / "fixtures" / "registered_cohort.json",
            tmp_path / "runs",
            strict_environment=True,
        )


def test_replicate_seed_changes_execution_identity_without_changing_protocol_seed(tmp_path):
    cohort = NEX326 / "fixtures" / "registered_cohort.json"
    first = NEX326Runner.from_paths(
        cohort, tmp_path / "first", n_samples=8, replicate_seed=101
    )
    second = NEX326Runner.from_paths(
        cohort, tmp_path / "second", n_samples=8, replicate_seed=202
    )
    arm = first.spec.arms[0]
    first_record = first.run_one(arm, arm.subconfigs[0])
    second_record = second.run_one(arm, arm.subconfigs[0])

    assert first_record["seed"] == second_record["seed"] == 20260814
    assert (first_record["replicate_seed"], second_record["replicate_seed"]) == (101, 202)
    assert first_record["execution_seed"] != second_record["execution_seed"]
    assert first_record["run_id"] != second_record["run_id"]
    assert first_record["result_id"] != second_record["result_id"]


def test_multi_seed_batch_requires_distinct_valid_replicates():
    assert validate_replicate_seeds([101, 202, 303]) == (101, 202, 303)
    with pytest.raises(MultiSeedError, match="at least two"):
        validate_replicate_seeds([101])
    with pytest.raises(MultiSeedError, match="unique"):
        validate_replicate_seeds([101, 101])
    with pytest.raises(MultiSeedError, match=r"\[0, 2\*\*32\)"):
        validate_replicate_seeds([101, -1])


def test_pilot_receipt_binds_records_and_rejects_status_disagreement(tmp_path):
    cohort = {
        "schema_version": "nex326-cohort-v1",
        "dataset_id": "dsde-pilot",
        "data_version": "v1",
        "purpose": "pilot",
        "source": {"trajectory_sha256": "a" * 64},
        "selection": {"sample_fraction": 0.2},
    }
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    records_root = tmp_path / "runs"
    relative_records = []
    for index in range(36):
        relative = f"run-{index:02d}/run_record.json"
        relative_records.append(relative)
        record_path = records_root / relative
        record_path.parent.mkdir(parents=True)
        record_path.write_text(
            json.dumps(
                {
                    "experiment_id": "NEX326",
                    "spec_version": "nex326-process-v2",
                    "arm_id": index % 22 + 1,
                    "subconfig_id": f"config-{index:02d}",
                    "dataset": {"dataset_id": "dsde-pilot"},
                    "runtime": {
                        "requested_prediction_samples": 64,
                        "effective_prediction_samples": 64,
                    },
                    "seed": 20260814,
                    "replicate_seed": 20260815,
                    "run_status": "succeeded",
                    "verdict": "not_assessed",
                }
            ),
            encoding="utf-8",
        )
    manifest = {
        "experiment_id": "NEX326",
        "spec_version": "nex326-process-v2",
        "record_count": 36,
        "requested_prediction_samples": 64,
        "protocol_seed": 20260814,
        "replicate_seed": 20260815,
        "run_status": {"succeeded": 36},
        "run_records": relative_records,
    }
    records_root.mkdir(exist_ok=True)
    (records_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = {
        "spec_version": "nex326-process-v2",
        "arm_count": 22,
        "execution_count": 36,
        "protocol_seed": 20260814,
        "replicate_seed": 20260815,
        "run_status": {"succeeded": 36},
        "verdict": {"not_assessed": 36},
        "mechanism_status": {"passed": 36},
        "effective_prediction_samples": {"64": 36},
        "comparison": {
            "scientific_status": "exploratory_only",
            "assessment": "not_assessed",
        },
    }
    summary_path = tmp_path / "summary.json"
    aggregate_artifact = tmp_path / "comparison.csv"
    aggregate_artifact.write_text("arm_id,delta\n", encoding="utf-8")
    summary["artifacts"] = {
        "comparison.csv": {
            "path": aggregate_artifact.name,
            "size_bytes": aggregate_artifact.stat().st_size,
            "sha256": hashlib.sha256(aggregate_artifact.read_bytes()).hexdigest(),
        }
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    receipt = build_pilot_receipt(
        cohort_path, records_root, summary_path, tmp_path / "receipt.json"
    )
    assert receipt["execution"]["requested_prediction_samples"] == 64
    assert receipt["execution"]["replicate_seed"] == 20260815
    assert receipt["execution"]["effective_prediction_samples"] == {"64": 36}
    assert receipt["execution"]["comparison"]["assessment"] == "not_assessed"
    assert len(receipt["integrity"]["run_record_set_sha256"]) == 64

    summary["run_status"] = {"succeeded": 35, "data_unavailable": 1}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(PilotReceiptError, match="status counts differ"):
        build_pilot_receipt(
            cohort_path, records_root, summary_path, tmp_path / "invalid.json"
        )


def test_dsde_zhejiang_adapter_materializes_a_disjoint_20_percent_pilot(tmp_path):
    split_files = {
        "finetune": [f"GL_TEST_20240101{hour:02d}0000_F{hour}" for hour in range(9)],
        "val": [f"GL_TEST_20240102{hour:02d}0000_V{hour}" for hour in range(3)],
        "eval": [f"GL_TEST_20240103{hour:02d}0000_E{hour}" for hour in range(3)],
    }
    rows = []
    for file_index, file_id in enumerate(
        split_files["finetune"] + split_files["val"] + split_files["eval"]
    ):
        for mode, direction in enumerate(((1.0, 0.0), (1.0, 0.5), (0.0, 1.0))):
            for step in range(6):
                rows.append(
                    {
                        "file_id": file_id,
                        "segment_id": f"{file_index}_{mode}",
                        "t": float(step * 60),
                        "x": file_index + direction[0] * step,
                        "y": -file_index + direction[1] * step,
                        "region": "Zhejiang",
                        "city": f"city-{file_index % 3}",
                    }
                )
    trajectory = tmp_path / "zhejiang_holdout.parquet"
    pd.DataFrame(rows).to_parquet(trajectory, index=False)
    split_registry = tmp_path / "zhejiang_splits.json"
    split_registry.write_text(
        json.dumps(
            {
                "splits": {
                    name: {"files": file_ids}
                    for name, file_ids in split_files.items()
                }
            }
        ),
        encoding="utf-8",
    )
    cohort_path = tmp_path / "nex326_dsde_20pct.json"

    payload = materialize_dsde_zhejiang_pilot(
        trajectory,
        split_registry,
        cohort_path,
        sample_fraction=0.2,
    )
    cohort = load_cohort(cohort_path)

    assert payload["dataset_id"] == "NEX326-DSDE-ZHEJIANG-20P-PILOT-001"
    for split in ("train", "validation", "adapt", "evaluation"):
        source_count = payload["selection"]["source_usable_segments"][split]
        assert len(cohort.splits[split]) == math.ceil(source_count * 0.2)
    segment_ids = [
        segment.segment_id
        for split in ("train", "validation", "adapt", "evaluation")
        for segment in cohort.splits[split]
    ]
    assert len(segment_ids) == len(set(segment_ids))
    assert all(
        set(segment.conditions) == {"solar_elev", "is_day"}
        for split in ("train", "validation", "adapt", "evaluation")
        for segment in cohort.splits[split]
    )

    runner = NEX326Runner(load_experiment_spec(), cohort, tmp_path / "runs", n_samples=8)
    arm17 = next(arm for arm in runner.spec.arms if arm.arm_id == 17)
    weather = next(item for item in arm17.subconfigs if item["subconfig_id"] == "weather")
    weather_record = runner.run_one(arm17, weather)
    assert weather_record["run_status"] == "data_unavailable"
    assert weather_record["failure"]["reason"] == "DSDE Zhejiang pilot has no aligned weather"

    arm22 = next(arm for arm in runner.spec.arms if arm.arm_id == 22)
    bridge_record = runner.run_one(arm22, arm22.subconfigs[0])
    assert bridge_record["run_status"] == "data_unavailable"
    assert bridge_record["failure"]["reason"] == (
        "DSDE Zhejiang pilot has no independent endpoint-prior feed"
    )

    request_path = tmp_path / "endpoint_prior_request.json"
    request = write_endpoint_prior_request(cohort_path, request_path)
    assert request["schema_version"] == "nex326-endpoint-prior-request-v1"
    assert request["cohort"]["fingerprint"] == cohort.fingerprint
    assert len(request["records"]) == len(cohort.splits["evaluation"])
    requested = request["records"][0]
    source_segment = payload["splits"]["evaluation"][0]
    assert requested["segment_id"] == source_segment["segment_id"]
    assert len(requested["observed_state"]) < len(source_segment["state"])
    assert source_segment["state"][-1] not in requested["observed_state"]
    assert request["provider_requirements"]["prohibited_inputs"] == [
        "evaluation states after the final observed_time entry",
        "evaluation endpoint targets",
    ]
    with pytest.raises(EndpointPriorError, match="output already exists"):
        write_endpoint_prior_request(cohort_path, request_path)

    prior_feed = {
        "schema_version": "nex326-endpoint-prior-v1",
        "feed_id": "independent-planning-feed-test",
        "data_version": "1.0.0",
        "independence_attestation": {
            "derived_from_evaluation_truth": False,
            "method": "held-out external route plan",
            "responsible_party": "test fixture generator",
        },
        "records": [
            {
                "segment_id": segment["segment_id"],
                "mean": [100.0 + index, -20.0 - index],
                "covariance": [[4.0, 0.25], [0.25, 3.0]],
            }
            for index, segment in enumerate(payload["splits"]["evaluation"])
        ],
    }
    prior_path = tmp_path / "endpoint_priors.json"
    prior_path.write_text(json.dumps(prior_feed), encoding="utf-8")
    oracle_feed = json.loads(json.dumps(prior_feed))
    oracle_feed["independence_attestation"]["derived_from_evaluation_truth"] = True
    oracle_path = tmp_path / "oracle_priors.json"
    oracle_path.write_text(json.dumps(oracle_feed), encoding="utf-8")
    with pytest.raises(EndpointPriorError, match="non-truth-derived"):
        attach_endpoint_priors(cohort_path, oracle_path, tmp_path / "oracle-output.json")

    incomplete_feed = json.loads(json.dumps(prior_feed))
    incomplete_feed["records"].pop()
    incomplete_path = tmp_path / "incomplete_priors.json"
    incomplete_path.write_text(json.dumps(incomplete_feed), encoding="utf-8")
    with pytest.raises(EndpointPriorError, match="coverage must exactly match"):
        attach_endpoint_priors(
            cohort_path, incomplete_path, tmp_path / "incomplete-output.json"
        )

    indefinite_feed = json.loads(json.dumps(prior_feed))
    indefinite_feed["records"][0]["covariance"] = [[1.0, 2.0], [2.0, 1.0]]
    indefinite_path = tmp_path / "indefinite_priors.json"
    indefinite_path.write_text(json.dumps(indefinite_feed), encoding="utf-8")
    with pytest.raises(EndpointPriorError, match="not positive definite"):
        attach_endpoint_priors(
            cohort_path, indefinite_path, tmp_path / "indefinite-output.json"
        )

    enriched_path = tmp_path / "nex326_dsde_20pct_with_priors.json"
    enriched = attach_endpoint_priors(cohort_path, prior_path, enriched_path)
    assert enriched["endpoint_prior_status"]["status"] == "available"
    assert enriched["source"]["endpoint_prior"]["sha256"] == hashlib.sha256(
        prior_path.read_bytes()
    ).hexdigest()
    enriched_cohort = load_cohort(enriched_path)
    enriched_runner = NEX326Runner(
        load_experiment_spec(), enriched_cohort, tmp_path / "bridge-runs", n_samples=8
    )
    bridge_records = [
        enriched_runner.run_one(
            arm22,
            {
                **subconfig,
                "components": {
                    **subconfig["components"],
                    "model": "single_gaussian",
                },
            },
        )
        for subconfig in arm22.subconfigs
    ]
    assert {record["run_status"] for record in bridge_records} == {"succeeded"}


def test_segment_modes_are_balanced_for_directionally_skewed_real_data():
    cohort = load_cohort(NEX326 / "fixtures" / "registered_cohort.json")
    skewed = tuple(
        type(segment)(
            segment_id=segment.segment_id,
            source_domain=segment.source_domain,
            region=segment.region,
            time=segment.time,
            state=segment.state * [1.0, 0.01],
            conditions=segment.conditions,
            has_terrain=segment.has_terrain,
            endpoint_prior_mean=segment.endpoint_prior_mean,
            endpoint_prior_covariance=segment.endpoint_prior_covariance,
            endpoint_prior_source=segment.endpoint_prior_source,
            endpoint_prior_derived_from_truth=segment.endpoint_prior_derived_from_truth,
        )
        for segment in cohort.splits["train"]
    )
    transitions = build_transition_data(skewed, ["solar_elev"], "segment_constant_mode")
    counts = [int((transitions.mode_labels == mode).sum()) for mode in range(3)]
    assert all(count >= 2 for count in counts)
