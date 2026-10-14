"""Synthetic origin cases; no research data, fit authorization or forecasts."""
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from data.pirc20 import PIRC20Sample, _sample_id
from experiments.pirc17 import formal_origins as module
from experiments.pirc17.formal_inputs import bind_final_prefix
from experiments.pirc17.method_development import MethodDevelopment
from experiments.pirc17.method_inputs import AssignedSample, bind_method_prefix, development_training_segment
from experiments.pirc17.protocol_core import digest, envelope
from tests.test_pirc17_formal_eligibility import metadata_fixture, check


def development_fixture(*, validation_shift=0., per_role=2):
    prefixes, segments = [], []
    population = digest("SYNTHETIC POPULATION")
    for i, role in enumerate(role for role in ("train", "adapt", "validation") for _ in range(per_role)):
        fields = dict(data_version="synthetic", file_id=f"file-{i}", segment_id=f"segment-{i}",
            split="validation" if role == "validation" else "train", independent_block_id=f"block-{i}",
            history_start=0, history_end=2, target_start=3, target_end=63)
        sample = PIRC20Sample(sample_id=_sample_id(fields), factor_availability={}, **fields)
        times = 1_700_000_000_000_000_000+np.arange(64, dtype=np.int64)*60_000_000_000
        ticks = np.arange(64)
        lonlat = np.column_stack((110.+ticks*(i+1)*.0001+np.sin(ticks)*.00003,
                                 35.+ticks*(i+2)*.00008+np.cos(ticks)*.00004))
        if role == "validation": lonlat[:, 0] += validation_shift*np.arange(64)
        prefix = bind_method_prefix(AssignedSample(sample, role, population), times[:3], lonlat[:3],
                                    source_identity=digest(["software", i]))
        segment, _ = development_training_segment(prefix, times, lonlat, region=f"fixture-city-{i%2}",
                                                   region_identity=digest("synthetic-regions"))
        prefixes.append(prefix); segments.append(segment)
    order = sorted(range(len(prefixes)), key=lambda i: prefixes[i].assignment.sample.sample_id)
    prefixes, segments = tuple(prefixes[i] for i in order), tuple(segments[i] for i in order)
    counts = dict(Counter(p.assignment.method_role for p in prefixes))
    identity = dict(population_identity=population, sample_counts=counts,
        sample_ids=[p.assignment.sample.sample_id for p in prefixes], frame_policy="software-fixture", solar_policy="software-fixture")
    identity["sha256"] = digest(identity)
    prepared = MethodDevelopment(prefixes, segments, identity)
    contract = dict(population_identity=population, method_roles=counts, outer_train_windows=2*per_role, validation_windows=per_role,
                    methods={"max_transitions_per_fit": 80000})
    return prepared, contract


def prior_fixture():
    prepared, contract = development_fixture()
    return module.build_velocity_prior(prepared, training_contract=contract,
                                       expected_input_sha256=prepared.identity["sha256"])


def final_fixture(n=8, *, change_hidden_history=False):
    windows, lonlats, selected = [], [], []
    for i in range(n):
        sample, rows = metadata_fixture(f"final-{i}")
        _, window = check(sample, rows)
        lonlat = np.array([[110., 35.], [110.001, 35.001], [110.002, 35.002]])
        if change_hidden_history:
            lonlat[:2] -= [1., .5]
        windows.append(window); lonlats.append(lonlat)
        selected.append({k: sample[k] for k in ("sample_id", "independent_block_id", "split")})
    population = envelope({"selection": {"selected": selected, "secondary_selected": selected[:6]}})
    prefixes = tuple(bind_final_prefix(w, xy, population_sha256=population["sha256"], condition_sha256="4"*64)
                     for w, xy in zip(windows, lonlats))
    return prefixes, population


def test_prior_uses_all_outer_train_and_no_validation_or_targets():
    prepared, contract = development_fixture()
    prior = module.build_velocity_prior(prepared, training_contract=contract, expected_input_sha256=prepared.identity["sha256"])
    record = module.validate_prior(prior)
    assert len(prior.prior.velocities_mps) == 4
    assert Counter(r["method_role"] for r in record["training_rows"]) == {"train": 2, "adapt": 2}
    assert all(r["outer_split"] == "train" for r in record["training_rows"])
    assert record["target_coordinates_used"] is False and record["validation_used"] is False
    changed, _ = development_fixture(validation_shift=.05)
    # Targets are deliberately no longer correlated with their prefix. The
    # prior must not inspect any future segment states to derive velocity.
    changed = replace(changed, segments=tuple(replace(s, state=s.state+1000) for s in changed.segments))
    second = module.build_velocity_prior(changed, training_contract=contract, expected_input_sha256=changed.identity["sha256"])
    np.testing.assert_array_equal(prior.prior.velocities_mps, second.prior.velocities_mps)
    for row, vector in zip(record["training_rows"], prior.prior.velocities_mps):
        np.testing.assert_array_equal(vector, row["velocity_east_north_mps"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "input", "population", "velocity", "role"])
def test_prior_rejects_partial_relabelled_or_uncausal_training(change):
    prepared, contract = development_fixture()
    expected = prepared.identity["sha256"]
    if change == "missing": prepared = replace(prepared, prefixes=prepared.prefixes[:-1])
    elif change == "duplicate": prepared = replace(prepared, prefixes=prepared.prefixes+(prepared.prefixes[0],))
    elif change == "input": expected = "0"*64
    elif change == "population": contract = {**contract, "population_identity": "0"*64}
    else:
        i = next(i for i, p in enumerate(prepared.prefixes) if p.assignment.sample.split == "train")
        prefixes = list(prepared.prefixes); p = prefixes[i]
        if change == "velocity": prefixes[i] = replace(p, origin=replace(p.origin, velocity_mps=p.origin.velocity_mps+1))
        else:
            # Frozen role object bypassed deliberately to exercise this boundary.
            assignment = object.__new__(type(p.assignment))
            object.__setattr__(assignment, "sample", replace(p.assignment.sample, split="final_eval"))
            object.__setattr__(assignment, "method_role", "train")
            object.__setattr__(assignment, "population_identity", p.assignment.population_identity)
            prefixes[i] = replace(p, assignment=assignment)
        prepared = replace(prepared, prefixes=tuple(prefixes))
    with pytest.raises(ValueError):
        module.build_velocity_prior(prepared, training_contract=contract, expected_input_sha256=expected)


def test_all_three_modes_keep_exact_primary_and_first_six_secondary_ranks():
    prefixes, population = final_fixture()
    prior = prior_fixture()
    cases = module.build_origin_cases(prefixes, population, prior=prior)
    assert {k: len(v) for k, v in cases.items()} == {"causal_prefix": 8, "known_velocity": 6, "point_only": 6}
    for mode, group in cases.items():
        assert [c.sample_id for c in group] == [p.sample_id for p in prefixes[:len(group)]]
        for case in group:
            assert case.mode == case.method_origin.mode == case.terrain_origin.mode == mode
            assert case.method_origin.velocity_error_mps is None
            assert not case.score_seconds.flags.writeable
            if mode != "causal_prefix":
                assert len(case.method_origin.history_times_seconds) == 1
                np.testing.assert_array_equal(case.method_origin.position_m, [0, 0])
                assert case.condition_at.frame == case.scoring_frame
                assert case.identity()["role"] == "secondary-descriptive"
            if mode == "point_only":
                np.testing.assert_array_equal(case.method_origin.velocity_prior_mps, prior.prior.velocities_mps)
                assert case.prior_identity == prior.prior.identity
    np.testing.assert_array_equal(cases["known_velocity"][0].terrain_origin.velocity_mps, prefixes[0].terrain_origin.velocity_mps)


def test_hidden_prefix_cannot_leak_into_point_only_through_frame_or_velocity():
    prefix_a, population = final_fixture(1)
    prefix_b, _ = final_fixture(1, change_hidden_history=True)
    prior = prior_fixture()
    a = module.build_origin_cases(prefix_a, population, prior=prior)
    b = module.build_origin_cases(prefix_b, population, prior=prior)
    assert a["point_only"][0].identity() == b["point_only"][0].identity()
    np.testing.assert_array_equal(a["point_only"][0].method_origin.sample_velocities(40, np.random.default_rng(1)),
                                  b["point_only"][0].method_origin.sample_velocities(40, np.random.default_rng(1)))
    assert not np.array_equal(a["known_velocity"][0].method_origin.velocity_mps, b["known_velocity"][0].method_origin.velocity_mps)


@pytest.mark.parametrize("change", ["order", "omission", "secondary", "population", "prior"])
def test_case_population_is_not_replaced_or_expanded(change):
    prefixes, population = final_fixture()
    prior = prior_fixture()
    if change == "order": prefixes = prefixes[::-1]
    elif change == "omission": prefixes = prefixes[:-1]
    elif change == "secondary":
        payload = population["payload"]; payload["selection"]["secondary_selected"] = payload["selection"]["selected"][:7]
        population = envelope(payload)
    elif change == "population": prefixes = (replace(prefixes[0], population_sha256="0"*64),)+prefixes[1:]
    else: prior = replace(prior, prior=replace(prior.prior, velocities_mps=prior.prior.velocities_mps+1))
    with pytest.raises(ValueError): module.build_origin_cases(prefixes, population, prior=prior)
