"""Software fixtures only: never empirical qualification or invented research."""
from dataclasses import replace
from datetime import datetime, timezone
import calendar
import math

import numpy as np
import pytest

from data.pirc20 import PIRC20Sample
from experiments.nex326.pirc20_adapter import _roles
from experiments.pirc17.features import LocalFrame
from experiments.pirc17.method_inputs import (
    AssignedSample, SolarConditionField, assign_development_samples,
    bind_method_prefix, development_training_segment, positions_in_scoring_frame,
    resample_training_segment, solar_conditions,
)
from experiments.pirc17.method_rollout import forecast_method
from tests.test_pirc17_method_rollout import bound, fitted


EPOCH = 1_700_000_000_000_000_000


def sample(index=0, split="train"):
    return PIRC20Sample(f"sample-{index}", "software-fixture", f"file-{index}", f"segment-{index}",
                       split, f"block-{index}", 0, 2, 3, 5, {})


def fixture(names=("solar_elev",)):
    assignment = AssignedSample(sample(), "train", "software-only-population")
    times = EPOCH + np.array([0, 5, 10, 17, 22, 29], dtype=np.int64) * 10**9
    lonlat = np.array([[114., 22.], [114.0001, 22.0001], [114.0002, 22.0002],
                       [114.0004, 22.0004], [114.0007, 22.0007], [114.001, 22.001]])
    prefix = bind_method_prefix(assignment, times[:3], lonlat[:3],
                               source_identity="software-fixture-not-research", condition_names=names)
    return prefix, times, lonlat


def segment_fixture(names=("solar_elev",)):
    prefix, times, lonlat = fixture(names)
    segment, report = development_training_segment(prefix, times, lonlat,
        region="software-region", region_identity="software-region-provenance")
    return prefix, segment, report


def test_roles_use_full_global_training_blocks_before_subsetting():
    samples = tuple(sample(i) for i in range(20)) + (sample(20, "validation"), sample(21, "final_eval"))
    expected = _roles(samples, False)
    adapt = next(s for s in samples if expected.get(s.segment_id) == "adapt")
    train = next(s for s in samples if expected.get(s.segment_id) == "train")
    selected = [adapt.sample_id, train.sample_id, "sample-20"]
    assigned = assign_development_samples(samples, selected)
    assert {a.sample.sample_id: a.method_role for a in assigned} == {
        s: expected[next(x.segment_id for x in samples if x.sample_id == s)] for s in selected}
    assert len({a.population_identity for a in assigned}) == 1
    assert [a.population_identity for a in assigned] == [
        a.population_identity for a in assign_development_samples(samples[::-1], selected[::-1])]
    with pytest.raises(ValueError, match="forbid final-eval"):
        assign_development_samples(samples, ["sample-21"])


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "empty", "block_overlap", "file_overlap"])
def test_role_population_rejects_ambiguous_or_cross_split_identity(kind):
    samples = [sample(0), sample(1), sample(2, "validation")]
    selected = ["sample-0"]
    if kind == "duplicate":
        selected *= 2
    elif kind == "unknown":
        selected = ["missing"]
    elif kind == "empty":
        selected = []
    elif kind == "block_overlap":
        samples[2] = replace(samples[2], independent_block_id="block-0")
    else:
        samples[2] = replace(samples[2], file_id="file-0")
    with pytest.raises(ValueError):
        assign_development_samples(samples, selected)


def test_visible_frame_clock_velocity_and_provider_are_explicit():
    prefix, times, lonlat = fixture()
    assert prefix.condition_at.origin_epoch_ns == int(times[2])
    assert prefix.origin.epoch_seconds == 0
    assert prefix.condition_at.frame == LocalFrame(*lonlat[0])
    np.testing.assert_allclose(prefix.condition_at.frame.to_lonlat(prefix.origin.position_m), lonlat[2], atol=1e-12)
    np.testing.assert_array_equal(prefix.origin.history_times_seconds, [-10, -5, 0])
    np.testing.assert_allclose(prefix.origin.velocity_mps, (prefix.visible_positions_m[2]-prefix.visible_positions_m[0])/10)
    expected = solar_conditions(lonlat[2:3], times[2:3])
    np.testing.assert_array_equal(prefix.condition_at(prefix.origin.position_m[None], 0), expected)
    wrong_1970 = solar_conditions(lonlat[2:3], np.array([0], dtype=np.int64))
    assert not np.allclose(expected, wrong_1970)
    assert not prefix.identity()["historical_fit_reuse_qualified"]


def test_prefix_is_copied_and_forecast_does_not_capture_future_conditions():
    prefix, times, lonlat = fixture()
    original = prefix.identity()
    times[:] += 999_000_000_000
    lonlat[:] += 1
    assert prefix.identity() == original
    assert set(vars(prefix.condition_at)) == {"frame", "origin_epoch_ns", "names"}
    assert not prefix.visible_epoch_ns.flags.writeable
    assert not prefix.visible_positions_m.flags.writeable
    assert len(prefix.visible_epoch_ns) == 3


@pytest.mark.parametrize("propagation", ["fp", "mc", "crn"])
def test_actual_method_forecast_uses_bound_absolute_solar_without_any_segment(propagation):
    prefix, _, _ = fixture()
    model = fitted(names=("solar_elev",))
    weights = model.weights[0].copy()
    weights[3, 0] = .02
    model = replace(model, weights=(weights,))
    settings = dict(propagation=propagation, integrator="exact", particles=8, seed=3,
        max_step_seconds=5., history_step_seconds=5., max_steps=10, max_particle_steps=80,
        origin_id="fixture", run_id="fixture", condition_names=("solar_elev",),
        condition_at=prefix.condition_at, crn_pair_id="fixture-pair" if propagation == "crn" else None)
    result = forecast_method(bound(model), prefix.origin, [5., 10.], **settings)
    assert result.forecast.positions_m.shape == (8, 2, 2)
    wrong_clock = replace(prefix.condition_at, origin_epoch_ns=0)
    wrong = forecast_method(bound(model), prefix.origin, [5., 10.], **{**settings, "condition_at": wrong_clock})
    assert not np.allclose(result.forecast.positions_m, wrong.forecast.positions_m)


def test_training_and_prediction_share_frame_and_solar_not_historical_columns():
    prefix, segment, report = segment_fixture(("solar_elev", "is_day"))
    for i, relative in enumerate(segment.time):
        actual = prefix.condition_at(segment.state[i:i+1], relative)[0]
        np.testing.assert_array_equal(actual, [segment.conditions[n][i] for n in prefix.condition_at.names])
    np.testing.assert_array_equal(segment.state[:3], prefix.visible_positions_m)
    assert segment.region == "software-region"
    assert report["stored_future_conditions_used"] is False
    assert report["training_policy_sealed"] is False


def test_training_resampling_retains_partial_tail_and_recomputes_solar():
    prefix, segment, _ = segment_fixture()
    sampled, report = resample_training_segment(prefix, segment, 10.)
    np.testing.assert_array_equal(sampled.time, [-10., 0., 10., 19.])
    assert report["transition_count"] == 3
    assert report["short_tail_count"] == 1
    assert report["minimum_interval_seconds"] == 9.
    assert report["solar_recomputed_not_interpolated"]
    assert not report["noise_embedding_qualified"]
    for i, t in enumerate(sampled.time):
        np.testing.assert_array_equal(prefix.condition_at(sampled.state[i:i+1], t)[0],
                                      [sampled.conditions["solar_elev"][i]])
    full, exact = resample_training_segment(prefix, segment, 1.)
    assert exact["short_tail_count"] == 0
    assert full.time[-1] == segment.time[-1]


@pytest.mark.parametrize("field", ["state", "time", "conditions"])
def test_resampling_cannot_silently_reuse_legacy_frame_clock_or_condition_schema(field):
    prefix, segment, _ = segment_fixture()
    bad = replace(segment, **{field: ({} if field == "conditions" else getattr(segment, field)+1.)})
    with pytest.raises(ValueError, match="bound visible"):
        resample_training_segment(prefix, bad, 10.)


def test_common_scoring_frame_conversion_preserves_origin_and_geography():
    prefix, _, lonlat = fixture()
    scoring = LocalFrame(*lonlat[2])
    np.testing.assert_allclose(positions_in_scoring_frame(prefix, prefix.origin.position_m, scoring), [0, 0], atol=1e-8)
    predicted = prefix.condition_at.frame.from_lonlat(lonlat[3:])[None]
    np.testing.assert_allclose(positions_in_scoring_frame(prefix, predicted, scoring),
                              scoring.from_lonlat(lonlat[3:])[None], atol=1e-8)


def noaa_scalar_oracle(longitude, latitude, epoch_ns):
    # Independent scalar/calendar implementation of the published NOAA sheet;
    # software arithmetic oracle, not measured astronomical/research data.
    timestamp = datetime.fromtimestamp(epoch_ns // 10**9, timezone.utc)
    hours = timestamp.hour + timestamp.minute/60 + (timestamp.second + (epoch_ns % 10**9)/1e9)/3600
    gamma = 2*math.pi/(366 if calendar.isleap(timestamp.year) else 365) * (
        timestamp.timetuple().tm_yday-1+(hours-12)/24)
    eqtime = 229.18*(.000075+.001868*math.cos(gamma)-.032077*math.sin(gamma)
        -.014615*math.cos(2*gamma)-.040849*math.sin(2*gamma))
    decl = .006918-.399912*math.cos(gamma)+.070257*math.sin(gamma)-.006758*math.cos(2*gamma)+.000907*math.sin(2*gamma)-.002697*math.cos(3*gamma)+.00148*math.sin(3*gamma)
    ha = math.radians((hours*60+eqtime+4*longitude)/4-180)
    lat = math.radians(latitude)
    return math.degrees(math.asin(math.sin(lat)*math.sin(decl)+math.cos(lat)*math.cos(decl)*math.cos(ha)))


def test_solar_matches_published_noaa_sheet_scalar_calendar_oracle():
    epochs = EPOCH + np.array([0, 3600, 86400, -86400], dtype=np.int64)*10**9
    coords = np.array([[0, 0], [120, 30], [-70, -30], [20, 60]])
    expected = np.array([noaa_scalar_oracle(*xy, int(t)) for xy, t in zip(coords, epochs)])
    actual = solar_conditions(coords, epochs, ("is_day", "solar_elev"))
    np.testing.assert_allclose(actual[:, 1], expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual[:, 0], (expected >= 0).astype(float))


def test_utc_subseconds_and_leap_year_boundaries_are_not_rounded_away():
    midnight = int(np.datetime64("2024-01-01T00:00:00", "ns").astype(np.int64))
    epochs = np.array([midnight-500_000_000, midnight,
                       midnight+250_000_000, midnight+500_000_000, midnight+1_500_000_000], dtype=np.int64)
    coords = np.tile([114., 22.], (5, 1))
    expected = np.array([noaa_scalar_oracle(*xy, int(t)) for xy, t in zip(coords, epochs)])
    actual = solar_conditions(coords, epochs)[:, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert abs(actual[2]-actual[1]) > 1e-6
    dates = np.array(['2000-02-29T12:15:30.125', '2023-12-31T23:59:59.875',
                     '2024-02-29T12:15:30.125', '2100-03-01T12:15:30.125'], dtype='datetime64[ns]').astype(np.int64)
    coords = np.tile([-75., 40.], (4, 1))
    expected = [noaa_scalar_oracle(*xy, int(t)) for xy, t in zip(coords, dates)]
    np.testing.assert_allclose(solar_conditions(coords, dates)[:, 0], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("mutation", ["float_time", "future_in_prefix", "duplicate_time", "bad_lat", "empty_source", "bad_role"])
def test_binding_rejects_malformed_or_future_prefix(mutation):
    prefix, times, coords = fixture()
    assignment = prefix.assignment
    times, coords, identity = times[:3].copy(), coords[:3].copy(), "fixture"
    if mutation == "float_time": times = times.astype(float)
    elif mutation == "future_in_prefix": times, coords = np.append(times, times[-1]+10**9), np.vstack([coords, coords[-1]])
    elif mutation == "duplicate_time": times[1] = times[0]
    elif mutation == "bad_lat": coords[0, 1] = 91
    elif mutation == "empty_source": identity = ""
    else:
        with pytest.raises(ValueError, match="forbid"):
            AssignedSample(replace(assignment.sample, split="final_eval"), "train", "fixture")
        return
    with pytest.raises(ValueError):
        bind_method_prefix(assignment, times, coords, source_identity=identity)


@pytest.mark.parametrize("bad", [np.nan, np.inf, True, -1., 0.])
def test_training_interval_requires_positive_finite_explicit_value(bad):
    prefix, segment, _ = segment_fixture()
    with pytest.raises(ValueError):
        resample_training_segment(prefix, segment, bad)


def test_segment_binding_rejects_changed_prefix_and_unknown_source_region():
    prefix, times, coords = fixture()
    for changed_times, changed_coords, region in [(times+1, coords, "r"), (times, coords+.1, "r"), (times, coords, "")]:
        with pytest.raises(ValueError):
            development_training_segment(prefix, changed_times, changed_coords, region=region, region_identity="fixture")


@pytest.mark.parametrize("names", [("weather",), ("solar_elev", "solar_elev")])
def test_unknown_condition_schema_is_not_silently_filled(names):
    with pytest.raises(ValueError):
        fixture(names)
