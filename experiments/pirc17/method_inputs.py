"""Development-only causal geography/clock inputs for the method matrix.

This is a candidate preprocessing policy, not a sealed training population or a
fit-compatibility certificate. Reusing file-mean-centred historical coefficients
is NOT justified by this adapter. Training and prediction use the same declared
NOAA solar kernel; stored historical solar columns are not silently equated to
it. All real final-eval loading remains outside this development-only interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from data.pirc20 import PIRC20Sample
from experiments.nex326.cohort import Segment, resample_segment
from experiments.nex326.pirc20_adapter import ADAPT_FRACTION, ADAPT_SEED, _roles
from .features import EARTH_RADIUS_M, LocalFrame
from .origins import Origin, causal_prefix, frozen_array

VERSION = "pirc17-method-development-inputs-v1"
FRAME_POLICY = "first-visible-history-point-equirectangular-v1"
SOLAR_POLICY = "noaa-fractional-year-geometric-utc-subsecond-v1"
SOLAR_REFERENCE = "https://www.gml.noaa.gov/grad/solcalc/solareqns.PDF"
NANOSECONDS = 1_000_000_000
DEVELOPMENT_ROLES = {"train", "adapt", "validation"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {name} required")
    return value


def _epochs(value):
    values = np.asarray(value)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
        raise ValueError("nonempty integer UTC nanosecond vector required")
    limits = np.iinfo(np.int64)
    if any(not limits.min < int(x) <= limits.max for x in values):
        raise ValueError("UTC nanoseconds outside supported int64 range")
    return np.array(values, dtype=np.int64, copy=True)


def _lonlat(value, count):
    values = frozen_array(value, (count, 2))
    if np.any(np.abs(values[:, 0]) > 180) or np.any(np.abs(values[:, 1]) >= 89):
        raise ValueError("coordinates outside supported nonpolar WGS84 extent")
    return values


def _elapsed(epochs, origin_epoch_ns):
    # Subtract Python integers first: neither int64 overflow nor loss of the
    # source's subsecond differences from subtracting large float epochs.
    return np.array([(int(t) - origin_epoch_ns) / NANOSECONDS for t in epochs])


@dataclass(frozen=True)
class AssignedSample:
    sample: PIRC20Sample
    method_role: str
    population_identity: str

    def __post_init__(self):
        if self.method_role not in DEVELOPMENT_ROLES or self.sample.split not in {"train", "validation"}:
            raise ValueError("method development inputs forbid final-eval")
        if (self.sample.split == "validation") != (self.method_role == "validation"):
            raise ValueError("outer and method role disagree")
        _identity(self.population_identity, "population identity")


def assign_development_samples(all_samples, selected_sample_ids):
    """Apply the ORIGINAL global block split before selecting any pilot subset.

    all_samples is immutable release metadata, not trajectory/condition values.
    No final-eval sample can be selected, and no role is inferred from a source
    directory name such as the legacy cond_slices/eval directory.
    """
    samples = tuple(all_samples)
    selected = tuple(selected_sample_ids)
    by_id = {s.sample_id: s for s in samples}
    if not samples or len(by_id) != len(samples) or len({s.segment_id for s in samples}) != len(samples):
        raise ValueError("unique release sample and segment identities required")
    block_splits = {}
    file_splits = {}
    for sample in samples:
        if sample.split not in {"train", "validation", "final_eval"}:
            raise ValueError("unknown outer split")
        for identity, registry in ((sample.independent_block_id, block_splits), (sample.file_id, file_splits)):
            if registry.setdefault(identity, sample.split) != sample.split:
                raise ValueError("independent block/file crosses outer splits")
    if not selected or len(set(selected)) != len(selected) or any(s not in by_id for s in selected):
        raise ValueError("unique known selected samples required")
    if any(by_id[s].split == "final_eval" for s in selected):
        raise ValueError("method development inputs forbid final-eval")
    roles = _roles(samples, final_eval_unlocked=False)
    identity = _digest({"version": VERSION, "adapt_seed": ADAPT_SEED,
        "adapt_fraction": ADAPT_FRACTION,
        "global_train_blocks": sorted(b for b, split in block_splits.items() if split == "train"),
        "selected": [[s, by_id[s].segment_id, by_id[s].independent_block_id,
                      by_id[s].split, roles[by_id[s].segment_id]] for s in sorted(selected)]})
    return tuple(AssignedSample(by_id[s], roles[by_id[s].segment_id], identity) for s in sorted(selected))


def solar_conditions(lonlat, epoch_ns, names=("solar_elev",)):
    """NOAA fractional-year geometric elevation, not a refraction correction.

    Uses east-positive longitude, UTC (timezone=0), actual 365/366-day years and
    fractional hours. Integer day decomposition preserves source subseconds.
    Reference: SOLAR_REFERENCE. Does not certify historical generator parity or
    ephemeris accuracy; no constant is selected from trajectory outcomes.
    """
    epochs = _epochs(epoch_ns)
    positions = _lonlat(lonlat, len(epochs))
    names = tuple(names)
    if len(set(names)) != len(names) or any(n not in {"solar_elev", "is_day"} for n in names):
        raise ValueError("unsupported or duplicate solar condition names")
    days, remainder = np.divmod(epochs, 86400 * NANOSECONDS)
    calendar_days = days.astype("datetime64[D]")
    years = calendar_days.astype("datetime64[Y]")
    year_start = years.astype("datetime64[D]")
    days_in_year = ((years + np.timedelta64(1, "Y")).astype("datetime64[D]") - year_start).astype(int)
    day_of_year = (calendar_days - year_start).astype(int) + 1
    hour = remainder.astype(float) / (3600 * NANOSECONDS)
    gamma = 2 * np.pi / days_in_year * (day_of_year - 1 + (hour - 12) / 24)
    equation_minutes = 229.18 * (.000075 + .001868 * np.cos(gamma) - .032077 * np.sin(gamma)
                               - .014615 * np.cos(2 * gamma) - .040849 * np.sin(2 * gamma))
    declination = (.006918 - .399912 * np.cos(gamma) + .070257 * np.sin(gamma)
                   - .006758 * np.cos(2 * gamma) + .000907 * np.sin(2 * gamma)
                   - .002697 * np.cos(3 * gamma) + .00148 * np.sin(3 * gamma))
    hour_angle = np.deg2rad((hour * 60 + equation_minutes + 4 * positions[:, 0]) / 4 - 180)
    latitude = np.deg2rad(positions[:, 1])
    sine = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    elevation = np.rad2deg(np.arcsin(np.clip(sine, -1., 1.)))
    columns = {"solar_elev": elevation, "is_day": (elevation >= 0).astype(float)}
    return np.column_stack([columns[name] for name in names]) if names else np.empty((len(epochs), 0))


@dataclass(frozen=True)
class SolarConditionField:
    """Holds geography and clock only: no trajectory, targets or stored solar."""
    frame: LocalFrame
    origin_epoch_ns: int
    names: tuple[str, ...] = ("solar_elev",)

    def __post_init__(self):
        if not isinstance(self.frame, LocalFrame) or type(self.origin_epoch_ns) is not int:
            raise ValueError("explicit local frame and integer UTC origin required")
        _epochs([self.origin_epoch_ns])
        object.__setattr__(self, "names", tuple(self.names))
        solar_conditions([[self.frame.longitude, self.frame.latitude]], [self.origin_epoch_ns], self.names)

    def __call__(self, predicted_positions_m, relative_seconds):
        if not np.isscalar(relative_seconds) or isinstance(relative_seconds, (bool, np.bool_)) or not np.isfinite(relative_seconds):
            raise ValueError("finite scalar time relative to the bound origin required")
        positions = np.asarray(predicted_positions_m)
        if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions):
            raise ValueError("nonempty predicted position matrix required")
        epoch = self.origin_epoch_ns + int(round(float(relative_seconds) * NANOSECONDS))
        _epochs([epoch])
        return solar_conditions(self.frame.to_lonlat(positions), np.full(len(positions), epoch, dtype=np.int64), self.names)

    def identity(self):
        return {"solar_policy": SOLAR_POLICY, "time_input": "seconds relative to explicit UTC origin",
                "solar_reference": SOLAR_REFERENCE,
                "origin_epoch_ns": self.origin_epoch_ns, "condition_names": list(self.names),
                "frame_longitude": self.frame.longitude, "frame_latitude": self.frame.latitude,
                "earth_radius_m": EARTH_RADIUS_M, "historical_solar_parity_claimed": False}


@dataclass(frozen=True)
class MethodPrefix:
    assignment: AssignedSample
    origin: Origin
    condition_at: SolarConditionField
    visible_epoch_ns: np.ndarray
    visible_positions_m: np.ndarray
    source_identity: str

    def identity(self):
        return {"version": VERSION, "sample_id": self.assignment.sample.sample_id,
                "block_id": self.assignment.sample.independent_block_id,
                "method_role": self.assignment.method_role, "population_identity": self.assignment.population_identity,
                "source_identity": self.source_identity, "frame_policy": FRAME_POLICY,
                "condition_provider": self.condition_at.identity(),
                "visible_sha256": hashlib.sha256(self.visible_epoch_ns.astype("<i8").tobytes()
                    + self.visible_positions_m.astype("<f8").tobytes()).hexdigest(),
                "historical_fit_reuse_qualified": False, "formal_training_accepted": False}


def bind_method_prefix(assignment, visible_epoch_ns, visible_lonlat, *, source_identity,
                       condition_names=("solar_elev",)):
    """Accept ONLY visible observations, never full-route data or target arrays.

    The first visible point fixes the training/prediction frame. The forecast
    origin is still the SAME final visible observation, with the last-three
    causal secant. Its clock remains relative zero; the solar provider binds UTC.
    """
    if not isinstance(assignment, AssignedSample):
        raise ValueError("release-bound development assignment required")
    _identity(source_identity, "source identity")
    times = _epochs(visible_epoch_ns)
    sample = assignment.sample
    if len(times) != sample.history_end - sample.history_start + 1 or len(times) < 2:
        raise ValueError("visible observations differ from sample history bounds")
    if any(int(b) <= int(a) for a, b in zip(times, times[1:])):
        raise ValueError("visible timestamps must strictly increase; no implicit duplicate removal")
    lonlat = _lonlat(visible_lonlat, len(times))
    frame = LocalFrame(*lonlat[0])
    positions = frozen_array(frame.from_lonlat(lonlat))
    epoch = int(times[-1])
    origin = causal_prefix(positions[-3:], _elapsed(times[-3:], epoch))
    times.setflags(write=False)
    return MethodPrefix(assignment, origin, SolarConditionField(frame, epoch, tuple(condition_names)),
                        times, positions, source_identity)


def development_training_segment(prefix, epoch_ns, lonlat, *, region, region_identity):
    """Build a train/adapt/calibration segment in the SAME frame/solar policy.

    The caller supplies bound real region metadata (used by Reptile), never a
    fabricated geographic task. Raw observations are not resampled here. This
    object belongs to fitting/calibration, NOT to forecast_method's input API.
    """
    if not isinstance(prefix, MethodPrefix):
        raise ValueError("bound method prefix required")
    _identity(region, "source region")
    _identity(region_identity, "region provenance")
    times = _epochs(epoch_ns)
    sample = prefix.assignment.sample
    if len(times) != sample.target_end - sample.history_start + 1:
        raise ValueError("development observations differ from sample bounds")
    n = len(prefix.visible_epoch_ns)
    if not np.array_equal(times[:n], prefix.visible_epoch_ns):
        raise ValueError("training prefix timestamps differ from forecast prefix")
    positions = _lonlat(lonlat, len(times))
    xy = prefix.condition_at.frame.from_lonlat(positions)
    if not np.array_equal(xy[:n], prefix.visible_positions_m):
        raise ValueError("training prefix coordinates differ from forecast prefix")
    conditions = solar_conditions(positions, times, prefix.condition_at.names)
    segment = Segment(sample.segment_id, "human", region,
        frozen_array(_elapsed(times, prefix.condition_at.origin_epoch_ns)), frozen_array(xy),
        {name: frozen_array(conditions[:, i]) for i, name in enumerate(prefix.condition_at.names)}, False)
    segment.validate()
    return segment, {"input": prefix.identity(), "region_identity": region_identity,
        "region": region, "observed_points": len(times), "condition_policy": SOLAR_POLICY,
        "stored_future_conditions_used": False, "training_policy_sealed": False}


def resample_training_segment(prefix, segment, interval_seconds):
    """Retain linear state interpolation, explicitly report short tail steps.

    This is training preprocessing only; interpolated positions must never serve
    as evaluation truth. The report does not qualify Q=R*tau for irregular data.
    Solar is recomputed at the resampled positions/times, NOT interpolated from
    stored solar. This is another explicit incompatibility with historical fits.
    """
    if not isinstance(prefix, MethodPrefix) or segment.segment_id != prefix.assignment.sample.segment_id:
        raise ValueError("training segment and bound prefix identity differ")
    if isinstance(interval_seconds, (bool, np.bool_)) or not np.isscalar(interval_seconds) or not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("positive finite training interval required")
    segment.validate()
    visible_count = len(prefix.visible_epoch_ns)
    if (not np.array_equal(segment.state[:visible_count], prefix.visible_positions_m)
        or not np.array_equal(segment.time[:visible_count],
                              _elapsed(prefix.visible_epoch_ns, prefix.condition_at.origin_epoch_ns))
        or tuple(segment.conditions) != prefix.condition_at.names):
        raise ValueError("training segment does not share the bound visible frame/clock/conditions")
    sampled = resample_segment(segment, float(interval_seconds))
    if len(sampled.time) < 2:
        raise ValueError("training grid has no transition")
    intervals = np.diff(sampled.time)
    if np.any(intervals <= 0):
        raise ValueError("training grid is not strictly increasing")
    partial = ~np.isclose(intervals, interval_seconds, rtol=0, atol=1e-9)
    epochs = _epochs([prefix.condition_at.origin_epoch_ns + int(round(float(t) * NANOSECONDS))
                      for t in sampled.time])
    values = solar_conditions(prefix.condition_at.frame.to_lonlat(sampled.state), epochs,
                              prefix.condition_at.names)
    sampled = Segment(sampled.segment_id, sampled.source_domain, sampled.region,
        frozen_array(sampled.time), frozen_array(sampled.state),
        {name: frozen_array(values[:, i]) for i, name in enumerate(prefix.condition_at.names)}, False)
    return sampled, {"nominal_interval_seconds": float(interval_seconds),
        "transition_count": len(intervals), "short_tail_count": int(np.count_nonzero(partial)),
        "minimum_interval_seconds": float(intervals.min()), "maximum_interval_seconds": float(intervals.max()),
        "interpolated_training_only": True, "solar_recomputed_not_interpolated": True,
        "noise_embedding_qualified": False}


def positions_in_scoring_frame(prefix, positions_m, scoring_frame):
    """Convert method predictions to the common origin-anchored metric frame.

    All methods and terrain configurations must be scored in the same frame;
    first-visible and forecast-origin frames have different eastward scales.
    The conversion reads neither target positions nor future observations.
    """
    if not isinstance(prefix, MethodPrefix) or not isinstance(scoring_frame, LocalFrame):
        raise ValueError("bound prefix and explicit common scoring frame required")
    return frozen_array(scoring_frame.from_lonlat(prefix.condition_at.frame.to_lonlat(positions_m)))
