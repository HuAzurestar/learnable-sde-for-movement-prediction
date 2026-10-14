"""Information-restricted formal origin cases and an outer-train-only prior.

These pure consumers do not open data or authorize execution. The registered
input operation supplies verified MethodDevelopment/FinalPrefix objects. A
point/velocity-only method frame is anchored at its sole visible point (t0),
never at a hidden earlier observation. Truth is not accepted by this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .features import LocalFrame
from .formal_inputs import FinalPrefix
from .method_development import MethodDevelopment
from .method_inputs import MethodPrefix, SolarConditionField, _elapsed
from .origins import Origin, VelocityPrior, causal_prefix, frozen_array, known_velocity
from .protocol_core import digest, envelope, sha256, unpack

PRIOR_VERSION = "pirc17-formal-outer-train-velocity-prior-v1"
CASE_VERSION = "pirc17-formal-origin-case-v1"


@dataclass(frozen=True)
class RegisteredVelocityPrior:
    prior: VelocityPrior
    evidence: dict


def build_velocity_prior(prepared, *, training_contract, expected_input_sha256):
    """Use every outer-train prefix, including its frozen method-adapt subset.

    Units are physical east/north m/s in each prefix's origin-local frame, not
    coordinates from a hidden evaluation history or training target velocity.
    Validation is checked as a separate population and never enters the fit.
    """
    if not isinstance(prepared, MethodDevelopment):
        raise ValueError("source-bound method development required")
    identity = dict(prepared.identity)
    identity_sha = identity.pop("sha256", None)
    if (identity_sha != sha256(expected_input_sha256) or digest(identity) != identity_sha
            or identity["population_identity"] != training_contract["population_identity"]):
        raise ValueError("fixed training input/population identity differs")
    counts = Counter()
    samples, train_blocks, validation_blocks = set(), set(), set()
    rows, origins = [], []
    for prefix in sorted(prepared.prefixes, key=lambda p: p.assignment.sample.sample_id):
        if not isinstance(prefix, MethodPrefix):
            raise ValueError("verified development prefixes required")
        sample, role = prefix.assignment.sample, prefix.assignment.method_role
        if sample.sample_id in samples or prefix.assignment.population_identity != identity["population_identity"]:
            raise ValueError("duplicate or replaced development population member")
        samples.add(sample.sample_id)
        counts[role] += 1
        if sample.split == "validation" and role == "validation":
            validation_blocks.add(sample.independent_block_id)
            continue
        if sample.split != "train" or role not in {"train", "adapt"}:
            raise ValueError("prior forbids final-eval or relabelled development roles")
        train_blocks.add(sample.independent_block_id)
        if (len(prefix.visible_epoch_ns) != sample.history_end-sample.history_start+1
                or len(prefix.visible_epoch_ns) < 2 or prefix.visible_positions_m.shape != (len(prefix.visible_epoch_ns), 2)
                or prefix.condition_at.origin_epoch_ns != int(prefix.visible_epoch_ns[-1])):
            raise ValueError("prior prefix differs from original visible bounds")
        times = _elapsed(prefix.visible_epoch_ns[-3:], prefix.condition_at.origin_epoch_ns)
        original = causal_prefix(prefix.visible_positions_m[-3:], times)
        if (not np.array_equal(original.history_positions_m, prefix.origin.history_positions_m)
                or not np.array_equal(original.history_times_seconds, prefix.origin.history_times_seconds)
                or not np.array_equal(original.velocity_mps, prefix.origin.velocity_mps)
                or prefix.origin.mode != "causal_prefix"):
            raise ValueError("training origin is not its actual causal prefix secant")
        lonlat = prefix.condition_at.frame.to_lonlat(original.history_positions_m)
        frame = LocalFrame(*lonlat[-1])
        origin = causal_prefix(frame.from_lonlat(lonlat), times)
        origins.append(origin)
        rows.append({"sample_id": sample.sample_id, "segment_id": sample.segment_id,
            "independent_block_id": sample.independent_block_id, "outer_split": "train", "method_role": role,
            "prefix_sha256": digest(prefix.identity()), "velocity_east_north_mps": origin.velocity_mps.tolist()})
    if (dict(counts) != training_contract["method_roles"] or dict(counts) != identity["sample_counts"]
            or sorted(samples) != identity["sample_ids"] or train_blocks & validation_blocks
            or len(origins) != training_contract["outer_train_windows"]
            or counts["validation"] != training_contract["validation_windows"]):
        raise ValueError("complete disjoint registered train/adapt/validation population required")
    record = {"schema_version": PRIOR_VERSION, "input_sha256": identity_sha,
        "population_identity": identity["population_identity"], "coordinate_units": "origin-local-east-north-metres-per-second",
        "training_rows": rows, "outer_train_windows": len(rows), "validation_used": False,
        "final_eval_used": False, "target_coordinates_used": False}
    prior = VelocityPrior.fit(origins, split="train", training_identity=digest(record))
    return RegisteredVelocityPrior(prior, envelope({**record, "prior_identity": prior.identity}))


def validate_prior(value):
    if not isinstance(value, RegisteredVelocityPrior):
        raise ValueError("registered outer-train velocity prior required")
    p = dict(unpack(value.evidence))
    prior_identity = p.pop("prior_identity")
    if (p["schema_version"] != PRIOR_VERSION or p["validation_used"] is not False
            or p["final_eval_used"] is not False or p["target_coordinates_used"] is not False
            or value.prior.training_identity != digest(p) or prior_identity != value.prior.identity
            or p["outer_train_windows"] != len(value.prior.velocities_mps)
            or not np.array_equal(value.prior.velocities_mps, np.array([r["velocity_east_north_mps"] for r in p["training_rows"]]))):
        raise ValueError("prior evidence/parameters changed")
    return p


@dataclass(frozen=True)
class OriginCase:
    sample_id: str
    independent_block_id: str
    mode: str
    population_sha256: str
    window_sha256: str
    terrain_origin: Origin
    method_origin: Origin
    scoring_frame: LocalFrame
    condition_at: SolarConditionField
    score_seconds: np.ndarray
    prior_identity: str | None

    def to_scoring_frame(self, method_positions_m):
        return frozen_array(self.scoring_frame.from_lonlat(self.condition_at.frame.to_lonlat(method_positions_m)))

    def identity(self):
        def state(origin):
            return {"mode": origin.mode, "position_m": origin.position_m.tolist(),
                "velocity_mps": origin.velocity_mps.tolist(), "history_positions_m": origin.history_positions_m.tolist(),
                "history_times_seconds": origin.history_times_seconds.tolist(), "velocity_source": origin.velocity_source,
                "velocity_observed_at_seconds": origin.velocity_observed_at_seconds, "velocity_error_mps": origin.velocity_error_mps}
        return {"schema_version": CASE_VERSION, "sample_id": self.sample_id,
            "independent_block_id": self.independent_block_id, "origin_mode": self.mode,
            "population_sha256": self.population_sha256, "window_sha256": self.window_sha256,
            "terrain_origin": state(self.terrain_origin), "method_origin": state(self.method_origin),
            "solar": self.condition_at.identity(), "score_seconds": self.score_seconds.tolist(),
            "prior_identity": self.prior_identity, "role": "primary" if self.mode == "causal_prefix" else "secondary-descriptive"}


def build_origin_cases(prefixes, population, *, prior):
    """Require exact sealed order/secondary subset; no rank replacement or truth."""
    p = unpack(population)
    selection = p["selection"]
    prefixes = tuple(prefixes)
    if any(not isinstance(x, FinalPrefix) or x.split != "final_eval" for x in prefixes):
        raise ValueError("actual final causal prefixes required")
    identities = [{"sample_id": x.sample_id, "independent_block_id": x.independent_block_id, "split": x.split} for x in prefixes]
    if (identities != selection["selected"] or selection["secondary_selected"] != selection["selected"][:6]
            or len(identities) > 46 or len({x["independent_block_id"] for x in identities}) != len(identities)
            or any(x.population_sha256 != population["sha256"] for x in prefixes)):
        raise ValueError("origin cases must follow exact frozen primary/secondary population")
    validate_prior(prior)
    output = {"causal_prefix": [], "known_velocity": [], "point_only": []}
    for rank, prefix in enumerate(prefixes):
        def case(mode, terrain, method, solar, prior_identity=None):
            return OriginCase(prefix.sample_id, prefix.independent_block_id, mode, population["sha256"],
                prefix.window_sha256, terrain, method, prefix.scoring_frame, solar,
                frozen_array(prefix.score_seconds), prior_identity)
        output["causal_prefix"].append(case("causal_prefix", prefix.terrain_origin,
                                            prefix.method_origin, prefix.condition_at))
        if rank >= len(selection["secondary_selected"]):
            continue
        # One visible point means the first-visible frame is now t0. Carrying
        # the primary's earlier frame would leak hidden history to point_only.
        solar = SolarConditionField(prefix.scoring_frame, prefix.origin_epoch_ns)
        velocity = known_velocity([0., 0.], 0., prefix.terrain_origin.velocity_mps,
                                  source=prefix.terrain_origin.velocity_source, observed_at_seconds=0., error_mps=None)
        point = prior.prior.at([0., 0.], 0.)
        output["known_velocity"].append(case("known_velocity", velocity, velocity, solar))
        output["point_only"].append(case("point_only", point, point, solar, prior.prior.identity))
    return {k: tuple(v) for k, v in output.items()}
