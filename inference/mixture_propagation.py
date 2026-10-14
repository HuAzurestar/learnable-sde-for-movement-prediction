"""Bounded deterministic cubature branches and moment-preserving reduction.

Absolute weights keep discarded mass visible across steps. The reported
functional normalizes only the retained approximation, not the true law.
Lineage hashes commit to a bounded live state, never an exponential tree.
"""

from dataclasses import asdict, dataclass
import json
import math
import sys

import numpy as np

from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import content_hash, _hash, _encode
from domain.mixture import MixturePolicy, finite
from domain.propagation import ErrorComponent, FunctionalResult, NumericalErrorBudget
from .nonlinear_propagation import nonlinear_drift


STATE_BYTES = 65536
COUNTERS = ("generated", "pruned", "threshold_merges", "forced_merges")


def strict_root(covariance):
    value = np.array(covariance, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all() or not np.array_equal(value, value.T):
        raise NumericalError("NUMERICAL_FAILURE: finite symmetric mixture covariance required")
    eigenvalues, vectors = np.linalg.eigh(value)
    if np.any(eigenvalues < 0):
        raise NumericalError("NUMERICAL_FAILURE: mixture PSD root would require projection")
    # Unique symmetric square root avoids a cubature recipe dependent on signs
    # or rotations of eigenvectors in repeated eigenspaces. No PSD clipping.
    root = (vectors*np.sqrt(eigenvalues)[None, :])@vectors.T
    if not np.isfinite(root).all():
        raise NumericalError("NUMERICAL_FAILURE: nonfinite mixture root")
    return root


@dataclass(frozen=True)
class Component:
    weight: float
    mean: tuple
    covariance: tuple
    lineage_id: str

    def validate(self):
        if (not finite(self.weight) or self.weight <= 0 or not _hash(self.lineage_id)
                or type(self.mean) is not tuple or len(self.mean) != 4 or any(not finite(v) for v in self.mean)
                or type(self.covariance) is not tuple or len(self.covariance) != 4
                or any(type(row) is not tuple or len(row) != 4 or any(not finite(v) for v in row) for row in self.covariance)):
            raise NumericalError("NUMERICAL_FAILURE: invalid finite positive mixture component")
        strict_root(self.covariance)

    @classmethod
    def from_manifest(cls, value):
        if (type(value) is not dict or set(value) != set(cls.__dataclass_fields__)
                or type(value["mean"]) is not list or len(value["mean"]) != 4
                or type(value["covariance"]) is not list or len(value["covariance"]) != 4
                or any(type(row) is not list or len(row) != 4 for row in value["covariance"])):
            raise DataValidationError("bounded canonical mixture component required")
        component = cls(value["weight"], tuple(value["mean"]), tuple(tuple(row) for row in value["covariance"]), value["lineage_id"])
        component.validate()
        return component


def component(weight, mean, covariance, lineage):
    value = Component(float(weight), tuple(float(v) for v in mean),
        tuple(tuple(float(v) for v in row) for row in covariance), lineage)
    value.validate()
    return value


def _merge_accumulator(a, b, *, policy_hash, step):
    """Preserve absolute weight and first/second moments, not mixture density."""
    total = math.fsum((a.weight, b.weight))
    fraction = b.weight/total
    ma, mb = np.array(a.mean), np.array(b.mean)
    delta = mb-ma
    mean = ma+fraction*delta
    covariance = (1-fraction)*np.array(a.covariance)+fraction*np.array(b.covariance)
    covariance += fraction*(1-fraction)*np.outer(delta, delta)
    if not finite(total) or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise NumericalError("NUMERICAL_FAILURE: nonfinite mixture moment accumulator")
    # The accumulator is not sampled or saved mid-reduction. Check each final
    # live covariance before it becomes an accepted next-step component.
    return Component(total, tuple(float(v) for v in mean),
        tuple(tuple(float(v) for v in row) for row in covariance),
        content_hash({"operation": "moment-merge-v1", "step": step,
            "policy_hash": policy_hash, "parents": [a.lineage_id, b.lineage_id]}))


def merge(a, b, *, policy_hash, step):
    a.validate()
    b.validate()
    result = _merge_accumulator(a, b, policy_hash=policy_hash, step=step)
    result.validate()
    return result


def mixture_step(components, parameters, dt, policy, step):
    """Generate at most 8*cap candidates, reduce with at most cap clusters.

    Heavy-first stable generation-order ties; scaled Euclidean nearest live
    cluster; threshold merging, then forced cap merging. This is not Runnalls'
    KL-bound selection or a density-error certificate.
    """
    if (type(policy) is not MixturePolicy or type(policy.component_cap) is not int
            or not 1 <= policy.component_cap <= 32 or not finite(dt) or dt <= 0
            or type(step) is not int or step < 0
            or type(components) is not tuple or not 0 < len(components) <= policy.component_cap):
        raise DataValidationError("bounded immutable mixture state required")
    A, b, L, amplitude, scales = parameters
    noise = dt*(L@L.T)
    strict_root(noise)
    candidates, pruned = [], []
    count = 0
    for parent in components:
        parent.validate()
        points = np.array(parent.mean)+np.concatenate((2*strict_root(parent.covariance).T, -2*strict_root(parent.covariance).T))
        means = points+dt*nonlinear_drift(A, b, amplitude, scales, points)
        weight = parent.weight/8
        if not finite(weight) or weight <= 0 or not np.isfinite(means).all():
            raise NumericalError("NUMERICAL_FAILURE: mixture branch overflow/underflow")
        for ordinal, mean in enumerate(means):
            count += 1
            if weight < policy.prune_weight:
                pruned.append(weight)
            else:
                candidates.append(component(weight, mean, noise, content_hash({"operation": "cubature-branch-v1",
                    "parent": parent.lineage_id, "point": ordinal, "step": step, "policy_hash": policy.policy_hash})))
    if not candidates:
        raise NumericalError("NUMERICAL_FAILURE: all mixture mass pruned")
    # Stable sort deliberately does not use lineage hashes for numerical ties.
    candidates.sort(key=lambda c: -c.weight)
    reduced, threshold_merges, forced_merges = [], 0, 0
    for candidate in candidates:
        distances = [float(np.linalg.norm((np.array(candidate.mean)-c.mean)/policy.state_scales)) for c in reduced]
        if any(not finite(distance) for distance in distances):
            raise NumericalError("NUMERICAL_FAILURE: nonfinite scaled merge distance")
        nearest = min(range(len(distances)), key=distances.__getitem__) if distances else None
        near = nearest is not None and distances[nearest] <= policy.merge_distance
        if near or len(reduced) == policy.component_cap:
            reduced[nearest] = _merge_accumulator(reduced[nearest], candidate, policy_hash=policy.policy_hash, step=step)
            threshold_merges += int(near)
            forced_merges += int(not near)
        else:
            reduced.append(candidate)
    for live in reduced:
        live.validate()
    return tuple(reduced), {"generated": count, "pruned": len(pruned), "threshold_merges": threshold_merges,
        "forced_merges": forced_merges}, math.fsum(pruned)


def mixture_estimate(package, request, policy, *, resume_state=None, checkpoint=None):
    from .propagation_methods import _inputs, _expectation
    from experiments.pirc25.affine import code_hash
    if type(policy) is not MixturePolicy:
        raise DataValidationError("explicit frozen mixture policy required")
    policy.validate(package, request, code_hash())
    A, b, L, mean, _ = _inputs(package, request)
    p = package.manifest()["parameters"]
    parameters = A, b, L, p.get("amplitude", (0., 0.)), p.get("length_scale", (1., 1.))
    identity = {"schema_version": "bounded-mixture-state-v1", "request_hash": request.request_hash,
        "model_package_hash": package.package_hash, "policy_hash": policy.policy_hash}
    environment = {"scheme": "deterministic-cubature-no-sampled-rng-v1", "numpy_version": np.__version__,
        "float_byteorder": sys.byteorder}
    root_id = content_hash({**identity, "operation": "initial-Gaussian-v1"})
    components = (component(1., mean, request.initial_covariance, root_id),)
    completed, discarded = 0, 0.
    counters = dict.fromkeys(COUNTERS, 0)
    peak = 1
    history = root_id
    if resume_state is not None:
        state = resume_state
        required = {"step", "data_position", "method_state", "rng_state", "chunk_complete"}
        if (type(state) is not dict or set(state) != required or state["chunk_complete"] is not True
                or state["rng_state"] != environment or type(state["data_position"]) is not dict
                or set(state["data_position"]) != {"next_grid_step"}):
            raise DataValidationError("mixture checkpoint environment/boundary differs")
        completed = state["data_position"]["next_grid_step"]
        method = state["method_state"]
        if (type(completed) is not int or not 0 <= completed <= request.steps
                or type(state["step"]) is not int or state["step"] != completed*policy.work_per_step
                or type(method) is not dict or set(method) != set(identity)|{"components", "discarded_mass", "counters", "peak_components", "lineage_digest"}
                or any(method[k] != v for k, v in identity.items()) or type(method["components"]) is not list
                or not 0 < len(method["components"]) <= policy.component_cap
                or type(method["counters"]) is not dict or set(method["counters"]) != set(COUNTERS)
                or any(type(v) is not int or not 0 <= v <= completed*8*policy.component_cap for v in method["counters"].values())
                or type(method["peak_components"]) is not int
                or not len(method["components"]) <= method["peak_components"] <= policy.component_cap
                or sum(method["counters"][k] for k in ("pruned", "threshold_merges", "forced_merges")) > method["counters"]["generated"]
                or not _hash(method["lineage_digest"])):
            raise DataValidationError("mixture checkpoint identity/progress/counters differ")
        components = tuple(Component.from_manifest(c) for c in method["components"])
        discarded, counters, peak, history = method["discarded_mass"], dict(method["counters"]), method["peak_components"], method["lineage_digest"]
        if len(_encode(state)) > STATE_BYTES:
            raise DataValidationError("mixture state byte quota exceeded")
    def check_mass():
        retained = math.fsum(c.weight for c in components)
        if (not finite(discarded) or not 0 <= discarded <= policy.maximum_discarded_mass
                or not finite(retained) or retained <= 0
                or abs(retained+discarded-1) > 64*np.finfo(float).eps*(completed+1)):
            raise NumericalError("NUMERICAL_FAILURE: mixture mass accounting/pruning cap differs")
        return retained
    check_mass()
    for step in range(completed, request.steps):
        components, update, removed = mixture_step(components, parameters, request.horizons[0]/request.steps, policy, step)
        discarded = math.fsum((discarded, removed))
        counters = {k: counters[k]+update[k] for k in COUNTERS}
        completed, peak = step+1, max(peak, len(components))
        check_mass()
        history = content_hash({"previous": history, "grid_step": completed, "components": [asdict(c) for c in components],
            "removed_mass": removed, "reduction_counts": update})
        if checkpoint is not None:
            state = {"step": completed*policy.work_per_step, "data_position": {"next_grid_step": completed},
                "method_state": {**identity, "components": [asdict(c) for c in components], "discarded_mass": discarded,
                    "counters": counters, "peak_components": peak, "lineage_digest": history},
                "rng_state": environment, "chunk_complete": True}
            content = _encode(state)
            if len(content) > STATE_BYTES:
                raise DataValidationError("mixture state byte quota exceeded")
            checkpoint(json.loads(content), request.steps*policy.work_per_step)
    retained = check_mass()
    estimate = math.fsum(c.weight*_expectation(np.array(c.mean), np.array(c.covariance), request) for c in components)/retained
    if not finite(estimate):
        raise NumericalError("NUMERICAL_FAILURE: nonfinite retained-mixture functional")
    normalized_mean = np.array([math.fsum(c.weight*c.mean[i] for c in components)/retained for i in range(4)])
    covariance = sum((c.weight/retained)*(np.array(c.covariance)+np.outer(np.array(c.mean)-normalized_mean, np.array(c.mean)-normalized_mean)) for c in components)
    strict_root(covariance)
    units = "m" if request.functional == "endpoint-x" else "1"
    unknown = lambda why: ErrorComponent(None, units, why, "NOT_IDENTIFIABLE")
    errors = NumericalErrorBudget(unknown("no certified mixture reference"), unknown("Euler time-grid bias unqualified"),
        unknown("cubature branching, moment merge and retained-mass normalization errors unknown"),
        ErrorComponent(0., units, "deterministic approximation; no sampled Monte Carlo", "IDENTIFIED"),
        unknown("synthetic frozen law; no observed real dynamics"))
    return FunctionalResult(request.request_hash, "retained-cubature-mixture-euler-v1", "functional_estimate", estimate,
        0., None, "deterministic-mixture-no-sampling", 0, errors, "APPROXIMATION_ONLY",
        (("policy_hash", policy.policy_hash), ("component_cap", policy.component_cap), ("point_count", 8),
         ("components", tuple(asdict(c) for c in components)), ("retained_mass", retained), ("discarded_mass", discarded),
         ("mass_accounting", "absolute-weights; functional-normalizes-retained-approximation"),
         ("lineage_root", root_id), ("lineage_digest", history), ("lineage_scope", "bounded-live-hash-commitments-not-full-expanding-tree"),
         ("reduction_counts", counters), ("peak_components", peak), ("covariance_projection", False),
         ("mean", tuple(float(v) for v in normalized_mean)), ("covariance", tuple(tuple(float(v) for v in row) for row in covariance))))
