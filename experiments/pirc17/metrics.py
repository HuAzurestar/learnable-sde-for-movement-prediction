"""Position-distribution scores; never mix accuracy, entropy and runtime."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

METRICS_VERSION = "pirc17-position-metrics-v1"
COVERAGE_LEVELS = (.5, .8, .9, .95)


def _samples(samples, target):
    x, y = np.asarray(samples, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1:] != y.shape or not x.shape[1]:
        raise ValueError("at least two vector samples and matching target required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite forecasts/targets must be recorded as failures, not dropped")
    return x, y


def energy_score(samples, target, *, chunk_size=128):
    """Unbiased U-statistic Monte Carlo ES, same distance unit as positions."""
    x, y = _samples(samples, target)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("positive integer chunk size required")
    distances = 0.
    for start in range(0, len(x), chunk_size):
        distances += np.linalg.norm(x[start:start+chunk_size, None] - x[None], axis=-1).sum()
    return float(np.linalg.norm(x-y, axis=-1).mean() - distances/(2*len(x)*(len(x)-1)))


def marginal_crps(samples, target):
    """Per-coordinate unbiased ensemble CRPS; uses sorted pair differences."""
    x, y = _samples(samples, target)
    n = len(x)
    coefficients = (2*np.arange(n)-n+1)[:, None]
    unordered_pair_sum = (coefficients*np.sort(x, axis=0)).sum(axis=0)
    return np.abs(x-y).mean(axis=0) - unordered_pair_sum/(n*(n-1))


def radial_regions(samples, target, *, levels=COVERAGE_LEVELS):
    """Empirical quantile disks about ensemble mean, NOT highest-density regions."""
    x, y = _samples(samples, target)
    if x.shape[1] != 2:
        raise ValueError("radial region area requires 2D positions")
    center = x.mean(axis=0)
    distances = np.sort(np.linalg.norm(x-center, axis=1))
    target_distance = float(np.linalg.norm(y-center))
    results = []
    for level in levels:
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("coverage levels must lie strictly between zero and one")
        radius = float(distances[int(np.ceil(level*len(x)))-1])
        results.append({"level":float(level),"radius_m":radius,"area_m2":float(np.pi*radius**2),
            "covered":bool(target_distance <= radius),
            "empirical_mass":float(np.mean(distances <= radius))})
    return {"region":"ensemble_mean_radial_quantile_disk", "center_m":center.tolist(), "levels":results}


@dataclass(frozen=True)
class EntropyGrid:
    x_edges_m: tuple[float, ...]
    y_edges_m: tuple[float, ...]

    def __post_init__(self):
        for name in ("x_edges_m", "y_edges_m"):
            edges = np.asarray(getattr(self, name), dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
                raise ValueError("finite strictly increasing grid edges required")
            object.__setattr__(self, name, tuple(float(v) for v in edges))

    @property
    def identity(self):
        payload = {"x_edges_m":self.x_edges_m,"y_edges_m":self.y_edges_m,
                   "tail_policy":"one_explicit_overflow_bin","units":"nats_discrete"}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

    def entropy(self, samples):
        x = np.asarray(samples, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("finite nonempty 2D positions required")
        counts, _, _ = np.histogram2d(x[:,0],x[:,1],bins=(self.x_edges_m,self.y_edges_m))
        outside = int(len(x)-counts.sum())
        masses = np.append(counts.ravel(),outside)/len(x)
        positive = masses[masses>0]
        return {"entropy_nats":float(-np.sum(positive*np.log(positive))),
            "overflow_mass":float(outside/len(x)),"total_mass":float(masses.sum()),
            "grid_identity":self.identity,"particle_count":len(x)}


def score_path(particles, target, elapsed_seconds, *, time_weights, entropy_grid: EntropyGrid):
    """One origin/seed; aggregate origins and independent blocks separately.

    Position arrays are metric offsets in the same origin-anchored frame.
    Targets must be actual observed positions on the registered scoring grid;
    this function performs no interpolation or nearest-label selection.
    """
    x, y = np.asarray(particles,dtype=float),np.asarray(target,dtype=float)
    times, weights = np.asarray(elapsed_seconds,dtype=float),np.asarray(time_weights,dtype=float)
    if x.ndim != 3 or x.shape[2] != 2 or y.shape != x.shape[1:] or len(x)<2:
        raise ValueError("particles must be (N,T,2), targets (T,2), N>=2")
    t = x.shape[1]
    if times.shape != (t,) or weights.shape != (t,) or t<1:
        raise ValueError("time grid/weight dimensions differ")
    if not all(np.isfinite(a).all() for a in (x,y,times,weights)):
        raise ValueError("nonfinite input must remain a scored-run failure")
    if times[0]<=0 or np.any(np.diff(times)<=0) or np.any(weights<0) or not np.isclose(weights.sum(),1.,rtol=0,atol=1e-12):
        raise ValueError("positive increasing times and normalized nonnegative weights required")
    by_time = []
    for i, seconds in enumerate(times):
        by_time.append({"elapsed_seconds":float(seconds),"energy_score_m":energy_score(x[:,i],y[i]),
            "marginal_crps_m":marginal_crps(x[:,i],y[i]).tolist(),
            "region":radial_regions(x[:,i],y[i]),"position_entropy":entropy_grid.entropy(x[:,i])})
    errors = np.linalg.norm(x.mean(axis=0)-y,axis=1)
    endpoint_samples = np.linalg.norm(x[:,-1]-y[-1],axis=1)
    entropies = np.array([r["position_entropy"]["entropy_nats"] for r in by_time])
    weighted_particles = (x*np.sqrt(weights)[None,:,None]).reshape(len(x),-1)
    weighted_target = (y*np.sqrt(weights)[:,None]).ravel()
    return {"schema_version":METRICS_VERSION,"particle_count":len(x),"by_time":by_time,
        "time_weights":weights.tolist(),"time_weighted_energy_score_m":float(weights@np.array([r["energy_score_m"] for r in by_time])),
        "point_estimator":"ensemble_mean_not_best_of_N","ade_grid_mean_m":float(errors.mean()),
        "time_weighted_displacement_error_m":float(weights@errors),"fde_m":float(errors[-1]),
        "particle_endpoint_error_quantiles_m":{str(q):float(np.quantile(endpoint_samples,q)) for q in (.5,.9,.95)},
        "path_energy_score_m":energy_score(weighted_particles,weighted_target),
        "path_score_role":"supplementary_weighted_joint_path_energy_not_sum_of_marginals",
        "entropy_change_from_first_scoring_time_nats":(entropies-entropies[0]).tolist(),
        "nll":{"status":"unavailable","reason":"no qualified predictive density estimator; particles do not justify a Gaussian shortcut"},
        "mode_entropy":{"status":"unavailable","reason":"not a mode-labelled predictive model"},
        "joint_path_entropy":{"status":"unavailable","reason":"marginal grid entropies do not identify joint path entropy"}}
