"""Compute a particle suffix with the unchanged causal rollout kernel.

This is a numerical primitive, NOT an empirical executor or an inheritance
certificate. A future caller must independently admit the complete, closed
prefix, model/data/map identities, resource ancestry and whole-result audit.
No prefix predictions are accepted, relabeled or recomputed here. Only the
returned suffix incurs rollout feature-query counts. Brownian setup still
materializes the original full-budget driver and retains its 512 MiB guard.

Pathwise equivalence requires particle-local callbacks. This helper does not
certify arbitrary callbacks, map backends, scientific precision or runtime.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .brownian import BrownianPath, integration_grid
from .origins import Origin
from .rollout import Forecast, _horizons, rollout

VERSION = "pirc17-particle-suffix-kernel-v1"


@dataclass(frozen=True)
class ParticleSuffix:
    prefix_particles: int
    total_particles: int
    brownian_identity: dict
    forecast: Forecast
    version: str = VERSION

    @property
    def new_particles(self):
        return self.total_particles - self.prefix_particles


class _SuffixOrigin:
    """Keep the original origin, including point-only prior draw indices."""

    def __init__(self, origin, prefix, total):
        self._origin, self._prefix, self._total = origin, prefix, total

    def __getattr__(self, name):
        return getattr(self._origin, name)

    def sample_velocities(self, count, rng):
        if count != self._total - self._prefix:
            raise ValueError("suffix initial-velocity count changed")
        # Calling sample_velocities(count, rng) directly would incorrectly
        # reuse the first draws for point-only origins. Preserve original
        # full-budget sampling, then discard only the already computed prefix.
        return self._origin.sample_velocities(self._total, rng)[self._prefix:]


class _SuffixBrownian:
    def __init__(self, driver, prefix):
        self._driver, self._prefix = driver, prefix

    def __call__(self, start, end, particles, noise_dimensions):
        if type(particles) is not int or particles != self._driver.particles - self._prefix:
            raise ValueError("suffix Brownian particle count changed")
        # Reuse the original driver's interval/dimension validation and exact
        # float64 subtraction, rather than inventing a new random stream.
        return self._driver(start, end, self._driver.particles, noise_dimensions)[self._prefix:]


def rollout_suffix(origin, horizons_seconds, *, prefix_particles, driver,
                   seed, stream_id, max_step_seconds, base_drift, diffusion,
                   terrain=None, conditioner=None, history_step_seconds=5.0):
    """Return only [prefix_particles, driver.particles), with actual counters.

    The complete driver is supplied by the caller under its original allocation
    guard. Its origin namespace and seed must be explicitly bound. Requested
    integration boundaries must all exist before any model or map callback runs.
    This function neither joins a prefix nor validates its provenance.
    """
    if type(origin) is not Origin or type(driver) is not BrownianPath:
        raise ValueError("original Origin and BrownianPath required")
    if type(prefix_particles) is not int or not 0 < prefix_particles < driver.particles:
        raise ValueError("strictly positive proper particle prefix required")
    if (type(seed) is not int or seed != driver.identity["seed"]
            or not isinstance(stream_id, str) or stream_id != driver.identity["stream_id"]):
        raise ValueError("suffix must bind the same Brownian seed and origin stream")
    horizons = _horizons(horizons_seconds)
    grid = integration_grid(horizons, max_step_seconds, history_step_seconds)
    if horizons[-1] != driver.times[-1] or any(t not in driver.index for t in grid):
        raise ValueError("suffix integration grid differs from the supplied full driver")
    identity = deepcopy(driver.identity)
    forecast = rollout(
        _SuffixOrigin(origin, prefix_particles, driver.particles), horizons,
        particles=driver.particles-prefix_particles, seed=seed,
        max_step_seconds=max_step_seconds, history_step_seconds=history_step_seconds,
        base_drift=base_drift, diffusion=diffusion, terrain=terrain, conditioner=conditioner,
        brownian_increments=_SuffixBrownian(driver, prefix_particles))
    return ParticleSuffix(prefix_particles, driver.particles, identity, forecast)
