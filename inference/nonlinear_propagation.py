"""Smooth additive-noise synthetic drift and bounded discrete Gaussian closure."""

import math

import numpy as np

from domain.errors import NumericalError
from domain.propagation import ErrorComponent, FunctionalResult, NumericalErrorBudget


def nonlinear_drift(A, b, amplitude, length_scale, state):
    """Kinematic affine drift plus a*tanh(position/s) in velocity coordinates.

    Fixed scalar tanh and einsum arithmetic is independent of batch size. The
    recipe is globally Lipschitz for finite positive length scales, but an
    explicit coarse time grid is not thereby guaranteed stable or accurate.
    """
    drift = np.einsum("bi,ji->bj", state, A, optimize=False) + b
    for axis in range(2):
        drift[:, 2+axis] += np.fromiter((float(amplitude[axis])*math.tanh(float(x)/float(length_scale[axis]))
            for x in state[:, axis]), dtype=float, count=len(state))
    return drift


def _strict_root(covariance):
    from .affine_oracle import _covariance
    covariance = _covariance(covariance, "cubature covariance").numpy()
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues < 0):
        raise NumericalError("cubature square root would require PSD projection")
    return vectors*np.sqrt(eigenvalues)[None, :]


def cubature_moments(package, request):
    """Eight equal-weight points; Euler pushforward then additive diffusion.

    This is a discrete-time assumed Gaussian closure, not a nonlinear exact
    transition or a continuous Gaussian moment ODE integration.
    """
    from .propagation_methods import _inputs
    A, b, L, mean, _ = _inputs(package, request)
    parameters = package.manifest()["parameters"]
    amplitude, scales = parameters.get("amplitude", (0., 0.)), parameters.get("length_scale", (1., 1.))
    covariance = np.array(request.initial_covariance, dtype=float)
    dt = request.horizons[0]/request.steps
    for _ in range(request.steps):
        root = _strict_root(covariance)
        points = mean + np.concatenate((2*root.T, -2*root.T))
        propagated = points + dt*nonlinear_drift(A, b, amplitude, scales, points)
        mean = propagated.mean(axis=0)
        residuals = propagated-mean
        covariance = residuals.T@residuals/8 + dt*(L@L.T)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise NumericalError("nonfinite cubature moments")
    # Check the final root too: never silently repair the terminal covariance.
    _strict_root(covariance)
    return mean, covariance


def nonlinear_error_budget(request, standard_error, *, closure=False):
    units = "m" if request.functional == "endpoint-x" else "1"
    return NumericalErrorBudget(
        ErrorComponent(None, units, "no certified nonlinear reference", "NOT_IDENTIFIABLE"),
        ErrorComponent(None, units, "requires independently qualified grid comparison", "NOT_IDENTIFIABLE"),
        ErrorComponent(None if closure else 0.0, units,
            "Gaussian cubature closure error unknown" if closure else "path estimator; no additional Gaussian closure",
            "NOT_IDENTIFIABLE" if closure else "IDENTIFIED"),
        ErrorComponent(standard_error, units, "estimator standard error",
            "ESTIMATED" if standard_error is not None else "NOT_IDENTIFIABLE"),
        ErrorComponent(None, units, "synthetic generator; no observed real dynamics", "NOT_IDENTIFIABLE"))


def cubature_estimate(package, request):
    from .propagation_methods import _expectation
    mean, covariance = cubature_moments(package, request)
    return FunctionalResult(request.request_hash, "gaussian-cubature-euler-v1", "functional_estimate",
        _expectation(mean, covariance, request), 0.0, None, "deterministic-closure-no-sampling", 0,
        nonlinear_error_budget(request, 0.0, closure=True), "APPROXIMATION_ONLY",
        (("assumption", "Gaussian closure at each discrete Euler step; multimodality not represented"),
         ("point_count", 8), ("point_weight", 0.125), ("covariance_projection", False),
         ("mean", tuple(float(x) for x in mean)), ("covariance", tuple(tuple(float(x) for x in row) for row in covariance))))
