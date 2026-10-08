"""Outward bounds for declared affine finite-grid laws, not path roundoff.

The grid is frozen explicitly, including MLMC's chosen finest level. No pilot,
solver selection, qualification flag promotion, ledger or budget operation.
"""

from fractions import Fraction

from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import FrozenDynamicsPackage, _encode
from domain.propagation import PropagationRequest
from inference.affine_reference import (AffineReferenceCertificate, Arithmetic,
    MAX_BYTES, _add, _functional, _interval_manifest, _matmul, _strict_covariance,
    _transpose, _zeros, algorithm_manifest, bound_affine_reference)


SOLVERS = ("euler", "heun", "reversible-heun")
MAX_STEPS = 8192


def _grid(solver, steps):
    if type(solver) is not str or solver not in SOLVERS:
        raise DataValidationError("discrete reference requires an explicit supported solver")
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise DataValidationError("discrete reference requires explicit grid steps within 1..8192")
    return {"solver": solver, "steps": steps, "time_step": "exact-horizon-rational/steps",
            "noise": "independent-centered-Gaussian-increments",
            "roundoff_scope": "mathematical-recurrence-only"}


def _identity(a, size):
    return [[a.box(int(i == j)) for j in range(size)] for i in range(size)]


def _scale(a, matrix, scalar):
    return [[a.mul(value, scalar) for value in row] for row in matrix]


def _compose(a, after, before):
    F, c, Q = after
    G, d, R = before
    return (_matmul(a, F, G), _add(a, c, _matmul(a, F, d)),
            _add(a, Q, _matmul(a, _matmul(a, F, R), _transpose(F))))


def _transition(a, parameters, horizon, solver, steps):
    h = a.box(Fraction(horizon)/steps)
    A = [[a.box(value) for value in row] for row in parameters["A"]]
    b = [[a.box(value)] for value in parameters["b"]]
    L = [[a.box(value) for value in row] for row in parameters["L"]]
    identity = _identity(a, 4)
    hA, hb = _scale(a, A, h), _scale(a, b, h)
    F, c, B = _add(a, identity, hA), hb, L
    if solver != "euler":
        half_square = a.mul(a.box(Fraction(1, 2)), a.mul(h, h))
        A2 = _scale(a, _matmul(a, A, A), half_square)
        c = _add(a, hb, _scale(a, _matmul(a, A, b), half_square))
        B = _matmul(a, _add(a, identity, _scale(a, hA, a.box(Fraction(1, 2)))), L)
        if solver == "heun":
            F = _add(a, F, A2)
        else:
            # State is [physical y, auxiliary z], with z0=y0 exactly.
            F = [left+right for left, right in zip(F, A2)]
            F += [left+right for left, right in zip(_scale(a, identity, a.box(2)),
                [[a.sub(u, v) for u, v in zip(row, unit)] for row, unit in zip(hA, identity)])]
            c += hb
            B += L
    Q = _scale(a, _matmul(a, B, _transpose(B)), h)
    return F, c, Q


def _certificate(document):
    encoded = _encode(document)
    if len(encoded) > MAX_BYTES:
        raise NumericalError("discrete reference certificate exceeds fixed byte quota")
    certificate = AffineReferenceCertificate(encoded)
    certificate.manifest()  # Enforce the same bounded canonical parser on output.
    return certificate


def bound_affine_discrete(package, request, *, solver, steps):
    """Enclose an explicitly registered grid law using <=26 compositions.

    Input floats mean exact binary rational values, dt is exact horizon/steps.
    This does not bound the implemented sampler's floating point roundoff.
    """
    grid = _grid(solver, steps)
    if type(package) is not FrozenDynamicsPackage or type(request) is not PropagationRequest:
        raise DataValidationError("discrete reference requires frozen affine dynamics and endpoint request")
    package.validate()
    request.validate()
    if package.package_hash != request.model_package_hash:
        raise DataValidationError("discrete reference request/model content differs")
    covariance = _strict_covariance(request.initial_covariance)
    a = Arithmetic()
    base = _transition(a, package.manifest()["parameters"], request.horizons[0], solver, steps)
    size = len(base[0])
    result = (_identity(a, size), _zeros(a, size, 1), _zeros(a, size, size))
    remaining, compositions = steps, 0
    while remaining:
        if remaining & 1:
            result = _compose(a, base, result)
            compositions += 1
        remaining >>= 1
        if remaining:
            base = _compose(a, base, base)
            compositions += 1
    F, c, Q = result
    initial_mean = [[a.box(value)] for value in request.initial_mean]
    initial_covariance = [[a.box(value) for value in row] for row in covariance]
    if size == 8:
        initial_mean += initial_mean
        initial_covariance = [row+row for row in initial_covariance]*2
    mean = _add(a, _matmul(a, F, initial_mean), c)
    covariance = _add(a, _matmul(a, _matmul(a, F, initial_covariance), _transpose(F)), Q)
    # Only physical y is a model output. Auxiliary z never changes dimension.
    mean, covariance = mean[:4], [row[:4] for row in covariance[:4]]
    status, functional = _functional(a, request, mean, covariance)
    from experiments.pirc25.affine import code_hash
    return _certificate({"schema_version": "affine-discrete-certificate-v1",
        "algorithm": algorithm_manifest(), "code_hash": code_hash(),
        "model_package_hash": package.package_hash, "request_hash": request.request_hash,
        "grid": grid, "scope": "declared-affine-finite-grid-Gaussian-law",
        "scientific_qualification": False, "status": status,
        "physical_dimension": 4, "auxiliary_dimension": size-4,
        "compositions": compositions, "operations": a.operations,
        "units": "m" if request.functional == "endpoint-x" else "1",
        "functional_bounds": _interval_manifest(functional),
        "mean_bounds": [_interval_manifest(row[0]) for row in mean],
        "covariance_bounds": [[_interval_manifest(value) for value in row] for row in covariance],
        "transition_bounds": {"F": [[_interval_manifest(value) for value in row] for row in F],
            "offset": [_interval_manifest(row[0]) for row in c],
            "covariance": [[_interval_manifest(value) for value in row] for row in Q]}})


def verify_affine_discrete(certificate, package, request, *, solver, steps):
    if type(certificate) is not AffineReferenceCertificate:
        raise DataValidationError("explicit discrete reference certificate required")
    certificate.manifest()
    expected = bound_affine_discrete(package, request, solver=solver, steps=steps)
    if certificate._document != expected._document:
        raise DataValidationError("discrete certificate differs from recomputed source/request/grid")
    return expected


def bound_affine_time_bias(package, request, *, solver, steps):
    """Signed E[finite grid functional] minus E[continuous functional].

    Each component independently consumes the fixed arithmetic quota. Their
    total cost is exposed; this is not the error of a random sample estimator.
    """
    discrete = bound_affine_discrete(package, request, solver=solver, steps=steps)
    continuous = bound_affine_reference(package, request)
    dm, cm = discrete.manifest(), continuous.manifest()
    if dm["code_hash"] != cm["code_hash"]:
        raise DataValidationError("time-bias components were computed under different source hashes")
    a = Arithmetic()
    bias = a.sub(discrete.functional_bounds(), continuous.functional_bounds())
    return _certificate({"schema_version": "affine-time-bias-certificate-v1",
        "algorithm": algorithm_manifest(), "code_hash": dm["code_hash"],
        "model_package_hash": package.package_hash, "request_hash": request.request_hash,
        "grid": dm["grid"], "scope": "finite-grid-minus-continuous-expectation",
        "scientific_qualification": False,
        "status": "BOUNDED" if dm["status"] == cm["status"] == "BOUNDED" else "UNRESOLVED_COMPONENT",
        "units": dm["units"], "functional_bounds": _interval_manifest(bias),
        "continuous_certificate_hash": continuous.certificate_hash,
        "discrete_certificate_hash": discrete.certificate_hash,
        "component_operations": {"continuous": cm["operations"], "discrete": dm["operations"],
                                 "subtraction": a.operations},
        "operations": cm["operations"]+dm["operations"]+a.operations,
        "operation_quota_scope": "per-component-not-shared"})


def verify_affine_time_bias(certificate, package, request, *, solver, steps):
    if type(certificate) is not AffineReferenceCertificate:
        raise DataValidationError("explicit time-bias certificate required")
    certificate.manifest()
    expected = bound_affine_time_bias(package, request, solver=solver, steps=steps)
    if certificate._document != expected._document:
        raise DataValidationError("time-bias certificate differs from recomputed source/request/grid")
    return expected
