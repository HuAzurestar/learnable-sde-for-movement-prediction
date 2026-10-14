"""Versioned four-state oracle recipes, with no trajectory-data dependencies."""

from dataclasses import dataclass

from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash
from inference.affine_oracle import exact_moments


@dataclass(frozen=True)
class OracleCase:
    case_id: str
    package: FrozenDynamicsPackage
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]
    horizons: tuple[float, ...]
    tolerance: float = 1e-10

    def moments(self, horizon):
        if horizon not in self.horizons:
            raise ValueError("horizon is outside the registered oracle grid")
        return exact_moments(self.package, self.initial_mean, self.initial_covariance, horizon)


def affine_package(case_id, A, b, L):
    parameters = {"A": A, "b": b, "L": L}
    # Bind the complete numerical implementation, not only the recipe name.
    from application.research_registry import implementation_hash
    from inference.affine_oracle import exact_transition
    document = {"schema_version": "frozen-affine-dynamics-v1", "package_id": case_id,
        "family": "affine-oracle", "frozen": True, "state_names": ["x", "y", "vx", "vy"],
        "units": ["m", "m", "m/s", "m/s"], "coordinate_system": "local-cartesian", "time_unit": "s",
        "noise_convention": "ito", "parameters": parameters, "parameter_hash": content_hash(parameters),
        "code_hash": implementation_hash(exact_transition), "drift_implementation": "affine-constant-v1",
        "diffusion_implementation": "additive-velocity-v1", "noise_support": ["vx", "vy"],
        "context_hash": content_hash({"terrain": "none", "frozen": True}),
        "profile_hash": content_hash({"coordinate_system": "local-cartesian", "time_unit": "s"}),
        "training_source": {"kind": "synthetic", "training_data_hash": None,
            "reason": "fixed generator; no fitted trajectory data"},
        "protocol_hash": content_hash({"generator": "affine-oracle-suite-v1", "case_id": case_id}),
        "capabilities": ["affine-exact", "constant-diffusion", "velocity-noise"],
        "qualification": {"scope": "oracle-fixture", "scientific_qualification": False},
        "license_scope": {"input": "synthetic-only", "export": "public-generator"}}
    return FrozenDynamicsPackage.from_manifest(document, expected_hash=content_hash(document))


def oracle_suite():
    """Stable, near-critical, correlated-noise and uncertain-initial-state cases."""
    cases = []
    for name, damping, stiffness, noise, variances in (
        ("stable-v1", 0.8, 0.4, ((0.4, 0.0), (0.0, 0.4)), (0, 0, 0, 0)),
        ("near-critical-v1", 0.0001, 0.0, ((0.2, 0.0), (0.0, 0.2)), (0, 0, 0, 0)),
        ("anisotropic-v1", 0.6, 0.15, ((0.5, 0.1), (0.2, 0.08)), (0, 0, 0, 0)),
        ("uncertain-initial-v1", 0.4, 0.2, ((0.3, 0.0), (0.0, 0.15)), (1, 0.25, 0.09, 0.04)),
    ):
        A = [[0, 0, 1, 0], [0, 0, 0, 1], [-stiffness, 0, -damping, 0], [0, -stiffness, 0, -damping]]
        package = affine_package(name, A, [0, 0, 0.05, -0.02], [[0, 0], [0, 0], list(noise[0]), list(noise[1])])
        covariance = tuple(tuple(float(variances[i]) if i == j else 0.0 for j in range(4)) for i in range(4))
        cases.append(OracleCase(name, package, (0.0, 0.0, 1.0, -0.5), covariance, (0.0, 1.0, 10.0, 100.0, 1000.0)))
    return tuple(cases)


def matrix_cardinality(axis_sizes, *, maximum=10000):
    """Check the design before materializing any Cartesian-product cells."""
    from domain.errors import DataValidationError
    if type(maximum) is not int or not 0 < maximum <= 10000:
        raise DataValidationError("matrix maximum must be between 1 and 10000")
    if type(axis_sizes) not in (list, tuple) or not 0 < len(axis_sizes) <= 16:
        raise DataValidationError("matrix needs a bounded explicit axis list")
    count = 1
    for size in axis_sizes:
        if type(size) is not int or size <= 0:
            raise DataValidationError("matrix axes must be positive integer counts")
        count *= size
        if count > maximum:
            raise DataValidationError("matrix exceeds registered cell limit")
    return count
