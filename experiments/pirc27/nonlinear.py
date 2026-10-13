"""Versioned synthetic stress inputs; registration/runs require shared admission."""

from domain.frozen_dynamics import content_hash
from domain.nonlinear_dynamics import FrozenNonlinearPackage


def nonlinear_package(case_id="double-well-tanh-v1", *, amplitude=(3.0, 0.0), length_scale=(1.0, 1.0),
                      stiffness=0.5, damping=1.0, diffusion=0.15):
    from application.research_registry import implementation_hash
    from inference.nonlinear_propagation import nonlinear_drift
    parameters = {"A": [[0, 0, 1, 0], [0, 0, 0, 1], [-stiffness, 0, -damping, 0], [0, -stiffness, 0, -damping]],
        "b": [0, 0, 0, 0], "L": [[0, 0], [0, 0], [diffusion, 0], [0, diffusion]],
        "amplitude": list(amplitude), "length_scale": list(length_scale)}
    document = {"schema_version": "frozen-tanh-dynamics-v1", "package_id": case_id,
        "family": "synthetic-nonlinear", "frozen": True, "state_names": ["x", "y", "vx", "vy"],
        "units": ["m", "m", "m/s", "m/s"], "coordinate_system": "local-cartesian", "time_unit": "s",
        "noise_convention": "ito", "parameters": parameters, "parameter_hash": content_hash(parameters),
        "code_hash": implementation_hash(nonlinear_drift), "drift_implementation": "tanh-affine-velocity-v1",
        "diffusion_implementation": "additive-velocity-v1", "noise_support": ["vx", "vy"],
        "context_hash": content_hash({"terrain": "none", "frozen": True}),
        "profile_hash": content_hash({"coordinate_system": "local-cartesian", "time_unit": "s"}),
        "training_source": {"kind": "synthetic", "training_data_hash": None,
            "reason": "fixed generator; no fitted trajectory data"},
        "protocol_hash": content_hash({"generator": "tanh-stress-suite-v1", "case_id": case_id}),
        "capabilities": ["generic-drift", "constant-diffusion", "velocity-noise"],
        "qualification": {"scope": "stress-fixture", "scientific_qualification": False},
        "license_scope": {"input": "synthetic-only", "export": "public-generator"}}
    return FrozenNonlinearPackage.from_manifest(document, expected_hash=content_hash(document))
