"""A bounded immutable synthetic stress package, never a trained model grant."""

from dataclasses import dataclass
import hashlib
import json

from .errors import DataValidationError
from .frozen_dynamics import STATE_NAMES, STATE_UNITS, _array, _encode, _hash, content_hash


@dataclass(frozen=True)
class FrozenNonlinearPackage:
    _document: bytes

    @classmethod
    def from_manifest(cls, manifest, *, expected_hash):
        validate_nonlinear_package(manifest, expected_hash=expected_hash)
        return cls(_encode(manifest))

    @property
    def package_hash(self):
        return hashlib.sha256(self._document).hexdigest()

    def manifest(self):
        return json.loads(self._document)

    def validate(self):
        if type(self._document) is not bytes or len(self._document) > 65536:
            raise DataValidationError("nonlinear package requires bounded canonical bytes")
        validate_nonlinear_package(self.manifest(), expected_hash=self.package_hash)


def validate_nonlinear_package(manifest, *, expected_hash):
    required = {"schema_version", "package_id", "family", "frozen", "state_names", "units",
        "coordinate_system", "time_unit", "noise_convention", "parameters", "parameter_hash", "code_hash",
        "drift_implementation", "diffusion_implementation", "noise_support", "context_hash", "profile_hash",
        "training_source", "protocol_hash", "capabilities", "qualification", "license_scope"}
    def require(condition, detail):
        if not condition:
            raise DataValidationError(detail)
    require(type(manifest) is dict and set(manifest) == required, "invalid nonlinear package fields")
    require(manifest["schema_version"] == "frozen-tanh-dynamics-v1" and manifest["family"] == "synthetic-nonlinear"
            and manifest["frozen"] is True, "only the frozen versioned synthetic stress family is supported")
    require(type(manifest["package_id"]) is str and 0 < len(manifest["package_id"]) <= 128, "invalid package ID")
    require(manifest["state_names"] == list(STATE_NAMES) and manifest["units"] == list(STATE_UNITS)
            and manifest["coordinate_system"] == "local-cartesian" and manifest["time_unit"] == "s"
            and manifest["noise_convention"] == "ito", "incompatible state, units or convention")
    parameters = manifest["parameters"]
    require(type(parameters) is dict and set(parameters) == {"A", "b", "L", "amplitude", "length_scale"},
            "invalid tanh-affine parameters")
    require(_array(parameters["A"], (4, 4)) and _array(parameters["b"], (4,)) and _array(parameters["L"], (4, 2))
            and _array(parameters["amplitude"], (2,)) and _array(parameters["length_scale"], (2,)),
            "stress parameters must be finite fixed-size arrays")
    require(all(scale > 0 for scale in parameters["length_scale"]), "length scales must be positive metres")
    require(parameters["A"][0] == [0, 0, 1, 0] and parameters["A"][1] == [0, 0, 0, 1]
            and parameters["b"][:2] == [0, 0] and parameters["L"][:2] == [[0, 0], [0, 0]]
            and manifest["noise_support"] == ["vx", "vy"], "position kinematics or noise support violated")
    require(manifest["drift_implementation"] == "tanh-affine-velocity-v1"
            and manifest["diffusion_implementation"] == "additive-velocity-v1", "unsupported stress implementation")
    for key in ("parameter_hash", "code_hash", "context_hash", "profile_hash", "protocol_hash"):
        require(_hash(manifest[key]), "invalid content hash: " + key)
    require(manifest["parameter_hash"] == content_hash(parameters), "stress parameter hash differs")
    require(manifest["training_source"] == {"kind": "synthetic", "training_data_hash": None,
            "reason": "fixed generator; no fitted trajectory data"}, "no real/trained provenance is admitted")
    require(manifest["capabilities"] == ["generic-drift", "constant-diffusion", "velocity-noise"],
            "nonlinear stress must not claim an exact affine capability")
    require(manifest["qualification"] == {"scope": "stress-fixture", "scientific_qualification": False},
            "stress fixture metadata cannot grant scientific qualification")
    require(manifest["license_scope"] == {"input": "synthetic-only", "export": "public-generator"}, "invalid scope")
    require(_hash(expected_hash), "an expected stress package hash is required")
    require(content_hash(manifest) == expected_hash, "stress content differs from expected hash")
