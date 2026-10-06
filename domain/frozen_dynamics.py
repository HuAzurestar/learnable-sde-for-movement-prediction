"""Content-bound synthetic affine dynamics, independent of storage or runners.

This first package version covers four-state, constant-coefficient Itô models.
It carries no real-data grant and cannot qualify a trained nonlinear model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from .errors import DataValidationError


STATE_NAMES = ("x", "y", "vx", "vy")
STATE_UNITS = ("m", "m", "m/s", "m/s")


def content_hash(value) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _array(value, shape):
    if type(value) is not list or len(value) != shape[0]:
        return False
    if len(shape) > 1:
        return all(_array(row, shape[1:]) for row in value)
    try:
        return all(type(x) in (int, float) and math.isfinite(x) for x in value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class FrozenDynamicsPackage:
    """Keep canonical bytes, never mutable caller-owned dictionaries or arrays."""

    _document: bytes

    @classmethod
    def from_manifest(cls, manifest, *, expected_hash: str):
        validate_package(manifest, expected_hash=expected_hash)
        return cls(_encode(manifest))

    @property
    def package_hash(self):
        return hashlib.sha256(self._document).hexdigest()

    def manifest(self):
        return json.loads(self._document)

    def validate(self):
        if type(self._document) is not bytes or len(self._document) > 65536:
            raise DataValidationError("frozen package must contain bounded canonical bytes")
        validate_package(self.manifest(), expected_hash=self.package_hash)


def validate_package(manifest, *, expected_hash: str):
    required = {"schema_version", "package_id", "family", "frozen", "state_names", "units",
                "coordinate_system", "time_unit", "noise_convention", "parameters", "parameter_hash",
                "code_hash", "drift_implementation", "diffusion_implementation", "noise_support",
                "context_hash", "profile_hash", "training_source", "protocol_hash", "capabilities",
                "qualification", "license_scope"}
    def require(condition, detail):
        if not condition:
            raise DataValidationError(detail)
    require(type(manifest) is dict and set(manifest) == required, "invalid frozen dynamics fields")
    require(manifest["schema_version"] == "frozen-affine-dynamics-v1" and manifest["family"] == "affine-oracle"
            and manifest["frozen"] is True, "only frozen synthetic affine packages are supported")
    require(type(manifest["package_id"]) is str and 0 < len(manifest["package_id"]) <= 128, "invalid package ID")
    require(manifest["state_names"] == list(STATE_NAMES) and manifest["units"] == list(STATE_UNITS)
            and manifest["coordinate_system"] == "local-cartesian" and manifest["time_unit"] == "s"
            and manifest["noise_convention"] == "ito", "incompatible state, units or noise convention")
    parameters = manifest["parameters"]
    require(type(parameters) is dict and set(parameters) == {"A", "b", "L"}, "invalid affine parameters")
    require(_array(parameters["A"], (4, 4)) and _array(parameters["b"], (4,))
            and _array(parameters["L"], (4, 2)), "affine parameters must be finite fixed-size arrays")
    require(parameters["A"][0] == [0, 0, 1, 0] and parameters["A"][1] == [0, 0, 0, 1]
            and parameters["b"][:2] == [0, 0] and parameters["L"][:2] == [[0, 0], [0, 0]]
            and manifest["noise_support"] == ["vx", "vy"], "position kinematics or velocity noise support violated")
    require(manifest["drift_implementation"] == "affine-constant-v1"
            and manifest["diffusion_implementation"] == "additive-velocity-v1", "unsupported implementation")
    for name in ("parameter_hash", "code_hash", "context_hash", "profile_hash", "protocol_hash"):
        require(_hash(manifest[name]), "invalid content hash: " + name)
    require(manifest["parameter_hash"] == content_hash(parameters), "parameter content differs from hash")
    require(manifest["training_source"] == {"kind": "synthetic", "training_data_hash": None,
            "reason": "fixed generator; no fitted trajectory data"}, "unsupported training provenance")
    require(manifest["capabilities"] == ["affine-exact", "constant-diffusion", "velocity-noise"], "invalid capabilities")
    require(manifest["qualification"] == {"scope": "oracle-fixture", "scientific_qualification": False},
            "fixture metadata cannot grant scientific qualification")
    require(manifest["license_scope"] == {"input": "synthetic-only", "export": "public-generator"}, "invalid license scope")
    require(_hash(expected_hash), "an expected package hash is required")
    try:
        actual = content_hash(manifest)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError("package must contain finite JSON values") from exc
    require(actual == expected_hash, "package content differs from expected hash")
