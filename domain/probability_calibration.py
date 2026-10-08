"""Frozen affine spatial-probability controls, never execution/read authority."""

from dataclasses import dataclass, fields
import math

from domain.errors import DataValidationError
from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash, _hash, STATE_NAMES, STATE_UNITS
from domain.propagation import PropagationRequest


@dataclass(frozen=True)
class AffineHalfspaceCalibrationPolicy:
    source_request_hash: str
    model_package_hash: str
    code_hash: str
    target_probability: float
    maximum_relative_probability_error: float
    maximum_relative_probability_width: float
    maximum_operations: int
    maximum_job_seconds: float
    schema_version: str = "affine-halfspace-calibration-policy-v1"

    def _validate_scalars(self):
        def positive(value):
            try:
                return type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                return False
        if (type(self.schema_version) is not str
                or self.schema_version != "affine-halfspace-calibration-policy-v1"
                or any(not _hash(value) for value in
                    (self.source_request_hash, self.model_package_hash, self.code_hash))
                or any(not positive(value) for value in (self.target_probability,
                    self.maximum_relative_probability_error, self.maximum_relative_probability_width,
                    self.maximum_job_seconds))
                or not 1e-8 <= self.target_probability <= .1
                or self.maximum_relative_probability_error > 1
                or self.maximum_relative_probability_width > 1
                or type(self.maximum_operations) is not int
                or not 1 <= self.maximum_operations <= 200_000
                or self.maximum_job_seconds > 1800):
            raise DataValidationError("explicit bounded scalar probability calibration policy required")

    def validate(self, package, request, source_hash):
        self._validate_scalars()
        if type(package) is not FrozenDynamicsPackage or type(request) is not PropagationRequest:
            raise DataValidationError("calibration requires frozen affine dynamics and endpoint request")
        package.validate()
        request.validate()
        if (not _hash(source_hash) or self.code_hash != source_hash
                or self.model_package_hash != package.package_hash
                or request.model_package_hash != package.package_hash
                or self.source_request_hash != request.request_hash):
            raise DataValidationError("probability calibration source/model/request binding differs")
        if (request.functional != "endpoint-halfspace" or request.threshold != 0
                or request.normal[2:] != (0., 0.)):
            raise DataValidationError("calibration requires a zero-template position halfspace in metres")
        return self

    def manifest(self):
        # Validate before recursively copying or encoding any caller-owned value.
        self._validate_scalars()
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def policy_hash(self):
        return content_hash(self.manifest())

    @classmethod
    def from_manifest(cls, document):
        names = {field.name for field in fields(cls)}
        if (type(document) is not dict or set(document) != names
                or any(type(value) not in (str, int, float) for value in document.values())):
            raise DataValidationError("complete bounded scalar probability calibration policy required")
        result = cls(**document)
        result._validate_scalars()
        return result


def halfspace_geometry(package, request, *, causal_input_hash):
    """A physical event key, not method identity, independence or read authority.

    Callers must supply the genuine frozen causal input identity. The owner
    binds it to the original source; a caller-supplied hash alone is no proof.
    Numerical tolerance, seed, grid, sample count and budget arm are excluded.
    """
    if type(package) is not FrozenDynamicsPackage or type(request) is not PropagationRequest:
        raise DataValidationError("frozen affine package and halfspace request required")
    package.validate()
    request.validate()
    if (request.model_package_hash != package.package_hash or not _hash(causal_input_hash)
            or request.functional != "endpoint-halfspace" or request.normal[2:] != (0., 0.)):
        raise DataValidationError("explicit four-state spatial halfspace and causal input identity required")
    return {"schema_version": "affine-halfspace-geometry-v1", "model_package_hash": package.package_hash,
        "initial_mean": list(request.initial_mean),
        "initial_covariance": [list(row) for row in request.initial_covariance],
        "origin": request.origin, "history_cutoff": request.history_cutoff, "horizon": request.horizons[0],
        "functional": request.functional, "functional_version": request.functional_version,
        "normal": list(request.normal), "threshold": request.threshold, "closed": request.closed,
        "state_order": list(STATE_NAMES), "units": list(STATE_UNITS), "time_unit": "s",
        "coordinate_system": request.region_coordinate_system, "threshold_units": "m",
        "causal_input_hash": causal_input_hash}
