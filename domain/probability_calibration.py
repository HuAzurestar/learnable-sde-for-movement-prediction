"""Frozen affine spatial-probability controls, never execution/read authority."""

from dataclasses import dataclass, fields
import math

from domain.errors import DataValidationError
from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash, _hash
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
