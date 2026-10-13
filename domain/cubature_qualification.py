"""Per-request affine cubature numerical thresholds, never read authority."""

from dataclasses import asdict, dataclass
import math

from domain.errors import DataValidationError
from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash, _hash


@dataclass(frozen=True)
class CubatureQualificationPolicy:
    request_hash: str
    model_package_hash: str
    code_hash: str
    maximum_reference_width: float
    maximum_functional_roundoff: float
    maximum_time_bias: float
    maximum_scaled_transition_norm: float
    state_scales: tuple[float, ...]
    maximum_operations: int
    maximum_job_seconds: float
    schema_version: str = "affine-cubature-qualification-policy-v1"

    def validate(self, package, request, source_hash):
        def positive(value):
            try:
                return type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                return False
        if (type(package) is not FrozenDynamicsPackage
                or self.schema_version != "affine-cubature-qualification-policy-v1"
                or any(not _hash(value) for value in (self.request_hash, self.model_package_hash, self.code_hash))
                or self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or self.code_hash != source_hash
                or any(not positive(value) for value in (self.maximum_reference_width,
                    self.maximum_functional_roundoff, self.maximum_time_bias,
                    self.maximum_scaled_transition_norm, self.maximum_job_seconds))
                or self.maximum_job_seconds > 1800
                or type(self.maximum_operations) is not int or not 1 <= self.maximum_operations <= 400001
                or type(self.state_scales) is not tuple or len(self.state_scales) != 4
                or any(not positive(value) for value in self.state_scales)):
            raise DataValidationError("explicit affine cubature numerical policy differs or is invalid")
        package.validate()
        request.validate()
        return self

    def manifest(self):
        value = asdict(self)
        value["state_scales"] = list(self.state_scales)
        return value

    @property
    def policy_hash(self):
        return content_hash(self.manifest())

    @classmethod
    def from_manifest(cls, document):
        try:
            if type(document) is not dict or type(document.get("state_scales")) is not list:
                raise ValueError
            value = dict(document)
            value["state_scales"] = tuple(value["state_scales"])
            result = cls(**value)
            if result.manifest() != document:
                raise ValueError
            return result
        except (ValueError, TypeError) as exc:
            raise DataValidationError("complete canonical affine cubature policy required") from exc
