"""Frozen per-request numerical qualification thresholds, not permissions."""

from dataclasses import asdict, dataclass
import math

from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash


@dataclass(frozen=True)
class AffineQualificationPolicy:
    request_hash: str
    model_package_hash: str
    code_hash: str
    method: str
    maximum_reference_width: float
    maximum_functional_roundoff: float
    maximum_time_bias: float
    maximum_scaled_transition_norm: float
    state_scales: tuple[float, ...]
    maximum_operations: int
    maximum_job_seconds: float
    schema_version: str = "affine-analytic-qualification-policy-v1"

    def validate(self, package, request, method, source_hash):
        def finite_positive(value):
            try:
                return type(value) in (float, int) and math.isfinite(value) and value > 0
            except OverflowError:
                return False
        hashes = (self.request_hash, self.model_package_hash, self.code_hash)
        if (self.schema_version != "affine-analytic-qualification-policy-v1"
                or any(type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in hashes)
                or self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or self.code_hash != source_hash or self.method != method or method not in {"exact", "gaussian"}
                or any(not finite_positive(value) for value in (self.maximum_reference_width,
                    self.maximum_functional_roundoff, self.maximum_time_bias, self.maximum_scaled_transition_norm,
                    self.maximum_job_seconds))
                or self.maximum_job_seconds > 1800
                or type(self.maximum_operations) is not int or not 1 <= self.maximum_operations <= 400_001
                or type(self.state_scales) is not tuple or len(self.state_scales) != 4
                or any(not finite_positive(value) for value in self.state_scales)):
            raise DataValidationError("explicit bound affine analytic qualification policy differs or is invalid")
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
        except (TypeError, ValueError) as exc:
            raise DataValidationError("complete canonical affine qualification policy required") from exc
