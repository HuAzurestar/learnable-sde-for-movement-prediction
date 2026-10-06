"""Frozen affine reference caps for an independent MLMC pilot, not approval."""

from dataclasses import asdict, dataclass
import math

from .errors import DataValidationError
from .frozen_dynamics import content_hash


@dataclass(frozen=True)
class AffineMLMCReferencePolicy:
    request_hash: str
    model_package_hash: str
    code_hash: str
    pilot_policy_hash: str
    level_samples: tuple[int, ...]
    maximum_reference_width: float
    maximum_scaled_transition_norm: float
    state_scales: tuple[float, ...]
    maximum_operations: int
    maximum_job_seconds: float
    schema_version: str = "affine-mlmc-reference-policy-v1"

    def validate(self, package, request, pilot_policy, source_hash):
        from .mlmc_pilot import MLMCPilotPolicy
        if type(pilot_policy) is not MLMCPilotPolicy:
            raise DataValidationError("explicit frozen MLMC pilot policy required")
        pilot_policy.validate(request, self.level_samples)
        def positive(value):
            try:
                return type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                return False
        hashes = (self.request_hash, self.model_package_hash, self.code_hash, self.pilot_policy_hash)
        if (self.schema_version != "affine-mlmc-reference-policy-v1"
                or any(type(v) is not str or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in hashes)
                or self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or self.code_hash != source_hash or self.pilot_policy_hash != pilot_policy.policy_hash
                or type(self.level_samples) is not tuple
                or type(self.state_scales) is not tuple or len(self.state_scales) != 4
                or any(not positive(v) for v in self.state_scales)
                or any(not positive(v) for v in (self.maximum_reference_width,
                    self.maximum_scaled_transition_norm, self.maximum_job_seconds))
                or self.maximum_job_seconds > 1800 or type(self.maximum_operations) is not int
                or not 1 <= self.maximum_operations <= 400_001):
            raise DataValidationError("complete bound affine MLMC reference policy required")
        return self

    def manifest(self):
        value = asdict(self)
        value["level_samples"] = list(self.level_samples)
        value["state_scales"] = list(self.state_scales)
        return value

    @property
    def policy_hash(self):
        return content_hash(self.manifest())

    @classmethod
    def from_manifest(cls, value):
        try:
            if type(value) is not dict or any(type(value.get(k)) is not list for k in ("level_samples", "state_scales")):
                raise ValueError
            fields = dict(value)
            for key in ("level_samples", "state_scales"):
                fields[key] = tuple(fields[key])
            result = cls(**fields)
            if result.manifest() != value:
                raise ValueError
            return result
        except (TypeError, ValueError) as exc:
            raise DataValidationError("canonical affine MLMC reference policy required") from exc
