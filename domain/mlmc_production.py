"""Separately frozen production registration, never pilot-triggered execution."""

from dataclasses import asdict, dataclass
import math

from .errors import DataValidationError
from .frozen_dynamics import content_hash


@dataclass(frozen=True)
class MLMCProductionPolicy:
    request_hash: str
    model_package_hash: str
    code_hash: str
    pilot_request_hash: str
    pilot_policy_hash: str
    reference_policy_hash: str
    level_samples: tuple[int, ...]
    maximum_job_seconds: float
    schema_version: str = "affine-mlmc-production-policy-v1"

    def validate(self, package, request, source):
        request.validate()
        hashes = (self.request_hash, self.model_package_hash, self.code_hash,
            self.pilot_request_hash, self.pilot_policy_hash, self.reference_policy_hash)
        if (self.schema_version != "affine-mlmc-production-policy-v1"
                or any(type(v) is not str or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in hashes)
                or self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or self.code_hash != source or self.pilot_request_hash == request.request_hash
                or type(self.level_samples) is not tuple or not 3 <= len(self.level_samples) <= 9
                or any(type(n) is not int or not 2 <= n <= 1_000_000 for n in self.level_samples)
                or sum(self.level_samples) != request.samples or request.samples > 1_000_000
                or request.steps*2**(len(self.level_samples)-1) > 8192
                or type(self.maximum_job_seconds) not in (int, float)
                or not 0 < self.maximum_job_seconds <= 7200 or not math.isfinite(self.maximum_job_seconds)):
            raise DataValidationError("complete bound independent MLMC production policy required")
        return self

    def manifest(self):
        value = asdict(self)
        value["level_samples"] = list(self.level_samples)
        return value

    @property
    def policy_hash(self):
        return content_hash(self.manifest())

    @classmethod
    def from_manifest(cls, value):
        try:
            if type(value) is not dict or type(value.get("level_samples")) is not list:
                raise ValueError
            fields = {**value, "level_samples": tuple(value["level_samples"])}
            result = cls(**fields)
            if result.manifest() != value:
                raise ValueError
            return result
        except (TypeError, ValueError) as exc:
            raise DataValidationError("canonical MLMC production registration required") from exc
