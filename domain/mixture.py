"""Frozen bounded cubature-mixture reduction, never a qualification flag."""

from dataclasses import asdict, dataclass
import math

from .errors import DataValidationError
from .frozen_dynamics import content_hash, _hash


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class MixturePolicy:
    request_hash: str
    model_package_hash: str
    code_hash: str
    component_cap: int
    merge_distance: float
    prune_weight: float
    state_scales: tuple[float, ...]
    maximum_discarded_mass: float
    maximum_work_units: int
    maximum_job_seconds: float
    schema_version: str = "bounded-cubature-mixture-policy-v1"

    @property
    def work_per_step(self):
        # Conservative fixed proxy for candidate roots/moments and capped
        # nearest-cluster comparisons. Not measured FLOPs or a wall-time bound.
        return 8*self.component_cap*(self.component_cap+1+4**3)

    def validate(self, package, request, source_hash):
        request.validate()
        package.validate()
        if (self.schema_version != "bounded-cubature-mixture-policy-v1"
                or not all(_hash(h) for h in (self.request_hash, self.model_package_hash, self.code_hash))
                or self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or request.model_package_hash != package.package_hash or self.code_hash != source_hash
                or type(self.component_cap) is not int or not 1 <= self.component_cap <= 32
                or not finite(self.merge_distance) or self.merge_distance < 0
                or not finite(self.prune_weight) or not 0 <= self.prune_weight < 1
                or type(self.state_scales) is not tuple or len(self.state_scales) != 4
                or any(not finite(s) or s <= 0 for s in self.state_scales)
                or not finite(self.maximum_discarded_mass) or not 0 <= self.maximum_discarded_mass < 1
                or type(self.maximum_work_units) is not int or not 1 <= self.maximum_work_units <= 1_000_000
                or request.steps*self.work_per_step > self.maximum_work_units
                or not finite(self.maximum_job_seconds) or not 0 < self.maximum_job_seconds <= 7200):
            raise DataValidationError("invalid frozen bounded mixture policy/work/request/source")

    def manifest(self):
        value = asdict(self)
        value["state_scales"] = list(self.state_scales)
        return value

    @property
    def policy_hash(self):
        return content_hash(self.manifest())

    @classmethod
    def from_manifest(cls, value):
        try:
            if type(value) is not dict or set(value) != set(cls.__dataclass_fields__) or type(value["state_scales"]) is not list:
                raise ValueError
            policy = cls(**{**value, "state_scales": tuple(value["state_scales"])})
            if policy.manifest() != value:
                raise ValueError
            return policy
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError("complete canonical mixture policy required") from exc
