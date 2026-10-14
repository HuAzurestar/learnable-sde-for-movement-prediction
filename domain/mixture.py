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


def _validate_settings(value):
    # Inspect bounded primitive fields before dataclass deep-copy or JSON
    # serialization. Invalid nested/cyclic/oversized inputs are never copied.
    if (type(value.component_cap) is not int or not 1 <= value.component_cap <= 32
            or not finite(value.merge_distance) or value.merge_distance < 0
            or not finite(value.prune_weight) or not 0 <= value.prune_weight < 1
            or type(value.state_scales) is not tuple or len(value.state_scales) != 4
            or any(not finite(s) or s <= 0 for s in value.state_scales)
            or not finite(value.maximum_discarded_mass) or not 0 <= value.maximum_discarded_mass < 1
            or type(value.maximum_work_units) is not int or not 1 <= value.maximum_work_units <= 1_000_000
            or not finite(value.maximum_job_seconds) or not 0 < value.maximum_job_seconds <= 7200):
        raise DataValidationError("invalid bounded primitive mixture settings")


@dataclass(frozen=True)
class MixtureSettings:
    """Explicit immutable design settings; binding happens per full request."""

    component_cap: int
    merge_distance: float
    prune_weight: float
    state_scales: tuple[float, ...]
    maximum_discarded_mass: float
    maximum_work_units: int
    maximum_job_seconds: float

    def validate(self):
        _validate_settings(self)

    def bind(self, package, request, source_hash):
        self.validate()
        policy = MixturePolicy(request.request_hash, package.package_hash, source_hash, **asdict(self))
        policy.validate(package, request, source_hash)
        return policy


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
        self.validate_structure()
        return 8*self.component_cap*(self.component_cap+1+4**3)

    def validate_structure(self):
        _validate_settings(self)
        if (self.schema_version != "bounded-cubature-mixture-policy-v1"
                or not all(_hash(h) for h in (self.request_hash, self.model_package_hash, self.code_hash))):
            raise DataValidationError("invalid bounded mixture schema/hash structure")

    def validate(self, package, request, source_hash):
        self.validate_structure()
        request.validate()
        package.validate()
        if (self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or request.model_package_hash != package.package_hash or self.code_hash != source_hash
                or request.steps*self.work_per_step > self.maximum_work_units):
            raise DataValidationError("invalid frozen bounded mixture policy/work/request/source")

    def manifest(self):
        self.validate_structure()
        value = asdict(self)
        value["state_scales"] = list(self.state_scales)
        return value

    @property
    def policy_hash(self):
        return content_hash(self.manifest())

    @classmethod
    def from_manifest(cls, value):
        try:
            if (type(value) is not dict or len(value) != len(cls.__dataclass_fields__)
                    or set(value) != set(cls.__dataclass_fields__)
                    or type(value["state_scales"]) is not list or len(value["state_scales"]) != 4
                    or any(not finite(s) or s <= 0 for s in value["state_scales"])):
                raise ValueError
            policy = cls(**{**value, "state_scales": tuple(value["state_scales"])})
            policy.validate_structure()
            if policy.manifest() != value:
                raise ValueError
            return policy
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError("complete canonical mixture policy required") from exc
