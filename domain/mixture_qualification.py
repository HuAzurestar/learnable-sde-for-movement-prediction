"""Explicit per-functional mixture thresholds; never permissions or defaults."""

from dataclasses import asdict, dataclass

from .errors import DataValidationError
from .frozen_dynamics import FrozenDynamicsPackage, content_hash, _hash
from .mixture import MixturePolicy, finite


# Two hard-bounded200k reference computations plus one signed subtraction.
# A stricter qualification threshold must never shrink this allocation proxy.
REFERENCE_WORK_CAP = 400_001


@dataclass(frozen=True)
class MixtureQualificationPolicy:
    request_hash: str
    model_package_hash: str
    code_hash: str
    mixture_policy_hash: str
    maximum_reference_width: float
    maximum_retained_functional_error: float
    maximum_time_bias: float
    maximum_total_functional_error: float
    maximum_scaled_transition_norm: float
    maximum_reference_operations: int
    maximum_job_seconds: float
    schema_version: str = "affine-mixture-functional-qualification-policy-v1"

    def validate_structure(self):
        if (self.schema_version != "affine-mixture-functional-qualification-policy-v1"
                or not all(_hash(h) for h in (self.request_hash, self.model_package_hash,
                    self.code_hash, self.mixture_policy_hash))
                or any(not finite(v) or v <= 0 for v in (self.maximum_reference_width,
                    self.maximum_retained_functional_error, self.maximum_time_bias,
                    self.maximum_total_functional_error, self.maximum_scaled_transition_norm,
                    self.maximum_job_seconds))
                or self.maximum_job_seconds > 1800
                or type(self.maximum_reference_operations) is not int
                or not 1 <= self.maximum_reference_operations <= 400_001):
            raise DataValidationError("invalid bounded primitive mixture qualification policy")

    def validate(self, package, request, mixture, source_hash):
        self.validate_structure()
        if type(package) is not FrozenDynamicsPackage or type(mixture) is not MixturePolicy:
            raise DataValidationError("mixture qualification requires the declared affine package and policy")
        mixture.validate(package, request, source_hash)
        if (self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or self.code_hash != source_hash or self.mixture_policy_hash != mixture.policy_hash
                or self.maximum_total_functional_error > request.tolerance
                or self.maximum_job_seconds > mixture.maximum_job_seconds
                or request.steps*mixture.work_per_step+REFERENCE_WORK_CAP > 1_000_000):
            raise DataValidationError("mixture qualification request/model/source/policy/work binding differs")

    def manifest(self):
        self.validate_structure()
        return asdict(self)

    @property
    def policy_hash(self):
        return content_hash(self.manifest())

    @classmethod
    def from_manifest(cls, value):
        try:
            if (type(value) is not dict or len(value) != len(cls.__dataclass_fields__)
                    or set(value) != set(cls.__dataclass_fields__)):
                raise ValueError
            policy = cls(**value)
            policy.validate_structure()
            if policy.manifest() != value:
                raise ValueError
            return policy
        except (TypeError, ValueError) as exc:
            raise DataValidationError("complete canonical mixture qualification policy required") from exc
