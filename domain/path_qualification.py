"""Frozen per-request path/IS pilot thresholds, never scientific permissions."""

from dataclasses import asdict, dataclass

from .errors import DataValidationError
from .frozen_dynamics import FrozenDynamicsPackage, content_hash, _hash
from .mixture import finite
from .propagation import PropagationRequest


METHODS = ("euler", "heun", "reversible-heun", "importance")
# Two hard-bounded200k reference computations plus one signed subtraction.
# Stricter observed operation thresholds do not reduce the allocation proxy.
REFERENCE_WORK_CAP = 400_001


@dataclass(frozen=True)
class PathQualificationPolicy:
    request_hash: str
    model_package_hash: str
    code_hash: str
    method: str
    proposal: tuple[float, ...]
    state_scales: tuple[float, ...]
    maximum_reference_width: float
    maximum_observed_grid_error: float
    maximum_time_bias: float
    maximum_total_observed_functional_error: float
    maximum_sampling_uncertainty: float
    minimum_effective_sample_size: float
    maximum_scaled_transition_norm: float
    maximum_reference_operations: int
    maximum_job_seconds: float
    schema_version: str = "affine-path-functional-qualification-policy-v1"

    def validate_structure(self):
        if (type(self.schema_version) is not str
                or self.schema_version != "affine-path-functional-qualification-policy-v1"
                or not all(_hash(h) for h in (self.request_hash, self.model_package_hash, self.code_hash))
                or type(self.method) is not str or self.method not in METHODS
                or type(self.proposal) is not tuple or len(self.proposal) != 2
                or any(not finite(v) for v in self.proposal)
                or (self.method != "importance" and self.proposal != (0., 0.))
                or type(self.state_scales) is not tuple or len(self.state_scales) != 4
                or any(not finite(v) or v <= 0 for v in self.state_scales)
                or any(not finite(v) or v <= 0 for v in (self.maximum_reference_width,
                    self.maximum_observed_grid_error, self.maximum_time_bias,
                    self.maximum_total_observed_functional_error, self.maximum_sampling_uncertainty,
                    self.minimum_effective_sample_size, self.maximum_scaled_transition_norm,
                    self.maximum_job_seconds))
                or not 2 <= self.minimum_effective_sample_size <= 1_000_000
                or self.maximum_job_seconds > 1800
                or type(self.maximum_reference_operations) is not int
                or not 1 <= self.maximum_reference_operations <= REFERENCE_WORK_CAP):
            raise DataValidationError("invalid bounded primitive path qualification policy")

    def validate(self, package, request, source_hash):
        self.validate_structure()
        if type(package) is not FrozenDynamicsPackage or type(request) is not PropagationRequest:
            raise DataValidationError("path qualification requires the declared affine package/request")
        request.validate()
        package.validate()
        if (self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or request.model_package_hash != package.package_hash or self.code_hash != source_hash
                or self.maximum_total_observed_functional_error > request.tolerance
                or self.minimum_effective_sample_size > request.samples
                or request.samples*request.steps+REFERENCE_WORK_CAP > 1_000_000
                or (self.method == "importance" and request.functional != "endpoint-halfspace")):
            raise DataValidationError("path qualification request/model/source/work binding differs")

    def manifest(self):
        self.validate_structure()
        value = asdict(self)
        for name in ("proposal", "state_scales"):
            value[name] = list(value[name])
        return value

    @property
    def policy_hash(self):
        return content_hash(self.manifest())

    @classmethod
    def from_manifest(cls, value):
        try:
            if (type(value) is not dict or len(value) != len(cls.__dataclass_fields__)
                    or set(value) != set(cls.__dataclass_fields__)):
                raise ValueError
            for name, length in (("proposal", 2), ("state_scales", 4)):
                if type(value[name]) is not list or len(value[name]) != length:
                    raise ValueError
            fields = {**value, "proposal": tuple(value["proposal"]), "state_scales": tuple(value["state_scales"])}
            policy = cls(**fields)
            policy.validate_structure()
            if policy.manifest() != value:
                raise ValueError
            return policy
        except (TypeError, ValueError) as exc:
            raise DataValidationError("complete canonical path qualification policy required") from exc
