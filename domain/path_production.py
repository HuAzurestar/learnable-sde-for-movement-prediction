"""Explicit independent target registration; pilot evidence is not execution."""

from dataclasses import asdict, dataclass

from .errors import DataValidationError
from .frozen_dynamics import FrozenDynamicsPackage, content_hash, _hash
from .mixture import finite
from .path_qualification import PathQualificationPolicy
from .propagation import PropagationRequest


# Only these sampling/registration fields may differ. In particular horizons,
# grid, physical initial distribution, event geometry and original arm cannot.
SAMPLING_FIELDS = ("request_id", "seed", "coupling_id", "samples", "chunk_size", "tolerance")


@dataclass(frozen=True)
class PathProductionPolicy:
    request_hash: str
    model_package_hash: str
    code_hash: str
    source_request_hash: str
    qualification_policy_hash: str
    maximum_job_seconds: float
    schema_version: str = "affine-independent-path-production-policy-v1"

    def validate_structure(self):
        if (type(self.schema_version) is not str
                or self.schema_version != "affine-independent-path-production-policy-v1"
                or not all(_hash(h) for h in (self.request_hash, self.model_package_hash, self.code_hash,
                    self.source_request_hash, self.qualification_policy_hash))
                or self.request_hash == self.source_request_hash
                or not finite(self.maximum_job_seconds) or not 0 < self.maximum_job_seconds <= 7200):
            raise DataValidationError("invalid bounded independent path production policy")

    def validate(self, package, request, qualification, source):
        self.validate_structure()
        if (type(package) is not FrozenDynamicsPackage or type(request) is not PropagationRequest
                or type(qualification) is not PathQualificationPolicy):
            raise DataValidationError("declared affine path policies and request required")
        request.validate()
        package.validate()
        qualification.validate_structure()
        if (self.request_hash != request.request_hash or self.model_package_hash != package.package_hash
                or request.model_package_hash != package.package_hash or self.code_hash != source
                or self.source_request_hash != qualification.request_hash
                or self.qualification_policy_hash != qualification.policy_hash
                or qualification.model_package_hash != self.model_package_hash
                or qualification.code_hash != self.code_hash
                or qualification.maximum_total_observed_functional_error > request.tolerance
                or qualification.minimum_effective_sample_size > request.samples
                or request.samples*request.steps > 1_000_000
                or (qualification.method == "importance" and request.functional != "endpoint-halfspace")):
            raise DataValidationError("independent path target/model/source/policy/work binding differs")

    def validate_source_request(self, package, source_request, request, qualification, source):
        self.validate(package, request, qualification, source)
        qualification.validate(package, source_request, source)
        if (source_request.seed == request.seed or source_request.coupling_id == request.coupling_id
                or source_request.request_id == request.request_id
                or any(getattr(source_request, key) != getattr(request, key)
                    for key in PropagationRequest.__dataclass_fields__ if key not in SAMPLING_FIELDS)):
            raise DataValidationError("target must retain the same declared law and original arm with distinct sampling identities")

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
            raise DataValidationError("complete canonical independent path production policy required") from exc
