"""Frozen empirical pilot rules; no authority, allocation execution or budgets."""

from dataclasses import asdict, dataclass
import math

from .errors import DataValidationError
from .frozen_dynamics import content_hash


@dataclass(frozen=True)
class MLMCPilotPolicy:
    pilot_request_hash: str
    production_seed: int
    production_coupling_id: str
    sampling_tolerance: float
    bias_tolerance: float
    maximum_variance_ratio: float
    maximum_cost_ratio: float
    minimum_level_samples: int
    maximum_samples: int
    maximum_work_steps: int
    schema_version: str = "endpoint-mlmc-pilot-policy-v1"

    def validate(self, request, level_samples):
        request.validate()
        def finite(value):
            try:
                return type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                return False
        if (self.schema_version != "endpoint-mlmc-pilot-policy-v1"
                or self.pilot_request_hash != request.request_hash
                or type(self.production_seed) is not int or not 0 <= self.production_seed < 2**63
                or self.production_seed == request.seed
                or type(self.production_coupling_id) is not str or not 0 < len(self.production_coupling_id) <= 128
                or any(ord(char) < 32 for char in self.production_coupling_id)
                or not finite(self.sampling_tolerance) or self.sampling_tolerance <= 0
                or not finite(self.bias_tolerance) or self.bias_tolerance <= 0
                or not finite(self.maximum_variance_ratio) or not 0 < self.maximum_variance_ratio < 1
                or not finite(self.maximum_cost_ratio) or self.maximum_cost_ratio <= 1
                or type(self.minimum_level_samples) is not int or not 2 <= self.minimum_level_samples <= 1_000_000
                or type(self.maximum_samples) is not int or not 2 <= self.maximum_samples <= 1_000_000
                or type(self.maximum_work_steps) is not int or not 1 <= self.maximum_work_steps <= 1_000_000
                or type(level_samples) is not tuple or not 3 <= len(level_samples) <= 9
                or any(type(n) is not int or not self.minimum_level_samples <= n <= 1_000_000 for n in level_samples)
                or sum(level_samples) != request.samples or sum(level_samples) > self.maximum_samples
                or request.steps*2**(len(level_samples)-1) > 8192
                or sum(n*(request.steps*2**level + (request.steps*2**(level-1) if level else 0))
                       for level, n in enumerate(level_samples)) > self.maximum_work_steps):
            raise DataValidationError("invalid frozen independent MLMC pilot policy/allocation")

    @property
    def policy_hash(self):
        return content_hash(asdict(self))
