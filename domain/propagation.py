"""Immutable per-cell endpoint requests and estimator/error contracts."""

from dataclasses import asdict, dataclass
import math

from .errors import DataValidationError
from .frozen_dynamics import content_hash


@dataclass(frozen=True)
class PropagationRequest:
    request_id: str
    model_package_hash: str
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]
    origin: float
    history_cutoff: float
    horizons: tuple[float, ...]
    functional: str
    seed: int
    coupling_id: str
    arm_id: str
    steps: int = 32
    samples: int = 1024
    chunk_size: int = 256
    normal: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    threshold: float = 0.0
    closed: bool = True
    tolerance: float = 0.01
    functional_version: str = "endpoint-v1"
    region_coordinate_system: str = "local-cartesian"

    def validate(self):
        def finite(value):
            try:
                return type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                return False
        def vector(value):
            return type(value) is tuple and len(value) == 4 and all(finite(x) for x in value)
        valid = (all(type(x) is str and 0 < len(x) <= 128 for x in
                     (self.request_id, self.coupling_id, self.arm_id))
                 and type(self.model_package_hash) is str and len(self.model_package_hash) == 64
                 and all(x in "0123456789abcdef" for x in self.model_package_hash)
                 and vector(self.initial_mean) and type(self.initial_covariance) is tuple
                 and len(self.initial_covariance) == 4 and all(vector(row) for row in self.initial_covariance)
                 and finite(self.origin) and finite(self.history_cutoff) and self.history_cutoff <= self.origin
                 and type(self.horizons) is tuple and len(self.horizons) == 1
                 and finite(self.horizons[0]) and self.horizons[0] > 0
                 and self.functional in ("endpoint-x", "endpoint-halfspace")
                 and self.functional_version == "endpoint-v1"
                 and self.region_coordinate_system == "local-cartesian"
                 and type(self.seed) is int and 0 <= self.seed < 2**63
                 and type(self.steps) is int and 1 <= self.steps <= 8192
                 and type(self.samples) is int and 2 <= self.samples <= 1_000_000
                 and type(self.chunk_size) is int and 1 <= self.chunk_size <= 256
                 and vector(self.normal) and any(x != 0 for x in self.normal)
                 and finite(self.threshold) and type(self.closed) is bool
                 and finite(self.tolerance) and self.tolerance > 0)
        if not valid:
            raise DataValidationError("invalid frozen endpoint propagation request")

    @property
    def request_hash(self):
        self.validate()
        return content_hash(asdict(self))


@dataclass(frozen=True)
class ErrorComponent:
    value: float | None
    units: str
    estimated_by: str
    status: str


@dataclass(frozen=True)
class NumericalErrorBudget:
    reference: ErrorComponent
    time_discretization: ErrorComponent
    propagation_approximation: ErrorComponent
    sampling: ErrorComponent
    model: ErrorComponent


@dataclass(frozen=True)
class FunctionalResult:
    request_hash: str
    estimator_id: str
    kind: str
    estimate: float
    standard_error: float | None
    interval: tuple[float, float] | None
    interval_kind: str
    sample_count: int
    error_budget: NumericalErrorBudget
    status: str
    diagnostics: tuple[tuple[str, object], ...]

    def manifest(self):
        return asdict(self)
