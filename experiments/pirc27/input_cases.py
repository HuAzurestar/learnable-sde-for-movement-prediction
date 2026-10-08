"""Declared immutable source/case bindings, never read authority or independence.

One instance per source block is selected before exposure. Numerical seeds stay
inside that block. Real prefix consumption requires a separate admitted adapter.
"""

from dataclasses import asdict, dataclass, fields
import math

from application.research_preregistration import same_source
from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash, _hash
from infrastructure.research_store import ResearchError, identifier


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _vector(value):
    return type(value) is tuple and len(value) == 4 and all(_finite(x) for x in value)


@dataclass(frozen=True)
class StudyInputPolicy:
    policy_id: str
    initialization_policy_hash: str
    origin_policy_hash: str
    selection_hash: str
    instance_policy: str = "one-preselected-instance-per-source-block"

    def validate(self):
        identifier(self.policy_id)
        if (self.instance_policy != "one-preselected-instance-per-source-block"
                or any(not _hash(getattr(self, k)) for k in
                    ("initialization_policy_hash", "origin_policy_hash", "selection_hash"))):
            raise DataValidationError("explicit frozen initialization/origin/selection policy required")

    def manifest(self):
        self.validate()
        return asdict(self)

    @property
    def policy_hash(self):
        return content_hash(self.manifest())


@dataclass(frozen=True)
class StudyInputCase:
    block_id: str
    instance_id: str
    dataset_id: str
    release_id: str
    source_block_id: str
    source_sha256: str
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]
    origin: float
    history_cutoff: float
    source_kind: str = "synthetic-recipe"
    split_role: str = "validation"

    def validate(self):
        for key in ("block_id", "instance_id", "dataset_id", "release_id", "source_block_id"):
            identifier(getattr(self, key))
        if (not _hash(self.source_sha256) or not _vector(self.initial_mean)
                or type(self.initial_covariance) is not tuple or len(self.initial_covariance) != 4
                or not all(_vector(row) for row in self.initial_covariance)
                or not _finite(self.origin) or not _finite(self.history_cutoff)
                or self.history_cutoff > self.origin
                or type(self.source_kind) is not str or self.source_kind not in {"synthetic-recipe", "private-past-prefix"}
                or type(self.split_role) is not str or self.split_role not in {"train", "selection", "validation", "test", "final-eval"}):
            raise DataValidationError("invalid bounded source identity or frozen four-state input law")

    def manifest(self):
        self.validate()
        return asdict(self)

    def source_identity(self):
        return {"dataset_id": self.dataset_id, "release_id": self.release_id,
            "source_block_id": self.source_block_id, "sha256": self.source_sha256}


def validate_cases(cases, policy, *, selection_hash):
    if type(cases) is not tuple or len(cases) > 64 or any(type(c) is not StudyInputCase for c in cases):
        raise DataValidationError("bounded immutable input case axis required")
    if not cases:
        if policy is not None:
            raise DataValidationError("input policy without explicit cases")
        return
    if type(policy) is not StudyInputPolicy:
        raise DataValidationError("explicit input policy required for source cases")
    policy.validate()
    if policy.selection_hash != selection_hash:
        raise DataValidationError("case selection differs from frozen study selection")
    for case in cases:
        case.validate()
    if (len({c.block_id for c in cases}) != len(cases)
            or len({c.instance_id for c in cases}) != len(cases)):
        raise DataValidationError("one preselected instance per unique block is required")
    for i, case in enumerate(cases):
        if any(same_source(case.source_identity(), other.source_identity()) for other in cases[:i]):
            raise DataValidationError("source aliases or correlated instances cannot create independent blocks")


def input_binding(case, policy):
    body = {"schema_version": "propagation-input-binding-v1", "case": case.manifest(),
        "policy": policy.manifest(), "qualification": "declared-only-not-scientific"}
    return {**body, "binding_hash": content_hash(body)}


def _restore(kind, value):
    if type(value) is not dict or set(value) != {f.name for f in fields(kind)}:
        raise ValueError("complete input binding fields required")
    if kind is StudyInputCase:
        value = dict(value)
        if (type(value["initial_mean"]) is not list or len(value["initial_mean"]) != 4
                or type(value["initial_covariance"]) is not list or len(value["initial_covariance"]) != 4
                or any(type(row) is not list or len(row) != 4 for row in value["initial_covariance"])):
            raise ValueError("input law must contain immutable JSON arrays")
        value["initial_mean"] = tuple(value["initial_mean"])
        value["initial_covariance"] = tuple(tuple(row) for row in value["initial_covariance"])
    restored = kind(**value)
    restored.validate()
    return restored


def validate_cell_input_binding(cell, *, request=None, protocol_block=None, selection_hash=None):
    """Verify exact declared input and source before granting/reading any bytes.

    No file/store/grant is opened. A matching declaration is not causal
    initialization proof, permission, independent evidence or qualification.
    """
    if "input_binding" not in cell:
        if "instance_id" in cell:
            raise ResearchError("CONTRACT_MISMATCH", "input instance has no frozen source binding")
        return
    try:
        binding = cell["input_binding"]
        if (type(binding) is not dict or set(binding) != {"schema_version", "case", "policy", "qualification", "binding_hash"}
                or binding["schema_version"] != "propagation-input-binding-v1"
                or binding["qualification"] != "declared-only-not-scientific"):
            raise ValueError("input binding schema/claim differs")
        case, policy = _restore(StudyInputCase, binding["case"]), _restore(StudyInputPolicy, binding["policy"])
        if (content_hash(input_binding(case, policy)) != content_hash(binding) or cell["block_id"] != case.block_id
                or cell["instance_id"] != case.instance_id
                or cell["visibility"] != ("synthetic" if case.source_kind == "synthetic-recipe" else "restricted")):
            raise ValueError("input case/source/visibility identity differs")
        if selection_hash is not None and policy.selection_hash != selection_hash:
            raise ValueError("input selection differs from frozen study")
        actual_request = asdict(request) if request is not None else cell["propagation_request"]
        keys = ("initial_mean", "initial_covariance", "origin", "history_cutoff")
        if content_hash({k: actual_request[k] for k in keys}) != content_hash({k: case.manifest()[k] for k in keys}):
            raise ValueError("request differs from frozen input law/origin")
        horizons = actual_request["horizons"]
        if type(horizons) not in (list, tuple) or len(horizons) != 1:
            raise ValueError("case pairing requires one frozen horizon")
        paired = {"model_package_hash": actual_request["model_package_hash"],
            **{k: actual_request[k] for k in keys}, "horizon": horizons[0],
            "functional": {"functional_id": cell["functional_id"], "kind": actual_request["functional"],
                **{k: actual_request[k] for k in ("normal", "threshold", "closed", "tolerance")}},
            "seed": actual_request["seed"], "input_binding_hash": binding["binding_hash"]}
        if "calibration_binding" in cell:
            paired["calibration_binding_hash"] = cell["calibration_binding"]["binding_hash"]
        if actual_request["coupling_id"] != "paired-"+content_hash(paired):
            raise ValueError("case pairing root differs from frozen source/input/functional")
        dimensions = cell["comparison_dimensions"]
        if (dimensions.get("input_policy") != policy.policy_hash
                or dimensions.get("initialization") != {"policy_hash": policy.initialization_policy_hash}
                or dimensions.get("prediction_origin") != {"policy_hash": policy.origin_policy_hash}
                or dimensions.get("input_population") != {"dataset_id": case.dataset_id, "release_id": case.release_id,
                    "source_kind": case.source_kind, "split_role": case.split_role}):
            raise ValueError("input comparison policy/population differs")
        if protocol_block is not None:
            original = {"dataset_id": protocol_block["dataset_id"], "release_id": protocol_block["release_id"],
                "source_block_id": protocol_block.get("source_block_id", protocol_block["block_id"]),
                "sha256": protocol_block["sha256"]}
            if (original != case.source_identity() or protocol_block["block_id"] != case.block_id
                    or protocol_block["split_role"] != case.split_role):
                raise ValueError("case source differs from registered protocol block")
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "frozen input case: " + str(exc)) from exc
