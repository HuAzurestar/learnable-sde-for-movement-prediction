"""Bounded declarations for a complete geometry table; never source authority.

CALIBRATED describes declared saved evidence, not successful shared admission.
The owner must be checked afresh before any method consumes this table. Failed
and absent sources retain their slots rather than select an easier threshold.
"""

from dataclasses import dataclass
import json
import math

from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash, _encode, _hash
from domain.probability_calibration import AffineHalfspaceCalibrationPolicy
from infrastructure.research_store import identifier


MAX_DOCUMENT_BYTES = 8192
GEOMETRY_FIELDS = frozenset({"schema_version", "model_package_hash", "initial_mean", "initial_covariance",
    "origin", "history_cutoff", "horizon", "functional", "functional_version", "normal", "threshold",
    "closed", "state_order", "units", "time_unit", "coordinate_system", "threshold_units", "causal_input_hash"})
POINTER_FIELDS = frozenset({"schema_version", "policy", "source_attempt_id", "source_artifact_id",
    "source_authorization_id", "source_authorization_version"})
MANIFEST_FIELDS = frozenset({"schema_version", "functional_id", "model_family_id", "model_package_hash",
    "horizon", "input_case_id", "model_configuration_id", "status", "reason", "geometry", "source_pointer",
    "source_evidence_hash", "consumer_study_id", "geometry_hash", "binding_hash"})


def _require(condition, detail):
    if not condition:
        raise DataValidationError("frozen calibration table: " + detail)


def _bounded(value):
    # Check shape before JSON copy/hash; no recursive traversal of cycles.
    pending, visited, count = [(value, 0)], set(), 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _require(count <= 512 and depth <= 8, "metadata graph exceeds bound")
        if type(item) in (dict, list):
            _require(id(item) not in visited, "cyclic or aliased metadata graph")
            visited.add(id(item))
            _require(len(item) <= 64, "metadata container exceeds bound")
            if type(item) is dict:
                _require(all(type(k) is str and len(k) <= 128 for k in item), "invalid metadata keys")
                pending.extend((v, depth+1) for v in item.values())
            else:
                pending.extend((v, depth+1) for v in item)
        elif type(item) is str:
            _require(len(item) <= 256, "metadata string exceeds bound")
        elif type(item) in (int, float):
            _require((type(item) is not int or item.bit_length() <= 1023) and math.isfinite(item),
                "invalid finite metadata scalar")
        else:
            _require(item is None or type(item) is bool, "unsupported metadata value")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def _read(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_DOCUMENT_BYTES, "bounded canonical bytes required")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
        _bounded(value)
        _require(type(value) is dict and _encode(value) == raw, "noncanonical metadata")
        return value
    except (ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError("invalid bounded calibration metadata") from exc


@dataclass(frozen=True)
class StudyCalibration:
    functional_id: str
    model_family_id: str
    model_package_hash: str
    horizon: float
    input_case_id: str | None = None
    model_configuration_id: str | None = None
    status: str = "UNAVAILABLE"
    reason: str = "no qualified source"
    geometry_document: bytes | None = None
    source_pointer_document: bytes | None = None
    source_evidence_hash: str | None = None
    consumer_study_id: str | None = None

    @property
    def key(self):
        return (self.model_family_id, self.model_configuration_id, self.model_package_hash,
                self.functional_id, self.horizon, self.input_case_id)

    def manifest(self):
        for value in (self.functional_id, self.model_family_id):
            identifier(value)
        for value in (self.input_case_id, self.model_configuration_id, self.consumer_study_id):
            if value is not None:
                identifier(value)
        _require(_hash(self.model_package_hash) and type(self.horizon) in (int, float)
            and (type(self.horizon) is not int or self.horizon.bit_length() <= 1023)
            and math.isfinite(self.horizon) and self.horizon > 0, "invalid model/horizon")
        _require(type(self.reason) is str and 0 < len(self.reason) <= 256
            and type(self.status) is str and self.status in {"CALIBRATED", "FAILED", "UNAVAILABLE"}, "invalid disposition")
        geometry = _read(self.geometry_document) if self.geometry_document is not None else None
        pointer = _read(self.source_pointer_document) if self.source_pointer_document is not None else None
        if pointer is not None:
            _require(set(pointer) == POINTER_FIELDS and pointer["schema_version"] == "managed-affine-halfspace-calibration-v1",
                "invalid source pointer")
            AffineHalfspaceCalibrationPolicy.from_manifest(pointer["policy"])
            for key in ("source_attempt_id", "source_authorization_id"):
                identifier(pointer[key])
            _require(_hash(pointer["source_artifact_id"]), "invalid source artifact")
            # The shared owner uses an immutable string selector, not an
            # integer revision counter. None selects only the original grant;
            # never stringify another type or silently fall back to None.
            if pointer["source_authorization_version"] is not None:
                identifier(pointer["source_authorization_version"])
        if self.status == "CALIBRATED":
            _require(geometry is not None and pointer is not None and _hash(self.source_evidence_hash)
                and self.consumer_study_id is not None, "complete declared owner binding required")
            _require(set(geometry) == GEOMETRY_FIELDS and geometry["schema_version"] == "affine-halfspace-geometry-v1"
                and _hash(geometry["causal_input_hash"]), "invalid physical geometry")
        else:
            _require(geometry is None and self.source_evidence_hash is None and self.consumer_study_id is None,
                "unavailable/failed slot cannot claim a calibrated region")
            _require((self.status == "FAILED") == (pointer is not None), "failed slot retains its original pointer")
        value = {"schema_version": "propagation-study-calibration-v1", "functional_id": self.functional_id,
            "model_family_id": self.model_family_id, "model_package_hash": self.model_package_hash,
            "horizon": self.horizon, "input_case_id": self.input_case_id,
            "model_configuration_id": self.model_configuration_id, "status": self.status, "reason": self.reason,
            "geometry": geometry, "source_pointer": pointer, "source_evidence_hash": self.source_evidence_hash,
            "consumer_study_id": self.consumer_study_id,
            "geometry_hash": content_hash(geometry) if geometry is not None else None}
        return {**value, "binding_hash": content_hash(value)}

    @classmethod
    def from_manifest(cls, value):
        _bounded(value)
        _require(type(value) is dict and set(value) == MANIFEST_FIELDS
            and value["schema_version"] == "propagation-study-calibration-v1", "complete table entry required")
        result = cls(**{key: value[key] for key in ("functional_id", "model_family_id", "model_package_hash",
            "horizon", "input_case_id", "model_configuration_id", "status", "reason", "source_evidence_hash",
            "consumer_study_id")}, geometry_document=_encode(value["geometry"]) if value["geometry"] is not None else None,
            source_pointer_document=_encode(value["source_pointer"]) if value["source_pointer"] is not None else None)
        _require(content_hash(result.manifest()) == content_hash(value), "table entry hashes differ")
        return result


def require_consumer_support(spec, cell):
    """Validate declarations pre-store; hashes never grant source authority."""
    from infrastructure.research_store import ResearchError
    try:
        declaration = spec.get("propagation_design", {})
        _require(type(declaration) is dict, "invalid design declaration")
        axes = declaration.get("axis_manifest", {})
        _require(type(axes) is dict, "invalid axis declaration")
        functionals = axes.get("functionals", [])
        _require(type(functionals) is list and len(functionals) <= 64
            and all(type(f) is dict for f in functionals), "invalid bounded functional axis")
        calibrated = any(f.get("functional_id") == cell.get("functional_id")
            and f.get("target_probability") is not None for f in functionals)
        if "calibration_binding" not in cell and not calibrated:
            return None
        _require(calibrated and "calibration_binding" in cell, "stripped or orphan calibrated binding")
        from .preparation import registered_design
        registered_design(spec)
        _require(sum(c == cell for c in spec["cells"]) == 1, "consumer cell absent or duplicated")
        entry = StudyCalibration.from_manifest(cell["calibration_binding"]).manifest()
        _require(entry["status"] == "CALIBRATED" and entry["consumer_study_id"] == spec["study_id"],
            "complete calibrated consumer required")
        return entry
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("UNQUALIFIED", "calibrated consumer declaration malformed") from exc
