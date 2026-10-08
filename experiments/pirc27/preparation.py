"""Reload compiler output and bind an existing runtime without admitting work.

Only bounded synthetic design and connection metadata are read. No store is
opened, no ledger is changed, and no trajectory or result is read. The normal
shared registration/admission/budget path remains mandatory after preparation.
"""

from dataclasses import fields
import hashlib
import json
from pathlib import Path

from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash, _hash
from domain.mixture import MixtureSettings
from domain.nonlinear_dynamics import FrozenNonlinearPackage
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError, identifier

from .design import (MAX_MANIFEST_BYTES, PropagationStudyDesign, StudyArm,
    StudyFunctional, StudyMethod, StudyModel, freeze_design)


MAX_CONNECTION_BYTES = 64 * 1024
CONNECTION_FIELDS = frozenset({"schema_version", "runtime_root", "store_id", "contract_version",
    "participants", "data_source_root", "data_binding_status", "pirc38_runtime_baseline",
    "initialized_using_code_sha"})


def _require(condition, detail):
    if not condition:
        raise ResearchError("CONTRACT_MISMATCH", detail)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate preparation JSON field")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("nonfinite JSON")


def _read_document(path, maximum_bytes, *, expected_hash=None):
    if expected_hash is not None:
        _require(_hash(expected_hash), "explicit valid preparation content hash required")
    path = Path(path).absolute()
    try:
        with opened_regular_file(path.parent, path, maximum_bytes=maximum_bytes) as (stream, size, verify):
            raw = stream.read(size + 1)
            verify()
            _require(len(raw) == size, "preparation metadata size changed")
        raw_hash = hashlib.sha256(raw).hexdigest()
        if expected_hash is not None:
            _require(raw_hash == expected_hash, "runtime configuration content hash differs")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant)
        _require(type(document) is dict, "preparation metadata must be an object")
        return document, raw_hash
    except OSError as exc:
        raise ResearchError("STORE_MISSING", "required preparation metadata file is unavailable") from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("CONTRACT_MISMATCH", "invalid bounded preparation JSON") from exc


def _keys(value, required, optional=()):
    _require(type(value) is dict and set(required) <= set(value) <= set(required) | set(optional),
        "unexpected frozen preparation fields")


def _tuple(value, *, maximum=64, exact=None):
    _require(type(value) is list and len(value) <= maximum and (exact is None or len(value) == exact),
        "invalid bounded frozen preparation axis")
    return tuple(value)


def reload_design(path, *, expected_hash):
    """Regenerate every request/resource/refusal row, not just check a digest."""
    _require(_hash(expected_hash), "explicit frozen design hash required")
    document, _ = _read_document(path, MAX_MANIFEST_BYTES)
    try:
        _require(content_hash(document) == expected_hash, "frozen design content hash differs")
        _keys(document, {"schema_version", "study_id", "experiment_id", "comparison_family", "code_hash",
            "protocol_hash", "data_hash", "feature_hash", "selection_hash", "stopping_rule", "primary_metrics",
            "qualification", "expected_cells", "axis_manifest", "arms", "matrix"})
        _require(document["schema_version"] == "propagation-study-manifest-v1"
            and document["qualification"] == "preparation-only-not-scientific", "unsupported frozen design schema")
        from experiments.pirc25.affine import code_hash
        _require(document["code_hash"] == code_hash(), "frozen design current source hash differs")
        axes = document["axis_manifest"]
        _keys(axes, {"models", "methods", "functionals", "horizons", "seeds", "origin", "history_cutoff",
            "state_names", "time_unit", "coordinate_system"})
        _require(axes["state_names"] == ["x", "y", "vx", "vy"] and axes["time_unit"] == "s"
            and axes["coordinate_system"] == "local-cartesian", "only two-dimensional four-state SI motion is supported")
        models = []
        for value in _tuple(axes["models"]):
            _keys(value, {"family_id", "package", "initial_mean", "initial_covariance"}, {"configuration_id"})
            package = value["package"]
            _require(type(package) is dict, "explicit frozen synthetic package required")
            package_type = {"frozen-affine-dynamics-v1": FrozenDynamicsPackage,
                "frozen-tanh-dynamics-v1": FrozenNonlinearPackage}.get(package.get("schema_version"))
            _require(package_type is not None, "unsupported frozen synthetic package")
            models.append(StudyModel(value["family_id"],
                package_type.from_manifest(package, expected_hash=content_hash(package)),
                _tuple(value["initial_mean"], exact=4),
                tuple(_tuple(row, exact=4) for row in _tuple(value["initial_covariance"], exact=4)),
                value.get("configuration_id")))
        methods = []
        for value in _tuple(axes["methods"]):
            _keys(value, {f.name for f in fields(StudyMethod)} - {"configuration_id"}, {"configuration_id"})
            value = {**value, "level_samples": _tuple(value["level_samples"], maximum=9),
                "proposal": _tuple(value["proposal"], exact=2)}
            if value["mixture_settings"] is not None:
                settings = value["mixture_settings"]
                _keys(settings, {f.name for f in fields(MixtureSettings)})
                value["mixture_settings"] = MixtureSettings(**{**settings,
                    "state_scales": _tuple(settings["state_scales"], exact=4)})
            methods.append(StudyMethod(**value))
        functionals = []
        for value in _tuple(axes["functionals"]):
            _keys(value, {f.name for f in fields(StudyFunctional)})
            functionals.append(StudyFunctional(**{**value, "normal": _tuple(value["normal"], exact=4)}))
        arms = []
        for value in _tuple(document["arms"], maximum=10000):
            _keys(value, {f.name for f in fields(StudyArm)} | {"budget_seconds"})
            _require(type(value["budget_seconds"]) is int and value["budget_seconds"] == 86400,
                "original cumulative arm budget differs")
            arms.append(StudyArm(**{k: v for k, v in value.items() if k != "budget_seconds"}))
        count = document["expected_cells"]
        _require(type(count) is int and 0 < count <= 10000 and type(document["matrix"]) is list
            and len(document["matrix"]) == count, "complete bounded registered matrix required")
        design = PropagationStudyDesign(document["study_id"], document["experiment_id"], document["comparison_family"],
            tuple(models), tuple(methods), tuple(functionals), _tuple(axes["horizons"]), _tuple(axes["seeds"]),
            tuple(arms), *(document[k] for k in ("protocol_hash", "data_hash", "feature_hash", "selection_hash")),
            origin=axes["origin"], history_cutoff=axes["history_cutoff"], stopping_rule=document["stopping_rule"],
            primary_metrics=_tuple(document["primary_metrics"]))
        frozen = freeze_design(design)
        _require(frozen.manifest_hash == expected_hash, "frozen design differs from regenerated compiler output")
        return frozen
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("CONTRACT_MISMATCH", "invalid complete four-state frozen design") from exc


def load_connection(path, *, expected_hash):
    """Read identity metadata only; this is not a ledger health/admission check."""
    _require(_hash(expected_hash), "explicit frozen runtime configuration hash required")
    connection, config_hash = _read_document(path, MAX_CONNECTION_BYTES, expected_hash=expected_hash)
    _keys(connection, CONNECTION_FIELDS)
    _require(connection["schema_version"] == "pirc25-local-runtime-config-v1"
        and connection["contract_version"] == "pirc25-contract-v1"
        and connection["data_binding_status"] == "source_location_only_not_study_registration_or_read_authority",
        "connection metadata is not research or data-access authority")
    participants = connection["participants"]
    _require(type(participants) is list and 0 < len(participants) <= 3
        and all(type(v) is str and v in {"PIRC-26", "PIRC-27", "PIRC-28"} for v in participants)
        and len(set(participants)) == len(participants) and "PIRC-27" in participants,
        "shared runtime must explicitly include PIRC-27")
    for key in ("pirc38_runtime_baseline", "initialized_using_code_sha"):
        value = connection[key]
        _require(type(value) is str and len(value) == 40 and all(c in "0123456789abcdef" for c in value),
            "invalid declared runtime source ref")
    for key in ("runtime_root", "data_source_root"):
        _require(type(connection[key]) is str and Path(connection[key]).is_absolute(),
            "connection roots must be explicit absolute paths")
    try:
        root = Path(connection["runtime_root"]).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "invalid shared runtime root") from exc
    _require(not any((p / ".git").exists() for p in (root, *root.parents)), "runtime root must stay outside Git")
    store_id = identifier(connection["store_id"])
    identity, identity_hash = _read_document(root / "pirc25" / "store.json", MAX_CONNECTION_BYTES)
    _require(identity == {"schema_version": "pirc25-contract-v1", "store_id": store_id,
        "runtime_root": str(root)}, "existing shared store identity or root differs")
    return {"root": str(root), "store_id": store_id}, config_hash, identity_hash


def prepare_study(path, *, expected_hash, runtime_config, expected_runtime_config_hash):
    frozen = reload_design(path, expected_hash=expected_hash)
    binding, config_hash, identity_hash = load_connection(runtime_config, expected_hash=expected_runtime_config_hash)
    spec = {**frozen.study_spec(expected_hash=expected_hash), "runtime_binding": binding}
    return {"schema_version": "pirc27-prepared-study-v1", "preparation_status": "NOT_REGISTERED_NOT_ADMITTED",
        "design_hash": frozen.manifest_hash, "runtime_config_hash": config_hash,
        "store_identity_hash": identity_hash, "study_spec_hash": content_hash(spec), "study_spec": spec,
        "motion_space_dimension": 2, "state_dimension": 4,
        "terrain_dependency": "only-terrain-dependent-comparisons-wait-for-PIRC-17"}
