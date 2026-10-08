"""Method consumers of settled geometry, separate from method qualification.

Only bounded saved proof is examined here. All calibration/reference searches
remain in their originally charged source worker. Independent export support
is deliberately required before any calibrated study can disclose a bundle.
"""

from domain.probability_calibration import halfspace_geometry
from experiments.pirc27.calibration_bindings import require_consumer_support
from infrastructure.research_store import ResearchError, digest, encode

from .probability_calibration_admission import _bounded, prepare_probability_calibration
from .propagation_execution import validate_propagation_cell


def _require(condition, detail):
    if not condition:
        raise ResearchError("UNQUALIFIED", "calibrated method consumer: " + detail)


def prepare_calibrated_consumer(store, spec, cell, *, preregistration=None):
    entry = require_consumer_support(spec, cell)
    if entry is None:
        return None
    if preregistration is not None:
        from experiments.pirc27.design import MAX_MANIFEST_BYTES
        table = spec["propagation_design"]["axis_manifest"]["calibrations"]
        declared = preregistration.get("probability_calibration_bindings")
        _bounded(declared, nodes=2000000, depth_limit=30, string_limit=16384)
        _require(type(declared) is list and 0 < len(declared) <= 10000
            and len(encode(declared)) <= MAX_MANIFEST_BYTES, "bounded preregistration table required")
        _require(digest(declared) == digest(table)
            and preregistration.get("probability_calibration_bindings_hash") == digest(table),
            "complete geometry table absent from frozen preregistration")
    # Actual current permissions, original native settlement and canonical
    # saved source bytes, not the table author's declaration, own this check.
    evidence = prepare_probability_calibration(store, entry["source_pointer"],
        consumer_study_id=spec["study_id"])
    _require(evidence["evidence_hash"] == entry["source_evidence_hash"]
        and evidence["geometry"] == entry["geometry"]
        and evidence["geometry_hash"] == entry["geometry_hash"],
        "fresh settled owner differs from frozen region")
    package, request, _, _ = validate_propagation_cell(spec, cell)
    # The source causal identity identifies the calibrated mathematical law.
    # It does NOT grant a real-prefix identity or target-data access; the target
    # input binding and protocol are independently checked by AdmissionGate.
    geometry = halfspace_geometry(package, request,
        causal_input_hash=evidence["geometry"]["causal_input_hash"])
    _require(geometry == evidence["geometry"], "target law/time/physical event differs")
    return evidence


def validate_calibrated_result(store, receipt, spec, cell, result):
    declared = require_consumer_support(spec, cell)
    saved = receipt.get("documents", {}).get("probability_calibration")
    if declared is None:
        _require(saved is None, "orphan calibration proof")
        return
    _require(saved is not None, "result lacks settled geometry admission")
    _bounded(saved, nodes=25000, depth_limit=30, string_limit=16384)
    _require(type(saved) is dict and len(encode(saved)) <= 2*1024*1024
        and saved.get("evidence_hash") == digest({k: v for k, v in saved.items() if k != "evidence_hash"}),
        "saved calibration proof content binding differs")
    fresh = prepare_calibrated_consumer(store, spec, cell,
        preregistration=receipt.get("documents", {}).get("preregistration"))
    _require(digest(saved) == digest(fresh), "saved calibration proof differs from current owner")
    package, request, _, _ = validate_propagation_cell(spec, cell)
    forecast = result.get("forecast", {})
    _require(type(forecast) is dict and forecast.get("model_package_hash") == package.package_hash
        and forecast.get("request_hash") == request.request_hash
        and type(forecast.get("horizons")) is list and len(forecast["horizons"]) == 1
        and type(forecast["horizons"][0]) in (int, float)
        and digest(forecast["horizons"]) == digest(list(request.horizons)),
        "current output model/request/horizon differs")
    return fresh


def require_calibration_export_support(spec, *, cells=()):
    axes = spec.get("propagation_design", {}).get("axis_manifest", {})
    if (axes.get("calibrations") or any(f.get("target_probability") is not None
            for f in axes.get("functionals", []) if type(f) is dict)
            or any("calibration_binding" in c for c in spec.get("cells", []))
            or any("probability_calibration" in (c.get("admission") or {}).get("documents", {})
                for c in cells)):
        raise ResearchError("UNQUALIFIED", "independent calibration export support is not yet available")
