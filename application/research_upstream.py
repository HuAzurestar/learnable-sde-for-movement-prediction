"""Fresh selected-cell metadata takeover, independent of data permission.

The registered catalog is an operator attestation, not a new acceptance
decision. No trajectory/provider reads, grants or cached ready decisions.
"""
from pathlib import Path

from infrastructure.research_store import ResearchError, digest
from .research_preregistration import hash_reference


def validate_acceptance_catalog(value):
    if (not isinstance(value, dict) or value.get("schema_version") != "pirc25-upstream-acceptance-v1"
            or not isinstance(value.get("source"), str) or not value["source"].strip()
            or not isinstance(value.get("entries"), list)
            or any(not isinstance(entry, dict) or not isinstance(entry.get("input"), dict)
                   for entry in value["entries"])
            or not isinstance(value.get("source_evidence"), dict)
            or digest(value["source_evidence"]) != value.get("source_evidence_hash")
            or value["source_evidence"].get("entries") != value["entries"]):
        raise ResearchError("UNACCEPTED_VERSION", "frozen operator acceptance catalog/evidence required")


def register_acceptance_catalog(store, value, expected_hash):
    validate_acceptance_catalog(value)
    if not hash_reference(expected_hash) or digest(value) != expected_hash:
        raise ResearchError("IDENTITY_MISMATCH", "acceptance catalog content binding differs")
    return store.publish("upstream-acceptance-" + expected_hash, value)


def study_binding(snapshot_hash, catalog_hash, study):
    return {"snapshot_hash": snapshot_hash, "acceptance_catalog_hash": catalog_hash,
            "pirc22_cutover": study.get("pirc22_cutover")}


def _document(store, kind, reference):
    if not hash_reference(reference):
        raise ResearchError("MISSING_ARTIFACT", "frozen upstream " + kind + " reference required")
    try:
        value = store.manifest("upstream-" + kind + "-" + reference)
    except ResearchError as exc:
        code = "MISSING_ARTIFACT" if exc.code == "MISSING_INPUT" else "IDENTITY_MISMATCH"
        raise ResearchError(code, "registered upstream " + kind + " unavailable") from exc
    if digest(value) != reference:
        raise ResearchError("IDENTITY_MISMATCH", "upstream " + kind + " content binding differs")
    return value


def prepare_upstream(store, spec, cell, package, prereg=None, *, attempt_id, run_id):
    try:
        return _prepare_upstream(store, spec, cell, package, prereg, attempt_id=attempt_id, run_id=run_id)
    except ResearchError as exc:
        settings = spec["admission"]
        store.append("UPSTREAM_REFUSED", {"study_id": spec["study_id"], "cell_hash": digest(cell),
            "attempt_id": attempt_id, "run_id": run_id,
            "snapshot_hash": settings.get("upstream_snapshot_hash"),
            "acceptance_catalog_hash": settings.get("upstream_acceptance_hash"), "error_code": exc.code})
        raise


def _prepare_upstream(store, spec, cell, package, prereg=None, *, attempt_id, run_id):
    from experiments.pirc25.snapshot import UpstreamSnapshot

    settings = spec["admission"]
    snapshot_hash, catalog_hash = settings.get("upstream_snapshot_hash"), settings.get("upstream_acceptance_hash")
    definition = _document(store, "snapshot", snapshot_hash)
    catalog = _document(store, "acceptance", catalog_hash)
    validate_acceptance_catalog(catalog)
    snapshot = UpstreamSnapshot(definition)
    if (snapshot.snapshot_hash != snapshot_hash or package.get("upstream_snapshot_hash") != snapshot_hash
            or package.get("upstream_acceptance_hash") != catalog_hash):
        raise ResearchError("IDENTITY_MISMATCH", "package differs from frozen upstream snapshot/catalog")
    studies = {study["study_id"]: study for study in definition["studies"]}
    study = studies.get(spec["study_id"])
    if study is None:
        raise ResearchError("MISSING_ARTIFACT", "study absent from frozen upstream snapshot")
    frozen_cells = [item for item in definition["cells"] if item.get("study_id") == spec["study_id"]]
    actual_cells = {digest(item) for item in spec["cells"]}
    if (len(frozen_cells) != len(actual_cells)
            or {item["cell_id"] for item in frozen_cells} != actual_cells
            or digest(cell) not in actual_cells):
        raise ResearchError("IDENTITY_MISMATCH", "complete execution matrix differs from frozen upstream cells")
    if prereg is not None:
        expected = study_binding(snapshot_hash, catalog_hash, study)
        bindings = prereg.get("upstream_bindings")
        if not isinstance(bindings, dict) or bindings.get(spec["study_id"]) != expected:
            raise ResearchError("IDENTITY_MISMATCH", "upstream snapshot/cutover differs from pre-read frozen plan")
        events = store.events()
        objects = ("upstream-snapshot-" + snapshot_hash, "upstream-acceptance-" + catalog_hash,
                   "preregistration-" + digest(prereg))
        positions = {event["payload"]["object_id"]: event["sequence"] for event in events
                     if event["event_kind"] == "MANIFEST" and event["payload"].get("object_id") in objects}
        if (any(name not in positions for name in objects)
                or max(positions[name] for name in objects[:2]) >= positions[objects[2]]):
            raise ResearchError("IDENTITY_MISMATCH", "upstream records were not frozen before preregistration")
    root = settings.get("upstream_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ResearchError("MISSING_ARTIFACT", "explicit lexical upstream metadata root required")
    # Only selected dependencies are opened. An unrelated absent/rejected
    # input is not an excuse to relabel or block this cell.
    validation = snapshot.resolve(root=root, accepted_versions=catalog["entries"], cell_id=digest(cell))
    validation["acceptance_catalog_hash"] = catalog_hash
    validation["consumer"] = {"attempt_id": attempt_id, "run_id": run_id,
                              "study_id": spec["study_id"], "cell_hash": digest(cell), "spec_hash": digest(spec)}
    validation_hash = digest(validation)
    store.publish("upstream-validation-" + validation_hash, validation)
    selected = validation["cells"][0]
    validation_event = store.append("UPSTREAM_VALIDATION", {"study_id": spec["study_id"], "cell_hash": digest(cell),
        "attempt_id": attempt_id, "run_id": run_id,
        "snapshot_hash": snapshot_hash, "acceptance_catalog_hash": catalog_hash,
        "validation_hash": validation_hash, "status": selected["status"],
        "rejected_inputs": selected["rejected_inputs"]})
    if selected["status"] != "ready":
        raise ResearchError(selected["rejected_inputs"][0]["code"], "selected frozen upstream cell rejected")
    # Attach observed immutable event objects, never manufactured timestamps or
    # sequence defaults. The Paper consumer verifies their cross-object order
    # without importing a provider or reopening protected source data.
    objects = {"snapshot": "upstream-snapshot-" + snapshot_hash,
               "acceptance_catalog": "upstream-acceptance-" + catalog_hash,
               "validation": "upstream-validation-" + validation_hash}
    with store._read_transaction():
        for name, expected in (("snapshot", definition), ("acceptance_catalog", catalog), ("validation", validation)):
            if store.manifest(objects[name]) != expected:
                raise ResearchError("IDENTITY_MISMATCH", "upstream evidence changed before admission")
        events = {event["payload"]["object_id"]: event for event in store.events()
                  if event["event_kind"] == "MANIFEST" and event["payload"].get("object_id") in objects.values()}
        if any(object_id not in events for object_id in objects.values()):
            raise ResearchError("MISSING_ARTIFACT", "upstream publication event missing")
        publications = {name: events[object_id] for name, object_id in objects.items()}
    return {"snapshot": definition, "acceptance_catalog": catalog,
            "validation": validation, "validation_hash": validation_hash,
            "publication_events": publications, "validation_event": validation_event}
