"""Conservative source lineage shared by publication and disclosure boundaries.

Resolvers may be lock-free store internals when the caller already holds the
store lock. This module never reads artifact bytes or changes authority.
"""

from infrastructure.research_store import ResearchError, digest


def combine_visibility(values):
    labels = set(values)
    if not labels or labels - {"synthetic", "public"}:
        return "restricted"
    return "public" if "public" in labels else "synthetic"


def upstream_metadata_visibility(snapshot, catalog):
    # Metadata-only licensing is not public disclosure permission. The receipt
    # retains the whole definition/catalog, including unrelated input metadata;
    # every explicit narrower input label must remain conservative too.
    if (not isinstance(snapshot, dict) or not isinstance(catalog, dict)
            or not isinstance(snapshot.get("inputs"), list) or not isinstance(catalog.get("entries"), list)
            or any(not isinstance(record, dict) for record in snapshot["inputs"])
            or any(not isinstance(entry, dict) or not isinstance(entry.get("input"), dict)
                   for entry in catalog["entries"])):
        raise ResearchError("CONTRACT_MISMATCH", "source upstream metadata lineage malformed")
    snapshot_label = snapshot.get("visibility", "restricted")
    catalog_label = catalog.get("visibility", "restricted")
    labels = [snapshot_label, catalog_label]
    labels.extend(record.get("visibility", snapshot_label) for record in snapshot.get("inputs", []))
    labels.extend(entry.get("input", {}).get("visibility", catalog_label) for entry in catalog.get("entries", []))
    return combine_visibility([label if isinstance(label, str) else "restricted" for label in labels])


def admission_visibility(manifest, receipt):
    documents = receipt.get("documents", {})
    labels = [receipt.get("cell", {}).get("visibility", "restricted")]
    for name in ("package", "frozen_model"):
        if name in documents:
            labels.append(documents[name].get("visibility", "restricted"))
    for name in ("qualification_evidence", "model_qualification_evidence"):
        for item in documents.get(name, []):
            labels.append(item["artifact"].get("visibility", "restricted"))
            labels.append(manifest("artifact-" + item["artifact"]["artifact_id"])["visibility"])
    if "upstream_snapshot" in documents:
        upstream = documents["upstream_snapshot"]
        labels.append(upstream_metadata_visibility(upstream["snapshot"], upstream["acceptance_catalog"]))
    return combine_visibility(labels)


def study_visibility(manifest, spec):
    """Include declared source packages even before admission or worker launch."""
    labels = [cell.get("visibility", "restricted") for cell in spec["cells"]]
    settings = spec.get("admission")
    if settings:
        if "upstream_snapshot_hash" in settings or "upstream_acceptance_hash" in settings:
            snapshot_hash, catalog_hash = settings.get("upstream_snapshot_hash"), settings.get("upstream_acceptance_hash")
            if not snapshot_hash or not catalog_hash:
                labels.append("restricted")
            else:
                snapshot = manifest("upstream-snapshot-" + snapshot_hash)
                catalog = manifest("upstream-acceptance-" + catalog_hash)
                if digest(snapshot) != snapshot_hash or digest(catalog) != catalog_hash:
                    raise ResearchError("CONTRACT_MISMATCH", "source upstream metadata binding differs")
                labels.append(upstream_metadata_visibility(snapshot, catalog))
        reference = settings.get("package_hash")
        seen = set()
        while reference:
            if reference in seen or len(seen) >= 32:
                raise ResearchError("CONTRACT_MISMATCH", "source lineage is cyclic or too deep")
            seen.add(reference)
            package = manifest("package-" + reference)
            if digest(package) != reference:
                raise ResearchError("CONTRACT_MISMATCH", "source package binding differs")
            labels.append(package.get("visibility", "restricted"))
            if package.get("qualification_hash"):
                report = manifest("qualification-" + package["qualification_hash"])
                if digest(report) != package["qualification_hash"]:
                    raise ResearchError("CONTRACT_MISMATCH", "source qualification binding differs")
                for check in report.get("checks", []):
                    labels.append(manifest("artifact-" + check["artifact_id"])["visibility"])
            reference = package.get("model_hash") if package.get("requires_frozen_model") else None
    return combine_visibility(labels)
