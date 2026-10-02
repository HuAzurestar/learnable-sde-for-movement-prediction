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
    return combine_visibility(labels)


def study_visibility(manifest, spec):
    """Include declared source packages even before admission or worker launch."""
    labels = [cell.get("visibility", "restricted") for cell in spec["cells"]]
    settings = spec.get("admission")
    if settings:
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
