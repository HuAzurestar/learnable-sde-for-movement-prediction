"""Owner-side saved-resource bookkeeping, independent of the Paper producer.

No native observer, model/reference replay or statistical score calculation.
The frozen source bundle remains the authority; a self-rehashed summary is not.
"""
from .research_dimensions import comparison_dimensions
from infrastructure.research_store import ResearchError, encode


def resource_work(rows):
    observed_rows = attempts = 0
    for row in rows:
        if "resource_usage" not in row:
            continue
        usage = row["resource_usage"]
        if type(usage) is not dict or type(usage.get("attempts")) is not list:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "resource history allocation")
        observed_rows += 1
        attempts += len(usage["attempts"])
        if attempts > 100_000:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "resource attempt allocation")
    return (64*observed_rows + attempts*(128 + 8*attempts.bit_length()),
            64*observed_rows + 16*attempts)


def _summary(cells):
    legacy = sum("resource_usage" not in c for c in cells)
    unattempted = sum("resource_usage" in c and not c["resource_usage"]["attempts"] for c in cells)
    entries = []
    for cell in cells:
        if "resource_usage" not in cell:
            continue
        sources = cell["resource_usage"]["attempts"]
        ids = [a["attempt_id"] for a in sources]
        if len(set(ids)) != len(ids) or ids != [a["attempt_id"] for a in cell["history"]]:
            raise ResearchError("CONTRACT_MISMATCH", "resource summary omits original attempts")
        entries.extend(sources)
    recorded = [a for a in entries if a["status"] == "RECORDED"]
    unavailable = sorted(a["attempt_id"] for a in entries if a["status"] != "RECORDED")
    memory_unknown = sorted(a["attempt_id"] for a in recorded if a["packet"]["memory"]["status"] != "MEASURED")
    peaks = [a["packet"]["memory"]["peak_resident_bytes"] for a in recorded
             if a["packet"]["memory"]["status"] == "MEASURED"]
    wall = sum(a["packet"]["duration"]["wall_ns"] for a in recorded) if recorded else None
    cpu = sum(a["packet"]["duration"]["cpu_ns"] for a in recorded) if recorded else None
    duration_complete = bool(recorded) and not legacy and not unavailable
    memory_complete = bool(peaks) and not legacy and not unavailable and not memory_unknown
    return {"schema_version": "pirc27-resource-summary-v1", "scope": "all-cell-attempts-including-failures",
        "duration_scope": "worker-validation-method-reference-qualification-pipeline", "duration_unit": "ns",
        "cpu_scope": "current-process-all-threads", "memory_scope": "current-process-lifetime-through-observation",
        "memory_unit": "bytes", "peak_aggregation": "max-of-observed-self-peaks;not-additive;not-phase-delta;not-tree-peak",
        "observed_attempt_count": len(recorded), "unattempted_cell_count": unattempted,
        "legacy_unavailable_cell_count": legacy, "unavailable_attempt_ids": unavailable,
        "memory_unavailable_attempt_ids": sorted(unavailable+memory_unknown),
        "observed_pipeline_wall_ns": wall, "observed_pipeline_cpu_ns": cpu,
        "pipeline_wall_ns": wall if duration_complete else None,
        "pipeline_cpu_ns": cpu if duration_complete else None,
        "duration_complete": duration_complete, "memory_complete": memory_complete,
        "maximum_observed_self_peak_bytes": max(peaks) if peaks else None,
        "source_artifact_ids": sorted(a["artifact"]["artifact_id"] for a in recorded),
        "source_publication_event_hashes": sorted(a["publication_event"]["hash"] for a in entries
                                                 if a["publication_event"] is not None)}


def validate_resource_summaries(aggregate, *, enabled=None):
    try:
        arms = aggregate["arms"]
        present = any("resource_usage_summary" in arm for arm in arms)
        if enabled is None:
            enabled = present
        if type(enabled) is not bool:
            raise ResearchError("CONTRACT_MISMATCH", "resource opt-in must be a boolean")
        if not enabled:
            if present:
                raise ResearchError("CONTRACT_MISMATCH", "resource summaries were not requested")
            return False
        cells = aggregate["cell_dispositions"]
        resource_work(cells)
        def key(row):
            return row["arm_id"], encode(comparison_dimensions(row))
        grouped = {}
        for cell in cells:
            grouped.setdefault(key(cell), []).append(cell)
        keys = [key(arm) for arm in arms]
        if len(set(keys)) != len(keys) or set(keys) != set(grouped):
            raise ResearchError("CONTRACT_MISMATCH", "resource summary stratum inventory changed")
        for arm in arms:
            sources = grouped[key(arm)]
            expected = _summary(sources) if any("resource_usage" in c for c in sources) else None
            if (expected is None and "resource_usage_summary" in arm) or (
                    expected is not None and encode(arm.get("resource_usage_summary")) != encode(expected)):
                raise ResearchError("CONTRACT_MISMATCH", "resource summary differs from frozen all-attempt sources")
        return present
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "malformed resource summary source") from exc
