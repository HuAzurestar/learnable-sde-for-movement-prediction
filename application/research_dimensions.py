"""Comparison strata are not budget arms or independent sampling units."""

from infrastructure.research_store import ResearchError, encode

AXES = ("horizon", "horizons", "region", "scenario", "initialization",
        "prediction_origin", "context_profile", "time_grid")


def comparison_dimensions(cell):
    dimensions = cell.get("comparison_dimensions", {})
    if not isinstance(dimensions, dict) or any(not isinstance(k, str) or not k for k in dimensions):
        raise ResearchError("CONTRACT_MISMATCH", "comparison dimensions must be a named mapping")
    dimensions = dict(dimensions)
    if set(dimensions) & {"arm_id", "block_id", "seed"}:
        raise ResearchError("CONTRACT_MISMATCH", "sampling identity cannot be a comparison dimension")
    for key in AXES:
        if key in cell:
            if key in dimensions and encode(dimensions[key]) != encode(cell[key]):
                raise ResearchError("CONTRACT_MISMATCH", "conflicting comparison dimension")
            dimensions[key] = cell[key]
    encode(dimensions)
    return dimensions
