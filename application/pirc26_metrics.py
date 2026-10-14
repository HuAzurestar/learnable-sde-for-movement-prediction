"""Pure consumer bindings, not another adjudication or horizon-weight policy.

The target definition is position-only Euclidean Energy U (ordered non-diagonal
pairs, all 2..256 paths), at the terminal elapsed horizon, mean across distinct
registered origins. Intermediate solver-grid points have no aggregate weight.
Horizon matching uses relative tolerance 1e-12, zero absolute tolerance, to
account for subtraction of ordinary timestamp floats; ambiguous grids refuse.
Cross-stratum weighting/seed aggregation remain in the existing adjudicator.
"""

import math

from application.research_dimensions import comparison_dimensions
from infrastructure.research_store import ResearchError, digest

TARGET_DEFINITION = "pirc26-position-energy-u-exact-target-horizon-mean-origins-v1"
FIXTURE_DEFINITION = "pirc26-position-energy-u-exact-noninitial-grid-origin-mean-v1"
PRIMARY = {"name": "energy_score", "definition": TARGET_DEFINITION, "unit": "m", "direction": "minimize"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def metric_binding(job, receipt):
    """No data/engine/store reads; consume only the owner's frozen metadata."""
    origins = job["origins"]
    identities = set()
    for recipe in origins:
        identity = (recipe["segment_id"], recipe["origin_index"])
        grid = recipe["time_grid"]
        if (identity in identities or len(grid) < 2 or any(not _number(t) for t in grid)
                or any(last <= first for first, last in zip(grid, grid[1:]))
                or type(recipe["sample_count"]) is not int or not 2 <= recipe["sample_count"] <= 256):
            raise ResearchError("CONTRACT_MISMATCH", "distinct origins, increasing grids and exact-U path counts required")
        identities.add(identity)
    documents, spec, cell = receipt["documents"], receipt["spec"], receipt["cell"]
    block = next(b for b in documents["protocol"]["blocks"] if b["block_id"] == cell["block_id"])
    protected = receipt["mode"] == "formal" or block["split_role"] in {"test", "final-eval"}
    plan = spec.get("comparison_plan", {})
    policy = plan.get("adjudication_spec")
    pilot_diagnostic = (receipt["mode"] == "pilot" and not protected and policy is None
                        and "adjudication_hash" not in plan)
    if policy is None and "adjudication_hash" not in plan and receipt["mode"] == "fixture" and not protected:
        # Explicit engineering-only legacy diagnostic, never a formal default.
        definition, indices = FIXTURE_DEFINITION, [list(range(1, len(r["time_grid"]))) for r in origins]
        horizon, policy_hash = None, None
    else:
        if not pilot_diagnostic and (type(policy) is not dict or policy.get("schema_version") != "pirc25-adjudication-spec-v1"
                or plan.get("adjudication_hash") != digest(policy)):
            raise ResearchError("UNQUALIFIED", "NEEDS_PREREGISTRATION: bound AdjudicationSpec required before source reads")
        if not pilot_diagnostic and policy.get("primary_metric") != PRIMARY:
            raise ResearchError("OBJECTIVE_INCOMPATIBLE", "unsupported frozen primary metric definition/unit/direction")
        if protected:
            prereg = documents.get("preregistration", {})
            if (prereg.get("adjudication_spec") != policy or prereg.get("primary_metrics") != [PRIMARY["name"]]
                    or plan.get("preregistration_hash") != digest(prereg)
                    or documents["protocol"].get("preregistration_hash") != digest(prereg)):
                raise ResearchError("UNQUALIFIED", "primary metric differs from the owner's pre-read preregistration")
        dimensions = comparison_dimensions(cell)
        horizon = dimensions.get("horizon")
        if not _number(horizon) or horizon <= 0 or "horizons" in dimensions:
            raise ResearchError("CONTRACT_MISMATCH", "one positive elapsed horizon in seconds required per cell")
        if not pilot_diagnostic:
            contrasts = policy.get("contrasts")
            if type(contrasts) is not list or not 0 < len(contrasts) <= 256:
                raise ResearchError("CONTRACT_MISMATCH", "registered comparison family required")
            participating = [c for c in contrasts if type(c) is dict and cell["arm_id"] in (c.get("reference"), c.get("candidate"))]
            if not participating or any(type(c.get("stratum_weights")) is not list
                    or not 0 < len(c["stratum_weights"]) <= 256 or not any(type(w) is dict
                    and digest(w.get("comparison_dimensions")) == digest(dimensions)
                    for w in c["stratum_weights"]) for c in participating):
                raise ResearchError("CONTRACT_MISMATCH", "cell is absent from a participating frozen weighted stratum")
        indices = []
        for recipe in origins:
            grid = recipe["time_grid"]
            matches = [i for i in range(1, len(grid)) if math.isclose(grid[i] - grid[0], horizon, rel_tol=1e-12, abs_tol=0)]
            if matches != [len(grid) - 1]:
                raise ResearchError("CONTRACT_MISMATCH", "terminal solver time must uniquely match the registered elapsed horizon")
            if "time_grid" in dimensions and dimensions["time_grid"] != grid:
                raise ResearchError("CONTRACT_MISMATCH", "cell time grid differs from its immutable origin recipe")
            indices.append(matches)
        definition, policy_hash = TARGET_DEFINITION, None if pilot_diagnostic else digest(policy)
    body = {"schema_version": "pirc26-cell-metric-binding-v1", "metric_name": "energy_score", "definition": definition,
        "unit": "m", "target_horizon_seconds": horizon, "selected_grid_indices": indices,
        "sample_counts": [r["sample_count"] for r in origins],
        "origin_recipe_hashes": [digest(r) for r in origins], "adjudication_hash": policy_hash,
        "cross_stratum_weights_applied": False}
    return {**body, "binding_hash": digest(body)}


def aggregate_metric(forecasts, binding):
    """Aggregate only complete, exact estimator rows covered by this binding."""
    if (not forecasts or len(forecasts) != len(binding["selected_grid_indices"])
            or len(forecasts) != len(binding["sample_counts"])):
        raise ResearchError("CONTRACT_MISMATCH", "metric origin population differs")
    scores = []
    for forecast, indices, expected_count in zip(forecasts, binding["selected_grid_indices"], binding["sample_counts"]):
        evaluation = forecast["evaluation"]
        if (evaluation["status"] != "SUCCEEDED" or evaluation["failed_paths"]
                or evaluation["requested_paths"] != expected_count):
            raise ResearchError("NONFINITE", "partial forecast cannot enter a primary metric")
        for index in indices:
            row = evaluation["rows"][index]
            value, count = row["metrics"]["energy_score"], row["sample_count"]
            if (row["position_units"] != "m" or row["estimator"] != {
                    "estimator_id": "energy-u-exact-v1", "pair_count": count * (count - 1)}
                    or not _number(value) or type(count) is not int or count != expected_count or not 2 <= count <= 256):
                raise ResearchError("CONTRACT_MISMATCH", "metric row differs from its exact position Energy definition")
            scores.append(value)
    return {"energy_score": math.fsum(scores) / len(scores)}
