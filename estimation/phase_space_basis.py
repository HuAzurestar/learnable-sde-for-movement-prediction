"""Bounded streaming O1 basis fit, with explicit restart-only semantics.

The affine base, velocity covariance, normalization and basis recipe stay
frozen. Coefficients are committed only after all numerical/cancellation gates.
There is no normal-equation inverse, N-by-N kernel, or implicit ridge repair.
"""

from dataclasses import dataclass
import math
import time

import torch

from infrastructure.research_store import digest
from models.phase_space import ModelContractError, RBFResidualDrift, SplineResidualDrift


@dataclass(frozen=True)
class BasisPlan:
    solver_id: str = "streaming-qr-v1"
    ridge: float = 0.
    curvature_penalty: float = 0.
    condition_number_max: float = 1e6
    max_batch_rows: int = 4096
    identifiability: str = "reference-coded-additive-v1"

    def validate(self):
        if (self.solver_id != "streaming-qr-v1" or self.identifiability not in
                ("reference-coded-additive-v1", "full-rbf-v1")
                or type(self.max_batch_rows) is not int or not 1 <= self.max_batch_rows <= 4096
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in
                       (self.ridge, self.curvature_penalty, self.condition_number_max))
                or self.ridge < 0 or self.curvature_penalty < 0 or self.condition_number_max < 1):
            raise ModelContractError("MODEL_CONTRACT_ERROR: bounded explicit basis plan required")


def _basis_at(x, knots, degree):
    x = x[:, None]
    values = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
    for order in range(1, degree + 1):
        count = len(knots) - order - 1
        left = knots[order:order+count] - knots[:count]
        right = knots[order+1:order+count+1] - knots[1:count+1]
        values = ((x - knots[:count]) / torch.where(left > 0, left, torch.ones_like(left)) * (left > 0) * values[:, :count]
            + (knots[order+1:order+count+1] - x) / torch.where(right > 0, right, torch.ones_like(right)) * (right > 0) * values[:, 1:count+1])
    return values


def spline_curvature_factor(drift):
    """P such that ||P theta||² integrates squared second derivatives in u.

    Two Gauss nodes per positive knot span integrate the quadratic products
    exactly for degree <=3. This is normalized-feature curvature on the frozen
    knot domain, not a claim about clamped tails or physical-coordinate units.
    Active curvature requires C1 continuity; discontinuous derivative jumps are
    rejected rather than omitted from a purported Sobolev roughness penalty.
    """
    if not isinstance(drift, SplineResidualDrift) or drift.degree < 2:
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: curvature requires degree-two/three splines")
    width, rows = len(drift.coefficients), []
    offset = 0
    for knots in drift.knots:
        degree = drift.degree
        unique, counts = torch.unique_consecutive(knots, return_counts=True)
        if (counts[1:-1] > degree - 1).any():
            raise ModelContractError("OBJECTIVE_INCOMPATIBLE: active curvature requires C1 interior knots")
        lower, upper = unique[:-1], unique[1:]
        half, middle = (upper - lower) / 2, (upper + lower) / 2
        x = torch.stack((middle - half / math.sqrt(3), middle + half / math.sqrt(3)), 1).flatten()
        weight = half.repeat_interleave(2).sqrt()
        lower_basis = _basis_at(x, knots, degree - 2)
        derivative_count = len(knots) - degree
        dl = knots[degree-1:degree-1+derivative_count] - knots[:derivative_count]
        dr = knots[degree:degree+derivative_count] - knots[1:derivative_count+1]
        first = (degree-1) * ((dl > 0) / torch.where(dl > 0, dl, torch.ones_like(dl)) * lower_basis[:, :derivative_count]
            - (dr > 0) / torch.where(dr > 0, dr, torch.ones_like(dr)) * lower_basis[:, 1:derivative_count+1])
        count = derivative_count - 1
        dl = knots[degree:degree+count] - knots[:count]
        dr = knots[degree+1:degree+count+1] - knots[1:count+1]
        second = degree * ((dl > 0) / torch.where(dl > 0, dl, torch.ones_like(dl)) * first[:, :count]
            - (dr > 0) / torch.where(dr > 0, dr, torch.ones_like(dr)) * first[:, 1:count+1])
        part = knots.new_zeros((len(x), width))
        part[:, offset:offset+count] = second * weight[:, None]
        rows.append(part)
        offset += count
    factor = torch.cat(rows)
    if not torch.isfinite(factor).all():
        raise ModelContractError("NONFINITE: normalized spline curvature factor")
    return factor


def free_coefficients(drift, identifiability):
    count = len(drift.coefficients)
    if isinstance(drift, RBFResidualDrift):
        if identifiability != "full-rbf-v1":
            raise ModelContractError("OBJECTIVE_INCOMPATIBLE: RBF needs its full fixed basis")
        return list(range(count)), []
    if not isinstance(drift, SplineResidualDrift) or identifiability != "reference-coded-additive-v1":
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: registered additive spline identifiability required")
    block = drift.knots.shape[1] - drift.degree - 1
    # Every component partitions unity. Fix one coefficient in each component
    # after the first to zero; its constant can be represented by the first.
    # This removes exactly F-1 redundant constants without dropping functions.
    constrained = [block * (i + 1) - 1 for i in range(1, len(drift.features))]
    return [i for i in range(count) if i not in constrained], constrained


def fit_basis(model, batches, plan, *, cancellation=None, progress=None):
    from estimation.phase_space import TransitionBatch, local_velocity_nll
    from estimation.phase_space_checkpoint import tensor_identity, training_scope
    plan.validate()
    drift = model.acceleration_model
    if (not isinstance(drift, (RBFResidualDrift, SplineResidualDrift)) or not drift.residual_enabled
            or model.velocity_factor.device.type != "cpu" or not isinstance(batches, (tuple, list))
            or not 1 <= len(batches) <= 256):
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: enabled bounded CPU basis fit required")
    if isinstance(drift, RBFResidualDrift) and plan.curvature_penalty:
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: spline curvature is not an RBF penalty")
    free, constrained = free_coefficients(drift, plan.identifiability)
    chol, info = torch.linalg.cholesky_ex(model.velocity_factor @ model.velocity_factor.T)
    if info.item() != 0:
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: basis O1 requires rank-two velocity covariance")
    identities = []
    for batch in batches:
        if not isinstance(batch, TransitionBatch) or not 0 < len(batch.time) <= plan.max_batch_rows:
            raise ModelContractError("RESOURCE_PLAN_REJECTED: registered basis batch quota")
        batch.validate(model)
        identities.append({"time": tensor_identity(batch.time), "state": tensor_identity(batch.state),
            "next_state": tensor_identity(batch.next_state), "dt": tensor_identity(batch.dt),
            "context": tensor_identity(batch.context.condition), "split_role": batch.split_role,
            "train_binding_hash": batch.train_binding_hash})
    scope = training_scope(model, plan, identities, "O1")
    scope.update(initial_model_hash=model.checkpoint()["sha256"], gradient_route="G0", resume_level="restart-only")
    started, q = time.perf_counter(), len(free)
    r, projected = model.velocity_factor.new_empty((0, q)), model.velocity_factor.new_empty((0, 2))
    count = 0
    def check():
        if cancellation is not None and cancellation():
            raise ModelContractError("INTERRUPTED: restart-only basis fit cancelled before coefficient publication")
    check()
    with torch.no_grad():
        curvature = spline_curvature_factor(drift)[:, free] if plan.curvature_penalty else r.new_empty((0, q))
        for step, batch in enumerate(batches, 1):
            check()
            c = model._inputs(batch.time, batch.state, batch.context)
            basis = drift.basis(drift.selected_features(batch.state, c))[:, free]
            target = (batch.next_state[:, 2:] - batch.state[:, 2:]) / batch.dt[:, None] - drift.affine(batch.time, batch.state, c)
            weight = batch.dt.sqrt()[:, None]
            combined, responses = torch.cat((r, basis * weight)), torch.cat((projected, target * weight))
            if not torch.isfinite(combined).all() or not torch.isfinite(responses).all():
                raise ModelContractError("NONFINITE: weighted basis inputs")
            orthogonal, r = torch.linalg.qr(combined, mode="reduced")
            projected = orthogonal.T @ responses
            if not torch.isfinite(r).all() or not torch.isfinite(projected).all():
                raise ModelContractError("NONFINITE: streaming QR state")
            count += len(batch.time)
            if progress is not None:
                progress({"step": step, "total_steps": len(batches), "observations": count, "solver_id": plan.solver_id})
            check()
        singular = torch.linalg.svdvals(r)
        if len(singular) != q or singular[-1] <= torch.finfo(r.dtype).eps * max(count, q) * singular[0]:
            raise ModelContractError("ILL_CONDITIONED: rank-deficient registered identifiable basis")
        raw_condition = (singular[0] / singular[-1]).item()
        raw_r = r.clone()
        penalties = []
        if plan.ridge:
            penalties.append(math.sqrt(plan.ridge) * torch.eye(q, dtype=r.dtype, device=r.device))
        if plan.curvature_penalty:
            penalties.append(math.sqrt(plan.curvature_penalty) * curvature)
        penalty = torch.cat(penalties) if penalties else r.new_empty((0, q))
        if len(penalty):
            combined, responses = torch.cat((r, penalty)), torch.cat((projected, projected.new_zeros((len(penalty), 2))))
            orthogonal, r = torch.linalg.qr(combined, mode="reduced")
            projected = orthogonal.T @ responses
        regularized_singular = torch.linalg.svdvals(r)
        condition = (regularized_singular[0] / regularized_singular[-1]).item()
        if not math.isfinite(condition) or condition > plan.condition_number_max:
            raise ModelContractError("ILL_CONDITIONED: registered condition threshold exceeded")
        coefficients = torch.linalg.solve_triangular(r, projected, upper=True)
        if not torch.isfinite(coefficients).all():
            raise ModelContractError("NONFINITE: residual basis coefficients")
        degrees = torch.linalg.solve_triangular(r.T, raw_r.T, upper=False).square().sum().item() if len(penalty) else float(q)
        check()
        # Monitor a detached candidate before mutating the caller's coefficients.
        from models.phase_space import PhaseSpaceSDE
        candidate = PhaseSpaceSDE.from_checkpoint(model.checkpoint())
        candidate.acceleration_model.coefficients.zero_()
        candidate.acceleration_model.coefficients[free] = coefficients
        objective = sum(local_velocity_nll(candidate, part).item() * len(part.time) for part in batches) / count
        penalty_values = penalty @ coefficients
        penalty_norm = torch.linalg.solve_triangular(chol, penalty_values.T, upper=False).square().sum().item() if len(penalty) else 0.
        penalized = objective + .5 * penalty_norm / count
        if not math.isfinite(penalized) or not math.isfinite(degrees):
            raise ModelContractError("NONFINITE: basis fit diagnostics")
        checkpoint = candidate.checkpoint()
        check()
        drift.coefficients.copy_(candidate.acceleration_model.coefficients)
    return {"schema_version": "pirc26-basis-fit-v2", "status": "SUCCEEDED", "objective_id": "O1", "gradient_route": "G0",
        "solver_id": plan.solver_id, "objective_hash": digest(scope), "scope": scope, "steps": len(batches),
        "observations": count, "basis_count": len(drift.coefficients), "free_basis_count": q,
        "identifiability": plan.identifiability, "constrained_coefficient_indices": constrained,
        "ridge": plan.ridge, "curvature_penalty": plan.curvature_penalty,
        "curvature_definition": "normalized-feature-integrated-second-derivative-v1" if plan.curvature_penalty else None,
        "penalty_strength_units": "s", "condition_number": condition, "raw_condition_number": raw_condition,
        "effective_degrees_of_freedom": degrees, "train_objective": objective, "penalized_train_objective": penalized,
        "resume_level": "restart-only", "wall_seconds": time.perf_counter() - started, "peak_memory_bytes": None,
        "checkpoint": checkpoint}
