"""Observed-state O1 components; job execution belongs to the shared runner."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from domain import ModelContext
from models.phase_space import ModelContractError, PhaseSpaceSDE, RBFResidualDrift, SplineResidualDrift


@dataclass(frozen=True)
class TransitionBatch:
    time: torch.Tensor
    state: torch.Tensor
    next_state: torch.Tensor
    dt: torch.Tensor
    context: ModelContext
    train_binding_hash: str
    split_role: str = "train"

    def validate(self, model):
        if self.split_role != "train" or self.train_binding_hash != model.spec.train_binding_hash:
            raise ModelContractError("UNAUTHORIZED_DATA: O1 fits only the frozen train binding")
        model._inputs(self.time, self.state, self.context)
        if (self.next_state.shape != self.state.shape or self.next_state.dtype != self.state.dtype
                or self.next_state.device != self.state.device or not torch.isfinite(self.next_state).all()
                or self.dt.shape != self.time.shape or self.dt.dtype != self.time.dtype
                or self.dt.device != self.time.device or not torch.isfinite(self.dt).all() or not (self.dt > 0).all()):
            raise ModelContractError("MODEL_CONTRACT_ERROR: finite aligned transitions and positive dt required")


def local_velocity_nll(model, batch, factor=None):
    """Two-dimensional Gaussian quasi-likelihood; no invented position noise."""
    batch.validate(model)
    factor = model.velocity_factor if factor is None else factor
    if (factor.shape != (2, 2) or factor.dtype != batch.state.dtype
            or factor.device != batch.state.device or not torch.isfinite(factor).all()):
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: velocity factor profile")
    covariance = factor @ factor.T
    chol, info = torch.linalg.cholesky_ex(covariance)
    if info.item() != 0:
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: O1 requires rank-two velocity covariance")
    residual = batch.next_state[:, 2:] - batch.state[:, 2:] - model.acceleration(
        batch.time, batch.state, batch.context) * batch.dt[:, None]
    whitened = torch.linalg.solve_triangular(chol, residual.T, upper=False).T / batch.dt.sqrt()[:, None]
    logdet = 2 * torch.log(torch.diag(chol)).sum() + 2 * batch.dt.log()
    values = .5 * (2 * math.log(2 * math.pi) + logdet + whitened.square().sum(-1))
    if not torch.isfinite(values).all():
        raise ModelContractError("NONFINITE: local velocity objective")
    return values.mean()


class VelocityCholesky(nn.Module):
    """Learn constant rank-two covariance with a registered positive diagonal floor."""

    def __init__(self, factor, diagonal_floor):
        super().__init__()
        if type(diagonal_floor) not in (int, float) or not math.isfinite(diagonal_floor) or diagonal_floor <= 0:
            raise ModelContractError("OBJECTIVE_INCOMPATIBLE: positive diffusion diagonal floor required")
        chol, info = torch.linalg.cholesky_ex(factor @ factor.T)
        if info.item() != 0 or not (torch.diag(chol) > diagonal_floor).all():
            raise ModelContractError("OBJECTIVE_INCOMPATIBLE: initial covariance below registered floor")
        self.floor = float(diagonal_floor)
        delta = torch.diag(chol) - self.floor
        # Stable inverse softplus, including large physical diffusion scales.
        self.raw_diagonal = nn.Parameter(delta + torch.log(-torch.expm1(-delta)))
        self.off_diagonal = nn.Parameter(chol[1, 0].detach().clone())

    def factor(self):
        diagonal = F.softplus(self.raw_diagonal) + self.floor
        return torch.stack((torch.stack((diagonal[0], diagonal.new_zeros(()))),
                            torch.stack((self.off_diagonal, diagonal[1]))))


@dataclass(frozen=True)
class O1Plan:
    max_steps: int = 200
    learning_rate: float = .01
    patience: int = 30
    tolerance: float = 1e-6
    gradient_norm_limit: float = 100.
    fit_diffusion: bool = True
    diffusion_diagonal_floor: float = .001

    def validate(self):
        if (type(self.max_steps) is not int or not 1 <= self.max_steps <= 10000
                or type(self.patience) is not int or not 1 <= self.patience <= self.max_steps
                or type(self.fit_diffusion) is not bool
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in
                       (self.learning_rate, self.tolerance, self.gradient_norm_limit, self.diffusion_diagonal_floor))
                or not 0 < self.learning_rate <= 1 or self.tolerance < 0
                or self.gradient_norm_limit <= 0 or self.diffusion_diagonal_floor <= 0):
            raise ModelContractError("MODEL_CONTRACT_ERROR: unregistered O1 plan")


def fit_o1(model: PhaseSpaceSDE, batches, plan: O1Plan, *, cancellation=None, progress=None,
           resume_state=None, checkpoint_requested=None, checkpoint_handler=None):
    """Finite local-gradient fitting component, called within a managed worker.

    This component is also used by numerical fixtures. It grants no permission
    to read data and cannot create studies, authorize inputs or reserve budgets.
    The registered execution adapter must admit inputs and enforce OS deadlines.
    """
    plan.validate()
    if model.velocity_factor.device.type != "cpu":
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: exact O1 continuation declares CPU only")
    if not isinstance(batches, (tuple, list)) or not 1 <= len(batches) <= 256:
        raise ModelContractError("MODEL_CONTRACT_ERROR: bounded train batches required")
    for batch in batches:
        if not isinstance(batch, TransitionBatch) or not 0 < len(batch.time) <= 4096:
            raise ModelContractError("MODEL_CONTRACT_ERROR: bounded TransitionBatch required")
        batch.validate(model)
    diffusion = VelocityCholesky(model.velocity_factor, plan.diffusion_diagonal_floor) if plan.fit_diffusion else None
    parameters = list(model.acceleration_model.parameters()) + ([] if diffusion is None else list(diffusion.parameters()))
    from estimation.phase_space_checkpoint import tensor_identity, training_scope, train_loop
    identities = [{"time": tensor_identity(b.time), "state": tensor_identity(b.state),
                   "next_state": tensor_identity(b.next_state), "dt": tensor_identity(b.dt),
                   "context": tensor_identity(b.context.condition), "split_role": b.split_role,
                   "train_binding_hash": b.train_binding_hash} for b in batches]
    scope = training_scope(model, plan, identities, "O1")

    def objective(step):
        batch = batches[step % len(batches)]
        factor = model.velocity_factor if diffusion is None else diffusion.factor()
        return local_velocity_nll(model, batch, factor), {"batch_index": step % len(batches)}

    def monitor():
        # Always synchronize the current factor before serializing optimizer/model.
        if diffusion is not None:
            model.velocity_factor.copy_(diffusion.factor())
        count = sum(len(part.time) for part in batches)
        return sum(local_velocity_nll(model, part).item() * len(part.time) for part in batches) / count

    return train_loop(model, parameters, plan, scope, objective, monitor, auxiliary=diffusion,
                      resume_state=resume_state, cancellation=cancellation, progress=progress,
                      checkpoint_requested=checkpoint_requested, checkpoint_handler=checkpoint_handler)


def fit_residual_basis(model, batches, *, ridge, condition_number_max, cancellation=None):
    """Stream weighted QR with a frozen affine base and explicit regularization.

    Workspace is O(Bq + q²); the implementation never forms N by N kernels or
    inverts normal equations. Ridge corresponds to a registered covariance-
    scaled coefficient penalty under the common velocity quasi-likelihood.
    """
    drift = model.acceleration_model
    if (not isinstance(drift, (RBFResidualDrift, SplineResidualDrift))
            or type(ridge) not in (float, int) or not math.isfinite(ridge) or ridge < 0
            or type(condition_number_max) not in (float, int) or not math.isfinite(condition_number_max)
            or condition_number_max < 1 or not isinstance(batches, (tuple, list)) or not 1 <= len(batches) <= 256):
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: registered bounded basis fit required")
    q = len(drift.coefficients)
    r = model.velocity_factor.new_empty((0, q))
    projected = model.velocity_factor.new_empty((0, 2))
    count = 0
    with torch.no_grad():
        for batch in batches:
            if cancellation is not None and cancellation():
                raise ModelContractError("INTERRUPTED: residual basis fit cancelled")
            batch.validate(model)
            if not 0 < len(batch.time) <= 4096:
                raise ModelContractError("RESOURCE_PLAN_REJECTED: basis batch quota")
            context = model._inputs(batch.time, batch.state, batch.context)
            basis = drift.basis(drift.selected_features(batch.state, context))
            target = (batch.next_state[:, 2:] - batch.state[:, 2:]) / batch.dt[:, None]
            target -= drift.affine(batch.time, batch.state, context)
            weight = batch.dt.sqrt()[:, None]
            combined = torch.cat((r, basis * weight), dim=0)
            responses = torch.cat((projected, target * weight), dim=0)
            orthogonal, r = torch.linalg.qr(combined, mode="reduced")
            projected = orthogonal.T @ responses
            count += len(batch.time)
        singular = torch.linalg.svdvals(r)
        if len(singular) != q or singular[-1] <= torch.finfo(r.dtype).eps * max(count, q) * singular[0]:
            raise ModelContractError("ILL_CONDITIONED: rank-deficient registered basis")
        raw_condition = (singular[0] / singular[-1]).item()
        if ridge:
            combined = torch.cat((r, math.sqrt(ridge) * torch.eye(q, dtype=r.dtype, device=r.device)))
            responses = torch.cat((projected, projected.new_zeros((q, 2))))
            orthogonal, r = torch.linalg.qr(combined, mode="reduced")
            projected = orthogonal.T @ responses
        regularized_singular = torch.linalg.svdvals(r)
        condition = (regularized_singular[0] / regularized_singular[-1]).item()
        if condition > condition_number_max:
            raise ModelContractError("ILL_CONDITIONED: registered condition threshold exceeded")
        coefficients = torch.linalg.solve_triangular(r, projected, upper=True)
        if not torch.isfinite(coefficients).all():
            raise ModelContractError("NONFINITE: residual basis coefficients")
        drift.coefficients.copy_(coefficients)
        degrees = (singular.square() / (singular.square() + ridge)).sum().item()
    return {"schema_version": "pirc26-basis-fit-v1", "status": "SUCCEEDED", "observations": count,
            "basis_count": q, "ridge": ridge, "condition_number": condition,
            "raw_condition_number": raw_condition, "effective_degrees_of_freedom": degrees,
            "resume_level": "restart-only", "checkpoint": model.checkpoint()}
