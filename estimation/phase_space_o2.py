"""Bounded observed-state M2 proper-score fine-tuning, after immutable O1."""

from dataclasses import asdict, dataclass, replace
import math

import torch

from estimation.phase_space_checkpoint import tensor_identity, training_scope, train_loop
from evaluation.phase_space import energy_score_value
from inference.phase_space import ForecastRequest, forecast
from infrastructure.research_store import digest
from models.phase_space import ModelContractError, NeuralResidualAccelerationDrift


@dataclass(frozen=True)
class HorizonTrainingExample:
    request: ForecastRequest
    target: torch.Tensor
    train_binding_hash: str
    split_role: str = "train"

    def validate(self, model):
        if self.split_role != "train" or self.train_binding_hash != model.spec.train_binding_hash:
            raise ModelContractError("UNAUTHORIZED_DATA: O2 fits only the frozen train binding")
        if not isinstance(self.request, ForecastRequest):
            raise ModelContractError("MODEL_CONTRACT_ERROR: causal O2 request required")
        self.request.validate(model)
        if (not isinstance(self.target, torch.Tensor)
                or self.target.shape != (len(self.request.time_grid), 4)
                or self.target.dtype != model.velocity_factor.dtype
                or self.target.device != model.velocity_factor.device or not torch.isfinite(self.target).all()):
            raise ModelContractError("MODEL_CONTRACT_ERROR: aligned finite O2 train target required")
        initial = self.target.new_tensor(self.request.initial_state)
        if not torch.equal(self.target[0], initial):
            raise ModelContractError("MODEL_CONTRACT_ERROR: target origin differs from known initial state")


@dataclass(frozen=True)
class O2Plan:
    horizon_indices: tuple[int, ...] = (1, 3)
    max_steps: int = 100
    learning_rate: float = .005
    patience: int = 20
    tolerance: float = 1e-6
    gradient_norm_limit: float = 100.
    curriculum_steps: int = 20
    max_grid_steps: int = 256
    max_gradient_state_elements: int = 262144

    def __post_init__(self):
        object.__setattr__(self, "horizon_indices", tuple(self.horizon_indices))

    def validate(self):
        if (not 1 <= len(self.horizon_indices) <= 32
                or any(type(i) is not int or i < 1 for i in self.horizon_indices)
                or tuple(sorted(set(self.horizon_indices))) != self.horizon_indices
                or type(self.max_steps) is not int or not 1 <= self.max_steps <= 10000
                or type(self.patience) is not int or not 1 <= self.patience <= self.max_steps
                or type(self.curriculum_steps) is not int or not 1 <= self.curriculum_steps <= self.max_steps
                or type(self.max_grid_steps) is not int or not 1 <= self.max_grid_steps <= 256
                or type(self.max_gradient_state_elements) is not int
                or not 8 <= self.max_gradient_state_elements <= 262144
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in (self.learning_rate, self.tolerance, self.gradient_norm_limit))
                or not 0 < self.learning_rate <= 1 or self.tolerance < 0 or self.gradient_norm_limit <= 0):
            raise ModelContractError("RESOURCE_PLAN_REJECTED: bounded O2 plan required")


def fit_o2(model, examples, plan, o1_result, *, resume_state=None, cancellation=None,
           progress=None, checkpoint_requested=None, checkpoint_handler=None,
           checkpoint_byte_limit=4 * 1024 * 1024):
    """G1 through the common solver; no latent model or alternate data authority.

    Each update uses one train origin with short-to-long cumulative horizons.
    Model selection within training compares all origins/all horizons against a
    separate fixed monitoring Brownian stream, never changing curriculum losses.
    Diffusion stays at the O1 checkpoint. This component does not admit studies.
    """
    if not isinstance(plan, O2Plan):
        raise ModelContractError("MODEL_CONTRACT_ERROR: O2Plan required")
    plan.validate()
    if not isinstance(model.acceleration_model, NeuralResidualAccelerationDrift) or model.velocity_factor.device.type != "cpu":
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: O2 G1 declares observed-state M2 CPU only")
    if (type(o1_result) is not dict or o1_result.get("schema_version") != "pirc26-fit-result-v1"
            or o1_result.get("objective") != "O1" or o1_result.get("gradient_route") != "G0"
            or o1_result.get("status") not in ("MAX_STEPS", "CONVERGED")
            or type(o1_result.get("steps")) is not int or o1_result["steps"] < 1
            or o1_result.get("checkpoint") != model.checkpoint()):
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: immutable completed O1 M2 lineage required")
    if not isinstance(examples, (list, tuple)) or not 1 <= len(examples) <= 32:
        raise ModelContractError("RESOURCE_PLAN_REJECTED: bounded O2 train origins required")
    for example in examples:
        if not isinstance(example, HorizonTrainingExample):
            raise ModelContractError("MODEL_CONTRACT_ERROR: O2 train example required")
        example.validate(model)
        req = example.request
        if (req.sample_count > 256 or len(req.time_grid) - 1 > plan.max_grid_steps
                or plan.horizon_indices[-1] >= len(req.time_grid)
                or req.sample_count * len(req.time_grid) * 4 > plan.max_gradient_state_elements):
            raise ModelContractError("RESOURCE_PLAN_REJECTED: O2 direct-gradient path quota")
    lineage = digest(o1_result)
    identities = {"o1_lineage_hash": lineage, "examples": [
        {"request": asdict(e.request), "target": tensor_identity(e.target),
         "train_binding_hash": e.train_binding_hash, "split_role": e.split_role} for e in examples]}
    scope = training_scope(model, plan, identities, "O2")

    def score(example, horizons, root, *, stop=None):
        req = replace(example.request, brownian_root_id=root,
                      time_grid=example.request.time_grid[:horizons[-1] + 1])
        result = forecast(model, req, cancellation=stop)
        if result["valid_paths"] != req.sample_count:
            # Failed paths never silently change the training objective population.
            raise ModelContractError("NONFINITE: O2 path failure; no survivor-only training score")
        values = [energy_score_value(result["samples"][:, i, :2], example.target[i, :2])[0] for i in horizons]
        return torch.stack(values).mean()

    def objective(step):
        index = step % len(examples)
        horizons = plan.horizon_indices[:min(len(plan.horizon_indices), 1 + step // plan.curriculum_steps)]
        root = digest({"root": examples[index].request.brownian_root_id, "phase": "o2-train", "step": step})
        return score(examples[index], horizons, root, stop=cancellation), {
            "batch_index": index, "horizon_indices": list(horizons), "brownian_root_id": root,
            "estimator_id": "energy-u-exact-v1"}

    def monitor():
        values = [score(e, plan.horizon_indices, digest({"root": e.request.brownian_root_id,
                                                        "phase": "o2-monitor", "origin": i}), stop=cancellation)
                  for i, e in enumerate(examples)]
        return torch.stack(values).mean().item()

    result = train_loop(model, list(model.acceleration_model.parameters()), plan, scope, objective, monitor,
                        resume_state=resume_state, cancellation=cancellation, progress=progress,
                        checkpoint_requested=checkpoint_requested, checkpoint_handler=checkpoint_handler,
                        checkpoint_byte_limit=checkpoint_byte_limit)
    return {**result, "o1_lineage_hash": lineage, "diffusion_policy": "frozen-o1",
            "training_score": "position-energy-u-exact-v1", "monitor_policy": "all-train-origins-horizons-fixed-independent-noise"}
