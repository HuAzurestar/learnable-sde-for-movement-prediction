"""Causal, bounded common Euler-Maruyama components for PIRC-26 models."""

from dataclasses import dataclass, replace
import hashlib
import math
import re
import struct
import sys
import time

import torch

from domain import ModelContext
from models.phase_space import ModelContractError


@dataclass(frozen=True)
class ForecastRequest:
    initial_state: tuple[float, ...]
    time_grid: tuple[float, ...]
    history_cutoff: float
    sample_count: int
    brownian_root_id: str
    context: tuple[float, ...] = ()
    chunk_size: int = 256
    maximum_state_norm: float = 1e8

    def __post_init__(self):
        for key in ("initial_state", "time_grid", "context"):
            object.__setattr__(self, key, tuple(getattr(self, key)))

    def validate(self, model):
        if (len(self.initial_state) != 4 or len(self.context) != model.spec.context_dim
                or not 2 <= len(self.time_grid) <= 10000
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in
                       (*self.initial_state, *self.context, *self.time_grid, self.history_cutoff, self.maximum_state_norm))
                or any(a >= b for a, b in zip(self.time_grid, self.time_grid[1:]))
                or self.history_cutoff > self.time_grid[0]
                or type(self.sample_count) is not int or not 2 <= self.sample_count <= 100000
                or type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 256
                or self.sample_count * len(self.time_grid) * 4 > 2_000_000
                or self.maximum_state_norm <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(self.brownian_root_id))):
            raise ModelContractError("RESOURCE_PLAN_REJECTED: causal bounded forecast request required")
        if model.velocity_factor.dtype == torch.float32:
            try:
                grid = [struct.unpack("f", struct.pack("f", t))[0] for t in self.time_grid]
            except (OverflowError, struct.error) as exc:
                raise ModelContractError("RESOURCE_PLAN_REJECTED: time grid exceeds model dtype") from exc
            if any(not math.isfinite(t) for t in grid) or any(a >= b for a,b in zip(grid,grid[1:])):
                raise ModelContractError("RESOURCE_PLAN_REJECTED: time grid collapses after model dtype conversion")


def brownian_increments(request, model, first, last):
    """Independent sample-ID streams: chunking preserves exact paired noise."""
    if not 0 <= first < last <= request.sample_count:
        raise ModelContractError("MODEL_CONTRACT_ERROR: Brownian sample range")
    dtype, device = model.velocity_factor.dtype, model.velocity_factor.device
    grid = torch.tensor(request.time_grid, dtype=dtype, device=device)
    streams = []
    for sample_id in range(first, last):
        seed = int(hashlib.sha256((request.brownian_root_id + ":" + str(sample_id)).encode()).hexdigest()[:16], 16) % (2**63 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        streams.append(torch.randn((len(grid) - 1, 2), generator=generator, dtype=dtype, device=device))
    return torch.stack(streams) * (grid[1:] - grid[:-1]).sqrt()[None, :, None]


def _forecast(model, request: ForecastRequest, *, cancellation=None, increments_override=None,
              resume_state=None, checkpoint_requested=None, checkpoint_handler=None):
    """No future truth parameter; evaluation is a separate authorized operation.

    The result is bounded to two million state elements. Research artifacts can
    later stream larger registered sample ranges through the same component.
    """
    request.validate(model)
    managed = checkpoint_requested is not None or resume_state is not None or checkpoint_handler is not None
    if managed and ((checkpoint_requested is None) != (checkpoint_handler is None)
            or increments_override is not None or torch.is_grad_enabled()
            or model.velocity_factor.device.type != "cpu" or torch.get_num_threads() != 1
            or model.velocity_factor.dtype not in (torch.float32,torch.float64) or sys.byteorder != "little"):
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: exact inference continuation requires no-grad single-thread CPU and paired controls")
    if managed:
        width = model.velocity_factor.element_size()
        raw_bytes = request.sample_count * len(request.time_grid) * 4 * width + len(request.time_grid) * 20 * 8
        if (raw_bytes * 4 + 2) // 3 + request.sample_count * 24 + 16384 > 4 * 1024 * 1024:
            raise ModelContractError("RESOURCE_PLAN_REJECTED: full forecast continuation exceeds bounded frame")
    if increments_override is not None and (increments_override.shape != (request.sample_count, len(request.time_grid) - 1, 2)
            or increments_override.dtype != model.velocity_factor.dtype or increments_override.device != model.velocity_factor.device
            or not torch.isfinite(increments_override).all()):
        raise ModelContractError("MODEL_CONTRACT_ERROR: coupled Brownian increment profile")
    started = time.perf_counter()
    failed_ids, sample_ids = [], []
    dtype, device = model.velocity_factor.dtype, model.velocity_factor.device
    from inference.phase_space_resume import scope, restore, snapshot, moments
    identity = scope(model, request) if managed else None
    first, step, states, alive = 0, 0, None, None
    if resume_state is not None:
        first, step, completed, sample_ids, failed_ids, states, alive, mean, m2 = restore(resume_state, identity, request, dtype)
    else:
        completed = torch.empty((0, len(request.time_grid), 4), dtype=dtype, device=device)
        mean = torch.zeros((len(request.time_grid),4), dtype=torch.float64, device=device)
        m2 = torch.zeros((len(request.time_grid),4,4), dtype=torch.float64, device=device)
    grid = torch.tensor(request.time_grid, dtype=dtype, device=device)
    chunks = [completed] if len(completed) else []
    def all_samples():
        return torch.cat(chunks) if chunks else completed
    def save_if_requested():
        if checkpoint_requested is not None and checkpoint_requested():
            saved = snapshot(identity, first, step, all_samples(), sample_ids, failed_ids, states, alive, mean, m2)
            work = first * (len(grid) - 1) + (0 if states is None else len(alive) * step)
            checkpoint_handler(saved, {"completed_steps": work, "total_steps": request.sample_count * (len(grid) - 1)})
            return {"status": "CHECKPOINTED", "forecast_state": saved}
        return None
    while first < request.sample_count:
        saved = save_if_requested()
        if saved is not None:
            return saved
        if cancellation is not None and cancellation():
            raise ModelContractError("INTERRUPTED: common forecast cancelled")
        last = min(request.sample_count, first + request.chunk_size)
        state = (torch.tensor(request.initial_state, dtype=dtype, device=device).expand(last - first, 4).clone()
                 if states is None else states[-1])
        context = ModelContext(torch.tensor(request.context, dtype=dtype, device=device).expand(last - first, model.spec.context_dim))
        increments = (brownian_increments(request, model, first, last) if increments_override is None
                      else increments_override[first:last])
        if states is None:
            states, alive, step = [state], torch.ones(len(state), dtype=torch.bool, device=device), 0
        while step < len(grid) - 1:
            saved = save_if_requested()
            if saved is not None:
                return saved
            if cancellation is not None and cancellation():
                raise ModelContractError("INTERRUPTED: common forecast cancelled")
            dt = grid[step + 1] - grid[step]
            time_batch = grid[step].expand(len(state))
            drift = model.drift(time_batch, state, context)
            diffusion = model.diffusion(time_batch, state, context)
            candidate = state + drift * dt + torch.einsum("bij,bj->bi", diffusion, increments[:, step])
            alive = alive & torch.isfinite(candidate).all(-1) & (candidate.norm(dim=-1) <= request.maximum_state_norm)
            # Failed paths are quarantined from later model calls and excluded
            # only from samples; their IDs/counts remain in the returned result.
            state = torch.where(alive[:, None], candidate, torch.zeros_like(candidate))
            states.append(state)
            step += 1
        chunk = torch.stack(states, dim=1)
        valid = chunk[alive]
        chunks.append(valid)
        ids = list(range(first, last))
        selected = alive.detach().cpu().tolist()
        sample_ids.extend(i for i, keep in zip(ids, selected) if keep)
        failed_ids.extend(i for i, keep in zip(ids, selected) if not keep)
        # Fixed sample-ID update order, independent of save and chunk boundaries.
        count = len(sample_ids) - len(valid)
        for sample in valid:
            count += 1
            sample = sample.detach().double()
            delta = sample - mean
            mean = mean + delta / count
            m2 = m2 + delta[:, :, None] * (sample - mean)[:, None, :]
        first, step, states, alive = last, 0, None, None
    saved = save_if_requested()
    if saved is not None:
        return saved
    samples = all_samples()
    result = {"schema_version": "pirc26-forecast-result-v1", "samples": samples,
            "sample_ids": sample_ids, "failed_sample_ids": failed_ids,
            "requested_paths": request.sample_count, "valid_paths": len(sample_ids),
            "failure_rate": len(failed_ids) / request.sample_count,
            "solver": "euler-maruyama", "time_grid": list(request.time_grid),
            "brownian_root_id": request.brownian_root_id, "brownian_convention": "sample-id-stream-on-frozen-grid-v1",
            "wall_seconds": time.perf_counter() - started,
            "moments": {"estimator_id": "ordered-sample-welford-full-state-v1", "sample_count": len(samples),
                "mean": mean.detach().cpu().tolist() if len(samples) else None,
                "covariance": (m2 / (len(samples) - 1)).detach().cpu().tolist() if len(samples) >= 2 else None},
            "error_budget": {"time_discretization_sensitivity": None,
                "mean_state_standard_error": ((samples.detach().std(0) / math.sqrt(len(samples))).cpu().tolist()
                                               if len(samples) >= 2 else None)}}
    if managed:
        result["completion_state"] = snapshot(identity, first, step, samples, sample_ids, failed_ids, None, None, mean, m2)
    return result


def forecast(model, request: ForecastRequest, *, cancellation=None, resume_state=None,
             checkpoint_requested=None, checkpoint_handler=None):
    return _forecast(model, request, cancellation=cancellation, resume_state=resume_state,
                     checkpoint_requested=checkpoint_requested, checkpoint_handler=checkpoint_handler)


def forecast_coupled_levels(model, request: ForecastRequest, *, cancellation=None):
    """Two registered step levels, coupled by summing consecutive fine increments.

    Fine-grid identity is explicit; two unrelated per-grid Gaussian streams are
    not mislabeled as the same Brownian path. Costs include both rollouts.
    """
    request.validate(model)
    grid = []
    for left, right in zip(request.time_grid, request.time_grid[1:]):
        grid.extend((left, (left + right) / 2))
    grid.append(request.time_grid[-1])
    fine_request = replace(request, time_grid=tuple(grid))
    fine_request.validate(model)
    fine_increments = brownian_increments(fine_request, model, 0, request.sample_count)
    coarse_increments = fine_increments.reshape(request.sample_count, len(request.time_grid) - 1, 2, 2).sum(2)
    coarse = _forecast(model, request, cancellation=cancellation, increments_override=coarse_increments)
    fine = _forecast(model, fine_request, cancellation=cancellation, increments_override=fine_increments)
    for result in (coarse, fine):
        result["brownian_convention"] = "sum-of-two-fine-increments-v1"
        result["brownian_reference_grid"] = list(fine_request.time_grid)
    # Failure identities are aligned explicitly, not by array row offsets.
    coarse_rows = {key: index for index, key in enumerate(coarse["sample_ids"])}
    fine_rows = {key: index for index, key in enumerate(fine["sample_ids"])}
    paired = sorted(coarse_rows.keys() & fine_rows.keys())
    sensitivity = None
    if paired:
        a = coarse["samples"][[coarse_rows[i] for i in paired]]
        b = fine["samples"][[fine_rows[i] for i in paired]][:, ::2]
        sensitivity = (a - b).detach().square().mean(0).sqrt().cpu().tolist()
    coarse["error_budget"]["time_discretization_sensitivity"] = sensitivity
    return {"coarse": coarse, "fine": fine, "paired_sample_ids": paired,
            "step_sensitivity_estimator": "paired-strong-rms-v1"}
