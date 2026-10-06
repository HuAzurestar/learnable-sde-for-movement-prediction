"""Causal, bounded common Euler-Maruyama components for PIRC-26 models."""

from dataclasses import dataclass, replace
import hashlib
import math
import re
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


def _forecast(model, request: ForecastRequest, *, cancellation=None, increments_override=None):
    """No future truth parameter; evaluation is a separate authorized operation.

    The result is bounded to two million state elements. Research artifacts can
    later stream larger registered sample ranges through the same component.
    """
    request.validate(model)
    if increments_override is not None and (increments_override.shape != (request.sample_count, len(request.time_grid) - 1, 2)
            or increments_override.dtype != model.velocity_factor.dtype or increments_override.device != model.velocity_factor.device
            or not torch.isfinite(increments_override).all()):
        raise ModelContractError("MODEL_CONTRACT_ERROR: coupled Brownian increment profile")
    started = time.perf_counter()
    chunks, failed_ids, sample_ids = [], [], []
    dtype, device = model.velocity_factor.dtype, model.velocity_factor.device
    grid = torch.tensor(request.time_grid, dtype=dtype, device=device)
    for first in range(0, request.sample_count, request.chunk_size):
        if cancellation is not None and cancellation():
            raise ModelContractError("INTERRUPTED: common forecast cancelled")
        last = min(request.sample_count, first + request.chunk_size)
        state = torch.tensor(request.initial_state, dtype=dtype, device=device).expand(last - first, 4).clone()
        context = ModelContext(torch.tensor(request.context, dtype=dtype, device=device).expand(last - first, model.spec.context_dim))
        increments = (brownian_increments(request, model, first, last) if increments_override is None
                      else increments_override[first:last])
        states, alive = [state], torch.ones(len(state), dtype=torch.bool, device=device)
        for step, dt in enumerate(grid[1:] - grid[:-1]):
            if cancellation is not None and cancellation():
                raise ModelContractError("INTERRUPTED: common forecast cancelled")
            time_batch = grid[step].expand(len(state))
            drift = model.drift(time_batch, state, context)
            diffusion = model.diffusion(time_batch, state, context)
            candidate = state + drift * dt + torch.einsum("bij,bj->bi", diffusion, increments[:, step])
            alive = alive & torch.isfinite(candidate).all(-1) & (candidate.norm(dim=-1) <= request.maximum_state_norm)
            # Failed paths are quarantined from later model calls and excluded
            # only from samples; their IDs/counts remain in the returned result.
            state = torch.where(alive[:, None], candidate, torch.zeros_like(candidate))
            states.append(state)
        chunk = torch.stack(states, dim=1)
        chunks.append(chunk[alive])
        ids = list(range(first, last))
        selected = alive.detach().cpu().tolist()
        sample_ids.extend(i for i, keep in zip(ids, selected) if keep)
        failed_ids.extend(i for i, keep in zip(ids, selected) if not keep)
    samples = torch.cat(chunks)
    return {"schema_version": "pirc26-forecast-result-v1", "samples": samples,
            "sample_ids": sample_ids, "failed_sample_ids": failed_ids,
            "requested_paths": request.sample_count, "valid_paths": len(sample_ids),
            "failure_rate": len(failed_ids) / request.sample_count,
            "solver": "euler-maruyama", "time_grid": list(request.time_grid),
            "brownian_root_id": request.brownian_root_id, "brownian_convention": "sample-id-stream-on-frozen-grid-v1",
            "wall_seconds": time.perf_counter() - started,
            "error_budget": {"time_discretization_sensitivity": None,
                "mean_state_standard_error": ((samples.detach().std(0) / math.sqrt(len(samples))).cpu().tolist()
                                               if len(samples) >= 2 else None)}}


def forecast(model, request: ForecastRequest, *, cancellation=None):
    return _forecast(model, request, cancellation=cancellation)


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
