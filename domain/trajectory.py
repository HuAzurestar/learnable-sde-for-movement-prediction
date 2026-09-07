"""Pure adapters between observed trajectories and model state layouts."""

from __future__ import annotations

import torch

from .errors import DataValidationError
from .types import TrajectorySegment


def to_phase_space_1d(
    segment: TrajectorySegment,
    coord: int = 0,
) -> torch.Tensor:
    """Convert one position coordinate to the I1 ``[X, V]`` state layout."""

    position = segment.x[:, coord].to(torch.float64)
    time = segment.t.to(torch.float64)
    label = f"seg:{segment.meta.get('segment_id', '?')}"
    if not torch.isfinite(position).all():
        raise DataValidationError(f"{label} x contains NaN or infinity")
    if not torch.isfinite(time).all():
        raise DataValidationError(f"{label} t contains NaN or infinity")
    elapsed = torch.diff(time)
    if torch.any(elapsed <= 0):
        raise DataValidationError(f"{label} t must be strictly increasing")
    velocity = torch.diff(position) / elapsed.clamp(min=1e-6)
    velocity = torch.cat([velocity, velocity[-1:]])
    return torch.stack([position, velocity], dim=-1)


__all__ = ["to_phase_space_1d"]
