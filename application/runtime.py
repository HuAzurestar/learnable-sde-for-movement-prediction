"""Explicit runtime tensor policy and independent random streams."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


@dataclass(frozen=True)
class RandomStreams:
    training: torch.Generator
    inference: torch.Generator
    bootstrap: torch.Generator

    @classmethod
    def from_seed(cls, seed: int, device: torch.device | str = "cpu") -> "RandomStreams":
        target = torch.device(device)
        return cls(
            training=_generator(seed + 1, target),
            inference=_generator(seed + 2, target),
            bootstrap=_generator(seed + 3, target),
        )


@dataclass(frozen=True)
class RunContext:
    device: torch.device
    dtype: torch.dtype
    random: RandomStreams

    @classmethod
    def create(
        cls,
        seed: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "RunContext":
        target = torch.device(device)
        return cls(target, dtype, RandomStreams.from_seed(seed, target))
