"""Reusable estimator contract and explicit fitting context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from domain import FitResult
from models.base import SDEModel

ModelT = TypeVar("ModelT", bound=SDEModel)
DataT = TypeVar("DataT")


@dataclass(frozen=True)
class FitContext:
    generator: torch.Generator
    device: torch.device
    dtype: torch.dtype


class Estimator(Generic[ModelT, DataT], ABC):
    """A fitting strategy; model, data, and runtime are explicit inputs."""

    @abstractmethod
    def fit(self, model: ModelT, data: DataT, context: FitContext) -> FitResult:
        ...


__all__ = ["Estimator", "FitContext"]
