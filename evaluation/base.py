"""Evaluation contracts kept separate from model fitting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from domain import Forecast


class ScoringRule(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def score(self, forecast: Forecast, observation: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class EvaluationReport:
    per_item: Mapping[str, tuple[float, ...]]
    aggregate: Mapping[str, float]


class Evaluator:
    def __init__(self, rules: Sequence[ScoringRule]) -> None:
        if not rules:
            raise ValueError("Evaluator requires at least one scoring rule")
        self.rules = tuple(rules)

    def evaluate(
        self,
        forecasts: Sequence[Forecast],
        observations: Sequence[torch.Tensor],
    ) -> EvaluationReport:
        if len(forecasts) != len(observations):
            raise ValueError("forecast/observation lengths differ")
        values = {
            rule.name: tuple(
                float(rule.score(forecast, observation))
                for forecast, observation in zip(forecasts, observations)
            )
            for rule in self.rules
        }
        aggregate = {
            name: sum(scores) / len(scores) if scores else float("nan")
            for name, scores in values.items()
        }
        return EvaluationReport(values, aggregate)
