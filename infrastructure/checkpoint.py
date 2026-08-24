"""PyTorch checkpoint adapter using the standard module state lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from domain import DataValidationError
from models.base import SDEModel


class TorchModelStore:
    def save(
        self,
        model: SDEModel,
        metadata: Mapping[str, Any],
        destination: Path,
    ) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "metadata": dict(metadata)},
            destination,
        )

    def load_state(
        self,
        source: Path,
        map_location: torch.device | str = "cpu",
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
        payload = torch.load(Path(source), map_location=map_location)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise DataValidationError(f"无效 checkpoint: {source}")
        return payload["state_dict"], payload.get("metadata", {})
