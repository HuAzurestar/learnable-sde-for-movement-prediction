"""UTF-8 JSON artifact adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from domain import DataValidationError


class JsonArtifactStore:
    def write(self, payload: Mapping[str, Any], destination: Path) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read(self, source: Path) -> Mapping[str, Any]:
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"invalid JSON artifact: {source}") from exc
        if not isinstance(payload, dict):
            raise DataValidationError(f"JSON artifact must contain an object: {source}")
        return payload
