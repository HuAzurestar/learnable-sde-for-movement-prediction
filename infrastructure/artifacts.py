"""UTF-8 JSON artifact adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class JsonArtifactStore:
    def write(self, payload: Mapping[str, Any], destination: Path) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
