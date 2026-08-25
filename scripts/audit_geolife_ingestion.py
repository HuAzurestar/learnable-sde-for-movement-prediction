"""Audit a local cleaned GeoLife source without publishing row-level data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.geolife import audit_geolife_ingestion
from data.validation import DataValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geolife-root", required=True, type=Path)
    parser.add_argument("--osm-root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    try:
        report = audit_geolife_ingestion(
            args.geolife_root,
            osm_root=args.osm_root,
            registry_path=args.registry,
            verify_sha256=not args.skip_sha256,
        )
    except DataValidationError as exc:
        report = {
            "status": "failed",
            "technical_status": "failed",
            "release_status": "blocked",
            "failure_ledger": [
                {"area": "ingestion_contract", "status": "failed", "reason": str(exc)}
            ],
            "boundary_ledger": [],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "technical_status": report.get("technical_status"),
                "release_status": report.get("release_status"),
                "output": str(args.output),
            }
        )
    )
    return {"pass": 0, "provisional": 2, "failed": 3}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
