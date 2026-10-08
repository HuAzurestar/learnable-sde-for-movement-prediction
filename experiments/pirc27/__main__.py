"""Bounded offline study preparation; never register, grant or run research."""

import argparse
import json
from pathlib import Path

from infrastructure.research_store import ResearchError
from .preparation import prepare_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="revalidate a frozen four-state design and existing runtime identity")
    prepare.add_argument("design", type=Path)
    prepare.add_argument("--expected-hash", required=True, help="canonical frozen design hash")
    prepare.add_argument("--runtime-config", type=Path, required=True)
    prepare.add_argument("--expected-runtime-config-hash", required=True, help="SHA256 of exact connection file bytes")
    args = parser.parse_args(argv)
    try:
        result = prepare_study(args.design, expected_hash=args.expected_hash, runtime_config=args.runtime_config,
            expected_runtime_config_hash=args.expected_runtime_config_hash)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except ResearchError as exc:
        print(json.dumps({"error": exc.envelope()}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
