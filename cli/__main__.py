"""``python -m cli {train,predict,ablate}`` dispatcher."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="learnable-sde")
    parser.add_argument("command", choices=("train", "predict", "ablate"))
    args, remainder = parser.parse_known_args(argv)
    if args.command == "train":
        from .train import main as command
    elif args.command == "predict":
        from .predict import main as command
    else:
        from .ablate import main as command
    return command(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
