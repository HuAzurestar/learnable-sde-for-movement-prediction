"""Run the existing numerical ablation service through the public CLI."""

from __future__ import annotations

from experiments.ablate import main as run_ablation


def main(argv: list[str] | None = None) -> int:
    return run_ablation(argv)


if __name__ == "__main__":
    raise SystemExit(main())
