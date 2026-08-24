"""Compatibility entry point; public training now lives in :mod:`cli.train`."""

from cli.train import main


if __name__ == "__main__":
    raise SystemExit(main())
