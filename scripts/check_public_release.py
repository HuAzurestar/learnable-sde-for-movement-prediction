"""Fail when public release candidates contain common private artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = {
    ".log",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pth",
    ".pyc",
    ".zip",
}
TEXT_PATTERNS = {
    "workstation path": re.compile(r"(?i)(?:E:[\\/]|[\\/]Users[\\/][^\\/]+|[\\/]home[\\/][^\\/]+)"),
    "internal work item": re.compile(r"\bNEX[-_]?\d+\b", re.IGNORECASE),
    "internal role": re.compile(r"\b(?:CEO|CRO)\b|算法研究员|审核专员|研究助理"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "common access token": re.compile(
        r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})"
    ),
}

# These references are part of the already-reviewed public preregistration
# record.  Keep the exception deliberately narrow: new internal work items and
# role names remain release blockers everywhere else.
PUBLIC_PREREGISTRATION = Path("experiments/capacity_preregistration")
PUBLIC_PREREGISTRATION_ID = "NEX-381"
PUBLIC_CRO_RECORD = Path("experiments/capacity_preregistration/PREREGISTRATION.md")
PUBLIC_PREREGISTRATION_REFERENCES = {
    Path("models/neural.py"),
    Path("tests/test_capacity_preregistration.py"),
}


def is_approved_public_reference(label: str, relative: Path, matched: str) -> bool:
    if (
        label == "internal work item"
        and (
            relative.is_relative_to(PUBLIC_PREREGISTRATION)
            or relative in PUBLIC_PREREGISTRATION_REFERENCES
        )
        and matched.upper() == PUBLIC_PREREGISTRATION_ID
    ):
        return True
    return label == "internal role" and relative == PUBLIC_CRO_RECORD and matched == "CRO"


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return [ROOT / line for line in result.stdout.splitlines() if line]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in {".git", ".lake", ".local"} for part in path.parts)
    ]


def main() -> int:
    problems: list[str] = []
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if relative == Path("scripts/check_public_release.py"):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden artifact: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in TEXT_PATTERNS.items():
            for match in pattern.finditer(content):
                if is_approved_public_reference(label, relative, match.group(0)):
                    continue
                line = content.count("\n", 0, match.start()) + 1
                problems.append(f"{label}: {relative}:{line}")

    if problems:
        print("Public release scan failed:")
        for problem in sorted(set(problems)):
            print(f"  - {problem}")
        return 1
    print("Public release scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
