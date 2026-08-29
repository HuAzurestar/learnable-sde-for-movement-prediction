"""Validate the repository's Issue-first pull-request title convention."""

from __future__ import annotations

import re
import sys

TYPE_TO_PREFIX = {
    "feat": "feature", "fix": "fix", "docs": "docs", "refactor": "refactor",
    "perf": "perf", "test": "test", "build": "build", "ci": "ci", "chore": "chore",
}
HEADER = re.compile(r"^#([1-9][0-9]*) (" + "|".join(TYPE_TO_PREFIX) + r")(?:\([a-z0-9][a-z0-9-]*\))?!?: .+")


def main(title: str, branch: str) -> int:
    if re.fullmatch(r"chore: repository bootstrap(?: .+)?", title):
        return 0 if branch.startswith("chore/") else 1
    match = HEADER.fullmatch(title)
    if match and branch.startswith(f"{TYPE_TO_PREFIX[match.group(2)]}/{match.group(1)}-"):
        return 0
    print("PR title/branch must use #<issue> <type>: summary and <prefix>/<issue>-<summary>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
