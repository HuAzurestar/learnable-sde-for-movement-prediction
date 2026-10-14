"""Exhaustive module-stable CI partitions with independently collected coverage.

This is CI scheduling only. It does not shorten a test, alter numerical clocks,
or touch any production research input, authorization, supervisor or budget.
Normal ``python -m pytest`` is unchanged; only this explicit CI entry partitions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


SCHEMA = "pytest-exhaustive-partition-v1"
FIELDS = {"schema_version", "source_sha", "python_version", "partition_count",
          "partition_index", "full_suite", "collection_skips", "outcomes", "exit_code"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module_for(node):
    require(type(node) is str and "::" in node, "pytest node identity required")
    module = node.split("::", 1)[0]
    path = PurePosixPath(module)
    require(module.endswith(".py") and not path.is_absolute() and ".." not in path.parts
            and "\\" not in module, "repository-relative test module required")
    return module


def partition_for(node, count):
    require(type(count) is int and 1 <= count <= 16, "bounded partition count required")
    return int.from_bytes(hashlib.sha256(module_for(node).encode("utf-8")).digest(), "big") % count


def canonical_suite(nodes):
    require(type(nodes) is list and nodes and len(nodes) == len(set(nodes)),
            "nonempty unique complete collection required")
    for node in nodes:
        module_for(node)
    return sorted(nodes)


def terminal_outcome(reports):
    phases = [phase for phase, _ in reports]
    if len(phases) != len(set(phases)):
        return "incomplete"  # No best-of-retries or duplicated call report.
    if any(outcome == "failed" for _, outcome in reports):
        return "failed"
    if any(outcome == "skipped" for _, outcome in reports):
        return "skipped"
    if set(phases) == {"setup", "call", "teardown"} and all(o == "passed" for _, o in reports):
        return "passed"
    return "incomplete"


class Collection:
    def __init__(self, index=None, count=None):
        self.index, self.count = index, count
        self.full_suite, self.skips, self.selected = [], [], []
        self.reports = {}
        self.partitioning = False

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skips.append(report.nodeid)

    def pytest_deselected(self, items):
        if items and not self.partitioning:
            raise ValueError("external deselection cannot qualify the full suite")

    def pytest_collection_modifyitems(self, session, config, items):
        self.full_suite = canonical_suite([item.nodeid for item in items])
        if self.index is None:
            return  # Independent full recollection, no selection/filtering.
        self.selected = [item for item in items if partition_for(item.nodeid, self.count) == self.index]
        other = [item for item in items if partition_for(item.nodeid, self.count) != self.index]
        require(self.selected, "empty CI partition is not successful execution")
        self.partitioning = True
        try:
            config.hook.pytest_deselected(items=other)
        finally:
            self.partitioning = False
        items[:] = self.selected  # Keep original order and every module fixture.

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, []).append((report.when, report.outcome))


def pytest_session(collection, *, collect_only=False):
    import pytest
    require(not os.environ.get("PYTEST_ADDOPTS", "").strip(), "unregistered pytest options refused")
    # Capture after other collection hooks; refuse externally deselected items.
    Collection.pytest_collection_modifyitems = pytest.hookimpl(trylast=True)(
        Collection.pytest_collection_modifyitems)
    return int(pytest.main(["-qq", "--collect-only"] if collect_only else ["-q"], plugins=[collection]))


def source_sha():
    root = Path(__file__).resolve().parents[1]
    require(Path.cwd().resolve() == root, "run CI from the repository root")
    require(not subprocess.check_output(["git", "status", "--porcelain"]).strip(),
            "CI source must be clean before collection")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def receipt(collection, sha, version, exit_code):
    return {"schema_version": SCHEMA, "source_sha": sha, "python_version": version,
            "partition_count": collection.count, "partition_index": collection.index,
            "full_suite": collection.full_suite, "collection_skips": sorted(set(collection.skips)),
            "outcomes": {item.nodeid: terminal_outcome(collection.reports.get(item.nodeid, []))
                         for item in collection.selected}, "exit_code": exit_code}


def verify(receipts, *, sha, version, count, full_suite, collection_skips):
    require(re.fullmatch(r"[0-9a-f]{40}", sha) is not None, "literal source SHA required")
    require(type(count) is int and 1 <= count <= 16, "bounded partition count required")
    expected = canonical_suite(full_suite)
    require(len(receipts) == count, "missing or extra partition receipt")
    seen, covered = set(), set()
    totals = {"passed": 0, "skipped": 0, "failed": 0, "incomplete": 0}
    for value in receipts:
        require(type(value) is dict and set(value) == FIELDS, "closed coverage receipt required")
        require(value["schema_version"] == SCHEMA and value["source_sha"] == sha
                and value["python_version"] == version, "borrowed source or Python version")
        index = value["partition_index"]
        require(type(index) is int and 0 <= index < count and index not in seen
                and type(value["partition_count"]) is int and value["partition_count"] == count,
                "duplicate or wrong partition identity")
        seen.add(index)
        require(value["full_suite"] == expected and value["collection_skips"] == collection_skips,
                "partition collection differs from independent full roster")
        assigned = {node for node in expected if partition_for(node, count) == index}
        outcomes = value["outcomes"]
        require(assigned and type(outcomes) is dict and set(outcomes) == assigned,
                "missing, extra or wrongly assigned test outcome")
        require(not covered.intersection(outcomes), "overlapping partition coverage")
        require(type(value["exit_code"]) is int and value["exit_code"] == 0,
                "non-successful pytest process")
        for outcome in outcomes.values():
            require(outcome in ("passed", "skipped"), "failed or incomplete test cannot pass coverage")
            totals[outcome] += 1
        covered.update(outcomes)
    require(covered == set(expected), "partition union is not the entire independent suite")
    return {"collected": len(expected), **totals, "collection_skips": len(collection_skips)}


def read_receipts(directory):
    root = directory.resolve(strict=True)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate coverage JSON key")
            value[key] = item
        return value
    result = []
    for path in sorted(root.rglob("*.json")):
        require(not path.is_symlink() and path.resolve().is_relative_to(root)
                and path.stat().st_size <= 8 * 1024 * 1024, "bounded regular coverage file required")
        result.append(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--count", type=int, required=True)
    run.add_argument("--report", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--count", type=int, required=True)
    check.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    sha = source_sha()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if args.command == "run":
        require(type(args.count) is int and 1 <= args.count <= 16
                and 0 <= args.index < args.count, "invalid partition arguments")
        require(not args.report.exists(), "coverage receipt already exists; no silent overwrite")
        collection = Collection(args.index, args.count)
        code = pytest_session(collection)
        value = receipt(collection, sha, version, code)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
        return code
    collection = Collection()
    require(pytest_session(collection, collect_only=True) == 0, "independent complete collection failed")
    result = verify(read_receipts(args.directory), sha=sha, version=version, count=args.count,
                    full_suite=collection.full_suite, collection_skips=sorted(set(collection.skips)))
    print("Verified exhaustive supported-version coverage:", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
