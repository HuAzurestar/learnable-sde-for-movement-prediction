"""CI coverage scheduling controls; never use research fixtures as science."""

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from scripts.ci_pytest_partition import (
    Collection, canonical_suite, partition_for, read_receipts, receipt, terminal_outcome, verify,
)


SHA = "a" * 40
SUITE = sorted(f"tests/test_module_{m}.py::test_case[{p}]" for m in range(64) for p in range(3))


def reports():
    result = []
    for index in range(4):
        capture = Collection(index, 4)
        capture.full_suite = SUITE
        capture.selected = [SimpleNamespace(nodeid=n) for n in SUITE if partition_for(n, 4) == index]
        capture.reports = {item.nodeid: [("setup", "passed"), ("call", "passed"), ("teardown", "passed")]
                           for item in capture.selected}
        result.append(receipt(capture, SHA, "3.10", 0))
    return result


def check(values):
    return verify(values, sha=SHA, version="3.10", count=4, full_suite=SUITE, collection_skips=[])


def test_complete_disjoint_module_stable_union():
    values = reports()
    result = check(values)
    assert result == {"collected": 192, "passed": 192, "skipped": 0,
                      "failed": 0, "incomplete": 0, "collection_skips": 0}
    for module in range(64):
        assert len({partition_for(f"tests/test_module_{module}.py::test_case[{p}]", 4) for p in range(3)}) == 1
    assert sum(len(v["outcomes"]) for v in values) == len(SUITE)


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "source", "version", "schema",
                                   "count", "boolean-index", "roster", "all-rosters", "collection-skips",
                                   "omitted-node", "foreign-node", "moved-node", "failed", "incomplete", "exit"])
def test_coverage_rejects_non_exhaustive_or_unsuccessful_receipts(fault):
    values = reports()
    node = next(iter(values[0]["outcomes"]))
    if fault == "missing":
        values.pop()
    elif fault == "extra":
        values.append(deepcopy(values[0]))
    elif fault == "duplicate":
        values[1] = deepcopy(values[0])
    elif fault == "source":
        values[0]["source_sha"] = "b" * 40
    elif fault == "version":
        values[0]["python_version"] = "3.12"
    elif fault == "schema":
        values[0]["bypass"] = True
    elif fault == "count":
        values[0]["partition_count"] = 3
    elif fault == "boolean-index":
        values[0]["partition_index"] = False
    elif fault == "roster":
        values[0]["full_suite"] = SUITE[:-1]
    elif fault == "all-rosters":
        for value in values:
            value["full_suite"] = SUITE[:-1]
            value["outcomes"].pop(SUITE[-1], None)
    elif fault == "collection-skips":
        values[0]["collection_skips"] = ["hidden-module"]
    elif fault == "omitted-node":
        del values[0]["outcomes"][node]
    elif fault == "foreign-node":
        values[0]["outcomes"]["tests/foreign.py::test_foreign"] = "passed"
    elif fault == "moved-node":
        values[1]["outcomes"][node] = values[0]["outcomes"].pop(node)
    elif fault in ("failed", "incomplete"):
        values[0]["outcomes"][node] = fault
    else:
        values[0]["exit_code"] = 1
    with pytest.raises(ValueError):
        check(values)


@pytest.mark.parametrize("phases,expected", [
    ([], "incomplete"), ([("setup", "passed"), ("call", "passed")], "incomplete"),
    ([("setup", "skipped"), ("teardown", "passed")], "skipped"),
    ([("call", "passed"), ("teardown", "failed")], "failed"),
    ([("call", "failed"), ("call", "passed")], "incomplete"),
    ([("setup", "passed"), ("call", "passed"), ("teardown", "passed")], "passed"),
])
def test_outcome_requires_genuine_terminal_phases(phases, expected):
    assert terminal_outcome(phases) == expected


def test_skips_are_not_promoted_to_passed():
    values = reports()
    node = next(iter(values[0]["outcomes"]))
    values[0]["outcomes"][node] = "skipped"
    result = check(values)
    assert result["passed"] == 191 and result["skipped"] == 1


def test_external_deselection_and_duplicate_collection_refused():
    with pytest.raises(ValueError, match="external deselection"):
        Collection().pytest_deselected([object()])
    with pytest.raises(ValueError):
        canonical_suite([SUITE[0], SUITE[0]])


@pytest.mark.parametrize("fault", ["duplicate-keys", "oversized"])
def test_receipt_reader_refuses_duplicate_keys_and_oversized_inputs(tmp_path, fault):
    # Keep the collected node identity bounded; do not put megabytes of fixture
    # payload into PYTEST_CURRENT_TEST or every partition's full-suite receipt.
    contents = '{"outcomes": {}, "outcomes": {}}' if fault == "duplicate-keys" else "x" * (8 * 1024 * 1024 + 1)
    (tmp_path / "coverage.json").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        read_receipts(tmp_path)


def test_workflow_preserves_versions_required_names_caps_and_full_verification():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    shards = jobs["python-shards"]
    assert shards["timeout-minutes"] == jobs["python"]["timeout-minutes"] == 20
    assert shards["strategy"]["fail-fast"] is False
    assert shards["strategy"]["matrix"] == {"python-version": ["3.10", "3.12"], "partition": [0, 1, 2, 3]}
    assert jobs["python"]["name"] == "test (${{ matrix.python-version }})"
    assert jobs["python"]["needs"] == ["python-shards"]
    assert jobs["python"]["if"] == "${{ always() }}"
    assert jobs["required"]["needs"] == ["policy", "python"]
    commands = [step.get("run", "") for step in shards["steps"]]
    assert any("ci_pytest_partition.py run" in command and "--count 4" in command for command in commands)
    assert not any("-k " in command or "-m pytest" in command for command in commands)
    gates = [step.get("run", "") for step in jobs["python"]["steps"]]
    assert any('test "$SHARDS" = success' in command for command in gates)
    assert any("ci_pytest_partition.py verify" in command and "--count 4" in command for command in gates)


def test_actual_partition_processes_and_independent_recollection(tmp_path):
    """Real subprocesses over a disposable Git-backed suite, not synthetic flags."""
    root = tmp_path / "suite"
    (root / "scripts").mkdir(parents=True)
    (root / "tests").mkdir()
    helper = Path(__file__).resolve().parents[1] / "scripts/ci_pytest_partition.py"
    (root / "scripts/ci_pytest_partition.py").write_text(helper.read_text(encoding="utf-8"), encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    for module in range(64):
        body = ("import pytest\n"
                "calls = 0\n"
                "@pytest.fixture(scope='module', autouse=True)\n"
                "def owned_module():\n"
                "    global calls\n    calls += 1\n    assert calls == 1\n"
                "    yield\n    assert calls == 1\n"
                "def test_first():\n    assert calls == 1\n"
                "def test_second():\n    assert calls == 1\n")
        if module == 0:
            body += "def test_declared_skip():\n    pytest.skip('disposable declared skip')\n"
        (root / "tests" / f"test_module_{module}.py").write_text(body, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=CI fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "disposable suite"],
                   cwd=root, check=True, capture_output=True)
    directory = tmp_path / "coverage"
    values = []
    for index in range(4):
        path = directory / f"coverage-{index}.json"
        completed = subprocess.run([sys.executable, "scripts/ci_pytest_partition.py", "run",
            "--count", "4", "--index", str(index), "--report", str(path)], cwd=root,
            capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        values.append(json.loads(path.read_text(encoding="utf-8")))
    verified = subprocess.run([sys.executable, "scripts/ci_pytest_partition.py", "verify",
        "--count", "4", "--directory", str(directory)], cwd=root,
        capture_output=True, text=True, timeout=60)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert '"collected": 129' in verified.stdout
    assert '"passed": 128' in verified.stdout and '"skipped": 1' in verified.stdout
    # Every worker can consistently omit the same node in its reported roster;
    # only the independently recollected full source exposes that omission.
    omitted = values[0]["full_suite"][-1]
    for index, value in enumerate(values):
        value["full_suite"].remove(omitted)
        value["outcomes"].pop(omitted, None)
        (directory / f"coverage-{index}.json").write_text(json.dumps(value), encoding="utf-8")
    denied = subprocess.run([sys.executable, "scripts/ci_pytest_partition.py", "verify",
        "--count", "4", "--directory", str(directory)], cwd=root,
        capture_output=True, text=True, timeout=60)
    assert denied.returncode != 0 and "independent full roster" in denied.stderr
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
