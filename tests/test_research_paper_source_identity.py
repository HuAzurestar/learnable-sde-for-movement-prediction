"""Actual package parent imports and bounded comparison source identity."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from application.research_computation import paper_identity, MAX_INPUT_BYTES
from infrastructure.research_store import ResearchError

RUNTIME = Path(__file__).resolve().parents[1]


def git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=PIRC-fixture",
        "-c", "user.email=fixture@example.invalid", *arguments], check=True,
        capture_output=True, text=True, timeout=10)


def replica(tmp_path, existing=False):
    root = tmp_path / "paper"
    shutil.copytree(RUNTIME.parent / "TSDE-SDE/scripts/pirc25", root / "scripts/pirc25",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    parent = root / "scripts/__init__.py"
    if existing:
        parent.write_text("VALUE = 'before'\n", encoding="utf-8")
    git(root, "init", "--quiet")
    git(root, "add", "scripts")
    git(root, "commit", "--quiet", "-m", "synthetic paper source fixture")
    return root, parent


@pytest.mark.parametrize("change", ["add", "modify", "remove"])
@pytest.mark.parametrize("field", ["source_tree_hash", "source_tree_dirty"])
def test_actually_imported_parent_changes_source_identity(tmp_path, change, field):
    root, parent = replica(tmp_path, existing=change != "add")
    before = paper_identity(root)
    if change == "remove":
        parent.unlink()
    else:
        parent.write_text("VALUE = 'after'\n", encoding="utf-8")
    after = paper_identity(root)
    if field == "source_tree_hash":
        assert after[field] != before[field]
    else:
        assert after[field] is True


def test_absent_parent_is_explicitly_bound(tmp_path):
    root, _ = replica(tmp_path)
    assert paper_identity(root)["files"].get("scripts/__init__.py", "omitted") is None


def test_present_parent_import_is_bound_to_its_actual_bytes(tmp_path):
    root, _ = replica(tmp_path, existing=True)
    identity = paper_identity(root)
    observed = subprocess.run([sys.executable, "-c",
        "import sys; sys.path.insert(0,sys.argv[1]); import scripts; print(scripts.VALUE)", str(root)],
        check=True, capture_output=True, text=True, timeout=10)
    assert observed.stdout.strip() == "before"
    assert "scripts/__init__.py" in identity["files"]
    assert identity["source_tree_dirty"] is False


def test_parent_symlink_is_rejected_before_source_read(tmp_path, monkeypatch):
    root, parent = replica(tmp_path, existing=True)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == parent or original(path))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        paper_identity(root)


def test_source_read_is_bounded_even_after_size_precheck(tmp_path, monkeypatch):
    root, _ = replica(tmp_path)
    target = root / "scripts/pirc25/__init__.py"
    original = Path.open
    reads = []
    class BoundedStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def read(self, size=-1):
            assert 0 <= size <= MAX_INPUT_BYTES + 1, "paper source read is not byte bounded"
            reads.append(size)
            return self.stream.read(size)
    def guarded(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return BoundedStream(stream) if path == target else stream
    monkeypatch.setattr(Path, "open", guarded)
    paper_identity(root)
    assert reads
