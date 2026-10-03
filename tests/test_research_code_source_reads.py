"""Actual code identity readers, opened-file races and digest compatibility."""
import hashlib
import os
from pathlib import Path
import shutil

import pytest

from application.research_computation import paper_identity, MAX_INPUT_BYTES
from experiments.pirc25 import affine
from infrastructure.research_store import ResearchError, digest
from tests.research_file_observation import observe_file
from tests.test_research_paper_source_identity import replica


@pytest.mark.parametrize("entry", ["runtime", "paper"])
@pytest.mark.parametrize("scenario", ["replace", "symlink", "root_initial", "root_open"])
def test_code_source_redirect_denied_before_bytes(tmp_path, monkeypatch, entry, scenario):
    if os.name == "nt" and scenario != "replace":
        pytest.skip("POSIX-native source/root links; same-byte replacement tested on Windows")
    if entry == "paper":
        root, _ = replica(tmp_path)
        target = root / "scripts/pirc25/__init__.py"
        read = lambda: paper_identity(root)
    else:
        root = tmp_path / "runtime"
        root.mkdir()
        for name in ("config.py", "numerics.py", "registry.py"):
            (root / name).write_bytes(b"# synthetic source\n")
        target = root / "config.py"
        monkeypatch.setattr(affine, "ROOT", root)
        read = affine.code_hash
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / target.relative_to(root)
    external.parent.mkdir(parents=True, exist_ok=True)
    external.write_bytes(target.read_bytes())
    if scenario.startswith("root_"):
        shutil.copytree(root, outside, dirs_exist_ok=True)
    touched = []
    def change():
        assert not touched
        touched.append(True)
        if scenario == "replace":
            os.replace(external, target)
        elif scenario == "symlink":
            target.unlink()
            target.symlink_to(external)
        else:
            root.rename(tmp_path / "original")
            root.symlink_to(outside, target_is_directory=True)
    try:
        if scenario == "root_initial":
            change()
        observed = external if entry == "paper" and scenario == "root_initial" else target
        reads, _ = observe_file(monkeypatch, observed,
            before_open=None if scenario == "root_initial" else change)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA|CORRUPT_ARTIFACT"):
            read()
        assert touched == [True]
        assert reads == [], "identity reader consumed replaced/outside source bytes"
    finally:
        if root.is_symlink():
            root.unlink()


@pytest.mark.parametrize("entry", ["runtime", "paper"])
def test_source_identity_preserves_utf8_and_universal_newlines(tmp_path, monkeypatch, entry):
    content = ("# π synthetic\r\nx=1\ry=2\n" * 4096).encode("utf-8")
    if entry == "paper":
        root, _ = replica(tmp_path)
        target = root / "scripts/pirc25/__init__.py"
        target.write_bytes(content)
        actual = paper_identity(root)["files"]["scripts/pirc25/__init__.py"]
        expected = hashlib.sha256(target.read_text(encoding="utf-8").encode()).hexdigest()
    else:
        root = tmp_path / "runtime"
        root.mkdir()
        for name in ("config.py", "numerics.py", "registry.py"):
            (root / name).write_bytes(content)
        monkeypatch.setattr(affine, "ROOT", root)
        expected = digest({p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in root.glob("*.py")})
        actual = affine.code_hash()
    assert actual == expected


def test_runtime_source_large_file_streaming_remains_compatible(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    for name in ("config.py", "numerics.py", "registry.py"):
        (root / name).write_bytes(b"# synthetic\n")
    target = root / "config.py"
    # Above the paper quota: that admission cap does not apply to runtime sources.
    content = b"# x\r\n" * (MAX_INPUT_BYTES // 5 + 1)
    target.write_bytes(content)
    monkeypatch.setattr(affine, "ROOT", root)
    expected = digest({p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
        for p in root.glob("*.py")})
    def bounded(stream, size):
        assert 0 < size <= 1024 * 1024, "runtime source read allocated an entire file"
    reads, handles = observe_file(monkeypatch, target, before_read=bounded)
    assert affine.code_hash() == expected
    assert reads and handles == [True]
