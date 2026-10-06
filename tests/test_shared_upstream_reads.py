"""Real metadata reads preserve frozen upstream identity and lexical roots."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess

import pytest

from experiments.pirc25.upstream import (
    AdmissionError, Binding, PUBLIC_BINDINGS, audit_inputs, canonical_hash,
    validate_binding,
)
from tests.research_file_observation import observe_file


ROOT = Path(__file__).resolve().parents[1]


def synthetic_binding(tmp_path, *, payload_size=0):
    root = tmp_path / "holder" / "inputs"
    root.mkdir(parents=True)
    payload = {"schema_version": "synthetic-v1", "status": "frozen",
               "note": "合成元数据-π" + "x" * payload_size}
    binding = Binding("synthetic", "metadata.json", "synthetic-v1",
                      canonical_hash(payload), "frozen")
    target = root / binding.path
    target.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2)
                        + "\r\n").encode("utf-8"))
    return root, target, payload, binding


@contextmanager
def redirect_directory(alias, relocated, tmp_path):
    """Redirect only this newly created fixture, keeping its original inode."""
    assert alias.is_relative_to(tmp_path) and relocated.is_relative_to(tmp_path)
    alias.rename(relocated)
    installed = False
    try:
        if os.name == "nt":
            env = dict(os.environ, PIRC_TEST_JUNCTION_PATH=str(alias),
                       PIRC_TEST_JUNCTION_TARGET=str(relocated))
            try:
                subprocess.run([
                    "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                    '$ErrorActionPreference = "Stop"; New-Item -ItemType Junction -Path '
                    '$env:PIRC_TEST_JUNCTION_PATH -Target $env:PIRC_TEST_JUNCTION_TARGET | Out-Null',
                ], env=env, check=True, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.CalledProcessError) as exc:
                pytest.skip("native junction capability unavailable: " + str(exc))
        else:
            alias.symlink_to(relocated, target_is_directory=True)
        installed = True
        yield
    finally:
        if installed:
            # Remove the newly created link entry, never its target/content.
            if os.name == "nt":
                alias.rmdir()
            else:
                alias.unlink()
        relocated.rename(alias)


@pytest.mark.parametrize("kind", ["root", "ancestor"])
def test_upstream_does_not_bless_redirected_lexical_root(tmp_path, kind):
    root, _, _, binding = synthetic_binding(tmp_path)
    alias = root if kind == "root" else root.parent
    with redirect_directory(alias, tmp_path / "relocated", tmp_path):
        with pytest.raises(AdmissionError, match="UNAUTHORIZED_DATA"):
            validate_binding(root, binding)


def test_upstream_denies_same_bytes_replacement_before_real_open(tmp_path, monkeypatch):
    root, target, _, binding = synthetic_binding(tmp_path)
    content, changed = target.read_bytes(), []

    def replace():
        target.rename(root / "old-metadata.json")
        target.write_bytes(content)
        changed.append(True)

    reads, _ = observe_file(monkeypatch, target, before_open=replace)
    with pytest.raises(AdmissionError, match="CORRUPT_ARTIFACT|UNAUTHORIZED_DATA"):
        validate_binding(root, binding)
    assert changed == [True]
    assert reads == [], "a replacement must be rejected before metadata bytes"


@pytest.mark.parametrize("growth", [False, True])
def test_upstream_denies_real_change_after_read(tmp_path, monkeypatch, growth):
    root, target, _, binding = synthetic_binding(tmp_path)
    content, changed = target.read_bytes(), []
    before = target.stat()

    def mutate(stream, value):
        if not changed:
            target.write_bytes(content + b" " if growth else content[:-1] + b" ")
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            changed.append(True)

    reads, handles = observe_file(monkeypatch, target, after_read=mutate)
    with pytest.raises(AdmissionError, match="CORRUPT_ARTIFACT|UNAUTHORIZED_DATA"):
        validate_binding(root, binding)
    assert changed == [True] and reads and handles == [True]


def test_upstream_keeps_valid_large_formatted_unicode_metadata(tmp_path):
    root, target, payload, binding = synthetic_binding(tmp_path, payload_size=16 * 1024 * 1024 + 1)
    assert target.stat().st_size > 16 * 1024 * 1024
    assert validate_binding(root, binding) == payload


def terrain_files(tmp_path):
    binding = next(value for value in PUBLIC_BINDINGS if value.object_id == "terrain-selection")
    paths = [tmp_path / name for name in (binding.path,
        "experiments/pirc22/representation_matrix.json",
        "experiments/pirc22/representation_matrix.lock.json")]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / path.relative_to(tmp_path)).read_bytes())
    return paths


def test_terrain_validation_uses_one_actual_read_per_frozen_input(tmp_path, monkeypatch):
    paths = terrain_files(tmp_path)
    observations = [observe_file(monkeypatch, path) for path in paths]
    manifest = audit_inputs(tmp_path, ("terrain-selection",))
    assert manifest["data_authorization"] == "none"
    assert manifest["objects"][0]["sha256"] == next(
        value.sha256 for value in PUBLIC_BINDINGS if value.object_id == "terrain-selection")
    assert all(handles == [True] for _, handles in observations), (
        "the frozen validator must consume the admitted bytes, not reopen the selection")
    assert all(reads and all(read["size"] > 0 for read in reads)
               for reads, _ in observations), "every actual read needs a frozen-size bound"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_terrain_audit_denies_real_input_change_after_read(tmp_path, monkeypatch, index):
    paths = terrain_files(tmp_path)
    target = paths[index]
    content, changed = target.read_bytes(), []

    def mutate(stream, value):
        if not changed:
            target.write_bytes(content + b" ")
            changed.append(True)

    reads, _ = observe_file(monkeypatch, target, after_read=mutate)
    with pytest.raises(AdmissionError, match="CORRUPT_ARTIFACT|UNAUTHORIZED_DATA"):
        audit_inputs(tmp_path, ("terrain-selection",))
    assert changed == [True] and reads
