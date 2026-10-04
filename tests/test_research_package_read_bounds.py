"""Real incoming package files must retain their declared allocation bounds."""

import os
from pathlib import Path
import shutil

import pytest

from application.research_budget import BudgetSpec
from application.research_evidence import accept_evidence_package
from infrastructure.research_store import ResearchError
from tests.research_file_observation import is_query_only_open
from tests.test_research_comparison import comparison_source, compute


FILES = [
    ("manifest.json", 64 * 1024 * 1024),
    ("aggregate.json", 64 * 1024 * 1024),
    ("metrics.csv", 64 * 1024 * 1024),
    ("PaperEvidenceIndex.json", 64 * 1024 * 1024),
    ("ComputationReceipt.json", 1024 * 1024),
    ("FigureIndex.json", 2 * 1024 * 1024),
    ("figure", 2 * 1024 * 1024),
]


@pytest.fixture(scope="module")
def managed_package(tmp_path_factory):
    directory = tmp_path_factory.mktemp("actual-managed-package")
    source = comparison_source(directory)
    result = compute(source, budget=BudgetSpec(20), output=directory / "managed")
    assert result["state"] == "SUCCEEDED", result
    return source[0], directory / "managed", result["comparison"]["aggregate_hash"]


def observe_target(monkeypatch, target, *, grow_to=None, after_read=None):
    """Observe the real old Path.read_bytes and new descriptor-backed readers."""
    original_open, original_fdopen = Path.open, os.fdopen
    original_os_open = os.open
    descriptors, reads = set(), []

    class Stream:
        def __init__(self, actual):
            self.actual = actual

        def __enter__(self):
            self.actual.__enter__()
            return self

        def __exit__(self, *args):
            return self.actual.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def read(self, size=-1):
            if grow_to is not None:
                # Actual growth AFTER the reader's path/open-handle stat.
                with original_open(target, "r+b") as writer:
                    writer.truncate(grow_to)
            reads.append(size)
            content = self.actual.read(size)
            if after_read is not None:
                after_read()
            return content

    def path_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return Stream(stream) if path == target and mode == "rb" else stream

    def descriptor_open(path, *args, **kwargs):
        fd = original_os_open(path, *args, **kwargs)
        if Path(path) == target and not is_query_only_open(args, kwargs):
            descriptors.add(fd)
        return fd

    def descriptor_stream(fd, *args, **kwargs):
        stream = original_fdopen(fd, *args, **kwargs)
        observed = fd in descriptors
        descriptors.discard(fd)  # Do not observe a later unrelated reuse of fd.
        return Stream(stream) if observed else stream

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(os, "open", descriptor_open)
    monkeypatch.setattr(os, "fdopen", descriptor_stream)
    return reads


def incoming_copy(managed_package, tmp_path, name):
    store, original, aggregate_hash = managed_package
    incoming = tmp_path / "incoming"
    shutil.copytree(original, incoming)
    filename = next(path.name for path in incoming.glob("*.svg")) if name == "figure" else name
    return store, incoming, aggregate_hash, incoming / filename


@pytest.mark.parametrize("name,limit", FILES)
def test_actual_package_import_reads_each_file_with_explicit_bound(managed_package, tmp_path, monkeypatch, name, limit):
    store, incoming, aggregate_hash, target = incoming_copy(managed_package, tmp_path, name)
    before_attempts = store.attempts()
    reads = observe_target(monkeypatch, target)
    imported = accept_evidence_package(store, incoming, aggregate_hash)
    assert imported["aggregate_hash"] == aggregate_hash
    assert reads and all(0 <= size <= limit + 1 for size in reads), "incoming package read has no allocation bound"
    assert store.attempts() == before_attempts


@pytest.mark.parametrize("name,limit", FILES)
def test_growth_after_actual_package_stat_is_quota_error_before_parse_or_import(managed_package, tmp_path, monkeypatch, name, limit):
    store, incoming, aggregate_hash, target = incoming_copy(managed_package, tmp_path, name)
    before = store.events()
    reads = observe_target(monkeypatch, target, grow_to=limit + 1)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        accept_evidence_package(store, incoming, aggregate_hash)
    assert reads and all(0 <= size <= limit + 1 for size in reads)
    assert store.events() == before
    assert target.stat().st_size == limit + 1


def test_unchanged_real_managed_package_is_imported_idempotently(managed_package, tmp_path):
    store, incoming, aggregate_hash, _ = incoming_copy(managed_package, tmp_path, "manifest.json")
    first = accept_evidence_package(store, incoming, aggregate_hash)
    assert accept_evidence_package(store, incoming, aggregate_hash) == first
    assert store.manifest("comparison-" + aggregate_hash) == first


@pytest.mark.parametrize("name,limit", FILES)
def test_file_growing_at_actual_open_is_rejected_without_content_read(managed_package, tmp_path, monkeypatch, name, limit):
    store, incoming, aggregate_hash, target = incoming_copy(managed_package, tmp_path, name)
    reads = observe_target(monkeypatch, target)
    original = os.open

    def grow(path, *args, **kwargs):
        if Path(path) == target and not is_query_only_open(args, kwargs):
            with target.open("r+b") as writer:
                writer.truncate(limit + 1)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", grow)
    before = store.events()
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        accept_evidence_package(store, incoming, aggregate_hash)
    assert reads == [] and target.stat().st_size == limit + 1
    assert store.events() == before


@pytest.mark.parametrize("name,limit", FILES)
def test_replaced_path_at_actual_open_is_rejected_before_read(managed_package, tmp_path, monkeypatch, name, limit):
    store, incoming, aggregate_hash, target = incoming_copy(managed_package, tmp_path, name)
    replacement = tmp_path / "same-content-new-file"
    shutil.copy2(target, replacement)
    reads = observe_target(monkeypatch, target)
    original = os.open

    def replace(path, *args, **kwargs):
        if Path(path) == target and not is_query_only_open(args, kwargs):
            os.replace(replacement, target)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace)
    before = store.events()
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        accept_evidence_package(store, incoming, aggregate_hash)
    assert reads == [] and not replacement.exists()
    assert store.events() == before


@pytest.mark.parametrize("name,limit", FILES)
@pytest.mark.parametrize("fault", ["in-place", "replace"])
def test_actual_post_read_mutation_cannot_publish_old_bytes_as_stable_package(managed_package, tmp_path, monkeypatch, name, limit, fault):
    store, incoming, aggregate_hash, target = incoming_copy(managed_package, tmp_path, name)
    original_open = Path.open
    information = target.stat()
    replacement = tmp_path / "same-content-new-file"
    if fault == "replace":
        shutil.copy2(target, replacement)

    def change():
        if fault == "replace":
            os.replace(replacement, target)
        else:
            with original_open(target, "r+b") as writer:
                writer.write(b"!")
            os.utime(target, ns=(information.st_atime_ns, information.st_mtime_ns + 1_000_000))

    reads = observe_target(monkeypatch, target, after_read=change)
    before = store.events()
    if fault == "replace" and os.name == "nt":
        # The actual production descriptor denies delete sharing on Windows.
        # Preserve that native refusal; it is not evidence of the POSIX
        # post-replacement identity guard (covered by the native Linux probe).
        with pytest.raises(PermissionError):
            accept_evidence_package(store, incoming, aggregate_hash)
        assert replacement.exists()
        assert target.stat().st_ino == information.st_ino
    else:
        with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
            accept_evidence_package(store, incoming, aggregate_hash)
    assert reads and all(0 <= size <= limit + 1 for size in reads)
    assert store.events() == before


def test_actual_original_64_mib_quota_accepts_large_valid_manifest(managed_package, tmp_path, monkeypatch):
    store, incoming, aggregate_hash, target = incoming_copy(managed_package, tmp_path, "manifest.json")
    limit = 64 * 1024 * 1024
    original = target.read_bytes()
    target.write_bytes(original + b" " * (limit - len(original)))
    reads = observe_target(monkeypatch, target)
    imported = accept_evidence_package(store, incoming, aggregate_hash)
    assert imported == store.manifest("comparison-" + aggregate_hash)
    assert reads == [limit + 1] and target.stat().st_size == limit


def test_nonregular_package_file_is_rejected_before_read(managed_package, tmp_path, monkeypatch):
    store, incoming, aggregate_hash, target = incoming_copy(managed_package, tmp_path, "metrics.csv")
    target.unlink()
    target.mkdir()
    reads = observe_target(monkeypatch, target)
    before = store.events()
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        accept_evidence_package(store, incoming, aggregate_hash)
    assert reads == [] and store.events() == before


@pytest.mark.parametrize("name,limit", FILES)
def test_growth_between_actual_handle_stat_and_path_recheck_is_quota_error(managed_package, tmp_path, monkeypatch, name, limit):
    store, incoming, aggregate_hash, target = incoming_copy(managed_package, tmp_path, name)
    reads = observe_target(monkeypatch, target)
    original_open, original_fstat = os.open, os.fstat
    descriptors, touched = set(), []

    def track(path, *args, **kwargs):
        fd = original_open(path, *args, **kwargs)
        if Path(path) == target and not is_query_only_open(args, kwargs):
            descriptors.add(fd)
        return fd

    def grow_after_stat(fd):
        information = original_fstat(fd)
        if fd in descriptors and not touched:
            with target.open("r+b") as writer:
                writer.truncate(limit + 1)
            touched.append(True)
        return information

    monkeypatch.setattr(os, "open", track)
    monkeypatch.setattr(os, "fstat", grow_after_stat)
    before = store.events()
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        accept_evidence_package(store, incoming, aggregate_hash)
    assert reads == [] and touched == [True]
    assert store.events() == before and target.stat().st_size == limit + 1
