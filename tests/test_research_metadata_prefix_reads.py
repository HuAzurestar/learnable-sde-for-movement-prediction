"""Real physical-prefix identities, payload reads and directory admission work.

All file operations and parsers execute. Replacements contain the original
valid bytes: a refusal cannot be explained by a synthetic hash-chain failure.
"""

import os
from pathlib import Path
import shutil

import pytest

import infrastructure.research_files as files
import infrastructure.research_json as json_reader
from infrastructure.research_store import ResearchError, ResearchStore, encode


def prepared(tmp_path, count=8):
    store = ResearchStore(tmp_path / "runtime", "metadata-prefix", initialize=True)
    with store._read_transaction():
        for number in range(count):
            store.append("METADATA_PREFIX_FIXTURE", {"number": number}, "prefix-" + str(number))
    return store, store.events()


def identities(root):
    return {(info.st_dev, info.st_ino) for path in root.rglob("*.json")
            for info in [path.stat()]}


class ObservedStream:
    def __init__(self, stream, observe):
        self.stream, self.observe = stream, observe

    def read(self, count=-1):
        value = self.stream.read(count)
        self.observe(len(value))
        return value

    def __getattr__(self, name):
        return getattr(self.stream, name)


@pytest.mark.parametrize("directory", ["events", "store", "ancestor"])
@pytest.mark.parametrize("switch_at", ["first-event", "after-head"])
def test_physical_prefix_rejects_original_directory_replacement_without_foreign_bytes(
        tmp_path, monkeypatch, directory, switch_at):
    store, expected = prepared(tmp_path)
    targets = {"events": store.path / "events", "store": store.path,
               "ancestor": tmp_path / "runtime"}
    target, incoming, parked = targets[directory], tmp_path / "incoming", tmp_path / "parked"
    # Explicitly restrict every directory move to this test's owned fixture.
    for path in (target, incoming, parked):
        assert path.resolve().is_relative_to(tmp_path.resolve()) and path != tmp_path
    shutil.copytree(target, incoming)
    original_ids, foreign_ids = identities(target), identities(incoming)
    assert original_ids.isdisjoint(foreign_ids)
    original_json, original_read = store._json, json_reader.read_json
    observed = {"platform": os.name, "directory": directory, "switch_at": switch_at,
                "switched": False, "foreign_bytes": 0, "foreign_reads": [], "error": None}

    def parse(stream, size, **kwargs):
        information = os.fstat(stream.fileno())
        foreign = (information.st_dev, information.st_ino) in foreign_ids

        def read(count):
            if foreign and count:
                observed["foreign_bytes"] += count
                observed["foreign_reads"].append({"inode": information.st_ino, "bytes": count})

        return original_read(ObservedStream(stream, read), size, **kwargs)

    def read(path):
        result = original_json(path)
        boundary = (path.parent == store.path / "events" if switch_at == "first-event"
                    else path == store.path / "head.json")
        if boundary and not observed["switched"]:
            try:
                target.rename(parked)
                incoming.rename(target)
            except PermissionError as error:
                if os.name == "nt" and error.winerror == 5:
                    observed["native_move_denial"] = error.winerror
                    (tmp_path / "prefix-replacement-observed.json").write_bytes(encode(observed))
                    pytest.skip("actual native Windows directory move denied: WinError5")
                raise
            observed["switched"] = True
        return result

    monkeypatch.setattr(json_reader, "read_json", parse)
    monkeypatch.setattr(store, "_json", read)
    try:
        result = store.events()
        observed["accepted_original_chain"] = result == expected
    except ResearchError as error:
        observed["error"] = error.code
    finally:
        (tmp_path / "prefix-replacement-observed.json").write_bytes(encode(observed))
    assert observed["switched"], observed
    assert observed["error"] in {"CORRUPT_ARTIFACT", "UNAUTHORIZED_DATA"}, observed
    assert observed["foreign_bytes"] == 0, observed


@pytest.mark.parametrize("count", [8, 32])
def test_physical_prefix_directory_admission_is_bounded_without_skipping_any_json(
        tmp_path, monkeypatch, count):
    store, expected = prepared(tmp_path, count)
    original_source, original_json = files._source_in_root, store._json
    sources, reads = [], []

    def source(root, path):
        sources.append(str(path))
        return original_source(root, path)

    def read(path):
        reads.append(str(path))
        return original_json(path)

    monkeypatch.setattr(files, "_source_in_root", source)
    monkeypatch.setattr(store, "_json", read)
    assert store.events() == expected
    observed = {"platform": os.name, "events": len(expected), "sources": sources, "reads": reads}
    (tmp_path / "prefix-work-observed.json").write_bytes(encode(observed))
    expected_paths = [str(store.path / "events" / f"{event['sequence']:016d}.json")
                      for event in expected] + [str(store.path / "head.json")]
    assert reads == expected_paths, "every JSON must be read afresh, including the head"
    assert 0 < len(sources) <= 16, observed  # Whole-root admission, not a global request-complexity claim.


@pytest.mark.parametrize("count", [0, 8])
def test_unchanged_physical_prefix_remains_compatible(tmp_path, count):
    store, expected = prepared(tmp_path, count)
    assert store.events() == expected
    assert ResearchStore(tmp_path / "runtime", "metadata-prefix").events() == expected
