"""Real physical-prefix identities, payload reads and directory admission work.

All file operations and parsers execute. Replacements contain the original
valid bytes: a refusal cannot be explained by a synthetic hash-chain failure.
"""

from contextlib import contextmanager
import json
import os
import shutil

import pytest

import infrastructure.research_files as files
import infrastructure.research_json as json_reader
import infrastructure.research_publication as publication
from infrastructure.research_store import ResearchError, ResearchStore, encode
from tests.research_file_observation import observe_file


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


def test_physical_prefix_retains_native_same_inode_restored_mtime_rejection(tmp_path, monkeypatch):
    store, _ = prepared(tmp_path)
    target = store.path / "events" / "0000000000000001.json"
    before, changed = target.stat(), []
    original = target.read_bytes()
    replacement = original.replace(b'"number":0', b'"number":1')
    assert replacement != original and len(replacement) == len(original)

    def change(*_):
        if not changed:
            target.write_bytes(replacement)
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            changed.append(True)

    reads, handles = observe_file(monkeypatch, target, after_read=change)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        store.events()
    after = target.stat()
    assert changed == [True] and reads and handles == [True]
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino, before.st_size, before.st_mtime_ns)


@pytest.mark.parametrize("replacement", ["regular", "symlink"])
def test_physical_prefix_rejects_actual_leaf_replacement_before_payload(tmp_path, monkeypatch, replacement):
    store, _ = prepared(tmp_path)
    target = store.path / "events" / "0000000000000001.json"
    incoming = tmp_path / "incoming.json"
    incoming.write_bytes(target.read_bytes())
    changed = []

    def change():
        target.rename(tmp_path / "original.json")
        if replacement == "regular":
            incoming.rename(target)
        else:
            try:
                target.symlink_to(incoming)
            except OSError as error:
                if os.name == "nt" and error.winerror == 1314:
                    pytest.skip("actual native symlink creation denied: WinError1314")
                raise
        changed.append(True)

    reads, _ = observe_file(monkeypatch, target, before_open=change)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT|UNAUTHORIZED_DATA"):
        store.events()
    assert changed == [True] and reads == [], "replacement payload bytes must not be consumed"


def test_physical_prefix_keeps_large_formatted_head_and_lagging_head_compatibility(tmp_path, monkeypatch):
    store, expected = prepared(tmp_path)
    target = store.path / "head.json"
    # Actual valid legacy formatting above the JSON streaming threshold. No
    # new aggregate prefix or blanket metadata cap is introduced by root pins.
    content = b" \r\n\t" * (512 * 1024) + encode({"sequence": 0, "hash": "0" * 64})
    target.write_bytes(content)
    reads, handles = observe_file(monkeypatch, target)
    assert store.events() == expected
    assert reads and all(0 <= row["size"] <= 64 * 1024 for row in reads)
    assert handles == [True]


@pytest.mark.parametrize("damage", ["gap", "hash", "head", "invalid-json"])
def test_physical_prefix_still_refuses_real_corruption(tmp_path, damage):
    store, _ = prepared(tmp_path)
    target = store.path / "events" / "0000000000000001.json"
    if damage == "gap":
        target.rename(target.with_name("0000000000000000.json"))
    elif damage == "hash":
        event = json.loads(target.read_bytes())
        event["hash"] = "f" * 64
        target.write_bytes(encode(event))
    elif damage == "head":
        (store.path / "head.json").write_bytes(encode({"sequence": 99, "hash": "0" * 64}))
    else:
        target.write_bytes(b'{"damaged":')
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        store.events()


@pytest.mark.parametrize("damaged", [False, True])
def test_physical_prefix_closes_actual_original_directory_descriptors(tmp_path, monkeypatch, damaged):
    store, expected = prepared(tmp_path)
    original_directory, closed = publication.opened_directory, []

    @contextmanager
    def directory(path):
        fd = None
        try:
            with original_directory(path) as value:
                fd = value[1]
                yield value
        finally:
            if fd is not None:
                with pytest.raises(OSError):
                    os.fstat(fd)  # Actual native/CRT fd closure, not a Boolean substitute.
                closed.append(str(path))

    monkeypatch.setattr(publication, "opened_directory", directory)
    if damaged:
        (store.path / "events" / "0000000000000001.json").write_bytes(b"{}")
        with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
            store.events()
    else:
        assert store.events() == expected
    assert sorted(closed) == sorted([str(store.path), str(store.path / "events")])


def test_physical_prefix_missing_original_events_directory_is_typed_refusal(tmp_path):
    store, _ = prepared(tmp_path, 0)
    (store.path / "events").rename(tmp_path / "parked-events")
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        store.events()
