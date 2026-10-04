"""Actual member I/O and owned physical-prefix lifetime, no result stand-ins."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import time

import pytest

from infrastructure.research_store import ResearchError, ResearchStore, encode
from tests.research_file_observation import is_query_only_open
from tests.test_research_metadata_prefix_reads import prepared


def test_prefix_enumerates_and_opens_actual_members_relative_to_original_directories(tmp_path, monkeypatch):
    store, expected = prepared(tmp_path)
    original_glob, original_open, original_json = Path.glob, os.open, store._json
    path_enumerations, absolute_members, actual_json = [], [], []

    def glob(path, *args, **kwargs):
        if path == store.path / "events":
            path_enumerations.append(str(path))
        return original_glob(path, *args, **kwargs)

    def opened(path, *args, **kwargs):
        member = Path(path)
        if not is_query_only_open(args, kwargs) and member.is_absolute() and member.suffix == ".json" and member.parent in {
                store.path, store.path / "events"}:
            absolute_members.append(str(member))
        return original_open(path, *args, **kwargs)

    def read(path):
        result = original_json(path)
        actual_json.append(str(path))
        return result

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(store, "_json", read)
    assert store.events() == expected
    observation = {"platform": os.name, "path_enumerations": path_enumerations,
                   "absolute_members": absolute_members, "actual_json": actual_json}
    (tmp_path / "relative-members-observed.json").write_bytes(encode(observation))
    assert len(actual_json) == len(expected) + 1, "retain every actual event/head read"
    assert not path_enumerations and not absolute_members, observation


def test_captured_physical_member_cannot_reopen_after_its_original_scope_closed(tmp_path, monkeypatch):
    store, expected = prepared(tmp_path)
    original, captured = store._json, []

    def read(path):
        captured.append(path)
        return original(path)

    monkeypatch.setattr(store, "_json", read)
    assert store.events() == expected
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        original(captured[0])


def test_bound_member_cannot_inherit_another_actual_threads_directory_scope(tmp_path, monkeypatch):
    store, expected = prepared(tmp_path)
    original, errors, reached = store._json, [], []

    def read(path):
        if not reached:
            reached.append(True)

            def other_thread():
                try:
                    original(path)
                except ResearchError as error:
                    errors.append(error.code)

            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(other_thread).result(timeout=5)
        return original(path)

    monkeypatch.setattr(store, "_json", read)
    assert store.events() == expected
    assert reached == [True] and errors == ["CORRUPT_ARTIFACT"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="actual POSIX fork ownership control")
def test_bound_member_cannot_inherit_an_actual_forked_process_directory_scope(tmp_path, monkeypatch):
    store, expected = prepared(tmp_path)
    original, results, reached = store._json, [], []

    def read(path):
        if not reached:
            reached.append(True)
            incoming, outgoing = os.pipe()
            child = os.fork()
            if child == 0:
                os.close(incoming)
                try:
                    result = {"accepted": original(path) is not None}
                except ResearchError as error:
                    result = {"error": error.code}
                except BaseException as error:
                    result = {"unexpected": type(error).__name__}
                os.write(outgoing, encode(result))
                os.close(outgoing)
                os._exit(0)
            os.close(outgoing)
            deadline, waited = time.monotonic() + 5, False
            try:
                while time.monotonic() < deadline:
                    pid, status = os.waitpid(child, os.WNOHANG)
                    if pid:
                        waited = True
                        assert os.waitstatus_to_exitcode(status) == 0
                        results.append(json.loads(os.read(incoming, 4096)))
                        break
                    time.sleep(0.005)
                assert waited, "actual owned fork did not finish"
            finally:
                if not waited:
                    os.kill(child, signal.SIGKILL)  # Only this test's actual child.
                    os.waitpid(child, 0)
                os.close(incoming)
        return original(path)

    monkeypatch.setattr(store, "_json", read)
    assert store.events() == expected
    (tmp_path / "relative-fork-observed.json").write_bytes(encode(results))
    assert reached == [True] and results == [{"error": "CORRUPT_ARTIFACT"}]


def test_plain_standalone_json_on_another_thread_keeps_its_own_source_boundary(tmp_path, monkeypatch):
    store, expected = prepared(tmp_path)
    target, original, observed, reached = tmp_path / "standalone.json", store._json, [], []
    target.write_bytes(encode({"synthetic": "unbound"}))
    original_open = os.open

    def opened(path, *args, **kwargs):
        if Path(path) == target and not is_query_only_open(args, kwargs):
            observed.append(str(path))
        return original_open(path, *args, **kwargs)

    def read(path):
        if not reached:
            reached.append(True)
            with ThreadPoolExecutor(max_workers=1) as worker:
                assert worker.submit(ResearchStore._json, target).result(timeout=5) == {"synthetic": "unbound"}
        return original(path)

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(store, "_json", read)
    assert store.events() == expected
    assert observed == [str(target)], "plain reads must not inherit the owned prefix backend"


def test_actual_directory_enumeration_preserves_all_pages_and_ignored_unicode_members(tmp_path, monkeypatch):
    store, expected = prepared(tmp_path)
    directory = store.path / "events"
    for number in range(1024):
        (directory / (f"ignored-{number:04d}-" + "目录" * 12 + ".note")).touch()
    calls = []
    if os.name == "nt":
        import infrastructure.research_windows_files as native
        original_api = native._api

        def api():
            c, basic, query, create, close = original_api()

            def queried(handle, kind, *args):
                if kind in {10, 11}:
                    calls.append(kind)
                return query(handle, kind, *args)

            return c, basic, queried, create, close

        monkeypatch.setattr(native, "_api", api)
    else:
        original_list = os.listdir

        def listed(path):
            if isinstance(path, int):
                calls.append(path)
            return original_list(path)

        monkeypatch.setattr(os, "listdir", listed)
    assert store.events() == expected
    (tmp_path / "relative-enumeration-observed.json").write_bytes(encode({"platform": os.name, "calls": calls}))
    assert len(calls) >= (2 if os.name == "nt" else 1), "execute the original directory-handle enumeration"
