"""Stdlib-only actual OS containment checks, not full model integration.

Loads the real containment module directly so a native backend can be checked
without installing unrelated model dependencies. Never bypasses a budget or
admission gate for a research run: every input here is a disposable process
fixture. The caller must supply expected source hashes from its fixed checkout.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def source_hash(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def exercise(containment, root, name, *, early_exit=False, deadline_seconds=5, watchdog=False):
    directory = root / name
    directory.mkdir()
    marker, ready = directory / "escaped", directory / "started"
    child = ("import pathlib,sys,time; marker=pathlib.Path(sys.argv[1]); "
             "marker.with_name('started').touch(); time.sleep(1.4); marker.touch(); time.sleep(5)")
    worker = ("import pathlib,subprocess,sys,time; "
              "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
              "ready=pathlib.Path(sys.argv[2]).with_name('started'); "
              "\nwhile not ready.exists(): time.sleep(0.005)"
              + ("\n" if early_exit else "\ntime.sleep(5)"))
    deadline = time.monotonic() + deadline_seconds
    process = subprocess.Popen([sys.executable, str(ROOT / "infrastructure/research_worker.py"),
        str(directory / "heartbeat.json"), str(deadline), sys.executable, "-c", worker, child, str(marker)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    tree = containment.ProcessTree(process)
    stopped = False
    try:
        process.stdin.write(b"GO\n")
        process.stdin.flush()
        while not ready.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("fixture descendant did not start inside the contained tree")
            time.sleep(0.005)
        if watchdog:
            assert process.wait(timeout=5) != 0
            assert tree.wait_stopped(), "wrapper deadline left running descendants"
        else:
            if early_exit:
                assert process.wait(timeout=5) == 0
            assert tree.active(), "native probe missed a live descendant"
            tree.terminate()
            assert tree.wait_stopped(), "native kill did not confirm whole-tree stop"
            process.wait(timeout=5)
        stopped = True
        # Do not reissue a numeric process-group kill after observing stop.
        if not watchdog:
            tree.terminate()
        time.sleep(1.5)
        assert not marker.exists(), "descendant performed work after whole-tree stop"
        return name
    finally:
        # A failed observation still needs fixture cleanup. On the watchdog
        # success path the wrapper already killed its own group/job tree.
        if not stopped:
            tree.terminate()
        tree.close()
        process.stdin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tree-hash", required=True)
    parser.add_argument("--expected-wrapper-hash", required=True)
    args = parser.parse_args()
    files = {"infrastructure/process_tree.py": source_hash(ROOT / "infrastructure/process_tree.py"),
             "infrastructure/research_worker.py": source_hash(ROOT / "infrastructure/research_worker.py")}
    if (files["infrastructure/process_tree.py"] != args.expected_tree_hash or
            files["infrastructure/research_worker.py"] != args.expected_wrapper_hash):
        raise ValueError("native source differs from fixed-checkout hashes")
    spec = importlib.util.spec_from_file_location("pirc25_native_containment", ROOT / "infrastructure/process_tree.py")
    containment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(containment)
    with tempfile.TemporaryDirectory(prefix="shared-native-tree-") as temporary:
        root = Path(temporary)
        checks = [exercise(containment, root, "live-parent-tree"),
                  exercise(containment, root, "stopped-leader-tree", early_exit=True),
                  exercise(containment, root, "wrapper-independent-watchdog", deadline_seconds=1, watchdog=True)]
    if files != {name: source_hash(ROOT / name) for name in files}:
        raise ValueError("native implementation moved during verification")
    print(json.dumps({"status": "passed", "scope": "native containment and stdlib wrapper only",
        "checks": checks, "source_hashes": files, "python": platform.python_version(),
        "platform": platform.system(), "qualification": "engineering-only"}, sort_keys=True))


if __name__ == "__main__":
    main()
