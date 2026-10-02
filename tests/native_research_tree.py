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


def exercise(containment, root, name, *, early_exit=False, deadline_seconds=5, watchdog=False,
             blocked_heartbeat=False):
    directory = root / name
    directory.mkdir()
    marker, ready = directory / "escaped", directory / "started"
    if blocked_heartbeat:
        # No reader exists: the wrapper's real heartbeat open/write blocks.
        os.mkfifo(directory / "heartbeat.json")
    child = ("import pathlib,sys,time; marker=pathlib.Path(sys.argv[1]); "
             "marker.with_name('started').touch(); "
             "time.sleep(max(0,float(sys.argv[2])-time.monotonic()) if len(sys.argv)>2 else 1.4); "
             "marker.write_text(str(time.monotonic())); time.sleep(5)")
    worker = ("import pathlib,subprocess,sys,time; "
              "pathlib.Path(sys.argv[2]).with_name('worker-started').write_text(str(time.monotonic())); "
              "subprocess.Popen([sys.executable,'-c',sys.argv[1],*sys.argv[2:]]); "
              "ready=pathlib.Path(sys.argv[2]).with_name('started'); "
              "\nwhile not ready.exists(): time.sleep(0.005)"
              + ("\n" if early_exit else "\ntime.sleep(5)"))
    deadline = time.monotonic() + deadline_seconds
    # This native fixture includes all startup in its declared deadline. Its
    # post-deadline marker must not be confused with work permitted before it.
    child_arguments = [str(marker), str(deadline + 0.4)] if watchdog else [str(marker)]
    error_log = tempfile.TemporaryFile()
    process = subprocess.Popen([sys.executable, str(ROOT / "infrastructure/research_worker.py"),
        str(directory / "heartbeat.json"), str(deadline), sys.executable, "-c", worker, child, *child_arguments],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log,
        start_new_session=os.name != "nt", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    tree = containment.ProcessTree(process)
    stopped = False
    try:
        process.stdin.write(b"GO\n")
        process.stdin.flush()
        while not ready.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                error_log.seek(0)
                diagnostic = {"check": name, "returncode": process.poll(),
                    "elapsed_seconds": time.monotonic() - deadline + deadline_seconds,
                    "deadline_seconds": deadline_seconds, "ready": ready.exists(),
                    "worker_started": (directory / "worker-started").read_text()
                        if (directory / "worker-started").exists() else None,
                    "heartbeat": (directory / "heartbeat.json").exists(),
                    "stderr": error_log.read(4096).decode("utf-8", errors="replace")}
                raise AssertionError("fixture descendant did not start inside the contained tree: "
                                     + json.dumps(diagnostic, sort_keys=True))
            time.sleep(0.005)
        if watchdog:
            assert process.wait(timeout=5) != 0
            assert tree.wait_stopped(), "wrapper deadline left running descendants"
        else:
            if early_exit:
                assert process.wait(timeout=5) == 0
            if early_exit and os.name == "nt":
                assert tree.wait_stopped(), "wrapper exit left descendants outside its nested job"
            else:
                assert tree.active(), "native probe missed a live descendant"
            tree.terminate()
            assert tree.wait_stopped(), "native kill did not confirm whole-tree stop"
            process.wait(timeout=5)
        stopped = True
        # Do not reissue a numeric process-group kill after observing stop.
        if not watchdog:
            tree.terminate()
        time.sleep(1.5)
        assert not marker.exists(), "descendant performed work after whole-tree stop: " + json.dumps({
            "check": name, "deadline_seconds": deadline_seconds,
            "marker_seconds_after_deadline": float(marker.read_text()) - deadline if marker.exists() else None,
            "observed_seconds_after_deadline": time.monotonic() - deadline,
            "returncode": process.returncode}, sort_keys=True)
        return name
    finally:
        # A failed observation still needs fixture cleanup. On the watchdog
        # success path the wrapper already killed its own group/job tree.
        if not stopped:
            tree.terminate()
            process.wait(timeout=5)
        tree.close()
        process.stdin.close()
        error_log.close()


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
                  exercise(containment, root, "wrapper-independent-watchdog", watchdog=True)]
        if sys.platform.startswith("linux"):
            checks.append(exercise(containment, root, "blocked-heartbeat-watchdog",
                                   watchdog=True, blocked_heartbeat=True))
    if files != {name: source_hash(ROOT / name) for name in files}:
        raise ValueError("native implementation moved during verification")
    print(json.dumps({"status": "passed", "scope": "native containment and stdlib wrapper only",
        "checks": checks, "source_hashes": files, "python": platform.python_version(),
        "platform": platform.system(), "qualification": "engineering-only"}, sort_keys=True))


if __name__ == "__main__":
    main()
