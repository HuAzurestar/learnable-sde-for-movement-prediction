"""Actual OS roots/descendants; only the negative group's observation is injected."""

import json
import os
import subprocess
import sys
import time

from infrastructure.process_tree import ProcessTree


def launch(program, ready):
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(ready)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    tree = ProcessTree(process)
    process.stdin.write(b"GO\n")
    process.stdin.flush()
    return process, tree


def wait_ready(ready, process):
    until = time.monotonic() + 5
    while not ready.exists():
        assert process.poll() is None, "real contained helper exited before readiness"
        assert time.monotonic() < until, "actual helper readiness was not reached"
        time.sleep(0.001)


def cleanup(process, tree):
    try:
        tree.terminate()
        process.wait(timeout=5)
        assert tree.wait_stopped(timeout=1), "real owned group cleanup was not confirmed"
    finally:
        tree.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_actual_running_root_vetoes_empty_group_observation(tmp_path, monkeypatch):
    ready = tmp_path / "actual-root-ready"
    program = ("import sys,time; from pathlib import Path; "
               "assert sys.stdin.readline().strip()=='GO'; "
               "Path(sys.argv[1]).write_text('ready'); time.sleep(30)")
    process, tree = launch(program, ready)
    try:
        wait_ready(ready, process)
        assert process.poll() is None and tree.active()
        with monkeypatch.context() as injected:
            # Adversarial contradictory job/group observation, not an exit
            # surrogate: the ORIGINAL live root handle must veto stop approval.
            injected.setattr(tree, "active", lambda: False)
            stopped = tree.wait_stopped(timeout=0.03)
            code = process.poll()
        actual_active = tree.active()
        print(json.dumps({"stopped": stopped, "actual_root_code": code,
                          "actual_group_active": actual_active, "pid": process.pid}))
        assert code is None and actual_active, "actual running-root counterexample not reached"
    finally:
        cleanup(process, tree)
    assert not stopped, "group-empty observation approved stop while the real root was still alive"


def test_actual_exited_root_does_not_hide_live_descendant(tmp_path):
    ready = tmp_path / "actual-descendant-ready"
    program = r'''
import subprocess,sys,time
from pathlib import Path
assert sys.stdin.readline().strip()=='GO'
child_program = "import sys,time; from pathlib import Path; Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
child = subprocess.Popen([sys.executable,'-c',child_program,sys.argv[1]])
while not Path(sys.argv[1]).exists():
    time.sleep(0.001)
print(child.pid,flush=True)
'''
    process, tree = launch(program, ready)
    try:
        assert process.wait(timeout=5) == 0
        assert ready.exists() and tree.active(), "actual surviving descendant was not reached"
        stopped = tree.wait_stopped(timeout=0.03)
        print(json.dumps({"stopped": stopped, "actual_root_code": process.poll(),
                          "actual_group_active": tree.active(), "root_pid": process.pid}))
        assert process.poll() == 0 and tree.active()
        assert not stopped, "completed root hid an actual running descendant"
    finally:
        cleanup(process, tree)
